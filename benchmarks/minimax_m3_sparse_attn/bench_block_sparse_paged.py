#!/usr/bin/env python3
"""Original vs paged block-sparse attention, prefill and decode, over context.

Compares implementations of the *same* block-sparse attention, fed identical
pre-generated (causally valid) block selections at a fixed 2,048-token budget
(top-16 x block_size 128). Nothing about the math differs between them — only
how a selected block's physical slots are resolved:

  orig    ``flash_{prefill,decode}_with_gqa_share_sparse`` — one ``req_to_token``
          lookup per token (a ``block_size``-wide gather), then a runtime int64
          ``% max_slots`` guard applied to that whole vector.
  nomod   the same kernel with the vector modulo replaced by a select. Isolates
          the cost of the modulo alone; needs no page-size constraint.
  paged   ``..._paged`` — when ``page_size == block_size`` a block is exactly one
          physical page, so its slots are consecutive: one scalar lookup plus a
          scalar modulo replaces the gather entirely.

All variants must produce bit-identical output; the harness checks that and
reports max abs diff per point.

MiniMax-M3 shapes: 64 q heads / 4 kv heads / d128, bf16, page_size 128,
prefill = 4,096-token extend chunk on a growing prefix.

    CUDA_VISIBLE_DEVICES=4 python bench_block_sparse_paged.py
    python bench_block_sparse_paged.py --context-lens 32768,262144 --phases decode
    python bench_block_sparse_paged.py --variants orig,paged --no-plots

Outputs (default --out results/block_sparse_kernel_comparison): raw.json,
summary.csv, prefill_latency.png, decode_latency.png, speedup.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402
    bench_cuda,
    build_decode_inputs,
    build_prefill_inputs,
    gpu_info,
    wait_for_idle,
    warn_if_contended,
    write_results,
)
from m3_config import m3_config  # noqa: E402

from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (  # noqa: E402
    flash_decode_with_gqa_share_sparse,
)
from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse_nomod import (  # noqa: E402
    flash_decode_with_gqa_share_sparse_nomod,
)
from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse_paged import (  # noqa: E402
    flash_decode_with_gqa_share_sparse_paged,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse import (  # noqa: E402
    flash_prefill_with_gqa_share_sparse,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse_nomod import (  # noqa: E402
    flash_prefill_with_gqa_share_sparse_nomod,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse_paged import (  # noqa: E402
    flash_prefill_with_gqa_share_sparse_paged,
)

DEFAULT_CTXS = [8192, 16384, 32768, 65536, 131072, 262144, 524288]
DEFAULT_VARIANTS = ["orig", "nomod", "paged"]
BLOCK_SIZE = 128
TOKEN_BUDGET = 2048
BLOCK_TOPK = TOKEN_BUDGET // BLOCK_SIZE  # 16
PREFILL_CHUNK = 4096
BASELINE = "orig"


def _ctx_label(n: int) -> str:
    return f"{n // 1024}k" if n < 1 << 20 else f"{n >> 20}M"


def _rand_block_idx(
    abs_pos: torch.Tensor, kv_heads: int, topk: int, gen: torch.Generator
) -> torch.Tensor:
    """[kv_heads, rows, topk] int32 block ids, uniform in each row's causal range.

    Duplicates are possible and irrelevant for timing — it matches the unsorted
    output convention of the real selectors, and every variant sees the same
    tensor, so the comparison is unaffected.
    """
    rows = abs_pos.shape[0]
    u = torch.rand(kv_heads, rows, topk, device=abs_pos.device, generator=gen)
    hi = ((abs_pos + BLOCK_SIZE) // BLOCK_SIZE).to(torch.float32).view(1, rows, 1)
    return (u * hi).to(torch.int32)


def _prefill_runner(variant: str, inp, topk_idx, page_size: int):
    common = dict(
        q=inp.q, k_cache=inp.k_cache, v_cache=inp.v_cache, sink=None,
        req_to_token=inp.req_to_token, slot_ids=inp.slot_ids, topk_idx=topk_idx,
        block_size_q=1, block_size_k=BLOCK_SIZE, cu_seqlens=inp.cu_seqlens,
        seq_lens=inp.seq_lens, prefix_lens=inp.prefix_lens,
        max_seqlen_q=inp.max_seqlen_q, cu_seqblocks_q=inp.cu_seqblocks_q,
        max_seqblock_q=inp.max_seqblock_q,
    )
    if variant == "orig":
        return lambda: flash_prefill_with_gqa_share_sparse(**common)
    if variant == "nomod":
        return lambda: flash_prefill_with_gqa_share_sparse_nomod(**common)
    if variant == "paged":
        return lambda: flash_prefill_with_gqa_share_sparse_paged(
            **common, page_size=page_size
        )
    raise ValueError(f"unknown variant {variant!r}")


def _decode_runner(variant: str, inp, topk_idx, page_size: int):
    common = dict(
        q=inp.q, sink=None, k_cache=inp.k_cache, v_cache=inp.v_cache,
        req_to_token=inp.req_to_token, seq_lens=inp.seq_lens,
        slot_ids=inp.slot_ids, block_size=BLOCK_SIZE, topk_idx=topk_idx,
    )
    if variant == "orig":
        return lambda: flash_decode_with_gqa_share_sparse(**common)
    if variant == "nomod":
        return lambda: flash_decode_with_gqa_share_sparse_nomod(**common)
    if variant == "paged":
        return lambda: flash_decode_with_gqa_share_sparse_paged(
            **common, page_size=page_size
        )
    raise ValueError(f"unknown variant {variant!r}")


def build_point(cfg, phase: str, ctx: int, batch: int, dev, gen):
    """Inputs + per-variant runners for one (phase, ctx, batch) point."""
    if phase == "prefill":
        chunk = min(PREFILL_CHUNK, ctx)
        inp = build_prefill_inputs(
            cfg, batch_size=batch, context_len=ctx, chunk_len=chunk, device=dev
        )
        abs_pos = (torch.arange(chunk, device=dev) + (ctx - chunk)).repeat(batch)
        rows = int(inp.cu_seqblocks_q[-1].item())
        abs_pos = abs_pos[:rows]
        topk_idx = _rand_block_idx(abs_pos, cfg.num_kv_heads, BLOCK_TOPK, gen)
        return inp, topk_idx, rows, _prefill_runner
    inp = build_decode_inputs(cfg, batch_size=batch, context_len=ctx, device=dev)
    abs_pos = torch.full((batch,), ctx - 1, device=dev)
    topk_idx = _rand_block_idx(abs_pos, cfg.num_kv_heads, BLOCK_TOPK, gen)
    return inp, topk_idx, batch, _decode_runner


def run_point(cfg, phase, ctx, batch, variants, dev, gen, iters, page_size) -> list[dict]:
    inp, topk_idx, rows, runner_for = build_point(cfg, phase, ctx, batch, dev, gen)
    baseline_out = None
    baseline_ms = None
    out_rows: list[dict] = []

    for variant in variants:
        run = runner_for(variant, inp, topk_idx, page_size)
        out = run()
        if variant == BASELINE:
            baseline_out = out.float().clone()
            max_diff = 0.0
        else:
            max_diff = float((baseline_out - out.float()).abs().max().item())
        finite = bool(torch.isfinite(out.float()).all())
        timing = bench_cuda(run, warmup=max(3, iters // 4), iters=iters)
        if variant == BASELINE:
            baseline_ms = timing.median_ms

        # KV the kernel must gather: K+V of the selected budget per
        # (query row, kv head). Identical across variants by construction.
        kv_bytes = rows * cfg.num_kv_heads * TOKEN_BUDGET * cfg.head_dim * 2 * 2
        out_rows.append({
            "phase": phase,
            "variant": variant,
            "context_len": ctx,
            "context_label": _ctx_label(ctx),
            "batch_size": batch,
            "rows": rows,
            "page_size": page_size,
            "block_size": BLOCK_SIZE,
            "budget_tokens": TOKEN_BUDGET,
            "latency_mean_ms": round(timing.mean_ms, 6),
            "latency_median_ms": round(timing.median_ms, 6),
            "latency_min_ms": round(timing.min_ms, 6),
            "latency_p90_ms": round(timing.p90_ms, 6),
            "speedup_vs_orig": round(baseline_ms / timing.median_ms, 4),
            "max_abs_diff_vs_orig": max_diff,
            "bitwise_identical": max_diff == 0.0,
            "selected_kv_bytes": kv_bytes,
            "gather_gb_s": round(kv_bytes / (timing.median_ms / 1e3) / 1e9, 2),
            "output_finite": finite,
            "status": "ok",
        })
        del out
    del inp, topk_idx, baseline_out
    torch.cuda.empty_cache()
    return out_rows


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


def make_plots(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = sorted({r["variant"] for r in rows}, key=DEFAULT_VARIANTS.index)
    colors = {"orig": "#5145CC", "nomod": "#C25708", "paged": "#0093A3"}
    markers = {"orig": "o", "nomod": "s", "paged": "^"}

    for phase in sorted({r["phase"] for r in rows}):
        pr = [r for r in rows if r["phase"] == phase]
        if not pr:
            continue
        batches = sorted({r["batch_size"] for r in pr})
        fig, axes = plt.subplots(1, len(batches), figsize=(6 * len(batches), 4.4),
                                 squeeze=False)
        for ax, bs in zip(axes[0], batches):
            for v in variants:
                pts = sorted(
                    [r for r in pr if r["variant"] == v and r["batch_size"] == bs],
                    key=lambda r: r["context_len"],
                )
                if not pts:
                    continue
                ax.plot([p["context_len"] for p in pts],
                        [p["latency_median_ms"] for p in pts],
                        marker=markers[v], color=colors[v], label=v, lw=2, ms=6)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("context length")
            ax.set_ylabel("median latency (ms)")
            ax.set_title(f"{phase} · batch {bs}")
            ax.grid(alpha=0.3, which="both")
            ax.legend()
        fig.suptitle(f"block-sparse {phase}: original vs paged "
                     f"(budget {TOKEN_BUDGET}, block {BLOCK_SIZE}, page {rows[0]['page_size']})")
        fig.tight_layout()
        fig.savefig(out_dir / f"{phase}_latency.png", dpi=140)
        plt.close(fig)

    # speedup vs orig
    phases = sorted({r["phase"] for r in rows})
    fig, axes = plt.subplots(1, len(phases), figsize=(6 * len(phases), 4.4),
                             squeeze=False)
    for ax, phase in zip(axes[0], phases):
        pr = [r for r in rows if r["phase"] == phase]
        for v in variants:
            if v == BASELINE:
                continue
            for bs in sorted({r["batch_size"] for r in pr}):
                pts = sorted(
                    [r for r in pr if r["variant"] == v and r["batch_size"] == bs],
                    key=lambda r: r["context_len"],
                )
                if not pts:
                    continue
                label = v if len(set(r["batch_size"] for r in pr)) == 1 else f"{v} bs{bs}"
                ax.plot([p["context_len"] for p in pts],
                        [p["speedup_vs_orig"] for p in pts],
                        marker=markers[v], color=colors[v], label=label, lw=2, ms=6,
                        alpha=0.55 + 0.45 * (bs == max(r["batch_size"] for r in pr)))
        ax.axhline(1.0, color="#888", ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context length")
        ax.set_ylabel("speedup vs orig (x)")
        ax.set_title(phase)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("speedup over the original block-sparse kernel")
    fig.tight_layout()
    fig.savefig(out_dir / "speedup.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------


def parse_args(argv=None):
    def _int_list(s):
        return [int(x) for x in s.split(",") if x]

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--context-lens", type=_int_list, default=DEFAULT_CTXS)
    p.add_argument("--phases", type=lambda s: [x for x in s.split(",") if x],
                   default=["prefill", "decode"])
    p.add_argument("--variants", type=lambda s: [x for x in s.split(",") if x],
                   default=DEFAULT_VARIANTS,
                   help="subset of: orig,nomod,paged (orig is the baseline)")
    p.add_argument("--decode-batch-sizes", type=_int_list, default=[1, 32, 128])
    p.add_argument("--prefill-batch-size", type=int, default=1)
    p.add_argument("--page-size", type=int, default=BLOCK_SIZE,
                   help="KV page size; the paged fast path engages only when "
                        "this equals block_size (128), else it falls back")
    p.add_argument("--prefill-iters", type=int, default=30)
    p.add_argument("--decode-iters", type=int, default=200)
    p.add_argument("--wait-for-idle", type=float, default=0.0)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("-o", "--out", type=Path,
                   default=Path(__file__).resolve().parent
                   / "results" / "block_sparse_kernel_comparison")
    args = p.parse_args(argv)
    if BASELINE not in args.variants:
        p.error(f"--variants must include the {BASELINE!r} baseline")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark.")
        return 1

    cfg = m3_config()
    info = gpu_info()
    print("block-sparse attention — original vs paged")
    print(f"  device : {info['gpu']} (sm{info['sm']}, {info['memory_gb']} GB)")
    print(f"  config : {cfg.shape_tag()}")
    print(f"  budget : top-{BLOCK_TOPK} x {BLOCK_SIZE} = {TOKEN_BUDGET} tokens/query")
    print(f"  page   : {args.page_size} "
          f"({'fast path active' if args.page_size == BLOCK_SIZE else 'fallback'})")
    print(f"  variants: {', '.join(args.variants)}\n")

    if args.wait_for_idle > 0:
        wait_for_idle(args.wait_for_idle)
    warn_if_contended()

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    rows: list[dict] = []

    for phase in args.phases:
        iters = args.prefill_iters if phase == "prefill" else args.decode_iters
        batches = ([args.prefill_batch_size] if phase == "prefill"
                   else args.decode_batch_sizes)
        for batch in batches:
            for ctx in args.context_lens:
                tag = f"{phase} ctx={_ctx_label(ctx)} bs={batch}"
                try:
                    pts = run_point(cfg, phase, ctx, batch, args.variants,
                                    dev, gen, iters, args.page_size)
                except torch.cuda.OutOfMemoryError:
                    print(f"  {tag:<28} OOM — skipped")
                    torch.cuda.empty_cache()
                    continue
                rows.extend(pts)
                summary = "  ".join(
                    f"{p['variant']}={p['latency_median_ms']:.4f}ms"
                    f"({p['speedup_vs_orig']:.2f}x)" for p in pts
                )
                worst = max(p["max_abs_diff_vs_orig"] for p in pts)
                flag = "" if worst == 0.0 else f"  [maxdiff {worst:.1e}]"
                print(f"  {tag:<28} {summary}{flag}")

    args.out.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = write_results(rows, args.out, "raw")
    # write_results names both files after `name`; the sibling result sets use
    # raw.json + summary.csv, so rename the csv to match that convention.
    summary_path = args.out / "summary.csv"
    csv_path.replace(summary_path)
    print(f"\nwrote {json_path}\n      {summary_path}")

    if not args.no_plots:
        try:
            make_plots(rows, args.out)
            print(f"      {args.out}/*.png")
        except Exception as err:  # plotting must never lose the measurements
            print(f"  (plotting failed: {err})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
