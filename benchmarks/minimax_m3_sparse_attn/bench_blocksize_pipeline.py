#!/usr/bin/env python3
"""block_size x context sweep of the FULL sparse pipeline, two step-3 kernels.

Companion to ``bench_pagesize_pipeline.py``. That one holds block_size at 128 and
sweeps page_size; this one holds page_size and sweeps the *sparse block size*.

The token budget is held constant (``topk_blocks = budget // block_size``) so the
sweep isolates block granularity rather than also changing how much KV is read --
the same convention ``bench_kernels.py`` uses for its block_size sweep.

Note the fast path engages whenever ``block_size`` divides ``page_size``, so at
page_size 128 every block size in {16, 32, 64, 128} qualifies.

    CUDA_VISIBLE_DEVICES=7 python bench_blocksize_pipeline.py --wait-for-idle 5

Outputs (default --out results/blocksize_sweep): raw.json, summary.csv,
latency_vs_blocksize.png, speedup_vs_blocksize.png, stage_vs_blocksize.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_kernels import make_prefill_fn  # noqa: E402
from bench_pagesize_pipeline import (  # noqa: E402
    _attn_out,
    _ctx_label,
    _select,
)
from harness import (  # noqa: E402
    STAGE_ORDER,
    bench_cuda,
    build_prefill_inputs,
    gpu_info,
    profile_breakdown,
    wait_for_idle,
    warn_if_contended,
    write_results,
)
from m3_config import m3_config  # noqa: E402

DEFAULT_BLOCKS = [16, 32, 64, 128]
DEFAULT_CTXS = [8192, 32768, 131072, 524288]
VARIANTS = ("orig", "paged")


def run_cell(base, block_size, page_size, ctx, chunk, iters, profile_iters):
    # Hold the token budget constant so this isolates block granularity.
    topk = max(base.init_blocks + base.local_blocks + 1, base.token_budget // block_size)
    cfg = base.replace(block_size=block_size, topk_blocks=topk, page_size=page_size)
    inp = build_prefill_inputs(cfg, batch_size=1, context_len=ctx, chunk_len=chunk)
    fast = page_size >= block_size and page_size % block_size == 0
    rows, base_ms, ref, floor = [], None, None, None

    for variant in VARIANTS:
        _select(variant, page_size)
        fn = make_prefill_fn(cfg, inp)
        first = _attn_out(fn())
        timing = bench_cuda(fn, warmup=max(3, iters // 4), iters=iters)
        stages, _ = profile_breakdown(fn, iters=profile_iters, warmup=3)
        if variant == "orig":
            base_ms, ref = timing.median_ms, first.clone()
            floor = float((_attn_out(fn()) - ref).abs().max().item())
        row = {
            "variant": variant, "block_size": block_size, "topk_blocks": topk,
            "token_budget": topk * block_size, "page_size": page_size,
            "context_len": ctx, "context_label": _ctx_label(ctx), "chunk_len": chunk,
            "fast_path": variant == "paged" and fast,
            "latency_median_ms": round(timing.median_ms, 6),
            "latency_min_ms": round(timing.min_ms, 6),
            "speedup_vs_orig": round(base_ms / timing.median_ms, 4),
            "max_abs_diff_vs_orig": float((first - ref).abs().max().item()),
            "nondeterminism_floor": floor,
            "status": "ok",
        }
        for st in STAGE_ORDER:
            row[f"stage_{st}_ms"] = round(stages.get(st, 0.0), 6)
        rows.append(row)
        del fn, first
    del inp, ref
    torch.cuda.empty_cache()
    return rows


def make_plots(rows, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    blocks = sorted({r["block_size"] for r in rows})
    ctxs = sorted({r["context_len"] for r in rows})
    lut = {(r["variant"], r["block_size"], r["context_len"]): r for r in rows}
    C = {"orig": "#5145CC", "paged": "#0093A3"}

    # 1. latency vs block_size, one line per context, both variants
    fig, axes = plt.subplots(1, len(ctxs), figsize=(3.5 * len(ctxs) + 1, 4.2),
                             squeeze=False, sharey=False)
    for ax, c in zip(axes[0], ctxs):
        for v in VARIANTS:
            ys = [lut[(v, b, c)]["latency_median_ms"] for b in blocks if (v, b, c) in lut]
            ax.plot(blocks[:len(ys)], ys, "-o", color=C[v], lw=2, ms=6,
                    label=v, alpha=1.0 if v == "paged" else 0.75)
        ax.set_xscale("log", base=2)
        ax.set_xticks(blocks, [str(b) for b in blocks])
        ax.set_xlabel("block_size")
        ax.set_title(f"ctx {_ctx_label(c)}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
    axes[0][0].set_ylabel("layer latency (ms)")
    axes[0][0].legend(fontsize=9)
    fig.suptitle("full sparse prefill layer vs sparse block size "
                 "(token budget held at 2,048; page_size 128)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_blocksize.png", dpi=140)
    plt.close(fig)

    # 2. speedup vs block_size
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(ctxs):
        ys = [lut[("paged", b, c)]["speedup_vs_orig"] for b in blocks if ("paged", b, c) in lut]
        ax.plot(blocks[:len(ys)], ys, "-o", lw=2, ms=6,
                color=cmap(0.12 + 0.72 * i / max(1, len(ctxs) - 1)),
                label=f"ctx {_ctx_label(c)}")
    ax.axhline(1.0, color="#888", ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(blocks, [str(b) for b in blocks])
    ax.set_xlabel("block_size")
    ax.set_ylabel("layer speedup, paged vs orig (×)")
    ax.set_title("paged speedup vs sparse block size")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "speedup_vs_blocksize.png", dpi=140)
    plt.close(fig)

    # 3. stage times vs block_size (orig), to show what block_size really moves
    stages = [("stage_indexer_score_ms", "indexer", "#5145CC"),
              ("stage_topk_select_ms", "top-k select", "#C25708"),
              ("stage_sparse_attn_ms", "sparse attn", "#0093A3")]
    fig, axes = plt.subplots(1, len(ctxs), figsize=(3.5 * len(ctxs) + 1, 4.2),
                             squeeze=False)
    for ax, c in zip(axes[0], ctxs):
        for key, name, col in stages:
            ys = [lut[("orig", b, c)][key] for b in blocks if ("orig", b, c) in lut]
            ax.plot(blocks[:len(ys)], ys, "-o", color=col, lw=2, ms=5, label=name)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(blocks, [str(b) for b in blocks])
        ax.set_xlabel("block_size")
        ax.set_title(f"ctx {_ctx_label(c)}", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0][0].set_ylabel("stage time (ms, log)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("original kernel — where block_size actually lands", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "stage_vs_blocksize.png", dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    def _ints(s):
        return [int(x) for x in s.split(",") if x]

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--block-sizes", type=_ints, default=DEFAULT_BLOCKS)
    p.add_argument("--context-lens", type=_ints, default=DEFAULT_CTXS)
    p.add_argument("--page-size", type=int, default=128)
    p.add_argument("--chunk-len", type=int, default=4096)
    p.add_argument("--prefill-iters", type=int, default=20)
    p.add_argument("--profile-iters", type=int, default=10)
    p.add_argument("--wait-for-idle", type=float, default=0.0)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("-o", "--out", type=Path,
                   default=Path(__file__).resolve().parent / "results" / "blocksize_sweep")
    args = p.parse_args(argv)

    if args.plot_only:
        import json
        make_plots(json.loads((args.out / "raw.json").read_text()), args.out)
        print(f"replotted from {args.out / 'raw.json'}")
        return 0
    if not torch.cuda.is_available():
        print("CUDA is required.")
        return 1

    base = m3_config().replace(granularity="block")
    info = gpu_info()
    print("full sparse pipeline — block_size x context sweep")
    print(f"  device : {info['gpu']} (sm{info['sm']})")
    print(f"  budget : held at {base.token_budget} tokens (topk = budget // block_size)")
    print(f"  page   : {args.page_size}  chunk={args.chunk_len}")
    if args.wait_for_idle > 0:
        wait_for_idle(args.wait_for_idle)
    warn_if_contended()

    rows = []
    for ctx in args.context_lens:
        chunk = min(args.chunk_len, ctx)
        print(f"\n=== ctx {_ctx_label(ctx)} ===")
        print(f"{'block':>6}{'topk':>6}{'fast':>6}{'orig ms':>10}{'paged ms':>10}"
              f"{'speedup':>9}   orig stages (idx|topk|attn)")
        for b in args.block_sizes:
            try:
                pair = run_cell(base, b, args.page_size, ctx, chunk,
                                args.prefill_iters, args.profile_iters)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{b:>6}   OOM — skipped")
                continue
            o, g = pair
            rows.extend(pair)
            print(f"{b:>6}{o['topk_blocks']:>6}{('ON' if g['fast_path'] else '-'):>6}"
                  f"{o['latency_median_ms']:>10.3f}{g['latency_median_ms']:>10.3f}"
                  f"{g['speedup_vs_orig']:>8.3f}x   "
                  f"{o['stage_indexer_score_ms']:>6.3f}|"
                  f"{o['stage_topk_select_ms']:>6.3f}|{o['stage_sparse_attn_ms']:>7.3f}")
    _select("orig", 0)

    args.out.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = write_results(rows, args.out, "raw")
    summary = args.out / "summary.csv"
    csv_path.replace(summary)
    print(f"\nwrote {json_path}\n      {summary}")
    if not args.no_plots:
        try:
            make_plots(rows, args.out)
            print(f"      {args.out}/*.png")
        except Exception as err:
            print(f"  (plotting failed: {err})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
