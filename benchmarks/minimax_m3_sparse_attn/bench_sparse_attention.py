#!/usr/bin/env python3
"""Sparse attention kernels only — no indexer, no selector.

Benchmarks the main GQA attention restricted to a fixed 2,048-token budget,
fed with pre-generated random (but causally valid) selections:

  token   ``gqa_token_sparse_attn`` — per-query token positions,
          topk_idx [kv_heads, rows, 2048]
  block   ``flash_prefill_with_gqa_share_sparse`` /
          ``flash_decode_with_gqa_share_sparse`` — per-query block ids,
          topk_idx [kv_heads, rows, 16] with block_size 128 (16 x 128 = the
          same 2,048-token budget)

Both read exactly the same number of selected KV bytes per query; the
difference under test is the *access pattern*: block gathers 16 contiguous
128-token runs (one page each at page_size 128), token gathers 2,048
scattered positions. Context length only changes how far apart the selected
tokens sit in the paged pool.

MiniMax-M3 shapes: 64 q heads / 4 kv heads / d128, bf16, bs=1,
prefill = 8,192-token extend chunk.

    CUDA_VISIBLE_DEVICES=4 python bench_sparse_attention.py
    python bench_sparse_attention.py --context-lens 16384,131072 --phases decode

Outputs (default --out results/bench_sparse_attention): raw.json,
summary.csv, sparse_attn_prefill.png, sparse_attn_decode.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402
    bench_cuda,
    build_decode_inputs,
    build_prefill_inputs,
    measure_transient_bytes,
    wait_for_idle,
    warn_if_contended,
    write_results,
)
from m3_config import m3_config  # noqa: E402

from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (  # noqa: E402
    flash_decode_with_gqa_share_sparse,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse import (  # noqa: E402
    flash_prefill_with_gqa_share_sparse,
)
from sglang.kernels.ops.attention.minimax_sparse.token.sparse_attn import (  # noqa: E402
    gqa_token_sparse_attn,
)

DEFAULT_CTXS = [4096, 8192, 16384, 65536, 131072, 524288, 1048576]
TOKEN_BUDGET = 2048
BLOCK_SIZE = 128
BLOCK_TOPK = TOKEN_BUDGET // BLOCK_SIZE  # 16


def _ctx_label(n: int) -> str:
    return f"{n // 1024}k" if n < 1 << 20 else f"{n >> 20}M"


def _rand_token_idx(abs_pos: torch.Tensor, kv_heads: int, budget: int,
                    gen: torch.Generator) -> torch.Tensor:
    """[kv_heads, rows, budget] int32 positions, uniform in each row's causal
    range (duplicates possible — irrelevant for kernel timing, matches the
    unsorted output convention of the real selectors)."""
    rows = abs_pos.shape[0]
    u = torch.rand(kv_heads, rows, budget, device=abs_pos.device, generator=gen)
    hi = (abs_pos + 1).to(torch.float32).view(1, rows, 1)
    return (u * hi).to(torch.int32)


def _rand_block_idx(abs_pos: torch.Tensor, kv_heads: int, topk: int,
                    gen: torch.Generator) -> torch.Tensor:
    rows = abs_pos.shape[0]
    u = torch.rand(kv_heads, rows, topk, device=abs_pos.device, generator=gen)
    hi = ((abs_pos + BLOCK_SIZE) // BLOCK_SIZE).to(torch.float32).view(1, rows, 1)
    return (u * hi).to(torch.int32)


def build_prefill(cfg, ctx: int, gran: str, dev, gen):
    chunk = min(8192, ctx)
    inp = build_prefill_inputs(
        cfg, batch_size=1, context_len=ctx, chunk_len=chunk, device=dev
    )
    kv_heads = cfg.num_kv_heads
    abs_pos = torch.arange(chunk, device=dev) + (ctx - chunk)
    if gran == "token":
        topk_idx = _rand_token_idx(abs_pos, kv_heads, TOKEN_BUDGET, gen)
        q_slot_ids = torch.zeros(chunk, dtype=torch.int64, device=dev)

        def run():
            return gqa_token_sparse_attn(
                q=inp.q, k_cache=inp.k_cache, v_cache=inp.v_cache,
                req_to_token=inp.req_to_token, q_slot_ids=q_slot_ids,
                topk_idx=topk_idx,
            )
    else:
        topk_idx = _rand_block_idx(abs_pos, kv_heads, BLOCK_TOPK, gen)

        def run():
            return flash_prefill_with_gqa_share_sparse(
                q=inp.q, k_cache=inp.k_cache, v_cache=inp.v_cache, sink=None,
                req_to_token=inp.req_to_token, slot_ids=inp.slot_ids,
                topk_idx=topk_idx, block_size_q=1, block_size_k=BLOCK_SIZE,
                cu_seqlens=inp.cu_seqlens, seq_lens=inp.seq_lens,
                prefix_lens=inp.prefix_lens, max_seqlen_q=inp.max_seqlen_q,
                cu_seqblocks_q=inp.cu_seqblocks_q,
                max_seqblock_q=inp.max_seqblock_q,
            )

    rows = chunk
    return inp, run, rows


def build_decode(cfg, ctx: int, gran: str, dev, gen):
    inp = build_decode_inputs(cfg, batch_size=1, context_len=ctx, device=dev)
    kv_heads = cfg.num_kv_heads
    abs_pos = torch.full((1,), ctx - 1, device=dev)
    if gran == "token":
        topk_idx = _rand_token_idx(abs_pos, kv_heads, TOKEN_BUDGET, gen)
        q_slot_ids = torch.zeros(1, dtype=torch.int64, device=dev)

        def run():
            return gqa_token_sparse_attn(
                q=inp.q, k_cache=inp.k_cache, v_cache=inp.v_cache,
                req_to_token=inp.req_to_token, q_slot_ids=q_slot_ids,
                topk_idx=topk_idx,
            )
    else:
        topk_idx = _rand_block_idx(abs_pos, kv_heads, BLOCK_TOPK, gen)

        def run():
            return flash_decode_with_gqa_share_sparse(
                q=inp.q, sink=None, k_cache=inp.k_cache, v_cache=inp.v_cache,
                req_to_token=inp.req_to_token, seq_lens=inp.seq_lens,
                slot_ids=inp.slot_ids, block_size=BLOCK_SIZE,
                topk_idx=topk_idx,
            )

    return inp, run, 1


def run_point(cfg, phase: str, gran: str, ctx: int, dev, gen, iters: int) -> dict:
    builder = build_prefill if phase == "prefill" else build_decode
    inp, run, rows = builder(cfg, ctx, gran, dev, gen)
    out = run()
    finite = bool(torch.isfinite(out.float()).all()) if torch.is_tensor(out) else True
    timing = bench_cuda(run, warmup=max(3, iters // 4), iters=iters)
    # Bytes the kernel must gather: K+V of the selected 2,048 tokens per
    # (query row, kv head), plus q/o. Identical for both granularities —
    # the access-pattern difference is what the latency measures.
    kv_bytes = rows * cfg.num_kv_heads * TOKEN_BUDGET * cfg.head_dim * 2 * 2
    qo_bytes = rows * cfg.num_q_heads * cfg.head_dim * 2 * 2
    row = {
        "phase": phase, "impl": gran, "context_len": ctx, "batch_size": 1,
        "rows": rows, "budget_tokens": TOKEN_BUDGET,
        "latency_median_ms": round(timing.median_ms, 6),
        "latency_min_ms": round(timing.min_ms, 6),
        "selected_kv_bytes": kv_bytes,
        "qo_bytes": qo_bytes,
        "gather_gb_s": round(kv_bytes / (timing.median_ms / 1e3) / 1e9, 2),
        "transient_bytes": measure_transient_bytes(run),
        "output_finite": finite,
        "status": "ok",
    }
    print(f"  {phase:<7} {gran:<6} ctx={ctx:<8} {timing.median_ms:9.4f} ms  "
          f"gather {row['gather_gb_s']:8.1f} GB/s  finite={finite}")
    del inp
    torch.cuda.empty_cache()
    return row


def make_plots(rows: list[dict], out: Path, ctxs: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"block": "#898781", "token": "#2a78d6"}
    for phase in ("prefill", "decode"):
        sub = [r for r in rows if r["phase"] == phase and r["status"] == "ok"]
        if not sub:
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=160)
        for gran, c in colors.items():
            pts = sorted((r["context_len"], r["latency_median_ms"],
                          r["gather_gb_s"]) for r in sub if r["impl"] == gran)
            xs = [p[0] for p in pts]
            ax1.plot(xs, [p[1] for p in pts], marker="o", markersize=5,
                     linewidth=1.8, color=c,
                     linestyle="--" if gran == "block" else "-",
                     label=f"{gran} ({'16x128 blocks' if gran == 'block' else '2048 tokens'})")
            ax2.plot(xs, [p[2] for p in pts], marker="o", markersize=5,
                     linewidth=1.8, color=c,
                     linestyle="--" if gran == "block" else "-", label=gran)
        for ax, ylab, title, logy in (
            (ax1, "median ms (log)", f"{phase}: sparse attention latency", True),
            (ax2, "selected-KV gather GB/s", f"{phase}: effective gather bandwidth", False),
        ):
            ax.set_xscale("log", base=2)
            if logy:
                ax.set_yscale("log")
            ax.set_xticks(ctxs)
            ax.set_xticklabels([_ctx_label(c) for c in ctxs])
            ax.minorticks_off()
            ax.set_xlabel("context length")
            ax.set_ylabel(ylab)
            ax.set_title(title, loc="left", fontsize=11)
            ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
            ax.legend(frameon=False, fontsize=9)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f"sparse_attn_{phase}.png")
        plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--context-lens", default=",".join(map(str, DEFAULT_CTXS)))
    p.add_argument("--phases", default="prefill,decode")
    p.add_argument("--prefill-iters", type=int, default=10)
    p.add_argument("--decode-iters", type=int, default=30)
    p.add_argument("--wait-for-idle", type=float, default=600.0)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent
                   / "results" / "bench_sparse_attention")
    args = p.parse_args(argv)
    ctxs = [int(x) for x in args.context_lens.split(",") if x.strip()]
    phases = [s.strip() for s in args.phases.split(",") if s.strip()]

    torch.cuda.init()
    wait_for_idle(args.wait_for_idle)
    warn_if_contended()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = m3_config()
    print("MiniMax-M3 sparse attention kernels only (indices pre-generated)")
    print(f"  device : {torch.cuda.get_device_name(0)}")
    print(f"  config : {cfg.shape_tag()}  budget {TOKEN_BUDGET} tokens "
          f"({BLOCK_TOPK} blocks x {BLOCK_SIZE})")

    rows: list[dict] = []
    for ctx in ctxs:
        print(f"--- ctx={ctx} ---")
        for phase in phases:
            iters = args.prefill_iters if phase == "prefill" else args.decode_iters
            for gran in ("block", "token"):
                try:
                    rows.append(run_point(cfg, phase, gran, ctx, dev, gen, iters))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    rows.append({"phase": phase, "impl": gran,
                                 "context_len": ctx, "status": "oom"})
                    print(f"  {phase} {gran} ctx={ctx} OOM")

    (args.out / "raw.json").write_text(json.dumps(rows, indent=1))
    ok = [r for r in rows if r.get("status") == "ok"]
    if ok:
        fields = list(ok[0].keys())
        with (args.out / "summary.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(ok)
        make_plots(rows, args.out, ctxs)
    print(f"\nwrote {args.out}/raw.json, summary.csv + plots")
    return 0 if all(r.get("status") == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
