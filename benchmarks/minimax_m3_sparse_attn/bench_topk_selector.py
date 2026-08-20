#!/usr/bin/env python3
"""Top-k selector comparison: token-sparse vs block-sparse, selection only.

Feeds pre-materialized random score matrices to the *selection* stage of each
granularity — no scoring kernel involved — at MiniMax-M3 shapes:

  token   exact top-2048 token positions per query row over a [4 heads, rows,
          ctx] fp32 matrix, processed in the production 512 MiB query chunks.
          Two selector backends: the production one (FlashInfer exact top-k,
          via index_score._topk_positions) and plain torch.topk.
  block   top-16 block ids per row over [4, rows, ctx/128] — both the budget
          and the row width are 128x smaller (the block size). Uses the
          production JIT radix selector (minimax_prefill_topk) up to 4,096
          blocks and the Triton bitonic fallback (_topk_index_kernel) above,
          exactly like flash_prefill_with_topk_index does.

Latency is per full 8,192-token extend step (token chunks are looped inside
the timed region). L2 is flushed between iterations.

    CUDA_VISIBLE_DEVICES=4 python bench_topk_selector.py
    python bench_topk_selector.py --context-lens 16384,131072

Outputs (default --out results/selector_comparison): raw.json, summary.csv,
selector_latency.png, selector_bandwidth.png, selector_matrix_bytes.png.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import torch
import triton
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402
    bench_cuda,
    measure_transient_bytes,
    wait_for_idle,
    warn_if_contended,
)

from sglang.kernels.ops.attention.minimax_decode_topk import (  # noqa: E402
    minimax_prefill_topk,
)
from sglang.kernels.ops.attention.minimax_sparse.common.utils import (  # noqa: E402
    get_cu_seqblocks,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (  # noqa: E402
    _topk_index_kernel,
)
from sglang.kernels.ops.attention.minimax_sparse.token.index_score import (  # noqa: E402
    _topk_positions,
    plan_query_chunk,
)

DEFAULT_CTXS = [4096, 8192, 16384, 65536, 131072, 524288, 1048576]
HEADS = 4
CHUNK = 8192
TOKEN_TOPK = 2048
BLOCK_SIZE = 128
BLOCK_TOPK = TOKEN_TOPK // BLOCK_SIZE  # 16
INIT_BLOCKS, LOCAL_BLOCKS = 1, 2


def _ctx_label(n: int) -> str:
    return f"{n // 1024}k" if n < 1 << 20 else f"{n >> 20}M"


# ---------------------------------------------------------------------------
# builders: (fn to time, logical read bytes, live buffer bytes, output bytes)
# ---------------------------------------------------------------------------


def build_token(ctx: int, backend: str, dev) -> tuple[Callable, int, int, int]:
    chunk = min(CHUNK, ctx)
    _, chunk_rows = plan_query_chunk(
        batch_size=1, max_seqlen_q=chunk, max_seqlen_k=ctx, num_idx_heads=HEADS
    )
    n_chunks = -(-chunk // chunk_rows)
    scores = torch.randn(HEADS, chunk_rows, ctx, dtype=torch.float32, device=dev)

    if backend == "flashinfer":
        def run():
            for _ in range(n_chunks):
                _topk_positions(scores, TOKEN_TOPK)
    else:  # torch.topk
        flat = scores.view(-1, ctx)

        def run():
            for _ in range(n_chunks):
                torch.topk(flat, min(TOKEN_TOPK, ctx), dim=-1, sorted=False)

    logical = HEADS * chunk * ctx * 4  # every score read once per extend
    live = scores.numel() * 4
    out_bytes = HEADS * chunk * TOKEN_TOPK * 4
    return run, logical, live, out_bytes


def build_block(ctx: int, dev) -> tuple[Callable, int, int, int]:
    chunk = min(CHUNK, ctx)
    prefix = ctx - chunk
    width = -(-ctx // BLOCK_SIZE)
    cu_seqlens = torch.tensor([0, chunk], dtype=torch.int32, device=dev)
    prefix_lens = torch.tensor([prefix], dtype=torch.int32, device=dev)
    cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
        cu_seqlens, chunk, 1, BLOCK_SIZE, [chunk]
    )
    scores = torch.randn(HEADS, chunk, width, dtype=torch.float32, device=dev)

    if BLOCK_TOPK <= 32 and width <= 4096:  # the wrapper's radix-path gate
        def run():
            minimax_prefill_topk(
                scores, cu_seqlens, cu_seqblocks_q, prefix_lens,
                max_seqblock_q, all_seqblock_q, 1, BLOCK_SIZE, BLOCK_TOPK,
                INIT_BLOCKS, LOCAL_BLOCKS,
            )
    else:  # Triton bitonic fallback, launched as the production wrapper does
        def run():
            topk_idx = torch.full(
                (HEADS, all_seqblock_q, BLOCK_TOPK), -1,
                dtype=torch.int32, device=dev,
            )
            grid = (max_seqblock_q, 1, HEADS)
            _topk_index_kernel[grid](
                scores, topk_idx, 1, BLOCK_SIZE,
                cu_seqlens, cu_seqblocks_q, prefix_lens,
                BLOCK_TOPK, INIT_BLOCKS, LOCAL_BLOCKS,
                scores.stride(0), scores.stride(1), scores.stride(2),
                topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
                MASK_INIT=False, MASK_LOCAL=False,
            )

    logical = scores.numel() * 4
    out_bytes = HEADS * chunk * BLOCK_TOPK * 4
    return run, logical, logical, out_bytes


# ---------------------------------------------------------------------------


def run_point(name: str, ctx: int, builder, iters: int) -> dict:
    run, logical, live, out_bytes = builder
    timing = bench_cuda(run, warmup=max(3, iters // 4), iters=iters)
    row = {
        "impl": name,
        "context_len": ctx,
        "topk": BLOCK_TOPK if name == "block" else TOKEN_TOPK,
        "row_width": -(-ctx // BLOCK_SIZE) if name == "block" else ctx,
        "latency_median_ms": round(timing.median_ms, 6),
        "latency_min_ms": round(timing.min_ms, 6),
        "matrix_logical_bytes": logical,
        "matrix_live_bytes": live,
        "output_bytes": out_bytes,
        "read_gb_s": round(logical / (timing.median_ms / 1e3) / 1e9, 2),
        "transient_bytes": measure_transient_bytes(run),
    }
    print(f"  {name:<16} ctx={ctx:<8} {timing.median_ms:9.4f} ms  "
          f"read {row['read_gb_s']:8.1f} GB/s  "
          f"matrix {logical / 2**20:9.1f} MiB  ws {row['transient_bytes'] / 2**20:7.1f} MiB")
    return row


def make_plots(rows: list[dict], out: Path, ctxs: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"token_flashinfer": "#2a78d6", "token_torch": "#1baf7a",
              "block": "#898781"}

    def line(field, ylabel, title, fname, logy=True):
        fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
        for impl, c in colors.items():
            pts = sorted((r["context_len"], r[field]) for r in rows
                         if r["impl"] == impl)
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                    markersize=5, linewidth=1.8, color=c,
                    linestyle="--" if impl == "block" else "-",
                    label=impl + (" (k=16, w/128)" if impl == "block"
                                  else " (k=2048)"))
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(ctxs)
        ax.set_xticklabels([_ctx_label(c) for c in ctxs])
        ax.minorticks_off()
        ax.set_xlabel("context length")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / fname)
        plt.close(fig)

    line("latency_median_ms", "median ms per 8k-token extend (log)",
         "Top-k selector latency (selection only)", "selector_latency.png")
    line("read_gb_s", "effective matrix read GB/s",
         "Selector effective read bandwidth", "selector_bandwidth.png",
         logy=False)
    line("matrix_logical_bytes", "bytes read per extend (log)",
         "Score matrix bytes the selector must read",
         "selector_matrix_bytes.png")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context-lens",
                   default=",".join(map(str, DEFAULT_CTXS)))
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--wait-for-idle", type=float, default=600.0)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent
                   / "results" / "selector_comparison")
    args = p.parse_args(argv)
    ctxs = [int(x) for x in args.context_lens.split(",") if x.strip()]

    torch.cuda.init()
    wait_for_idle(args.wait_for_idle)
    warn_if_contended()
    dev = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    print("MiniMax-M3 top-k selector comparison (selection stage only)")
    print(f"  device : {torch.cuda.get_device_name(0)}")
    print(f"  token  : top-{TOKEN_TOPK} over ctx-wide rows, 512 MiB chunks")
    print(f"  block  : top-{BLOCK_TOPK} over ctx/{BLOCK_SIZE}-wide rows")

    rows = []
    for ctx in ctxs:
        print(f"--- ctx={ctx} ---")
        iters = max(5, args.iters // 4) if ctx >= 524288 else args.iters
        rows.append(run_point("token_flashinfer", ctx,
                              build_token(ctx, "flashinfer", dev), iters))
        rows.append(run_point("token_torch", ctx,
                              build_token(ctx, "torch", dev), iters))
        rows.append(run_point("block", ctx, build_block(ctx, dev), iters))
        torch.cuda.empty_cache()

    (args.out / "raw.json").write_text(json.dumps(rows, indent=1))
    fields = list(rows[0].keys())
    with (args.out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    make_plots(rows, args.out, ctxs)
    print(f"\nwrote {args.out}/raw.json, summary.csv + 3 plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
