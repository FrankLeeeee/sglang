#!/usr/bin/env python3
"""Stage 2: how long does it take to LOAD the KV for one block?

The paged fast path has two separable stages:

  1. slot-compute  — turn a block id into ``slots`` via ``req_to_token``.
     Cost here scales with how many *pages* a block spans (M lookups, not 128).
     Measured in ``bench_pagesize_pipeline.py`` / the multipage variant.

  2. KV load       — use ``slots`` to fetch K and V out of ``k_cache``.
     THIS SCRIPT. Cost here depends on how *fragmented* ``slots`` is:
        page_size == block_size -> one run of 128 consecutive slots
        page_size == block/2    -> two runs of 64, far apart
        page_size == 1          -> 128 scattered singletons

The kernel does the two loads and nothing else -- no dots, no softmax. Run bases
are generated arithmetically in-kernel (three integer ops per run, against 64 KB
of K+V per block), so there is no slots table to read and nothing but the gather
pattern varies.

"Pool" below means ``max_slots``: the number of physical KV slots the gather
draws from, i.e. the size of the KV cache being indexed. It decides whether the
working set is served from L2 (60 MB on H200) or from DRAM.

Access pattern matches the real kernel exactly: K as [head_dim, tokens]
(transposed for tl.dot), V as [tokens, head_dim], one program per
(query row, kv head), looping over topk blocks.

    CUDA_VISIBLE_DEVICES=1 python bench_kv_fragmentation.py --wait-for-idle 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402
    bench_cuda, emit_plots, gpu_info, wait_for_idle, warn_if_contended, write_results,
)

NUM_Q, NUM_KV, HEAD_DIM = 64, 4, 128
BLOCK_K, TOPK = 128, 16
ROWS = 4096            # query tokens in the extend chunk
DT = torch.bfloat16


@triton.jit
def _gather_kv_kernel(
    k_ptr, v_ptr, out_ptr,
    stride_ks, stride_kh, stride_kd,
    stride_vs, stride_vh, stride_vd,
    num_kv_heads, topk, n_runs_mask,
    SPAN: tl.constexpr,          # tokens per consecutive run
    RUNS: tl.constexpr,          # runs per block  (SPAN * RUNS == BLOCK_N)
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)                      # one program per (row, kv head)
    kh = pid % num_kv_heads
    off_d = tl.arange(0, BLOCK_D)
    off_j = tl.arange(0, RUNS)
    off_s = tl.arange(0, SPAN)
    acc = tl.zeros((), dtype=tl.float32)
    for b in range(topk):
        # Pseudo-random aligned run starts, computed not fetched. n_runs is a
        # power of two so this is a mask, never a division.
        r = (pid.to(tl.int64) * 1103515245 + b * 1664525
             + off_j.to(tl.int64) * 1013904223) & n_runs_mask
        bases = r * SPAN
        slots = tl.reshape(bases[:, None] + off_s.to(tl.int64)[None, :], BLOCK_N)
        k = tl.load(k_ptr + slots[None, :] * stride_ks + kh * stride_kh
                    + off_d[:, None] * stride_kd)          # [D, N], as the real kernel
        v = tl.load(v_ptr + slots[:, None] * stride_vs + kh * stride_vh
                    + off_d[None, :] * stride_vd)          # [N, D]
        acc += tl.sum(k.to(tl.float32)) + tl.sum(v.to(tl.float32))
    tl.store(out_ptr + pid, acc)


def run_point(runs, pool_slots, dev, iters):
    span = BLOCK_K // runs
    n_runs = pool_slots // span
    assert n_runs & (n_runs - 1) == 0, "pool/span must be a power of two"
    n_prog = ROWS * NUM_KV
    k = torch.randn(pool_slots, NUM_KV, HEAD_DIM, dtype=DT, device=dev)
    v = torch.randn(pool_slots, NUM_KV, HEAD_DIM, dtype=DT, device=dev)
    out = torch.empty(n_prog, dtype=torch.float32, device=dev)

    def fn():
        _gather_kv_kernel[(n_prog,)](
            k, v, out,
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            NUM_KV, TOPK, n_runs - 1,
            SPAN=span, RUNS=runs, BLOCK_N=BLOCK_K, BLOCK_D=HEAD_DIM,
            num_warps=4, num_stages=2,          # pinned: no autotune confound
        )

    fn()
    torch.cuda.synchronize()
    t = bench_cuda(fn, warmup=max(3, iters // 4), iters=iters)
    kv_bytes = n_prog * TOPK * BLOCK_K * HEAD_DIM * 2 * 2
    pool_mib = pool_slots * NUM_KV * HEAD_DIM * 2 * 2 / 2**20
    row = {
        "pool_slots": pool_slots, "pool_mib": round(pool_mib, 1),
        "runs_per_block": runs, "span_tokens": span,
        "run_bytes_per_head": span * HEAD_DIM * 2,
        "latency_median_ms": round(t.median_ms, 6),
        "kv_bytes": kv_bytes,
        "gather_gb_s": round(kv_bytes / (t.median_ms / 1e3) / 1e9, 1),
        "pct_hbm_peak": round(kv_bytes / (t.median_ms / 1e3) / 1e9 / 4800 * 100, 1),
        "status": "ok",
    }
    del k, v, out
    torch.cuda.empty_cache()
    return row


def make_plots(rows, out_dir: Path) -> None:
    """Absolute latency and achieved bandwidth vs fragmentation.

    Deliberately NOT normalised: normalising each pool to its own 1-run point
    hides three different baselines behind one axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pools = sorted({r["pool_slots"] for r in rows})
    cmap = plt.get_cmap("viridis")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.7))
    for i, pool in enumerate(pools):
        rs = sorted([r for r in rows if r["pool_slots"] == pool],
                    key=lambda r: r["runs_per_block"])
        col = cmap(0.12 + 0.7 * i / max(1, len(pools) - 1))
        lbl = f"KV pool {pool // 1024}k slots = {rs[0]['pool_mib']:.0f} MiB"
        xs = [r["runs_per_block"] for r in rs]
        ax.plot(xs, [r["latency_median_ms"] for r in rs], "-o", color=col, lw=2, ms=5, label=lbl)
        ax2.plot(xs, [r["gather_gb_s"] for r in rs], "-o", color=col, lw=2, ms=5, label=lbl)
    ticks = sorted({r["runs_per_block"] for r in rows})
    for a in (ax, ax2):
        a.set_xscale("log", base=2)
        a.set_xticks(ticks, [f"{t}\n({BLOCK_K // t} tok)" for t in ticks], fontsize=8)
        a.set_xlabel("runs per block   (tokens per consecutive run)")
        a.grid(alpha=0.3)
        a.set_ylim(0, None)
        a.legend(fontsize=8, loc="lower right")
    ax.set_ylabel("time to load K+V for the whole sweep (ms)")
    ax.set_title("absolute KV-load time — no baseline, no normalisation", fontsize=11)
    ax2.axhline(4800, color="#C25708", ls="--", lw=1.4)
    ax2.text(ticks[0] * 1.05, 4950, "HBM peak ~ 4800 GB/s", color="#C25708", fontsize=8.5)
    ax2.set_ylabel("achieved gather bandwidth (GB/s)")
    ax2.set_title("above the line = served from L2, not DRAM", fontsize=11)
    fig.suptitle("Stage 2 — fetching K+V as the block splits into more, shorter runs",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fragmentation.png", dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool-slots", type=lambda s: [int(x) for x in s.split(",") if x],
                   default=[32768, 262144, 1048576],
                   help="max_slots: size of the KV slot pool being indexed")
    p.add_argument("--runs", type=lambda s: [int(x) for x in s.split(",") if x],
                   default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--wait-for-idle", type=float, default=0.0)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("-o", "--out", type=Path,
                   default=Path(__file__).resolve().parent / "results" / "kv_fragmentation")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is required.")
        return 1
    info = gpu_info()
    print("KV-load stage in isolation — fragmentation of `slots` only")
    print(f"  device : {info['gpu']} (sm{info['sm']})")
    print(f"  shape  : {ROWS} rows x {NUM_KV} kv heads, top-{TOPK} x {BLOCK_K}, d{HEAD_DIM} bf16")
    print("  note   : run bases generated in-kernel (no slots table); only the")
    print("           gather pattern varies. 'pool' = max_slots, the KV slot pool.\n")
    if args.wait_for_idle > 0:
        wait_for_idle(args.wait_for_idle)
    warn_if_contended()

    rows = []
    for pool in args.pool_slots:
        mib = pool * NUM_KV * HEAD_DIM * 2 * 2 / 2**20
        print(f"\n=== KV pool = {pool:,} slots = {mib:.0f} MiB of K+V "
              f"({'>' if mib > 60 else '<'} 60 MB L2) ===")
        print(f"{'runs/blk':>9}{'span':>6}{'B/run/head':>12}{'ms':>9}{'GB/s':>9}{'%HBM':>7}")
        for r in args.runs:
            try:
                row = run_point(r, pool, torch.device("cuda"), args.iters)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{r:>9}   OOM — skipped")
                continue
            rows.append(row)
            print(f"{r:>9}{row['span_tokens']:>6}{row['run_bytes_per_head']:>12}"
                  f"{row['latency_median_ms']:>9.3f}{row['gather_gb_s']:>9.1f}"
                  f"{row['pct_hbm_peak']:>6.0f}%")

    args.out.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = write_results(rows, args.out, "raw")
    summary = args.out / "summary.csv"
    csv_path.replace(summary)
    print(f"\nwrote {json_path}\n      {summary}")
    if not args.no_plots:
        emit_plots(make_plots, rows, args.out)
        print(f"      {args.out}/fragmentation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
