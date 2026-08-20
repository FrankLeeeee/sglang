#!/usr/bin/env python3
"""Stage 1: how much does resolving `slots` cost, as page_size varies?

Companion to ``bench_kv_fragmentation.py`` (stage 2, the KV load). Everything
downstream of ``slots`` is identical across the variants measured here, so the
differences are purely slot-compute.

Four measurements per page size, all against the SAME inputs:

  orig        the shipped kernel — no paged lookup exists in it at all.
              This is the baseline every speedup is quoted against.
  paged_off   the paged variant with its lookup DISABLED (guard forced to fail).
              A control: it should equal ``orig`` exactly, which is what proves
              the compile-time branch costs nothing when not taken.
  paged       the paged variant at this page_size — the current guard, which
              needs block_size | page_size and otherwise falls back.
  multipage   the M-page variant — resolves a block spanning M pages with M
              lookups instead of 128, so it degrades gracefully instead of
              cliffing to the fallback.

    CUDA_VISIBLE_DEVICES=1 python bench_stage1_lookup.py --wait-for-idle 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402
    bench_cuda, build_page_table, emit_plots, gpu_info, wait_for_idle,
    warn_if_contended, write_results,
)

from sglang.kernels.ops.attention.minimax_sparse.common.utils import (  # noqa: E402
    get_cu_seqblocks,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse import (  # noqa: E402
    flash_prefill_with_gqa_share_sparse as ORIG,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse_multipage import (  # noqa: E402
    flash_prefill_with_gqa_share_sparse_multipage as MULTI,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse_paged import (  # noqa: E402
    flash_prefill_with_gqa_share_sparse_paged as PAGED,
)

NUM_Q, NUM_KV, HEAD_DIM = 64, 4, 128
BLOCK_K, BLOCK_Q, TOPK = 128, 1, 16
# page_size that no guard accepts (128 % 3 != 0 and 3 % 128 != 0) -> lookup off
LOOKUP_OFF = 3


def build(ctx, chunk, page_size, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    torch.manual_seed(seed)
    prefix = ctx - chunk
    r2t, _ = build_page_table(batch_size=1, context_len=ctx,
                              page_size=page_size, device=dev, generator=g)
    max_slots = r2t.max().item() + 1
    max_slots = max(max_slots, ctx)
    k = torch.randn(max_slots, NUM_KV, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    v = torch.randn(max_slots, NUM_KV, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    q = torch.randn(chunk, NUM_Q, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    cs = torch.tensor([0, chunk], dtype=torch.int32, device=dev)
    sl = torch.full((1,), ctx, dtype=torch.int32, device=dev)
    pf = torch.full((1,), prefix, dtype=torch.int32, device=dev)
    sid = torch.arange(1, dtype=torch.int64, device=dev)
    csb, msb, _, _, _, _ = get_cu_seqblocks(cs, chunk, BLOCK_Q, BLOCK_K, [chunk])
    rows = int(csb[-1].item())
    ti = torch.randint(0, ctx // BLOCK_K, (NUM_KV, rows, TOPK), dtype=torch.int32,
                       device=dev, generator=g)
    lim = ((torch.arange(rows, device=dev) + prefix) // BLOCK_K).clamp_min(0)
    ti = torch.minimum(ti, lim.view(1, -1, 1).to(torch.int32))
    return dict(q=q, k_cache=k, v_cache=v, sink=None, req_to_token=r2t, slot_ids=sid,
                topk_idx=ti, block_size_q=BLOCK_Q, block_size_k=BLOCK_K, cu_seqlens=cs,
                seq_lens=sl, prefix_lens=pf, max_seqlen_q=chunk,
                cu_seqblocks_q=csb, max_seqblock_q=msb)


def make_plots(rows, out_dir: Path) -> None:
    """Speedup vs page_size for each variant, against the measured `orig` baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = sorted(rows, key=lambda r: r["page_size"])
    pages = [r["page_size"] for r in rs]
    x = range(len(pages))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for key, name, col, style in [
        ("lookup_off_vs_orig", "lookup OFF (control)", "#6B7885", ":"),
        ("paged_speedup", "current guard", "#5145CC", "--"),
        ("multipage_speedup", "M-page lookup", "#0093A3", "-"),
    ]:
        ax.plot(x, [r[key] for r in rs], style, marker="o", color=col, lw=2, ms=6, label=name)
    ax.axhline(1.0, color="#888", ls="--", lw=1)
    ax.set_xticks(list(x), [f"{p}\nM={r['pages_per_block']}" for p, r in zip(pages, rs)],
                  fontsize=8.5)
    ax.set_xlabel("page_size   (M = pages per block)")
    ax.set_ylabel("speedup vs orig (×)")
    ax.set_title("Stage 1 — slot-compute cost vs page_size\n"
                 "control at 1.00× shows the constexpr branch is free when untaken",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "stage1_speedup.png", dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    def _ints(s):
        return [int(x) for x in s.split(",") if x]

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--page-sizes", type=_ints, default=[1, 8, 16, 32, 64, 128, 256])
    p.add_argument("--context-len", type=int, default=32768)
    p.add_argument("--chunk-len", type=int, default=4096)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--wait-for-idle", type=float, default=0.0)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("-o", "--out", type=Path,
                   default=Path(__file__).resolve().parent / "results" / "stage1_slot_compute")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is required.")
        return 1
    info = gpu_info()
    dev = torch.device("cuda")
    print("Stage 1 — slot-compute cost vs page_size")
    print(f"  device : {info['gpu']} (sm{info['sm']})")
    print(f"  shape  : ctx {args.context_len}, chunk {args.chunk_len}, "
          f"block {BLOCK_K}, top-{TOPK}")
    print(f"  control: paged variant with lookup forced OFF (page_size={LOOKUP_OFF})\n")
    if args.wait_for_idle > 0:
        wait_for_idle(args.wait_for_idle)
    warn_if_contended()

    rows = []
    print(f"{'page':>6}{'M':>5}{'orig':>9}{'pagedOFF':>10}{'paged':>9}{'multi':>9}"
          f"{'  ':>2}{'OFF/orig':>9}{'paged×':>8}{'multi×':>8}{'exact':>7}")
    for ps in args.page_sizes:
        kw = build(args.context_len, args.chunk_len, ps, dev)
        M = max(1, BLOCK_K // min(ps, BLOCK_K))
        t_orig = bench_cuda(lambda: ORIG(**kw), warmup=8, iters=args.iters).median_ms
        t_off = bench_cuda(lambda: PAGED(**kw, page_size=LOOKUP_OFF),
                           warmup=8, iters=args.iters).median_ms
        t_paged = bench_cuda(lambda: PAGED(**kw, page_size=ps),
                             warmup=8, iters=args.iters).median_ms
        t_multi = bench_cuda(lambda: MULTI(**kw, page_size=ps),
                             warmup=8, iters=args.iters).median_ms
        ref = ORIG(**kw)
        exact = (torch.equal(ref, PAGED(**kw, page_size=ps))
                 and torch.equal(ref, MULTI(**kw, page_size=ps))
                 and torch.equal(ref, PAGED(**kw, page_size=LOOKUP_OFF)))
        row = {
            "page_size": ps, "pages_per_block": M, "context_len": args.context_len,
            "chunk_len": args.chunk_len, "block_size": BLOCK_K,
            "orig_ms": round(t_orig, 6), "paged_lookup_off_ms": round(t_off, 6),
            "paged_ms": round(t_paged, 6), "multipage_ms": round(t_multi, 6),
            "lookup_off_vs_orig": round(t_off / t_orig, 4),
            "paged_speedup": round(t_orig / t_paged, 4),
            "multipage_speedup": round(t_orig / t_multi, 4),
            "paged_fast_path": ps >= BLOCK_K and ps % BLOCK_K == 0,
            "all_bit_exact": bool(exact), "status": "ok",
        }
        rows.append(row)
        print(f"{ps:>6}{M:>5}{t_orig:>9.4f}{t_off:>10.4f}{t_paged:>9.4f}{t_multi:>9.4f}"
              f"{'  ':>2}{t_off/t_orig:>8.3f}x{t_orig/t_paged:>7.2f}x"
              f"{t_orig/t_multi:>7.2f}x{str(exact):>7}")
        del kw, ref
        torch.cuda.empty_cache()

    args.out.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = write_results(rows, args.out, "raw")
    summary = args.out / "summary.csv"
    csv_path.replace(summary)
    print(f"\nwrote {json_path}\n      {summary}")
    if not args.no_plots:
        emit_plots(make_plots, rows, args.out)
        print(f"      {args.out}/stage1_speedup.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
