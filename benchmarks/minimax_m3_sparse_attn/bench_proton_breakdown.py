#!/usr/bin/env python3
"""Intra-kernel breakdown of the block-sparse prefill kernel via Triton Proton.

Instruments the inner loop of ``_gqa_share_sparse_fwd_kernel`` with proton
scopes and reports per-scope cycle counts, for both slot-resolution paths:

  orig   per-token ``req_to_token`` gather + runtime int64 ``% max_slots`` on
         the whole BLOCK_SIZE_K-wide vector  (page_size != block_size)
  paged  one scalar lookup + one scalar modulo                (page_size == block_size)

Both come from the SAME instrumented kernel
(``minimax_sparse/prefill/topk_sparse_proton.py``), selected by ``page_size``,
so the scopes are identical and directly comparable.

READ THE CAVEATS in the printed footer and in
``results/block_sparse_kernel_comparison/ANALYSIS.md`` before drawing
conclusions from the per-scope numbers -- instrumentation perturbs the
schedule substantially and asymmetrically here.

    CUDA_VISIBLE_DEVICES=4 python bench_proton_breakdown.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import triton.profiler as proton
import triton.profiler.language as pl_lang

# Intra-kernel scopes default to Gluon-only in Triton 3.6; opt the standard
# Triton semantic in so plain @triton.jit kernels can carry proton records.
pl_lang.enable_semantic("triton")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import bench_cuda, build_page_table, gpu_info, write_results  # noqa: E402

from sglang.kernels.ops.attention.minimax_sparse.common.utils import (  # noqa: E402
    get_cu_seqblocks,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse_paged import (  # noqa: E402
    flash_prefill_with_gqa_share_sparse_paged as PLAIN,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse_proton import (  # noqa: E402
    flash_prefill_with_gqa_share_sparse_proton as INSTR,
)

NUM_Q, NUM_KV, D = 64, 4, 128
BLOCK_K, BLOCK_Q, TOPK, PAGE = 128, 1, 16, 128
# The default CIRCULAR *shared-memory* buffer overflows with 7 scopes x 16 loop
# iterations x many warps and silently reports inverted totals. A large global
# buffer is required for these counts to be self-consistent.
BUFFER_SIZE = 32768
BUFFER_TYPE = "global"


def build_inputs(ctx: int, chunk: int, dev, gen):
    torch.manual_seed(0)
    prefix = ctx - chunk
    r2t, max_slots = build_page_table(
        batch_size=1, context_len=ctx, page_size=PAGE, device=dev, generator=gen
    )
    k = torch.randn(max_slots, NUM_KV, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(max_slots, NUM_KV, D, dtype=torch.bfloat16, device=dev)
    q = torch.randn(chunk, NUM_Q, D, dtype=torch.bfloat16, device=dev)
    cu = torch.tensor([0, chunk], dtype=torch.int32, device=dev)
    sl = torch.full((1,), ctx, dtype=torch.int32, device=dev)
    pf = torch.full((1,), prefix, dtype=torch.int32, device=dev)
    sid = torch.arange(1, dtype=torch.int64, device=dev)
    csb, msb, _, _, _, _ = get_cu_seqblocks(cu, chunk, BLOCK_Q, BLOCK_K, [chunk])
    rows = int(csb[-1].item())
    ti = torch.randint(0, ctx // BLOCK_K, (NUM_KV, rows, TOPK),
                       dtype=torch.int32, device=dev, generator=gen)
    lim = ((torch.arange(rows, device=dev) + prefix) // BLOCK_K).clamp_min(0)
    ti = torch.minimum(ti, lim.view(1, -1, 1).to(torch.int32))
    return dict(
        q=q, k_cache=k, v_cache=v, sink=None, req_to_token=r2t, slot_ids=sid,
        topk_idx=ti, block_size_q=BLOCK_Q, block_size_k=BLOCK_K, cu_seqlens=cu,
        seq_lens=sl, prefix_lens=pf, max_seqlen_q=chunk, cu_seqblocks_q=csb,
        max_seqblock_q=msb,
    )


def collect_scopes(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text())
    root = data[0] if isinstance(data, list) else data
    scopes: dict[str, int] = {}

    def walk(node):
        metrics = node.get("metrics", {})
        if "cycles" in metrics:
            scopes[node["frame"]["name"]] = int(metrics["cycles"])
        for child in node.get("children", []):
            walk(child)

    walk(root)
    return scopes


def profile_variant(kw, page_size: int, out_stem: Path) -> dict[str, int]:
    INSTR(**kw, page_size=page_size)  # compile outside the profiled region
    torch.cuda.synchronize()
    proton.start(
        str(out_stem), backend="instrumentation",
        mode=proton.mode.Default(buffer_size=BUFFER_SIZE, buffer_type=BUFFER_TYPE),
    )
    INSTR(**kw, page_size=page_size)
    torch.cuda.synchronize()
    proton.finalize()
    return collect_scopes(out_stem.with_suffix(".hatchet"))


def make_plot(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    scopes = sorted({r["scope"] for r in rows})
    variants = ["orig", "paged"]
    colors = {"orig": "#5145CC", "paged": "#0093A3"}
    import statistics
    vals = {v: [statistics.mean([r["share_pct"] for r in rows
                                 if r["scope"] == s and r["variant"] == v])
                for s in scopes] for v in variants}
    y = np.arange(len(scopes))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, v in enumerate(variants):
        ax.barh(y + (i - 0.5) * h, vals[v], height=h, color=colors[v], label=v)
    for i, s in enumerate(scopes):
        ax.text(max(vals["orig"][i], vals["paged"][i]) * 1.03, i,
                f'{vals["paged"][i] - vals["orig"][i]:+.1f}pp',
                va="center", fontsize=9, color="#444")
    ax.set_yticks(y)
    ax.set_yticklabels([s.split("_", 1)[1] for s in scopes])
    ax.set_xlabel("share of in-kernel cycles (%, mean over reps)")
    ax.set_title("Proton intra-kernel breakdown — block-sparse prefill (ctx 32k)\n"
                 "shares are stable; absolute totals are NOT -- see ANALYSIS.md", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "proton_breakdown.png", dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--context-len", type=int, default=32768)
    p.add_argument("--chunk-len", type=int, default=4096)
    p.add_argument("--reps", type=int, default=3,
                   help="repetitions; shares are averaged, totals are not "
                        "reproducible across runs")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("-o", "--out", type=Path,
                   default=Path(__file__).resolve().parent / "results"
                   / "block_sparse_kernel_comparison")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is required.")
        return 1
    info = gpu_info()
    print(f"proton intra-kernel breakdown — {info['gpu']} (sm{info['sm']})")
    print(f"  ctx={args.context_len} chunk={args.chunk_len} "
          f"block={BLOCK_K} topk={TOPK} page={PAGE}")
    print(f"  buffer: {BUFFER_TYPE} size={BUFFER_SIZE}\n")

    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    kw = build_inputs(args.context_len, args.chunk_len, dev, gen)

    # Wall-clock control: plain (production) and instrumented, both paths.
    control = {}
    for label, fn in [("plain", PLAIN), ("instrumented", INSTR)]:
        a = bench_cuda(lambda: fn(**kw, page_size=1), warmup=8, iters=30).median_ms
        b = bench_cuda(lambda: fn(**kw, page_size=PAGE), warmup=8, iters=30).median_ms
        control[label] = {"orig_ms": round(a, 5), "paged_ms": round(b, 5),
                          "speedup": round(a / b, 3)}
        print(f"  wall clock {label:<13} orig={a:.4f}ms paged={b:.4f}ms  {a / b:.2f}x")
    print()

    # Absolute cycle totals are NOT reproducible across runs here -- they track
    # machine state and have been observed to invert the orig/paged ordering
    # outright. The per-variant *share* distribution is stable, so repeat and
    # report mean +/- population stdev of the shares.
    import statistics

    rows: list[dict] = []
    per_rep: dict[str, list[dict[str, int]]] = {"orig": [], "paged": []}
    for rep in range(args.reps):
        for variant, page_size in (("orig", 1), ("paged", PAGE)):
            stem = args.out / f"proton_{variant}"
            scopes = profile_variant(kw, page_size, stem)
            per_rep[variant].append(scopes)
            total = sum(scopes.values())
            for scope, cycles in scopes.items():
                rows.append({
                    "variant": variant, "rep": rep, "scope": scope,
                    "cycles": cycles,
                    "share_pct": round(cycles / total * 100, 2),
                    "total_cycles": total,
                    "context_len": args.context_len, "chunk_len": args.chunk_len,
                })

    def shares(variant, scope):
        return [s[scope] / sum(s.values()) * 100 for s in per_rep[variant]]

    all_scopes = sorted(per_rep["orig"][0])
    print(f"{'scope':<22}{'orig share':>18}{'paged share':>18}")
    for scope in all_scopes:
        o, p = shares("orig", scope), shares("paged", scope)
        print(f"{scope:<22}{statistics.mean(o):>13.1f}% "
              f"±{statistics.pstdev(o):<4.1f}{statistics.mean(p):>12.1f}% "
              f"±{statistics.pstdev(p):<4.1f}")
    for variant in ("orig", "paged"):
        tots = [sum(s.values()) / 1e6 for s in per_rep[variant]]
        print(f"  {variant} total Mcycles across reps: "
              f"{', '.join(f'{t:.0f}' for t in tots)}  <- unstable, do not compare")

    json_path, csv_path = write_results(rows, args.out, "proton_breakdown")
    (args.out / "proton_control.json").write_text(json.dumps(control, indent=2))
    print(f"\nwrote {json_path}\n      {csv_path}\n"
          f"      {args.out / 'proton_control.json'}")
    if not args.no_plots:
        try:
            make_plot(rows, args.out)
            print(f"      {args.out / 'proton_breakdown.png'}")
        except Exception as err:
            print(f"  (plotting failed: {err})")

    print("\nCAVEAT: instrumentation slows the two paths asymmetrically "
          f"({control['plain']['speedup']:.2f}x plain vs "
          f"{control['instrumented']['speedup']:.2f}x instrumented), and scope "
          "boundaries act as scheduling barriers, so per-scope deltas do NOT "
          "carry the uninstrumented magnitudes. Use the shape, not the numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
