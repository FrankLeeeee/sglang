#!/usr/bin/env python3
"""Block-sparse vs token-sparse (seg) indexer: per-operation runtime comparison.

For each context length this measures, for both score kernels:

  * stage totals (score kernel / top-k select / glue) from torch.profiler
    kernel self-time,
  * the absolute wall time of every intermediate operation inside the score
    kernel (q_load, page_table, k_load, qk_dot, mask_bias, pool, score_store)
    via Triton Proton instrumentation, scaled onto the uninstrumented kernel
    time (see inner_profile.py for the methodology and caveats).

Each context runs in its OWN subprocess: the block kernel's autotune key has
no context bucket, so measuring several contexts in one process would reuse
whichever config the first context tuned (measured ~1.5-1.7x pessimisation).

    CUDA_VISIBLE_DEVICES=4 python bench_indexer_compare.py
    python bench_indexer_compare.py --context-lens 16384,131072

Outputs (default --out results/indexer_comparison): raw.json, summary.csv,
score_kernel_breakdown.png, score_kernel_totals.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_CTXS = [4096, 8192, 16384, 65536, 131072, 524288, 1048576]
OPS = ("q_load", "page_table", "k_load", "qk_dot", "mask_bias", "pool",
       "score_store")
OP_COLORS = {
    "q_load": "#2a78d6", "page_table": "#eb6834", "k_load": "#1baf7a",
    "qk_dot": "#eda100", "mask_bias": "#e87ba4", "pool": "#008300",
    "score_store": "#4a3aa7",
}
IMPL_COLORS = {"block": "#898781", "seg": "#2a78d6"}


def _ctx_label(n: int) -> str:
    return f"{n // 1024}k" if n < 1 << 20 else f"{n >> 20}M"


# ---------------------------------------------------------------------------
# single-context measurement (subprocess entry)
# ---------------------------------------------------------------------------


def run_single(ctx: int, json_path: Path) -> None:
    import torch

    from harness import (
        bench_cuda,
        build_prefill_inputs,
        profile_breakdown,
        warn_if_contended,
    )
    from m3_config import m3_config
    from bench_indexer import BUILDERS
    from inner_profile import profile_score_inner
    from sglang.kernels.ops.attention.minimax_sparse.token.index_score_seg import (
        token_select_prefill_seg,
    )

    torch.cuda.init()
    warn_if_contended()
    chunk = min(8192, ctx)
    inner_iters = 2 if ctx >= 524288 else 3
    out_dir = json_path.parent / "inner"
    rows = json.loads(json_path.read_text()) if json_path.exists() else []
    dev = torch.device("cuda")

    def measure(name: str, gran: str, cfg, inp, run) -> None:
        bench_cuda(run, warmup=3, iters=8)  # settle autotune first
        stages, _ = profile_breakdown(run, iters=6, warmup=3)
        score_ms = stages.get("indexer_score", 0.0)
        select_ms = stages.get("topk_select", 0.0)
        row = {
            "impl": name, "context_len": ctx, "chunk_len": chunk,
            "score_ms": round(score_ms, 4), "select_ms": round(select_ms, 4),
            "glue_ms": round(sum(stages.values()) - score_ms - select_ms, 4),
        }
        iters = inner_iters
        while iters >= 1:
            try:
                inner = profile_score_inner(
                    cfg, inp, granularity=gran, phase="prefill", iters=iters,
                    out_dir=out_dir, tag=f"cmp_{name}_{ctx}",
                )
                total = sum(inner.values())
                for op in OPS:
                    row[f"op_{op}_ms"] = round(
                        score_ms * inner.get(op, 0.0) / total, 4)
                break
            except Exception as err:  # Proton clock overflow at long context
                print(f"  [inner profile retry ({iters} -> {iters - 1}): {err}]")
                iters -= 1
        rows.append(row)
        print(f"  {name:<6} ctx={ctx:<8} score={score_ms:9.4f} ms  "
              f"select={select_ms:9.4f} ms")

    cfg_b = m3_config(granularity="block")
    inp_b = build_prefill_inputs(
        cfg_b, batch_size=1, context_len=ctx, chunk_len=chunk, device=dev)
    measure("block", "block", cfg_b, inp_b,
            BUILDERS[("block", "prefill")](cfg_b, inp_b))
    del inp_b
    torch.cuda.empty_cache()

    cfg_t = m3_config(granularity="token")
    inp_t = build_prefill_inputs(
        cfg_t, batch_size=1, context_len=ctx, chunk_len=chunk, device=dev)
    kw = dict(
        idx_q=inp_t.idx_q, idx_k_cache=inp_t.idx_k_cache,
        req_to_token=inp_t.req_to_token, slot_ids=inp_t.slot_ids,
        cu_seqlens=inp_t.cu_seqlens, seq_lens=inp_t.seq_lens,
        prefix_lens=inp_t.prefix_lens, max_seqlen_q=inp_t.max_seqlen_q,
        max_seqlen_k=inp_t.max_seqlen_k, topk=cfg_t.effective_topk_tokens,
        init_tokens=cfg_t.init_tokens, local_tokens=cfg_t.local_tokens,
        seqlens_cpu=inp_t.seqlens_cpu, prefix_lens_cpu=inp_t.prefix_lens_cpu,
    )
    measure("seg", "token_seg", cfg_t, inp_t,
            lambda: token_select_prefill_seg(**kw))

    json_path.write_text(json.dumps(rows, indent=1))


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def make_plots(rows: list[dict], out: Path, ctxs: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # per-op stacked bars, one panel per context
    ncol = 4
    nrow = -(-len(ctxs) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.1 * nrow),
                             dpi=160)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax in axes[len(ctxs):]:
        ax.axis("off")
    for ax, ctx in zip(axes, ctxs):
        for yi, impl in enumerate(("block", "seg")):
            r = next((x for x in rows if x["impl"] == impl
                      and x["context_len"] == ctx), None)
            if r is None or f"op_{OPS[0]}_ms" not in r:
                continue
            left = 0.0
            for op in OPS:
                v = r.get(f"op_{op}_ms", 0.0)
                if v <= 0:
                    continue
                ax.barh(yi, v, left=left, height=0.55, color=OP_COLORS[op])
                left += v
            ax.text(left * 1.02 + 0.005, yi, f"{r['score_ms']:.2f}",
                    va="center", fontsize=8.5, color="#52514e")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["block", "seg"], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"@ {_ctx_label(ctx)}", loc="left", fontsize=10)
        ax.margins(x=0.15)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.legend(handles=[Patch(color=OP_COLORS[o], label=o) for o in OPS],
               ncol=7, frameon=False, fontsize=9, loc="lower center")
    fig.suptitle("Indexer score kernel — absolute per-op wall time "
                 "(block-sparse vs token-sparse seg)", x=0.02, ha="left",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out / "score_kernel_breakdown.png", bbox_inches="tight")
    plt.close(fig)

    # totals vs context, log-log
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    for impl in ("block", "seg"):
        pts = sorted((r["context_len"], r["score_ms"]) for r in rows
                     if r["impl"] == impl)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                markersize=5, linewidth=1.8, color=IMPL_COLORS[impl],
                linestyle="--" if impl == "block" else "-",
                label=impl + (" (ref)" if impl == "block" else ""))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ctxs)
    ax.set_xticklabels([_ctx_label(c) for c in ctxs])
    ax.minorticks_off()
    ax.set_xlabel("context length")
    ax.set_ylabel("score kernel, median ms (log)")
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Score kernel total vs context (per-shape autotuned)",
                 loc="left", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "score_kernel_totals.png")
    plt.close(fig)


def write_csv(rows: list[dict], out: Path) -> None:
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with (out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context-lens", default=",".join(map(str, DEFAULT_CTXS)))
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent
                   / "results" / "indexer_comparison")
    p.add_argument("--wait-for-idle", type=float, default=600.0)
    p.add_argument("--single-ctx", type=int, default=None,
                   help="internal: measure one context and append to --json")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    if args.single_ctx is not None:
        run_single(args.single_ctx, args.json)
        return 0

    ctxs = [int(x) for x in args.context_lens.split(",") if x.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "raw.json"
    if json_path.exists():
        json_path.unlink()

    from harness import wait_for_idle  # noqa: PLC0415  (torch import cost)

    wait_for_idle(args.wait_for_idle)
    for ctx in ctxs:
        print(f"--- ctx={ctx} ---")
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--single-ctx", str(ctx), "--json", str(json_path)],
        )
        if r.returncode != 0:
            print(f"  ctx={ctx} FAILED (rc={r.returncode}); continuing")

    rows = json.loads(json_path.read_text())
    write_csv(rows, args.out)
    make_plots(rows, args.out, ctxs)
    print(f"\nwrote {args.out}/raw.json, summary.csv, "
          f"score_kernel_breakdown.png, score_kernel_totals.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
