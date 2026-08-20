#!/usr/bin/env python3
"""Plot the indexer+selector comparison from existing results (no GPU run).

    python plot_indexer_selector.py \
        --results results/indexer_selector_comparison/indexer_selector.json

Produces, next to the results file:
    latency_vs_context.png    total selection latency, log-log, all impls
    stage_breakdown.png       score/select/other stacks at --stack-ctx
    workspace_vs_context.png  peak transient workspace, log-log
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMPL_ORDER = ["current", "fused", "fused_cuda", "seg", "onepass", "tau_emit"]
IMPL_COLORS = {
    "current": "#2a78d6",
    "fused": "#4a3aa7",
    "fused_cuda": "#eb6834",
    "seg": "#1baf7a",
    "onepass": "#eda100",
    "tau_emit": "#e87b a4".replace(" ", ""),
}
STAGE_COLORS = {"score": "#2a78d6", "select": "#eb6834", "other": "#b6b4ac"}


def _ctx_label(n: int) -> str:
    return f"{n // 1024}k" if n < 1 << 20 else f"{n >> 20}M"


def _style(ax, ctxs, ylabel):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ctxs)
    ax.set_xticklabels([_ctx_label(c) for c in ctxs])
    ax.minorticks_off()
    ax.set_xlabel("context length")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def line_plot(rows, ctxs, field, ylabel, title, out_path, scale=1.0):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for impl in IMPL_ORDER:
        pts = sorted((r["context_len"], r[field] * scale) for r in rows
                     if r["impl"] == impl and r.get(field) is not None)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                    markersize=5, linewidth=1.8, color=IMPL_COLORS[impl],
                    label=impl)
    _style(ax, ctxs, ylabel)
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def stack_plot(rows, stack_ctxs, out_path):
    fig, axes = plt.subplots(1, len(stack_ctxs),
                             figsize=(6 * len(stack_ctxs), 4.4), dpi=160)
    if len(stack_ctxs) == 1:
        axes = [axes]
    for ax, ctx in zip(axes, stack_ctxs):
        impls, parts = [], []
        for impl in IMPL_ORDER:
            r = next((x for x in rows if x["impl"] == impl
                      and x["context_len"] == ctx), None)
            if r is None:
                continue
            impls.append(impl)
            score = r["stage_indexer_score_ms"]
            select = r["stage_topk_select_ms"]
            parts.append((score, select, r["stage_sum_ms"] - score - select))
        y = range(len(impls))
        left = [0.0] * len(impls)
        for si, stage in enumerate(("score", "select", "other")):
            vals = [p[si] for p in parts]
            ax.barh(y, vals, left=left, height=0.6,
                    color=STAGE_COLORS[stage], label=stage)
            left = [a + b for a, b in zip(left, vals)]
        for i, total in enumerate(left):
            ax.text(total + max(left) * 0.01, i, f"{total:.1f}", va="center",
                    fontsize=8.5, color="#52514e")
        ax.set_yticks(list(y))
        ax.set_yticklabels(impls, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("ms (kernel self-time)")
        ax.set_title(f"stage breakdown @ {_ctx_label(ctx)}", loc="left",
                     fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path,
                   default=Path(__file__).resolve().parent / "results"
                   / "indexer_selector_comparison" / "indexer_selector.json")
    p.add_argument("--stack-ctx", default="131072,1048576")
    args = p.parse_args(argv)

    rows = [r for r in json.loads(args.results.read_text())
            if r.get("status") == "ok"]
    out = args.results.parent
    ctxs = sorted({r["context_len"] for r in rows})
    stack_ctxs = [int(x) for x in args.stack_ctx.split(",") if x.strip()]

    line_plot(rows, ctxs, "latency_median_ms", "median ms (log)",
              "Indexer + selector latency — all token-sparse implementations "
              "(bs=1, 8k extend chunk, H200)",
              out / "latency_vs_context.png")
    stack_plot(rows, stack_ctxs, out / "stage_breakdown.png")
    line_plot(rows, ctxs, "transient_bytes", "peak workspace GiB (log)",
              "Peak transient workspace",
              out / "workspace_vs_context.png", scale=1 / 2**30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
