#!/usr/bin/env python3
"""Plot the unified indexer+top-k sweep (bench_indexer_topk.py results).

    python plot_indexer_topk.py \
        --results results/indexer_topk_stages.json --out results/plots

Produces:
    prefill_latency_vs_context.png       total median latency, log-log
    prefill_select_stage_vs_context.png  topk_select stage only, log-log —
                                         the flat-vs-linear scaling story
    prefill_stage_breakdown.png          score/select/glue stacks at 128k & 1M
    decode_latency_vs_context.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Color follows the implementation on every chart; block is the reference
# (different output contract) and is drawn dashed gray.
IMPL_COLORS = {
    "current": "#2a78d6",
    "fused_cuda": "#eb6834",
    "seg": "#1baf7a",
    "onepass": "#eda100",
    "tau_emit": "#e87ba4",
    "block": "#898781",
    "token": "#2a78d6",
}
PREFILL_ORDER = ["block", "current", "fused_cuda", "seg", "onepass", "tau_emit"]
DECODE_ORDER = ["block", "token"]
STAGE_COLORS = {"score": "#2a78d6", "select": "#eb6834", "glue": "#b6b4ac"}


def _ctx_label(n: int) -> str:
    return f"{n // 1024}k" if n < 1 << 20 else f"{n >> 20}M"


def _load(path: Path) -> list[dict]:
    rows = [r for r in json.loads(path.read_text()) if r.get("status") == "ok"]
    if not rows:
        raise SystemExit(f"no ok rows in {path}")
    return rows


def _series(rows, phase, impl, field):
    pts = sorted(
        (r["context_len"], r[field])
        for r in rows
        if r["phase"] == phase and r["impl"] == impl and field in r
    )
    return [p[0] for p in pts], [p[1] for p in pts]


def _style_axes(ax, ctxs, ylabel):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ctxs)
    ax.set_xticklabels([_ctx_label(c) for c in ctxs])
    ax.minorticks_off()
    ax.set_xlabel("context length")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", axis="y", color="#e1e0d9", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _line_plot(rows, phase, order, field, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ctxs = sorted({r["context_len"] for r in rows if r["phase"] == phase})
    for impl in order:
        xs, ys = _series(rows, phase, impl, field)
        if not xs:
            continue
        dashed = impl == "block"
        ax.plot(
            xs, ys,
            marker="o", markersize=5, linewidth=1.8,
            linestyle="--" if dashed else "-",
            color=IMPL_COLORS[impl],
            label=impl + (" (ref)" if dashed else ""),
        )
    _style_axes(ax, ctxs, "median ms (log)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _stack_plot(rows, ctx_list, out_path):
    fig, axes = plt.subplots(
        1, len(ctx_list), figsize=(6 * len(ctx_list), 4.2), dpi=160
    )
    if len(ctx_list) == 1:
        axes = [axes]
    for ax, ctx in zip(axes, ctx_list):
        impls, score, select, glue = [], [], [], []
        for impl in PREFILL_ORDER:
            r = next(
                (x for x in rows
                 if x["phase"] == "prefill" and x["impl"] == impl
                 and x["context_len"] == ctx and "stage_sum_ms" in x),
                None,
            )
            if r is None:
                continue
            impls.append(impl)
            score.append(r["stage_indexer_score_ms"])
            select.append(r["stage_topk_select_ms"])
            glue.append(r["stage_sum_ms"] - r["stage_indexer_score_ms"]
                        - r["stage_topk_select_ms"])
        y = range(len(impls))
        left = [0.0] * len(impls)
        for vals, stage in ((score, "score"), (select, "select"), (glue, "glue")):
            ax.barh(y, vals, left=left, height=0.62,
                    color=STAGE_COLORS[stage], label=stage)
            left = [a + b for a, b in zip(left, vals)]
        for i, total in enumerate(left):
            ax.text(total + max(left) * 0.01, i, f"{total:.0f}",
                    va="center", fontsize=8.5, color="#52514e")
        ax.set_yticks(list(y))
        ax.set_yticklabels(impls, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("ms (kernel self-time)")
        ax.set_title(f"prefill stage breakdown @ {_ctx_label(ctx)}",
                     loc="left", fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path,
                   default=Path(__file__).resolve().parent
                   / "results" / "indexer_topk_stages.json")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "results" / "plots")
    p.add_argument("--stack-ctx", default="131072,1048576",
                   help="comma-separated contexts for the stage-stack panels")
    args = p.parse_args(argv)

    rows = _load(args.results)
    args.out.mkdir(parents=True, exist_ok=True)

    _line_plot(rows, "prefill", PREFILL_ORDER, "latency_median_ms",
               "Prefill selection latency (bs=1, 8k extend chunk, H200)",
               args.out / "prefill_latency_vs_context.png")
    if any("stage_topk_select_ms" in r for r in rows):
        _line_plot(rows, "prefill", PREFILL_ORDER, "stage_topk_select_ms",
                   "Prefill topk_select stage — flat vs linear scaling",
                   args.out / "prefill_select_stage_vs_context.png")
        stack_ctxs = [int(x) for x in args.stack_ctx.split(",") if x.strip()]
        _stack_plot(rows, stack_ctxs, args.out / "prefill_stage_breakdown.png")
    _line_plot(rows, "decode", DECODE_ORDER, "latency_median_ms",
               "Decode selection latency (bs=1, H200)",
               args.out / "decode_latency_vs_context.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
