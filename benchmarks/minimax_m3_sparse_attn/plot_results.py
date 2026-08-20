#!/usr/bin/env python3
"""Turn the MiniMax-M3 sparse attention benchmark JSON into trend plots.

    python benchmarks/minimax_m3_sparse_attn/plot_results.py \
        --results benchmarks/minimax_m3_sparse_attn/results \
        --out benchmarks/minimax_m3_sparse_attn/results/plots

Produces, when the matching rows exist:
    <phase>_latency_vs_context.png
    <phase>_breakdown_absolute.png
    <phase>_breakdown_share.png
    kv_memory_vs_context.png      phase-independent whole-model KV-cache footprint
    <phase>_workspace_vs_context.png
    <phase>_sweep_<dimension>.png
    <phase>_granularity_ratio.png

Here <phase> is ``prefill`` or ``decode``. A phase-specific file is emitted
only when matching result rows exist.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGE_COLORS = {
    "projection_gemm": "#4C6EF5",
    "qk_norm_rope": "#7048E8",
    "kv_store": "#0CA678",
    "buffer_init": "#96F2D7",
    "indexer_score": "#F59F00",
    "topk_select": "#E8590C",
    "select_overhead": "#FFC078",
    "topk_union": "#C2255C",
    "sparse_attn": "#1971C2",
    "dense_attn": "#37B24D",
    "merge": "#868E96",
    "other": "#CED4DA",
}
STAGE_LABELS = {
    "projection_gemm": "QKV / o_proj GEMM",
    "qk_norm_rope": "QK-norm + RoPE",
    "kv_store": "KV + index cache store",
    "buffer_init": "workspace init (-inf fill)",
    "indexer_score": "indexer (block scores)",
    "topk_select": "top-k select",
    "select_overhead": "index bookkeeping",
    "topk_union": "top-k union (torch)",
    "sparse_attn": "sparse attention",
    "dense_attn": "dense attention",
    "merge": "split-k merge",
    "other": "other",
}
MiB = 1024**2
GiB = 1024**3


# Which latency column the plots use.
#   gpu    stage_sum_ms   pure GPU kernel time, excludes launch overhead. The
#                         only estimator that survives a shared GPU, since a
#                         co-tenant inflates launch latency without bound.
#   min    latency_min_ms fastest wall-clock iteration; good on a mostly-idle box.
#   median                only meaningful on a dedicated machine.
METRIC_KEYS = {
    "gpu": "stage_sum_ms",
    "min": "latency_min_ms",
    "median": "latency_median_ms",
}
LATENCY_KEY = METRIC_KEYS["min"]


def load_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            for row in data:
                row.setdefault("source", path.stem)
                row.setdefault("granularity", "block")
                rows.append(row)
    return [r for r in rows if r.get("status") == "ok"]


def _stages_present(rows: list[dict]) -> list[str]:
    present = []
    for stage in STAGE_COLORS:
        key = f"stage_{stage}_ms"
        if any(r.get(key, 0) for r in rows):
            present.append(stage)
    return present


def _sorted_by(rows: list[dict], key: str) -> list[dict]:
    return sorted(rows, key=lambda r: r[key])


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------


GRAN_COLOR = {"block": "#1971C2", "token": "#E8590C", "dense": "#37B24D"}
GRAN_LABEL = {
    "block": "block-sparse",
    "token": "token-sparse",
    # The --granularity dense variant: sglang's production Triton attention over
    # the paged KV cache, which is what a server actually runs.
    "dense": "dense (no indexer)",
}


# Presentation order everywhere: least sparse first, so each panel adds one
# more layer of selection machinery than the one to its left.
GRAN_ORDER = ("dense", "token", "block")
# Phases are emitted in this order.
PHASE_ORDER = ("prefill", "decode")

TITLE_SIZE = 17
PANEL_TITLE_SIZE = 13


def _budget_tag(rows: list[dict], gran: str) -> str:
    """The selection budget for one panel: `top-k 16 × 128-token blocks`.

    Only names what is actually constant across the panel — the `topk_blocks`
    and `block_size` sweeps vary those very fields, and there the shared
    2048-token budget is the honest label.
    """
    if gran == "dense":  # attends to the whole causal context, no budget
        return ""

    def const(key: str):
        vals = {r.get(key) for r in rows if r.get(key) is not None}
        return vals.pop() if len(vals) == 1 else None

    topk, blk = const("topk_blocks"), const("block_size")
    # Layer rows carry no explicit token budget; there blocks × block size is
    # it. A budget that is *present but varying* (the topk_blocks sweep) must
    # not fall back to that product — the panel has no single budget to name.
    if any(r.get("topk_tokens") is not None for r in rows):
        budget = const("topk_tokens")
    else:
        budget = topk * blk if topk and blk else None
    if gran == "token":
        return f"top-k {budget} tokens" if budget else ""
    if topk and blk:
        return f"top-k {topk} × {blk}-token blocks"
    if topk:
        return f"top-k {topk} blocks"
    if blk:
        return f"{blk}-token blocks"
    return f"{budget}-token budget" if budget else ""


def _panel_title(rows: list[dict], gran: str, suffix: str = "") -> str:
    """`block-sparse: breakdown` over its budget on a second line.

    ``suffix`` continues the granularity line; the budget always lands below it.
    """
    tag = _budget_tag(rows, gran)
    title = GRAN_LABEL.get(gran, gran) + suffix
    return f"{title}\n{tag}" if tag else title


def _granularities(rows: list[dict]) -> list[str]:
    present = {r.get("granularity", "block") for r in rows}
    ordered = [g for g in GRAN_ORDER if g in present]
    return ordered + sorted(present - set(GRAN_ORDER))


def _phases(rows: list[dict]) -> list[str]:
    present = {r.get("phase") for r in rows if r.get("phase")}
    return [p for p in PHASE_ORDER if p in present] + sorted(
        present - set(PHASE_ORDER)
    )


def _shape_tag(rows: list[dict]) -> str:
    """`8 Q : 1 KV heads` when the shard is constant, else `varying`."""
    shapes = {
        (r["num_q_heads"], r["num_kv_heads"])
        for r in rows
        if r.get("num_q_heads") and r.get("num_kv_heads")
    }
    if len(shapes) == 1:
        q, kv = shapes.pop()
        return f"{q} Q : {kv} KV heads"
    return "varying Q:KV heads"


def _phase_detail(rows: list[dict], phase: str) -> str:
    if phase != "prefill":
        return "1 token/sequence, one layer"
    chunks = {
        r["num_query_tokens"] // max(1, r.get("batch_size", 1))
        for r in rows
        if r.get("num_query_tokens")
    }
    if len(chunks) == 1:
        chunk = chunks.pop()
        if any(chunk < r["context_len"] for r in rows):
            return f"{chunk}-token extend chunk, one layer"
    return "whole context, one layer"


def _suptitle(fig, text: str, rows: list[dict]) -> None:
    fig.suptitle(f"{text}  —  {_shape_tag(rows)}", fontsize=TITLE_SIZE)


def _share_y(axes) -> None:
    """Put a list of axes on one y-scale so panels are visually comparable."""
    axes = [a for a in axes if a.has_data()]
    if len(axes) < 2:
        return
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)


def _label_totals(ax, totals: list[float], fmt: str = "{:.3g}") -> None:
    """Print each stacked bar's total above it."""
    if not totals:
        return
    span = max(totals) or 1.0
    for i, t in enumerate(totals):
        ax.annotate(
            fmt.format(t),
            (i, t),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylim(top=ax.get_ylim()[1] + 0.10 * span)


def plot_latency_vs_context(rows: list[dict], out: Path) -> None:
    ctx_rows = [r for r in rows if r.get("sweep") == "context"]
    if not ctx_rows:
        return
    styles = {1: ":", 4: "-.", 8: "--", 32: "-", 64: "-"}
    for phase in _phases(ctx_rows):
        phase_rows = [r for r in ctx_rows if r["phase"] == phase]
        grans = _granularities(phase_rows)
        fig, ax = plt.subplots(figsize=(7, 4.8))
        batches = (
            sorted({r["batch_size"] for r in phase_rows})
            if phase == "decode"
            else [None]
        )
        for gran in grans:
            for bs in batches:
                group = _sorted_by(
                    [
                        r
                        for r in phase_rows
                        if r["granularity"] == gran
                        and (bs is None or r["batch_size"] == bs)
                    ],
                    "context_len",
                )
                if not group:
                    continue
                label = GRAN_LABEL.get(gran, gran)
                if bs is not None:
                    label += f" bs={bs}"
                ax.plot(
                    [r["context_len"] for r in group],
                    [r[LATENCY_KEY] for r in group],
                    marker="o",
                    linestyle=styles.get(bs, "-"),
                    color=GRAN_COLOR.get(gran),
                    label=label,
                )
        detail = _phase_detail(phase_rows, phase)
        ax.set_title(f"{phase.capitalize()} ({detail})", fontsize=PANEL_TITLE_SIZE)
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("latency (ms)")
        ax.set_xscale("log", base=2)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        _suptitle(
            fig,
            f"{phase.capitalize()} latency vs context length",
            phase_rows,
        )
        _save(fig, out, f"{phase}_latency_vs_context.png")


def _stacked(
    ax, rows: list[dict], xkey: str, stages: list[str], normalize: bool
) -> list[float]:
    xs = [r[xkey] for r in rows]
    bottom = [0.0] * len(rows)
    totals = [max(1e-12, sum(r.get(f"stage_{s}_ms", 0.0) for s in stages)) for r in rows]
    idx = list(range(len(rows)))
    for stage in stages:
        vals = [r.get(f"stage_{stage}_ms", 0.0) for r in rows]
        if normalize:
            vals = [v / t for v, t in zip(vals, totals)]
        ax.bar(
            idx,
            vals,
            bottom=bottom,
            color=STAGE_COLORS[stage],
            label=STAGE_LABELS[stage],
            width=0.7,
        )
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(idx)
    ax.set_xticklabels([str(x) for x in xs], rotation=45, ha="right")
    _label_totals(ax, bottom, "{:.2f}" if normalize else "{:.3g}")
    return bottom


def plot_breakdown_vs_context(rows: list[dict], out: Path, normalize: bool = False) -> None:
    ctx_rows = [r for r in rows if r.get("sweep") == "context"]
    if not ctx_rows:
        return
    for phase in _phases(ctx_rows):
        phase_rows = [r for r in ctx_rows if r["phase"] == phase]
        grans = _granularities(phase_rows)
        stages = _stages_present(phase_rows)
        batches = sorted({r["batch_size"] for r in phase_rows})
        bs = batches[-1] if phase == "decode" else None
        fig, axes = plt.subplots(
            1, len(grans), figsize=(4.8 * len(grans), 4.8), squeeze=False
        )
        active_axes = []
        for ax, gran in zip(axes[0], grans):
            sel = _sorted_by(
                [
                    r
                    for r in phase_rows
                    if r["granularity"] == gran
                    and (bs is None or r["batch_size"] == bs)
                ],
                "context_len",
            )
            if not sel:
                continue
            _stacked(ax, sel, "context_len", stages, normalize)
            title = _panel_title(sel, gran)
            if bs is not None:
                title += f"\nbs={bs}"
            ax.set_title(title, fontsize=PANEL_TITLE_SIZE)
            ax.set_xlabel("context length")
            ax.set_ylabel("fraction of runtime" if normalize else "ms")
            ax.grid(alpha=0.25, axis="y")
            active_axes.append(ax)
        _share_y(active_axes)
        if active_axes:
            active_axes[-1].legend(fontsize=7, loc="upper left")
        kind = "share" if normalize else "absolute"
        _suptitle(
            fig,
            f"{phase.capitalize()} runtime breakdown ({kind})",
            phase_rows,
        )
        _save(fig, out, f"{phase}_breakdown_{kind}.png")


def plot_memory(rows: list[dict], out: Path) -> None:
    mem = _sorted_by([r for r in rows if r.get("sweep") == "kv_footprint"], "context_len")
    ctx_rows = [r for r in rows if r.get("sweep") == "context"]
    if not mem and not ctx_rows:
        return

    if mem:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        xs = [r["context_len"] for r in mem]
        ax.plot(
            xs, [r["kv_bytes_model_per_gpu"] / GiB for r in mem], "o-", label="M3 (main + index)"
        )
        ax.plot(
            xs,
            [r["kv_bytes_model_no_index"] / GiB for r in mem],
            "s--",
            color="#868E96",
            label="main KV only",
        )
        ax.set_title("Whole-model KV cache per GPU, one request")
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("GiB")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        _suptitle(fig, "KV-cache memory", mem)
        _save(fig, out, "kv_memory_vs_context.png")

    for phase in _phases(ctx_rows):
        phase_rows = [r for r in ctx_rows if r["phase"] == phase]
        batches = sorted({r["batch_size"] for r in phase_rows})
        bs = batches[-1] if phase == "decode" else None
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for gran in _granularities(phase_rows):
            sel = _sorted_by(
                [
                    r
                    for r in phase_rows
                    if r["granularity"] == gran
                    and (bs is None or r["batch_size"] == bs)
                ],
                "context_len",
            )
            if not sel:
                continue
            label = GRAN_LABEL.get(gran, gran)
            ax.plot(
                [r["context_len"] for r in sel],
                [r["transient_bytes"] / MiB for r in sel],
                "o-",
                color=GRAN_COLOR.get(gran),
                label=f"{label}: measured transient",
            )
            ax.plot(
                [r["context_len"] for r in sel],
                [r["score_buffer_bytes"] / MiB for r in sel],
                ":",
                color=GRAN_COLOR.get(gran),
                alpha=0.75,
                label=f"{label}: analytic score buffer",
            )
        suffix = f", bs={bs}" if bs is not None else ""
        ax.set_title(f"{phase.capitalize()} workspace{suffix}, one layer")
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("MiB")
        ax.set_xscale("log", base=2)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        _suptitle(
            fig,
            f"{phase.capitalize()} workspace memory",
            phase_rows,
        )
        _save(fig, out, f"{phase}_workspace_vs_context.png")


# sweep name -> the row field to plot it against. Usually the same field, but
# the budget sweep is the exception: it walks `topk_blocks` for the block path
# while the token path holds `topk_blocks` at 16 and varies the token budget
# underneath it. Plotting `topk_blocks` there stacks every token point on x=16.
# `topk_tokens` is what both paths actually vary, and it is what the two have
# in common — so it is the axis that makes them comparable.
SWEEP_XKEYS = {
    "num_q_heads": "num_q_heads",
    "num_kv_heads": "num_kv_heads",
    "head_dim": "head_dim",
    "topk_blocks": "topk_tokens",
    "block_size": "block_size",
    "page_size": "page_size",
    "batch_size": "batch_size",
}
XLABELS = {"topk_tokens": "top-k budget (tokens)"}


def plot_sweeps(rows: list[dict], out: Path) -> None:
    for sweep, xkey in SWEEP_XKEYS.items():
        sel = [r for r in rows if r.get("sweep") == sweep]
        if not sel:
            continue
        xlabel = XLABELS.get(xkey, xkey)
        for phase in _phases(sel):
            phase_rows = [r for r in sel if r["phase"] == phase]
            stages = _stages_present(phase_rows)
            grans = _granularities(phase_rows)
            # One row: the latency comparison, then a breakdown per granularity.
            ncols = 1 + len(grans)
            fig, axes = plt.subplots(
                1, ncols, figsize=(5.0 * ncols, 4.8), squeeze=False
            )
            top = axes[0][0]
            for gran in grans:
                group = _sorted_by(
                    [r for r in phase_rows if r["granularity"] == gran],
                    xkey,
                )
                if not group:
                    continue
                top.plot(
                    [r[xkey] for r in group],
                    [r[LATENCY_KEY] for r in group],
                    "o-",
                    color=GRAN_COLOR.get(gran),
                    label=GRAN_LABEL.get(gran, gran),
                )
            top.set_title(f"{phase}: latency vs {xlabel}", fontsize=PANEL_TITLE_SIZE)
            top.set_xlabel(xlabel)
            top.set_ylabel("ms")
            top.grid(alpha=0.3)
            top.legend(fontsize=8)
            for i, gran in enumerate(grans):
                ax = axes[0][1 + i]
                group = _sorted_by(
                    [r for r in phase_rows if r["granularity"] == gran],
                    xkey,
                )
                if not group:
                    continue
                _stacked(ax, group, xkey, stages, normalize=False)
                ax.set_title(
                    _panel_title(group, gran, ": breakdown"),
                    fontsize=PANEL_TITLE_SIZE,
                )
                ax.set_xlabel(xlabel)
                ax.set_ylabel("ms")
                ax.grid(alpha=0.25, axis="y")
            _share_y([axes[0][1 + i] for i in range(len(grans))])
            axes[0][-1].legend(fontsize=7)
            ctx = phase_rows[0].get("context_len")
            _suptitle(
                fig,
                f"{phase.capitalize()} {xlabel} sweep (context={ctx})",
                phase_rows,
            )
            _save(fig, out, f"{phase}_sweep_{sweep}.png")


def plot_granularity_ratio(rows: list[dict], out: Path) -> None:
    """Token-vs-block cost ratio per stage, the headline comparison."""
    ctx = [r for r in rows if r.get("sweep") == "context"]
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in ctx:
        by_key[(r["phase"], r["batch_size"], r["context_len"])][r["granularity"]] = r
    paired = {k: v for k, v in by_key.items() if "block" in v and "token" in v}
    if not paired:
        return

    stages = ["indexer_score", "topk_select", "select_overhead", "sparse_attn"]
    keys = {(p, b) for (p, b, _) in paired}
    for phase in _phases(ctx):
        panels = sorted(
            (key for key in keys if key[0] == phase), key=lambda key: key[1] or 0
        )
        if not panels:
            continue
        fig, axes = plt.subplots(
            1, len(panels), figsize=(5.2 * len(panels), 4.6), squeeze=False
        )
        active_axes = []
        for ax, (_, bs) in zip(axes[0], panels):
            xs = sorted(c for (p, b, c) in paired if p == phase and b == bs)
            if not xs:
                continue
            total = []
            for c in xs:
                pair = paired[(phase, bs, c)]
                tb = sum(pair["block"].get(f"stage_{s}_ms", 0.0) for s in stages)
                tt = sum(pair["token"].get(f"stage_{s}_ms", 0.0) for s in stages)
                total.append(tt / tb if tb else float("nan"))
            for stage in stages:
                ys = []
                for c in xs:
                    pair = paired[(phase, bs, c)]
                    block_ms = pair["block"].get(f"stage_{stage}_ms", 0.0)
                    token_ms = pair["token"].get(f"stage_{stage}_ms", 0.0)
                    ys.append(token_ms / block_ms if block_ms > 0 else float("nan"))
                ax.plot(
                    xs,
                    ys,
                    "o-",
                    color=STAGE_COLORS[stage],
                    label=STAGE_LABELS[stage],
                )
            ax.plot(xs, total, "k--", linewidth=2, label="total attention")
            # parity line: above means token granularity costs more
            ax.axhline(1.0, color="#868E96", linewidth=1)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("context length (tokens)")
            ax.set_ylabel("token time / block time")
            ax.set_title(
                phase if bs is None else f"{phase} bs={bs}",
                fontsize=PANEL_TITLE_SIZE,
            )
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
            active_axes.append(ax)
        _share_y(active_axes)
        if active_axes:
            active_axes[-1].legend(fontsize=7)
        phase_rows = [r for r in ctx if r["phase"] == phase]
        # Both granularities select the same number of tokens; say which.
        budget = _budget_tag(
            [r for r in phase_rows if r["granularity"] == "token"], "token"
        ).replace("top-k ", "").replace(" tokens", "-token")
        _suptitle(
            fig,
            f"{phase.capitalize()} token cost relative to block, "
            f"same {budget or 'matched'} budget",
            phase_rows,
        )
        _save(fig, out, f"{phase}_granularity_ratio.png")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    p.add_argument("--results", type=Path, default=here / "results")
    p.add_argument("--out", type=Path, default=here / "results" / "plots")
    p.add_argument(
        "--metric",
        default="min",
        choices=["gpu", "min", "median"],
        help="latency estimator: gpu = kernel time only (best on a shared box), "
        "min = fastest wall-clock iteration, median = needs a dedicated GPU",
    )
    args = p.parse_args(argv)
    global LATENCY_KEY
    LATENCY_KEY = METRIC_KEYS[args.metric]

    rows = load_rows(args.results)
    if not rows:
        print(f"no result rows found under {args.results}")
        return 1
    print(f"loaded {len(rows)} rows from {args.results}")

    plot_latency_vs_context(rows, args.out)
    plot_breakdown_vs_context(rows, args.out, normalize=False)
    plot_breakdown_vs_context(rows, args.out, normalize=True)
    plot_memory(rows, args.out)
    plot_sweeps(rows, args.out)
    plot_granularity_ratio(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
