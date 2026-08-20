#!/usr/bin/env python3
"""One figure comparing the four MiniMax-M3 indexers.

    block            shipped: score every key, max-pool per 128-key block, top-16
    token            exact: score every key, top-2048 positions
    two-level pre    LongCat/HISA: score L/P mean-pooled keys -> top-M blocks,
                     then score those M*P tokens -> top-2048
    two-level post   the same, with stage 1 scoring every key and pooling after
                     the dot — i.e. the block indexer used as the recall stage

Reads the JSON ``bench_two_level_indexer.py`` writes (it needs the ``latency``
and ``coverage`` modes) and draws the four panels that decide between them:

    A  prefill selection latency vs context
    B  decode selection latency vs context
    C  what each actually selects, at the same 2048-token budget

    python plot_indexer_comparison.py
    python plot_indexer_comparison.py --in results/two_level_indexer/raw.json

Latency is GPU kernel time, which is the comparable measure: the two-level
driver's wall clock is host-launch bound at decode, and a server hides that in a
CUDA graph.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Categorical slots 1-3 of the validated palette, assigned by entity and held
# across all four panels. Text never wears these — the marks carry identity.
BLOCK = "#eb6834"  # slot 2, orange — the shipped path
TOKEN = "#1baf7a"  # slot 3, aqua — the exact one
TWO_LEVEL = "#2a78d6"  # slot 1, blue — the prototype, pooling before the dot
TWO_LEVEL_POST = "#eda100"  # slot 4, yellow — pooling after the dot
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
DPI = 200

SERIES = (
    ("block", BLOCK, "block"),
    ("flat", TOKEN, "token"),
    ("two_level", TWO_LEVEL, "two-level (pre-pool)"),
    ("two_level_post", TWO_LEVEL_POST, "two-level (post-pool)"),
)


def _ctx_label(n: int) -> str:
    return f"{n // 1024}k" if n < 1 << 20 else f"{n >> 20}M"


def _style(ax, *, xlabel=None, ylabel=None, title=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(True, axis="y", color=GRID, linewidth=1.0, solid_capstyle="butt")
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=INK_2)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)


def _ctx_axis(ax, ctxs):
    """Context on a log2 x (it is sampled in doublings); y stays linear."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(ctxs)
    ax.set_xticklabels([_ctx_label(c) for c in ctxs])
    ax.minorticks_off()


def _line(ax, xs, ys, colour, *, dashed=False):
    """2px line, >=8px end marker with a 2px surface ring."""
    ax.plot(
        xs,
        ys,
        color=colour,
        linewidth=2,
        solid_capstyle="round",
        dash_capstyle="round",
        linestyle=(0, (6, 3)) if dashed else "-",
        zorder=3,
    )
    ax.plot(
        xs[-1:],
        ys[-1:],
        marker="o",
        markersize=8,
        color=colour,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=4,
    )


def _end_label_group(ax, items, *, min_gap_pt=26, ha="left"):
    """Place several end labels, nudged apart so none collides.

    Series that converge (post-pool and block at 1M, the two pooling curves at
    saturation) would otherwise stack their labels on the same pixel. Nudge in
    display space, top-down, keeping every label at least `min_gap_pt` apart —
    the leader is where the mark is, so a shifted label still reads.
    """
    dpi = ax.figure.dpi
    disp = [ax.transData.transform((x, y))[1] for x, y, _ in items]
    gap_px = min_gap_pt * dpi / 72
    placed: list[float] = []
    target = {}
    for i in sorted(range(len(items)), key=lambda j: -disp[j]):
        t = disp[i]
        for p in placed:
            if abs(t - p) < gap_px:
                t = p - gap_px
        target[i] = t
        placed.append(t)
    for i, (x, y, text) in enumerate(items):
        _end_label(ax, x, y, text, dy_pt=(target[i] - disp[i]) * 72 / dpi, ha=ha)


def _end_label(ax, x, y, text, *, dy_pt=0.0, ha="left"):
    """Direct label in ink — the coloured end dot beside it carries identity."""
    ax.annotate(
        text,
        (x, y),
        xytext=(9 if ha == "left" else -9, dy_pt),
        textcoords="offset points",
        va="center",
        ha=ha,
        fontsize=9,
        color=INK_2,
    )


def panel_latency(ax, rows, *, phase, batch, title):
    pts = [
        r
        for r in rows
        if r.get("phase") == phase
        and r.get("batch_size") == batch
        and "gpu_kernel_ms" in r
    ]
    ctxs = sorted({r["context_len"] for r in pts})
    labels = []
    for impl, colour, label in SERIES:
        series = sorted(
            (r["context_len"], r["gpu_kernel_ms"]) for r in pts if r["impl"] == impl
        )
        if not series:
            continue
        xs = [p[0] for p in series]
        ys = [p[1] for p in series]
        _line(ax, xs, ys, colour)
        name = label.replace(" (", " ").replace(")", "")
        labels.append((xs[-1], ys[-1], f"{name}\n{ys[-1]:.2f} ms"))
    _ctx_axis(ax, ctxs)
    _style(ax, xlabel="context length", ylabel="GPU kernel time, ms", title=title)
    ax.set_xlim(ctxs[0] / 1.3, ctxs[-1] * 6.5)
    ax.set_ylim(0, max(y for _, y, _ in labels) * 1.12)
    _end_label_group(ax, labels)


def panel_coverage(ax, rows, *, pool_block):
    """Coverage vs the coarse budget M — all four indexers on one curve.

    The four are points on one axis, not four algorithms: post-pooling at
    M = topk/P is the block indexer exactly, and either variant at M = L/P is the
    exact token one. Plotting them together is the honest picture.
    """
    cov = [r for r in rows if r.get("metric") == "coverage"]
    if not cov:
        ax.set_visible(False)
        return
    ctx = cov[0]["context_len"]
    budget = cov[0]["budget_tokens"]

    curve_labels = []
    for position, colour, label in (
        ("pre", TWO_LEVEL, "pre-pool"),
        ("post", TWO_LEVEL_POST, "post-pool"),
    ):
        pts = sorted(
            (r["coarse_blocks"], r["recall_of_exact"] * 100)
            for r in cov
            if r.get("pool_position") == position
        )
        if not pts:
            continue
        _line(ax, [p[0] for p in pts], [p[1] for p in pts], colour)
        curve_labels.append((pts[-1][0], pts[-1][1], label))

    # The two one-level indexers are levels, not points on this axis: neither has
    # an M to sit at. A dotted rule is the right mark for a reference level — the
    # one case where dashing carries meaning rather than adding noise.
    for impl, colour, label in (("block", BLOCK, "block"), ("flat", TOKEN, "token")):
        r = next((x for x in cov if x["impl"] == impl), None)
        if r is None:
            continue
        y = r["recall_of_exact"] * 100
        ax.axhline(
            y,
            color=colour,
            linewidth=2,
            linestyle=(0, (1, 2.4)),
            dash_capstyle="round",
            zorder=2,
        )
        ax.annotate(
            f"{label}  {y:.1f}%",
            (0.995, y),
            xycoords=("axes fraction", "data"),
            xytext=(0, 6),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            color=INK_2,
        )

    ax.set_xscale("log", base=2)
    xs = sorted({r["coarse_blocks"] for r in cov if r.get("coarse_blocks")})
    ax.set_xticks(xs)
    # Second line spells M out in tokens: M blocks of `pool_block` tokens is what
    # stage 2 actually gets to choose from, and it is the quantity that decides
    # both the cost and the coverage. It also says what a block is without
    # needing a symbol for it.
    ax.set_xticklabels([f"{m}\n{m * pool_block:,}" for m in xs])
    ax.minorticks_off()
    ax.set_ylim(60, 106)
    _style(
        ax,
        xlabel="M — blocks stage 1 recalls  /  candidate tokens stage 2 scores",
        ylabel=f"share of the exact top-{budget} (%)",
        title=f"C  One axis, four indexers (decode, {_ctx_label(ctx)})",
    )
    _end_label_group(ax, curve_labels, min_gap_pt=16)


def make_figure(rows, out: Path, *, dpi: int = DPI) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    lat = [r for r in rows if "gpu_kernel_ms" in r]
    tl = next(r for r in lat if r["impl"] == "two_level")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    panel_latency(
        axes[0],
        lat,
        phase="prefill",
        batch=1,
        title="A  Prefill selection, one 2048-token extend",
    )
    panel_latency(
        axes[1], lat, phase="decode", batch=32, title="B  Decode selection, batch 32"
    )
    panel_coverage(axes[2], rows, pool_block=tl["pool_block"])

    handles = [
        Line2D(
            [],
            [],
            color=c,
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=l,
        )
        for _, c, l in SERIES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.005),
        labelcolor=INK_2,
    )

    fig.suptitle(
        "MiniMax-M3 indexers: block (shipped), token (exact), and two-level "
        "with stage-1 pooling before / after the dot",
        x=0.011,
        y=0.985,
        ha="left",
        fontsize=13.5,
        color=INK,
    )
    fig.text(
        0.008,
        0.935,
        "H200, 64Q/4KV/4 index heads, 2048-token budget, clustered index keys. "
        "Panels A and B are GPU kernel time.",
        ha="left",
        fontsize=9.5,
        color=INK_2,
    )
    fig.text(
        0.008,
        0.895,
        f"Two-level: stage 1 recalls M = {tl['coarse_blocks']} blocks, so stage 2 "
        f"scores {tl['candidate_width']:,} candidate tokens.",
        ha="left",
        fontsize=9.5,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.875))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i",
        "--in",
        dest="src",
        type=Path,
        default=here / "results" / "two_level_indexer" / "raw.json",
    )
    p.add_argument("-o", "--out", type=Path, default=None)
    args = p.parse_args(argv)

    rows = json.loads(args.src.read_text())
    out = args.out or args.src.parent / "indexer_comparison.png"
    make_figure(rows, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
