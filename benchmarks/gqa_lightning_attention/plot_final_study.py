"""Speedup-vs-context plots for the final study, one figure per main config and phase.

Rows are the indexer type, columns the batch size, color the selection method and
line style the indexer head dim. Requires ``final_study_results.json`` (see
summarize_final_study.py) and matplotlib.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Categorical slots 1-3 of the reference palette (validated all-pairs).
METHODS = {"token": ("Token (single stage)", "#2a78d6"), "block": ("Block (top 16 x 128)", "#eb6834"), "two_stage": ("Two-stage (128 blocks)", "#1baf7a")}
DIM_STYLE = {64: ("-", "o"), 128: ("--", "s")}
MAIN_LABEL = {"48q4kv": "48 Q / 4 KV", "64q8kv": "64 Q / 8 KV", "80q8kv": "80 Q / 8 KV", "64q4kv": "64 Q / 4 KV"}


def speedup_formatter(value, _):
    return f"{value:g}x"


def draw_panel(ax, rows, main, phase, indexer, batch, contexts):
    ax.axhline(1.0, color=INK_2, linewidth=1.0, zorder=1)
    for method, (_, color) in METHODS.items():
        for dim, (linestyle, marker) in DIM_STYLE.items():
            points = sorted(
                (r["context"], r["speedup"])
                for r in rows
                if (r["main"], r["phase"], r["indexer"], r["batch"], r["method"], r["dim"])
                == (main, phase, indexer, batch, method, dim)
                and "speedup" in r
            )
            if points:
                xs = [contexts.index(c) for c, _ in points]
                ax.plot(
                    xs,
                    [s for _, s in points],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=5,
                    linewidth=2,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1,
                    zorder=3,
                )
    ax.set_yscale("log")
    ax.set_xticks(range(len(contexts)), [f"{c // 1024}k" if c < 1 << 20 else "1M" for c in contexts])
    ax.yaxis.set_major_formatter(FuncFormatter(speedup_formatter))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def figure(rows, main, phase, contexts, batches, indexers, path):
    fig, axes = plt.subplots(
        len(indexers), len(batches), figsize=(4.2 * len(batches), 3.4 * len(indexers) + 0.9), sharex=True, facecolor=SURFACE, squeeze=False
    )
    for i, indexer in enumerate(indexers):
        for j, batch in enumerate(batches):
            ax = axes[i][j]
            draw_panel(ax, rows, main, phase, indexer, batch, contexts)
            if i == 0:
                ax.set_title(f"batch {batch}", color=INK, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{indexer} indexer\nspeedup over best dense", color=INK_2, fontsize=10)
            if i == len(indexers) - 1:
                ax.set_xlabel("context length", color=INK_2, fontsize=10)
    # Shared log range per row so batch panels compare directly.
    for i in range(len(indexers)):
        lows, highs = zip(*(axes[i][j].get_ylim() for j in range(len(batches))))
        for j in range(len(batches)):
            axes[i][j].set_ylim(min(lows), max(highs))
    handles = [Line2D([0], [0], color=c, linewidth=2, label=name) for name, c in METHODS.values()]
    handles += [
        Line2D([0], [0], color=INK_2, linestyle=DIM_STYLE[d][0], marker=DIM_STYLE[d][1], markersize=5, linewidth=2, label=f"indexer head dim {d}")
        for d in DIM_STYLE
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, labelcolor=INK_2, fontsize=9)
    fig.suptitle(
        f"{MAIN_LABEL[main]} - {phase}: sparse speedup over best dense (above 1x = sparse faster)",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path(__file__).parent / "results/final_study")
    args = p.parse_args()
    data = json.loads((args.results / "final_study_results.json").read_text())
    rows = [r for r in data["measurements"] if r["method"] != "dense" and r["status"] == "ok"]
    out = args.results / "plots"
    out.mkdir(exist_ok=True)
    for main_cfg in data["main_configs"]:
        for phase in ("prefill", "decode"):
            figure(rows, main_cfg, phase, data["contexts"], data["batches"], ["identical", "reduced"], out / f"speedup_{main_cfg}_{phase}.png")
    print(f"wrote {len(data['main_configs']) * 2} figures to {out}")


if __name__ == "__main__":
    main()
