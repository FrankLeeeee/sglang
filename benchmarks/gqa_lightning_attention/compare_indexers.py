"""Compare full and reduced indexers using measured total-forward latencies."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(folder):
    rows = []
    for ci in (1, 2):
        data = json.loads((folder / f"config{ci}.json").read_text())
        for r in data["rows"]:
            assert r["status"] == "ok"
            assert len(r["samples_ms"]) == 15
            assert all(math.isfinite(x) and x > 0 for x in r["samples_ms"])
            assert statistics.median(r["samples_ms"]) == r["median_ms"]
        rows.extend(data["rows"])
    assert len(rows) == 192
    by_key = {
        (r["config"], r["phase"], r["context"], r["batch"], r["variant"]): r
        for r in rows
    }
    assert len(by_key) == 192
    return by_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reduced", type=Path, required=True)
    args = parser.parse_args()
    baseline, reduced = load(args.baseline), load(args.reduced)
    assert baseline.keys() == reduced.keys()
    rows = []
    for (ci, phase, length, batch, variant), r in sorted(reduced.items()):
        if variant != "sparse":
            continue
        old = baseline[(ci, phase, length, batch, variant)]
        dense = reduced[(ci, phase, length, batch, "dense")]
        old_dense = baseline[(ci, phase, length, batch, "dense")]
        assert r["indexer_q_heads"] == r["num_k_heads"] and r["indexer_kv_heads"] == 1
        assert (
            old["indexer_q_heads"] == old["num_q_heads"]
            and old["indexer_kv_heads"] == old["num_k_heads"]
        )
        for k in (
            "query_tokens",
            "cuda_graph",
            "page_size",
            "top_k",
            "num_q_heads",
            "num_k_heads",
            "head_dim",
            "hidden_dim",
        ):
            assert r[k] == old[k], k
        rows.append(
            {
                "config": ci,
                "phase": phase,
                "context": length,
                "batch": batch,
                "dense_ms": dense["median_ms"],
                "original_sparse_ms": old["median_ms"],
                "reduced_sparse_ms": r["median_ms"],
                "speedup_vs_original_sparse": old["median_ms"] / r["median_ms"],
                "speedup_vs_dense": dense["median_ms"] / r["median_ms"],
                "dense_rerun_ratio": dense["median_ms"] / old_dense["median_ms"],
            }
        )
    out = args.reduced
    with (out / "indexer_comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "indexer_comparison.json").write_text(json.dumps(rows, indent=2))
    lines = [
        "# Reduced-query, shared-key indexer experiment",
        "",
        "Configuration 1: indexer 48Q/4KV → 4Q/1KV; main attention remains 48Q/4KV.",
        "Configuration 2: indexer 64Q/8KV → 8Q/1KV; main attention remains 64Q/8KV.",
        "",
        "H200, BF16, TP=1, head dimension 128, indexer head dimension 64, hidden dimension 8192, exact top-2048 per attention group. All projection, norm, cache-write, scoring, top-k and attention costs are included. Each median uses 15 samples.",
        "",
        "Prefill is a 2,048-total-token final chunk, not full-prompt latency. Decode uses one new token per request and CUDA graph replay. Previous sparse results come from `lark_rerun_20260929`; dense and reduced sparse were freshly measured together. Cross-run differences include timing noise and changed score distributions.",
        "",
        "With one indexer query head per group, I[t,g,s] = w[t,g] * ReLU(q[t,g]·k[s]). For w>0 the gate cannot change the ranking; for w=0 all causal scores tie. The random-weight benchmark can therefore select the first 2,048 causal tokens for many groups under the smaller-index tie-break. This changes access locality and may affect timing; the experiment does not measure trained-model quality or isolate arithmetic savings alone.",
        "",
    ]
    charts = out / "indexer_charts"
    charts.mkdir(exist_ok=True)
    for ci in (1, 2):
        for phase in ("prefill", "decode"):
            section = [r for r in rows if r["config"] == ci and r["phase"] == phase]
            wins = sum(r["speedup_vs_dense"] > 1 for r in section)
            lines += [
                f"## Configuration {ci}: {phase}",
                "",
                f"Reduced sparse is faster than dense at {wins}/24 measured points.",
                "",
                "| Context | Batch | Dense ms | Original sparse ms | Reduced sparse ms | Original / reduced | Dense / reduced |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for r in section:
                lines.append(
                    f"| {r['context']//1024}k | {r['batch']} | {r['dense_ms']:.3f} | {r['original_sparse_ms']:.3f} | {r['reduced_sparse_ms']:.3f} | {r['speedup_vs_original_sparse']:.2f}× | {r['speedup_vs_dense']:.2f}× |"
                )
            lines.append("")
            for length in (16384, 32768, 65536, 131072):
                selected = [r for r in section if r["context"] == length]
                fig, ax = plt.subplots(figsize=(12, 5), dpi=180)
                fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.75)
                x = np.arange(6)
                labels = []
                for key, label, color, dx, hatch in [
                    ("dense_ms", "Dense (rerun)", "#2775AE", -0.26, None),
                    ("original_sparse_ms", "Original sparse", "#999999", 0, None),
                    (
                        "reduced_sparse_ms",
                        "Reduced indexer sparse",
                        "#E6A13A",
                        0.26,
                        "//",
                    ),
                ]:
                    bars = ax.bar(
                        x + dx,
                        [r[key] for r in selected],
                        0.24,
                        label=label,
                        color=color,
                        hatch=hatch,
                        linewidth=0.4,
                        edgecolor="#444444",
                    )
                    labels += ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
                ymax = max(
                    r[k]
                    for r in selected
                    for k in ("dense_ms", "original_sparse_ms", "reduced_sparse_ms")
                )
                ax.set_ylim(0, ymax * 1.3)
                ax.set_xticks(x, [str(r["batch"]) for r in selected])
                ax.set_xlabel("Batch size")
                ax.set_ylabel("Median total-forward latency (ms)")
                ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
                ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=9)
                fig.suptitle(
                    f"Configuration {ci} · {phase} · {length//1024}k context",
                    y=0.96,
                    fontsize=17,
                    fontweight="bold",
                )
                desc = "48Q/4KV → 4Q/1KV" if ci == 1 else "64Q/8KV → 8Q/1KV"
                fig.text(
                    0.5,
                    0.87,
                    f"Indexer {desc} · Main attention unchanged · H200 BF16",
                    ha="center",
                )
                scope = (
                    "Prefill: 2,048 total new tokens across batch, eager"
                    if phase == "prefill"
                    else "Decode: one new token/request, CUDA graph replay"
                )
                fig.text(
                    0.08,
                    0.08,
                    scope + ". All indexing costs included. Lower is better.",
                    fontsize=10,
                )
                fig.text(
                    0.08,
                    0.035,
                    "15 samples; random weights. Many zero-gated reduced-indexer rows tie. Independent zero-based y-axes.",
                    fontsize=9,
                )
                fig.canvas.draw()
                boxes = [
                    label.get_window_extent(fig.canvas.get_renderer())
                    for label in labels
                ]
                assert not any(
                    a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :]
                )
                path = charts / f"config{ci}_{phase}_{length//1024}k.png"
                fig.savefig(path, facecolor="white")
                plt.close(fig)
                lines += [
                    f"![Configuration {ci}, {phase}, {length//1024}k](indexer_charts/{path.name})",
                    "",
                ]
    (out / "indexer_comparison.md").write_text("\n".join(lines))
    print(f"Saved {len(rows)} comparisons and 16 three-way labeled charts to {out}")


if __name__ == "__main__":
    main()
