"""Export labeled latency bars and Feishu XML from a completed benchmark sweep."""

import argparse
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
import zipfile
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def table(headers, rows):
    def cells(values, tag):
        return (
            "<tr>"
            + "".join(f"<{tag}><p>{escape(str(v))}</p></{tag}>" for v in values)
            + "</tr>"
        )

    return (
        "<table><thead>"
        + cells(headers, "th")
        + "</thead><tbody>"
        + "".join(cells(r, "td") for r in rows)
        + "</tbody></table>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    charts = root / "charts"
    charts.mkdir(exist_ok=True)
    drafts = root / "document"
    drafts.mkdir(exist_ok=True)
    datasets = [json.loads((root / f"config{c}.json").read_text()) for c in (1, 2)]
    configs = {ci: datasets[ci - 1]["rows"][0] for ci in (1, 2)}
    experiment = datasets[0]["metadata"]["arguments"].get("experiment")
    rows = [r for data in datasets for r in data["rows"]]
    contexts, batches = [16384, 32768, 65536, 131072], [1, 2, 4, 8, 16, 32]
    expected = {
        (c, p, l, b, v)
        for c in (1, 2)
        for p in ("prefill", "decode")
        for l in contexts
        for b in batches
        for v in ("dense", "sparse")
    }
    actual = {
        (r["config"], r["phase"], r["context"], r["batch"], r["variant"]) for r in rows
    }
    assert actual == expected and len(rows) == len(expected) == 192
    for r in rows:
        assert r["status"] == "ok"
        assert r["indexer_kv_heads"] in (1, r["num_k_heads"])
        assert r["indexer_q_heads"] % r["num_k_heads"] == 0
        assert len(r["samples_ms"]) == 15
        assert all(math.isfinite(v) and v > 0 for v in r["samples_ms"])
        assert r["median_ms"] == statistics.median(r["samples_ms"])
        assert r["query_tokens"] == (2048 if r["phase"] == "prefill" else r["batch"])
    lookup = {
        (r["config"], r["phase"], r["context"], r["batch"], r["variant"]): r
        for r in rows
    }
    comparisons = json.loads((root / "comparison.json").read_text())["comparisons"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelcolor": "#242424",
            "text.color": "#242424",
        }
    )
    manifest = []
    for ci in (1, 2):
        for phase in ("prefill", "decode"):
            for length in contexts:
                fig, ax = plt.subplots(figsize=(11, 4.8), dpi=180)
                fig.subplots_adjust(left=0.085, right=0.985, bottom=0.21, top=0.77)
                x = np.arange(len(batches))
                all_labels = []
                for variant, offset, color, hatch in [
                    ("dense", -0.2, "#2775AE", None),
                    ("sparse", 0.2, "#E6A13A", "//"),
                ]:
                    vals = [
                        lookup[(ci, phase, length, b, variant)]["median_ms"]
                        for b in batches
                    ]
                    bars = ax.bar(
                        x + offset,
                        vals,
                        width=0.36,
                        label=variant.capitalize(),
                        color=color,
                        edgecolor="#454545",
                        linewidth=0.5,
                        hatch=hatch,
                    )
                    labels = ax.bar_label(
                        bars, labels=[f"{v:.3f}" for v in vals], padding=5, fontsize=10
                    )
                    all_labels.extend(labels)
                    for batch, val, label in zip(batches, vals, labels):
                        manifest.append(
                            {
                                "file": f"config{ci}_{phase}_{length//1024}k.png",
                                "config": ci,
                                "phase": phase,
                                "context": length,
                                "batch": batch,
                                "variant": variant,
                                "median_ms": val,
                                "bar_label": label.get_text(),
                            }
                        )
                ymax = max(
                    lookup[(ci, phase, length, b, v)]["median_ms"]
                    for b in batches
                    for v in ("dense", "sparse")
                )
                ax.set_ylim(0, ymax * 1.23)
                ax.set_xticks(x, [str(b) for b in batches])
                ax.set_xlabel("Batch size")
                ax.set_ylabel("Median latency (ms)")
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", color="#DDDDDD", linewidth=0.65)
                ax.set_axisbelow(True)
                label = "Chunked prefill" if phase == "prefill" else "Decode"
                fig.suptitle(
                    f"Configuration {ci} · {label} · {length//1024}k context",
                    y=0.96,
                    fontsize=17,
                    fontweight="bold",
                )
                c = configs[ci]
                fig.text(
                    0.5,
                    0.865,
                    f"Main {c['num_q_heads']}Q/{c['num_k_heads']}KV · Indexer {c['indexer_q_heads']}Q/{c['indexer_kv_heads']}KV · H200 · BF16 · TP=1",
                    ha="center",
                    fontsize=11,
                )
                ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=10)
                timing = (
                    "Eager; 2,048 total new tokens across batch"
                    if phase == "prefill"
                    else "CUDA graph replay; one new token per request"
                )
                fig.text(0.085, 0.07, f"{timing}. Lower is better.", fontsize=10)
                fig.text(
                    0.085,
                    0.025,
                    "15 samples; all sparse indexing costs included. Each chart uses its own zero-based y-axis scale.",
                    fontsize=9,
                )
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                boxes = [t.get_window_extent(renderer) for t in all_labels]
                assert all(
                    fig.bbox.contains(bb.x0, bb.y0) and fig.bbox.contains(bb.x1, bb.y1)
                    for bb in boxes
                )
                assert not any(
                    a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :]
                ), "Bar labels overlap"
                name = charts / f"config{ci}_{phase}_{length//1024}k"
                fig.savefig(name.with_suffix(".png"), facecolor="white")
                fig.savefig(name.with_suffix(".pdf"), facecolor="white")
                plt.close(fig)
    assert len(manifest) == 192
    (charts / "manifest.json").write_text(json.dumps(manifest, indent=2))
    when = datasets[0]["metadata"]["timestamp_utc"]
    intro = [
        "<h1>Dense vs Lightning sparse GQA — full benchmark rerun</h1>",
        f"<p>Run started {escape(when)} (UTC). All 192 measurements completed: 2 configurations × 2 phases × 4 context lengths × 6 batch sizes × 2 variants. Each measurement has 15 timing samples.</p>",
        f"<p>Experiment: {experiment if experiment is not None else 'see configuration table'}. Main and indexer head counts are listed explicitly below.</p>",
        "<h2>Configuration and measurement scope</h2>",
        table(
            ["Parameter", "Configuration 1", "Configuration 2"],
            [
                ["Query heads", configs[1]["num_q_heads"], configs[2]["num_q_heads"]],
                ["Main KV heads", configs[1]["num_k_heads"], configs[2]["num_k_heads"]],
                ["Head dimension", 128, 128],
                ["Hidden dimension", 8192, 8192],
                [
                    "Indexer Q heads",
                    configs[1]["indexer_q_heads"],
                    configs[2]["indexer_q_heads"],
                ],
                [
                    "Indexer KV heads",
                    configs[1]["indexer_kv_heads"],
                    configs[2]["indexer_kv_heads"],
                ],
                ["Indexer head dimension", 64, 64],
                ["Selected tokens per group", 2048, 2048],
            ],
        ),
        "<p>Contexts are 16,384 / 32,768 / 65,536 / 131,072 tokens per request. Batch sizes are 1 / 2 / 4 / 8 / 16 / 32. Each configuration ran on one NVIDIA H200; BF16 weights, activations and cache; tensor parallelism 1; shuffled physical pages of 128 tokens.</p>",
        "<p><b>Prefill:</b> one final scheduled chunk of 2,048 total new tokens across the batch (2,048 down to 64 per request), measured eagerly. This is not full-prompt prefill or TTFT. <b>Decode:</b> one token per request, measured by CUDA graph replay.</p>",
        "<p>Latency includes Q/K/V projections, per-head RMSNorm, cache writes and attention. Sparse additionally includes indexer projections, index-cache writes, grouped scoring, exact top-2048 selection and padding conversion. No indexer RoPE or scaling. No main output projection or MLP.</p>",
        "<p>Dense kernels: FlashInfer paged FA2 prefill and tensor-core paged decode. Sparse kernels: SGLang autotuned Triton token-sparse GQA, fused Triton grouped Lightning scoring, and FlashInfer exact top-k. Three warmups precede 15 CUDA-event samples. A 64 MiB L2 flush occurs outside each timed interval. Allocation, metadata planning, compilation, graph capture and scheduler overhead are excluded.</p>",
        "<h2>Results at 128k context and batch 32</h2>",
    ]
    summary = []
    for ci in (1, 2):
        for phase in ("prefill", "decode"):
            r = next(
                r
                for r in comparisons
                if (r["config"], r["phase"], r["context"], r["batch"])
                == (ci, phase, 131072, 32)
            )
            summary.append(
                [
                    ci,
                    phase,
                    f"{r['dense_ms']:.3f}",
                    f"{r['sparse_ms']:.3f}",
                    f"{r['speedup_dense_over_sparse']:.2f}×",
                ]
            )
    intro.append(
        table(
            ["Config", "Phase", "Dense (ms)", "Sparse (ms)", "Dense / sparse"], summary
        )
    )
    prefill_wins = sum(
        r["speedup_dense_over_sparse"] > 1
        for r in comparisons
        if r["phase"] == "prefill"
    )
    intro.append(
        f"<p>Sparse was faster at {prefill_wins} of 48 prefill comparisons. Sparse decode becomes more competitive as context and batch increase. Dense / sparse ratios above 1 mean sparse is faster; near-parity differences should be interpreted with the raw timing spread.</p>"
    )
    intro.append(
        "<p>Every bar below is the median in milliseconds, with its value printed above it. Blue solid bars are dense; gold hatched bars are sparse. All axes start at zero, but y-axis ranges vary across charts for readability. Compare numeric labels when moving between charts.</p>"
    )
    (drafts / "00_intro.xml").write_text("\n".join(intro))
    n = 1
    for ci in (1, 2):
        for phase in ("prefill", "decode"):
            section = [
                f"<h1>Configuration {ci}: {'chunked prefill' if phase == 'prefill' else 'decode'}</h1>"
            ]
            for length in contexts:
                path = charts / f"config{ci}_{phase}_{length//1024}k.png"
                section += [
                    f"<h2>{length//1024}k context</h2>",
                    f'<img path="@./{path.as_posix()}" width="820" caption="Configuration {ci}, {phase}, {length//1024}k context. Median latency in milliseconds; lower is better."/>',
                ]
            (drafts / f"{n:02d}_config{ci}_{phase}.xml").write_text("\n".join(section))
            n += 1
    bundle = root / "benchmark_evidence.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for name in (
            "config1.json",
            "config2.json",
            "comparison.csv",
            "comparison.md",
            "comparison.json",
        ):
            z.write(root / name, name)
        z.write(charts / "manifest.json", "chart_manifest.json")
        for name in (
            "attention.py",
            "kernels.py",
            "benchmark.py",
            "summarize.py",
            "plot_results.py",
            "test_attention.py",
            "experiments.json",
            "compare_experiments.py",
        ):
            z.write(Path(__file__).parent / name, f"source/{name}")
    tail = [
        "<h1>Source data and reproducibility</h1>",
        "<p>CSV contains all 96 dense/sparse pairs. The evidence ZIP contains raw JSON with all 2,880 timing samples, software/GPU metadata, comparison tables, chart-label mappings and the benchmark source used for this rerun.</p>",
        f'<source path="@./{(root / "comparison.csv").as_posix()}" name="gqa_dense_sparse_comparison.csv"/>',
        f'<source path="@./{bundle.as_posix()}" name="gqa_benchmark_evidence.zip"/>',
        "<p>These are synthetic single-attention-layer measurements, not whole-model or whole-server latency. Historical caches and weights are random BF16 data. The benchmark measures execution cost, not model quality after sparsification. Results depend on kernel backend, precision, page size, score distribution and scheduling.</p>",
    ]
    (drafts / "05_sources.xml").write_text("\n".join(tail))
    for p in drafts.glob("*.xml"):
        ET.fromstring("<root>" + p.read_text() + "</root>")
    validation = {
        "measurements": len(rows),
        "comparisons": len(comparisons),
        "samples": sum(len(r["samples_ms"]) for r in rows),
        "charts": 16,
        "bar_labels": len(manifest),
        "label_overlap_check": "passed",
        "xml_parse": "passed",
        "source_sha256": {
            f"config{ci}.json": hashlib.sha256(
                (root / f"config{ci}.json").read_bytes()
            ).hexdigest()
            for ci in (1, 2)
        },
    }
    (root / "validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
