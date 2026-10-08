"""Summarize stage GPU spans and render reproducible matrix plots."""

import argparse
import gzip
import json
import statistics
from pathlib import Path

from analyze_results import trace_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.results.glob("baseline-*/*-b*.json")):
        data = json.loads(path.read_text())
        traces = sorted(
            Path(data["profile_dir"]).glob("*TP-0.trace.json.gz"),
            key=lambda p: p.stat().st_mtime,
        )
        if not traces:
            continue
        summary = trace_summary(traces[-1])
        stages = summary["stage_ranges"]
        row = {
            "config": path.parent.name.removeprefix("baseline-"),
            "batch_size": data["batch_size"],
            "tokens_per_second": data["median_tokens_per_second"],
            "accept_length": statistics.mean(
                o["meta_info"]["spec_accept_length"] for o in data["outputs"]
            ),
            "trace_summary": summary,
        }
        for stage in ("draft", "verify", "draft_extend"):
            row[stage + "_ms"] = stages.get(stage, {}).get("median_ms")
            row[stage + "_count"] = stages.get(stage, {}).get("count", 0)
        with gzip.open(traces[-1], "rt") as stream:
            events = json.load(stream)["traceEvents"]
        row["communication_kernels"] = sorted(
            {
                e["name"]
                for e in events
                if e.get("cat") == "kernel"
                and any(
                    token in e["name"].lower()
                    for token in (
                        "nccl",
                        "all_gather",
                        "all_reduce",
                        "alltoall",
                        "reduce_scatter",
                    )
                )
            }
        )
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "baseline.json").write_text(json.dumps(rows, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = sorted({r["config"] for r in rows}, key=lambda s: int(s.split("-")[0]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for config in configs:
        values = sorted(
            (r for r in rows if r["config"] == config), key=lambda r: r["batch_size"]
        )
        for ax, key, title in zip(
            axes,
            ("draft_ms", "verify_ms", "draft_extend_ms"),
            ("Draft", "Verify + accept", "Draft extension"),
        ):
            ax.plot(
                [r["batch_size"] for r in values],
                [r[key] for r in values],
                "o-",
                label=config,
            )
            ax.set(
                title=title,
                xlabel="Batch size",
                ylabel="Median GPU span (ms)",
                xticks=[1, 8, 16, 32],
            )
            ax.grid(alpha=0.25)
    axes[-1].legend(title="steps-topk-tokens")
    fig.tight_layout()
    fig.savefig(args.output / "baseline_stages.png", dpi=180)
    plt.close(fig)
    print(f"Summarized {len(rows)} cases into {args.output}")


if __name__ == "__main__":
    main()
