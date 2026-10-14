"""Validate fixed-batch evidence and export the context sweep without hiding gaps."""

import argparse
import csv
import gzip
import json
import re
import statistics
from pathlib import Path

MODES = ("no-spec", "spec", "launch", "phase", "phase-serial")


def summarize(root):
    rows = []
    for mode in MODES:
        directory = root / mode
        status_file = directory / "status.json"
        if not status_file.exists():
            continue
        for status in json.loads(status_file.read_text()):
            row = dict(mode=mode, **status)
            result_path = directory / status.get("output", "missing.json")
            if status["status"] not in {"ok", "existing"} or not result_path.exists():
                rows.append(row)
                continue
            result = json.loads(result_path.read_text())
            outputs = result["outputs"]
            metadata = [o["meta_info"] for o in outputs]
            batch, context = status["batch"], status["context"]
            expected = result["output_len"]
            row.update(
                tokens_per_second=result["median_tokens_per_second"],
                cold_seconds=result["cold_seconds"],
                warm_seconds=statistics.median(r["seconds"] for r in result["runs"]),
                throughput_min=min(r["tokens_per_second"] for r in result["runs"]),
                throughput_max=max(r["tokens_per_second"] for r in result["runs"]),
                completed_requests=len(outputs),
                retractions=sum(m.get("num_retractions", 0) for m in metadata),
                cached_tokens_min=min(m["cached_tokens"] for m in metadata),
                prompt_tokens_min=min(m["prompt_tokens"] for m in metadata),
                completion_tokens_min=min(m["completion_tokens"] for m in metadata),
            )
            errors = []
            verify_count = sum(m.get("spec_verify_ct", 0) for m in metadata)
            proposed = sum(m.get("spec_num_proposed_drafts", 0) for m in metadata)
            if verify_count:
                row["spec_accept_length"] = (
                    sum(m["completion_tokens"] for m in metadata) / verify_count
                )
            if proposed:
                row["spec_accept_rate"] = (
                    sum(m.get("spec_num_correct_drafts", 0) for m in metadata)
                    / proposed
                )
            if len(outputs) != batch or any(
                m["completion_tokens"] != expected for m in metadata
            ):
                errors.append("incomplete_outputs")
            if any(
                m["prompt_tokens"]
                != context - expected - result.get("context_reserve_tokens", 0)
                for m in metadata
            ):
                errors.append("wrong_context")
            if row["retractions"]:
                errors.append("retractions")
            profile = Path(result["profile_dir"])
            for rank in range(2):
                traces = sorted(profile.glob(f"*TP-{rank}*.trace.json.gz"))
                if len(traces) != 1:
                    errors.append(f"rank{rank}_trace_count_{len(traces)}")
                    continue
                with gzip.open(traces[0], "rt") as stream:
                    events = json.load(stream)["traceEvents"]
                batches = [
                    int(match[1])
                    for event in events
                    if event.get("cat") == "user_annotation"
                    and (
                        match := re.match(
                            r"step\[(?:DECODE|VERIFY) bs=(\d+)", event.get("name", "")
                        )
                    )
                ]
                row[f"rank{rank}_profile_batches"] = sorted(set(batches))
                gpu_steps = [
                    e["dur"] / 1000
                    for e in events
                    if e.get("cat") == "gpu_user_annotation"
                    and re.match(r"step\[(?:DECODE|VERIFY) bs=", e.get("name", ""))
                ]
                if gpu_steps:
                    row[f"rank{rank}_profiled_gpu_step_median_ms"] = statistics.median(
                        gpu_steps
                    )
                row[f"rank{rank}_graph_launches"] = sum(
                    "cudaGraphLaunch" in e.get("name", "") for e in events
                )
                row[f"rank{rank}_phase_pairs"] = sum(
                    e.get("name") == "microbatch.phase_pair"
                    and e.get("cat") == "user_annotation"
                    for e in events
                )
                if not batches or any(size != batch for size in batches):
                    errors.append(f"rank{rank}_nonfixed_batch")
                if not row[f"rank{rank}_graph_launches"]:
                    errors.append(f"rank{rank}_no_graph")
            row["validation"] = ";".join(errors) if errors else "passed"
            rows.append(row)
    return rows


def plot(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharey=True)
    for ax, context in zip(axes, (4096, 16384, 32768, 65536, 131072)):
        for mode in MODES:
            data = sorted(
                (
                    r
                    for r in rows
                    if r["mode"] == mode
                    and r["context"] == context
                    and r.get("validation") == "passed"
                ),
                key=lambda r: r["batch"],
            )
            if data:
                ax.plot(
                    [r["batch"] for r in data],
                    [r["tokens_per_second"] for r in data],
                    marker="o",
                    markersize=3,
                    label=mode,
                )
        ax.set(
            title=f"{context // 1024}k total context",
            xlabel="Batch size",
            xscale="log",
            yscale="log",
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Generated tokens/s (cached prefix)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = summarize(args.results)
    (args.output / "context-summary.json").write_text(json.dumps(rows, indent=2))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "context-summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output / "throughput.png")
    print(
        json.dumps(
            {
                mode: {
                    "completed": sum(
                        r["mode"] == mode and "tokens_per_second" in r for r in rows
                    ),
                    "validation_failures": [
                        f"b{r['batch']}-c{r['context']}: {r['validation']}"
                        for r in rows
                        if r["mode"] == mode
                        and r.get("validation", "passed") != "passed"
                    ],
                }
                for mode in MODES
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
