"""Validate fixed-batch evidence and export the context sweep without hiding gaps."""

import argparse
import csv
import gzip
import json
import re
import statistics
from pathlib import Path

MODES = ("no-spec", "spec", "launch", "phase", "phase-serial")
LABELS = {
    "no-spec": "Target only",
    "spec": "Standard EAGLE3",
    "launch": "Previous overlap",
    "phase": "Phase-fenced overlap",
    "phase-serial": "Phase serial control",
}


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
            cold_outputs = result["cold_outputs"]
            if isinstance(cold_outputs, dict):
                cold_outputs = [cold_outputs]
            cold_tokens = [o["meta_info"]["completion_tokens"] for o in cold_outputs]
            row.update(
                tokens_per_second=result["median_tokens_per_second"],
                cold_seconds=result["cold_seconds"],
                cold_output_tokens_per_second=sum(cold_tokens) / result["cold_seconds"],
                warm_seconds=statistics.median(r["seconds"] for r in result["runs"]),
                throughput_min=min(r["tokens_per_second"] for r in result["runs"]),
                throughput_max=max(r["tokens_per_second"] for r in result["runs"]),
                completed_requests=len(outputs),
                retractions=sum(m.get("num_retractions", 0) for m in metadata),
                cached_tokens_min=min(m["cached_tokens"] for m in metadata),
                prompt_tokens_min=min(m["prompt_tokens"] for m in metadata),
                completion_tokens_min=min(m["completion_tokens"] for m in metadata),
                repeated_outputs_stable=all(
                    count == batch
                    for count in result["matching_requests_between_repeats"]
                ),
            )
            errors = []
            if len(cold_tokens) != batch or any(n != expected for n in cold_tokens):
                errors.append("incomplete_cold_outputs")
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
            if any(run["tokens"] != batch * expected for run in result["runs"]):
                errors.append("incomplete_timed_repeat")
            if any(
                len(sequences) != batch
                or any(len(ids) != expected for ids in sequences)
                for sequences in result["repeat_outputs"]
            ):
                errors.append("incomplete_repeated_token_ids")
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
                steps = [
                    (event, int(match[1]))
                    for event in events
                    if event.get("cat") == "user_annotation"
                    and (
                        match := re.match(
                            r"step\[(?:DECODE|VERIFY) bs=(\d+)", event.get("name", "")
                        )
                    )
                ]
                batches = [size for event, size in steps]
                row[f"rank{rank}_profile_batches"] = sorted(set(batches))
                grouped_batches = []
                iteration_ids = []
                for parent in events:
                    if (
                        parent.get("cat") != "user_annotation"
                        or parent.get("name") != "scheduler.run_batch"
                    ):
                        continue
                    sizes = [
                        size
                        for event, size in steps
                        if event["pid"] == parent["pid"]
                        and event["tid"] == parent["tid"]
                        and parent["ts"] <= event["ts"]
                        and event["ts"] + event["dur"]
                        <= parent["ts"] + parent["dur"] + 0.01
                    ]
                    if sizes:
                        grouped_batches.append(sorted(sizes))
                        iteration_ids.append(parent["args"]["External id"])
                expected_sizes = [batch]
                if mode in {"launch", "phase", "phase-serial"} and batch > 1:
                    percent = result["server_info"][
                        "speculative_microbatch_split_percent"
                    ]
                    cut = max(1, min(batch - 1, batch * percent // 100))
                    expected_sizes = sorted([cut, batch - cut])
                row[f"rank{rank}_iteration_request_batches"] = sorted(
                    {sum(sizes) for sizes in grouped_batches}
                )
                gpu_iterations = []
                for external_id in iteration_ids:
                    ranges = [
                        (event["ts"], event["ts"] + event["dur"])
                        for event in events
                        if event.get("cat") == "gpu_user_annotation"
                        and event.get("name") == "scheduler.run_batch"
                        and event.get("args", {}).get("External id") == external_id
                    ]
                    if ranges:
                        gpu_iterations.append(
                            (
                                max(end for start, end in ranges)
                                - min(start for start, end in ranges)
                            )
                            / 1000
                        )
                if gpu_iterations:
                    row[f"rank{rank}_profiled_gpu_iteration_median_ms"] = (
                        statistics.median(gpu_iterations)
                    )
                gpu_steps = [
                    e["dur"] / 1000
                    for e in events
                    if e.get("cat") == "gpu_user_annotation"
                    and re.match(r"step\[(?:DECODE|VERIFY) bs=", e.get("name", ""))
                ]
                if gpu_steps:
                    row[f"rank{rank}_profiled_gpu_model_call_median_ms"] = (
                        statistics.median(gpu_steps)
                    )
                row[f"rank{rank}_graph_launches"] = sum(
                    "cudaGraphLaunch" in e.get("name", "") for e in events
                )
                row[f"rank{rank}_phase_pairs"] = sum(
                    e.get("name") == "microbatch.phase_pair"
                    and e.get("cat") == "user_annotation"
                    for e in events
                )
                if (
                    not grouped_batches
                    or any(sizes != expected_sizes for sizes in grouped_batches)
                    or sum(map(len, grouped_batches)) != len(steps)
                ):
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

    fig, grid = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    axes = grid.ravel()
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
                    label=LABELS[mode],
                )
        ax.set(
            title=f"{context // 1024}k total context",
            xlabel="Batch size",
            xscale="log",
            yscale="log",
            xlim=(1, 256),
        )
        batches = (1, 4, 8, 16, 32, 64, 128, 256)
        ax.set_xticks(batches, [str(batch) for batch in batches], fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Generated tokens/s (cached prefix)")
    axes[3].set_ylabel("Generated tokens/s (cached prefix)")
    axes[-1].set_axis_off()
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, loc="center", fontsize=11)
    axes[-1].text(
        0.5,
        0.1,
        "EP=TP=2; two H200 GPUs\n128 output tokens; median of 3 runs\nNative TBO disabled",
        ha="center",
        transform=axes[-1].transAxes,
    )
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
    plot_cold(rows, args.output / "cold-latency-b16.png")
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


def plot_cold(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for mode in MODES:
        data = sorted(
            (
                row
                for row in rows
                if row["mode"] == mode
                and row["batch"] == 16
                and row.get("validation") == "passed"
            ),
            key=lambda row: row["context"],
        )
        if data:
            ax.plot(
                [row["context"] // 1024 for row in data],
                [row["cold_seconds"] for row in data],
                marker="o",
                label=LABELS[mode],
            )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Total context budget (k tokens)",
        ylabel="Cold request latency (seconds)",
        title="Batch 16: cold prefix cache, prefill + 128 output tokens/request",
    )
    ax.set_xticks([4, 16, 32, 64, 128], ["4", "16", "32", "64", "128"])
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
