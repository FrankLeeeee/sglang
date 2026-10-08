"""Compare useful throughput/output identity and GPU interval unions."""

import argparse
import gzip
import json
import statistics
from pathlib import Path


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def duration(intervals):
    return sum(end - start for start, end in union(intervals))


def intersection(first, second):
    first, second = union(first), union(second)
    i = j = 0
    total = 0
    while i < len(first) and j < len(second):
        a, b = first[i], second[j]
        total += max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] < b[1]:
            i += 1
        else:
            j += 1
    return total


def trace_summary(path):
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    stages = {}
    stage_events = []
    for event in events:
        name = event.get("name", "")
        if event.get("cat") == "gpu_user_annotation" and (
            name
            in {
                "draft",
                "verify",
                "draft_extend",
                "microbatch.draft",
                "microbatch.draft_chunk",
                "microbatch.target_chunk",
                "microbatch.verify",
                "microbatch.draft_extend",
            }
            or name.startswith(("step[VERIFY ", "step[DRAFT"))
        ):
            stage_events.append(event)
            stages.setdefault(name, []).append(
                (event["ts"], event["ts"] + event["dur"])
            )

    def stage_role(name):
        if (
            "verify" in name
            or name == "microbatch.target_chunk"
            or name.startswith("step[VERIFY ")
        ):
            return "verify"
        if "draft" in name or name.startswith("step[DRAFT"):
            return "draft"
        return None

    target = [
        v for k, values in stages.items() if stage_role(k) == "verify" for v in values
    ]
    draft = [
        v for k, values in stages.items() if stage_role(k) == "draft" for v in values
    ]
    # This measures GPU stage-span concurrency, including waits and launch gaps.
    # It must not be reported as useful compute overlap or saved cycle time.
    kernels = [e for e in events if e.get("cat") == "kernel"]
    comm, compute = [], []
    for event in kernels:
        name = event["name"].lower()
        intervals = (
            comm
            if any(
                token in name
                for token in (
                    "nccl",
                    "all_gather",
                    "all_reduce",
                    "alltoall",
                    "reduce_scatter",
                )
            )
            else compute
        )
        intervals.append((event["ts"], event["ts"] + event["dur"]))

    def inside(event, span):
        return (
            event["pid"] == span["pid"]
            and event["tid"] == span["tid"]
            and event["ts"] >= span["ts"]
            and event["ts"] + event["dur"] <= span["ts"] + span["dur"] + 1
        )

    def belongs(event, role):
        return any(
            stage_role(span["name"]) == role and inside(event, span)
            for span in stage_events
        )

    # Captured graphs may repeat "draft" on internal streams. The outer
    # stage range measures the complete proposal, rather than one graph step.
    outer_streams = {
        (e["pid"], e["tid"])
        for e in stage_events
        if e["name"] in {"verify", "microbatch.verify"}
    }
    measured_spans = [
        e
        for e in stage_events
        if e["name"] != "draft"
        or not outer_streams
        or (e["pid"], e["tid"]) in outer_streams
    ]
    # Prefill also extends the draft KV cache; exclude it from decode E.
    first_verify = min(
        (
            e["ts"]
            for e in measured_spans
            if e["name"] in {"verify", "microbatch.verify"}
        ),
        default=0,
    )
    measured_spans = [
        e
        for e in measured_spans
        if e["name"] != "draft_extend" or e["ts"] >= first_verify
    ]
    measured_stages = {}
    for e in measured_spans:
        measured_stages.setdefault(e["name"], []).append((e["ts"], e["ts"] + e["dur"]))
    activity = {}
    for span in measured_spans:
        active = (
            duration(
                [(e["ts"], e["ts"] + e["dur"]) for e in kernels if inside(e, span)]
            )
            / 1000
        )
        activity.setdefault(span["name"], []).append(
            (active, span["dur"] / 1000 - active)
        )

    def communication(event):
        name = event["name"].lower()
        return any(
            token in name
            for token in (
                "nccl",
                "all_gather",
                "all_reduce",
                "alltoall",
                "reduce_scatter",
            )
        )

    def intervals(role, is_comm, gemm_only=False):
        return [
            (e["ts"], e["ts"] + e["dur"])
            for e in kernels
            if belongs(e, role)
            and communication(e) == is_comm
            and (
                not gemm_only
                or any(
                    token in e["name"].lower() for token in ("gemm", "matmul", "nvjet")
                )
            )
        ]

    return {
        "trace": str(path),
        "draft_gemm_target_communication_intersection_ms": intersection(
            intervals("draft", False, gemm_only=True), intervals("verify", True)
        )
        / 1000,
        "target_gemm_draft_communication_intersection_ms": intersection(
            intervals("verify", False, gemm_only=True), intervals("draft", True)
        )
        / 1000,
        "draft_compute_target_communication_intersection_ms": intersection(
            intervals("draft", False), intervals("verify", True)
        )
        / 1000,
        "target_compute_draft_communication_intersection_ms": intersection(
            intervals("verify", False), intervals("draft", True)
        )
        / 1000,
        "stage_ranges": {
            name: {
                "count": len(values),
                "median_ms": statistics.median(b - a for a, b in values) / 1000,
                "median_kernel_union_ms": statistics.median(
                    a for a, _ in activity[name]
                ),
                "median_unoccupied_stream_ms": statistics.median(
                    g for _, g in activity[name]
                ),
            }
            for name, values in measured_stages.items()
        },
        "draft_target_stage_intersection_ms": intersection(draft, target) / 1000,
        "compute_communication_kernel_intersection_ms": intersection(compute, comm)
        / 1000,
        "communication_kernel_union_ms": duration(comm) / 1000,
        "compute_kernel_union_ms": duration(compute) / 1000,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).parent / "results"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = {}
    for path in sorted(args.results.glob("*-b*.json")):
        data = json.loads(path.read_text())
        runs[path.stem] = data
    report = {"throughput": {}, "traces": []}
    for name, data in runs.items():
        model, _mode, batch = name.rsplit("-", 2)
        baseline = runs.get(f"{model}-off-{batch}")
        serial = runs.get(f"{model}-serial-{batch}")
        fine_serial = runs.get(f"{model}-fine_serial-{batch}")
        identity = None
        if baseline:
            identity = [
                output.get("output_ids", output["text"])
                == expected.get("output_ids", expected["text"])
                for output, expected in zip(
                    data["outputs"], baseline["outputs"], strict=True
                )
            ]
        repeat_identity = None
        if baseline and data.get("repeat_outputs"):
            expected_ids = [o.get("output_ids", o["text"]) for o in baseline["outputs"]]
            repeat_identity = [
                sum(a == b for a, b in zip(expected_ids, observed, strict=True))
                for observed in data["repeat_outputs"]
            ]
        rate = data["median_tokens_per_second"]
        report["throughput"][name] = {
            "tokens_per_second": rate,
            "speedup_vs_original": rate / baseline["median_tokens_per_second"]
            if baseline
            else None,
            "speedup_vs_fine_serial": rate / fine_serial["median_tokens_per_second"]
            if fine_serial
            else None,
            "speedup_vs_split_serial": rate / serial["median_tokens_per_second"]
            if serial
            else None,
            "matching_requests_vs_original_each_repeat": repeat_identity,
            "exact_token_check_passed": all(
                count == len(data["outputs"]) for count in repeat_identity
            )
            if repeat_identity is not None
            else None,
            "output_comparison": "token_ids"
            if "output_ids" in data["outputs"][0]
            else "text",
            "matching_requests_vs_original": sum(identity)
            if identity is not None
            else None,
            "num_requests": len(data["outputs"]),
            "matching_requests_between_repeats": data.get(
                "matching_requests_between_repeats"
            ),
            "mean_accept_length": statistics.mean(
                o["meta_info"]["spec_accept_length"] for o in data["outputs"]
            ),
        }
    directories = {
        Path(data["profile_dir"]).resolve()
        for data in runs.values()
        if data.get("profile_dir")
    }
    directories.update(p.resolve() for p in args.results.glob("*-profile"))
    for directory in sorted(directories):
        paths = list(directory.glob("*TP-0.trace.json.gz"))
        if paths:
            # Re-runs retain prior captures; compare the latest capture per run.
            path = max(paths, key=lambda candidate: candidate.stat().st_mtime)
            report["traces"].append(trace_summary(path))
    output = args.output or args.results / "comparison.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
