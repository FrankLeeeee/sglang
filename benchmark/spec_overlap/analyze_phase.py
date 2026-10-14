"""Measure actual joint-graph kernel intersections using CUPTI clone identities."""

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_results import duration, intersection

from sglang.srt.speculative.graph_interleave import is_communication_kernel


def is_model_kernel(event):
    # CUDA emits these copy helpers for graph memory nodes without a source
    # kernel-node identity. Exclude them consistently from both schedules.
    return event.get("cat") == "kernel" and event.get("name") != "memcpy32_post"


def analyze(trace_path, server_log, mode="phase"):
    with gzip.open(trace_path, "rt") as stream:
        trace = json.load(stream)
    events = trace["traceEvents"]
    if mode == "launch":
        cycles = launch_cycles(events)
        return (
            summarize_cycles(
                trace_path, cycles, "CPU graph-launch scopes and CUPTI correlations"
            ),
            cycles,
        )
    cpu_pids = {e["pid"] for e in events if e.get("cat") == "cpu_op"}
    mapping = {}
    log = Path(server_log).read_text()
    decoder = json.JSONDecoder()
    position = 0
    while (position := log.find("PHASE_GRAPH_MAP ", position)) >= 0:
        # Worker writes can put two complete records on the same physical line.
        record, position = decoder.raw_decode(log, position + len("PHASE_GRAPH_MAP "))
        if record["pid"] in cpu_pids:
            mapping.update(record["nodes"])
    if not mapping:
        raise ValueError(f"No CUPTI graph-node map for {trace_path}")
    cycles = defaultdict(list)
    for event in events:
        if not is_model_kernel(event):
            continue
        args = event.get("args", {})
        metadata = mapping.get(str(args.get("graph node id")))
        if metadata:
            event["phase_metadata"] = metadata
            # A graph launch has one correlation id for all its kernels.
            cycles[args["correlation"]].append(event)
    return (
        summarize_cycles(trace_path, cycles, "CUPTI graph-node clone identities"),
        cycles,
    )


def launch_cycles(events):
    """Attribute separate graph launches, including the draft nested in verify A."""
    scopes = [e for e in events if e.get("cat") == "user_annotation"]
    launches = [
        e for e in events if e.get("name") in {"cudaGraphLaunch", "cuGraphLaunch"}
    ]
    kernels = defaultdict(list)
    for event in events:
        if is_model_kernel(event):
            kernels[event["args"]["correlation"]].append(event)

    def inside(inner, outer):
        return (
            inner["pid"] == outer["pid"]
            and inner["tid"] == outer["tid"]
            and outer["ts"] <= inner["ts"]
            and inner["ts"] + inner.get("dur", 0) <= outer["ts"] + outer["dur"] + 0.01
        )

    cycles = defaultdict(list)
    for index, verify in enumerate(
        e for e in scopes if e.get("name") == "microbatch.verify"
    ):
        draft_scopes = [
            s
            for s in scopes
            if s.get("name") == "microbatch.draft" and inside(s, verify)
        ]
        if not draft_scopes:
            continue  # Verification B has no independent drafting branch.
        target_scopes = [
            s
            for s in scopes
            if s.get("name") == "microbatch.target_chunk" and inside(s, verify)
        ]
        for role, role_scopes in (("target", target_scopes), ("draft", draft_scopes)):
            correlations = {
                launch["args"]["correlation"]
                for launch in launches
                if any(inside(launch, scope) for scope in role_scopes)
            }
            for correlation in correlations:
                for event in kernels[correlation]:
                    event["phase_metadata"] = {
                        "role": role,
                        "phase": (
                            "communication"
                            if is_communication_kernel(event["name"])
                            else "compute"
                        ),
                    }
                    cycles[index].append(event)
    return cycles


def summarize_cycles(trace_path, cycles, attribution):
    rows = []
    for correlation, kernels in cycles.items():
        lanes = defaultdict(list)
        for event in kernels:
            info = event["phase_metadata"]
            actual_comm = is_communication_kernel(event["name"])
            if actual_comm != (info["phase"] == "communication"):
                raise ValueError("Kernel classification disagrees with phase metadata")
            lanes[(info["role"], info["phase"])].append(
                (event["ts"], event["ts"] + event["dur"])
            )
        start = min(e["ts"] for e in kernels)
        end = max(e["ts"] + e["dur"] for e in kernels)
        tc, tm = lanes["target", "compute"], lanes["target", "communication"]
        dc, dm = lanes["draft", "compute"], lanes["draft", "communication"]
        if not all((tc, tm, dc, dm)):
            raise ValueError(f"Incomplete paired-graph attribution: {correlation}")
        row = dict(
            correlation=correlation, kernels=len(kernels), span_ms=(end - start) / 1000
        )
        for name, intervals in [
            ("target_compute", tc),
            ("target_comm", tm),
            ("draft_compute", dc),
            ("draft_comm", dm),
        ]:
            row[name + "_ms"] = duration(intervals) / 1000
        row["draft_compute_target_comm_ms"] = intersection(dc, tm) / 1000
        row["draft_comm_target_compute_ms"] = intersection(dm, tc) / 1000
        row["comm_comm_ms"] = intersection(dm, tm) / 1000
        row["draft_comm_draft_compute_ms"] = intersection(dm, dc) / 1000
        row["compute_compute_ms"] = intersection(dc, tc) / 1000
        row["draft_compute_coverage"] = intersection(dc, tm) / duration(dc) if dc else 0
        row["draft_comm_coverage"] = intersection(dm, tc) / duration(dm) if dm else 0
        row["draft_compute_spread_fraction"] = (
            max(b for a, b in dc) - min(a for a, b in dc)
        ) / (end - start)
        deciles = [
            intersection(
                dc,
                [
                    (
                        start + index * (end - start) / 10,
                        start + (index + 1) * (end - start) / 10,
                    )
                ],
            )
            for index in range(10)
        ]
        row["draft_compute_occupied_deciles"] = sum(value > 0 for value in deciles)
        row["draft_compute_decile_cv"] = statistics.pstdev(deciles) / statistics.mean(
            deciles
        )
        draft_intervals = dc + dm
        row["draft_spread_fraction"] = (
            max(b for a, b in draft_intervals) - min(a for a, b in draft_intervals)
        ) / (end - start)
        draft_kernels = sorted(
            (e for e in kernels if e["phase_metadata"]["role"] == "draft"),
            key=lambda e: e["ts"],
        )
        comm_starts = []
        previous_phase = None
        for event in draft_kernels:
            phase = event["phase_metadata"]["phase"]
            if phase == "communication" and phase != previous_phase:
                comm_starts.append(event["ts"])
            previous_phase = phase
        gaps = [b - a for a, b in zip(comm_starts, comm_starts[1:])]
        row["draft_comm_groups"] = len(comm_starts)
        row["draft_comm_gap_cv"] = (
            statistics.pstdev(gaps) / statistics.mean(gaps) if len(gaps) > 1 else 0
        )
        row["draft_comm_spread_fraction"] = (
            (max(b for a, b in dm) - min(a for a, b in dm)) / (end - start) if dm else 0
        )
        rows.append(row)
    if not rows:
        raise ValueError("No mapped kernel launches")
    medians = {
        key: statistics.median(row[key] for row in rows)
        for key in rows[0]
        if key != "correlation"
    }
    return {
        "trace": str(trace_path),
        "attribution": attribution,
        "kernel_scope": "Model kernels; CUDA memcpy32_post graph-copy helpers excluded in both schedules",
        "cycles": rows,
        "median": medians,
        "maximum_forbidden_ms": {
            key: max(row[key] for row in rows)
            for key in (
                "comm_comm_ms",
                "draft_comm_draft_compute_ms",
                "compute_compute_ms",
            )
        },
    }


def plot(cycles, output, mode="phase"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kernels = sorted(cycles.values(), key=lambda ks: min(e["ts"] for e in ks))[
        len(cycles) // 2
    ]
    start = min(e["ts"] for e in kernels)
    end = max(e["ts"] + e["dur"] for e in kernels)
    keys = [
        ("target", "compute"),
        ("target", "communication"),
        ("draft", "compute"),
        ("draft", "communication"),
    ]
    fig, ax = plt.subplots(figsize=(13, 3.5))
    for index, key in enumerate(keys):
        spans = [
            ((e["ts"] - start) / 1000, e["dur"] / 1000)
            for e in kernels
            if (e["phase_metadata"]["role"], e["phase_metadata"]["phase"]) == key
        ]
        ax.broken_barh(
            spans,
            (index - 0.3, 0.6),
            facecolors=["#2455a4", "#ec9e21", "#38a169", "#c95b87"][index],
        )
    ax.set(
        yticks=range(4),
        yticklabels=[" ".join(key).title() for key in keys],
        xlim=(0, (end - start) / 1000),
        xlabel="Time within paired graph (ms)",
        title=f"Measured kernels: verify A + draft B ({mode} schedule)",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--mode", choices=["phase", "launch"], default="phase")
    args = parser.parse_args()
    report, cycles = analyze(args.trace, args.server_log, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["median"], indent=2))
    if args.plot:
        plot(cycles, args.plot, args.mode)


if __name__ == "__main__":
    main()
