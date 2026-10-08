"""Draw an actual decode interval, retaining GPU kernel-level concurrency."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    annotations = [e for e in events if e.get("cat") == "gpu_user_annotation"]
    targets = [
        e
        for e in annotations
        if e["name"] in {"microbatch.verify", "microbatch.target_chunk"}
        or e["name"].startswith("step[VERIFY ")
    ]
    drafts = [
        e
        for e in annotations
        if e["name"].startswith("microbatch.draft")
        or e["name"] == "draft"
        or e["name"].startswith("step[DRAFT")
    ]
    first = min(
        (e for e in targets if e["name"] == "microbatch.verify"), key=lambda e: e["ts"]
    )
    start, end = first["ts"], first["ts"] + first["dur"]
    lanes = {
        name: []
        for name in (
            "Target compute",
            "Target communication",
            "Draft compute",
            "Draft communication",
        )
    }

    def inside(kernel, spans):
        return any(
            kernel["pid"] == e["pid"]
            and kernel["tid"] == e["tid"]
            and kernel["ts"] >= e["ts"]
            and kernel["ts"] + kernel["dur"] <= e["ts"] + e["dur"] + 1
            for e in spans
        )

    for e in events:
        if e.get("cat") != "kernel" or e["ts"] >= end or e["ts"] + e["dur"] <= start:
            continue
        role = (
            "Target" if inside(e, targets) else "Draft" if inside(e, drafts) else None
        )
        if role is None:
            continue
        comm = any(
            t in e["name"].lower()
            for t in ("nccl", "all_gather", "all_reduce", "alltoall", "reduce_scatter")
        )
        lane = role + (" communication" if comm else " compute")
        left, right = max(start, e["ts"]), min(end, e["ts"] + e["dur"])
        lanes[lane].append(((left - start) / 1000, (right - left) / 1000))
    fig, ax = plt.subplots(figsize=(12, 3.3))
    for index, (lane, intervals) in enumerate(lanes.items()):
        ax.broken_barh(
            intervals,
            (index - 0.3, 0.6),
            facecolors=["#2455a4", "#ec9e21", "#38a169", "#c95b87"][index],
        )
    ax.set(
        yticks=range(4),
        yticklabels=list(lanes),
        xlabel="Time from first verify A (ms)",
        title="Measured GPU kernels: verify A / independent draft B",
        xlim=(0, (end - start) / 1000),
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
