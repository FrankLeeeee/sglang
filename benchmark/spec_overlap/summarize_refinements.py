"""Collect ten measured schedules, source evidence and kernel intersections."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows = []
    for path in sorted(args.results.glob("*/experiment.json")):
        experiment = json.loads(path.read_text())
        directory = path.parent
        measured = directory / "gpt-oss-fine_overlap-b16.json"
        serial = directory / "gpt-oss-fine_serial-b16.json"
        if not measured.exists() or not serial.exists():
            continue
        report = directory / "comparison.json"
        if not report.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("analyze_results.py")),
                    "--results",
                    str(directory),
                ],
                stdout=subprocess.DEVNULL,
                check=True,
            )
        analysis = json.loads(report.read_text())
        prefix = "gpt-oss-"
        overlap = analysis["throughput"][prefix + "fine_overlap-b16"]
        control = analysis["throughput"][prefix + "fine_serial-b16"]
        original = analysis["throughput"][prefix + "off-b16"]
        traces = {
            mode: next(
                (t for t in analysis["traces"] if f"-{mode}-b16-profile" in t["trace"]),
                None,
            )
            for mode in ("off", "fine_serial", "fine_overlap")
        }
        trace = traces["fine_overlap"]
        cycles = (
            trace["stage_ranges"].get("microbatch.verify", {}).get("count", 0) / 2
            if trace
            else 0
        )
        row = dict(
            experiment,
            original_tps=original["tokens_per_second"],
            serial_tps=control["tokens_per_second"],
            overlap_tps=overlap["tokens_per_second"],
            overlap_vs_serial=overlap["speedup_vs_fine_serial"],
            overlap_vs_original=overlap["speedup_vs_original"],
            normal_matching_requests=overlap[
                "matching_requests_vs_original_each_repeat"
            ],
            profiled_cycles=cycles,
            traces=traces,
        )
        if trace:
            row["draft_gemm_target_comm_ms"] = trace[
                "draft_gemm_target_communication_intersection_ms"
            ]
            row["target_gemm_draft_comm_ms"] = trace[
                "target_gemm_draft_communication_intersection_ms"
            ]
            trace_path = Path(trace["trace"])
            row["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        rows.append(row)
    if args.require_complete and len(rows) != 10:
        raise RuntimeError(f"Expected ten complete rounds, found {len(rows)}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "refinements.json").write_text(json.dumps(rows, indent=2))
    if rows:
        fig, ax = plt.subplots(figsize=(10, 4))
        indices = [r["round"] for r in rows]
        ax.plot(
            indices,
            [r["original_tps"] for r in rows],
            "o--",
            label="Original full batch",
        )
        ax.plot(
            indices,
            [r["serial_tps"] for r in rows],
            "o-",
            label="Matching serial control",
        )
        ax.plot(indices, [r["overlap_tps"] for r in rows], "o-", label="Overlap")
        ax.set(
            xlabel="Refinement round",
            ylabel="Tokens/s (no profiler)",
            xticks=indices,
            title="GPT-OSS 20B + EAGLE3: batch 16, 128 output tokens",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output / "refinements_throughput.png", dpi=180)
        plt.close(fig)
    print(f"Summarized {len(rows)} rounds")


if __name__ == "__main__":
    main()
