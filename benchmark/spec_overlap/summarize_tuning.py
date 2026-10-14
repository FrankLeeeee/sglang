"""Compare granularity experiments using throughput and actual kernel overlap."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from run_matrix import compare_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    rows = []
    for directory in sorted(args.results.iterdir()):
        summary = directory / "analysis" / "summary.json"
        if not summary.exists():
            continue
        profiles = json.loads(summary.read_text())
        for status in json.loads((directory / "status.json").read_text()):
            if status["status"] not in {"ok", "existing"}:
                continue
            result = json.loads((directory / status["output"]).read_text())
            ranks = [
                row
                for row in profiles
                if row["batch"] == status["batch"]
                and row["context"] == status["context"]
            ]
            if {row["rank"] for row in ranks} != {0, 1}:
                raise ValueError(f"Missing rank: {directory}, {status}")
            row = dict(
                schedule=directory.name,
                batch=status["batch"],
                context=status["context"],
                tokens_per_second=result["median_tokens_per_second"],
                throughput_min=min(r["tokens_per_second"] for r in result["runs"]),
                throughput_max=max(r["tokens_per_second"] for r in result["runs"]),
                forbidden_max_ms=max(
                    value
                    for rank in ranks
                    for value in rank["maximum_forbidden_ms"].values()
                ),
                repeated_outputs_stable=all(
                    count == status["batch"]
                    for count in result["matching_requests_between_repeats"]
                ),
                reference_token_status="not_checked",
            )
            if args.reference:
                comparison = compare_tokens(
                    result,
                    json.loads((args.reference / status["output"]).read_text()),
                )
                row["reference_token_status"] = comparison["status"]
                row["mismatched_sequences"] = len(comparison["mismatches"])
            for key in (
                "span_ms",
                "draft_compute_coverage",
                "draft_comm_coverage",
                "draft_compute_spread_fraction",
                "draft_compute_occupied_deciles",
                "draft_compute_decile_cv",
                "draft_comm_gap_cv",
                "draft_comm_spread_fraction",
            ):
                row[key] = statistics.mean(rank[key] for rank in ranks)
            rows.append(row)
    (args.results / "summary.json").write_text(json.dumps(rows, indent=2))
    if rows:
        with (args.results / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
