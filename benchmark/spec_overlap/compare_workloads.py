"""Compare every timed token sequence with a matching workload's reference output."""

import argparse
import json
from pathlib import Path

from run_matrix import compare_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for directory in args.candidate:
        for path in sorted(directory.glob("b*-c*.json")):
            reference = args.reference / path.name
            if not reference.exists():
                raise ValueError(f"Missing reference: {reference}")
            row = dict(candidate=str(path), reference=str(reference))
            row.update(
                compare_tokens(
                    json.loads(path.read_text()), json.loads(reference.read_text())
                )
            )
            rows.append(row)
    if not rows:
        raise ValueError("No completed workloads to compare")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    failed = [row for row in rows if row["status"] != "passed"]
    print(
        json.dumps(
            dict(
                workloads=len(rows),
                sequences=sum(row["compared_sequences"] for row in rows),
                failed_workloads=len(failed),
            )
        )
    )
    if failed:
        raise SystemExit("Token comparisons failed; inspect the saved report")


if __name__ == "__main__":
    main()
