"""Batch actual-kernel overlap analysis for both ranks of a completed sweep."""

import argparse
import json
from pathlib import Path

from analyze_phase import analyze, plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["phase", "launch"], default="phase")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for status in json.loads((args.results / "status.json").read_text()):
        if status["status"] not in {"ok", "existing"} or status["batch"] == 1:
            continue
        result = json.loads((args.results / status["output"]).read_text())
        profile = Path(result["profile_dir"])
        for rank in (0, 1):
            trace = next(profile.glob(f"*TP-{rank}*.trace.json.gz"))
            report, cycles = analyze(trace, args.results / "server.log", args.mode)
            name = f"b{status['batch']}-c{status['context']}-rank{rank}"
            (args.output / f"{name}.json").write_text(json.dumps(report, indent=2))
            row = dict(
                batch=status["batch"],
                context=status["context"],
                rank=rank,
                cycles=len(report["cycles"]),
                **report["median"],
                maximum_forbidden_ms=report["maximum_forbidden_ms"],
            )
            rows.append(row)
            if rank == 0 and status["batch"] in {4, 16, 64}:
                plot(cycles, args.output / f"{name}.png", args.mode)
            print(json.dumps(row), flush=True)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
