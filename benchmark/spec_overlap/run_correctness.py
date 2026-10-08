"""Deterministic token gates for the selected GPT-OSS overlap schedule."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from profile_matrix import CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=CONFIGS)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--release-chunk", type=int, default=0)
    parser.add_argument("--split-percent", type=int, default=50)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    for config in args.configs:
        if args.summarize_only:
            continue
        directory = args.results / config
        directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_matrix.py")),
            "--models",
            "gpt-oss",
            "--modes",
            "off",
            "fine_overlap",
            "--batch-sizes",
            "1",
            "5",
            "8",
            "16",
            "32",
            "--spec-config",
            config,
            "--deterministic",
            "--decode-cuda-graph",
            "full",
            "--output-len",
            "64",
            "--repeats",
            "2",
            "--results",
            str(directory),
        ]
        if config == "3-1-4":
            command.append("--sanity-eval")
        command += [
            "--extra-server-args",
            "--speculative-microbatch-graph-chunks",
            str(args.chunks),
            "--speculative-microbatch-release-chunk",
            str(args.release_chunk),
            "--speculative-microbatch-split-percent",
            str(args.split_percent),
        ]
        print(f"Checking {config}", flush=True)
        with (directory / "matrix.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = []
    for config in args.configs:
        for batch_size in (1, 5, 8, 16, 32):
            directory = args.results / config
            baseline = json.loads(
                (directory / f"gpt-oss-off-b{batch_size}.json").read_text()
            )
            overlap = json.loads(
                (directory / f"gpt-oss-fine_overlap-b{batch_size}.json").read_text()
            )
            expected = [o["output_ids"] for o in baseline["outputs"]]
            assert len(expected) == batch_size
            assert all(
                ids == expected
                for ids in baseline["repeat_outputs"] + overlap["repeat_outputs"]
            )
            for mode in ("off", "fine_overlap"):
                profile = directory / f"gpt-oss-{mode}-b{batch_size}-profile"
                for rank in range(2):
                    assert list(profile.glob(f"*TP-{rank}.trace.json.gz")), (
                        f"Missing trace: {config}/{mode}/{batch_size}/TP-{rank}"
                    )
            rows.append(
                {
                    "config": config,
                    "batch_size": batch_size,
                    "requests_per_repeat": len(expected),
                    "repeat_count": len(overlap["repeat_outputs"]),
                    "exact_match": True,
                    "batch_one_fallback": batch_size == 1,
                }
            )
    (args.results / "correctness.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Passed {len(rows)} deterministic cases", flush=True)


if __name__ == "__main__":
    main()
