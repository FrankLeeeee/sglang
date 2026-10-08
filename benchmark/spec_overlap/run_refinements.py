"""Ten explicit schedule experiments; each directory preserves its source hash."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Each round changes one schedule dimension, or tests a wider proposal tree.
ROUNDS = [
    ("01-reference", "3-1-4", 8, 1, 50),
    ("02-tree", "5-2-6", 8, 1, 50),
    ("03-four-cuts", "3-1-4", 4, 0, 50),
    ("04-two-cuts", "3-1-4", 2, 0, 50),
    ("05-whole-target", "3-1-4", 1, -1, 50),
    ("06-immediate", "3-1-4", 2, -1, 50),
    ("07-small-first", "3-1-4", 2, 0, 25),
    ("08-large-first", "3-1-4", 2, 0, 75),
    ("09-late-release", "3-1-4", 2, 1, 50),
    ("10-single-cut-release", "3-1-4", 1, 0, 50),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--rounds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    for index in args.rounds:
        name, config, cuts, release, split = ROUNDS[index - 1]
        # Later experiments refine the measured best schedule, rather than
        # assuming the two-cut variant won. Keep tree and chain costs separate.
        if index >= 7:
            candidates = []
            for experiment in args.results.glob("*/experiment.json"):
                previous = json.loads(experiment.read_text())
                measured = experiment.parent / "gpt-oss-fine_overlap-b16.json"
                if (
                    previous["spec_config"] == config
                    and previous.get("deterministic", False) == args.deterministic
                    and not list(experiment.parent.glob("*-error.json"))
                    and (experiment.parent / "gpt-oss-fine_serial-b16.json").exists()
                    and measured.exists()
                ):
                    data = json.loads(measured.read_text())
                    candidates.append((data["median_tokens_per_second"], previous))
            if candidates:
                _, best = max(candidates, key=lambda entry: entry[0])
                cuts, release = best["chunks"], best["release_chunk"]
                if index == 9:
                    release = cuts - 1 if release != cuts - 1 else -1
                if index == 10:
                    cuts, release, split = 1, 0, best["split_percent"]
        directory = args.results / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "experiment.json").write_text(
            json.dumps(
                {
                    "round": index,
                    "name": name,
                    "spec_config": config,
                    "chunks": cuts,
                    "release_chunk": release,
                    "split_percent": split,
                    "deterministic": args.deterministic,
                },
                indent=2,
            )
        )
        modes = ["off", "fine_serial", "fine_overlap"]
        baseline = args.results.parent / f"baseline-{config}"
        if not args.deterministic and (baseline / "gpt-oss-off-b16.json").exists():
            modes.remove("off")
            for source in baseline.glob("gpt-oss-off*"):
                if source.is_file():
                    shutil.copy2(source, directory / source.name)
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_matrix.py")),
            "--models",
            "gpt-oss",
            "--modes",
            *modes,
            "--batch-sizes",
            "16",
            "--spec-config",
            config,
            "--decode-cuda-graph",
            "full",
            "--output-len",
            "128",
            "--repeats",
            "3",
            "--results",
            str(directory),
        ]
        if args.deterministic:
            command.append("--deterministic")
        # argparse.REMAINDER allows flags to be passed verbatim.
        command += [
            "--extra-server-args",
            "--speculative-microbatch-graph-chunks",
            str(cuts),
            "--speculative-microbatch-release-chunk",
            str(release),
            "--speculative-microbatch-split-percent",
            str(split),
        ]
        print(f"Starting {name}", flush=True)
        with (directory / "matrix.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
