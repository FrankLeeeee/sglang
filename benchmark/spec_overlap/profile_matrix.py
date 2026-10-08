"""Run the requested GPT-OSS speculative configurations, with owned servers."""

import argparse
import subprocess
import sys
from pathlib import Path

CONFIGS = ("3-1-4", "4-1-5", "5-2-6", "7-3-8", "10-4-11")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=CONFIGS)
    args = parser.parse_args()
    for config in args.configs:
        directory = args.results / f"baseline-{config}"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "matrix.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("run_matrix.py")),
                    "--models",
                    "gpt-oss",
                    "--modes",
                    "off",
                    "--batch-sizes",
                    "1",
                    "8",
                    "16",
                    "32",
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
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )


if __name__ == "__main__":
    main()
