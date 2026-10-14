"""Fixed-concurrency/context sweeps with explicit capacity and support records."""

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from run_workload import warm_phase_mapping

from sglang.srt.environ import envs


def read_info(url):
    with urllib.request.urlopen(url + "/get_server_info", timeout=10) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["no_spec", "spec", "launch", "phase", "phase_serial"],
        required=True,
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64, 128, 256]
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=[4096, 16384, 32768, 65536, 131072],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--port", type=int, default=31080)
    parser.add_argument("--tbo", action="store_true")
    parser.add_argument("--compute-chunks", type=int, default=4)
    parser.add_argument("--sanity-eval", action="store_true")
    parser.add_argument("--allow-draft-context-extension", action="store_true")
    parser.add_argument("--extra-server-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    command = [sys.executable]
    if args.mode in {"phase", "phase_serial"}:
        command += [str(Path(__file__).with_name("launch_profiled_server.py"))]
    else:
        command += ["-m", "sglang.launch_server"]
    command += [
        "--model-path",
        "openai/gpt-oss-20b",
        "--tp-size",
        "2",
        "--ep-size",
        "2",
        "--attention-backend",
        "fa3",
        "--context-length",
        str(max(args.contexts)),
        "--port",
        str(args.port),
        "--mem-fraction-static",
        "0.8",
        "--max-running-requests",
        "256",
        "--swa-full-tokens-ratio",
        "0.1",
        "--chunked-prefill-size",
        "8192",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--disable-custom-all-reduce",
        "--enforce-disable-flashinfer-allreduce-fusion",
        "--disable-overlap-schedule",
        "--return-output-ids",
        "--random-seed",
        "42",
        "--cuda-graph-bs-decode",
        "1",
        "2",
        "4",
        "8",
        "16",
        "32",
        "64",
        "128",
        "256",
    ]
    if args.mode != "no_spec":
        command += [
            "--speculative-draft-model-path",
            "zhuyksir/EAGLE3-gpt-oss-20b-bf16",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
        ]
    if args.mode in {"launch", "phase", "phase_serial"}:
        command += [
            "--speculative-microbatch-mode",
            "fine_overlap",
            "--speculative-microbatch-graph-chunks",
            "1",
            "--speculative-microbatch-release-chunk",
            "0",
            "--speculative-microbatch-schedule",
            args.mode,
            "--speculative-microbatch-compute-chunks",
            str(args.compute_chunks),
        ]
    if args.tbo:
        command.append("--enable-two-batch-overlap")
    command += args.extra_server_args
    (args.results / "command.json").write_text(json.dumps(command, indent=2))
    (args.results / "experiment-options.json").write_text(
        json.dumps(
            {
                "allow_draft_context_extension": args.allow_draft_context_extension,
            },
            indent=2,
        )
    )
    sources = [
        Path(__file__),
        Path(__file__).with_name("run_workload.py"),
        Path("python/sglang/srt/speculative/microbatch_overlap.py"),
        Path("python/sglang/srt/speculative/graph_interleave.py"),
        Path(
            "python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py"
        ),
    ]
    (args.results / "source-hashes.json").write_text(
        json.dumps(
            {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources
            },
            indent=2,
        )
    )
    entries = []

    def record(entry):
        entries.append(entry)
        (args.results / "status.json").write_text(json.dumps(entries, indent=2))
        print(json.dumps(entry), flush=True)

    process = None
    try:
        with (args.results / "server.log").open("a") as log:
            log.write(f"\nMATRIX_SERVER_START {time.time()}\n")
            log.flush()
            with contextlib.ExitStack() as stack:
                stack.enter_context(envs.SGLANG_DEEPGEMM_PDL.override(False))
                if args.allow_draft_context_extension:
                    stack.enter_context(
                        envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(True)
                    )
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            url = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    for context in args.contexts:
                        for batch in args.batch_sizes:
                            record(
                                dict(
                                    batch=batch,
                                    context=context,
                                    status="startup_failed",
                                    log="server.log",
                                )
                            )
                    return
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Server startup exceeded 900s")
            info = read_info(url)
            (args.results / "server-info.json").write_text(json.dumps(info, indent=2))
            capacities = [
                state["memory_usage"]["token_capacity"]
                for state in info["internal_states"]
            ]
            capacity = min(capacities)
            if args.mode in {"phase", "phase_serial"}:
                warm_phase_mapping(url, args.results / "phase-map-warmup.json")
            if args.sanity_eval:
                with (args.results / "gsm8k.log").open("w") as output:
                    subprocess.run(
                        [
                            "sgl-eval",
                            "run",
                            "gsm8k",
                            "--base-url",
                            url + "/v1",
                            "--model",
                            "openai/gpt-oss-20b",
                            "--num-examples",
                            "20",
                            "--num-threads",
                            "4",
                            "--max-tokens",
                            "1024",
                            "--reasoning-effort",
                            "low",
                            "--temperature",
                            "0",
                            "--out-dir",
                            str((args.results / "gsm8k").resolve()),
                        ],
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                metrics = json.loads(
                    next((args.results / "gsm8k").glob("*/metrics.json")).read_text()
                )
                if metrics["aggregate"]["score"] <= 0.8:
                    raise RuntimeError("GSM8K sanity score <= 0.8")
            for context in args.contexts:
                for batch in args.batch_sizes:
                    if (args.results / "STOP").exists():
                        print(
                            "Stopping between cases; existing outputs can be resumed",
                            flush=True,
                        )
                        return
                    entry = dict(batch=batch, context=context, capacity=capacity)
                    if context > 131072:
                        record(dict(entry, status="unsupported_native_context"))
                        continue
                    # Require the entire fixed batch to remain resident, including
                    # allocator rounding/speculation; do not silently queue it.
                    required = batch * (context + 64)
                    if required > capacity * 0.98:
                        record(
                            dict(
                                entry,
                                status="exceeds_fixed_batch_capacity",
                                required_tokens=required,
                            )
                        )
                        continue
                    output = args.results / f"b{batch}-c{context}.json"
                    profile = output.parent / (output.stem + "-profile")
                    if output.exists() and all(
                        list(profile.glob(f"*TP-{rank}*.trace.json.gz"))
                        for rank in range(2)
                    ):
                        record(dict(entry, status="existing", output=output.name))
                        continue
                    probe = [
                        sys.executable,
                        str(Path(__file__).with_name("run_workload.py")),
                        "--url",
                        url,
                        "--output",
                        str(output),
                        "--batch-size",
                        str(batch),
                        "--context-tokens",
                        str(context),
                        "--output-len",
                        str(args.output_len),
                        "--repeats",
                        str(args.repeats),
                        "--profile",
                    ]
                    started = time.monotonic()
                    with (args.results / f"b{batch}-c{context}.log").open(
                        "w"
                    ) as output_log:
                        result = subprocess.run(
                            probe,
                            stdout=output_log,
                            stderr=subprocess.STDOUT,
                            timeout=1800,
                        )
                    record(
                        dict(
                            entry,
                            status="ok" if result.returncode == 0 else "failed",
                            output=output.name,
                            elapsed=time.monotonic() - started,
                        )
                    )
                    if process.poll() is not None:
                        raise RuntimeError("Server exited during workload")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
