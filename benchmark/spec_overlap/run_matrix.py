"""Own server lifecycle for original, split-serial, and split-overlap trials."""

import argparse
import gzip
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PAIRS = {
    "llama": (
        "meta-llama/Llama-3.1-8B-Instruct",
        "lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge",
    ),
    "gpt-oss": ("openai/gpt-oss-20b", "zhuyksir/EAGLE3-gpt-oss-20b-bf16"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=PAIRS, default=list(PAIRS))
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["off", "serial", "overlap", "fine_serial", "fine_overlap"],
        default=["off", "serial", "overlap"],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--port", type=int, default=31080)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--decode-cuda-graph", choices=["disabled", "full"], default="disabled"
    )
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).parent / "results"
    )
    args = parser.parse_args()
    if args.decode_cuda_graph != "disabled" and any(
        mode not in {"off", "fine_serial", "fine_overlap"} for mode in args.modes
    ):
        parser.error("Full decode graphs support off/fine_serial/fine_overlap only")
    if args.deterministic and "off" not in args.modes:
        for model in args.models:
            for batch_size in args.batch_sizes:
                baseline = args.results / f"{model}-off-b{batch_size}.json"
                if not baseline.exists():
                    parser.error(
                        f"Exact token checks require {baseline}; include --modes off first"
                    )
    args.results.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        target, draft = PAIRS[model]
        for mode in args.modes:
            name = f"{model}-{mode}"
            log = args.results / f"{name}-server.log"
            source_files = [
                "python/sglang/srt/speculative/microbatch_overlap.py",
                "python/sglang/srt/speculative/graph_chunks.py",
                "python/sglang/srt/speculative/eagle_worker_v2.py",
                "python/sglang/srt/model_executor/input_buffers.py",
                "python/sglang/srt/model_executor/graph_shared_output.py",
                "python/sglang/srt/model_executor/forward_context.py",
                "python/sglang/srt/runtime_context.py",
                "python/sglang/srt/arg_groups/fields/spec.py",
                "python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py",
                "python/sglang/srt/models/llama.py",
                "python/sglang/srt/models/gpt_oss.py",
                "benchmark/spec_overlap/run_matrix.py",
                "benchmark/spec_overlap/run_workload.py",
            ]
            (args.results / f"{name}-sources.json").write_text(
                json.dumps(
                    {
                        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                        for path in source_files
                    },
                    indent=2,
                )
            )
            command = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                target,
                "--speculative-draft-model-path",
                draft,
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--speculative-microbatch-mode",
                mode,
                "--tp-size",
                "2",
                "--context-length",
                "2048",
                "--port",
                str(args.port),
                "--mem-fraction-static",
                "0.55",
                "--cuda-graph-backend-decode",
                args.decode_cuda_graph,
                "--cuda-graph-backend-prefill",
                "disabled",
                "--disable-custom-all-reduce",
                "--enforce-disable-flashinfer-allreduce-fusion",
                "--disable-overlap-schedule",
                "--return-output-ids",
                "--random-seed",
                "42",
            ]
            if args.decode_cuda_graph == "full":
                command.extend(
                    ["--cuda-graph-bs-decode", "1", "2", "4", "8", "16", "32"]
                )
            if args.deterministic:
                command.append("--enable-deterministic-inference")
            env = dict(os.environ, SGLANG_DEEPGEMM_PDL="0")
            if args.deterministic:
                env["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"
            print(f"Launching {name}; log={log}", flush=True)
            process = None
            try:
                with log.open("w") as stream:
                    process = subprocess.Popen(
                        command,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        env=env,
                        start_new_session=True,
                    )
                    url = f"http://127.0.0.1:{args.port}"
                    deadline = time.monotonic() + 900
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError(f"{name} exited; inspect {log}")
                        try:
                            with urllib.request.urlopen(
                                url + "/health", timeout=2
                            ) as response:
                                if response.status == 200:
                                    break
                        except OSError:
                            pass
                        time.sleep(1)
                    else:
                        raise TimeoutError(f"{name} did not become ready")
                    for batch_size in args.batch_sizes:
                        output = args.results / f"{name}-b{batch_size}.json"
                        probe = [
                            sys.executable,
                            str(Path(__file__).with_name("run_workload.py")),
                            "--url",
                            url,
                            "--output",
                            str(output),
                            "--batch-size",
                            str(batch_size),
                            "--output-len",
                            str(args.output_len),
                            "--repeats",
                            str(args.repeats),
                            "--profile",
                        ]
                        subprocess.run(probe, check=True)
                        if args.deterministic:
                            measured = json.loads(output.read_text())
                            baseline_path = (
                                args.results / f"{model}-off-b{batch_size}.json"
                            )
                            if baseline_path.exists():
                                baseline = json.loads(baseline_path.read_text())
                                expected = [
                                    o["output_ids"] for o in baseline["outputs"]
                                ]
                                if any(
                                    ids != expected
                                    for ids in measured["repeat_outputs"]
                                ):
                                    raise RuntimeError(
                                        f"{name}, batch {batch_size} failed exact token checks; "
                                        "do not interpret its speedup"
                                    )
                    # Let the profiler complete its export before stopping workers.
                    deadline = time.monotonic() + 120
                    while time.monotonic() < deadline:
                        if all(
                            list(
                                (args.results / f"{name}-b{bs}-profile").glob(
                                    "*.trace.json.gz"
                                )
                            )
                            for bs in args.batch_sizes
                        ):
                            break
                        time.sleep(1)
                    if mode != "off":
                        for bs in args.batch_sizes:
                            traces = sorted(
                                (args.results / f"{name}-b{bs}-profile").glob(
                                    "*TP-0.trace.json.gz"
                                )
                            )
                            if not traces:
                                raise RuntimeError(
                                    f"No trace exported for {name}, batch {bs}"
                                )
                            with gzip.open(traces[-1], "rt") as trace:
                                events = json.load(trace)["traceEvents"]
                            if not any(
                                e.get("name") == "microbatch.verify" for e in events
                            ):
                                raise RuntimeError(
                                    f"{name}, batch {bs} fell back: no microbatch stage in trace"
                                )
                            if args.decode_cuda_graph == "full" and not any(
                                e.get("name") == "microbatch.target_chunk"
                                for e in events
                            ):
                                raise RuntimeError(
                                    f"{name}, batch {bs} fell back: no graph chunks in trace"
                                )
            except Exception as error:
                (args.results / f"{name}-error.json").write_text(
                    json.dumps({"error": str(error), "command": command}, indent=2)
                )
                print(f"FAILED {name}: {error}", flush=True)
                raise
            finally:
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    main()
