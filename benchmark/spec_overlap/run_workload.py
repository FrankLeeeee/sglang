"""Reproducible fixed-length throughput probe and optional torch trace capture."""

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def post(url, route, payload):
    request = urllib.request.Request(
        url + route,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            body = response.read().decode()
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return body
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"{route}: HTTP {error.code}: {error.read().decode()}"
        ) from error


def warm_phase_mapping(url, output):
    # Batch one takes the ordinary EAGLE path. Initialize pair mappings before
    # any torch profile claims CUPTI, even when the measured grid starts at one.
    result = post(
        url,
        "/generate",
        {
            "input_ids": [[1, 2, 3, 4], [1, 2, 3, 5]],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 8,
                "ignore_eos": True,
            },
        },
    )
    output.write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:31080")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=0,
        help="Total sequence budget: prompt tokens plus output tokens",
    )
    args = parser.parse_args()
    workload_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.url + "/get_server_info") as response:
        initial_server_info = json.load(response)
    # Reserve the four-token speculative tree plus two scheduler positions at
    # the hard context boundary. Use the same prompt in every compared mode,
    # including the non-speculative baseline.
    context_reserve_tokens = (
        6 if args.context_tokens == initial_server_info.get("context_length") else 0
    )
    prompts = [
        f"Problem {i}: Explain how a computer executes a program. Discuss memory, "
        "instructions, and arithmetic with examples. Give a detailed explanation."
        for i in range(args.batch_size)
    ]
    payload = {
        "text": prompts,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.output_len,
            "ignore_eos": True,
        },
    }
    if args.context_tokens:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        paragraph = tokenizer.encode(
            "A computer executes instructions by fetching them from memory, decoding "
            "the operation, and using registers and arithmetic units to calculate "
            "results. Caches reduce memory latency. Programs combine these simple "
            "steps into useful work. The following notes explain this process.\n",
            add_special_tokens=False,
        )
        input_len = args.context_tokens - args.output_len - context_reserve_tokens
        suffix = tokenizer.encode(
            "\nSummarize these notes in detail:", add_special_tokens=False
        )
        inputs = []
        for index in range(args.batch_size):
            prefix = tokenizer.encode(
                f"Document {index:06d}:\n", add_special_tokens=False
            )
            count = input_len - len(prefix) - len(suffix)
            if count < 0:
                raise ValueError("Context budget is too small")
            inputs.append(
                prefix
                + (paragraph * ((count + len(paragraph) - 1) // len(paragraph)))[:count]
                + suffix
            )
        payload.pop("text")
        payload["input_ids"] = inputs
        prompts = {
            "recipe": "unique document prefix + repeated computer paragraph + summary suffix",
            "input_tokens_per_request": input_len,
            "sha256": hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
        }
        post(args.url, "/flush_cache", {})
    cold_start = time.perf_counter()
    cold_outputs = post(args.url, "/generate", payload)
    cold_seconds = time.perf_counter() - cold_start
    post(args.url, "/generate", payload)
    runs = []
    repeat_outputs = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        outputs = post(args.url, "/generate", payload)
        seconds = time.perf_counter() - start
        if isinstance(outputs, dict):
            outputs = [outputs]
        repeat_outputs.append([o.get("output_ids", o["text"]) for o in outputs])
        tokens = sum(o["meta_info"]["completion_tokens"] for o in outputs)
        runs.append(
            {
                "seconds": seconds,
                "tokens": tokens,
                "tokens_per_second": tokens / seconds,
            }
        )
    trace_dir = args.output.parent / (args.output.stem + "-profile")
    with urllib.request.urlopen(args.url + "/get_server_info") as response:
        server_info = json.load(response)
    result = {
        "batch_size": args.batch_size,
        "context_tokens": args.context_tokens,
        "context_reserve_tokens": context_reserve_tokens,
        "workload_sha256": workload_sha256,
        "cold_seconds": cold_seconds,
        "cold_outputs": cold_outputs,
        "output_len": args.output_len,
        "prompts": prompts,
        "runs": runs,
        "median_tokens_per_second": statistics.median(
            r["tokens_per_second"] for r in runs
        ),
        "outputs": outputs,
        "repeat_outputs": repeat_outputs,
        "matching_requests_between_repeats": [
            sum(a == b for a, b in zip(repeat_outputs[0], other, strict=True))
            for other in repeat_outputs[1:]
        ],
        "server_info": server_info,
        "profile_dir": str(trace_dir) if args.profile else None,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "batch_size",
                    "output_len",
                    "median_tokens_per_second",
                    "profile_dir",
                )
            }
        )
    )
    if args.profile:
        post(
            args.url,
            "/start_profile",
            {
                "output_dir": str(trace_dir.resolve()),
                "num_steps": 8,
                "activities": ["CPU", "GPU"],
                "record_shapes": True,
                "with_stack": False,
            },
        )
        post(args.url, "/generate", payload)


if __name__ == "__main__":
    main()
