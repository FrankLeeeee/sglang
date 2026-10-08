"""Reproducible fixed-length throughput probe and optional torch trace capture."""

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


def post(url, route, payload):
    request = urllib.request.Request(
        url + route,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = response.read().decode()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:31080")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
    for _ in range(2):
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
