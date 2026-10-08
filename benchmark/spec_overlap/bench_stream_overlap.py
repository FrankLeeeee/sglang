"""Synthetic TP compute/communication probe; not a model throughput benchmark.

Run with torchrun --standalone --nproc-per-node=2. Private weights, activations,
and communicators isolate stream feasibility from SGLang metadata ownership.
"""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 2 or args.batch_size % 2 or args.repeats < 1:
        parser.error("batch-size must be positive and even; repeats must be positive")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    draft_group = dist.new_group(list(range(dist.get_world_size())))
    torch.manual_seed(42 + rank)
    h = args.hidden
    if h % dist.get_world_size():
        raise ValueError("hidden size must be divisible by the TP world size")
    shard = h // dist.get_world_size()

    def weights(layers):
        return [
            (
                torch.randn(h, shard, device="cuda", dtype=torch.bfloat16) / h**0.5,
                torch.randn(shard, h, device="cuda", dtype=torch.bfloat16) / h**0.5,
            )
            for _ in range(layers)
        ]

    target_weights, draft_weights = weights(args.layers), weights(2)
    inputs = [
        torch.randn(args.batch_size // 2, h, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    dist.broadcast(inputs[0], 0)
    dist.broadcast(inputs[1], 0)
    main_stream = torch.cuda.current_stream()
    draft_stream = torch.cuda.Stream()

    def forward(x, parameters, group=None):
        for up, down in parameters:
            x = torch.mm(torch.relu(torch.mm(x, up)), down)
            dist.all_reduce(x, group=group)
            x = torch.tanh(x)
        return x

    def run(mode):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        outputs = []
        if mode == "serial":
            for x in inputs:
                proposal = forward(x, draft_weights, draft_group)
                verified = forward(proposal, target_weights)
                outputs.append(forward(verified, draft_weights[:1], draft_group))
        else:
            draft_stream.wait_stream(main_stream)
            with torch.cuda.stream(draft_stream):
                first = forward(inputs[0], draft_weights, draft_group)
                first_ready = draft_stream.record_event()
            main_stream.wait_event(first_ready)
            first_verified = forward(first, target_weights)
            first_verified_ready = main_stream.record_event()
            with torch.cuda.stream(draft_stream):
                second = forward(inputs[1], draft_weights, draft_group)
                second_ready = draft_stream.record_event()
                draft_stream.wait_event(first_verified_ready)
                outputs.append(forward(first_verified, draft_weights[:1], draft_group))
            main_stream.wait_event(second_ready)
            second_verified = forward(second, target_weights)
            second_verified_ready = main_stream.record_event()
            with torch.cuda.stream(draft_stream):
                draft_stream.wait_event(second_verified_ready)
                outputs.append(forward(second_verified, draft_weights[:1], draft_group))
                finished = draft_stream.record_event()
            main_stream.wait_event(finished)
        end.record()
        end.synchronize()
        return start.elapsed_time(end), outputs

    for _ in range(3):
        run("serial")
        run("overlap")
    reference = run("serial")[1]
    times = {"serial": [], "overlap": []}
    exact = True
    for _ in range(args.repeats):
        for mode in times:
            milliseconds, outputs = run(mode)
            # Report the slowest rank, with this reduction outside the timer.
            elapsed = torch.tensor(milliseconds, device="cuda")
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            times[mode].append(elapsed.item())
            exact &= all(
                torch.equal(a, b) for a, b in zip(reference, outputs, strict=True)
            )
    medians = {mode: statistics.median(values) for mode, values in times.items()}
    valid = torch.tensor(int(exact), device="cuda")
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if rank == 0:
        result = {
            "configuration": vars(args) | {"output": str(args.output)},
            "median_ms": medians,
            "samples_ms": times,
            "speedup": medians["serial"] / medians["overlap"],
            "exact_outputs_all_ranks": bool(valid.item()),
            "kind": "synthetic; no model weights, KV cache, or acceptance",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
