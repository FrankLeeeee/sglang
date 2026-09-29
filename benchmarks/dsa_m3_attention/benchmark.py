"""Matched dense/native sparse sweep, including projection and learned selection."""

import argparse
import gc

# Import the existing measurement contract without a conflicting benchmark name.
import importlib.util
import json
import platform
import subprocess
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import flashinfer
import torch
import triton
from implementations import DSACase, M3Case, init_runtime

spec = importlib.util.spec_from_file_location(
    "gqa_timing",
    Path(__file__).resolve().parents[1] / "gqa_lightning_attention/benchmark.py",
)
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def ints(s):
    return list(map(int, s.split(",")))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", choices=["dsa", "m3"], required=True)
    p.add_argument("--configs", type=ints, default=[1, 2])
    p.add_argument("--contexts", type=ints, default=[16384, 32768, 65536, 131072])
    p.add_argument("--batches", type=ints, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--phases", default="prefill,decode")
    p.add_argument("--variants", default="dense_fa2,dense_fa3,sparse")
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--prefill-budget", type=int, default=2048)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stages", action="store_true")
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.warmup < 3:
        p.error("Every measurement requires at least 3 warmup forwards")
    if a.iterations < 1:
        p.error("iterations must be positive")
    if a.architecture == "dsa":
        if a.topk != 2048:
            p.error("Native DSA fused prefill top-k requires 2048")
        init_runtime()
    metadata = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        architecture=a.architecture,
        gpu=torch.cuda.get_device_name(),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        flashinfer=flashinfer.__version__,
        triton=triton.__version__,
        python=platform.python_version(),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        scope="Synthetic projected attention adapter; projections, norms, cache writes, native indexer and selection, attention. No output projection, model MLP, communication, planning, compilation, allocation, or scheduler.",
        prefill="Final chunk, 2048 total query tokens (unless overridden); eager. Not TTFT.",
        decode="One query per request; CUDA graph replay; L2 flush outside each sample.",
        architecture_details=(
            "Absorbed MLA 512+64, BF16 KV, 64Q/1KV 128-dim FP8 production indexer; 64-token pages; 48 main heads padded to 64 for FlashMLA; dense FlashInfer MLA FA2 and FA3."
            if a.architecture == "dsa"
            else "128-dim GQA 48/4 or 64/8; one 128-dim index Q per main KV group, shared index K; 16 blocks x128; native M3 max pooling/local block; dense FlashInfer paged FA2 and FA3 (tensor-core decode)."
        ),
    )
    rows = (
        json.loads(a.output.read_text())["rows"]
        if a.resume and a.output.exists()
        else []
    )
    done = {
        (r["config"], r["phase"], r["context"], r["batch"], r["variant"]) for r in rows
    }
    cls = DSACase if a.architecture == "dsa" else M3Case
    for ci in a.configs:
        for phase in a.phases.split(","):
            for ctx in a.contexts:
                for b in a.batches:
                    for variant in a.variants.split(","):
                        key = (ci, phase, ctx, b, variant)
                        if key in done:
                            continue
                        qpr = 1 if phase == "decode" else a.prefill_budget // b
                        row = dict(
                            config=ci,
                            phase=phase,
                            context=ctx,
                            batch=b,
                            variant=variant,
                            architecture=a.architecture,
                            query_tokens=b * qpr,
                            topk=a.topk if variant == "sparse" else 0,
                            cuda_graph=phase == "decode",
                            status="ok",
                        )
                        case = None
                        try:
                            torch.manual_seed(42)
                            case = cls(
                                ci,
                                b,
                                ctx,
                                qpr,
                                variant == "sparse",
                                phase,
                                a.topk,
                                dense_backend=(
                                    variant.removeprefix("dense_")
                                    if variant.startswith("dense_")
                                    else "fa3"
                                ),
                            )
                            torch.cuda.reset_peak_memory_stats()
                            out = case.run()
                            torch.cuda.synchronize()
                            assert torch.isfinite(out).all(), "Nonfinite output"
                            del out
                            row.update(
                                timing.measure(
                                    case.run, phase == "decode", a.warmup, a.iterations
                                )
                            )
                            row["peak_allocated_gib"] = (
                                torch.cuda.max_memory_allocated() / 1024**3
                            )
                            if a.stages:
                                if a.architecture == "dsa":
                                    q, qa = case.project()
                                    idx = case.select(qa) if case.sparse else None
                                    fs = {
                                        "project_cache": case.project,
                                        "attention_only": partial(case.attend, q, idx),
                                    }
                                    if case.sparse:
                                        fs["indexer_total"] = partial(case.select, qa)
                                else:
                                    q = case.project()
                                    qi = case.index_project() if case.sparse else None
                                    fs = {
                                        "project_cache": case.project,
                                        (
                                            "sparse_pipeline"
                                            if case.sparse
                                            else "attention_only"
                                        ): partial(case.attend, q, qi),
                                    }
                                    if case.sparse:
                                        fs["index_projection_cache"] = (
                                            case.index_project
                                        )
                                row["stages"] = {
                                    k: timing.measure(
                                        fn, phase == "decode", a.warmup, a.iterations
                                    )
                                    for k, fn in fs.items()
                                }
                                del fs, q
                        except torch.OutOfMemoryError as e:
                            row.update(status="oom", error=str(e))
                        rows.append(row)
                        a.output.write_text(
                            json.dumps(dict(metadata=metadata, rows=rows), indent=2)
                            + "\n"
                        )
                        print(
                            a.architecture,
                            key,
                            row.get("median_ms", row["status"]),
                            flush=True,
                        )
                        del case
                        gc.collect()
                        torch.cuda.empty_cache()
    if a.architecture == "dsa":
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
