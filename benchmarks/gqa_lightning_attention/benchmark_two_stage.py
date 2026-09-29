"""Full-grid comparison of dense, original Lightning, and two-stage Lightning."""

import argparse
import csv
import gc
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import flashinfer
import torch
import triton
from attention import CONFIGS, Attention, PagedState

from benchmark import measure


def ints(s):
    return list(map(int, s.split(",")))


def config_for(experiment, ci):
    cfg = CONFIGS[ci]
    if experiment >= 2:
        cfg = replace(cfg, indexer_q_heads=cfg.num_k_heads, indexer_kv_heads=1)
    if experiment == 3:
        cfg = replace(cfg, num_k_heads=1)
    return cfg


def use_fa3(state):
    c = state.cfg
    p = state.page_size
    b = state.batch
    pages = (state.req_to_token[:, ::p] // p).flatten().contiguous()
    indptr = torch.arange(b + 1, device="cuda", dtype=torch.int32) * (state.length // p)
    last = torch.full((b,), p, device="cuda", dtype=torch.int32)
    if state.phase == "decode":
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            state.workspace,
            use_cuda_graph=True,
            use_tensor_cores=True,
            backend="fa3",
            paged_kv_indptr_buffer=indptr,
            paged_kv_indices_buffer=pages,
            paged_kv_last_page_len_buffer=last,
        )
        wrapper.plan(
            indptr,
            pages,
            last,
            c.num_q_heads,
            c.num_k_heads,
            c.head_dim,
            p,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
    else:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            state.workspace, backend="fa3"
        )
        cuq = (
            torch.arange(b + 1, device="cuda", dtype=torch.int32) * state.q_per_request
        )
        wrapper.plan(
            cuq,
            indptr,
            pages,
            last,
            c.num_q_heads,
            c.num_k_heads,
            c.head_dim,
            p,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
    state.wrapper = wrapper


def build_case(experiment, ci, batch, context, qpr, phase, variant, candidates=64):
    cfg = config_for(experiment, ci)
    sparse = variant in ("sparse", "two_stage")
    two_stage = variant == "two_stage"
    torch.manual_seed(42)
    layer = Attention(
        cfg, sparse=sparse, two_stage=two_stage, candidate_blocks=candidates
    )
    torch.manual_seed(43)
    state = PagedState(
        cfg, batch, context, qpr, sparse=sparse, phase=phase, two_stage=two_stage
    )
    if variant == "dense_fa3":
        use_fa3(state)
    torch.manual_seed(44)
    x = torch.randn(batch * qpr, 8192, device="cuda", dtype=torch.bfloat16)
    return layer, state, x


def save(path, metadata, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(metadata=metadata, rows=rows), indent=2) + "\n")
    temp.replace(path)
    fields = sorted({k for r in rows for k in r if k != "samples_ms"})
    with path.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiments", type=ints, default=[1, 2, 3])
    p.add_argument("--configs", type=ints, default=[1, 2])
    p.add_argument("--contexts", type=ints, default=[16384, 32768, 65536, 131072])
    p.add_argument("--batches", type=ints, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--phases", default="prefill,decode")
    p.add_argument("--variants", default="dense_fa2,dense_fa3,sparse,two_stage")
    p.add_argument("--candidate-blocks", type=int, default=64)
    p.add_argument("--prefill-budget", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results/two_stage"
    )
    a = p.parse_args()
    if a.warmup < 3 or a.iterations < 1:
        p.error("Require >=3 warmups and >=1 timed sample")
    if a.candidate_blocks < 16:
        p.error("Need at least 16 candidate blocks for top-2048")
    for experiment in a.experiments:
        if experiment not in (1, 2, 3):
            p.error("Experiment must be 1,2,3")
        for ci in a.configs:
            path = a.output / f"experiment{experiment}" / f"config{ci}.json"
            metadata = dict(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                gpu=torch.cuda.get_device_name(),
                gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                triton=triton.__version__,
                flashinfer=flashinfer.__version__,
                python=platform.python_version(),
                git_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                source_sha256={
                    n: hashlib.sha256(
                        (Path(__file__).parent / n).read_bytes()
                    ).hexdigest()
                    for n in (
                        "attention.py",
                        "kernels.py",
                        "two_stage.py",
                        "benchmark.py",
                        "benchmark_two_stage.py",
                    )
                },
                arguments={
                    k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
                },
                scope="Same main projections/RMSNorm/paged cache as experiments 1-3. Full indexer and selection included. Two-stage additionally updates cached block means, scores blocks, selects candidate blocks, scores their tokens, selects final tokens, and converts indices. Cache initialization/planning/JIT/capture excluded.",
                two_stage="Mean-pooled 128-token keys; 64 candidate blocks by default; force current block; exact top-2048 within candidates. Approximate relative to global token top-k.",
                timing="Final prefill chunk 2048 total tokens eager; decode one token/request CUDA graph; L2 flush outside timing; >=3 warmups.",
            )
            rows = (
                json.loads(path.read_text())["rows"]
                if a.resume and path.exists()
                else []
            )
            done = {(r["phase"], r["context"], r["batch"], r["variant"]) for r in rows}
            for phase in a.phases.split(","):
                if phase not in ("prefill", "decode"):
                    p.error("Invalid phase")
                for context in a.contexts:
                    for batch in a.batches:
                        qpr = 1 if phase == "decode" else a.prefill_budget // batch
                        for variant in a.variants.split(","):
                            if variant not in (
                                "dense_fa2",
                                "dense_fa3",
                                "sparse",
                                "two_stage",
                            ):
                                p.error("Invalid variant")
                            if (phase, context, batch, variant) in done:
                                continue
                            row = dict(
                                experiment=experiment,
                                config=ci,
                                phase=phase,
                                context=context,
                                batch=batch,
                                variant=variant,
                                **asdict(config_for(experiment, ci)),
                                candidate_blocks=(
                                    a.candidate_blocks if variant == "two_stage" else 0
                                ),
                                top_k=2048 if variant in ("sparse", "two_stage") else 0,
                                query_tokens=qpr * batch,
                                cuda_graph=phase == "decode",
                                status="ok",
                            )
                            layer = state = x = None
                            try:
                                layer, state, x = build_case(
                                    experiment,
                                    ci,
                                    batch,
                                    context,
                                    qpr,
                                    phase,
                                    variant,
                                    a.candidate_blocks,
                                )
                                out = layer.forward(x, state)
                                torch.cuda.synchronize()
                                assert torch.isfinite(out).all()
                                del out
                                torch.cuda.reset_peak_memory_stats()
                                row.update(
                                    measure(
                                        partial(layer.forward, x, state),
                                        phase == "decode",
                                        a.warmup,
                                        a.iterations,
                                    )
                                )
                                row["peak_allocated_gib"] = (
                                    torch.cuda.max_memory_allocated() / 1024**3
                                )
                            except torch.OutOfMemoryError as e:
                                row.update(status="oom", error=str(e))
                            rows.append(row)
                            save(path, metadata, rows)
                            print(
                                experiment,
                                ci,
                                phase,
                                context,
                                batch,
                                variant,
                                row.get("median_ms", row["status"]),
                                flush=True,
                            )
                            del layer, state, x
                            gc.collect()
                            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
