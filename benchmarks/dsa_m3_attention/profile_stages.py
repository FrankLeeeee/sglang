"""Diagnostic CUDA kernel attribution; separately measured from latency sweep."""

import argparse
import gc
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

# Initialize CUPTI before SGLang/sgl-kernel loads.
with profile(activities=[ProfilerActivity.CUDA]):
    torch.zeros(1, device="cuda")

from implementations import DSACase, M3Case, init_runtime


def classify(name):
    if "merge_topk_attn" in name:
        return "sparse_attention"
    if any(x in name.lower() for x in ("topk", "top_k", "radixselect", "radix_select")):
        return "selection"
    if any(
        x in name
        for x in (
            "mqa_logits",
            "_flash_attn_fwd_with_block_score",
            "_decode_score",
            "fp8_mqa",
        )
    ):
        return "index_scoring"
    if any(x in name.lower() for x in ("sparse", "flash_fwd", "flashmla", "flash_mla")):
        return "sparse_attention"
    return "projection_cache_other"


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", choices=["dsa", "m3"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.architecture == "dsa":
        init_runtime()
    rows = []
    for ci in (1, 2):
        for phase in ("prefill", "decode"):
            for length in (16384, 131072):
                for batch in (1, 32):
                    torch.manual_seed(42)
                    c = (DSACase if a.architecture == "dsa" else M3Case)(
                        ci,
                        batch,
                        length,
                        2048 // batch if phase == "prefill" else 1,
                        True,
                        phase,
                    )
                    for _ in range(3):
                        c.run()
                    torch.cuda.synchronize()
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                    ) as prof:
                        for _ in range(3):
                            with record_function("main_projection_cache"):
                                projected = c.project()
                            if a.architecture == "dsa":
                                q, qa = projected
                                with record_function("dsa_indexer_total"):
                                    idx = c.select(qa)
                                with record_function("main_sparse_attention"):
                                    c.attend(q, idx)
                            else:
                                with record_function("m3_index_projection_cache"):
                                    qi = c.index_project()
                                with record_function("m3_native_sparse_pipeline"):
                                    c.attend(projected, qi)
                        torch.cuda.synchronize()
                    scopes = {
                        e.key: e.device_time_total / 3000
                        for e in prof.key_averages()
                        if e.key
                        in (
                            "main_projection_cache",
                            "dsa_indexer_total",
                            "main_sparse_attention",
                            "m3_index_projection_cache",
                            "m3_native_sparse_pipeline",
                        )
                    }
                    kernels = {}
                    for e in prof.events():
                        if (
                            e.device_type == torch.autograd.DeviceType.CUDA
                            and e.name not in scopes
                        ):
                            kernels[e.name] = (
                                kernels.get(e.name, 0) + e.device_time_total / 3000
                            )
                    assert kernels, "CUPTI returned no CUDA events"
                    stages = {}
                    for name, ms in kernels.items():
                        cl = classify(name)
                        stages[cl] = stages.get(cl, 0) + ms
                    rows.append(
                        dict(
                            architecture=a.architecture,
                            config=ci,
                            phase=phase,
                            context=length,
                            batch=batch,
                            scope_ms=scopes,
                            kernel_stage_ms=stages,
                            kernels_ms=kernels,
                            warmup=3,
                            profile_iterations=3,
                            timing_note="Eager profiler CUDA kernel durations; excludes launch gaps; diagnostic only, not latency benchmark.",
                        )
                    )
                    a.output.parent.mkdir(parents=True, exist_ok=True)
                    a.output.write_text(json.dumps(rows, indent=2) + "\n")
                    print(a.architecture, ci, phase, length, batch, scopes, flush=True)
                    del c, prof, projected
                    gc.collect()
                    torch.cuda.empty_cache()
    if a.architecture == "dsa":
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
