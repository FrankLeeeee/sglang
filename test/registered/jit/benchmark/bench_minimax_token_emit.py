"""Benchmark the CUDA-emit fused token selector against the materializing path.

The interesting comparison is end-to-end selection latency: the fused path
(Triton pool-max + CUDA ballot emitter + candidate trim) against
``index_score.token_select_prefill`` which materializes the full score matrix.
The full sweep lives in benchmarks/minimax_m3_sparse_attn/bench_token_select.py;
this keeps a CI-sized point so a regression in the emit kernel shows up.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")

NUM_HEADS = 4
HEAD_DIM = 128
TOPK = 2048


def _build(batch_size: int, context_len: int, chunk_len: int):
    torch.manual_seed(0)
    device = "cuda"
    prefix = context_len - chunk_len
    total_q = batch_size * chunk_len
    max_slots = (batch_size + 1) * context_len
    idx_q = torch.randn(total_q, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    idx_k_cache = torch.randn(max_slots, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    req_to_token = torch.zeros(batch_size + 1, context_len, dtype=torch.int32, device=device)
    for b in range(batch_size):
        req_to_token[b] = torch.randperm(max_slots, device=device)[:context_len].to(torch.int32)
    return dict(
        idx_q=idx_q,
        idx_k_cache=idx_k_cache,
        req_to_token=req_to_token,
        slot_ids=torch.arange(batch_size, dtype=torch.int64, device=device),
        cu_seqlens=torch.arange(
            0, (batch_size + 1) * chunk_len, chunk_len, dtype=torch.int32, device=device
        ),
        seq_lens=torch.full((batch_size,), context_len, dtype=torch.int32, device=device),
        prefix_lens=torch.full((batch_size,), prefix, dtype=torch.int32, device=device),
        max_seqlen_q=chunk_len,
        max_seqlen_k=context_len,
        topk=TOPK,
        init_tokens=0,
        local_tokens=128,
        seqlens_cpu=[chunk_len] * batch_size,
        prefix_lens_cpu=[prefix] * batch_size,
    )


@marker.parametrize("context_len", [32768, 131072], [32768])
@marker.benchmark("impl", ["fused_cuda", "current"], unit="ms")
def benchmark(context_len: int, impl: str):
    from sglang.kernels.ops.attention.minimax_sparse.token.flash_with_topk_idx_optimized import (
        token_select_prefill_fused_cuda,
    )
    from sglang.kernels.ops.attention.minimax_sparse.token.index_score import (
        token_select_prefill,
    )

    fns = {
        "fused_cuda": token_select_prefill_fused_cuda,
        "current": token_select_prefill,
    }
    kwargs = _build(batch_size=1, context_len=context_len, chunk_len=4096)
    return marker.do_bench(
        lambda: fns[impl](**kwargs),
        # The fused path syncs on the overflow counter, so it cannot be
        # captured in a CUDA graph.
        use_cuda_graph=False,
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
