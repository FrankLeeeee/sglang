"""Exactness tests for the CUDA emitter of the MiniMax token-sparse selector.

``token_select_prefill_fused_cuda`` must return the same index *set* per row
as the reference ``token_select_prefill`` — the same contract the benchmark
suite enforces (benchmarks/minimax_m3_sparse_attn/bench_token_select.py).
Inputs mirror what ``MiniMaxSparseAttnBackend`` hands the kernels: a paged
index-K cache behind a req_to_token page table, chunked queries with a prefix.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

TOPK = 2048
NUM_HEADS = 4
HEAD_DIM = 128
LOCAL_TOKENS = 128


def _build_inputs(batch_size, context_len, chunk_len, device, dtype):
    torch.manual_seed(context_len + batch_size)
    prefix = context_len - chunk_len
    total_q = batch_size * chunk_len
    max_slots = (batch_size + 1) * context_len

    idx_q = torch.randn(total_q, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)
    idx_k_cache = torch.randn(max_slots, 1, HEAD_DIM, device=device, dtype=dtype)
    req_to_token = torch.zeros(
        batch_size + 1, context_len, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        req_to_token[b] = torch.randperm(max_slots, device=device)[:context_len].to(
            torch.int32
        )
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * chunk_len, chunk_len, dtype=torch.int32, device=device
    )
    seq_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device=device)
    prefix_lens = torch.full((batch_size,), prefix, dtype=torch.int32, device=device)
    slot_ids = torch.arange(batch_size, dtype=torch.int64, device=device)
    return dict(
        idx_q=idx_q,
        idx_k_cache=idx_k_cache,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=chunk_len,
        max_seqlen_k=context_len,
        topk=TOPK,
        init_tokens=0,
        local_tokens=LOCAL_TOKENS,
        seqlens_cpu=[chunk_len] * batch_size,
        prefix_lens_cpu=[prefix] * batch_size,
    )


def _index_sets(t):
    return [set(r[r >= 0].tolist()) for r in t.reshape(-1, t.shape[-1]).cpu()]


def _assert_equivalent(out, ref, kwargs):
    """Set-equal per row, allowing swaps of near-tied rank-k boundary scores.

    The CUDA emitter computes q.k through wmma while the reference goes
    through Triton's wgmma lowering; the two round differently by ~1 ulp,
    which can swap the pair straddling rank topk when their fp64 scores agree
    to ~1e-8. Any disputed position whose fp64 score is not within 1e-5 of
    the others' median is a real bug and fails.
    """
    A, B = _index_sets(out), _index_sets(ref)
    if A == B:
        return
    n_heads, total_q, _ = ref.shape
    cu = kwargs["cu_seqlens"].tolist()
    for i, (a, b) in enumerate(zip(A, B)):
        if a == b:
            continue
        assert len(a) == len(b), f"row {i}: cardinality {len(a)} vs {len(b)}"
        h, g_row = divmod(i, total_q)
        req = next(j for j in range(len(cu) - 1) if cu[j] <= g_row < cu[j + 1])
        q = kwargs["idx_q"][g_row, h].double()
        sid = int(kwargs["slot_ids"][req])
        scores = []
        for pos in (*(a - b), *(b - a)):
            slot = int(kwargs["req_to_token"][sid, pos])
            scores.append(float(q @ kwargs["idx_k_cache"][slot, 0].double()))
        med = sorted(scores)[len(scores) // 2]
        for s in scores:
            assert abs(s - med) <= 1e-5 * (1.0 + abs(med)), (
                f"row {i}: disputed scores {scores} are not a near-tie"
            )


@pytest.mark.parametrize(
    "batch_size,context_len,chunk_len",
    [
        # ragged: chunk not a multiple of the 64-row q tile, context not a
        # multiple of the 128-key tile — guards the partial-tile guards
        (1, 8000, 1000),
        # prefix = 0: whole-sequence chunk, rows with tiny causal widths
        (1, 8192, 8192),
        # batch > 1: guards per-request row indexing (the Triton emitters
        # shipped with a batch>1 aliasing bug this would have caught)
        (2, 16384, 4096),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_cuda_matches_reference(batch_size, context_len, chunk_len, dtype):
    from sglang.kernels.ops.attention.minimax_sparse.token.flash_with_topk_idx_optimized import (
        token_select_prefill_fused_cuda,
    )
    from sglang.kernels.ops.attention.minimax_sparse.token.index_score import (
        token_select_prefill,
    )

    kwargs = _build_inputs(batch_size, context_len, chunk_len, "cuda", dtype)
    ref = token_select_prefill(**kwargs)
    out = token_select_prefill_fused_cuda(**kwargs)
    assert out.shape == ref.shape
    _assert_equivalent(out, ref, kwargs)


def test_fused_cuda_short_rows_pad():
    """Rows shorter than topk must come back -1 padded, like the reference."""
    from sglang.kernels.ops.attention.minimax_sparse.token.flash_with_topk_idx_optimized import (
        token_select_prefill_fused_cuda,
    )
    from sglang.kernels.ops.attention.minimax_sparse.token.index_score import (
        token_select_prefill,
    )

    kwargs = _build_inputs(1, 1024, 1024, "cuda", torch.bfloat16)
    ref = token_select_prefill(**kwargs)
    out = token_select_prefill_fused_cuda(**kwargs)
    _assert_equivalent(out, ref, kwargs)
    assert int((out[:, 0] >= 0).sum(dim=-1).max()) == 1  # first row sees 1 token


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
