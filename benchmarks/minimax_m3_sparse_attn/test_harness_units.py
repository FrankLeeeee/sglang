"""CPU-only unit tests for the benchmark harness's bookkeeping.

These pin the two things that have silently gone wrong before:

  * kernel -> stage attribution, which is ordered substring matching. The dense
    rule "_fwd_kernel" is a *suffix* of the sparse kernels' names, so a careless
    reordering silently charges sparse work to the dense stage.
  * the prefill score-buffer budget, which must be a hard cap.

Run with:  python -m pytest benchmarks/minimax_m3_sparse_attn/test_harness_units.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import STAGE_ORDER, analytic_memory, classify_kernel
from m3_config import (
    DEFAULT_CONTEXT_LENS,
    DEFAULT_PREFILL_CHUNK,
    M3_MAX_POSITION,
    m3_config,
)

MIB = 1024**2


def test_default_context_sweep_reaches_model_limit_safely():
    assert DEFAULT_CONTEXT_LENS[-1] == M3_MAX_POSITION == 1048576
    assert DEFAULT_CONTEXT_LENS == sorted(set(DEFAULT_CONTEXT_LENS))
    assert DEFAULT_PREFILL_CHUNK == 4096


@pytest.mark.parametrize(
    "kernel,stage",
    [
        # sparse attention — these END in "_fwd_kernel" and must not be
        # swallowed by the generic dense rule
        ("_gqa_share_sparse_fwd_kernel", "sparse_attn"),
        ("_gqa_share_sparse_decode_kernel", "sparse_attn"),
        ("_gqa_token_sparse_fwd_kernel", "sparse_attn"),
        # dense
        ("_fwd_kernel", "dense_attn"),
        ("_fwd_kernel_unified", "dense_attn"),
        ("_fwd_grouped_kernel_stage1", "dense_attn"),
        ("_fwd_kernel_stage2", "dense_attn"),
        # indexer
        ("_flash_attn_fwd_with_block_score_kernel", "indexer_score"),
        ("_decode_score_kernel", "indexer_score"),
        ("_token_index_score_prefill_kernel", "indexer_score"),
        ("_token_index_score_decode_kernel", "indexer_score"),
        # selection
        ("_topk_index_kernel", "topk_select"),
        ("minimax_prefill_topk_block_kernel", "topk_select"),
        ("minimax_decode_topk_block_kernel", "topk_select"),
        (
            "(anonymous namespace)::topk_kernel("
            "(anonymous namespace)::FastTopKParams)",
            "topk_select",
        ),
        ("void at::native::mbtopk::computeBlockDigitCounts<float,...>", "topk_select"),
        # merge
        ("_merge_attn_out_kernel", "merge"),
        ("_merge_chunks_kernel", "merge"),
    ],
)
def test_stage_attribution(kernel, stage):
    assert classify_kernel(kernel) == stage


def test_every_stage_is_declared_in_order():
    for _, stage in __import__("harness").STAGE_RULES:
        assert stage in STAGE_ORDER, f"{stage} missing from STAGE_ORDER"


@pytest.mark.parametrize(
    "batch_size,context_len,num_idx_heads",
    [
        (1, 131072, 1),
        (8, 131072, 1),
        (64, 131072, 1),
        (64, 131072, 4),
        (256, 131072, 1),
        # the case that used to blow the cap by 8x: batch and context both large
        (256, 1048576, 4),
        (4096, 131072, 4),
    ],
)
def test_prefill_score_budget_is_a_hard_cap(batch_size, context_len, num_idx_heads):
    """The chunk planner must never plan a buffer larger than the budget."""
    from sglang.kernels.ops.attention.minimax_sparse.token.index_score import (
        plan_query_chunk,
    )

    budget = 512 * MIB
    batch_chunk, chunk = plan_query_chunk(
        batch_size=batch_size,
        max_seqlen_q=context_len,
        max_seqlen_k=context_len,
        num_idx_heads=num_idx_heads,
        score_budget_bytes=budget,
    )
    assert 1 <= batch_chunk <= batch_size
    assert chunk >= 1
    planned = num_idx_heads * batch_chunk * chunk * context_len * 4
    assert planned <= budget, (
        f"planned {planned / MIB:.0f} MiB for a {budget / MIB:.0f} MiB budget"
    )


def test_impossible_budget_raises_rather_than_over_allocating():
    """One query row of one request over the cap is irreducible — fail loudly."""
    from sglang.kernels.ops.attention.minimax_sparse.token.index_score import (
        plan_query_chunk,
    )

    with pytest.raises(ValueError, match="cannot honour"):
        plan_query_chunk(
            batch_size=1,
            max_seqlen_q=1024,
            max_seqlen_k=1048576,
            num_idx_heads=4,
            score_budget_bytes=1 * MIB,
        )


def test_dense_reports_no_indexer_memory():
    cfg = m3_config(granularity="dense")
    for is_prefill, bs, nq in ((True, 1, 4096), (False, 32, 32)):
        mem = analytic_memory(
            cfg,
            batch_size=bs,
            context_len=4096,
            num_query_tokens=nq,
            is_prefill=is_prefill,
        )
        assert mem["score_buffer_bytes"] == 0
        assert mem["topk_idx_bytes"] == 0
        assert mem["kv_index_bytes"] == 0
        # main KV only: K+V * 4 kv heads * 128 dims * 2 B
        assert mem["kv_bytes_per_token"] == 2048


def test_token_score_buffer_is_block_size_larger_per_row():
    """Dropping the pooling multiplies the per-row score buffer by block_size."""
    block = m3_config(granularity="block")
    token = m3_config(granularity="token")
    kw = dict(batch_size=32, context_len=32768, num_query_tokens=32, is_prefill=False)
    b = analytic_memory(block, **kw)["score_buffer_bytes"]
    t = analytic_memory(token, **kw)["score_buffer_bytes"]
    assert t == b * block.block_size


def test_token_prefill_reports_live_chunk_not_full_matrix():
    cfg = m3_config(granularity="token")
    mem = analytic_memory(
        cfg,
        batch_size=1,
        context_len=131072,
        num_query_tokens=131072,
        is_prefill=True,
        score_budget_bytes=512 * MIB,
    )
    assert mem["score_buffer_bytes"] <= 512 * MIB
    # the full matrix it stands in for is far larger — and one row per index
    # head, which is what makes it 4x the naive [queries, keys] estimate
    assert (
        mem["score_buffer_unchunked_bytes"]
        == cfg.num_idx_heads * 131072 * 131072 * 4
    )
    assert mem["score_buffer_unchunked_bytes"] > 100 * mem["score_buffer_bytes"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
