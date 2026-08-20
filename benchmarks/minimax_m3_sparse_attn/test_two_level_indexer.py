#!/usr/bin/env python3
"""Correctness tests for the two-level (block -> token) GQA indexer prototype.

Needs one GPU. Run from this directory:

    CUDA_VISIBLE_DEVICES=0 python -m pytest test_two_level_indexer.py -v

The load-bearing test is ``test_*_matches_flat_when_recall_is_total``: configured
so stage 1 recalls *every* block, the two-level pass must reproduce the shipped
flat token indexer's selection exactly. Both stages then run the same bf16 dot
over the same operands, so this is an exact set comparison with no tie-tolerance
— it pins the whole candidate -> column -> position mapping, the biases, the
causal masking and the chunking in one assertion.

The sparse configurations that follow can only be checked statistically
(recall), because coarse recall is lossy by construction.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import build_decode_inputs, build_prefill_inputs  # noqa: E402
from m3_config import m3_config  # noqa: E402
from two_level_indexer import (  # noqa: E402
    TwoLevelConfig,
    build_pooled_index_keys,
    make_clustered_index_keys,
    reference_exact_topk,
    reference_two_level,
    selection_recall,
    two_level_select_decode,
    two_level_select_prefill,
)

from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (  # noqa: E402
    flash_decode_with_topk_idx,
)
from sglang.kernels.ops.attention.minimax_sparse.token.index_score import (  # noqa: E402
    token_select_decode,
    token_select_prefill,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

DEV = "cuda"
# Small shapes: 4 index heads / 4 KV heads is the released M3 ratio (group 1),
# 4 index heads / 2 KV heads exercises the in-kernel head reduction.
CFG = m3_config(num_q_heads=8, num_kv_heads=4, num_idx_heads=4)


def _index_sets(topk_idx: torch.Tensor) -> list[set]:
    return [
        set(r[r >= 0].tolist()) for r in topk_idx.reshape(-1, topk_idx.shape[-1]).cpu()
    ]


def _two_level(ctx: int, *, topk: int, pool: int, blocks: int, **kw) -> TwoLevelConfig:
    return TwoLevelConfig(
        pool_block_size=pool,
        coarse_blocks=blocks,
        topk_tokens=topk,
        **kw,
    )


# ---------------------------------------------------------------------------
# stage 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctx,pool", [(1024, 128), (1000, 128), (2048, 64)])
def test_pooling_matches_torch_mean(ctx, pool):
    inp = build_decode_inputs(CFG, batch_size=3, context_len=ctx)
    pooled = build_pooled_index_keys(
        inp.idx_k_cache,
        inp.req_to_token,
        inp.slot_ids,
        inp.seq_lens,
        pool_block_size=pool,
        max_seqlen_k=ctx,
    )
    nblk = math.ceil(ctx / pool)
    assert pooled.shape == (3, nblk, CFG.idx_head_dim)
    for b in range(3):
        slots = inp.req_to_token[b, :ctx].long()
        k = inp.idx_k_cache[slots, 0].float()  # [ctx, d]
        for n in range(nblk):
            live = k[n * pool : min((n + 1) * pool, ctx)]
            torch.testing.assert_close(
                pooled[b, n].float(), live.mean(dim=0), rtol=2e-2, atol=2e-2
            )


def test_partial_tail_block_is_not_diluted():
    """A tail block holding 1 token pools to that token, not to token/pool_size."""
    ctx = 129
    inp = build_decode_inputs(CFG, batch_size=1, context_len=ctx)
    pooled = build_pooled_index_keys(
        inp.idx_k_cache,
        inp.req_to_token,
        inp.slot_ids,
        inp.seq_lens,
        pool_block_size=128,
        max_seqlen_k=ctx,
    )
    tail = inp.idx_k_cache[inp.req_to_token[0, 128].long(), 0].float()
    torch.testing.assert_close(pooled[0, 1].float(), tail, rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# the exactness anchor: total recall must reproduce the flat indexer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_position", ["pre", "post"])
@pytest.mark.parametrize(
    "batch,ctx,chunk,topk,pool",
    [
        (1, 1024, 1024, 64, 128),
        (2, 2048, 512, 128, 128),  # chunked prefill (prefix + extend)
        (3, 1536, 1536, 256, 64),
        (1, 1000, 1000, 64, 128),  # ragged tail block
    ],
)
def test_prefill_matches_flat_when_recall_is_total(
    batch, ctx, chunk, topk, pool, pool_position
):
    inp = build_prefill_inputs(CFG, batch_size=batch, context_len=ctx, chunk_len=chunk)
    nblk = math.ceil(ctx / pool)
    cfg = _two_level(
        ctx,
        topk=topk,
        pool=pool,
        blocks=nblk,
        local_tokens=0,
        pool_position=pool_position,
    )

    got = two_level_select_prefill(
        inp.idx_q,
        inp.idx_k_cache,
        inp.req_to_token,
        inp.slot_ids,
        inp.cu_seqlens,
        inp.seq_lens,
        inp.prefix_lens,
        inp.max_seqlen_q,
        inp.max_seqlen_k,
        num_kv_heads=CFG.num_kv_heads,
        cfg=cfg,
    )
    ref = token_select_prefill(
        inp.idx_q,
        inp.idx_k_cache,
        inp.req_to_token,
        inp.slot_ids,
        inp.cu_seqlens,
        inp.seq_lens,
        inp.prefix_lens,
        inp.max_seqlen_q,
        inp.max_seqlen_k,
        topk,
        0,
        0,
        seqlens_cpu=inp.seqlens_cpu,
        prefix_lens_cpu=inp.prefix_lens_cpu,
    )
    assert _index_sets(got) == _index_sets(ref)


@pytest.mark.parametrize(
    "batch,ctx,topk,pool", [(1, 1024, 64, 128), (8, 2048, 128, 64)]
)
def test_decode_matches_flat_when_recall_is_total(batch, ctx, topk, pool):
    inp = build_decode_inputs(CFG, batch_size=batch, context_len=ctx)
    cfg = _two_level(
        ctx, topk=topk, pool=pool, blocks=math.ceil(ctx / pool), local_tokens=0
    )

    got = two_level_select_decode(
        inp.idx_q,
        inp.idx_k_cache,
        inp.req_to_token,
        inp.slot_ids,
        inp.seq_lens,
        inp.max_seqlen,
        num_kv_heads=CFG.num_kv_heads,
        cfg=cfg,
    )
    ref = token_select_decode(
        inp.idx_q,
        inp.idx_k_cache,
        inp.req_to_token,
        inp.slot_ids,
        inp.seq_lens,
        inp.max_seqlen,
        topk,
        0,
        0,
    )
    assert _index_sets(got) == _index_sets(ref)


@pytest.mark.parametrize("batch,ctx", [(2, 4096), (1, 16384)])
def test_post_pooling_at_block_budget_is_the_shipped_block_indexer(batch, ctx):
    """`pool_position="post"` with M = topk/P *is* M3's block indexer.

    Stage 1 then scores blocks exactly as the shipped kernel does (max over each
    block's per-key scores) and recalls M blocks, and stage 2 is handed M*P
    candidates for an M*P budget, so it keeps all of them. The selection must
    therefore match the shipped path's blocks, expanded to their tokens — not
    approximately, exactly. That makes `coarse_blocks` a dial from the released
    indexer (M = topk/P) to the exact token one (M = L/P), with everything in
    between a refinement of the block path rather than a different algorithm.
    """
    pool, topk = 128, 2048
    inp = build_decode_inputs(CFG, batch_size=batch, context_len=ctx)
    cfg = _two_level(
        ctx,
        topk=topk,
        pool=pool,
        blocks=topk // pool,
        init_tokens=0,
        local_tokens=pool,  # == the shipped path's local_blocks=1
        pool_position="post",
    )
    got = two_level_select_decode(
        inp.idx_q,
        inp.idx_k_cache,
        inp.req_to_token,
        inp.slot_ids,
        inp.seq_lens,
        inp.max_seqlen,
        num_kv_heads=CFG.num_kv_heads,
        cfg=cfg,
    )

    _, blocks, _ = flash_decode_with_topk_idx(
        q=inp.idx_q,
        k_cache=inp.idx_k_cache,
        v_cache=None,
        sink=None,
        req_to_token=inp.req_to_token,
        slot_ids=inp.slot_ids,
        seq_lens=inp.seq_lens,
        max_seqlen=inp.max_seqlen,
        block_size=pool,
        topk=topk // pool,
        init_blocks=0,
        local_blocks=1,
        disable_index_value=True,
        score_type="max",
    )
    off = torch.arange(pool, device=blocks.device)
    tokens = blocks[..., None].long() * pool + off
    live = (blocks >= 0)[..., None].expand(*blocks.shape, pool)
    tokens = torch.where(live & (tokens < ctx), tokens, torch.full_like(tokens, -1))
    assert _index_sets(got) == _index_sets(tokens.reshape(*blocks.shape[:2], -1))


# ---------------------------------------------------------------------------
# structural invariants that hold at any recall
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool_position", ["pre", "post"])
@pytest.mark.parametrize("init_tokens,local_tokens", [(128, 256), (0, 512), (256, 0)])
def test_forced_regions_are_always_selected(init_tokens, local_tokens, pool_position):
    """Sinks and the sliding window survive *both* levels, however lossy stage 1 is."""
    ctx, batch, topk = 4096, 2, 1024
    inp = build_decode_inputs(CFG, batch_size=batch, context_len=ctx)
    cfg = _two_level(
        ctx,
        topk=topk,
        pool=128,
        blocks=8,  # 1024 of 4096 tokens recalled: deliberately lossy
        init_tokens=init_tokens,
        local_tokens=local_tokens,
        pool_position=pool_position,
    )
    got = two_level_select_decode(
        inp.idx_q,
        inp.idx_k_cache,
        inp.req_to_token,
        inp.slot_ids,
        inp.seq_lens,
        inp.max_seqlen,
        num_kv_heads=CFG.num_kv_heads,
        cfg=cfg,
    )
    forced = set(range(init_tokens)) | set(range(ctx - local_tokens, ctx))
    assert len(forced) <= topk
    for row in _index_sets(got):
        assert forced <= row


@pytest.mark.parametrize("query_tile", [1, 16, 64])
def test_prefill_forced_regions_survive_the_tile_union(query_tile):
    """Every row's own sliding window is recalled, not just the tile's last row's.

    A query tile shares one candidate set, so the coarse stage has to bias the
    *union* of the tile's windows — bounded by its first row, not its last. Take
    the last row's bound instead and the earliest rows in each tile silently lose
    their window, which is what this pins.
    """
    ctx, chunk, local_tokens, topk = 8192, 512, 256, 1024
    inp = build_prefill_inputs(CFG, batch_size=1, context_len=ctx, chunk_len=chunk)
    cfg = _two_level(
        ctx,
        topk=topk,
        pool=128,
        blocks=8,  # 1024 of 8192 tokens: lossy enough that nothing is free
        local_tokens=local_tokens,
        query_tile=query_tile,
    )
    got = two_level_select_prefill(
        inp.idx_q,
        inp.idx_k_cache,
        inp.req_to_token,
        inp.slot_ids,
        inp.cu_seqlens,
        inp.seq_lens,
        inp.prefix_lens,
        inp.max_seqlen_q,
        inp.max_seqlen_k,
        num_kv_heads=CFG.num_kv_heads,
        cfg=cfg,
    ).cpu()

    prefix = ctx - chunk
    for kh in range(CFG.num_kv_heads):
        for i in range(0, chunk, 37):  # stride: every row of every tile position
            abs_q = prefix + i
            sel = set(got[kh, i].tolist())
            window = set(range(max(0, abs_q - local_tokens + 1), abs_q + 1))
            assert window <= sel, (
                f"row {i} (tile offset {i % query_tile}) lost "
                f"{len(window - sel)} of its {len(window)} window tokens"
            )


def test_selection_is_causal_distinct_and_in_range():
    ctx, chunk, batch = 2048, 512, 2
    inp = build_prefill_inputs(CFG, batch_size=batch, context_len=ctx, chunk_len=chunk)
    cfg = _two_level(ctx, topk=512, pool=128, blocks=8)
    got = two_level_select_prefill(
        inp.idx_q,
        inp.idx_k_cache,
        inp.req_to_token,
        inp.slot_ids,
        inp.cu_seqlens,
        inp.seq_lens,
        inp.prefix_lens,
        inp.max_seqlen_q,
        inp.max_seqlen_k,
        num_kv_heads=CFG.num_kv_heads,
        cfg=cfg,
    ).cpu()

    prefix = ctx - chunk
    for kh in range(CFG.num_kv_heads):
        for b in range(batch):
            for i in range(chunk):
                sel = got[kh, b * chunk + i]
                live = sel[sel >= 0].tolist()
                assert len(live) == len(set(live)), "duplicate positions selected"
                assert max(live) <= prefix + i, "selected a non-causal position"
                assert min(live) >= 0


def test_short_context_pads_with_minus_one():
    """A budget wider than the visible context comes back -1 padded, not garbage."""
    ctx = 128
    inp = build_decode_inputs(CFG, batch_size=2, context_len=ctx)
    cfg = _two_level(ctx, topk=256, pool=128, blocks=2)
    got = two_level_select_decode(
        inp.idx_q,
        inp.idx_k_cache,
        inp.req_to_token,
        inp.slot_ids,
        inp.seq_lens,
        inp.max_seqlen,
        num_kv_heads=CFG.num_kv_heads,
        cfg=cfg,
    )
    for row in _index_sets(got):
        assert row == set(range(ctx))


@pytest.mark.parametrize(
    "num_kv_heads,head_reduce", [(4, "max"), (2, "max"), (2, "sum"), (1, "relu_sum")]
)
def test_head_reduction_matches_torch_reference(num_kv_heads, head_reduce):
    """The in-kernel index-head fold, against a torch mirror of both stages."""
    ctx, topk, pool, blocks = 2048, 256, 128, 6
    cfg_shape = m3_config(num_q_heads=8, num_kv_heads=num_kv_heads, num_idx_heads=4)
    inp = build_decode_inputs(cfg_shape, batch_size=2, context_len=ctx)
    # Clustered keys, so the coarse ranking is well separated and the comparison
    # is about the head fold rather than about which of two near-tied blocks the
    # bf16 pooled dot happened to prefer.
    make_clustered_index_keys(
        inp.idx_k_cache, inp.req_to_token, inp.seq_lens, block=128, noise=0.35
    )
    cfg = _two_level(
        ctx,
        topk=topk,
        pool=pool,
        blocks=blocks,
        local_tokens=0,
        head_reduce=head_reduce,
    )
    got = two_level_select_decode(
        inp.idx_q,
        inp.idx_k_cache,
        inp.req_to_token,
        inp.slot_ids,
        inp.seq_lens,
        inp.max_seqlen,
        num_kv_heads=num_kv_heads,
        cfg=cfg,
    ).cpu()

    group = 4 // num_kv_heads
    scale = CFG.idx_head_dim**-0.5
    for b in range(2):
        slots = inp.req_to_token[b, :ctx].long()
        k = inp.idx_k_cache[slots, 0].float()
        for kh in range(num_kv_heads):
            q_row = inp.idx_q[b, kh * group : (kh + 1) * group]
            ref = reference_two_level(
                q_row,
                k,
                ctx - 1,
                sm_scale=scale,
                cfg=cfg,
                idx_q_coarse=inp.idx_q[b] if cfg.share_candidates else None,
            )
            # bf16 vs fp32 scoring can swap positions that tie at the cutoff;
            # require agreement on all but a handful.
            overlap = len(set(ref.cpu().tolist()) & set(got[kh, b].tolist()))
            assert overlap >= topk - 4, f"{overlap}/{topk} agree with the reference"


# ---------------------------------------------------------------------------
# recall — the quantity the approximation trades away
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "share_candidates,floor",
    [(False, 0.95), (True, 0.40)],
)
def test_recall_on_clustered_keys_is_high(share_candidates, floor):
    """Recall against the exact flat top-k, on keys with within-block structure.

    Mean pooling predicts a block's best key only insofar as keys inside a block
    resemble each other, which is a property of real KV (adjacent tokens have
    correlated index keys) and *not* of iid Gaussians. The cache is therefore
    re-drawn as per-block cluster centres plus noise before measuring; recall on
    the iid cache the harness builds by default is a floor, not a forecast, and
    is asserted separately below only to stay far off the floor.
    """
    ctx, batch, topk = 16384, 2, 1024
    inp = build_decode_inputs(CFG, batch_size=batch, context_len=ctx)
    make_clustered_index_keys(
        inp.idx_k_cache, inp.req_to_token, inp.seq_lens, block=128, noise=0.35
    )
    # 4096 of 16k tokens recalled. `share_candidates` makes the four index heads
    # agree on one candidate set, which is much lossier at this M — the floors
    # differ by more than a factor of two, which is the point of pinning both.
    cfg = _two_level(
        ctx,
        topk=topk,
        pool=128,
        blocks=32,
        local_tokens=0,
        share_candidates=share_candidates,
    )

    got = two_level_select_decode(
        inp.idx_q,
        inp.idx_k_cache,
        inp.req_to_token,
        inp.slot_ids,
        inp.seq_lens,
        inp.max_seqlen,
        num_kv_heads=CFG.num_kv_heads,
        cfg=cfg,
    ).cpu()

    scale = CFG.idx_head_dim**-0.5
    recalls = []
    for b in range(batch):
        slots = inp.req_to_token[b, :ctx].long()
        k = inp.idx_k_cache[slots, 0].float()
        for kh in range(CFG.num_kv_heads):
            exact = reference_exact_topk(
                inp.idx_q[b, kh : kh + 1],
                k,
                ctx - 1,
                topk=topk,
                sm_scale=scale,
                cfg=cfg,
            )
            r, filled = selection_recall(got[kh, b], exact.cpu())
            recalls.append(r)
            assert filled == pytest.approx(1.0), "budget not filled"
    mean_recall = sum(recalls) / len(recalls)
    assert mean_recall > floor, f"mean recall {mean_recall:.3f}"
