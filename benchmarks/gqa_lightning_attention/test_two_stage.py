"""Independent checks for block summaries, fine scores, and causal isolation."""

from dataclasses import replace

import pytest
import torch
from attention import CONFIGS, Attention, LightningIndexer, PagedState
from kernels import lightning_scores


@pytest.mark.parametrize("experiment", [1, 2, 3])
@pytest.mark.parametrize("ci", [1, 2])
@pytest.mark.parametrize("qpr", [1, 8])
@torch.no_grad()
def test_two_stage(experiment, ci, qpr):
    cfg = CONFIGS[ci]
    if experiment >= 2:
        cfg = replace(cfg, indexer_q_heads=cfg.num_k_heads, indexer_kv_heads=1)
    if experiment == 3:
        cfg = replace(cfg, num_k_heads=1)
    torch.manual_seed(7)
    layer = Attention(cfg, sparse=True, two_stage=True, top_k=128, candidate_blocks=2)
    state = PagedState(
        cfg,
        2,
        512,
        qpr,
        sparse=True,
        phase="decode" if qpr == 1 else "prefill",
        two_stage=True,
    )
    expected = (
        state.ik.view(-1, 128, cfg.indexer_kv_heads, 64).float().mean(1).bfloat16()
    )
    torch.testing.assert_close(state.block_ik, expected)
    x = torch.randn(2 * qpr, 8192, device="cuda", dtype=torch.bfloat16)
    result = layer.forward(x, state)
    qi, _, w = layer.indexer.project(x, x)
    blocks = layer.indexer.shortlist(qi, w, state, 0, len(x))
    assert ((blocks == state.block_positions[None, :, None]).any(-1)).all()
    scores = layer.indexer.candidate_scores(qi, state.ik, w, state, 0, len(x), blocks)
    full = lightning_scores(
        qi,
        state.ik,
        w,
        state.req_to_token,
        state.request_ids,
        state.positions,
        512,
        cfg.num_k_heads,
        1 if qpr == 1 else 4,
    )
    tokens = (blocks[:, :, :, None] * 128 + torch.arange(128, device="cuda")).flatten(2)
    ref = full.gather(-1, tokens.clamp_min(0).long()).masked_fill(
        tokens < 0, -float("inf")
    )
    torch.testing.assert_close(scores, ref, atol=2e-4, rtol=2e-5)
    selected = layer.indexer.select(qi, state.ik, w, state, 0, len(x), 128)
    assert (selected >= 0).all() and (selected <= state.positions[None, :, None]).all()
    torch.testing.assert_close(
        full.gather(-1, selected.long()).sort(-1).values,
        scores.topk(128, dim=-1).values.sort(-1).values,
        atol=2e-4,
        rtol=2e-5,
    )
    assert torch.isfinite(result).all()
    # Selecting every candidate block recovers global top-k scores.
    layer.indexer.candidate_blocks = 4
    all_indices = layer.indexer.select(qi, state.ik, w, state, 0, len(x), 128)
    global_indices = LightningIndexer.select(
        layer.indexer, qi, state.ik, w, state, 0, len(x), 128
    )
    torch.testing.assert_close(
        full.gather(-1, all_indices.long()).sort(-1).values,
        full.gather(-1, global_indices.long()).sort(-1).values,
        atol=2e-4,
        rtol=2e-5,
    )
    if qpr == 1:
        for _ in range(3):
            layer.forward(x, state)
        expected = layer.forward(x, state)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = layer.forward(x, state)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected)
        graph.reset()


@torch.no_grad()
def test_future_and_other_request_do_not_affect_selection():
    cfg = CONFIGS[1]
    torch.manual_seed(8)
    layer = Attention(cfg, sparse=True, two_stage=True, top_k=128, candidate_blocks=2)
    state = PagedState(cfg, 2, 512, 8, sparse=True, two_stage=True)
    x = torch.randn(16, 8192, device="cuda", dtype=torch.bfloat16)
    layer.forward(x, state)
    qi, _, w = layer.indexer.project(x, x)
    old = layer.indexer.select(qi, state.ik, w, state, 0, 8, 128)
    # Current block is always included; its future-contaminated summary must not
    # influence which other block is selected. Fine token scores mask the future.
    state.ik[state.req_to_token[0, 505:].long()] = 1000
    state.ik[state.req_to_token[1].long()] = -1000
    from two_stage import prepare_block_cache

    prepare_block_cache(state)
    new = layer.indexer.select(qi, state.ik, w, state, 0, 8, 128)
    torch.testing.assert_close(old[:, 0], new[:, 0])


@torch.no_grad()
def test_early_query_padding():
    from two_stage import prepare_block_cache

    cfg = CONFIGS[1]
    layer = Attention(cfg, sparse=True, two_stage=True, top_k=128, candidate_blocks=2)
    state = PagedState(cfg, 2, 512, 8, sparse=True, two_stage=True)
    state.positions.copy_(torch.arange(8, device="cuda", dtype=torch.int32).repeat(2))
    prepare_block_cache(state)
    x = torch.randn(16, 8192, device="cuda", dtype=torch.bfloat16)
    q, _, w = layer.indexer.project(x, x)
    selected = layer.indexer.select(q, state.ik, w, state, 0, 16, 128)
    for g in range(cfg.num_k_heads):
        for row in range(16):
            ids = selected[g, row]
            valid = ids[ids >= 0]
            assert len(valid) == row % 8 + 1
            assert valid.unique().numel() == len(valid)
            assert (valid <= row % 8).all()
            assert (ids[ids < 0] == -1).all()
