"""Checks for the final study: head dim 64/128, block-max scores, block and
two-stage selection against independent PyTorch references."""

from dataclasses import replace

import pytest
import torch
from attention import Attention, LightningIndexer, PagedState
from block_select import BlockIndexer
from kernels import lightning_scores
from study_config import MAIN, make_cfg
from two_stage import TwoStageIndexer



def inputs(cfg, rows):
    qi = torch.randn(
        rows,
        cfg.indexer_q_heads,
        cfg.indexer_head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    w = torch.rand(rows, cfg.indexer_q_heads, device="cuda", dtype=torch.bfloat16)
    return qi, w


def reference_scores(cfg, state, qi, w, row):
    """Independent einsum for one query row -> [G, N] with causal -inf."""
    g, d = cfg.num_k_heads, cfg.indexer_head_dim
    q = qi.float().view(len(qi), g, -1, d)[row]
    k = state.ik[state.req_to_token[int(state.request_ids[row])].long()].float()
    k = k.expand(-1, g, -1)
    ref = (
        torch.einsum("gjd,ngd->gjn", q, k).relu() * w[row].float().view(g, -1, 1)
    ).sum(1)
    ref[:, int(state.positions[row]) + 1 :] = -float("inf")
    return ref


def reference_blocks(cfg, state, qi, w, row, count):
    """Top blocks by max token score, own block forced -> [G, count] sets."""
    ref = reference_scores(cfg, state, qi, w, row)
    blocks = ref.view(ref.shape[0], -1, 128).amax(-1)
    blocks[:, int(state.positions[row]) // 128] = float("inf")
    return blocks.topk(count, dim=-1).indices


@pytest.mark.parametrize("main", MAIN)
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("qpr", [1, 16])
@pytest.mark.parametrize("block_max", [False, True])
def test_scores(main, reduced, dim, qpr, block_max):
    torch.manual_seed(3)
    cfg = make_cfg(main, reduced, dim)
    state = PagedState(cfg, 2, 512, qpr, sparse=True)
    qi, w = inputs(cfg, 2 * qpr)
    actual = lightning_scores(
        qi,
        state.ik,
        w,
        state.req_to_token,
        state.request_ids,
        state.positions,
        512,
        cfg.num_k_heads,
        state.score_query_tile,
        block_max=block_max,
    )
    for row in range(2 * qpr):
        ref = reference_scores(cfg, state, qi, w, row)
        if block_max:
            ref = ref.view(ref.shape[0], -1, 128).amax(-1)
        torch.testing.assert_close(actual[:, row], ref, atol=3e-4, rtol=3e-5)


@pytest.mark.parametrize("main", MAIN)
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("qpr", [1, 16])
@torch.no_grad()
def test_block_selection(main, reduced, dim, qpr):
    torch.manual_seed(5)
    cfg = make_cfg(main, reduced, dim)
    state = PagedState(cfg, 2, 2048, qpr, sparse=True)
    indexer = BlockIndexer(cfg, top_blocks=4)
    qi, w = inputs(cfg, 2 * qpr)
    idx = indexer.select(qi, state.ik, w, state, 0, 2 * qpr, 512)
    assert idx.shape == (cfg.num_k_heads, 2 * qpr, 512)
    for row in range(2 * qpr):
        expected = reference_blocks(cfg, state, qi, w, row, 4)
        for g in range(cfg.num_k_heads):
            got = idx[g, row][idx[g, row] >= 0].long()
            assert (got <= int(state.positions[row])).all()
            assert got.unique().numel() == got.numel()
            assert set((got // 128).tolist()) == set(expected[g].tolist())
            own = int(state.positions[row])
            assert own in got.tolist()


@pytest.mark.parametrize("main", MAIN)
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("qpr", [1, 16])
@torch.no_grad()
def test_two_stage_matches_flat_when_shortlist_is_everything(main, reduced, dim, qpr):
    torch.manual_seed(9)
    cfg = make_cfg(main, reduced, dim)
    state = PagedState(cfg, 2, 1024, qpr, sparse=True, two_stage=True)
    qi, w = inputs(cfg, 2 * qpr)
    flat = LightningIndexer.select(
        LightningIndexer(cfg), qi, state.ik, w, state, 0, 2 * qpr, 256
    )
    staged = TwoStageIndexer(cfg, candidate_blocks=8).select(
        qi, state.ik, w, state, 0, 2 * qpr, 256
    )
    assert (staged <= state.positions[None, :, None]).all()
    for g in range(cfg.num_k_heads):
        for row in range(2 * qpr):
            a = set(flat[g, row][flat[g, row] >= 0].tolist())
            b = set(staged[g, row][staged[g, row] >= 0].tolist())
            assert a == b


@pytest.mark.parametrize("selector", ["token", "block", "two_stage"])
@pytest.mark.parametrize("qpr", [1, 16])
@torch.no_grad()
def test_forward_matches_reference_attention(selector, qpr):
    """Sparse output equals dense attention restricted to the selected tokens."""
    torch.manual_seed(13)
    cfg = make_cfg((48, 4), True, 128)
    layer = Attention(
        cfg,
        sparse=True,
        top_k=256,
        two_stage=selector == "two_stage",
        block=selector == "block",
        candidate_blocks=4,
    )
    state = PagedState(
        cfg,
        2,
        1024,
        qpr,
        sparse=True,
        phase="decode" if qpr == 1 else "prefill",
        two_stage=selector == "two_stage",
    )
    x = torch.randn(2 * qpr, cfg.hidden_dim, device="cuda", dtype=torch.bfloat16)
    out = layer.forward(x, state)
    qi, _, w = layer.indexer.project(x, x)
    idx = layer.indexer.select(qi, state.ik, w, state, 0, len(x), 256)
    from kernels import rms_norm_per_head
    import torch.nn.functional as F

    q = rms_norm_per_head(F.linear(x, layer.wq).view(-1, cfg.num_q_heads, 128))
    group = cfg.num_q_heads // cfg.num_k_heads
    for row in range(len(x)):
        slots_all = state.req_to_token[int(state.request_ids[row])].long()
        for g in range(cfg.num_k_heads):
            tokens = idx[g, row][idx[g, row] >= 0].long()
            k = state.k[slots_all[tokens], g].float()
            v = state.v[slots_all[tokens], g].float()
            heads = q[row, g * group : (g + 1) * group].float()
            p = (heads @ k.T * 128**-0.5).softmax(-1)
            torch.testing.assert_close(
                out[row, g * group : (g + 1) * group].float(),
                p @ v,
                atol=2e-2,
                rtol=2e-2,
            )
