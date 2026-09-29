"""GPU correctness: formula, causal isolation, padding, paged gather, graphs."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from attention import CONFIGS, Attention, LightningIndexer, PagedState
from kernels import lightning_scores, rms_norm_per_head


@pytest.mark.parametrize("ci", [1, 2])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("qpr", [1, 8, 16])
@pytest.mark.parametrize("reduced", [False, True, "mqa"])
def test_scores(ci, grouped, qpr, reduced):
    torch.manual_seed(3)
    c = CONFIGS[ci]
    c = replace(c, indexer_kv_heads=c.num_k_heads if grouped else 1)
    if reduced:
        c = replace(c, indexer_q_heads=c.num_k_heads)
    if reduced == "mqa":
        c = replace(c, num_k_heads=1, indexer_kv_heads=1)
    s = PagedState(c, 2, 128, qpr, sparse=True)
    qi = torch.randn(
        2 * qpr, c.indexer_q_heads, 64, device="cuda", dtype=torch.bfloat16
    )
    w = torch.rand(2 * qpr, c.indexer_q_heads, device="cuda", dtype=torch.bfloat16)
    actual = lightning_scores(
        qi,
        s.ik,
        w,
        s.req_to_token,
        s.request_ids,
        s.positions,
        128,
        c.num_k_heads,
        1 if qpr == 1 else 4,
    )
    q = qi.float().view(2 * qpr, c.num_k_heads, -1, 64)
    for r in range(2 * qpr):
        k = s.ik[s.req_to_token[r // qpr].long()].float()
        if not grouped:
            k = k.expand(-1, c.num_k_heads, -1)
        ref = (
            torch.einsum("gjd,ngd->gjn", q[r], k).relu()
            * w[r].float().view(c.num_k_heads, -1, 1)
        ).sum(1)
        ref[:, int(s.positions[r]) + 1 :] = -float("inf")
        torch.testing.assert_close(actual[:, r], ref, atol=2e-4, rtol=2e-5)


@pytest.mark.parametrize("ci", [1, 2])
@pytest.mark.parametrize("qpr", [1, 8, 16])
@pytest.mark.parametrize("topk", [16, 128])
@pytest.mark.parametrize("reduced", [False, True, "mqa"])
@torch.no_grad()
def test_forward(ci, qpr, topk, reduced):
    torch.manual_seed(17)
    c = CONFIGS[ci]
    if reduced:
        c = replace(c, indexer_q_heads=c.num_k_heads, indexer_kv_heads=1)
    if reduced == "mqa":
        c = replace(c, num_k_heads=1)
    sparse = Attention(c, sparse=True, top_k=topk)
    s = PagedState(
        c, 2, 128, qpr, sparse=True, phase="decode" if qpr == 1 else "prefill"
    )
    # Start the second request at position zero to exercise causal -1 padding.
    if qpr == 8:
        s.positions[qpr:] = torch.arange(qpr, device="cuda")
        s.write_slots = s.req_to_token[
            s.request_ids.long(), s.positions.long()
        ].contiguous()
    x = torch.randn(2 * qpr, c.hidden_dim, device="cuda", dtype=torch.bfloat16)
    out = sparse.forward(x, s)
    q = rms_norm_per_head(F.linear(x, sparse.wq).view(-1, c.num_q_heads, 128))
    qi, _, w = sparse.indexer.project(x, x)
    idx = sparse.indexer.select(qi, s.ik, w, s, 0, len(x), topk)
    scores = lightning_scores(
        qi,
        s.ik,
        w,
        s.req_to_token,
        s.request_ids,
        s.positions,
        128,
        c.num_k_heads,
        1 if qpr == 1 else 4,
    )
    selected = scores.gather(-1, idx.clamp_min(0).long()).masked_fill(
        idx < 0, -float("inf")
    )
    torch.testing.assert_close(
        selected.sort(-1).values, scores.topk(topk, dim=-1).values.sort(-1).values
    )
    reference = torch.empty_like(out)
    group_size = c.num_q_heads // c.num_k_heads
    for r in range(len(x)):
        for g in range(c.num_k_heads):
            pos = idx[g, r].long()
            assert ((pos == -1) | ((pos >= 0) & (pos <= s.positions[r]))).all()
            valid = pos[pos >= 0]
            assert valid.unique().numel() == valid.numel()
            slots = s.req_to_token[r // qpr, valid].long()
            qr = q[r, g * group_size : (g + 1) * group_size].float()
            k, v = s.k[slots, g].float(), s.v[slots, g].float()
            reference[r, g * group_size : (g + 1) * group_size] = (
                (qr @ k.T / 128**0.5).softmax(-1) @ v
            ).bfloat16()
    torch.testing.assert_close(out, reference, atol=0.02, rtol=0.02)
    # Graph replay must also reproduce the actual projection/selection pipeline.
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_out = sparse.forward(x, s)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, out, atol=0, rtol=0)


@pytest.mark.parametrize("ci", [1, 2])
@pytest.mark.parametrize("qpr", [1, 8, 16])
@pytest.mark.parametrize("mqa", [False, True])
@torch.no_grad()
def test_full_selection_matches_dense(ci, qpr, mqa):
    c = CONFIGS[ci]
    if mqa:
        c = replace(c, num_k_heads=1, indexer_q_heads=c.num_k_heads, indexer_kv_heads=1)
    torch.manual_seed(9)
    sparse = Attention(c, sparse=True, top_k=128)
    dense = Attention(c)
    dense.wq, dense.wk, dense.wv = sparse.wq, sparse.wk, sparse.wv
    s = PagedState(
        c, 2, 128, qpr, sparse=True, phase="decode" if qpr == 1 else "prefill"
    )
    d = PagedState(c, 2, 128, qpr, sparse=False, phase=s.phase)
    # Copy logically ordered caches between different physical page permutations.
    d.k[d.req_to_token.long()] = s.k[s.req_to_token.long()]
    d.v[d.req_to_token.long()] = s.v[s.req_to_token.long()]
    x = torch.randn(2 * qpr, c.hidden_dim, device="cuda", dtype=torch.bfloat16)
    actual, expected = sparse.forward(x, s), dense.forward(x, d)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


def test_norm():
    x = torch.randn(7, 48, 128, device="cuda", dtype=torch.bfloat16)
    ref = (
        x.float() * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    ).bfloat16()
    torch.testing.assert_close(rms_norm_per_head(x), ref, atol=0, rtol=0)


@pytest.mark.parametrize("reduced", [False, True, "mqa"])
@torch.no_grad()
def test_chunked_scores(reduced):
    c = CONFIGS[2]
    if reduced:
        c = replace(c, indexer_q_heads=c.num_k_heads, indexer_kv_heads=1)
    if reduced == "mqa":
        c = replace(c, num_k_heads=1)
    layer = Attention(c, sparse=True, top_k=32)
    state = PagedState(c, 2, 128, 32, sparse=True)
    x = torch.randn(64, c.hidden_dim, device="cuda", dtype=torch.bfloat16)
    reference = layer.forward(x, state)
    layer.score_budget = 16 * c.num_k_heads * 128 * 4
    actual = layer.forward(x, state)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize("length", [16384, 131072])
@pytest.mark.parametrize("reduced", [False, True, "mqa"])
@torch.no_grad()
def test_top2048_long_context(length, reduced):
    """Exercise the real selector at both ends of the measured length range."""
    c = CONFIGS[2]
    if reduced:
        c = replace(c, indexer_q_heads=c.num_k_heads, indexer_kv_heads=1)
    if reduced == "mqa":
        c = replace(c, num_k_heads=1)
    indexer = LightningIndexer(c)
    state = PagedState(c, 2, length, 1, sparse=True, phase="decode")
    q = torch.randn(2, c.indexer_q_heads, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(2, c.indexer_q_heads, device="cuda", dtype=torch.bfloat16)
    scores = lightning_scores(
        q,
        state.ik,
        w,
        state.req_to_token,
        state.request_ids,
        state.positions,
        length,
        c.num_k_heads,
        query_tile=1,
    )
    selected = indexer.select(q, state.ik, w, state, 0, 2, 2048)
    actual = scores.gather(-1, selected.long()).sort(-1).values
    expected = scores.topk(2048, dim=-1).values.sort(-1).values
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for row in selected.flatten(0, 1):
        assert row.unique().numel() == 2048
