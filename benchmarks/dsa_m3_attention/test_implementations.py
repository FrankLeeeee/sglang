"""Adapter correctness against independent dense and selected-token references."""

import gc

import pytest
import torch
from implementations import DSACase, M3Case, init_runtime


@pytest.fixture(scope="module", autouse=True)
def runtime():
    init_runtime()
    yield
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("ci", [1, 2])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@torch.no_grad()
def test_m3_full_selection_matches_dense(ci, phase):
    qpr = 4 if phase == "prefill" else 1
    torch.manual_seed(42)
    case = M3Case(ci, 2, 512, qpr, True, phase, topk=512)
    q = case.project()
    qi = case.index_project()
    out = case.attend(q, qi)
    ref = []
    for row in range(2 * qpr):
        req, pos = row // qpr, 512 - qpr + row % qpr
        slots = case.s.req_to_token[req, : pos + 1].long()
        k = (
            case.s.k[slots]
            .float()
            .repeat_interleave(case.c.num_q_heads // case.c.num_k_heads, dim=1)
        )
        v = (
            case.s.v[slots]
            .float()
            .repeat_interleave(case.c.num_q_heads // case.c.num_k_heads, dim=1)
        )
        logits = torch.einsum("hd,nhd->hn", q[row].float(), k) * 128**-0.5
        ref.append(torch.einsum("hn,nhd->hd", logits.softmax(-1), v))
    torch.testing.assert_close(out.float(), torch.stack(ref), atol=0.025, rtol=0.025)
    if phase == "decode":
        expected = case.run().clone()
        for _ in range(3):
            case.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = case.run()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, atol=0.01, rtol=0.01)
        graph.reset()
    del case
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("ci", [1, 2])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@torch.no_grad()
def test_dsa_selected_attention_and_request_causality(ci, phase):
    qpr = 4 if phase == "prefill" else 1
    torch.manual_seed(42)
    case = DSACase(ci, 2, 4096, qpr, True, phase, topk=2048)
    q, qa = case.project()
    idx = case.select(qa)
    out = case.attend(q, idx)
    inv = torch.empty(8192, device="cuda", dtype=torch.long)
    inv[case.table.flatten().long()] = torch.arange(8192, device="cuda")
    ref = []
    for row in range(2 * qpr):
        valid = idx[row][idx[row] >= 0].long()
        assert len(valid) == 2048
        assert len(valid.unique()) == len(valid)
        logical = inv[valid]
        assert (logical // 4096 == row // qpr).all()
        assert (logical % 4096 <= case.positions[row]).all()
        kv = case.kv[valid, 0].float()
        logits = q[row].float() @ kv.T * 192**-0.5
        ref.append(logits.softmax(-1) @ kv[:, :512])
    torch.testing.assert_close(out.float(), torch.stack(ref), atol=0.03, rtol=0.03)
    if phase == "decode":
        expected = case.run().clone()
        for _ in range(3):
            case.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = case.run()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, atol=0.01, rtol=0.01)
        graph.reset()
    del case
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("ci", [1, 2])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@torch.no_grad()
def test_dsa_dense_matches_reference(ci, phase):
    qpr = 4 if phase == "prefill" else 1
    torch.manual_seed(42)
    case = DSACase(ci, 2, 512, qpr, False, phase)
    q, _ = case.project()
    out = case.attend(q)
    ref = []
    for row in range(2 * qpr):
        slots = case.table[row // qpr, : int(case.positions[row]) + 1].long()
        kv = case.kv[slots, 0].float()
        logits = q[row].float() @ kv.T * 192**-0.5
        ref.append(logits.softmax(-1) @ kv[:, :512])
    torch.testing.assert_close(out.float(), torch.stack(ref), atol=0.03, rtol=0.03)
    if phase == "decode":
        expected = case.run().clone()
        for _ in range(3):
            case.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = case.run()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, atol=0.01, rtol=0.01)
        graph.reset()
    del case
    gc.collect()
    torch.cuda.empty_cache()
