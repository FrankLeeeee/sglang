"""Correctness for the MiniMax-M3 token-granularity sparse attention kernels.

Covers the two stages of the DeepSeek-style token-level path:

  * ``token_select_prefill`` / ``token_select_decode`` — per-key indexer logits
    with *no* block pooling, plus the forced attention-sink / sliding-window
    bias, selecting individual token positions.
  * ``gqa_token_sparse_attn`` — GQA flash attention restricted to those
    positions, through the paged KV cache.

The selection is checked as an exact index *set* against a PyTorch reference
(random logits make ties vanishingly unlikely), and the attention output against
a gather-and-softmax reference. Split-K decode must agree with the single-chunk
result, and the chunked prefill logits buffer must not change the answer.
"""

import pytest
import torch

from sglang.kernels.ops.attention.minimax_sparse.token import (
    INIT_BIAS,
    LOCAL_BIAS,
    gqa_token_sparse_attn,
    token_select_decode,
    token_select_prefill,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEV = "cuda"
# bf16 q/k/v through two matmuls and a softmax; this is the round-off floor.
RTOL = ATOL = 2e-2


# ---------------------------------------------------------------------------
# input construction
# ---------------------------------------------------------------------------


def _build(
    batch_size,
    num_q_heads,
    num_kv_heads,
    num_idx_heads,
    head_dim,
    idx_head_dim,
    seq_lens,
    chunk_lens,
    page_size,
    dtype=torch.bfloat16,
    seed=0,
):
    """Paged KV pool + ragged extend batch, laid out like sglang's allocator."""
    torch.manual_seed(seed)
    max_len = max(seq_lens)
    pages_per_req = (max_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_req
    max_slots = total_pages * page_size

    # tokens contiguous within a page, pages scattered across the pool
    page_perm = torch.randperm(total_pages, device=DEV)
    within = torch.arange(max_len, device=DEV) % page_size
    page_of = torch.arange(max_len, device=DEV) // page_size
    req_to_token = (
        (page_perm.view(batch_size, pages_per_req)[:, page_of] * page_size + within)
        .to(torch.int32)
        .contiguous()
    )

    total_q = sum(chunk_lens)
    cu = torch.tensor(
        [0] + torch.tensor(chunk_lens).cumsum(0).tolist(), dtype=torch.int32, device=DEV
    )
    return dict(
        q=torch.randn(total_q, num_q_heads, head_dim, dtype=dtype, device=DEV),
        idx_q=torch.randn(
            total_q, num_idx_heads, idx_head_dim, dtype=dtype, device=DEV
        ),
        k_cache=torch.randn(max_slots, num_kv_heads, head_dim, dtype=dtype, device=DEV),
        v_cache=torch.randn(max_slots, num_kv_heads, head_dim, dtype=dtype, device=DEV),
        idx_k_cache=torch.randn(max_slots, 1, idx_head_dim, dtype=dtype, device=DEV),
        req_to_token=req_to_token,
        cu_seqlens=cu,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
        prefix_lens=torch.tensor(
            [s - c for s, c in zip(seq_lens, chunk_lens)], dtype=torch.int32, device=DEV
        ),
        slot_ids=torch.arange(batch_size, dtype=torch.int64, device=DEV),
    )


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------


def _biased_scores(idx_q_rows, idx_k, abs_q, seq_len, init_tokens, local_tokens):
    """[num_idx_heads, n_query, seq_len] logits with the forced-position biases."""
    scale = idx_q_rows.shape[-1] ** -0.5
    s = torch.einsum("nhd,ld->hnl", idx_q_rows.float(), idx_k.float()) * scale
    pos = torch.arange(seq_len, device=s.device)
    causal = abs_q[:, None] >= pos[None, :]
    init_m = causal & (pos[None, :] < init_tokens)
    local_m = causal & (pos[None, :] > abs_q[:, None] - local_tokens)
    s = torch.where(init_m[None], torch.tensor(INIT_BIAS, device=s.device), s)
    s = torch.where(
        (local_m & ~init_m)[None], torch.tensor(LOCAL_BIAS, device=s.device), s
    )
    return torch.where(causal[None], s, torch.tensor(float("-inf"), device=s.device))


def _ref_select_prefill(t, seq_lens, chunk_lens, topk, init_tokens, local_tokens):
    num_idx_heads = t["idx_q"].shape[1]
    out = torch.full(
        (num_idx_heads, t["idx_q"].shape[0], topk), -1, dtype=torch.int32, device=DEV
    )
    for b, (seq_len, chunk) in enumerate(zip(seq_lens, chunk_lens)):
        lo, hi = int(t["cu_seqlens"][b]), int(t["cu_seqlens"][b + 1])
        slots = t["req_to_token"][b, :seq_len].long()
        abs_q = (seq_len - chunk) + torch.arange(chunk, device=DEV)
        s = _biased_scores(
            t["idx_q"][lo:hi],
            t["idx_k_cache"][slots, 0],
            abs_q,
            seq_len,
            init_tokens,
            local_tokens,
        )
        k = min(topk, seq_len)
        values, indices = torch.topk(s, k, dim=-1)
        out[:, lo:hi, :k] = torch.where(
            values > float("-inf"),
            indices.int(),
            torch.full_like(indices, -1, dtype=torch.int32),
        )
    return out


def _ref_select_decode(t, seq_lens, topk, init_tokens, local_tokens):
    num_idx_heads = t["idx_q"].shape[1]
    batch_size = len(seq_lens)
    out = torch.full(
        (num_idx_heads, batch_size, topk), -1, dtype=torch.int32, device=DEV
    )
    for b, seq_len in enumerate(seq_lens):
        slots = t["req_to_token"][b, :seq_len].long()
        abs_q = torch.tensor([seq_len - 1], device=DEV)
        s = _biased_scores(
            t["idx_q"][b : b + 1],
            t["idx_k_cache"][slots, 0],
            abs_q,
            seq_len,
            init_tokens,
            local_tokens,
        )  # [h, 1, seq_len]
        k = min(topk, seq_len)
        _, indices = torch.topk(s[:, 0], k, dim=-1)
        out[:, b, :k] = indices.int()
    return out


def _ref_attn(q, k_cache, v_cache, req_to_token, q_slot_ids, topk_idx):
    """Gather the selected tokens and run a plain softmax attention per query."""
    n, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group = num_q_heads // num_kv_heads
    scale = head_dim**-0.5
    out = torch.zeros(n, num_q_heads, head_dim, dtype=torch.float32, device=q.device)
    for i in range(n):
        sid = int(q_slot_ids[i])
        for kh in range(num_kv_heads):
            sel = topk_idx[kh, i]
            sel = sel[sel >= 0].long()
            if sel.numel() == 0:
                continue
            slots = req_to_token[sid, sel].long()
            logits = (
                q[i, kh * group : (kh + 1) * group].float()
                @ k_cache[slots, kh].float().T
            ) * scale
            out[i, kh * group : (kh + 1) * group] = (
                torch.softmax(logits, dim=-1) @ v_cache[slots, kh].float()
            )
    return out


def _align(topk_idx, num_kv_heads):
    num_idx_heads = topk_idx.shape[0]
    if num_idx_heads == num_kv_heads:
        return topk_idx
    if num_idx_heads == 1:
        return topk_idx.expand(num_kv_heads, -1, -1).contiguous()
    return topk_idx.repeat_interleave(num_kv_heads // num_idx_heads, dim=0)


def _index_sets(topk_idx):
    return [
        set(r[r >= 0].tolist()) for r in topk_idx.reshape(-1, topk_idx.shape[-1]).cpu()
    ]


@pytest.mark.parametrize(
    "cfg,expected",
    [
        (
            {"sparse_init_tokens": 3, "sparse_local_tokens": 129},
            (1, 3, 3, 129),
        ),
        (
            {"sparse_init_block": 2, "sparse_local_block": 4},
            (2, 4, 256, 512),
        ),
        (
            {
                "sparse_init_block": 2,
                "sparse_local_block": 4,
                "sparse_init_tokens": 3,
                "sparse_local_tokens": 129,
            },
            (2, 4, 3, 129),
        ),
    ],
)
def test_token_windows_preserve_exact_config_counts(cfg, expected):
    from sglang.srt.layers.attention.minimax_sparse_backend import (
        _resolve_sparse_window_counts,
    )

    assert _resolve_sparse_window_counts(cfg, 128) == expected


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


def _case(
    bs, nqh, nkh, nih, hd, ihd, seqs, chunks, topk, init, local, page, budget=None
):
    tag = (
        f"bs{bs}_q{nqh}kv{nkh}idx{nih}_d{hd}-{ihd}_topk{topk}"
        f"_init{init}_local{local}_page{page}"
        + ("_chunkedbuf" if budget else "")
        + ("_chunkedprefill" if seqs != chunks else "")
    )
    return pytest.param(
        bs,
        nqh,
        nkh,
        nih,
        hd,
        ihd,
        seqs,
        chunks,
        topk,
        init,
        local,
        page,
        budget,
        id=tag,
    )


PREFILL_CASES = [
    # MiniMax-M3 per-rank shape at tp=8 (8 Q : 1 KV : 1 index head)
    _case(1, 8, 1, 1, 128, 128, [512], [512], 64, 0, 128, 128),
    # unpaged
    _case(1, 8, 1, 1, 128, 128, [512], [512], 64, 0, 128, 1),
    # ragged batch
    _case(2, 8, 1, 1, 128, 128, [512, 300], [512, 300], 64, 0, 128, 128),
    # chunked prefill (non-zero prefix) + forced sinks
    _case(2, 8, 1, 1, 128, 128, [512, 384], [128, 96], 64, 4, 64, 128),
    # full-model shape: 16 Q : 4 KV : 4 index heads
    _case(1, 16, 4, 4, 128, 128, [768], [768], 128, 0, 128, 64),
    # one index head broadcast over several KV heads, head_dim 64
    _case(1, 8, 2, 1, 64, 64, [640], [640], 32, 2, 32, 32),
    # tiny logits budget forces several query chunks
    _case(1, 8, 1, 1, 128, 128, [1024], [1024], 64, 0, 128, 128, budget=1 << 18),
    # budget small enough that the *batch* axis must chunk too, not just query
    _case(4, 8, 1, 1, 128, 128, [512] * 4, [512] * 4, 64, 0, 128, 128, budget=1 << 17),
    _case(
        3,
        8,
        1,
        1,
        128,
        128,
        [512, 384, 256],
        [128, 96, 64],
        64,
        4,
        64,
        64,
        budget=1 << 16,
    ),
    # topk exceeds the context: every row is short and -1 padded
    _case(1, 8, 1, 1, 128, 128, [96], [96], 256, 0, 128, 1),
]

DECODE_CASES = [
    _case(4, 8, 1, 1, 128, 128, [512, 511, 300, 1024], [1] * 4, 64, 0, 128, 128),
    _case(1, 16, 4, 4, 128, 128, [777], [1], 128, 4, 64, 1),
    _case(2, 8, 1, 1, 128, 128, [50, 64], [1, 1], 128, 0, 128, 1),
    _case(8, 8, 2, 2, 128, 128, [333] * 8, [1] * 8, 32, 0, 128, 64),
]


@pytest.mark.parametrize(
    "bs,nqh,nkh,nih,hd,ihd,seqs,chunks,topk,init,local,page,budget", PREFILL_CASES
)
def test_prefill_selection_matches_reference(
    bs, nqh, nkh, nih, hd, ihd, seqs, chunks, topk, init, local, page, budget
):
    t = _build(bs, nqh, nkh, nih, hd, ihd, seqs, chunks, page)
    kwargs = {} if budget is None else {"score_budget_bytes": budget}
    got = token_select_prefill(
        idx_q=t["idx_q"],
        idx_k_cache=t["idx_k_cache"],
        req_to_token=t["req_to_token"],
        slot_ids=t["slot_ids"],
        cu_seqlens=t["cu_seqlens"],
        seq_lens=t["seq_lens"],
        prefix_lens=t["prefix_lens"],
        max_seqlen_q=max(chunks),
        max_seqlen_k=max(seqs),
        topk=topk,
        init_tokens=init,
        local_tokens=local,
        **kwargs,
    )
    ref = _ref_select_prefill(t, seqs, chunks, topk, init, local)
    assert _index_sets(got) == _index_sets(ref)


@pytest.mark.parametrize(
    "bs,nqh,nkh,nih,hd,ihd,seqs,chunks,topk,init,local,page,budget", PREFILL_CASES
)
def test_prefill_attention_matches_reference(
    bs, nqh, nkh, nih, hd, ihd, seqs, chunks, topk, init, local, page, budget
):
    t = _build(bs, nqh, nkh, nih, hd, ihd, seqs, chunks, page)
    topk_idx = _align(_ref_select_prefill(t, seqs, chunks, topk, init, local), nkh)
    q_slot_ids = torch.repeat_interleave(
        t["slot_ids"], torch.tensor(chunks, device=DEV)
    )
    got = gqa_token_sparse_attn(
        q=t["q"],
        k_cache=t["k_cache"],
        v_cache=t["v_cache"],
        req_to_token=t["req_to_token"],
        q_slot_ids=q_slot_ids,
        topk_idx=topk_idx,
        num_kv_chunks=1,
    )
    ref = _ref_attn(
        t["q"], t["k_cache"], t["v_cache"], t["req_to_token"], q_slot_ids, topk_idx
    )
    torch.testing.assert_close(got.float(), ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "bs,nqh,nkh,nih,hd,ihd,seqs,chunks,topk,init,local,page,budget", DECODE_CASES
)
def test_decode_selection_matches_reference(
    bs, nqh, nkh, nih, hd, ihd, seqs, chunks, topk, init, local, page, budget
):
    t = _build(bs, nqh, nkh, nih, hd, ihd, seqs, chunks, page)
    got = token_select_decode(
        idx_q=t["idx_q"],
        idx_k_cache=t["idx_k_cache"],
        req_to_token=t["req_to_token"],
        slot_ids=t["slot_ids"],
        seq_lens=t["seq_lens"],
        max_seqlen=max(seqs),
        topk=topk,
        init_tokens=init,
        local_tokens=local,
    )
    ref = _ref_select_decode(t, seqs, topk, init, local)
    assert _index_sets(got) == _index_sets(ref)


@pytest.mark.parametrize(
    "bs,nqh,nkh,nih,hd,ihd,seqs,chunks,topk,init,local,page,budget", DECODE_CASES
)
def test_decode_attention_matches_reference(
    bs, nqh, nkh, nih, hd, ihd, seqs, chunks, topk, init, local, page, budget
):
    t = _build(bs, nqh, nkh, nih, hd, ihd, seqs, chunks, page)
    topk_idx = _align(_ref_select_decode(t, seqs, topk, init, local), nkh)
    got = gqa_token_sparse_attn(
        q=t["q"],
        k_cache=t["k_cache"],
        v_cache=t["v_cache"],
        req_to_token=t["req_to_token"],
        q_slot_ids=t["slot_ids"],
        topk_idx=topk_idx,
    )
    ref = _ref_attn(
        t["q"], t["k_cache"], t["v_cache"], t["req_to_token"], t["slot_ids"], topk_idx
    )
    torch.testing.assert_close(got.float(), ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_kv_chunks", [1, 2, 4, 8])
def test_split_k_agrees_with_single_chunk(num_kv_chunks):
    """The decode split-K merge must reproduce the unsplit softmax exactly enough."""
    seqs, chunks = [1024, 777], [1, 1]
    t = _build(2, 8, 1, 1, 128, 128, seqs, chunks, page_size=128)
    topk_idx = _align(_ref_select_decode(t, seqs, 128, 0, 128), 1)
    ref = gqa_token_sparse_attn(
        q=t["q"],
        k_cache=t["k_cache"],
        v_cache=t["v_cache"],
        req_to_token=t["req_to_token"],
        q_slot_ids=t["slot_ids"],
        topk_idx=topk_idx,
        num_kv_chunks=1,
    )
    got = gqa_token_sparse_attn(
        q=t["q"],
        k_cache=t["k_cache"],
        v_cache=t["v_cache"],
        req_to_token=t["req_to_token"],
        q_slot_ids=t["slot_ids"],
        topk_idx=topk_idx,
        num_kv_chunks=num_kv_chunks,
    )
    torch.testing.assert_close(got.float(), ref.float(), rtol=RTOL, atol=ATOL)


def test_all_padding_rows_produce_zero_output():
    """A query that selected nothing must yield zeros, not NaN."""
    seqs, chunks = [256], [256]
    t = _build(1, 8, 1, 1, 128, 128, seqs, chunks, page_size=1)
    topk_idx = torch.full((1, 256, 32), -1, dtype=torch.int32, device=DEV)
    out = gqa_token_sparse_attn(
        q=t["q"],
        k_cache=t["k_cache"],
        v_cache=t["v_cache"],
        req_to_token=t["req_to_token"],
        q_slot_ids=torch.zeros(256, dtype=torch.int64, device=DEV),
        topk_idx=topk_idx,
        num_kv_chunks=4,
    )
    assert torch.isfinite(out.float()).all()
    assert out.abs().max().item() == 0.0


@pytest.mark.parametrize("budget", [1 << 14, 1 << 16, 1 << 18, 1 << 30])
def test_score_budget_does_not_change_selection(budget):
    """Chunking the query and batch axes is a memory strategy, not a semantic one.

    Whatever the budget forces — full batch in one pass, or one request and a
    partial query tile at a time — the selected index sets must be identical.
    """
    seqs, chunks = [512, 384, 300, 256], [512, 384, 300, 256]
    t = _build(4, 8, 1, 1, 128, 128, seqs, chunks, page_size=128)
    kwargs = dict(
        idx_q=t["idx_q"],
        idx_k_cache=t["idx_k_cache"],
        req_to_token=t["req_to_token"],
        slot_ids=t["slot_ids"],
        cu_seqlens=t["cu_seqlens"],
        seq_lens=t["seq_lens"],
        prefix_lens=t["prefix_lens"],
        max_seqlen_q=max(chunks),
        max_seqlen_k=max(seqs),
        topk=64,
        init_tokens=0,
        local_tokens=128,
    )
    got = token_select_prefill(**kwargs, score_budget_bytes=budget)
    ref = _ref_select_prefill(t, seqs, chunks, 64, 0, 128)
    assert _index_sets(got) == _index_sets(ref)


@pytest.mark.parametrize("budget", [1 << 10, 1 << 12, 1 << 14, 1 << 30])
def test_decode_key_window_does_not_change_selection(budget):
    """Decode windows the *key* axis; the merge of per-window top-k must be exact.

    Soundness rests on a derivation: a position in the global top-k is
    necessarily in the top-k of the single window containing it, so taking
    ``topk`` per window can miss nothing. A window plan or merge that broke that
    property — selecting fewer than ``topk`` per window, dropping the -inf fill,
    or mixing up window-relative and absolute positions — would show up here and
    nowhere else, since every other decode case fits in one window.
    """
    seqs = [512, 384, 300, 256]
    t = _build(4, 8, 1, 1, 128, 128, seqs, [1] * 4, page_size=128)
    got = token_select_decode(
        idx_q=t["idx_q"],
        idx_k_cache=t["idx_k_cache"],
        req_to_token=t["req_to_token"],
        slot_ids=t["slot_ids"],
        seq_lens=t["seq_lens"],
        max_seqlen=max(seqs),
        topk=64,
        init_tokens=0,
        local_tokens=128,
        score_budget_bytes=budget,
    )
    ref = _ref_select_decode(t, seqs, 64, 0, 128)
    assert _index_sets(got) == _index_sets(ref)


@pytest.mark.parametrize("seqs", [[2048, 1536], [2050, 1499]])
def test_wide_row_selection_matches_reference(seqs):
    """Wide rows go through FlashInfer's selector rather than torch.topk.

    The two are meant to be interchangeable — same exact answer, different
    speed — so availability must never change the result. A build where
    FlashInfer is present but disagrees (or where the 2D reshape the call needs
    mangles the row identity) shows up here and nowhere else, since every other
    selection case is narrow enough to be uninteresting.
    """
    t = _build(len(seqs), 8, 1, 1, 128, 128, seqs, [1] * len(seqs), page_size=128)
    kwargs = dict(
        idx_q=t["idx_q"],
        idx_k_cache=t["idx_k_cache"],
        req_to_token=t["req_to_token"],
        slot_ids=t["slot_ids"],
        seq_lens=t["seq_lens"],
        max_seqlen=max(seqs),
        topk=8,
        init_tokens=0,
        local_tokens=128,
    )
    got = token_select_decode(**kwargs)
    assert _index_sets(got) == _index_sets(_ref_select_decode(t, seqs, 8, 0, 128))


def test_full_budget_matches_dense_attention():
    """With topk >= context the selection is a no-op and the result is dense attention."""
    seq_len = 192
    t = _build(1, 8, 1, 1, 128, 128, [seq_len], [seq_len], page_size=64)
    topk_idx = token_select_prefill(
        idx_q=t["idx_q"],
        idx_k_cache=t["idx_k_cache"],
        req_to_token=t["req_to_token"],
        slot_ids=t["slot_ids"],
        cu_seqlens=t["cu_seqlens"],
        seq_lens=t["seq_lens"],
        prefix_lens=t["prefix_lens"],
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        topk=seq_len,
        init_tokens=0,
        local_tokens=0,
    )
    q_slot_ids = torch.zeros(seq_len, dtype=torch.int64, device=DEV)
    got = gqa_token_sparse_attn(
        q=t["q"],
        k_cache=t["k_cache"],
        v_cache=t["v_cache"],
        req_to_token=t["req_to_token"],
        q_slot_ids=q_slot_ids,
        topk_idx=topk_idx,
        num_kv_chunks=1,
    )

    slots = t["req_to_token"][0, :seq_len].long()
    k = t["k_cache"][slots, 0].float()
    v = t["v_cache"][slots, 0].float()
    logits = torch.einsum("nhd,ld->nhl", t["q"].float(), k) * (128**-0.5)
    pos = torch.arange(seq_len, device=DEV)
    logits = logits.masked_fill(
        (pos[:, None] < pos[None, :])[:, None, :], float("-inf")
    )
    ref = torch.einsum("nhl,ld->nhd", torch.softmax(logits, dim=-1), v)
    torch.testing.assert_close(got.float(), ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# dense variant: no indexer, no top-k, full causal attention
# ---------------------------------------------------------------------------


def _ref_dense_prefill(t, seq_len, chunk, num_q_heads, num_kv_heads, head_dim):
    slots = t["req_to_token"][0, :seq_len].long()
    group = num_q_heads // num_kv_heads
    k = t["k_cache"][slots].float().repeat_interleave(group, dim=1)
    v = t["v_cache"][slots].float().repeat_interleave(group, dim=1)
    logits = torch.einsum("nhd,lhd->nhl", t["q"].float(), k) * (head_dim**-0.5)
    abs_q = (seq_len - chunk) + torch.arange(chunk, device=DEV)
    pos = torch.arange(seq_len, device=DEV)
    logits = logits.masked_fill(
        (abs_q[:, None] < pos[None, :])[:, None, :], float("-inf")
    )
    return torch.einsum("nhl,lhd->nhd", torch.softmax(logits, dim=-1), v)


DENSE_CASES = [
    # (batch, q heads, kv heads, head_dim, seq_len, chunk, page_size)
    pytest.param(1, 8, 1, 128, 512, 512, 1, id="bs1_q8kv1_full_page1"),
    pytest.param(1, 8, 1, 128, 512, 128, 128, id="bs1_q8kv1_chunkedprefill_page128"),
    pytest.param(1, 16, 4, 128, 768, 768, 64, id="bs1_q16kv4_full_page64"),
    pytest.param(1, 8, 2, 64, 640, 640, 32, id="bs1_q8kv2_d64_page32"),
]


@pytest.mark.parametrize("bs,nqh,nkh,hd,seq_len,chunk,page", DENSE_CASES)
def test_dense_prefill_matches_causal_reference(bs, nqh, nkh, hd, seq_len, chunk, page):
    from sglang.srt.layers.attention.minimax_sparse_ops.minimax_dense import (
        minimax_dense_prefill,
    )

    t = _build(bs, nqh, nkh, nkh, hd, hd, [seq_len], [chunk], page)
    prefix = seq_len - chunk
    ext_slots = t["req_to_token"][:, prefix:seq_len].reshape(-1).long()
    _, out = minimax_dense_prefill(
        t["q"],
        t["k_cache"],
        t["v_cache"],
        None,
        None,
        None,
        None,
        None,
        t["req_to_token"],
        t["slot_ids"],
        t["cu_seqlens"],
        t["seq_lens"],
        t["prefix_lens"],
        chunk,
        k_extend=t["k_cache"][ext_slots],
        v_extend=t["v_cache"][ext_slots],
        prefix_lens_cpu=[prefix],
    )
    ref = _ref_dense_prefill(t, seq_len, chunk, nqh, nkh, hd)
    torch.testing.assert_close(out.float(), ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("bs,nqh,nkh,hd,seq_len,chunk,page", DENSE_CASES)
def test_dense_decode_matches_causal_reference(bs, nqh, nkh, hd, seq_len, chunk, page):
    from sglang.srt.layers.attention.minimax_sparse_ops.minimax_dense import (
        minimax_dense_decode,
    )

    t = _build(bs, nqh, nkh, nkh, hd, hd, [seq_len], [1], page)
    _, out = minimax_dense_decode(
        t["q"],
        None,
        t["k_cache"],
        t["v_cache"],
        None,
        None,
        None,
        None,
        t["req_to_token"],
        t["slot_ids"],
        t["seq_lens"],
        total_kv=seq_len,
    )
    slots = t["req_to_token"][0, :seq_len].long()
    group = nqh // nkh
    k = t["k_cache"][slots].float().repeat_interleave(group, dim=1)
    v = t["v_cache"][slots].float().repeat_interleave(group, dim=1)
    logits = torch.einsum("hd,lhd->hl", t["q"][0].float(), k) * (hd**-0.5)
    ref = torch.einsum("hl,lhd->hd", torch.softmax(logits, dim=-1), v).unsqueeze(0)
    torch.testing.assert_close(out.float(), ref, rtol=RTOL, atol=ATOL)


def test_dense_equals_sparse_when_budget_covers_context():
    """Full-budget token-sparse must reproduce dense attention on the same inputs."""
    from sglang.srt.layers.attention.minimax_sparse_ops.minimax_dense import (
        minimax_dense_decode,
    )

    seq_len = 320
    t = _build(1, 8, 1, 1, 128, 128, [seq_len], [1], page_size=64)
    topk_idx = _align(_ref_select_decode(t, [seq_len], seq_len, 0, 0), 1)
    sparse = gqa_token_sparse_attn(
        q=t["q"],
        k_cache=t["k_cache"],
        v_cache=t["v_cache"],
        req_to_token=t["req_to_token"],
        q_slot_ids=t["slot_ids"],
        topk_idx=topk_idx,
    )
    _, dense = minimax_dense_decode(
        t["q"],
        None,
        t["k_cache"],
        t["v_cache"],
        None,
        None,
        None,
        None,
        t["req_to_token"],
        t["slot_ids"],
        t["seq_lens"],
        total_kv=seq_len,
    )
    torch.testing.assert_close(sparse.float(), dense.float(), rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
