"""Paged step-3 kernels vs the originals.

``flash_{prefill,decode}_with_gqa_share_sparse_paged`` differ from the originals
only in how a selected block's physical slots are resolved. When page and block
sizes divide one another a block covers whole page-aligned runs of consecutive
slots, so one ``req_to_token`` lookup per run replaces the ``block_size``-entry
gather. The math is untouched.

Two separate claims are tested, because they hold to different tolerances:

* **Same autotune config -> bit-identical.** The slot arithmetic is the only
  difference, so with both kernels pinned to one config the outputs must be
  equal element for element. This is what actually validates the fast path.
* **As dispatched -> equal within bf16 rounding.** The two kernels autotune
  independently over the same config space, and the winner is decided by a
  timing race: at some shapes the original lands on ``num_warps=4`` while the
  paged copy lands on ``num_warps=8``. Different warp counts reassociate the
  split-K accumulation, which moves a handful of elements by ~1 ulp. That is
  autotune nondeterminism, not the paging change -- re-tuning the original
  kernel produces the same spread.

Every fallback condition is covered alongside the fast path, because the kernel
picks between them on its own and the production dispatch
(``SGLANG_OPT_USE_MINIMAX_SPARSE_PAGED_PREFILL``, default on) routes every page
size through the paged entry point. Both phases are covered here regardless of
which one ships enabled, since the decode flag exists and can be flipped on.
"""

import contextlib

import pytest
import torch
import triton

from sglang.kernels.ops.attention.minimax_sparse.common.utils import get_cu_seqblocks
from sglang.kernels.ops.attention.minimax_sparse.decode import topk_sparse as decode_mod
from sglang.kernels.ops.attention.minimax_sparse.decode import (
    topk_sparse_paged as decode_paged_mod,
)
from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)
from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse_paged import (
    MIN_PAGE_SPAN,
    flash_decode_with_gqa_share_sparse_paged,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill import (
    topk_sparse as prefill_mod,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill import (
    topk_sparse_paged as prefill_paged_mod,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse import (
    flash_prefill_with_gqa_share_sparse,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse_paged import (
    flash_prefill_with_gqa_share_sparse_paged,
)

DEVICE = "cuda"
NUM_Q_HEADS = 16
NUM_KV_HEADS = 1
HEAD_DIM = 128
DTYPE = torch.bfloat16
CONTEXT_LEN = 4096

# One bf16 ulp at the output magnitudes these shapes produce. Only reachable
# when the two kernels tune to different configs; see the module docstring.
ULP_ATOL = 1e-3

# page_size -> does the fast path engage at block_size 128? The fallbacks matter
# as much as the fast path: production sends every page size through here.
PAGE_CASES = [
    pytest.param(128, True, id="page128_eq_block"),
    pytest.param(256, True, id="page256_gt_block"),
    pytest.param(64, True, id="page64_two_runs"),
    pytest.param(16, True, id="page16_min_span"),
    pytest.param(8, False, id="page8_below_min_span"),
    pytest.param(48, False, id="page48_indivisible"),
    pytest.param(1, False, id="page1_per_token_gather"),
]


def _autotuner(kernel):
    """The ``triton.autotune`` node under a stack of decorators."""
    node = kernel
    while node is not None and not getattr(node, "configs", None):
        node = getattr(node, "fn", None)
    if node is None:
        raise RuntimeError("no autotuner found on kernel")
    return node


@contextlib.contextmanager
def pinned_autotune(*kernels, num_warps=4, num_stages=3):
    """Force every kernel onto one config so outputs are directly comparable.

    Both kernels autotune over ``Config({}, num_warps, num_stages)`` only -- no
    meta-parameters -- so a single config is always valid for either.
    """
    tuners = [_autotuner(k) for k in kernels]
    saved = [(t, t.configs, dict(t.cache)) for t in tuners]
    for tuner in tuners:
        tuner.configs = [triton.Config({}, num_warps=num_warps, num_stages=num_stages)]
        tuner.cache.clear()
    try:
        yield
    finally:
        for tuner, configs, cache in saved:
            tuner.configs = configs
            tuner.cache.clear()
            tuner.cache.update(cache)


def build_page_table(batch_size, context_len, page_size, device):
    """Tokens contiguous within a page, pages scattered across the pool.

    This is how sglang's paged allocator lays a request out, and it is what the
    fast path relies on: consecutive positions inside one page are consecutive
    slots, while nothing is guaranteed across pages.
    """
    pages_per_req = (context_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_req
    max_slots = total_pages * page_size
    perm = torch.randperm(total_pages, device=device)
    within = torch.arange(context_len, device=device) % page_size
    page_of_tok = torch.arange(context_len, device=device) // page_size
    phys_page = perm.view(batch_size, pages_per_req)[:, page_of_tok]
    req_to_token = (phys_page * page_size + within).to(torch.int32)
    return req_to_token, max_slots


def _randn(*shape, generator):
    return torch.randn(*shape, dtype=DTYPE, device=DEVICE, generator=generator)


def _topk_idx(rows, num_blocks, topk, generator):
    """Random selections, clamped so no block id runs past the context."""
    u = torch.rand(NUM_KV_HEADS, rows, topk, device=DEVICE, generator=generator)
    return (u * num_blocks).to(torch.int32).clamp_(max=num_blocks - 1)


def _causal_topk_idx(abs_pos, topk, block_size, generator):
    """Selections a query may legally attend to: its own block or an earlier one.

    Rows whose every selection lies in the future would softmax over an all-masked
    row and produce NaN, which no reference can be compared against -- and NaN is
    never ``torch.equal`` to itself, so it would mask a real divergence too.
    """
    rows = abs_pos.numel()
    u = torch.rand(NUM_KV_HEADS, rows, topk, device=DEVICE, generator=generator)
    highest = ((abs_pos + block_size) // block_size).to(torch.float32)
    return (u * highest.view(1, rows, 1)).to(torch.int32)


def build_decode_inputs(batch_size, page_size, block_size, topk, generator):
    req_to_token, max_slots = build_page_table(
        batch_size, CONTEXT_LEN, page_size, DEVICE
    )
    num_blocks = (CONTEXT_LEN + block_size - 1) // block_size
    return dict(
        q=_randn(batch_size, NUM_Q_HEADS, HEAD_DIM, generator=generator),
        sink=None,
        k_cache=_randn(max_slots, NUM_KV_HEADS, HEAD_DIM, generator=generator),
        v_cache=_randn(max_slots, NUM_KV_HEADS, HEAD_DIM, generator=generator),
        req_to_token=req_to_token,
        seq_lens=torch.full(
            (batch_size,), CONTEXT_LEN, dtype=torch.int32, device=DEVICE
        ),
        slot_ids=torch.arange(batch_size, dtype=torch.int64, device=DEVICE),
        block_size=block_size,
        topk_idx=_topk_idx(batch_size, num_blocks, topk, generator),
    )


def build_prefill_inputs(batch_size, page_size, block_size, topk, chunk_len, generator):
    req_to_token, max_slots = build_page_table(
        batch_size, CONTEXT_LEN, page_size, DEVICE
    )
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * chunk_len, chunk_len, dtype=torch.int32, device=DEVICE
    )
    cu_seqblocks_q, max_seqblock_q, _, _, _, _ = get_cu_seqblocks(
        cu_seqlens, chunk_len, 1, block_size
    )
    rows = int(cu_seqblocks_q[-1].item())
    # block_size_q is 1, so one row per query token: the chunk sits at the end of
    # the context, after a prefix of CONTEXT_LEN - chunk_len tokens.
    abs_pos = (torch.arange(chunk_len, device=DEVICE) + CONTEXT_LEN - chunk_len).repeat(
        batch_size
    )
    assert abs_pos.numel() == rows
    return dict(
        q=_randn(batch_size * chunk_len, NUM_Q_HEADS, HEAD_DIM, generator=generator),
        k_cache=_randn(max_slots, NUM_KV_HEADS, HEAD_DIM, generator=generator),
        v_cache=_randn(max_slots, NUM_KV_HEADS, HEAD_DIM, generator=generator),
        sink=None,
        req_to_token=req_to_token,
        slot_ids=torch.arange(batch_size, dtype=torch.int64, device=DEVICE),
        topk_idx=_causal_topk_idx(abs_pos, topk, block_size, generator),
        block_size_q=1,
        block_size_k=block_size,
        cu_seqlens=cu_seqlens,
        seq_lens=torch.full(
            (batch_size,), CONTEXT_LEN, dtype=torch.int32, device=DEVICE
        ),
        prefix_lens=torch.full(
            (batch_size,), CONTEXT_LEN - chunk_len, dtype=torch.int32, device=DEVICE
        ),
        max_seqlen_q=chunk_len,
        cu_seqblocks_q=cu_seqblocks_q,
        max_seqblock_q=max_seqblock_q,
    )


DECODE_KERNELS = (
    decode_mod._gqa_share_sparse_decode_kernel,
    decode_paged_mod._gqa_share_sparse_decode_kernel,
)
PREFILL_KERNELS = (
    prefill_mod._gqa_share_sparse_fwd_kernel,
    prefill_paged_mod._gqa_share_sparse_fwd_kernel,
)


@pytest.mark.parametrize("page_size,fast_path", PAGE_CASES)
@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_decode_paged_is_bit_identical(page_size, fast_path, batch_size):
    """Same config, so any difference would be the slot resolution itself."""
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    kwargs = build_decode_inputs(
        batch_size, page_size=page_size, block_size=128, topk=16, generator=gen
    )
    with pinned_autotune(*DECODE_KERNELS):
        o_orig = flash_decode_with_gqa_share_sparse(**kwargs)
        o_paged = flash_decode_with_gqa_share_sparse_paged(
            **kwargs, page_size=page_size
        )

    diff = (o_orig.float() - o_paged.float()).abs()
    assert torch.equal(o_orig, o_paged), (
        f"paged decode diverged at page_size={page_size}: "
        f"{int((diff > 0).sum())} of {diff.numel()} elements, max {diff.max().item()}"
    )


@pytest.mark.parametrize("page_size,fast_path", PAGE_CASES)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_prefill_paged_is_bit_identical(page_size, fast_path, batch_size):
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    kwargs = build_prefill_inputs(
        batch_size,
        page_size=page_size,
        block_size=128,
        topk=16,
        chunk_len=512,
        generator=gen,
    )
    with pinned_autotune(*PREFILL_KERNELS):
        o_orig = flash_prefill_with_gqa_share_sparse(**kwargs)
        o_paged = flash_prefill_with_gqa_share_sparse_paged(
            **kwargs, page_size=page_size
        )

    diff = (o_orig.float() - o_paged.float()).abs()
    assert torch.equal(o_orig, o_paged), (
        f"paged prefill diverged at page_size={page_size}: "
        f"{int((diff > 0).sum())} of {diff.numel()} elements, max {diff.max().item()}"
    )


@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
def test_decode_paged_across_block_sizes(block_size):
    """page_size 128 held fixed while block_size varies.

    Every one of these divides 128, so all take the fast path -- block == page
    resolves in one lookup, block < page in one lookup per run.
    """
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    kwargs = build_decode_inputs(
        2, page_size=128, block_size=block_size, topk=16, generator=gen
    )
    with pinned_autotune(*DECODE_KERNELS):
        o_orig = flash_decode_with_gqa_share_sparse(**kwargs)
        o_paged = flash_decode_with_gqa_share_sparse_paged(**kwargs, page_size=128)

    assert torch.equal(o_orig, o_paged)


@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_decode_paged_as_dispatched(batch_size):
    """No pinning: exactly how the production dispatch calls them.

    Independent autotuning may pick different configs, so this asserts bf16
    rounding rather than equality. See the module docstring.
    """
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    kwargs = build_decode_inputs(
        batch_size, page_size=128, block_size=128, topk=16, generator=gen
    )
    o_orig = flash_decode_with_gqa_share_sparse(**kwargs)
    o_paged = flash_decode_with_gqa_share_sparse_paged(**kwargs, page_size=128)

    torch.testing.assert_close(o_paged.float(), o_orig.float(), rtol=0, atol=ULP_ATOL)


def test_min_page_span_agrees_across_phases():
    """Prefill and decode must stop taking the fast path at the same width."""
    assert MIN_PAGE_SPAN == prefill_paged_mod.MIN_PAGE_SPAN
