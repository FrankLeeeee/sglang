"""Mean-pooled block shortlist followed by exact token scores within candidates.

The shortlist is approximate: it need not contain the global token top-k.
The current block is forced into the shortlist without consulting its pooled
key, so newly written future tokens cannot affect an earlier query's selection.
"""

import flashinfer
import torch
import triton
import triton.language as tl
from attention import LightningIndexer
from kernels import lightning_scores


@triton.jit
def _pool(K, POOLED, IDS, IK: tl.constexpr, BS: tl.constexpr, ALL: tl.constexpr):
    page, group = tl.program_id(0), tl.program_id(1)
    if not ALL:
        page = tl.load(IDS + page)
    n = tl.arange(0, BS)
    d = tl.arange(0, 64)
    k = tl.load(K + ((page * BS + n[:, None]) * IK + group) * 64 + d[None, :])
    mean = tl.sum(k.to(tl.float32), axis=0) / BS
    tl.store(POOLED + (page * IK + group) * 64 + d, mean)


def prepare_block_cache(state, block_size=128):
    if state.page_size != block_size or state.length % block_size:
        raise ValueError("Two-stage selection requires aligned block/page size 128")
    state.block_size = block_size
    state.block_ik = torch.empty(
        state.ik.shape[0] // block_size,
        state.ik.shape[1],
        64,
        device=state.ik.device,
        dtype=state.ik.dtype,
    )
    state.block_table = (state.req_to_token[:, ::block_size] // block_size).contiguous()
    state.block_positions = (state.positions // block_size).contiguous()
    state.updated_blocks = (state.write_slots // block_size).unique().contiguous()
    _pool[(state.block_ik.shape[0], state.ik.shape[1])](
        state.ik,
        state.block_ik,
        state.updated_blocks,
        state.ik.shape[1],
        block_size,
        True,
    )


@triton.jit
def _candidate_score(
    Q,
    K,
    W,
    R2T,
    REQ,
    POS,
    BLOCKS,
    OUT,
    M: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
    J: tl.constexpr,
    IK: tl.constexpr,
    H: tl.constexpr,
    BS: tl.constexpr,
):
    row, group, candidate = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    block = tl.load(BLOCKS + (group * M + row) * C + candidate)
    pos = tl.load(POS + row)
    request = tl.load(REQ + row)
    n = block * BS + tl.arange(0, BS)
    valid = (block >= 0) & (n <= pos) & (n < N)
    slots = tl.load(R2T + request * N + n, valid, 0).to(tl.int64)
    kg = 0 if IK == 1 else group
    d = tl.arange(0, 64)
    if J == 1:
        q = tl.load(Q + (row * G + group) * 64 + d).to(tl.float32)
        k = tl.load(
            K + (slots[:, None] * IK + kg) * 64 + d[None, :], valid[:, None], 0
        ).to(tl.float32)
        gate = tl.load(W + row * G + group).to(tl.float32)
        scores = tl.maximum(tl.sum(k * q[None, :], axis=1), 0.0) * gate
    else:
        h = tl.arange(0, H)
        q = tl.load(
            Q + (row * G * J + group * J + h[:, None]) * 64 + d[None, :],
            h[:, None] < J,
            0,
        )
        k = tl.load(K + (slots[None, :] * IK + kg) * 64 + d[:, None], valid[None, :], 0)
        dots = tl.maximum(tl.dot(q, k), 0.0)
        gates = tl.load(W + row * G * J + group * J + h, h < J, 0).to(tl.float32)
        scores = tl.sum(dots * gates[:, None], axis=0)
    scores = tl.where(valid, scores, -float("inf"))
    offsets = candidate * BS + tl.arange(0, BS)
    tl.store(OUT + (group * M + row) * (C * BS) + offsets, scores)


class TwoStageIndexer(LightningIndexer):
    def __init__(self, cfg, candidate_blocks=64, block_size=128):
        super().__init__(cfg)
        if candidate_blocks < 1 or block_size != 128:
            raise ValueError("Use positive candidate_blocks and block_size=128")
        self.candidate_blocks = candidate_blocks
        self.block_size = block_size

    def score_width(self, state):
        return min(self.candidate_blocks * self.block_size, state.length)

    def update_cache(self, state):
        _pool[(state.updated_blocks.numel(), state.ik.shape[1])](
            state.ik,
            state.block_ik,
            state.updated_blocks,
            state.ik.shape[1],
            self.block_size,
            False,
        )

    def shortlist(self, q, w, state, start, end):
        groups = self.cfg.num_k_heads
        scores = lightning_scores(
            q[start:end],
            state.block_ik,
            w[start:end],
            state.block_table,
            state.request_ids[start:end],
            state.block_positions[start:end],
            state.length // self.block_size,
            groups,
            state.score_query_tile,
        )
        # Never use a partially filled block's future-dependent mean to rank it.
        current = (
            state.block_positions[start:end].long()[None, :, None].expand(groups, -1, 1)
        )
        scores.scatter_(2, current, float("inf"))
        count = min(self.candidate_blocks, scores.shape[-1])
        vals, idx = flashinfer.top_k(
            scores.flatten(0, 1), count, deterministic=True, tie_break=1
        )
        idx = torch.where(vals != -float("inf"), idx, -1).to(torch.int32)
        return idx.view(groups, end - start, count)

    def candidate_scores(self, q, k_cache, w, state, start, end, blocks):
        m = end - start
        groups = self.cfg.num_k_heads
        j = self.cfg.indexer_q_heads // groups
        count = blocks.shape[-1]
        scores = torch.empty(
            groups, m, count * self.block_size, device=q.device, dtype=torch.float32
        )
        _candidate_score[(m, groups, count)](
            q[start:end],
            k_cache,
            w[start:end],
            state.req_to_token,
            state.request_ids[start:end],
            state.positions[start:end],
            blocks,
            scores,
            m,
            count,
            state.length,
            groups,
            j,
            k_cache.shape[1],
            max(16, triton.next_power_of_2(j)),
            self.block_size,
            num_warps=4,
        )
        return scores

    def select(self, q, k_cache, w, state, start, end, top_k):
        if self.score_width(state) < min(top_k, state.length):
            raise ValueError("Candidate token budget must cover final top-k")
        blocks = self.shortlist(q, w, state, start, end)
        scores = self.candidate_scores(q, k_cache, w, state, start, end, blocks)
        width = min(top_k, state.length)
        vals, ordinal = flashinfer.top_k(
            scores.flatten(0, 1), width, deterministic=True, tie_break=1
        )
        ordinal = ordinal.view(self.cfg.num_k_heads, end - start, width).long()
        block = blocks.gather(2, ordinal // self.block_size)
        indices = block * self.block_size + ordinal % self.block_size
        return torch.where(torch.isfinite(vals.view_as(indices)), indices, -1).int()
