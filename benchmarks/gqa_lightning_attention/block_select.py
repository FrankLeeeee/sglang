"""MSA-style block selector over the Lightning token scores.

Each 128-token block is scored by the maximum of its token scores (computed in
the score kernel, so per-token scores are never written). The top 16 blocks are
kept per query and KV group, with the query's own block always included, and
are expanded to the 16 * 128 = 2048 causal token positions that the shared
token-sparse attention kernel consumes.
"""

import flashinfer
import torch
from attention import LightningIndexer
from kernels import lightning_scores
from stages import stage

BLOCK_SIZE = 128


class BlockIndexer(LightningIndexer):
    def __init__(self, cfg, top_blocks=16):
        super().__init__(cfg)
        self.top_blocks = top_blocks

    def score_width(self, state):
        return state.length // BLOCK_SIZE

    def select(self, q, k_cache, w, state, start, end, top_k):
        groups, rows = self.cfg.num_k_heads, end - start
        with stage("score"):
            scores = lightning_scores(
                q[start:end],
                k_cache,
                w[start:end],
                state.req_to_token,
                state.request_ids[start:end],
                state.positions[start:end],
                state.length,
                groups,
                query_tile=state.score_query_tile,
                block_max=True,
                num_warps=state.score_warps,
            )
        with stage("select"):
            # The query's own block is partially future-dependent: always keep it.
            current = (state.positions[start:end] // BLOCK_SIZE).long()
            scores.scatter_(2, current[None, :, None].expand(groups, -1, 1), float("inf"))
            count = min(self.top_blocks, scores.shape[-1])
            vals, blocks = flashinfer.top_k(
                scores.flatten(0, 1), count, deterministic=True, tie_break=1
            )
        with stage("expand"):
            return self._tokens(vals, blocks, state.positions[start:end], groups, rows)

    @staticmethod
    def _tokens(vals, blocks, positions, groups, rows):
        """Block ids [G*M, B] -> causal request-local token ids [G, M, B * 128]."""
        blocks = torch.where(vals != -float("inf"), blocks, -1).view(groups, rows, -1)
        offsets = torch.arange(BLOCK_SIZE, device=blocks.device, dtype=blocks.dtype)
        tokens = blocks[..., None] * BLOCK_SIZE + offsets
        valid = (blocks[..., None] >= 0) & (tokens <= positions[None, :, None, None])
        return torch.where(valid, tokens, -1).flatten(2).int()
