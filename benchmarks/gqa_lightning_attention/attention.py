"""Single-layer paged dense / token-sparse GQA benchmark implementation."""

from dataclasses import dataclass

import flashinfer
import torch
import torch.nn.functional as F
from kernels import cache_write, lightning_scores, rms_norm_per_head

from sglang.kernels.ops.attention.minimax_sparse.token.sparse_attn import (
    gqa_token_sparse_attn,
)


@dataclass(frozen=True)
class Config:
    num_q_heads: int
    num_k_heads: int
    head_dim: int = 128
    hidden_dim: int = 8192
    indexer_head_dim: int = 64
    indexer_kv_heads: int | None = None
    indexer_q_heads: int | None = None

    def __post_init__(self):
        if self.num_k_heads < 1 or self.num_q_heads % self.num_k_heads:
            raise ValueError("Main KV heads must be positive and divide main Q heads")
        if self.indexer_q_heads is None:
            object.__setattr__(self, "indexer_q_heads", self.num_q_heads)
        if self.indexer_q_heads < 1 or self.indexer_q_heads % self.num_k_heads:
            raise ValueError(
                "Indexer query heads must be a positive multiple of main KV heads"
            )
        if self.indexer_kv_heads is None:
            object.__setattr__(self, "indexer_kv_heads", self.num_k_heads)
        if self.indexer_kv_heads not in (1, self.num_k_heads):
            raise ValueError("Indexer KV heads must be 1 or match attention KV heads")


CONFIGS = {1: Config(48, 4), 2: Config(64, 8)}


class LightningIndexer:
    def __init__(self, cfg):
        self.cfg = cfg

        def weight(out):
            return (
                torch.randn(out, cfg.hidden_dim, device="cuda", dtype=torch.bfloat16)
                / cfg.hidden_dim**0.5
            )

        self.wq = weight(cfg.indexer_q_heads * cfg.indexer_head_dim)
        self.wk = weight(cfg.indexer_kv_heads * cfg.indexer_head_dim)
        self.ww = weight(cfg.indexer_q_heads)

    def project(self, x, x_kv):
        c = self.cfg
        q = F.linear(x, self.wq).view(-1, c.indexer_q_heads, 64)
        k = F.linear(x_kv, self.wk).view(-1, c.indexer_kv_heads, 64)
        w = F.relu(F.linear(x, self.ww))
        return q, k, w

    def select(self, q, k_cache, w, state, start, end, top_k):
        scores = lightning_scores(
            q[start:end],
            k_cache,
            w[start:end],
            state.req_to_token,
            state.request_ids[start:end],
            state.positions[start:end],
            state.length,
            self.cfg.num_k_heads,
            query_tile=state.score_query_tile,
        )
        width = min(top_k, state.length)
        vals, idx = flashinfer.top_k(
            scores.flatten(0, 1), width, deterministic=True, tie_break=1
        )
        # Keep -1 untouched: adding a packed document offset to it is a bug in
        # the supplied pseudocode. The attention kernel expects local positions.
        idx = torch.where(torch.isfinite(vals), idx, -1).to(torch.int32)
        return idx.view(self.cfg.num_k_heads, end - start, width)


class PagedState:
    def __init__(
        self,
        cfg,
        batch,
        length,
        q_per_request,
        page_size=128,
        sparse=False,
        phase="prefill",
    ):
        assert length % page_size == 0
        assert q_per_request == 1 or q_per_request % 4 == 0
        self.cfg, self.batch, self.length = cfg, batch, length
        self.q_per_request = q_per_request
        # H200 tuning: 16-query tiles help the eight-head indexer groups.
        # Twelve-head groups were fastest at four queries per tile. Require
        # request boundaries to align with tiles so a tile never mixes caches.
        self.score_query_tile = (
            1
            if q_per_request == 1
            else (
                16
                if cfg.indexer_q_heads // cfg.num_k_heads in (1, 8)
                and q_per_request % 16 == 0
                else 4
            )
        )
        self.phase, self.page_size = phase, page_size
        pages_per_req = length // page_size
        pages = torch.randperm(batch * pages_per_req, device="cuda", dtype=torch.int32)
        self.pages = pages
        offsets = torch.arange(length, device="cuda")
        self.req_to_token = (
            pages.view(batch, -1)[:, offsets // page_size] * page_size
            + offsets % page_size
        ).int()
        self.request_ids = torch.arange(
            batch, device="cuda", dtype=torch.int32
        ).repeat_interleave(q_per_request)
        self.positions = torch.arange(
            length - q_per_request, length, device="cuda", dtype=torch.int32
        ).repeat(batch)
        self.write_slots = self.req_to_token[
            self.request_ids.long(), self.positions.long()
        ].contiguous()
        shape = (batch * length, cfg.num_k_heads, cfg.head_dim)
        self.k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        self.v = torch.randn_like(self.k)
        self.ik = (
            torch.randn(
                batch * length,
                cfg.indexer_kv_heads,
                64,
                device="cuda",
                dtype=torch.bfloat16,
            )
            if sparse
            else None
        )
        self.wrapper = None
        if not sparse:
            # FlashInfer tensor-core decode reserves larger split-K workspaces
            # at the 48:1 / 64:1 MQA ratios, even for a short context.
            workspace_mib = 1024 if cfg.num_k_heads == 1 else 128
            self.workspace = torch.empty(
                workspace_mib * 1024 * 1024, device="cuda", dtype=torch.uint8
            )
            indptr = (
                torch.arange(batch + 1, device="cuda", dtype=torch.int32)
                * pages_per_req
            )
            last = torch.full((batch,), page_size, device="cuda", dtype=torch.int32)
            if phase == "decode":
                self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace,
                    use_cuda_graph=True,
                    use_tensor_cores=True,
                    paged_kv_indptr_buffer=indptr,
                    paged_kv_indices_buffer=pages,
                    paged_kv_last_page_len_buffer=last,
                )
                self.wrapper.plan(
                    indptr,
                    pages,
                    last,
                    cfg.num_q_heads,
                    cfg.num_k_heads,
                    cfg.head_dim,
                    page_size,
                    q_data_type=torch.bfloat16,
                    kv_data_type=torch.bfloat16,
                )
            else:
                self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace, backend="fa2"
                )
                qptr = (
                    torch.arange(batch + 1, device="cuda", dtype=torch.int32)
                    * q_per_request
                )
                self.wrapper.plan(
                    qptr,
                    indptr,
                    pages,
                    last,
                    cfg.num_q_heads,
                    cfg.num_k_heads,
                    cfg.head_dim,
                    page_size,
                    causal=True,
                    q_data_type=torch.bfloat16,
                    kv_data_type=torch.bfloat16,
                )

    def dense(self, q):
        c, p = self.cfg, self.page_size
        return self.wrapper.run(
            q,
            (
                self.k.view(-1, p, c.num_k_heads, c.head_dim),
                self.v.view(-1, p, c.num_k_heads, c.head_dim),
            ),
        )


class Attention:
    """Turn sparsity on with ``Attention(cfg, sparse=True)``.

    Forward includes Q/K/V projections and RMSNorm, cache append and attention.
    No output projection, RoPE, residual, MLP, or learned norm gain was requested.
    """

    def __init__(self, cfg, sparse=False, top_k=2048, score_budget_mib=256):
        self.cfg, self.sparse, self.top_k = cfg, sparse, top_k
        self.score_budget = score_budget_mib * 1024**2

        def weight(out):
            return (
                torch.randn(out, cfg.hidden_dim, device="cuda", dtype=torch.bfloat16)
                / cfg.hidden_dim**0.5
            )

        self.wq = weight(cfg.num_q_heads * cfg.head_dim)
        self.wk = weight(cfg.num_k_heads * cfg.head_dim)
        self.wv = weight(cfg.num_k_heads * cfg.head_dim)
        self.indexer = LightningIndexer(cfg) if sparse else None

    @torch.no_grad()
    def forward(self, x, state, x_kv=None):
        x_kv = x if x_kv is None else x_kv
        c = self.cfg
        q = rms_norm_per_head(F.linear(x, self.wq).view(-1, c.num_q_heads, c.head_dim))
        k = rms_norm_per_head(
            F.linear(x_kv, self.wk).view(-1, c.num_k_heads, c.head_dim)
        )
        v = rms_norm_per_head(
            F.linear(x_kv, self.wv).view(-1, c.num_k_heads, c.head_dim)
        )
        cache_write(k, state.k, state.write_slots)
        cache_write(v, state.v, state.write_slots)
        if not self.sparse:
            return state.dense(q)
        qi, ki, w = self.indexer.project(x.detach(), x_kv.detach())
        cache_write(ki, state.ik, state.write_slots)
        # Bound score memory independently of prompt and batch size.
        tile = state.score_query_tile
        chunk = max(
            tile, self.score_budget // (c.num_k_heads * state.length * 4) // tile * tile
        )
        outputs = []
        for start in range(0, len(q), chunk):
            end = min(start + chunk, len(q))
            idx = self.indexer.select(qi, state.ik, w, state, start, end, self.top_k)
            outputs.append(
                gqa_token_sparse_attn(
                    q[start:end],
                    state.k,
                    state.v,
                    state.req_to_token,
                    state.request_ids[start:end],
                    idx,
                    num_kv_chunks=None if state.phase == "decode" else 1,
                )
            )
        return torch.cat(outputs) if len(outputs) > 1 else outputs[0]
