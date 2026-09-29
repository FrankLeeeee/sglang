"""Native DSA/M3 sparse pipelines inside explicit synthetic attention adapters.

Main projections are standardized adapters, not checkpoint model modules. DSA
uses SGLang's complete Indexer and FlashMLA; M3 uses its complete native block
scoring/selection/attention pipeline. See README for architectural differences.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import flashinfer
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gqa_lightning_attention"))
from attention import Config, PagedState
from kernels import cache_write, rms_norm_per_head


def weight(out, inp=8192):
    return torch.randn(out, inp, device="cuda", dtype=torch.bfloat16) / inp**0.5


def init_runtime():
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs

    get_context().set_server_args(
        ServerArgs(
            model_path="dummy",
            tp_size=1,
            dsa_prefill_backend="flashmla_sparse",
            dsa_decode_backend="flashmla_sparse",
            dsa_paged_mqa_logits_backend="deepgemm",
        )
    )
    import os

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f'tcp://127.0.0.1:{os.environ.get("BENCH_PORT", "29670")}',
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)


class M3Case:
    """128-dim GQA + native per-group block selection (16 x 128 tokens)."""

    def __init__(
        self, ci, batch, length, qpr, sparse, phase, topk=2048, dense_backend="fa3"
    ):
        from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
            minimax_sparse_decode,
            minimax_sparse_prefill,
        )

        self.prefill, self.decode = minimax_sparse_prefill, minimax_sparse_decode
        self.c = c = Config(48 if ci == 1 else 64, 4 if ci == 1 else 8)
        self.sparse, self.phase, self.topk = sparse, phase, topk
        self.wq, self.wk, self.wv = (
            weight(c.num_q_heads * 128),
            weight(c.num_k_heads * 128),
            weight(c.num_k_heads * 128),
        )
        if sparse:
            self.wiq, self.wik = weight(c.num_k_heads * 128), weight(128)
        torch.manual_seed(43)
        self.s = s = PagedState(c, batch, length, qpr, 128, sparse, phase)
        if sparse:
            s.ik = torch.randn(
                batch * length, 1, 128, device="cuda", dtype=torch.bfloat16
            )
        self.cuq = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * qpr
        self.seq = torch.full((batch,), length, device="cuda", dtype=torch.int32)
        if not sparse and dense_backend != "fa2":
            pages = (s.req_to_token[:, ::128] // 128).flatten().contiguous()
            indptr = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * (
                length // 128
            )
            last = torch.full((batch,), 128, device="cuda", dtype=torch.int32)
            if phase == "decode":
                s.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    s.workspace,
                    use_cuda_graph=True,
                    use_tensor_cores=True,
                    backend=dense_backend,
                    paged_kv_indptr_buffer=indptr,
                    paged_kv_indices_buffer=pages,
                    paged_kv_last_page_len_buffer=last,
                )
                s.wrapper.plan(
                    indptr,
                    pages,
                    last,
                    c.num_q_heads,
                    c.num_k_heads,
                    128,
                    128,
                    q_data_type=torch.bfloat16,
                    kv_data_type=torch.bfloat16,
                )
            else:
                s.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    s.workspace, backend=dense_backend
                )
                s.wrapper.plan(
                    self.cuq,
                    indptr,
                    pages,
                    last,
                    c.num_q_heads,
                    c.num_k_heads,
                    128,
                    128,
                    causal=True,
                    q_data_type=torch.bfloat16,
                    kv_data_type=torch.bfloat16,
                )
        self.prefix = self.seq - qpr
        self.req = torch.arange(batch, device="cuda", dtype=torch.int32)
        self.length, self.qpr, self.batch = length, qpr, batch
        torch.manual_seed(44)
        self.x = torch.randn(batch * qpr, 8192, device="cuda", dtype=torch.bfloat16)

    def project(self):
        x, c, s = self.x, self.c, self.s
        q = rms_norm_per_head(F.linear(x, self.wq).view(-1, c.num_q_heads, 128))
        k = rms_norm_per_head(F.linear(x, self.wk).view(-1, c.num_k_heads, 128))
        v = rms_norm_per_head(F.linear(x, self.wv).view(-1, c.num_k_heads, 128))
        cache_write(k, s.k, s.write_slots)
        cache_write(v, s.v, s.write_slots)
        return q

    def index_project(self):
        qi = rms_norm_per_head(
            F.linear(self.x, self.wiq).view(-1, self.c.num_k_heads, 128)
        )
        ki = rms_norm_per_head(F.linear(self.x, self.wik).view(-1, 1, 128))
        cache_write(ki, self.s.ik, self.s.write_slots)
        return qi

    def attend(self, q, qi=None):
        s = self.s
        if not self.sparse:
            return s.dense(q)
        kw = dict(
            score_type="max", disable_index_value=True, page_size=128, use_msa=False
        )
        if self.phase == "prefill":
            return self.prefill(
                q,
                s.k,
                s.v,
                None,
                qi,
                s.ik,
                None,
                None,
                s.req_to_token,
                self.req,
                self.cuq,
                self.seq,
                self.prefix,
                self.qpr,
                self.length,
                1,
                128,
                self.topk // 128,
                0,
                1,
                cu_seqblocks_q=self.cuq,
                max_seqblock_q=self.qpr,
                all_seqblock_q=self.batch * self.qpr,
                seqlens_cpu=[self.length] * self.batch,
                **kw,
            )[1]
        return self.decode(
            q,
            None,
            s.k,
            s.v,
            qi,
            None,
            s.ik,
            None,
            s.req_to_token,
            self.req,
            self.seq,
            self.length,
            1,
            128,
            self.topk // 128,
            0,
            1,
            **kw,
        )[1]

    def run(self):
        q = self.project()
        qi = self.index_project() if self.sparse else None
        return self.attend(q, qi)


class DSACase:
    """Absorbed MLA adapter + complete production DSA indexer.

    Shared main adapter: 8192 -> 1536 -> H*192 query, then absorb 128 -> 512;
    8192 -> 576 latent KV.
    V aliases the first 512 KV channels. No model output/up projection is timed.
    Native Hopper sparse kernel pads 48 logical Q heads to 64; cost is included.
    """

    def __init__(
        self, ci, batch, length, qpr, sparse, phase, topk=2048, dense_backend="fa3"
    ):
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
        from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer, deep_gemm
        from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
        from sglang.srt.layers.attention.dsa_backend import DSAIndexerMetadata
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )

        self.flashmla = flash_mla_sparse_fwd
        self.h = 48 if ci == 1 else 64
        self.sparse, self.phase, self.topk = sparse, phase, topk
        self.batch, self.length, self.qpr = batch, length, qpr
        self.wa, self.wq, self.wk = (
            weight(1536),
            weight(self.h * 192, 1536),
            weight(576),
        )
        self.wuk = weight(self.h * 128, 512).view(self.h, 128, 512)
        torch.manual_seed(43)
        pages = torch.randperm(batch * length // 64, device="cuda", dtype=torch.int32)
        self.page_table = pages.view(batch, -1)
        self.table = (
            self.page_table[:, :, None] * 64
            + torch.arange(64, device="cuda", dtype=torch.int32)
        ).reshape(batch, length)
        self.slots = self.table[:, length - qpr :].reshape(-1).long()
        self.positions = torch.arange(length - qpr, length, device="cuda").repeat(batch)
        self.req = torch.arange(
            batch, device="cuda", dtype=torch.int32
        ).repeat_interleave(qpr)
        self.cuq = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * qpr
        self.seq = torch.full((batch,), length, device="cuda", dtype=torch.int32)
        self.pool = DSATokenToKVPool(
            size=batch * length,
            page_size=64,
            kv_lora_rank=512,
            dtype=torch.bfloat16,
            qk_rope_head_dim=64,
            layer_num=1,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
        )
        self.kv = self.pool.get_key_buffer(0)
        self.kv.normal_()
        self.fb = ForwardBatch(
            forward_mode=(
                ForwardMode.EXTEND if phase == "prefill" else ForwardMode.DECODE
            ),
            batch_size=batch,
            input_ids=torch.zeros(batch * qpr, device="cuda", dtype=torch.int64),
            req_pool_indices=torch.arange(batch, device="cuda", dtype=torch.int64),
            seq_lens=self.seq,
            out_cache_loc=self.slots,
            seq_lens_sum=batch * length,
        )
        self.fb.seq_lens_cpu = self.seq.cpu()
        self.fb.extend_seq_lens_cpu = [qpr] * batch if phase == "prefill" else None
        self.fb.extend_seq_lens = (
            torch.full_like(self.seq, qpr) if phase == "prefill" else None
        )
        self.fb.extend_prefix_lens = self.seq - qpr if phase == "prefill" else None
        ks = self.req * length
        lengths = (self.positions + 1).int()
        meta = SimpleNamespace(
            cache_seqlens_int32=self.seq,
            real_page_table=self.page_table,
            page_table_1=self.table,
            dsa_seqlens_expanded=lengths,
            cu_seqlens_q=self.cuq,
            cu_seqlens_k=torch.arange(batch + 1, device="cuda", dtype=torch.int32)
            * length,
            indexer_k_start_end=(ks, ks + lengths),
            indexer_seq_lens=self.seq,
            indexer_seq_lens_cpu=self.seq.cpu(),
            dsa_extend_seq_lens_list=[qpr] * batch,
            token_to_batch_idx=self.req,
            topk_indices_offset=None,
            topk_v2_plan=plan_topk_v2(lengths) if phase == "decode" else None,
        )
        self.metadata = DSAIndexerMetadata(
            meta,
            TopkTransformMethod.PAGED,
            paged_mqa_schedule_metadata=(
                deep_gemm.get_paged_mqa_logits_metadata(
                    self.seq[:, None], 64, deep_gemm.get_num_sms()
                )
                if phase == "decode"
                else None
            ),
        )
        self.backend = SimpleNamespace(
            token_to_kv_pool=self.pool,
            req_to_token_pool=SimpleNamespace(req_to_token=self.table),
            get_indexer_metadata=lambda *a: self.metadata,
        )
        if sparse:
            torch.manual_seed(45)
            with torch.device("cuda"):
                prev = torch.get_default_dtype()
                torch.set_default_dtype(torch.bfloat16)
                try:
                    self.indexer = Indexer(
                        hidden_size=8192,
                        index_n_heads=64,
                        index_head_dim=128,
                        rope_head_dim=64,
                        index_topk=topk,
                        q_lora_rank=1536,
                        max_position_embeddings=max(length, 16384),
                        rope_theta=10000.0,
                        layer_id=0,
                        scale_fmt="ue8m0",
                        is_neox_style=False,
                    )
                finally:
                    torch.set_default_dtype(prev)
            for name, p in self.indexer.named_parameters():
                if "k_norm.weight" in name:
                    p.data.fill_(1)
                elif "k_norm.bias" in name:
                    p.data.zero_()
                else:
                    p.data.normal_(std=p.shape[-1] ** -0.5)
            # Initialize every physical index slot using the native quantized store.
            # Chunked to avoid a full hidden-state prefix allocation.
            from sglang.kernels.ops.attention.fused_store_index_cache import (
                fused_store_index_k_cache,
            )

            for start in range(0, batch * length, 16384):
                n = min(16384, batch * length - start)
                k = torch.randn(n, 128, device="cuda", dtype=torch.bfloat16)
                loc = torch.arange(start, start + n, device="cuda", dtype=torch.int64)
                fused_store_index_k_cache(
                    k, self.pool.get_index_k_with_scale_buffer(0), loc, 64
                )
        else:
            self.workspace = torch.empty(
                1024 * 1024**2, device="cuda", dtype=torch.uint8
            )
            ki = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * (
                length // 64
            )
            self.wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                self.workspace,
                use_cuda_graph=phase == "decode",
                qo_indptr=self.cuq,
                kv_indptr=ki,
                kv_indices=pages,
                kv_len_arr=self.seq,
                backend=dense_backend,
            )
            self.wrapper.plan(
                self.cuq,
                ki,
                pages,
                self.seq,
                self.h,
                512,
                64,
                64,
                True,
                192**-0.5,
                torch.bfloat16,
                torch.bfloat16,
            )
        torch.manual_seed(44)
        self.x = torch.randn(batch * qpr, 8192, device="cuda", dtype=torch.bfloat16)

    def project(self):
        qa = F.rms_norm(F.linear(self.x, self.wa), (1536,))
        q_raw = F.linear(qa, self.wq).view(-1, self.h, 192)
        q_nope = torch.einsum("thd,hdc->thc", q_raw[..., :128], self.wuk)
        k = F.linear(self.x, self.wk).view(-1, 1, 576)
        # Normalize the absorbed latent component, preserving the RoPE component.
        q_rope, k_rope = flashinfer.apply_rope_pos_ids(
            q_raw[..., 128:],
            k[..., 512:],
            self.positions,
            interleave=True,
            rope_theta=10000.0,
        )
        q = torch.cat((q_nope, q_rope), dim=-1)
        k = torch.cat((F.rms_norm(k[..., :512], (512,)), k_rope), dim=-1)
        self.pool.set_mla_kv_buffer(
            SimpleNamespace(layer_id=0), self.slots, k[..., :512], k[..., 512:]
        )
        return q, qa

    def select(self, qa):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        with forward_context(ForwardContext(attn_backend=self.backend)):
            return self.indexer(self.x, qa, self.positions, self.fb, 0)

    def attend(self, q, idx=None):
        if not self.sparse:
            kv = self.kv.view(-1, 64, 576)
            return self.wrapper.run(
                q[..., :512], q[..., 512:], kv[..., :512], kv[..., 512:]
            )
        if self.h != 64:
            q = F.pad(q, (0, 0, 0, 64 - self.h))
        return self.flashmla(q, self.kv, idx[:, None, :], 192**-0.5, 512)[0][
            :, : self.h
        ]

    def run(self):
        q, qa = self.project()
        idx = self.select(qa) if self.sparse else None
        return self.attend(q, idx)
