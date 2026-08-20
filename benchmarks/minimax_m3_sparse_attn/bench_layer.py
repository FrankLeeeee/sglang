#!/usr/bin/env python3
"""Level 2 — end-to-end benchmark of one full MiniMax-M3 sparse attention layer.

Where ``bench_kernels.py`` times the sparse attention kernels in isolation, this
builds the *real* ``MiniMaxM3Attention`` module with dummy weights and runs its
real ``forward()``, so the measured latency includes everything a serving layer
pays for:

    fused QKV + index-QKV projection
      -> per-head Gemma RMS-norm + partial (rotary_dim=64) NeoX RoPE
      -> fused main-KV + index-K paged cache store
      -> indexer attention -> block top-k -> sparse GQA attention
      -> o_proj

It runs against a real ``MiniMaxSparseKVPool`` and the real
``MiniMaxSparseAttnBackend``; only the weights are random and the module is
instantiated standalone rather than as part of the 60-layer model.

Usage:
    python benchmarks/minimax_m3_sparse_attn/bench_layer.py --help
    CUDA_VISIBLE_DEVICES=1 python .../bench_layer.py --context-lens 4096,32768
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (
    emit_plots,
    STAGE_ORDER,
    analytic_memory,
    bench_cuda,
    build_page_table,
    gpu_info,
    measure_transient_bytes,
    profile_breakdown,
    wait_for_idle,
    warn_if_contended,
    write_results,
)
from m3_config import (
    DEFAULT_CONTEXT_LENS,
    DEFAULT_PREFILL_CHUNK,
    M3_MAX_POSITION,
    M3_NUM_SPARSE_LAYERS,
    SparseAttnConfig,
    m3_config,
)

HF_MODEL_ID = "MiniMaxAI/MiniMax-M3"

# Released MiniMax-M3 text_config fields the attention module reads. Used when
# the HF config cannot be fetched (offline runs).
FALLBACK_TEXT_CONFIG = {
    "hidden_size": 6144,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000,
    "rotary_dim": 64,
    "partial_rotary_factor": 0.5,
    "max_position_embeddings": 1048576,
    "use_qk_norm": True,
    "qk_norm_type": "per_head",
    "use_gemma_norm": True,
    "attention_output_gate": False,
    "sparse_attention_config": {
        "use_sparse_attention": True,
        "sparse_index_dim": 128,
        "sparse_num_index_heads": 4,
        "sparse_topk_blocks": 16,
        "sparse_block_size": 128,
        "sparse_score_type": "max",
        "sparse_init_block": 0,
        "sparse_local_block": 1,
    },
}


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


def init_runtime(cfg: SparseAttnConfig, port: int = 29531) -> None:
    """Publish the process-wide ServerArgs and bring up a world-size-1 TP group.

    ``QKVParallelLinear``/``RowParallelLinear`` need a TP group, and ``get_rope``
    reads the published ServerArgs — both are process globals a real server sets
    up during ``ModelRunner.__init__``.
    """
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs

    get_context().set_server_args(
        ServerArgs(
            model_path="dummy",
            tp_size=1,
            attention_backend="triton",
            page_size=cfg.page_size,
            disable_cuda_graph=True,
        )
    )

    if model_parallel_is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=0,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)


def load_text_config(allow_download: bool = True):
    """The released M3 ``text_config`` as a PretrainedConfig, or the fallback."""
    from transformers import PretrainedConfig

    raw = None
    if allow_download:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(HF_MODEL_ID, "config.json")
            raw = json.loads(Path(path).read_text())["text_config"]
        except Exception as err:  # offline / no HF token — use the pinned copy
            print(f"  [config] could not fetch {HF_MODEL_ID}: {err}; using fallback")
    if raw is None:
        raw = dict(FALLBACK_TEXT_CONFIG)
    cfg = PretrainedConfig(**raw)
    # A standalone layer never sees the model's 60-entry freq list; the benchmark
    # decides sparseness per instantiated layer instead.
    cfg.sparse_attention_config = dict(raw["sparse_attention_config"])
    return cfg


def override_shapes(hf_cfg, cfg: SparseAttnConfig):
    """Point the HF config at the per-rank shapes under test (tp=1 module)."""
    hf_cfg.num_attention_heads = cfg.num_q_heads
    hf_cfg.num_key_value_heads = cfg.num_kv_heads
    hf_cfg.head_dim = cfg.head_dim
    if cfg.granularity == "dense":
        # A dense layer has no indexer at all — M3's own 3 dense layers are
        # built this way (sparse_attention_freq entry == 0).
        hf_cfg.sparse_attention_config = dict(
            hf_cfg.sparse_attention_config, sparse_attention_freq=[0]
        )
        return hf_cfg
    hf_cfg.sparse_attention_config = dict(
        hf_cfg.sparse_attention_config,
        sparse_num_index_heads=cfg.num_idx_heads,
        sparse_index_dim=cfg.idx_head_dim,
        sparse_topk_blocks=cfg.topk_blocks,
        sparse_block_size=cfg.block_size,
        sparse_init_block=cfg.init_blocks,
        sparse_local_block=cfg.local_blocks,
        sparse_score_type=cfg.score_type,
        # single instantiated layer, sparse, index pool is K-only (as M3 ships)
        sparse_attention_freq=[1],
        sparse_disable_index_value=[1 if cfg.disable_index_value else 0],
    )
    return hf_cfg


# ---------------------------------------------------------------------------
# standalone layer + backend
# ---------------------------------------------------------------------------


class _StubModelConfig:
    def __init__(self, hf_config, context_len: int):
        self.hf_config = hf_config
        self.context_len = context_len
        self.num_attention_heads = hf_config.num_attention_heads


class _StubReqToTokenPool:
    def __init__(self, req_to_token: torch.Tensor):
        self.req_to_token = req_to_token


class _StubModelRunner:
    """The handful of fields ``MiniMaxSparseAttnBackend.__init__`` reads.

    Passing the real ModelRunner would mean loading 435B of weights; the backend
    only needs the KV pool, the page table, and the model shape.
    """

    def __init__(self, kv_pool, req_to_token, hf_config, context_len):
        self.token_to_kv_pool = kv_pool
        self.req_to_token_pool = _StubReqToTokenPool(req_to_token)
        self.model_config = _StubModelConfig(hf_config, context_len)
        self.server_args = None


def build_kv_pool(cfg: SparseAttnConfig, num_slots: int, device: str = "cuda"):
    from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

    dense = cfg.granularity == "dense"
    return MiniMaxSparseKVPool(
        size=num_slots,
        page_size=cfg.page_size,
        dtype=cfg.torch_dtype,
        head_num=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        idx_head_dim=cfg.idx_head_dim,
        # A dense layer allocates no index cache — the +50% KV tax disappears.
        dense_layer_ids=[0] if dense else [],
        sparse_layer_ids=[] if dense else [0],
        disable_value_sparse_layer_ids=(
            [] if dense else ([0] if cfg.disable_index_value else [])
        ),
        device=device,
        start_layer=0,
        end_layer=1,
    )


def build_attention_layer(hf_cfg, device: str, dtype: torch.dtype, is_sparse: bool = True):
    from sglang.srt.models.minimax_m3 import MiniMaxM3Attention

    torch.set_default_dtype(dtype)
    try:
        layer = MiniMaxM3Attention(
            config=hf_cfg,
            layer_id=0,
            quant_config=None,
            prefix="model.layers.0.self_attn",
            is_sparse_attention_layer=is_sparse,
            disable_index_value=bool(
                hf_cfg.sparse_attention_config["sparse_disable_index_value"][0]
            )
            if is_sparse
            else False,
        )
    finally:
        torch.set_default_dtype(torch.float32)

    # Move, but do NOT cast: RotaryEmbedding.cos_sin_cache must stay fp32 on
    # CUDA. Casting it to bf16 silently disables the fused grouped
    # qk-norm+RoPE path (_combined_qknorm_ok) that production actually runs.
    layer = layer.to(device=device)
    assert layer.rotary_emb.cos_sin_cache.dtype == torch.float32
    # Dummy weights at a trained checkpoint's scale, so no denormals or infs
    # perturb the timing.
    with torch.no_grad():
        for param in layer.parameters():
            param.data = param.data.to(dtype)
            param.normal_(mean=0.0, std=0.02)
    layer.maybe_build_fused_qkv_index()
    layer.eval()
    return layer


# ---------------------------------------------------------------------------
# forward batches
# ---------------------------------------------------------------------------


def _make_forward_batch(
    *,
    forward_mode,
    batch_size: int,
    num_tokens: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    extend_seq_lens: Optional[torch.Tensor],
    extend_prefix_lens: Optional[torch.Tensor],
    extend_seq_lens_cpu: Optional[list[int]],
    device: torch.device,
):
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    fb = ForwardBatch(
        forward_mode=forward_mode,
        batch_size=batch_size,
        input_ids=torch.zeros(num_tokens, dtype=torch.int64, device=device),
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc,
        seq_lens_sum=int(seq_lens.sum().item()),
    )
    fb.seq_lens_cpu = seq_lens.cpu()
    fb.extend_seq_lens = extend_seq_lens
    fb.extend_prefix_lens = extend_prefix_lens
    fb.extend_seq_lens_cpu = extend_seq_lens_cpu
    return fb


class _DenseLayerBackend:
    """Minimal attention backend for the dense variant of the layer.

    A dense MiniMaxM3Attention calls ``RadixAttention`` with no ``idx_*``
    kwargs, so it needs a backend that just writes KV and runs full causal
    attention. Production gets this from ``MiniMaxHybridAttnBackend`` routing
    non-sparse layers to a real dense backend; this stands in for that with the
    same underlying sglang Triton kernels.
    """

    def __init__(self, kv_pool, req_to_token: torch.Tensor):
        self.kv_pool = kv_pool
        self.req_to_token = req_to_token
        self.token_to_kv_pool = kv_pool
        self._prefix_lens_cpu = None
        self._total_kv = None

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        if forward_batch.extend_seq_lens_cpu is not None:
            seq = forward_batch.seq_lens_cpu.tolist()
            self._prefix_lens_cpu = [
                int(s) - int(e)
                for s, e in zip(seq, forward_batch.extend_seq_lens_cpu)
            ]
            self._total_kv = None
        else:
            self._prefix_lens_cpu = None
            self._total_kv = int(forward_batch.seq_lens_cpu.sum().item())

    def init_forward_metadata(self, forward_batch):
        self.init_forward_metadata_out_graph(forward_batch)

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

        return AttentionBackend.forward(
            self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def _store(self, layer, forward_batch, k, v):
        self.kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        from sglang.srt.layers.attention.minimax_sparse_ops.minimax_dense import (
            minimax_dense_prefill,
        )

        if save_kv_cache:
            self._store(layer, forward_batch, k, v)
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=q.device),
                forward_batch.extend_seq_lens.to(torch.int32).cumsum(0).to(torch.int32),
            ]
        )
        # The dense branch of MiniMaxM3Attention.forward_core hands q over flat
        # ([n, num_heads*head_dim]); the kernels want it 3-D, exactly as
        # TritonAttnBackend views it.
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        _, o = minimax_dense_prefill(
            q,
            k_cache,
            v_cache,
            None,
            None,
            None,
            None,
            None,
            self.req_to_token,
            forward_batch.req_pool_indices,
            cu_seqlens,
            forward_batch.seq_lens.to(torch.int32),
            forward_batch.extend_prefix_lens.to(torch.int32),
            max(forward_batch.extend_seq_lens_cpu),
            # k/v here are the projection outputs, exactly what a server passes.
            k_extend=k,
            v_extend=v,
            sm_scale=layer.scaling,
            prefix_lens_cpu=self._prefix_lens_cpu,
        )
        return o.reshape(q.shape[0], -1).contiguous()

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        from sglang.srt.layers.attention.minimax_sparse_ops.minimax_dense import (
            minimax_dense_decode,
        )

        if save_kv_cache:
            self._store(layer, forward_batch, k, v)
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        _, o = minimax_dense_decode(
            q,
            None,
            k_cache,
            v_cache,
            None,
            None,
            None,
            None,
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens.to(torch.int32),
            total_kv=self._total_kv,
            sm_scale=layer.scaling,
        )
        return o.reshape(q.shape[0], -1).contiguous()


class LayerCase:
    """Everything needed to time one forward of the standalone layer."""

    def __init__(self, layer, backend, forward_batch, positions, hidden_states):
        self.layer = layer
        self.backend = backend
        self.forward_batch = forward_batch
        self.positions = positions
        self.hidden_states = hidden_states

    def run(self):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        with forward_context(ForwardContext(attn_backend=self.backend)):
            self.backend.init_forward_metadata_out_graph(self.forward_batch)
            return self.layer(
                positions=self.positions,
                hidden_states=self.hidden_states,
                forward_batch=self.forward_batch,
            )


def build_case(
    cfg: SparseAttnConfig,
    hf_cfg,
    *,
    phase: str,
    batch_size: int,
    context_len: int,
    chunk_len: Optional[int],
    device: str = "cuda",
) -> LayerCase:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    dev = torch.device(device)
    dtype = cfg.torch_dtype
    is_prefill = phase == "prefill"
    chunk = (context_len if chunk_len is None else min(chunk_len, context_len)) if is_prefill else 1
    prefix = context_len - chunk

    req_to_token, num_slots = build_page_table(
        batch_size=batch_size,
        context_len=context_len,
        page_size=cfg.page_size,
        device=dev,
    )
    kv_pool = build_kv_pool(cfg, num_slots, device=device)

    dense = cfg.granularity == "dense"
    layer = build_attention_layer(hf_cfg, device, dtype, is_sparse=not dense)
    if dense:
        backend = _DenseLayerBackend(kv_pool, req_to_token)
    else:
        runner = _StubModelRunner(kv_pool, req_to_token, hf_cfg, context_len)
        from sglang.srt.layers.attention.minimax_sparse_backend import (
            MiniMaxSparseAttnBackend,
        )

        backend = MiniMaxSparseAttnBackend(runner)

    num_tokens = batch_size * chunk
    req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=dev)
    seq_lens = torch.full((batch_size,), context_len, dtype=torch.int64, device=dev)
    # The slots the new tokens land in: the tail `chunk` positions of each request.
    out_cache_loc = req_to_token[:, prefix:context_len].reshape(-1).long().contiguous()

    if is_prefill:
        extend_seq_lens = torch.full((batch_size,), chunk, dtype=torch.int64, device=dev)
        extend_prefix_lens = torch.full(
            (batch_size,), prefix, dtype=torch.int64, device=dev
        )
        extend_cpu = [chunk] * batch_size
        mode = ForwardMode.EXTEND
        positions = (
            torch.arange(prefix, context_len, device=dev, dtype=torch.int64)
            .repeat(batch_size)
            .contiguous()
        )
    else:
        extend_seq_lens = extend_prefix_lens = extend_cpu = None
        mode = ForwardMode.DECODE
        positions = torch.full(
            (batch_size,), context_len - 1, device=dev, dtype=torch.int64
        )

    fb = _make_forward_batch(
        forward_mode=mode,
        batch_size=batch_size,
        num_tokens=num_tokens,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc,
        extend_seq_lens=extend_seq_lens,
        extend_prefix_lens=extend_prefix_lens,
        extend_seq_lens_cpu=extend_cpu,
        device=dev,
    )
    hidden_states = torch.randn(
        num_tokens, hf_cfg.hidden_size, dtype=dtype, device=dev
    )
    return LayerCase(layer, backend, fb, positions, hidden_states)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def measure_case(
    case: LayerCase,
    cfg: SparseAttnConfig,
    *,
    phase: str,
    batch_size: int,
    context_len: int,
    num_query_tokens: int,
    iters: int,
    profile_iters: int,
) -> dict:
    timing = bench_cuda(case.run, warmup=max(3, iters // 4), iters=iters)
    stages, kernels = profile_breakdown(case.run, iters=profile_iters, warmup=3)

    row = {
        "phase": phase,
        # The layer benchmark only ever walks context lengths, so every row
        # belongs to the context sweep. plot_results.py keys off this.
        "sweep": "context",
        "level": "layer",
        "granularity": cfg.granularity,
        "context_len": context_len,
        "batch_size": batch_size,
        "num_query_tokens": num_query_tokens,
        "num_q_heads": cfg.num_q_heads,
        "num_kv_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "num_idx_heads": cfg.num_idx_heads,
        "idx_head_dim": cfg.idx_head_dim,
        "block_size": cfg.block_size,
        "topk_blocks": cfg.topk_blocks,
        "page_size": cfg.page_size,
        "dtype": cfg.dtype,
        "latency_mean_ms": round(timing.mean_ms, 6),
        "latency_median_ms": round(timing.median_ms, 6),
        "latency_min_ms": round(timing.min_ms, 6),
        "latency_p90_ms": round(timing.p90_ms, 6),
    }
    for stage in STAGE_ORDER:
        row[f"stage_{stage}_ms"] = round(stages.get(stage, 0.0), 6)
    stage_sum = sum(stages.values())
    row["stage_sum_ms"] = round(stage_sum, 6)
    row["launch_gap_ms"] = round(max(0.0, timing.median_ms - stage_sum), 6)
    row["transient_bytes"] = measure_transient_bytes(case.run)
    row.update(
        analytic_memory(
            cfg,
            batch_size=batch_size,
            context_len=context_len,
            num_query_tokens=num_query_tokens,
            is_prefill=(phase == "prefill"),
        )
    )
    # what 57 sparse layers of this shape cost end to end
    row["layer_ms_x57"] = round(timing.median_ms * M3_NUM_SPARSE_LAYERS, 4)
    row["throughput_tok_per_s"] = round(
        num_query_tokens / (row["latency_median_ms"] / 1000.0), 1
    )
    row["status"] = "ok"
    return row


def run_point(cfg, hf_cfg, *, phase, batch_size, context_len, chunk_len, args) -> Optional[dict]:
    label = f"{phase:<7} bs={batch_size:<4} ctx={context_len:<7}"
    case = None
    try:
        case = build_case(
            cfg,
            hf_cfg,
            phase=phase,
            batch_size=batch_size,
            context_len=context_len,
            chunk_len=chunk_len,
        )
        n_tok = case.hidden_states.shape[0]
        row = measure_case(
            case,
            cfg,
            phase=phase,
            batch_size=batch_size,
            context_len=context_len,
            num_query_tokens=n_tok,
            iters=args.prefill_iters if phase == "prefill" else args.decode_iters,
            profile_iters=args.profile_iters,
        )
        print(
            f"  {label} {row['latency_median_ms']:9.4f} ms "
            f"(min {row['latency_min_ms']:8.4f})  "
            f"(gemm {row['stage_projection_gemm_ms']:.4f} | "
            + (f"dense {row['stage_dense_attn_ms']:.4f} | " if row.get("stage_dense_attn_ms") else "") +
            f"norm/rope {row['stage_qk_norm_rope_ms']:.4f} | "
            f"kv_store {row['stage_kv_store_ms']:.4f} | "
            f"index {row['stage_indexer_score_ms']:.4f} | "
            f"topk {row['stage_topk_select_ms']:.4f} | "
            f"sparse {row['stage_sparse_attn_ms']:.4f})"
        )
        return row
    except torch.OutOfMemoryError:
        print(f"  {label} OOM")
        return {"phase": phase, "context_len": context_len, "batch_size": batch_size, "status": "oom"}
    except Exception as err:
        print(f"  {label} FAILED: {type(err).__name__}: {err}")
        traceback.print_exc(limit=5)
        return {
            "phase": phase,
            "context_len": context_len,
            "batch_size": batch_size,
            "status": f"error: {type(err).__name__}: {err}",
        }
    finally:
        del case
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--context-lens", type=_int_list, default=DEFAULT_CONTEXT_LENS)
    p.add_argument("--prefill-batch-size", type=int, default=1)
    p.add_argument(
        "--prefill-chunk",
        type=int,
        default=DEFAULT_PREFILL_CHUNK,
        help=f"extend-chunk size (default: {DEFAULT_PREFILL_CHUNK}); "
        "0 = whole-context prefill",
    )
    p.add_argument("--decode-batch-sizes", type=_int_list, default=[1, 32])
    p.add_argument("--prefill-iters", type=int, default=20)
    p.add_argument("--decode-iters", type=int, default=200)
    p.add_argument("--profile-iters", type=int, default=20)
    p.add_argument(
        "--granularity",
        default="block",
        choices=["block", "token", "dense"],
        help="block/token selection, or dense = no indexer and no sparse kernel",
    )
    p.add_argument("--no-download", action="store_true", help="use the pinned config copy")
    p.add_argument("--port", type=int, default=29531)
    p.add_argument(
        "--wait-for-idle",
        type=float,
        default=0.0,
        help="seconds to wait for the GPU to go idle before starting (shared boxes)",
    )
    p.add_argument("-o", "--out", type=Path, default=Path(__file__).resolve().parent / "results")
    p.add_argument("--no-plots", action="store_true",
                   help="skip the plots this benchmark emits when it finishes")
    p.add_argument("--tag", default="layer")
    args = p.parse_args(argv)
    if not args.context_lens:
        p.error("--context-lens must contain at least one value")
    if any(ctx <= 0 or ctx > M3_MAX_POSITION for ctx in args.context_lens):
        p.error(
            f"every context length must be in [1, {M3_MAX_POSITION}] "
            "(MiniMax-M3's max_position_embeddings)"
        )
    if args.prefill_chunk < 0:
        p.error("--prefill-chunk must be non-negative")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA is required for these benchmarks.")
        return 1

    cfg = m3_config(granularity=args.granularity)
    # The backend reads this at construction; set it before any case is built.
    from sglang.srt.environ import envs

    envs.SGLANG_USE_MINIMAX_TOKEN_SPARSE.set(args.granularity == "token")
    init_runtime(cfg, args.port)
    hf_cfg = override_shapes(load_text_config(not args.no_download), cfg)

    info = gpu_info()
    print("MiniMax-M3 sparse attention — full-layer benchmark (dummy weights)")
    print(f"  device : {info['gpu']} (sm{info['sm']}, {info['memory_gb']} GB)")
    print(f"  layer  : hidden={hf_cfg.hidden_size}, {cfg.shape_tag()}")
    print(f"  select : granularity={args.granularity}, budget={cfg.token_budget} tokens")
    print("           built at tp=1 so it owns exactly that per-rank shard.")
    wait_for_idle(args.wait_for_idle)
    gpu_util = warn_if_contended()

    # Pay the Triton/JIT first-touch costs before any timed point. The decode
    # kernels specialize on a batch-size bucket, so prime every batch size the
    # run will use, not just one.
    prime = build_case(
        cfg, hf_cfg, phase="prefill", batch_size=1, context_len=4096, chunk_len=None
    )
    for _ in range(3):
        prime.run()
    del prime
    for bs in sorted(set(args.decode_batch_sizes)):
        prime_dec = build_case(
            cfg, hf_cfg, phase="decode", batch_size=bs, context_len=4096, chunk_len=None
        )
        for _ in range(3):
            prime_dec.run()
        del prime_dec
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    rows: list[dict] = []
    print("\n=== PREFILL ===")
    for ctx in args.context_lens:
        chunk = None if args.prefill_chunk == 0 else min(args.prefill_chunk, ctx)
        row = run_point(
            cfg,
            hf_cfg,
            phase="prefill",
            batch_size=args.prefill_batch_size,
            context_len=ctx,
            chunk_len=chunk,
            args=args,
        )
        if row:
            rows.append(row)

    print("\n=== DECODE ===")
    for bs in args.decode_batch_sizes:
        for ctx in args.context_lens:
            row = run_point(
                cfg,
                hf_cfg,
                phase="decode",
                batch_size=bs,
                context_len=ctx,
                chunk_len=None,
                args=args,
            )
            if row:
                rows.append(row)

    for row in rows:
        row.setdefault("gpu", info["gpu"])
        row.setdefault("gpu_util_before_pct", gpu_util)
    json_path, csv_path = write_results(rows, args.out, args.tag)
    print(f"\nwrote {json_path}\n      {csv_path}")
    if not args.no_plots:
        import plot_results
        emit_plots(plot_results.main,
                   ["--results", str(args.out), "--out", str(args.out / "plots")],
                   label="plots")
        print(f"      {args.out / 'plots'}/*.png")
    errors = [row for row in rows if str(row.get("status", "")).startswith("error:")]
    if errors:
        print(f"\nFAILED: {len(errors)} benchmark point(s) ended in an error.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
