"""Shared harness for the MiniMax-M3 sparse attention benchmarks.

Provides input construction that matches what ``MiniMaxSparseAttnBackend`` hands
to the kernels, plus latency / memory / per-kernel-breakdown measurement.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import msgspec
import torch


def _kineto_early_init() -> None:
    """Initialize kineto/CUPTI before any sglang extension loads.

    Once sgl_kernel (pulled in transitively by every sglang kernel import) is
    in the process, kineto's first CUPTI initialization fails with
    CUPTI_ERROR_UNKNOWN and ``profile_breakdown`` returns empty stage tables.
    Initializing the profiler once beforehand is sufficient, so run it at
    import time — the sglang import below is the first one in every bench.
    """
    if not torch.cuda.is_available():
        return
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]):
        torch.zeros(1, device="cuda")


_kineto_early_init()

from sglang.kernels.ops.attention.minimax_sparse.common.utils import (  # noqa: E402
    get_cu_seqblocks,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m3_config import SparseAttnConfig

# ---------------------------------------------------------------------------
# per-kernel -> stage attribution
# ---------------------------------------------------------------------------

# Ordered (substring, stage) rules; the first match wins, so put the more
# specific kernel names first. Anything unmatched lands in "other" and is listed
# by --show-unmapped so a stage is never silently folded away.
STAGE_RULES: tuple[tuple[str, str], ...] = (
    # --- indexer: lightweight attention producing selection scores ------------
    ("_flash_attn_fwd_with_block_score_kernel", "indexer_score"),
    ("_decode_score_attn_kernel", "indexer_score"),
    ("_decode_score_kernel", "indexer_score"),
    ("_token_index_score_prefill_kernel", "indexer_score"),
    ("_token_index_score_decode_kernel", "indexer_score"),
    # Both passes of the fused token selector recompute q.k, so they are score
    # work; the threshold's top-k over pool maxima matches the rules below.
    ("_score_poolmax_kernel", "indexer_score"),
    ("_emit_above_threshold_kernel", "indexer_score"),
    ("minimax_token_emit_kernel", "indexer_score"),
    # tau-emit's scorer (writes scores + pooled maxima from one q.k pass) and
    # the one-pass kernel (q.k fused with running-threshold selection) both
    # spend their time on the scan, so they book as score work.
    ("_token_score_seg_pool_kernel", "indexer_score"),
    ("minimax_token_onepass_kernel", "indexer_score"),
    # --- top-k selection ------------------------------------------------------
    # The token path selects with FlashInfer's exact top-k when it is installed
    # and ATen's multi-block radix select (at::native::mbtopk) otherwise. These
    # must match before the generic "sort"/"gather"/"reduce" rules further down.
    # NOTE: FlashInfer files its top-k under `sampling::`, so this entry has to
    # come before any sampling-shaped rule, and dropping it silently moves the
    # whole selection stage into the "other" bucket.
    ("filteredtopk", "topk_select"),
    ("topk_kernel", "topk_select"),
    # tau-emit's selection pair: the radix threshold over pooled maxima and
    # the streaming filter that reads stored scores (no q.k recompute).
    ("minimax_row_kth_kernel", "topk_select"),
    ("minimax_score_filter_kernel", "topk_select"),
    ("mbtopk", "topk_select"),
    ("radixfindkthvalues", "topk_select"),
    ("gathertopk", "topk_select"),
    ("radixselect", "topk_select"),
    ("sortkeyvalueinplace", "topk_select"),
    ("bitonicsortkv", "topk_select"),
    # Masking the -inf rows to -1 padding is part of producing the index list.
    ("compare_scalar_kernel", "topk_select"),
    ("minimax_prefill_topk_block_kernel", "topk_select"),
    ("minimax_decode_topk_page_table_kernel", "topk_select"),
    ("minimax_decode_topk_block_kernel", "topk_select"),
    ("_topk_index_partial_kernel", "topk_select"),
    ("_topk_index_merge_kernel", "topk_select"),
    ("_topk_index_kernel", "topk_select"),
    # --- sparse main attention over the selected blocks / tokens -------------
    ("_gqa_share_sparse_fwd_kernel", "sparse_attn"),
    ("_gqa_share_sparse_decode_kernel", "sparse_attn"),
    ("_gqa_token_sparse_fwd_kernel", "sparse_attn"),
    # --- dense attention (the no-selection variant) ---------------------------
    # sglang's production Triton attention: _fwd_kernel is the extend/prefill
    # kernel, _fwd_(grouped_)kernel_stage1/2 the split-K decode pair.
    # ORDER MATTERS: "_fwd_kernel" is a suffix of _gqa_*_sparse_fwd_kernel, so
    # these must stay *after* the sparse rules above. test_stage_attribution
    # pins this.
    ("_fwd_kernel_unified", "dense_attn"),
    ("_fwd_grouped_kernel_stage1", "dense_attn"),
    ("_fwd_kernel_stage1", "dense_attn"),
    ("_fwd_kernel_stage2", "dense_attn"),
    ("_fwd_kernel", "dense_attn"),
    ("create_flashinfer_kv_indices_triton", "dense_attn"),
    ("get_num_kv_splits_triton", "dense_attn"),
    # --- split-k output merge -------------------------------------------------
    ("_merge_topk_attn_out_kernel", "merge"),
    ("_merge_attn_out_kernel", "merge"),
    ("_merge_chunks_kernel", "merge"),
    # --- index bookkeeping around the selection --------------------------------
    # Cast of the int64 top-k indices to int32, the scatter of each query chunk
    # into the global index buffer, and repeat_interleave building the per-query
    # request map. Broken out rather than folded into topk_select because it is
    # pure glue — a fused topk-transform kernel (DSA has one) would remove it.
    ("direct_copy_kernel", "select_overhead"),
    ("memcpy dtod", "select_overhead"),
    ("devicescan", "select_overhead"),
    ("compute_cuda_kernel", "select_overhead"),
    # --- KV + index cache write ----------------------------------------------
    ("store_kv_index_kernel", "kv_store"),
    ("index_copy", "kv_store"),
    ("indexing_backward", "kv_store"),
    # --- workspace allocation: the score buffer is filled with -inf and the
    #     top-k index buffer with -1 before either kernel runs. Cheap at short
    #     context, not at 128k where the score buffer is hundreds of MiB.
    ("fillfunctor", "buffer_init"),
    ("memset", "buffer_init"),
    # --- layer-level extras (bench_layer only) -------------------------------
    ("qknorm_rope", "qk_norm_rope"),
    ("qk_norm_rope", "qk_norm_rope"),
    ("rotary", "qk_norm_rope"),
    ("rms_norm", "qk_norm_rope"),
    ("nvjet", "projection_gemm"),
    ("cutlass", "projection_gemm"),
    ("gemm", "projection_gemm"),
    ("gemv", "projection_gemm"),
    ("s16816", "projection_gemm"),
    ("gett", "projection_gemm"),
    # --- the pure-torch top-k union (only when num_idx_heads > num_kv_heads) --
    ("sort", "topk_union"),
    ("gather", "topk_union"),
    ("cat", "topk_union"),
    ("where", "topk_union"),
    ("cumsum", "topk_union"),
    ("reduce", "topk_union"),
)

STAGE_ORDER = (
    "kv_store",
    "qk_norm_rope",
    "projection_gemm",
    "buffer_init",
    "indexer_score",
    "topk_select",
    "select_overhead",
    "topk_union",
    "sparse_attn",
    "dense_attn",
    "merge",
    "other",
)


def classify_kernel(name: str) -> str:
    low = name.lower()
    for needle, stage in STAGE_RULES:
        if needle.lower() in low:
            return stage
    return "other"


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

_L2_FLUSH_BYTES = 256 * 1024 * 1024
_l2_buffer: Optional[torch.Tensor] = None


def _flush_l2(device: torch.device) -> None:
    global _l2_buffer
    if _l2_buffer is None or _l2_buffer.device != device:
        _l2_buffer = torch.empty(_L2_FLUSH_BYTES, dtype=torch.int8, device=device)
    _l2_buffer.zero_()


class Timing(msgspec.Struct, frozen=True):
    mean_ms: float
    median_ms: float
    min_ms: float
    p90_ms: float
    iters: int

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}mean_ms": self.mean_ms,
            f"{prefix}median_ms": self.median_ms,
            f"{prefix}min_ms": self.min_ms,
            f"{prefix}p90_ms": self.p90_ms,
        }


def bench_cuda(
    fn: Callable[[], object],
    *,
    warmup: int = 10,
    iters: int = 50,
    flush_l2: bool = True,
) -> Timing:
    """Time ``fn`` with CUDA events, flushing L2 between iterations.

    The flush is enqueued *before* the start event, so it is not timed but does
    guarantee each iteration re-reads the KV cache from DRAM — which is what a
    real decode step does.
    """
    device = torch.cuda.current_device()
    dev = torch.device("cuda", device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flush_l2:
            _flush_l2(dev)
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return Timing(
        mean_ms=sum(times) / len(times),
        median_ms=times[len(times) // 2],
        min_ms=times[0],
        p90_ms=times[min(len(times) - 1, int(0.9 * len(times)))],
        iters=iters,
    )


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


def measure_transient_bytes(fn: Callable[[], object], *, warmup: int = 3) -> int:
    """Peak *extra* device bytes one call needs beyond its already-live inputs.

    This is the attention op's workspace: block-score buffer, top-k indices,
    split-k partials and the output tensor. It excludes the KV cache and Q/K/V,
    which are allocated before the measurement window.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return max(0, peak - base)


def analytic_memory(
    cfg: SparseAttnConfig,
    *,
    batch_size: int,
    context_len: int,
    num_query_tokens: int,
    is_prefill: bool,
    score_budget_bytes: Optional[int] = None,
) -> dict[str, int]:
    """Closed-form byte counts for the pieces that dominate attention memory.

    The selection buffers differ by granularity, so this must branch:

    * **block** — one fp32 score per 128-key block.
    * **token** — one fp32 score per *key*, i.e. ``block_size`` times larger.
      Prefill never materializes that whole matrix; the query axis is chunked to
      a byte budget, so ``score_buffer_bytes`` reports the *live* chunk and
      ``score_buffer_unchunked_bytes`` the full matrix it stands in for.
    * **dense** — no indexer, so no score or top-k buffer at all.
    """
    from sglang.kernels.ops.attention.minimax_sparse.token.index_score import (
        DEFAULT_SCORE_BUDGET_BYTES,
        plan_query_chunk,
    )

    per_tok = cfg.kv_bytes_per_token_per_layer()
    kv_tokens = batch_size * context_len
    heads = cfg.num_idx_heads

    if cfg.granularity == "dense":
        score_bytes = unchunked_bytes = topk_bytes = 0
    elif cfg.granularity == "token":
        topk = cfg.effective_topk_tokens
        if is_prefill:
            budget = (
                DEFAULT_SCORE_BUDGET_BYTES
                if score_budget_bytes is None
                else score_budget_bytes
            )
            batch_chunk, chunk = plan_query_chunk(
                batch_size=batch_size,
                max_seqlen_q=max(1, num_query_tokens // max(1, batch_size)),
                max_seqlen_k=context_len,
                num_idx_heads=heads,
                score_budget_bytes=budget,
            )
            # live: [heads, batch_chunk * chunk, kv_width <= context] float32
            score_bytes = heads * batch_chunk * chunk * context_len * 4
            unchunked_bytes = heads * num_query_tokens * context_len * 4
            topk_bytes = heads * num_query_tokens * topk * 4
        else:
            score_bytes = unchunked_bytes = heads * batch_size * context_len * 4
            topk_bytes = heads * batch_size * topk * 4
    else:  # block
        num_blocks = math.ceil(context_len / cfg.block_size)
        rows = num_query_tokens if is_prefill else batch_size
        score_bytes = unchunked_bytes = heads * rows * num_blocks * 4
        topk_bytes = heads * rows * cfg.topk_blocks * 4

    return {
        "kv_main_bytes": per_tok["main_kv"] * kv_tokens,
        "kv_index_bytes": per_tok["index_kv"] * kv_tokens,
        "kv_total_bytes": per_tok["total"] * kv_tokens,
        "kv_bytes_per_token": per_tok["total"],
        "score_buffer_bytes": score_bytes,
        "score_buffer_unchunked_bytes": unchunked_bytes,
        "topk_idx_bytes": topk_bytes,
    }


# ---------------------------------------------------------------------------
# per-kernel breakdown
# ---------------------------------------------------------------------------


def _profile_once(fn: Callable[[], object], iters: int) -> dict[str, float]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    per_kernel: dict[str, float] = {}
    for evt in prof.key_averages():
        # Kernel events carry device time on themselves; the enclosing aten op
        # only carries it via children, so self_device_time_total isolates
        # actual GPU kernels and never double-counts.
        us = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
        if us <= 0:
            continue
        per_kernel[evt.key] = per_kernel.get(evt.key, 0.0) + us / 1000.0 / iters
    return per_kernel


def profile_breakdown(
    fn: Callable[[], object],
    *,
    iters: int = 10,
    warmup: int = 5,
    repeats: int = 3,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (stage -> ms/iter, kernel_name -> ms/iter) from CUDA kernel time.

    Runs the profiled loop ``repeats`` times and keeps the pass with the lowest
    total device time. A co-tenant landing on the GPU mid-pass inflates every
    kernel in that pass, so taking the best pass — rather than averaging — is
    what keeps the breakdown consistent with the ``min`` latency estimator.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    best: Optional[dict[str, float]] = None
    best_total = float("inf")
    for _ in range(max(1, repeats)):
        per_kernel = _profile_once(fn, iters)
        total = sum(per_kernel.values())
        if total > 0 and total < best_total:
            best, best_total = per_kernel, total
    per_kernel = best or {}

    per_stage: dict[str, float] = {}
    for name, ms in per_kernel.items():
        stage = classify_kernel(name)
        per_stage[stage] = per_stage.get(stage, 0.0) + ms
    return per_stage, per_kernel


# ---------------------------------------------------------------------------
# input construction
# ---------------------------------------------------------------------------


def build_page_table(
    *,
    batch_size: int,
    context_len: int,
    page_size: int,
    device: torch.device,
    shuffle_pages: bool = True,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, int]:
    """Build ``req_to_token`` [batch, context_len] plus the pool slot count.

    Tokens inside a page are contiguous in the pool, pages are scattered across
    it — exactly sglang's paged allocator layout. This is why ``page_size`` is
    measurable here even though the Triton sparse kernels index per token: a
    page_size of 128 makes a whole 128-token sparse block one contiguous DRAM
    run, while page_size=1 turns the same block into 128 random gathers.
    """
    pages_per_req = math.ceil(context_len / page_size)
    total_pages = batch_size * pages_per_req
    max_slots = total_pages * page_size

    if shuffle_pages:
        page_perm = torch.randperm(total_pages, device=device, generator=generator)
    else:
        page_perm = torch.arange(total_pages, device=device)

    within = torch.arange(context_len, device=device) % page_size
    page_of_tok = torch.arange(context_len, device=device) // page_size
    phys_page = page_perm.view(batch_size, pages_per_req)[:, page_of_tok]
    req_to_token = (phys_page * page_size + within).to(torch.int32)
    return req_to_token.contiguous(), max_slots


class PrefillInputs(msgspec.Struct):
    q: torch.Tensor
    idx_q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    idx_k_cache: torch.Tensor
    idx_v_cache: Optional[torch.Tensor]
    req_to_token: torch.Tensor
    slot_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    seq_lens: torch.Tensor
    prefix_lens: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    seqlens_cpu: list[int]
    prefix_lens_cpu: list[int]
    k_extend: torch.Tensor
    v_extend: torch.Tensor
    cu_seqblocks_q: torch.Tensor
    max_seqblock_q: int
    all_seqblock_q: int
    num_query_tokens: int


class DecodeInputs(msgspec.Struct):
    q: torch.Tensor
    idx_q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    idx_k_cache: torch.Tensor
    idx_v_cache: Optional[torch.Tensor]
    req_to_token: torch.Tensor
    slot_ids: torch.Tensor
    seq_lens: torch.Tensor
    max_seqlen: int


def _kv_pools(
    cfg: SparseAttnConfig, max_slots: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    dt = cfg.torch_dtype
    k = torch.randn(max_slots, cfg.num_kv_heads, cfg.head_dim, dtype=dt, device=device)
    v = torch.randn(max_slots, cfg.num_kv_heads, cfg.head_dim, dtype=dt, device=device)
    idx_k = torch.randn(max_slots, 1, cfg.idx_head_dim, dtype=dt, device=device)
    idx_v = (
        None
        if cfg.disable_index_value
        else torch.randn(max_slots, 1, cfg.idx_head_dim, dtype=dt, device=device)
    )
    return k, v, idx_k, idx_v


def build_prefill_inputs(
    cfg: SparseAttnConfig,
    *,
    batch_size: int,
    context_len: int,
    chunk_len: Optional[int] = None,
    device: str = "cuda",
    seed: int = 0,
) -> PrefillInputs:
    """One extend step: ``chunk_len`` new tokens on top of a ``context_len - chunk_len`` prefix.

    ``chunk_len=None`` means a single-shot prefill of the whole context, which is
    the worst case for the indexer's O(L^2) score pass.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    chunk = context_len if chunk_len is None else min(chunk_len, context_len)
    prefix = context_len - chunk

    req_to_token, max_slots = build_page_table(
        batch_size=batch_size,
        context_len=context_len,
        page_size=cfg.page_size,
        device=dev,
    )
    k, v, idx_k, idx_v = _kv_pools(cfg, max_slots, dev)

    n_tok = batch_size * chunk
    dt = cfg.torch_dtype
    q = torch.randn(n_tok, cfg.num_q_heads, cfg.head_dim, dtype=dt, device=dev)
    idx_q = torch.randn(
        n_tok, cfg.num_idx_heads, cfg.idx_head_dim, dtype=dt, device=dev
    )

    cu_seqlens = torch.arange(
        0, (batch_size + 1) * chunk, chunk, dtype=torch.int32, device=dev
    )
    seq_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device=dev)
    prefix_lens = torch.full((batch_size,), prefix, dtype=torch.int32, device=dev)
    slot_ids = torch.arange(batch_size, dtype=torch.int64, device=dev)

    cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
        cu_seqlens, chunk, 1, cfg.block_size, [chunk] * batch_size
    )

    # K/V of the extend tokens, which the dense path takes as a direct input
    # (in a server they come straight out of the projection). Gathered here, at
    # build time, so the dense path is not charged for a lookup it never does.
    ext_slots = req_to_token[:, prefix:context_len].reshape(-1).long()
    k_extend = k[ext_slots].contiguous()
    v_extend = v[ext_slots].contiguous()

    return PrefillInputs(
        q=q,
        idx_q=idx_q,
        k_cache=k,
        v_cache=v,
        idx_k_cache=idx_k,
        idx_v_cache=idx_v,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=chunk,
        max_seqlen_k=context_len,
        seqlens_cpu=[chunk] * batch_size,
        prefix_lens_cpu=[prefix] * batch_size,
        k_extend=k_extend,
        v_extend=v_extend,
        cu_seqblocks_q=cu_seqblocks_q,
        max_seqblock_q=max_seqblock_q,
        all_seqblock_q=all_seqblock_q,
        num_query_tokens=n_tok,
    )


def build_decode_inputs(
    cfg: SparseAttnConfig,
    *,
    batch_size: int,
    context_len: int,
    device: str = "cuda",
    seed: int = 0,
) -> DecodeInputs:
    torch.manual_seed(seed)
    dev = torch.device(device)
    req_to_token, max_slots = build_page_table(
        batch_size=batch_size,
        context_len=context_len,
        page_size=cfg.page_size,
        device=dev,
    )
    k, v, idx_k, idx_v = _kv_pools(cfg, max_slots, dev)

    dt = cfg.torch_dtype
    q = torch.randn(batch_size, cfg.num_q_heads, cfg.head_dim, dtype=dt, device=dev)
    idx_q = torch.randn(
        batch_size, cfg.num_idx_heads, cfg.idx_head_dim, dtype=dt, device=dev
    )
    seq_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device=dev)
    slot_ids = torch.arange(batch_size, dtype=torch.int64, device=dev)

    return DecodeInputs(
        q=q,
        idx_q=idx_q,
        k_cache=k,
        v_cache=v,
        idx_k_cache=idx_k,
        idx_v_cache=idx_v,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        seq_lens=seq_lens,
        max_seqlen=context_len,
    )


# ---------------------------------------------------------------------------
# results IO
# ---------------------------------------------------------------------------


def write_results(rows: Sequence[dict], out_dir: Path, name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    csv_path = out_dir / f"{name}.csv"

    json_path.write_text(json.dumps(list(rows), indent=2))

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _nvml_handle_for_visible_device(pynvml):
    """NVML handle for the GPU torch is actually using.

    ``torch.cuda.current_device()`` is an index into the *visible* devices, so
    under ``CUDA_VISIBLE_DEVICES=7`` it is 0 while NVML index 0 is physical GPU
    0. Keying NVML by that index silently watches the wrong GPU: the guard then
    reports a co-tenant that is not there, or — worse — reports idle while the
    measured device is saturated. Resolve by UUID, which is remap-proof.
    """
    dev = torch.cuda.current_device()
    uuid = torch.cuda.get_device_properties(dev).uuid
    return pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode())


def sample_gpu_utilization(samples: int = 5, interval_s: float = 0.1) -> Optional[float]:
    """Mean SM utilization of the visible GPU, sampled before we start timing.

    Foreign load on the same device inflates every latency here — often by
    several-fold, since these kernels are grid-wide. Returns None if NVML is
    unavailable.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = _nvml_handle_for_visible_device(pynvml)
        readings = []
        for _ in range(samples):
            readings.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            time.sleep(interval_s)
        return sum(readings) / len(readings)
    except Exception:
        return None


def wait_for_idle(max_wait_s: float, threshold_pct: float = 10.0) -> None:
    """Block until the target GPU looks idle, or until ``max_wait_s`` elapses.

    On a shared box, co-tenant load does not merely add noise — it inflates
    launch latency enough to swamp these kernels (a 15 us decode step reads as
    500 us). Starting in a clean window is the difference between usable and
    unusable numbers.
    """
    if max_wait_s <= 0:
        return
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        util = sample_gpu_utilization(samples=6, interval_s=0.25)
        if util is None or util <= threshold_pct:
            return
        remaining = deadline - time.time()
        print(
            f"  waiting for an idle GPU: {util:.0f}% busy, "
            f"{remaining:.0f}s left before giving up",
            flush=True,
        )
        time.sleep(5)
    print("  gave up waiting for an idle GPU; proceeding under contention")


def warn_if_contended(threshold_pct: float = 10.0) -> Optional[float]:
    """Print a loud warning when the target GPU is already busy. Returns the util."""
    util = sample_gpu_utilization()
    if util is None:
        print("  [warn] could not read GPU utilization (NVML unavailable)")
        return None
    if util > threshold_pct:
        print(
            f"\n  !! GPU is {util:.0f}% busy with other work before this benchmark "
            f"started.\n"
            f"  !! Latencies below are contended and will read high and noisy "
            f"(2-3x is typical).\n"
            f"  !! Re-run pinned to an idle GPU (CUDA_VISIBLE_DEVICES=<idle id>) "
            f"for numbers you can quote.\n"
        )
    else:
        print(f"  gpu load: {util:.0f}% idle-check OK")
    return util


def gpu_info() -> dict[str, object]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "gpu": props.name,
        "sm": f"{props.major}.{props.minor}",
        "memory_gb": round(props.total_memory / 1024**3, 1),
        "torch": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GiB"


def emit_plots(fn, *args, label: str = "plots", **kwargs) -> None:
    """Run a plotting routine right after a benchmark, without risking its data.

    Plotting is the last thing a benchmark does and the least important: a
    missing matplotlib, a stale schema or an empty sweep must never turn a
    completed measurement into a crashed run with nothing written. Failures are
    reported and swallowed.
    """
    try:
        fn(*args, **kwargs)
    except Exception as err:  # noqa: BLE001 - see docstring
        print(f"  ({label} failed: {type(err).__name__}: {err})")
