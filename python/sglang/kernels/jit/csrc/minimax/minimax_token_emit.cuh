// Pass 2 of the MiniMax-M3 token-sparse fused prefill selector: recompute the
// indexer q.k scores tile by tile and emit every score that clears the per-row
// threshold tau into a bounded per-(head, row) candidate list.
//
// This is the CUDA replacement for the Triton `_emit_above_threshold_kernel` in
// `sglang/kernels/ops/attention/minimax_sparse/token/flash_with_topk_idx_optimized.py`.
// The Triton emitter is instruction-bound: slot assignment (`tl.cumsum`) and the
// masked scatter stores each run on every element of the [rows, context] score
// matrix (~15 ops/element). Here a warp ballot compresses that to ~2 ops per
// element: `__ballot_sync` yields the survivor mask of 32 scores at once, one
// lane reserves the row's slots with a single atomic, and `__popc` prefix sums
// place the (value, position) pairs.
//
// Contract (matches the Triton emitter; verified in
// test/registered/jit/test_minimax_token_emit.py). One known deviation: wmma
// and Triton's wgmma lowering round the same bf16 dot differently by ~1 ulp,
// so when the two scores straddling rank topk agree to ~1e-8 (about one row
// in ~10^5 with random data) the selected boundary token can differ from the
// reference by that near-tied swap. The test and the bench adjudicate exactly
// this case in fp64 and reject anything larger.
//   * score  = q.k * sm_scale * log2(e), fp32 accumulation
//   * forced positions: pos <  init_tokens          -> INIT_BIAS
//                       pos >  abs_q - local_tokens -> LOCAL_BIAS (if not init)
//   * keep   = causal && score >= tau[head, row]
//   * cnt[head, row] counts every survivor (may exceed cap; the host detects
//     overflow and falls back), entries beyond cap are dropped, order in the
//     candidate list is unspecified.
//
// Shape restrictions (checked in the launcher): qk_head_dim == 128, q/k in
// bf16 or fp16 (the fp8 index cache falls back to the Triton path).

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For CHECK_HOST, div_ceil
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, bf16_t/fp16_t, CHECK_CUDA

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <math_constants.h>
#include <mma.h>

#include <cstdint>

namespace {

// Bias magnitudes, mirroring index_score.INIT_BIAS / LOCAL_BIAS. Passed as
// kernel arguments from Python so the two definitions cannot drift.

constexpr uint32_t kHeadDim = 128;   // the only idx head dim MiniMax-M3 ships
constexpr uint32_t kBlockQ = 64;     // query rows per CTA
constexpr uint32_t kBlockK = 128;    // key positions per CTA
constexpr uint32_t kNumWarps = 4;    // one 16-row wmma strip per warp
constexpr uint32_t kCTASize = kNumWarps * device::kWarpThreads;
constexpr uint32_t kWmmaTile = 16;

// Shared memory layout. K rows are padded to 136 elements (272 B) and score
// rows to 132 floats (528 B) to keep 16-byte alignment and stagger banks.
constexpr uint32_t kPadK = kHeadDim + 8;    // halves per K/Q shared row
constexpr uint32_t kPadS = kBlockK + 4;     // floats per score shared row
constexpr uint32_t kSlotsBytes = kBlockK * sizeof(int32_t);
constexpr uint32_t kKBytes = kBlockK * kPadK * 2;
constexpr uint32_t kQBytes = kBlockQ * kPadK * 2;
constexpr uint32_t kScoreBytes = kBlockQ * kPadS * sizeof(float);
// Q is only read while the wmma accumulators are being built and the score
// tile is only written after a barrier, so the two overlay the same region.
constexpr uint32_t kUnionBytes = kQBytes > kScoreBytes ? kQBytes : kScoreBytes;
constexpr uint32_t kSmemBytes = kSlotsBytes + kKBytes + kUnionBytes;

template <typename T>
__global__ void __launch_bounds__(kCTASize) minimax_token_emit_kernel(
    const T* __restrict__ idx_q,          // [total_q, num_heads, 128]
    const T* __restrict__ k_cache,        // [max_slots, 1, 128]
    const int32_t* __restrict__ req_to_token,  // [max_reqs, r2t_width]
    const int32_t* __restrict__ cu_seqlens,    // [batch + 1]
    const int32_t* __restrict__ seq_lens,      // [batch]
    const int32_t* __restrict__ prefix_lens,   // [batch]
    const int32_t* __restrict__ slot_ids,      // [batch]
    const float* __restrict__ tau,        // [num_heads, total_q]
    float* __restrict__ cand_val,         // [num_heads, total_q, cap]
    int32_t* __restrict__ cand_pos,       // [num_heads, total_q, cap]
    int32_t* __restrict__ cnt,            // [num_heads, total_q]
    const int64_t total_q,
    const int32_t num_heads,
    const int64_t max_slots,
    const int64_t r2t_width,
    const int32_t kv_width,
    const int32_t cap,
    const int32_t init_tokens,
    const int32_t local_tokens,
    const float sm_scale_log2e,
    const float init_bias,
    const float local_bias) {
  using namespace nvcuda;
  extern __shared__ char smem_raw[];
  int32_t* s_slots = reinterpret_cast<int32_t*>(smem_raw);
  T* s_k = reinterpret_cast<T*>(smem_raw + kSlotsBytes);
  T* s_q = reinterpret_cast<T*>(smem_raw + kSlotsBytes + kKBytes);
  float* s_score = reinterpret_cast<float*>(smem_raw + kSlotsBytes + kKBytes);

  const uint32_t pid_k = blockIdx.x;
  const uint32_t pid_b = blockIdx.y;
  const uint32_t pid_q = blockIdx.z;
  const uint32_t tx = threadIdx.x;
  const uint32_t warp_id = tx / device::kWarpThreads;
  const uint32_t lane_id = tx % device::kWarpThreads;

  const int32_t seq_start = cu_seqlens[pid_b];
  const int32_t q_len = cu_seqlens[pid_b + 1] - seq_start;
  const int32_t seq_len = seq_lens[pid_b];
  const int32_t prefix_len = prefix_lens[pid_b];
  const int64_t sid =
      (static_cast<int64_t>(slot_ids[pid_b]) + max_slots) % max_slots;

  const int32_t q_lo = pid_q * kBlockQ;
  const int32_t q_end = q_lo + static_cast<int32_t>(kBlockQ);
  const int32_t q_hi = q_end < q_len ? q_end : q_len;
  if (q_lo >= q_hi) return;
  const int32_t q_rows = q_hi - q_lo;

  const int32_t tile_lo = pid_k * kBlockK;
  // Nothing in this tile is causal for any of this CTA's rows, or the whole
  // tile is past the sequence: skip before touching shared memory.
  if (tile_lo > prefix_len + q_hi - 1 || tile_lo >= seq_len) return;

  // --- stage the page-table slots, then gather the K tile ------------------
  if (tx < kBlockK) {
    const int32_t pos = tile_lo + static_cast<int32_t>(tx);
    const bool valid = pos < seq_len && pos < kv_width;
    int32_t slot = -1;
    if (valid) {
      const int64_t raw = req_to_token[sid * r2t_width + pos];
      slot = static_cast<int32_t>((raw + max_slots) % max_slots);
    }
    s_slots[tx] = slot;
  }
  __syncthreads();

  constexpr uint32_t kSegPerRow = kHeadDim * sizeof(T) / sizeof(uint4);
#pragma unroll
  for (uint32_t i = tx; i < kBlockK * kSegPerRow; i += kCTASize) {
    const uint32_t key = i / kSegPerRow;
    const uint32_t seg = i % kSegPerRow;
    const int32_t slot = s_slots[key];
    uint4 v = {0u, 0u, 0u, 0u};
    if (slot >= 0) {
      v = reinterpret_cast<const uint4*>(k_cache +
                                         static_cast<int64_t>(slot) * kHeadDim)[seg];
    }
    *reinterpret_cast<uint4*>(reinterpret_cast<char*>(s_k) +
                              key * kPadK * sizeof(T) + seg * sizeof(uint4)) = v;
  }

  // Loop heads inside the CTA: the gathered K tile in shared memory is reused
  // for every head. The grid-folded alternative (one CTA per head) was
  // measured 37% slower — the sibling CTAs' 4x K re-gather does not stay
  // resident in L2 at long context.
  for (int32_t h = 0; h < num_heads; ++h) {
    // --- load this head's Q strip (zero-padded past q_rows) ----------------
    __syncthreads();  // previous head's score reads are done; safe to overwrite
#pragma unroll
    for (uint32_t i = tx; i < kBlockQ * kSegPerRow; i += kCTASize) {
      const uint32_t row = i / kSegPerRow;
      const uint32_t seg = i % kSegPerRow;
      uint4 v = {0u, 0u, 0u, 0u};
      if (static_cast<int32_t>(row) < q_rows) {
        const int64_t g_row = seq_start + q_lo + static_cast<int32_t>(row);
        v = reinterpret_cast<const uint4*>(idx_q +
                                           (g_row * num_heads + h) * kHeadDim)[seg];
      }
      *reinterpret_cast<uint4*>(reinterpret_cast<char*>(s_q) +
                                row * kPadK * sizeof(T) + seg * sizeof(uint4)) = v;
    }
    __syncthreads();

    // --- q.k for the warp's 16-row strip, fp32 accumulators ----------------
    wmma::fragment<wmma::accumulator, kWmmaTile, kWmmaTile, kWmmaTile, float>
        acc[kBlockK / kWmmaTile];
#pragma unroll
    for (auto& frag : acc) {
      wmma::fill_fragment(frag, 0.0f);
    }
    const uint32_t warp_row = warp_id * kWmmaTile;
#pragma unroll
    for (uint32_t kk = 0; kk < kHeadDim / kWmmaTile; ++kk) {
      wmma::fragment<wmma::matrix_a, kWmmaTile, kWmmaTile, kWmmaTile, T,
                     wmma::row_major>
          a_frag;
      wmma::load_matrix_sync(a_frag, s_q + warp_row * kPadK + kk * kWmmaTile,
                             kPadK);
#pragma unroll
      for (uint32_t n = 0; n < kBlockK / kWmmaTile; ++n) {
        // s_k is [key][dim], i.e. column-major (dim, key) with ld = kPadK.
        wmma::fragment<wmma::matrix_b, kWmmaTile, kWmmaTile, kWmmaTile, T,
                       wmma::col_major>
            b_frag;
        wmma::load_matrix_sync(b_frag, s_k + (n * kWmmaTile) * kPadK + kk * kWmmaTile,
                               kPadK);
        wmma::mma_sync(acc[n], a_frag, b_frag, acc[n]);
      }
    }

    // Every warp is done reading s_q; the score tile may now overlay it.
    __syncthreads();
#pragma unroll
    for (uint32_t n = 0; n < kBlockK / kWmmaTile; ++n) {
      wmma::store_matrix_sync(s_score + warp_row * kPadS + n * kWmmaTile, acc[n],
                              kPadS, wmma::mem_row_major);
    }

    // --- ballot-compacted emission of scores >= tau ------------------------
    // Each warp owns the 16 rows it just computed, so no CTA barrier is
    // needed — but the fragment store scatters elements across lanes, and the
    // reads below cross lanes, so the warp itself must synchronize.
    __syncwarp();

    const auto keep_val = [&](uint32_t r, uint32_t c, float tau_r,
                              float& v) -> bool {
      const uint32_t col = c * device::kWarpThreads + lane_id;
      const int32_t pos = tile_lo + static_cast<int32_t>(col);
      const int32_t abs_q = prefix_len + q_lo + static_cast<int32_t>(r);
      const bool causal = pos <= abs_q && pos < seq_len && pos < kv_width;
      v = s_score[r * kPadS + col] * sm_scale_log2e;
      if (pos < init_tokens) {
        v = init_bias;
      } else if (pos > abs_q - local_tokens) {
        v = local_bias;
      }
      return causal && v >= tau_r;
    };

    // Lane r < 16 holds row (warp_row + r)'s threshold, survivor count and
    // list base, so the 16 row counters are reserved by ONE warp-wide
    // atomicAdd on distinct addresses instead of a serialized round trip per
    // (row, 32-column chunk) — those chains dominated this kernel's runtime.
    const int32_t lane_row = warp_row + lane_id;
    const bool lane_has_row =
        lane_id < kWmmaTile && static_cast<int32_t>(lane_row) < q_rows;
    float tau_lane = CUDART_INF_F;
    if (lane_has_row) {
      tau_lane = tau[h * total_q + (seq_start + q_lo + static_cast<int32_t>(lane_row))];
    }

    // Pass A: count survivors per row; lane r keeps row r's total.
    int32_t cnt_lane = 0;
#pragma unroll
    for (uint32_t r0 = 0; r0 < kWmmaTile; ++r0) {
      const float tau_r = __shfl_sync(0xFFFFFFFFu, tau_lane, r0);
      int32_t row_cnt = 0;
#pragma unroll
      for (uint32_t c = 0; c < kBlockK / device::kWarpThreads; ++c) {
        float v;
        const bool keep = keep_val(warp_row + r0, c, tau_r, v);
        row_cnt += __popc(__ballot_sync(0xFFFFFFFFu, keep));
      }
      if (lane_id == r0) cnt_lane = row_cnt;
    }

    // Pass B: one warp-parallel reservation for all 16 rows.
    int32_t base_lane = 0;
    if (lane_has_row && cnt_lane > 0) {
      base_lane = atomicAdd(
          cnt + h * total_q + (seq_start + q_lo + static_cast<int32_t>(lane_row)),
          cnt_lane);
    }

    // Pass C: re-derive the survivor masks and place (value, position).
#pragma unroll
    for (uint32_t r0 = 0; r0 < kWmmaTile; ++r0) {
      if (__shfl_sync(0xFFFFFFFFu, cnt_lane, r0) == 0) continue;
      int32_t base = __shfl_sync(0xFFFFFFFFu, base_lane, r0);
      const float tau_r = __shfl_sync(0xFFFFFFFFu, tau_lane, r0);
      const int64_t g_row = seq_start + q_lo + static_cast<int32_t>(warp_row + r0);
      const int64_t list_base = (h * total_q + g_row) * cap;
#pragma unroll
      for (uint32_t c = 0; c < kBlockK / device::kWarpThreads; ++c) {
        float v;
        const bool keep = keep_val(warp_row + r0, c, tau_r, v);
        const uint32_t ballot = __ballot_sync(0xFFFFFFFFu, keep);
        if (keep) {
          const int32_t slot = base + __popc(ballot & ((1u << lane_id) - 1u));
          if (slot < cap) {
            cand_val[list_base + slot] = v;
            cand_pos[list_base + slot] =
                tile_lo + static_cast<int32_t>(c * device::kWarpThreads + lane_id);
          }
        }
        base += __popc(ballot);
      }
    }
  }
}

template <typename T>
void minimax_token_emit(
    tvm::ffi::TensorView idx_q,
    tvm::ffi::TensorView idx_k_cache,
    tvm::ffi::TensorView req_to_token,
    tvm::ffi::TensorView cu_seqlens,
    tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView prefix_lens,
    tvm::ffi::TensorView slot_ids,
    tvm::ffi::TensorView tau,
    tvm::ffi::TensorView cand_val,
    tvm::ffi::TensorView cand_pos,
    tvm::ffi::TensorView cnt,
    int64_t max_seqlen_q,
    int64_t kv_width,
    int64_t init_tokens,
    int64_t local_tokens,
    double sm_scale,
    double init_bias,
    double local_bias) {
  using namespace host;

  SymbolicSize NQ{"total_q"}, H{"num_idx_heads"}, D{"qk_head_dim"};
  SymbolicSize S{"max_slots"}, One{"one"}, R{"max_reqs"}, W{"r2t_width"};
  SymbolicSize B{"batch"}, Bp1{"batch_plus_1"}, CAP{"cand_cap"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({NQ, H, D}).with_dtype<T>().with_device(device_).verify(idx_q);
  TensorMatcher({S, One, D}).with_dtype<T>().with_device(device_).verify(idx_k_cache);
  TensorMatcher({R, W}).with_dtype<int32_t>().with_device(device_).verify(req_to_token);
  TensorMatcher({Bp1}).with_dtype<int32_t>().with_device(device_).verify(cu_seqlens);
  TensorMatcher({B})
      .with_dtype<int32_t>()
      .with_device(device_)
      .verify(seq_lens)
      .verify(prefix_lens)
      .verify(slot_ids);
  TensorMatcher({H, NQ}).with_dtype<fp32_t>().with_device(device_).verify(tau);
  TensorMatcher({H, NQ, CAP}).with_dtype<fp32_t>().with_device(device_).verify(cand_val);
  TensorMatcher({H, NQ, CAP}).with_dtype<int32_t>().with_device(device_).verify(cand_pos);
  TensorMatcher({H, NQ}).with_dtype<int32_t>().with_device(device_).verify(cnt);

  CHECK_HOST(D.unwrap() == kHeadDim)
      << "minimax_token_emit requires qk_head_dim == " << kHeadDim << ", got "
      << D.unwrap();
  CHECK_HOST(One.unwrap() == 1) << "index K cache must have a single KV head";
  CHECK_HOST(Bp1.unwrap() == B.unwrap() + 1)
      << "cu_seqlens must have batch + 1 entries";
  CHECK_HOST(kv_width <= W.unwrap())
      << "kv_width " << kv_width << " exceeds req_to_token width " << W.unwrap();

  const auto kernel = minimax_token_emit_kernel<T>;
  static const bool smem_ok = [&] {
    CHECK_CUDA(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes))
        << "raising the dynamic smem limit to " << kSmemBytes << " B";
    return true;
  }();
  (void)smem_ok;

  const dim3 grid(div_ceil<uint32_t>(kv_width, kBlockK),
                  static_cast<uint32_t>(B.unwrap()),
                  div_ceil<uint32_t>(max_seqlen_q, kBlockQ));
  LaunchKernel(grid, kCTASize, device_.unwrap(), kSmemBytes)(
      kernel,
      static_cast<const T*>(idx_q.data_ptr()),
      static_cast<const T*>(idx_k_cache.data_ptr()),
      static_cast<const int32_t*>(req_to_token.data_ptr()),
      static_cast<const int32_t*>(cu_seqlens.data_ptr()),
      static_cast<const int32_t*>(seq_lens.data_ptr()),
      static_cast<const int32_t*>(prefix_lens.data_ptr()),
      static_cast<const int32_t*>(slot_ids.data_ptr()),
      static_cast<const float*>(tau.data_ptr()),
      static_cast<float*>(cand_val.data_ptr()),
      static_cast<int32_t*>(cand_pos.data_ptr()),
      static_cast<int32_t*>(cnt.data_ptr()),
      static_cast<int64_t>(NQ.unwrap()),
      static_cast<int32_t>(H.unwrap()),
      static_cast<int64_t>(S.unwrap()),
      static_cast<int64_t>(W.unwrap()),
      static_cast<int32_t>(kv_width),
      static_cast<int32_t>(CAP.unwrap()),
      static_cast<int32_t>(init_tokens),
      static_cast<int32_t>(local_tokens),
      // fp32(sm_scale) * fp32(log2e), exactly as the Triton kernels compute
      // it — the double-precision product rounds differently by 1 ulp, which
      // is enough to flip a rank-k boundary pair against the reference.
      static_cast<float>(sm_scale) * 1.4426950408889634f,
      static_cast<float>(init_bias),
      static_cast<float>(local_bias));
}

}  // namespace
