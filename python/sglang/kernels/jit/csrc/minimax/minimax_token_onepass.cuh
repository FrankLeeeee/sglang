// One-pass MiniMax-M3 token-sparse prefill selector (prototype).
//
// The two-pass fused selector (minimax_token_emit.cuh) computes q.k twice:
// once to derive a per-row threshold from pooled maxima, once to emit the
// scores that clear it. This kernel does both in a single pass over the index
// K cache: scores go from Hopper WGMMA accumulators into bounded per-(head,
// row, key-segment) candidate lists guarded by *running* thresholds, and the
// full score matrix never exists.
//
//   * Layout: one 128-thread warpgroup computes a 64-row x 128-key tile with
//     SM90 WGMMA. Q and gathered K are staged directly into SW128 shared-memory
//     layouts with cp.async, and K is reused across all index heads. Grid =
//     (query strips, batch, key segments): each CTA owns one segment for its
//     rows, so selection state is CTA-private.
//   * Per (head, row, segment): a candidate list in global memory plus a
//     running threshold tau in shared memory. tau starts at -inf; emission is
//     the emit kernel's warp-ballot append. When a list nears capacity the
//     CTA compacts it with a two-level (16-bit) radix select over the
//     order-preserving float bits: tau rises to the floor of the bin holding
//     rank topk, which is <= the segment's true topk-th score seen so far —
//     so each list stays a superset of its segment's top-k, and the union
//     over segments a superset of the row's global top-k (any global top-k
//     member is in its own segment's top-k). The host-side exact top-k over
//     the union therefore matches the materializing path, modulo the same
//     ~1-ulp tensor-core boundary ties as the emit kernel.
//   * Expected appends per segment are ~topk * (1 + ln(seg_len / topk)); a
//     compaction that cannot shrink a list (a tie mass wider than capacity)
//     sets the overflow flag and the host falls back.
//
// Shape restrictions (checked in the launcher): qk_head_dim == 128, bf16/fp16,
// num_heads <= 4, cap >= topk + 2 * kBlockK.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For CHECK_HOST, div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, bf16_t/fp16_t, CHECK_CUDA

#include <cute/tensor.hpp>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "../sparse_mla_q8kv8_prefill_sm90/helpers.h"
#include <cstdint>
#include <math_constants.h>
#include <type_traits>

namespace {

constexpr uint32_t kOnepassHeadDim = 128;  // the only idx head dim M3 ships
constexpr uint32_t kOnepassBlockQ = 64;    // query rows per CTA
constexpr uint32_t kOnepassBlockK = 128;   // key positions per tile
constexpr uint32_t kOnepassWarps = 4;      // one Hopper warpgroup
constexpr uint32_t kOnepassCTA = kOnepassWarps * device::kWarpThreads;
constexpr uint32_t kOnepassMaxHeads = 4;
constexpr uint32_t kOnepassTile = 16;  // rows emitted by each warp

// Hopper GMMA uses exact swizzled Q/K tiles. The score buffer aliases Q after
// the asynchronous GMMA read completes and enables coalesced full-warp emits.
constexpr uint32_t kOnepassSlotsBytes = kOnepassBlockK * sizeof(int32_t);
constexpr uint32_t kOnepassKBytes = kOnepassBlockK * kOnepassHeadDim * 2;
constexpr uint32_t kOnepassPadS = kOnepassBlockK + 4;
constexpr uint32_t kOnepassScoreBytes = kOnepassBlockQ * kOnepassPadS * sizeof(float);
constexpr uint32_t kOnepassUnionBytes = kOnepassScoreBytes;
constexpr uint32_t kOnepassHistBytes = kOnepassWarps * 256 * sizeof(int32_t);  // one histogram per warp
constexpr uint32_t kOnepassQueueBytes = kOnepassMaxHeads * kOnepassBlockQ * sizeof(int32_t);
constexpr uint32_t kOnepassStateBytes = kOnepassMaxHeads * kOnepassBlockQ * (sizeof(float) + sizeof(int32_t));
constexpr uint32_t kOnepassMiscBytes = 4 * sizeof(int32_t);
constexpr uint32_t kOnepassSmemBytes = kOnepassSlotsBytes + kOnepassKBytes + kOnepassUnionBytes + kOnepassHistBytes +
                                       kOnepassQueueBytes + kOnepassStateBytes + kOnepassMiscBytes;

// Order-preserving float <-> uint mapping for the radix select.
SGL_DEVICE uint32_t onepass_ordered(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

SGL_DEVICE float onepass_unorder(uint32_t u) {
  return (u & 0x80000000u) ? __uint_as_float(u & 0x7fffffffu) : __uint_as_float(~u);
}

SGL_DEVICE void onepass_cp_async_16(void* dst, const void* src, bool valid) {
  const uint32_t dst_smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst_smem), "l"(src), "r"(valid ? 16 : 0));
}

SGL_DEVICE void onepass_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

SGL_DEVICE void onepass_cp_async_wait() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

template <typename T>
__global__ void __launch_bounds__(kOnepassCTA) minimax_token_onepass_kernel(
    const T* __restrict__ idx_q,               // [total_q, num_heads, 128]
    const T* __restrict__ k_cache,             // [max_slots, 1, 128]
    const int32_t* __restrict__ req_to_token,  // [max_reqs, r2t_width]
    const int32_t* __restrict__ cu_seqlens,    // [batch + 1]
    const int32_t* __restrict__ seq_lens,      // [batch]
    const int32_t* __restrict__ prefix_lens,   // [batch]
    const int32_t* __restrict__ slot_ids,      // [batch]
    float* __restrict__ cand_val,              // [num_heads, total_q, num_segs, cap]
    int32_t* __restrict__ cand_pos,            // [num_heads, total_q, num_segs, cap]
    int32_t* __restrict__ cnt_out,             // [num_heads, total_q, num_segs]
    int32_t* __restrict__ overflow,            // [1]
    const int64_t total_q,
    const int32_t num_heads,
    const int64_t max_slots,
    const int64_t r2t_width,
    const int32_t kv_width,
    const int32_t cap,
    const int32_t topk,
    const int32_t init_tokens,
    const int32_t local_tokens,
    const float sm_scale_log2e,
    const float init_bias,
    const float local_bias) {
  extern __shared__ char smem_raw[];
  int32_t* s_slots = reinterpret_cast<int32_t*>(smem_raw);
  T* s_k = reinterpret_cast<T*>(smem_raw + kOnepassSlotsBytes);
  T* s_q = reinterpret_cast<T*>(smem_raw + kOnepassSlotsBytes + kOnepassKBytes);
  float* s_score = reinterpret_cast<float*>(smem_raw + kOnepassSlotsBytes + kOnepassKBytes);
  int32_t* s_hist = reinterpret_cast<int32_t*>(smem_raw + kOnepassSlotsBytes + kOnepassKBytes + kOnepassUnionBytes);
  int32_t* s_queue = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(s_hist) + kOnepassHistBytes);
  float* s_tau = reinterpret_cast<float*>(reinterpret_cast<char*>(s_queue) + kOnepassQueueBytes);
  int32_t* s_cnt =
      reinterpret_cast<int32_t*>(reinterpret_cast<char*>(s_tau) + kOnepassMaxHeads * kOnepassBlockQ * sizeof(float));
  int32_t* s_misc =
      reinterpret_cast<int32_t*>(reinterpret_cast<char*>(s_cnt) + kOnepassMaxHeads * kOnepassBlockQ * sizeof(int32_t));
  // s_misc[0] = radix bin, s_misc[1] = compaction write count,
  // s_misc[2] = CTA overflow flag, s_misc[3] = survivors above level-1 bin

  const uint32_t pid_q = blockIdx.z;
  const uint32_t pid_b = blockIdx.y;
  const uint32_t seg = blockIdx.x;
  const uint32_t num_segs = gridDim.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t warp_id = tx / device::kWarpThreads;
  const uint32_t lane_id = tx % device::kWarpThreads;

  const int32_t seq_start = cu_seqlens[pid_b];
  const int32_t q_len = cu_seqlens[pid_b + 1] - seq_start;
  const int32_t seq_len = seq_lens[pid_b];
  const int32_t prefix_len = prefix_lens[pid_b];
  const int64_t sid = (static_cast<int64_t>(slot_ids[pid_b]) + max_slots) % max_slots;

  const int32_t q_lo = pid_q * kOnepassBlockQ;
  const int32_t q_end = q_lo + static_cast<int32_t>(kOnepassBlockQ);
  const int32_t q_hi = q_end < q_len ? q_end : q_len;
  if (q_lo >= q_hi) return;
  const int32_t q_rows = q_hi - q_lo;

  // No key beyond the last row's causal frontier (or the sequence) matters.
  int32_t key_hi = prefix_len + q_hi;
  if (seq_len < key_hi) key_hi = seq_len;
  if (kv_width < key_hi) key_hi = kv_width;

  // This CTA's slice of the key axis, in whole tiles.
  const int32_t tiles_total =
      (key_hi + static_cast<int32_t>(kOnepassBlockK) - 1) / static_cast<int32_t>(kOnepassBlockK);
  const int32_t tiles_per_seg = (tiles_total + static_cast<int32_t>(num_segs) - 1) / static_cast<int32_t>(num_segs);
  const int32_t seg_tile_lo = static_cast<int32_t>(seg) * tiles_per_seg;
  const int32_t seg_tile_hi = seg_tile_lo + tiles_per_seg < tiles_total ? seg_tile_lo + tiles_per_seg : tiles_total;
  if (seg_tile_lo >= seg_tile_hi) {
    // Empty segment (short context): the host reads the buffers unmasked, so
    // neutralize this CTA's lists before leaving.
    for (int32_t h = 0; h < num_heads; ++h) {
      for (int32_t r = 0; r < q_rows; ++r) {
        const int64_t g_row = seq_start + q_lo + r;
        const int64_t list_base = ((static_cast<int64_t>(h) * total_q + g_row) * num_segs + seg) * cap;
        for (int32_t i = tx; i < cap; i += kOnepassCTA) {
          cand_val[list_base + i] = -CUDART_INF_F;
        }
      }
    }
    return;
  }

  for (uint32_t i = tx; i < kOnepassMaxHeads * kOnepassBlockQ; i += kOnepassCTA) {
    s_tau[i] = -CUDART_INF_F;
    s_cnt[i] = 0;
  }
  if (tx == 0) s_misc[2] = 0;

  using CuteT = std::conditional_t<std::is_same_v<T, bf16_t>, cute::bfloat16_t, cute::half_t>;
  constexpr uint32_t kCutePerVec = sizeof(uint4) / sizeof(CuteT);
  using OnepassQLayout = decltype(cute::coalesce(
      cute::tile_to_shape(
          cute::GMMA::Layout_K_SW128_Atom<CuteT>{},
          cute::Shape<cute::Int<kOnepassBlockQ>, cute::Int<kOnepassHeadDim>>{},
          cute::Step<cute::_1, cute::_2>{}),
      cute::Shape<cute::_1, cute::_1>{}));
  using OnepassKLayout = decltype(cute::coalesce(
      cute::tile_to_shape(
          cute::GMMA::Layout_K_SW128_Atom<CuteT>{},
          cute::Shape<cute::Int<kOnepassBlockK>, cute::Int<kOnepassHeadDim>>{},
          cute::Step<cute::_1, cute::_2>{}),
      cute::Shape<cute::_1, cute::_1>{}));
  auto s_q_gmma = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<CuteT*>(s_q)), OnepassQLayout{});
  auto s_k_gmma = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<CuteT*>(s_k)), OnepassKLayout{});

  for (int32_t tile = seg_tile_lo; tile < seg_tile_hi; ++tile) {
    const int32_t tile_lo = tile * static_cast<int32_t>(kOnepassBlockK);
    // Previous iteration's compaction / emission is done with smem.
    __syncthreads();

    // --- stage the page-table slots, then gather the K tile ----------------
    if (tx < kOnepassBlockK) {
      const int32_t pos = tile_lo + static_cast<int32_t>(tx);
      int32_t slot = -1;
      if (pos < key_hi) {
        const int64_t raw = req_to_token[sid * r2t_width + pos];
        slot = static_cast<int32_t>((raw + max_slots) % max_slots);
      }
      s_slots[tx] = slot;
    }
    __syncthreads();

    for (uint32_t i = tx; i < kOnepassBlockK * (kOnepassHeadDim / kCutePerVec); i += kOnepassCTA) {
      const uint32_t key = i / (kOnepassHeadDim / kCutePerVec);
      const uint32_t vec = i % (kOnepassHeadDim / kCutePerVec);
      const int32_t slot = s_slots[key];
      void* dst = &s_k_gmma(key, vec * kCutePerVec);
      const void* src =
          slot >= 0 ? reinterpret_cast<const uint4*>(k_cache + static_cast<int64_t>(slot) * kOnepassHeadDim) + vec
                    : static_cast<const void*>(k_cache);
      onepass_cp_async_16(dst, src, slot >= 0);
    }
    onepass_cp_async_commit();

    for (int32_t h = 0; h < num_heads; ++h) {
      // Score emission for the previous head is complete before the shared Q
      // tile is overwritten for the next head.
      if (h > 0) __syncthreads();
      for (uint32_t i = tx; i < kOnepassBlockQ * (kOnepassHeadDim / kCutePerVec); i += kOnepassCTA) {
        const uint32_t row = i / (kOnepassHeadDim / kCutePerVec);
        const uint32_t vec = i % (kOnepassHeadDim / kCutePerVec);
        const bool valid = static_cast<int32_t>(row) < q_rows;
        const int64_t g_row = seq_start + q_lo + static_cast<int32_t>(row);
        void* dst = &s_q_gmma(row, vec * kCutePerVec);
        const void* src = valid
                              ? reinterpret_cast<const uint4*>(idx_q + (g_row * num_heads + h) * kOnepassHeadDim) + vec
                              : static_cast<const void*>(idx_q);
        onepass_cp_async_16(dst, src, valid);
      }
      onepass_cp_async_commit();
      onepass_cp_async_wait();
      __syncthreads();
      cutlass::arch::fence_view_async_shared();

      // --- Hopper warpgroup q.k: 64 queries x 128 keys x 128 dims ----------
      using OnepassWgmmaAtom = std::conditional_t<
          std::is_same_v<T, bf16_t>,
          cute::GMMA::MMA_64x128x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>,
          cute::GMMA::MMA_64x128x16_F32F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>>;
      using OnepassTiledMma =
          decltype(cute::make_tiled_mma(OnepassWgmmaAtom{}, cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>{}));
      auto acc = cute::partition_fragment_C(
          OnepassTiledMma{}, cute::Shape<cute::Int<kOnepassBlockQ>, cute::Int<kOnepassBlockK>>{});
      sm90::gemm_ss(true, OnepassTiledMma{}, s_q_gmma, s_k_gmma, acc, static_cast<int>(tx));
      cute::warpgroup_commit_batch();
      cute::warpgroup_wait<0>();
      cute::warpgroup_fence_operand(acc);

      // Store the GMMA accumulator in row-major form for full-warp emission.
      // The CTA barrier also prevents the aliased score tile from overwriting
      // Q until every warp has completed its asynchronous GMMA reads.
      __syncthreads();
#pragma unroll
      for (int32_t row_i = 0; row_i < 2; ++row_i) {
        const int32_t row = (static_cast<int32_t>(tx) / 32) * 16 + row_i * 8 + (static_cast<int32_t>(tx) % 32) / 4;
#pragma unroll
        for (int32_t i = row_i * 2; i < cute::size(acc); i += 4) {
          const int32_t col = 8 * (i / 4) + (static_cast<int32_t>(tx) % 4) * 2;
          s_score[row * kOnepassPadS + col] = acc(i);
          s_score[row * kOnepassPadS + col + 1] = acc(i + 1);
        }
      }
      __syncthreads();

      const uint32_t warp_row = warp_id * kOnepassTile;
#pragma unroll
      for (uint32_t r0 = 0; r0 < kOnepassTile; ++r0) {
        const int32_t row = static_cast<int32_t>(warp_row + r0);
        if (row >= q_rows) break;
        const int32_t abs_q = prefix_len + q_lo + row;
        const float tau_r = s_tau[h * kOnepassBlockQ + row];
        const int64_t g_row = seq_start + q_lo + row;
        const int64_t list_base = ((static_cast<int64_t>(h) * total_q + g_row) * num_segs + seg) * cap;
        int32_t base = 0;
        if (lane_id == 0) {
          base = s_cnt[h * kOnepassBlockQ + row];
        }
        base = __shfl_sync(0xFFFFFFFFu, base, 0);
#pragma unroll
        for (uint32_t c = 0; c < kOnepassBlockK / device::kWarpThreads; ++c) {
          const uint32_t col = c * device::kWarpThreads + lane_id;
          const int32_t pos = tile_lo + static_cast<int32_t>(col);
          float v = s_score[(warp_row + r0) * kOnepassPadS + col] * sm_scale_log2e;
          if (pos < init_tokens) {
            v = init_bias;
          } else if (pos > abs_q - local_tokens) {
            v = local_bias;
          }
          const bool causal = pos <= abs_q && pos < key_hi;
          const bool keep = causal && v >= tau_r;
          const uint32_t ballot = __ballot_sync(0xFFFFFFFFu, keep);
          if (keep) {
            const int32_t slot = base + __popc(ballot & ((1u << lane_id) - 1u));
            if (slot < cap) {
              cand_val[list_base + slot] = v;
              cand_pos[list_base + slot] = pos;
            }
          }
          base += __popc(ballot);
        }
        if (lane_id == 0) {
          s_cnt[h * kOnepassBlockQ + row] = base;
        }
      }
    }

    // --- compaction: raise tau for any row whose list is nearly full -------
    // Pending lists are queued, then each warp compacts a different list
    // independently (per-warp histogram, warp-synchronous rewrite) — no CTA
    // barriers on the hot path.
    __syncthreads();
    if (tx == 0) s_misc[0] = 0;  // queue length
    __syncthreads();
    for (uint32_t i = tx; i < static_cast<uint32_t>(num_heads) * kOnepassBlockQ; i += kOnepassCTA) {
      const int32_t r = i % kOnepassBlockQ;
      if (r < q_rows && s_cnt[i] > cap - static_cast<int32_t>(kOnepassBlockK)) {
        s_queue[atomicAdd(&s_misc[0], 1)] = static_cast<int32_t>(i);
      }
    }
    __syncthreads();
    const int32_t n_pending = s_misc[0];
    for (int32_t qi = static_cast<int32_t>(warp_id); qi < n_pending; qi += static_cast<int32_t>(kOnepassWarps)) {
      const int32_t state_i = s_queue[qi];
      const int32_t h = state_i / static_cast<int32_t>(kOnepassBlockQ);
      const int32_t r = state_i % static_cast<int32_t>(kOnepassBlockQ);
      // The count may briefly exceed cap when a compaction could not shrink
      // the list (overflow flag set); clamp so reads stay in bounds.
      const int32_t n_raw = s_cnt[state_i];
      const int32_t n = n_raw < cap ? n_raw : cap;
      const int64_t g_row = seq_start + q_lo + r;
      const int64_t list_base = ((static_cast<int64_t>(h) * total_q + g_row) * num_segs + seg) * cap;
      int32_t* hist = s_hist + warp_id * 256;

      // Two-level radix select over the order-preserving bits: real score
      // distributions concentrate in a few exponent buckets, so the top byte
      // alone often cannot separate rank topk — refine with the next byte
      // for a 16-bit threshold prefix.
      uint32_t thr_u = 0;
      int32_t above_hi = 0;
#pragma unroll
      for (int level = 0; level < 2; ++level) {
        for (uint32_t i = lane_id; i < 256; i += device::kWarpThreads) {
          hist[i] = 0;
        }
        __syncwarp();
        for (int32_t i = lane_id; i < n; i += static_cast<int32_t>(device::kWarpThreads)) {
          const uint32_t u = onepass_ordered(cand_val[list_base + i]);
          if (level == 0) {
            atomicAdd(&hist[u >> 24], 1);
          } else if ((u >> 24) == (thr_u >> 24)) {
            atomicAdd(&hist[(u >> 16) & 0xFFu], 1);
          }
        }
        __syncwarp();
        uint32_t bin = 0;
        int32_t above = 0;
        if (lane_id == 0) {
          const int32_t need = level == 0 ? topk : topk - above_hi;
          int32_t acc_cnt = 0;
          int32_t b = 255;
          for (; b > 0; --b) {
            acc_cnt += hist[b];
            if (acc_cnt >= need) break;
          }
          bin = static_cast<uint32_t>(b);
          above = level == 0 ? acc_cnt - hist[b] : 0;
        }
        bin = __shfl_sync(0xFFFFFFFFu, bin, 0);
        above = __shfl_sync(0xFFFFFFFFu, above, 0);
        if (level == 0) {
          thr_u = bin << 24;
          above_hi = above;
        } else {
          thr_u |= bin << 16;
        }
        __syncwarp();
      }
      const float tau_new = onepass_unorder(thr_u);

      // Warp-synchronous in-place compaction, one 256-entry chunk at a time:
      // the whole chunk is staged in registers before any write, and writes
      // never pass the read frontier (survivors <= entries processed).
      constexpr int32_t kChunkPer = 8;
      constexpr int32_t kChunk = static_cast<int32_t>(device::kWarpThreads) * kChunkPer;
      int32_t wcnt = 0;
      for (int32_t c0 = 0; c0 < n; c0 += kChunk) {
        float v[kChunkPer];
        int32_t p[kChunkPer];
        bool keep[kChunkPer];
#pragma unroll
        for (int32_t j = 0; j < kChunkPer; ++j) {
          const int32_t i = c0 + j * static_cast<int32_t>(device::kWarpThreads) + static_cast<int32_t>(lane_id);
          keep[j] = false;
          if (i < n) {
            v[j] = cand_val[list_base + i];
            p[j] = cand_pos[list_base + i];
            keep[j] = onepass_ordered(v[j]) >= thr_u;
          }
        }
        __syncwarp();
#pragma unroll
        for (int32_t j = 0; j < kChunkPer; ++j) {
          const uint32_t ballot = __ballot_sync(0xFFFFFFFFu, keep[j]);
          if (keep[j]) {
            const int32_t slot = wcnt + __popc(ballot & ((1u << lane_id) - 1u));
            cand_val[list_base + slot] = v[j];
            cand_pos[list_base + slot] = p[j];
          }
          wcnt += __popc(ballot);
        }
        __syncwarp();
      }
      if (lane_id == 0) {
        s_cnt[state_i] = wcnt;
        float& tau_ref = s_tau[state_i];
        tau_ref = tau_new > tau_ref ? tau_new : tau_ref;
        // A compaction that cannot make room for even two more tiles means a
        // tie mass wider than the list; give up and let the host fall back
        // rather than silently dropping candidates.
        if (wcnt > cap - 2 * static_cast<int32_t>(kOnepassBlockK)) {
          atomicOr(&s_misc[2], 1);
        }
      }
    }
  }

  __syncthreads();
  // Neutralize the stale tail of every owned list: the host runs the exact
  // top-k over the raw buffers, and entries past cnt are pre-compaction
  // leftovers (including duplicates of survivors).
  for (int32_t h = 0; h < num_heads; ++h) {
    for (int32_t r = 0; r < q_rows; ++r) {
      const int32_t c_raw = s_cnt[h * kOnepassBlockQ + r];
      const int32_t c = c_raw < cap ? c_raw : cap;
      const int64_t g_row = seq_start + q_lo + r;
      const int64_t list_base = ((static_cast<int64_t>(h) * total_q + g_row) * num_segs + seg) * cap;
      for (int32_t i = c + static_cast<int32_t>(tx); i < cap; i += kOnepassCTA) {
        cand_val[list_base + i] = -CUDART_INF_F;
      }
    }
  }
  for (uint32_t i = tx; i < static_cast<uint32_t>(num_heads) * kOnepassBlockQ; i += kOnepassCTA) {
    const int32_t h = i / kOnepassBlockQ;
    const int32_t r = i % kOnepassBlockQ;
    if (r < q_rows) {
      const int64_t g_row = seq_start + q_lo + r;
      const int32_t c = s_cnt[h * kOnepassBlockQ + r];
      cnt_out[(static_cast<int64_t>(h) * total_q + g_row) * num_segs + seg] = c < cap ? c : cap;
    }
  }
  if (tx == 0 && s_misc[2]) atomicOr(overflow, 1);
}

template <typename T>
void minimax_token_onepass(
    tvm::ffi::TensorView idx_q,
    tvm::ffi::TensorView idx_k_cache,
    tvm::ffi::TensorView req_to_token,
    tvm::ffi::TensorView cu_seqlens,
    tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView prefix_lens,
    tvm::ffi::TensorView slot_ids,
    tvm::ffi::TensorView cand_val,
    tvm::ffi::TensorView cand_pos,
    tvm::ffi::TensorView cnt,
    tvm::ffi::TensorView overflow,
    int64_t max_seqlen_q,
    int64_t kv_width,
    int64_t topk,
    int64_t init_tokens,
    int64_t local_tokens,
    double sm_scale,
    double init_bias,
    double local_bias) {
  using namespace host;

  SymbolicSize NQ{"total_q"}, H{"num_idx_heads"}, D{"qk_head_dim"};
  SymbolicSize S{"max_slots"}, One{"one"}, R{"max_reqs"}, W{"r2t_width"};
  SymbolicSize B{"batch"}, Bp1{"batch_plus_1"}, SEG{"num_segs"}, CAP{"cand_cap"};
  SymbolicSize OF{"overflow"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({NQ, H, D}).with_dtype<T>().with_device(device_).verify(idx_q);
  TensorMatcher({S, One, D}).with_dtype<T>().with_device(device_).verify(idx_k_cache);
  TensorMatcher({R, W}).with_dtype<int32_t>().with_device(device_).verify(req_to_token);
  TensorMatcher({Bp1}).with_dtype<int32_t>().with_device(device_).verify(cu_seqlens);
  TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens).verify(prefix_lens).verify(slot_ids);
  TensorMatcher({H, NQ, SEG, CAP}).with_dtype<fp32_t>().with_device(device_).verify(cand_val);
  TensorMatcher({H, NQ, SEG, CAP}).with_dtype<int32_t>().with_device(device_).verify(cand_pos);
  TensorMatcher({H, NQ, SEG}).with_dtype<int32_t>().with_device(device_).verify(cnt);
  TensorMatcher({OF}).with_dtype<int32_t>().with_device(device_).verify(overflow);

  CHECK_HOST(D.unwrap() == kOnepassHeadDim)
      << "minimax_token_onepass requires qk_head_dim == " << kOnepassHeadDim << ", got " << D.unwrap();
  CHECK_HOST(One.unwrap() == 1) << "index K cache must have a single KV head";
  CHECK_HOST(Bp1.unwrap() == B.unwrap() + 1) << "cu_seqlens must have batch + 1 entries";
  CHECK_HOST(H.unwrap() <= kOnepassMaxHeads) << "at most " << kOnepassMaxHeads << " index heads, got " << H.unwrap();
  CHECK_HOST(kv_width <= W.unwrap()) << "kv_width " << kv_width << " exceeds req_to_token width " << W.unwrap();
  CHECK_HOST(CAP.unwrap() >= static_cast<size_t>(topk + 2 * kOnepassBlockK))
      << "cand cap " << CAP.unwrap() << " must be >= topk + 2 * " << kOnepassBlockK;

  const auto kernel = minimax_token_onepass_kernel<T>;
  static const bool smem_ok = [&] {
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kOnepassSmemBytes))
        << "raising the dynamic smem limit to " << kOnepassSmemBytes << " B";
    return true;
  }();
  (void)smem_ok;

  const dim3 grid(
      static_cast<uint32_t>(SEG.unwrap()),
      static_cast<uint32_t>(B.unwrap()),
      div_ceil<uint32_t>(max_seqlen_q, kOnepassBlockQ));
  LaunchKernel(grid, kOnepassCTA, device_.unwrap(), kOnepassSmemBytes)(
      kernel,
      static_cast<const T*>(idx_q.data_ptr()),
      static_cast<const T*>(idx_k_cache.data_ptr()),
      static_cast<const int32_t*>(req_to_token.data_ptr()),
      static_cast<const int32_t*>(cu_seqlens.data_ptr()),
      static_cast<const int32_t*>(seq_lens.data_ptr()),
      static_cast<const int32_t*>(prefix_lens.data_ptr()),
      static_cast<const int32_t*>(slot_ids.data_ptr()),
      static_cast<float*>(cand_val.data_ptr()),
      static_cast<int32_t*>(cand_pos.data_ptr()),
      static_cast<int32_t*>(cnt.data_ptr()),
      static_cast<int32_t*>(overflow.data_ptr()),
      static_cast<int64_t>(NQ.unwrap()),
      static_cast<int32_t>(H.unwrap()),
      static_cast<int64_t>(S.unwrap()),
      static_cast<int64_t>(W.unwrap()),
      static_cast<int32_t>(kv_width),
      static_cast<int32_t>(CAP.unwrap()),
      static_cast<int32_t>(topk),
      static_cast<int32_t>(init_tokens),
      static_cast<int32_t>(local_tokens),
      // Match the Triton kernels' fp32 rounding exactly (see emit kernel).
      static_cast<float>(sm_scale) * 1.4426950408889634f,
      static_cast<float>(init_bias),
      static_cast<float>(local_bias));
}

}  // namespace
