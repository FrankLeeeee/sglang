// Threshold filter over a materialized score matrix (prototype).
//
// Part of the tau-emit token selector: the scorer writes the [heads, rows,
// width] fp32 logits once; tau per row is derived on the host from pooled
// maxima of the *stored* scores (no second q.k pass, unlike the two-pass
// fused selector); this kernel then streams the matrix once and appends every
// entry with score >= tau to a bounded per-(head, row) candidate list. The
// exact top-k runs over those few thousand candidates instead of the full
// row.
//
// The matrix already carries the causal -inf mask and the sink/window biases,
// so the filter is a pure compare: keep = (v >= tau) && (v > -inf). cnt may
// exceed cap (the host detects overflow and falls back); entries beyond cap
// are dropped; list order is unspecified.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For CHECK_HOST, div_ceil
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, CHECK_CUDA

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <math_constants.h>

#include <cstdint>

namespace {

// Order-preserving float <-> uint mapping (same trick as the one-pass kernel).
SGL_DEVICE uint32_t filter_ordered(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

SGL_DEVICE float filter_unorder(uint32_t u) {
  return (u & 0x80000000u) ? __uint_as_float(u & 0x7fffffffu)
                           : __uint_as_float(~u);
}

// tau[row] = a lower bound on the k-th largest of vals[row, :], from a
// two-level (16-bit) radix histogram — one CTA per row, two streaming passes.
// The bin floor is <= the true k-th value, so a filter keeping >= tau always
// retains a superset of the row's top-k.
constexpr uint32_t kKthCTA = 256;

__global__ void __launch_bounds__(kKthCTA) minimax_row_kth_kernel(
    const float* __restrict__ vals,  // [rows, width]
    float* __restrict__ tau,         // [rows]
    const int32_t width,
    const int32_t k) {
  __shared__ int32_t s_hist[256];
  __shared__ int32_t s_sel[2];  // [0] = bin, [1] = survivors above bin

  const int64_t row = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const float* v = vals + row * width;

  uint32_t thr_u = 0;
  int32_t above_hi = 0;
#pragma unroll
  for (int level = 0; level < 2; ++level) {
    for (uint32_t i = tx; i < 256; i += kKthCTA) s_hist[i] = 0;
    __syncthreads();
    for (int32_t i = tx; i < width; i += static_cast<int32_t>(kKthCTA)) {
      const uint32_t u = filter_ordered(v[i]);
      if (level == 0) {
        atomicAdd(&s_hist[u >> 24], 1);
      } else if ((u >> 24) == (thr_u >> 24)) {
        atomicAdd(&s_hist[(u >> 16) & 0xFFu], 1);
      }
    }
    __syncthreads();
    if (tx == 0) {
      const int32_t need = level == 0 ? k : k - above_hi;
      int32_t acc = 0;
      int32_t b = 255;
      bool found = false;
      for (; b > 0; --b) {
        acc += s_hist[b];
        if (acc >= need) {
          found = true;
          break;
        }
      }
      // Fewer than `need` entries above the lowest bin: rank k sits among
      // the -inf mass of a short causal row, so every finite score must
      // survive the filter. Signal tau = -inf (bin 0's floor would be NaN
      // under the order-preserving unmap, which poisons the compare).
      s_sel[0] = (level == 0 && !found) ? -1 : b;
      s_sel[1] = (level == 0 && found) ? acc - s_hist[b] : 0;
    }
    __syncthreads();
    if (level == 0 && s_sel[0] < 0) {
      if (tx == 0) tau[row] = -CUDART_INF_F;
      return;
    }
    if (level == 0) {
      thr_u = static_cast<uint32_t>(s_sel[0]) << 24;
      above_hi = s_sel[1];
    } else {
      thr_u |= static_cast<uint32_t>(s_sel[0]) << 16;
    }
    __syncthreads();
  }
  if (tx == 0) tau[row] = filter_unorder(thr_u);
}

void minimax_row_kth(
    tvm::ffi::TensorView vals, tvm::ffi::TensorView tau, int64_t k) {
  using namespace host;
  SymbolicSize N{"rows"}, W{"width"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({N, W}).with_dtype<fp32_t>().with_device(device_).verify(vals);
  TensorMatcher({N}).with_dtype<fp32_t>().with_device(device_).verify(tau);
  CHECK_HOST(k > 0) << "k must be positive, got " << k;
  LaunchKernel(static_cast<uint32_t>(N.unwrap()), kKthCTA, device_.unwrap())(
      minimax_row_kth_kernel,
      static_cast<const float*>(vals.data_ptr()),
      static_cast<float*>(tau.data_ptr()),
      static_cast<int32_t>(W.unwrap()),
      static_cast<int32_t>(k));
}

constexpr uint32_t kFilterCTA = 256;
constexpr uint32_t kFilterSpan = 2048;  // matrix columns per CTA
constexpr uint32_t kFilterWarps = kFilterCTA / device::kWarpThreads;

__global__ void __launch_bounds__(kFilterCTA) minimax_score_filter_kernel(
    const float* __restrict__ scores,  // [num_heads, rows, width] (chunk)
    const float* __restrict__ tau,     // [num_heads, rows] (chunk)
    float* __restrict__ cand_val,      // [num_heads, rows_total, cap], -inf filled
    int32_t* __restrict__ cand_pos,    // [num_heads, rows_total, cap]
    int32_t* __restrict__ cnt,         // [num_heads, rows_total], zeroed
    const int64_t rows,        // rows in this chunk (grid y)
    const int64_t rows_total,  // rows in the global candidate buffers
    const int64_t row_offset,  // this chunk's first global row
    const int32_t width,
    const int32_t cap) {
  __shared__ int32_t s_warp_cnt[kFilterWarps];
  __shared__ int32_t s_base;

  const uint32_t span = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int64_t head = blockIdx.z;
  const uint32_t tx = threadIdx.x;
  const uint32_t warp_id = tx / device::kWarpThreads;
  const uint32_t lane_id = tx % device::kWarpThreads;

  const int64_t row_base = (head * rows + row) * width;
  const float tau_r = tau[head * rows + row];
  const int64_t g_row = head * rows_total + row_offset + row;
  const int32_t col_lo = static_cast<int32_t>(span * kFilterSpan);

  // Stage this thread's elements and count survivors per warp.
  constexpr int32_t kPer = kFilterSpan / kFilterCTA;
  float v[kPer];
  bool keep[kPer];
  int32_t warp_keep = 0;
#pragma unroll
  for (int32_t j = 0; j < kPer; ++j) {
    const int32_t col =
        col_lo + j * static_cast<int32_t>(kFilterCTA) + static_cast<int32_t>(tx);
    keep[j] = false;
    if (col < width) {
      v[j] = scores[row_base + col];
      keep[j] = v[j] >= tau_r && v[j] > -CUDART_INF_F;
    }
  }
#pragma unroll
  for (int32_t j = 0; j < kPer; ++j) {
    warp_keep += __popc(__ballot_sync(0xFFFFFFFFu, keep[j]));
  }
  if (lane_id == 0) s_warp_cnt[warp_id] = warp_keep;
  __syncthreads();

  // One global atomic per CTA reserves the whole span's slots.
  if (tx == 0) {
    int32_t total = 0;
#pragma unroll
    for (uint32_t w = 0; w < kFilterWarps; ++w) total += s_warp_cnt[w];
    s_base = total > 0 ? atomicAdd(&cnt[g_row], total) : 0;
  }
  __syncthreads();

  int32_t base = s_base;
  for (uint32_t w = 0; w < warp_id; ++w) base += s_warp_cnt[w];

  const int64_t list_base = g_row * cap;
#pragma unroll
  for (int32_t j = 0; j < kPer; ++j) {
    const uint32_t ballot = __ballot_sync(0xFFFFFFFFu, keep[j]);
    if (keep[j]) {
      const int32_t slot = base + __popc(ballot & ((1u << lane_id) - 1u));
      if (slot < cap) {
        cand_val[list_base + slot] = v[j];
        cand_pos[list_base + slot] =
            col_lo + j * static_cast<int32_t>(kFilterCTA) +
            static_cast<int32_t>(tx);
      }
    }
    base += __popc(ballot);
  }
}

void minimax_score_filter(
    tvm::ffi::TensorView scores,
    tvm::ffi::TensorView tau,
    tvm::ffi::TensorView cand_val,
    tvm::ffi::TensorView cand_pos,
    tvm::ffi::TensorView cnt,
    int64_t row_offset) {
  using namespace host;

  SymbolicSize H{"num_heads"}, R{"rows"}, RT{"rows_total"}, W{"width"};
  SymbolicSize CAP{"cand_cap"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({H, R, W}).with_dtype<fp32_t>().with_device(device_).verify(scores);
  TensorMatcher({H, R}).with_dtype<fp32_t>().with_device(device_).verify(tau);
  TensorMatcher({H, RT, CAP}).with_dtype<fp32_t>().with_device(device_).verify(cand_val);
  TensorMatcher({H, RT, CAP}).with_dtype<int32_t>().with_device(device_).verify(cand_pos);
  TensorMatcher({H, RT}).with_dtype<int32_t>().with_device(device_).verify(cnt);
  CHECK_HOST(row_offset >= 0 &&
             row_offset + R.unwrap() <= RT.unwrap())
      << "chunk rows [" << row_offset << ", " << row_offset + R.unwrap()
      << ") out of the global buffer's " << RT.unwrap() << " rows";

  const dim3 grid(div_ceil<uint32_t>(W.unwrap(), kFilterSpan),
                  static_cast<uint32_t>(R.unwrap()),
                  static_cast<uint32_t>(H.unwrap()));
  LaunchKernel(grid, kFilterCTA, device_.unwrap())(
      minimax_score_filter_kernel,
      static_cast<const float*>(scores.data_ptr()),
      static_cast<const float*>(tau.data_ptr()),
      static_cast<float*>(cand_val.data_ptr()),
      static_cast<int32_t*>(cand_pos.data_ptr()),
      static_cast<int32_t*>(cnt.data_ptr()),
      static_cast<int64_t>(R.unwrap()),
      static_cast<int64_t>(RT.unwrap()),
      row_offset,
      static_cast<int32_t>(W.unwrap()),
      static_cast<int32_t>(CAP.unwrap()));
}

}  // namespace
