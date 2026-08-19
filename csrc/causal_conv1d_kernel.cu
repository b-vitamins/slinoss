// Causal depthwise conv1d kernels.
//
// Class: DRAM-bound, both directions.
//
// Analytic byte count, per token per channel, at bfloat16 with W = 4 and a
// 64-token tile:
//
//   forward   read x 2 B, write y 2 B, plus (W-1)/kTileT = 3/64 re-read of x
//             for the tile prologue: 4.09 B for 2W-1 = 7 flop, 1.7 flop/B.
//   backward  read x 2 B, read dy 2 B, write dx 2 B, plus the same prologue
//             re-read on x and dy and the (W-1)/kTileT extra ds recompute:
//             6.28 B for 4W+2 = 18 flop, 2.9 flop/B.
//
// The tap bank is D*W*4 B, resident in L2 for every reachable D, and the
// parameter-gradient partials are parts*D*(W+1)*4 B with parts = ceil(T/64), an
// order 1/16 addition to the traffic at W = 4. Both figures are analytic and
// hold no claim about achieved bandwidth.
//
// Decomposition. One thread per channel, walking kTileT timesteps with the
// window in registers. The layout is channels-last, so a warp at a fixed
// timestep reads 32 consecutive channels: one coalesced transaction. No shared
// memory, so there is no tile to swizzle and occupancy is bounded by registers
// alone.
//
// The forward grid is (channel tiles, time tiles, batch). The backward folds
// batch into the block's serial loop instead, which makes each block's
// parameter-gradient accumulator complete over the batch axis and leaves one
// partial per time tile.

#include "causal_conv1d.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace slinoss {
namespace {

__device__ __forceinline__ float sigmoid_of(float s) {
  // exp(-s) overflows to inf for very negative s, and 1/inf is zero, so the
  // saturated ends are exact rather than NaN.
  return 1.0f / (1.0f + expf(-s));
}

__device__ __forceinline__ float silu_of(float s) { return s * sigmoid_of(s); }

__device__ __forceinline__ float silu_grad_of(float s) {
  const float g = sigmoid_of(s);
  return g * (1.0f + s * (1.0f - g));
}

// One extended-index read of the activation stream. Index u < 0 addresses the
// incoming state, which holds the W-1 timesteps before the sequence; below that,
// and past the end, the stream is zero.
template <typename input_t>
__device__ __forceinline__ float
read_extended(const input_t *__restrict__ x, const input_t *__restrict__ state,
              long xbase, long sbase, int channels, int seqlen, int width,
              int u) {
  if (u >= 0) {
    return u < seqlen ? static_cast<float>(x[xbase + static_cast<long>(u) * channels])
                      : 0.0f;
  }
  const int i = u + width - 1;
  if (state == nullptr || i < 0) {
    return 0.0f;
  }
  return static_cast<float>(state[sbase + static_cast<long>(i) * channels]);
}

template <typename input_t>
__global__ void conv1d_fwd_kernel(const input_t *__restrict__ x,
                                  const input_t *__restrict__ weight,
                                  const input_t *__restrict__ bias,
                                  const input_t *__restrict__ initial_state,
                                  input_t *__restrict__ y,
                                  input_t *__restrict__ final_state, int seqlen,
                                  int channels, int width, bool activation) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }
  const int t0 = blockIdx.y * kTileT;
  const int t1 = min(t0 + kTileT, seqlen);
  const long xbase = static_cast<long>(blockIdx.z) * seqlen * channels + channel;
  const long sbase =
      static_cast<long>(blockIdx.z) * (width - 1) * channels + channel;

  // wr[j] is the tap that multiplies lag j, so tap width-1 is the current
  // token. Slots at or past width are hard zeros, which is what lets the
  // contraction run over the compile-time bound with no predicate.
  float wr[kMaxWidth];
  float xw[kMaxWidth];
#pragma unroll
  for (int j = 0; j < kMaxWidth; ++j) {
    wr[j] = j < width ? static_cast<float>(
                            weight[static_cast<long>(channel) * width + width - 1 - j])
                      : 0.0f;
    // Pre-shift state: the tile loop shifts before it loads, so slot j must
    // hold the sample that lands at lag j+1, i.e. x[t0-1-j]. Slots from
    // width-1 up shift into lags at or past width, where wr is zero.
    xw[j] = j < width - 1
                ? read_extended(x, initial_state, xbase, sbase, channels, seqlen,
                                width, t0 - 1 - j)
                : 0.0f;
  }
  const float bias_of_channel =
      bias == nullptr ? 0.0f : static_cast<float>(bias[channel]);

  for (int t = t0; t < t1; ++t) {
#pragma unroll
    for (int j = kMaxWidth - 1; j > 0; --j) {
      xw[j] = xw[j - 1];
    }
    xw[0] = static_cast<float>(x[xbase + static_cast<long>(t) * channels]);
    // Oldest tap first, which is the order the reference sums in.
    float acc = 0.0f;
#pragma unroll
    for (int j = kMaxWidth - 1; j >= 0; --j) {
      acc = fmaf(wr[j], xw[j], acc);
    }
    acc += bias_of_channel;
    y[xbase + static_cast<long>(t) * channels] =
        static_cast<input_t>(activation ? silu_of(acc) : acc);
  }

  // The next call's window: the W-1 timesteps that precede its first token.
  // Below T = W-1 that window straddles the incoming state, which the extended
  // read handles without a second path.
  if (final_state != nullptr && t1 == seqlen) {
    for (int i = 0; i < width - 1; ++i) {
      final_state[sbase + static_cast<long>(i) * channels] =
          static_cast<input_t>(read_extended(x, initial_state, xbase, sbase,
                                             channels, seqlen, width,
                                             seqlen - (width - 1) + i));
    }
  }
}

template <typename input_t>
__global__ void
conv1d_bwd_kernel(const input_t *__restrict__ dy,
                  const input_t *__restrict__ dfinal_state,
                  const input_t *__restrict__ x,
                  const input_t *__restrict__ weight,
                  const input_t *__restrict__ bias,
                  const input_t *__restrict__ initial_state,
                  input_t *__restrict__ dx, input_t *__restrict__ dinitial_state,
                  float *__restrict__ dweight_parts,
                  float *__restrict__ dbias_parts, int batch, int seqlen,
                  int channels, int width, bool activation) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }
  const int part = blockIdx.y;
  const int t0 = part * kTileT;
  const int t1 = min(t0 + kTileT, seqlen);
  // dx at index u needs ds at u .. u+W-1, so the walk runs W-1 steps past the
  // tile and recomputes those ds. The tile owns dx over [t0, t1) and, at t0 = 0,
  // the W-1 negative indices that are the gradient of the incoming state.
  const int tend = t1 - 1 + width - 1;
  const int u_min = t0 == 0 ? -(width - 1) : t0;

  float wr[kMaxWidth];
  float wf[kMaxWidth];
  float dwacc[kMaxWidth];
#pragma unroll
  for (int j = 0; j < kMaxWidth; ++j) {
    const bool live = j < width;
    wr[j] = live ? static_cast<float>(
                       weight[static_cast<long>(channel) * width + width - 1 - j])
                 : 0.0f;
    wf[j] = live
                ? static_cast<float>(weight[static_cast<long>(channel) * width + j])
                : 0.0f;
    dwacc[j] = 0.0f;
  }
  const float bias_of_channel =
      bias == nullptr ? 0.0f : static_cast<float>(bias[channel]);
  float dbacc = 0.0f;

  for (int b = 0; b < batch; ++b) {
    const long xbase = static_cast<long>(b) * seqlen * channels + channel;
    const long sbase = static_cast<long>(b) * (width - 1) * channels + channel;

    float xw[kMaxWidth];
    float dsw[kMaxWidth];
#pragma unroll
    for (int j = 0; j < kMaxWidth; ++j) {
      // Pre-shift state, as in the forward: slot j lands at lag j+1.
      xw[j] = j < width - 1
                  ? read_extended(x, initial_state, xbase, sbase, channels,
                                  seqlen, width, t0 - 1 - j)
                  : 0.0f;
      // ds before the tile is zero at t0 = 0 because there is no output there,
      // and is never read at t0 > 0 because u_min holds dx back until the
      // window is full of ds values this tile computed.
      dsw[j] = 0.0f;
    }

    for (int t = t0; t <= tend; ++t) {
#pragma unroll
      for (int j = kMaxWidth - 1; j > 0; --j) {
        xw[j] = xw[j - 1];
      }
      xw[0] = t < seqlen ? static_cast<float>(
                               x[xbase + static_cast<long>(t) * channels])
                         : 0.0f;

      float ds = 0.0f;
      if (dy != nullptr && t < seqlen) {
        ds = static_cast<float>(dy[xbase + static_cast<long>(t) * channels]);
        if (activation) {
          float acc = 0.0f;
#pragma unroll
          for (int j = kMaxWidth - 1; j >= 0; --j) {
            acc = fmaf(wr[j], xw[j], acc);
          }
          ds *= silu_grad_of(acc + bias_of_channel);
        }
      }

      // The tile owns the parameter gradient over [t0, t1); the overhang past
      // t1 belongs to the next tile and must not be counted twice.
      if (t < t1) {
#pragma unroll
        for (int j = 0; j < kMaxWidth; ++j) {
          dwacc[j] = fmaf(ds, xw[j], dwacc[j]);
        }
        dbacc += ds;
      }

#pragma unroll
      for (int j = kMaxWidth - 1; j > 0; --j) {
        dsw[j] = dsw[j - 1];
      }
      dsw[0] = ds;

      const int u = t - (width - 1);
      if (u >= u_min) {
        float acc = 0.0f;
#pragma unroll
        for (int j = kMaxWidth - 1; j >= 0; --j) {
          acc = fmaf(wf[j], dsw[j], acc);
        }
        // The trailing window is returned as the next call's state, so its
        // cotangent lands on the extended index it was sliced from. Below
        // T = W-1 that index is negative and the contribution belongs to the
        // gradient of the incoming state, which the same test covers.
        if (dfinal_state != nullptr) {
          const int i = u - (seqlen - (width - 1));
          if (i >= 0 && i < width - 1) {
            acc += static_cast<float>(
                dfinal_state[sbase + static_cast<long>(i) * channels]);
          }
        }
        if (u >= 0) {
          dx[xbase + static_cast<long>(u) * channels] =
              static_cast<input_t>(acc);
        } else if (dinitial_state != nullptr) {
          dinitial_state[sbase + static_cast<long>(u + width - 1) * channels] =
              static_cast<input_t>(acc);
        }
      }
    }
  }

  // Plain stores, one slice per time tile, so no output is read back and
  // nothing needs zeroing before the launch. dwacc is indexed by lag; tap k is
  // lag width-1-k. The loop bound is the compile-time one so that the register
  // array is never indexed dynamically, which would put it in local memory.
  float *dw = dweight_parts + static_cast<long>(part) * channels * width +
              static_cast<long>(channel) * width;
#pragma unroll
  for (int j = 0; j < kMaxWidth; ++j) {
    if (j < width) {
      dw[width - 1 - j] = dwacc[j];
    }
  }
  if (dbias_parts != nullptr) {
    dbias_parts[static_cast<long>(part) * channels + channel] = dbacc;
  }
}

int block_width(int channels) {
  const int warps = (channels + 31) / 32 * 32;
  return warps < kMaxChannelsPerBlock ? warps : kMaxChannelsPerBlock;
}

int time_tiles(int seqlen) { return (seqlen + kTileT - 1) / kTileT; }

template <typename input_t>
const input_t *data_or_null(const std::optional<at::Tensor> &t) {
  return t.has_value() ? t->const_data_ptr<input_t>() : nullptr;
}

template <typename input_t>
input_t *mutable_or_null(const std::optional<at::Tensor> &t) {
  return t.has_value() ? t->data_ptr<input_t>() : nullptr;
}

template <typename input_t>
void launch_fwd(const at::Tensor &x, const at::Tensor &weight,
                const std::optional<at::Tensor> &bias,
                const std::optional<at::Tensor> &initial_state,
                const at::Tensor &y,
                const std::optional<at::Tensor> &final_state, bool activation) {
  const int batch = static_cast<int>(x.size(0));
  const int seqlen = static_cast<int>(x.size(1));
  const int channels = static_cast<int>(x.size(2));
  const int width = static_cast<int>(weight.size(1));
  const int threads = block_width(channels);
  const dim3 grid((channels + threads - 1) / threads, time_tiles(seqlen), batch);
  conv1d_fwd_kernel<input_t>
      <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          x.const_data_ptr<input_t>(), weight.const_data_ptr<input_t>(),
          data_or_null<input_t>(bias), data_or_null<input_t>(initial_state),
          y.data_ptr<input_t>(), mutable_or_null<input_t>(final_state), seqlen,
          channels, width, activation);
}

template <typename input_t>
void launch_bwd(const std::optional<at::Tensor> &dy,
                const std::optional<at::Tensor> &dfinal_state,
                const at::Tensor &x, const at::Tensor &weight,
                const std::optional<at::Tensor> &bias,
                const std::optional<at::Tensor> &initial_state,
                const at::Tensor &dx,
                const std::optional<at::Tensor> &dinitial_state,
                const at::Tensor &dweight_parts,
                const std::optional<at::Tensor> &dbias_parts,
                bool activation) {
  const int batch = static_cast<int>(x.size(0));
  const int seqlen = static_cast<int>(x.size(1));
  const int channels = static_cast<int>(x.size(2));
  const int width = static_cast<int>(weight.size(1));
  const int threads = block_width(channels);
  const dim3 grid((channels + threads - 1) / threads, time_tiles(seqlen), 1);
  conv1d_bwd_kernel<input_t>
      <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          data_or_null<input_t>(dy), data_or_null<input_t>(dfinal_state),
          x.const_data_ptr<input_t>(), weight.const_data_ptr<input_t>(),
          data_or_null<input_t>(bias), data_or_null<input_t>(initial_state),
          dx.data_ptr<input_t>(), mutable_or_null<input_t>(dinitial_state),
          dweight_parts.data_ptr<float>(),
          dbias_parts.has_value() ? dbias_parts->data_ptr<float>() : nullptr,
          batch, seqlen, channels, width, activation);
}

} // namespace

int64_t causal_conv1d_bwd_parts(int64_t seqlen) {
  return (seqlen + kTileT - 1) / kTileT;
}

void causal_conv1d_fwd(const at::Tensor &x, const at::Tensor &weight,
                       const std::optional<at::Tensor> &bias,
                       const std::optional<at::Tensor> &initial_state,
                       const at::Tensor &y,
                       const std::optional<at::Tensor> &final_state,
                       bool activation) {
  const at::cuda::CUDAGuard guard(x.device());
  switch (x.scalar_type()) {
  case at::ScalarType::BFloat16:
    launch_fwd<at::BFloat16>(x, weight, bias, initial_state, y, final_state,
                             activation);
    break;
  case at::ScalarType::Half:
    launch_fwd<at::Half>(x, weight, bias, initial_state, y, final_state,
                         activation);
    break;
  case at::ScalarType::Float:
    launch_fwd<float>(x, weight, bias, initial_state, y, final_state,
                      activation);
    break;
  default:
    TORCH_CHECK(false, "causal_conv1d_fwd: unsupported dtype ",
                x.scalar_type());
  }
}

void causal_conv1d_bwd(const std::optional<at::Tensor> &dy,
                       const std::optional<at::Tensor> &dfinal_state,
                       const at::Tensor &x, const at::Tensor &weight,
                       const std::optional<at::Tensor> &bias,
                       const std::optional<at::Tensor> &initial_state,
                       const at::Tensor &dx,
                       const std::optional<at::Tensor> &dinitial_state,
                       const at::Tensor &dweight_parts,
                       const std::optional<at::Tensor> &dbias_parts,
                       bool activation) {
  const at::cuda::CUDAGuard guard(x.device());
  switch (x.scalar_type()) {
  case at::ScalarType::BFloat16:
    launch_bwd<at::BFloat16>(dy, dfinal_state, x, weight, bias, initial_state,
                             dx, dinitial_state, dweight_parts, dbias_parts,
                             activation);
    break;
  case at::ScalarType::Half:
    launch_bwd<at::Half>(dy, dfinal_state, x, weight, bias, initial_state, dx,
                         dinitial_state, dweight_parts, dbias_parts,
                         activation);
    break;
  case at::ScalarType::Float:
    launch_bwd<float>(dy, dfinal_state, x, weight, bias, initial_state, dx,
                      dinitial_state, dweight_parts, dbias_parts, activation);
    break;
  default:
    TORCH_CHECK(false, "causal_conv1d_bwd: unsupported dtype ",
                x.scalar_type());
  }
}

} // namespace slinoss
