// Causal depthwise conv1d kernels.
//
// Class: DRAM-bound, both directions. The arithmetic intensity below is 1.75 and
// 2.7 flop/B against a ridge of 164, and measured DRAM traffic is within 3% of
// the compulsory count for the forward and 6% for the backward, so there are no
// bytes left to recover and the class is not in question.
//
// OPEN DEFECT: neither direction clears that class's floor of 85% of the measured
// copy ceiling. On an RTX A6000, clocks unlocked, at the standard shape the
// forward reaches 66% and the backward 61%. Reproduce with
//
//   scripts/perf/profile_op.py --op conv --shape standard --mode step
//
// What is missing is not traffic but memory-level parallelism to cover DRAM
// latency. SERIAL-tiny is not the escape: the two are 29% and 54% of the conv
// step against that class's 2% ceiling.
//
// For the backward that shortfall is structural. Its grid is
// ceil(D/kMaxChannelsPerBlock) * ceil(T/kBwdTileT) and carries no batch axis,
// because folding batch into the block's serial loop is what makes each block's
// parameter-gradient accumulator complete over batch and leaves exactly one
// partial per time tile. Achieved bandwidth then tracks the block count the time
// tile alone can supply: 60% at the standard shape, which supplies 2304 blocks,
// against 76% at the long shape, which supplies 9216 from the identical code.
// Shrinking the tile further to buy blocks reverses at kBwdTileT = 4, where the
// partial count doubles and the host-side sum of the partials grows faster than
// the kernel shrinks. Moving batch onto the grid raises the block count at
// unchanged traffic, but the partial count then depends on batch, and
// causal_conv1d_bwd_parts takes only a sequence length; changing that signature
// changes the (P,D,W) allocation in slinoss/ops/conv/backends.py, so it is not a
// change this file can make alone.
//
// Compulsory byte count, per token per channel, at bfloat16 with W = 4:
//
//   forward   read x 2 B, write y 2 B: 4 B for 2W-1 = 7 flop, 1.75 flop/B.
//   backward  read x 2 B, read dy 2 B, write dx 2 B, plus the partials at
//             (W+1)*4/(kBwdTileT*B) = 0.63 B: 6.63 B for 4W+2 = 18 flop,
//             2.7 flop/B.
//
// Both kernels re-read past their tile: the forward re-reads W-1 activations for
// the prologue, the backward re-reads W-1 timesteps of both x and dy and
// recomputes their ds, because dx at u needs ds at u .. u+W-1. Those re-reads
// raise requested loads to 1.8x the compulsory ones for the backward at
// kBwdTileT = 8, and do not appear in the count above: at the shapes measured L2
// absorbs them, which is why DRAM traffic stays near the compulsory figure.
// The tap bank is D*W*4 B and is L2-resident for every reachable D. Every figure
// here is analytic and holds no claim about achieved bandwidth.
//
// Decomposition. One thread per channel, walking the time tile with the window
// in registers. The layout is channels-last, so a warp at a fixed timestep reads
// 32 consecutive channels: one coalesced transaction. No shared memory, so there
// is no tile to swizzle and occupancy is bounded by registers and by the
// blocks-per-SM limit alone.
//
// The forward grid is (channel tiles, time tiles, batch). The backward grid is
// (channel tiles, time tiles), with batch in the block's serial loop and
// kBwdStreams entries of it interleaved per thread.

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

template <typename input_t, int kWidth>
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
                  int channels, bool activation) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }
  const int part = blockIdx.y;
  const int t0 = part * kBwdTileT;
  const int t1 = min(t0 + kBwdTileT, seqlen);
  // dx at index u needs ds at u .. u+W-1, so the walk runs W-1 steps past the
  // tile and recomputes those ds. The tile owns dx over [t0, t1) and, at t0 = 0,
  // the W-1 negative indices that are the gradient of the incoming state.
  const int tend = t1 - 1 + kWidth - 1;
  const int u_min = t0 == 0 ? -(kWidth - 1) : t0;

  // wr[j] is the tap at lag j; wf is the same bank in the order the dx
  // contraction wants. Holding both is not a duplicated register cost: the two
  // index the same eight values, and nvcc allocates them as one bank.
  float wr[kWidth];
  float wf[kWidth];
  float dwacc[kWidth];
#pragma unroll
  for (int j = 0; j < kWidth; ++j) {
    wr[j] = static_cast<float>(
        weight[static_cast<long>(channel) * kWidth + kWidth - 1 - j]);
    wf[j] =
        static_cast<float>(weight[static_cast<long>(channel) * kWidth + j]);
    dwacc[j] = 0.0f;
  }
  const float bias_of_channel =
      bias == nullptr ? 0.0f : static_cast<float>(bias[channel]);
  float dbacc = 0.0f;

  // Batch entries are independent, so the walk carries kBwdStreams of them at
  // once rather than one after another. That is what puts more than one load per
  // thread in flight over a serial timestep walk: the stream loads at a given t
  // have no dependence on each other, and interleaving them costs no traffic,
  // because the same bytes are read either way.
  //
  // A dead stream in the final partial group addresses its own clamped batch
  // entry, so every load is in bounds and needs no predicate; only the
  // accumulations and the stores are held back, which is what keeps a dead
  // stream from double-counting the entry it was clamped onto.
  for (int b0 = 0; b0 < batch; b0 += kBwdStreams) {
    const int live = batch - b0;
    long xbase[kBwdStreams];
    long sbase[kBwdStreams];
    float xw[kBwdStreams][kWidth];
    float dsw[kBwdStreams][kWidth];
#pragma unroll
    for (int g = 0; g < kBwdStreams; ++g) {
      const int b = b0 + min(g, live - 1);
      xbase[g] = static_cast<long>(b) * seqlen * channels + channel;
      sbase[g] = static_cast<long>(b) * (kWidth - 1) * channels + channel;
#pragma unroll
      for (int j = 0; j < kWidth; ++j) {
        // Pre-shift state, as in the forward: slot j lands at lag j+1.
        xw[g][j] = j < kWidth - 1
                       ? read_extended(x, initial_state, xbase[g], sbase[g],
                                       channels, seqlen, kWidth, t0 - 1 - j)
                       : 0.0f;
        // ds before the tile is zero at t0 = 0 because there is no output there,
        // and is never read at t0 > 0 because u_min holds dx back until the
        // window is full of ds values this tile computed.
        dsw[g][j] = 0.0f;
      }
    }

    for (int t = t0; t <= tend; ++t) {
      // Every stream's loads are issued before any of them is consumed, so the
      // group's misses overlap instead of serializing.
      float xc[kBwdStreams];
      float dyc[kBwdStreams];
#pragma unroll
      for (int g = 0; g < kBwdStreams; ++g) {
        const long at = xbase[g] + static_cast<long>(t) * channels;
        xc[g] = t < seqlen ? static_cast<float>(x[at]) : 0.0f;
        dyc[g] = (dy != nullptr && t < seqlen) ? static_cast<float>(dy[at]) : 0.0f;
      }

#pragma unroll
      for (int g = 0; g < kBwdStreams; ++g) {
#pragma unroll
        for (int j = kWidth - 1; j > 0; --j) {
          xw[g][j] = xw[g][j - 1];
        }
        xw[g][0] = xc[g];

        float ds = dyc[g];
        if (activation && dy != nullptr) {
          float acc = 0.0f;
#pragma unroll
          for (int j = kWidth - 1; j >= 0; --j) {
            acc = fmaf(wr[j], xw[g][j], acc);
          }
          ds *= silu_grad_of(acc + bias_of_channel);
        }

        // The tile owns the parameter gradient over [t0, t1); the overhang past
        // t1 belongs to the next tile and must not be counted twice.
        if (t < t1 && g < live) {
#pragma unroll
          for (int j = 0; j < kWidth; ++j) {
            dwacc[j] = fmaf(ds, xw[g][j], dwacc[j]);
          }
          dbacc += ds;
        }

#pragma unroll
        for (int j = kWidth - 1; j > 0; --j) {
          dsw[g][j] = dsw[g][j - 1];
        }
        dsw[g][0] = ds;

        const int u = t - (kWidth - 1);
        if (u >= u_min && g < live) {
          float acc = 0.0f;
#pragma unroll
          for (int j = kWidth - 1; j >= 0; --j) {
            acc = fmaf(wf[j], dsw[g][j], acc);
          }
          // The trailing window is returned as the next call's state, so its
          // cotangent lands on the extended index it was sliced from. Below
          // T = W-1 that index is negative and the contribution belongs to the
          // gradient of the incoming state, which the same test covers.
          if (dfinal_state != nullptr) {
            const int i = u - (seqlen - (kWidth - 1));
            if (i >= 0 && i < kWidth - 1) {
              acc += static_cast<float>(
                  dfinal_state[sbase[g] + static_cast<long>(i) * channels]);
            }
          }
          if (u >= 0) {
            dx[xbase[g] + static_cast<long>(u) * channels] =
                static_cast<input_t>(acc);
          } else if (dinitial_state != nullptr) {
            dinitial_state[sbase[g] +
                           static_cast<long>(u + kWidth - 1) * channels] =
                static_cast<input_t>(acc);
          }
        }
      }
    }
  }

  // Plain stores, one slice per time tile, so no output is read back and
  // nothing needs zeroing before the launch. dwacc is indexed by lag; tap k is
  // lag kWidth-1-k. The bound is the template parameter so that the register
  // array is never indexed dynamically, which would put it in local memory.
  float *dw = dweight_parts + static_cast<long>(part) * channels * kWidth +
              static_cast<long>(channel) * kWidth;
#pragma unroll
  for (int j = 0; j < kWidth; ++j) {
    dw[kWidth - 1 - j] = dwacc[j];
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

int bwd_time_tiles(int seqlen) {
  return (seqlen + kBwdTileT - 1) / kBwdTileT;
}

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

// Backward operands, gathered so the width dispatch names them once instead of
// once per instantiated width.
template <typename input_t> struct BwdArgs {
  const input_t *dy;
  const input_t *dfinal_state;
  const input_t *x;
  const input_t *weight;
  const input_t *bias;
  const input_t *initial_state;
  input_t *dx;
  input_t *dinitial_state;
  float *dweight_parts;
  float *dbias_parts;
  int batch;
  int seqlen;
  int channels;
  bool activation;
};

template <typename input_t, int kWidth>
void launch_bwd_width(const BwdArgs<input_t> &a) {
  const int threads = block_width(a.channels);
  const dim3 grid((a.channels + threads - 1) / threads,
                  bwd_time_tiles(a.seqlen), 1);
  conv1d_bwd_kernel<input_t, kWidth>
      <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          a.dy, a.dfinal_state, a.x, a.weight, a.bias, a.initial_state, a.dx,
          a.dinitial_state, a.dweight_parts, a.dbias_parts, a.batch, a.seqlen,
          a.channels, a.activation);
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
  const BwdArgs<input_t> a{
      data_or_null<input_t>(dy),
      data_or_null<input_t>(dfinal_state),
      x.const_data_ptr<input_t>(),
      weight.const_data_ptr<input_t>(),
      data_or_null<input_t>(bias),
      data_or_null<input_t>(initial_state),
      dx.data_ptr<input_t>(),
      mutable_or_null<input_t>(dinitial_state),
      dweight_parts.data_ptr<float>(),
      dbias_parts.has_value() ? dbias_parts->data_ptr<float>() : nullptr,
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1)),
      static_cast<int>(x.size(2)),
      activation,
  };
  // Width is a template parameter, not an argument: it sizes the five register
  // arrays the walk carries, so a runtime width would size all of them at
  // kMaxWidth and cap occupancy at the widest case for every call. The
  // assertion is what keeps the case list from falling behind the bound; with it
  // the default arm is unreachable, because the host already refused every width
  // outside [1, kMaxWidth].
  static_assert(kMaxWidth == 8, "the width dispatch below enumerates 1 .. 8");
  switch (static_cast<int>(weight.size(1))) {
  case 1:
    launch_bwd_width<input_t, 1>(a);
    break;
  case 2:
    launch_bwd_width<input_t, 2>(a);
    break;
  case 3:
    launch_bwd_width<input_t, 3>(a);
    break;
  case 4:
    launch_bwd_width<input_t, 4>(a);
    break;
  case 5:
    launch_bwd_width<input_t, 5>(a);
    break;
  case 6:
    launch_bwd_width<input_t, 6>(a);
    break;
  case 7:
    launch_bwd_width<input_t, 7>(a);
    break;
  case 8:
    launch_bwd_width<input_t, 8>(a);
    break;
  default:
    TORCH_CHECK(false, "causal_conv1d_bwd: width ", weight.size(1),
                " has no instantiation; the bound is ", kMaxWidth);
  }
}

} // namespace

int64_t causal_conv1d_bwd_parts(int64_t seqlen) {
  return (seqlen + kBwdTileT - 1) / kBwdTileT;
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
