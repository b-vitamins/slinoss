// Causal depthwise conv1d kernels.
//
// Class: DRAM-bound, both directions. The arithmetic intensity below is 1.75 and
// 2.85 flop/B against a ridge of 164, and measured DRAM traffic is within 7% of
// the compulsory count for the forward and 14% for the backward, so there are few
// bytes left to recover and the class is not in question. SERIAL-tiny is not the
// escape either: the two are 29% and 56% of the conv step against that class's 2%
// ceiling. The reduction that closes the backward is a third kernel and declares
// SERIAL-tiny; its own block is at the end of this comment.
//
// Achieved fraction of the measured copy ceiling, on an RTX A6000 with clocks
// unlocked and the device contended, two runs per shape:
//
//   scripts/perf/profile_op.py --op conv --shape <name> --mode step
//
//               standard  wide  long  ragged
//   forward            88    85    92      88
//   backward           77    53    84      74
//
// OPEN DEFECT: the backward clears no shape's floor of 85%, and the forward
// straddles it at wide, where four runs read 84 to 85.
//
// The backward's shortfall is latency, not traffic: long_scoreboard is the
// dominant stall at 59% to 71%, and achieved occupancy is 43% to 59% against a
// ceiling the register arrays set, 67% at W = 4 and 58% at W = 8. The per-thread
// state is four arrays of length W, which is 68 registers at W = 8, one block slot
// short of the 16 the SM would hold. Four levers were measured against that and
// reverted for regressing duration: batch on grid.z, which quadruples the block
// count at the standard shape but scales the partial count and the per-block
// prologue with batch and costs 8% to 10%; a time-direction prefetch group at 2
// and at 4 timesteps, which raises requested loads and registers; dropping wf to
// index wr backwards, which costs 17 registers at W = 8; and __launch_bounds__ on
// either kernel, which buys the occupancy back as spill traffic and loses 4% at
// wide. Both of the last two also raise achieved bandwidth while slowing the
// kernel, because spill and re-read traffic count toward it. Clearing the floor at
// W = 8 needs 1.8x the throughput at unchanged traffic, which needs a
// decomposition that does not hold 4W floats per thread.
//
// Compulsory byte count, per token per channel, at bfloat16 with W = 4:
//
//   forward   read x 2 B, write y 2 B: 4 B for 2W-1 = 7 flop, 1.75 flop/B.
//   backward  read x 2 B, read dy 2 B, write dx 2 B, plus the partials at
//             (W+1)*4/(kBwdTileT*B) = 0.31 B: 6.31 B for 4W+2 = 18 flop,
//             2.85 flop/B.
//
// Both kernels re-read past their tile: the forward re-reads W-1 activations for
// the prologue, the backward re-reads W-1 timesteps of both x and dy and
// recomputes their ds, because dx at u needs ds at u .. u+W-1. Those re-reads
// raise requested loads to 1.4x the compulsory reads at W = 4 and 2.2x at W = 8,
// and do not appear in the count above: L2 absorbs most of them, which is why DRAM
// traffic stays near the compulsory figure. The tap bank is D*W*4 B and is
// L2-resident for every reachable D. Every figure in this block is analytic and
// holds no claim about achieved bandwidth.
//
// Decomposition. One thread per channel, walking the time tile with the window
// in registers. The layout is channels-last, so a warp at a fixed timestep reads
// 32 consecutive channels: one coalesced transaction. No shared memory, so there
// is no tile to swizzle and occupancy is bounded by registers and by the
// blocks-per-SM limit alone.
//
// The forward grid is (channel tiles, time tiles, batch). The backward grid is
// (channel tiles, time tiles), with batch in the block's serial loop.
//
// Output layout. The forward writes y token-major, (B,T,D), or head-major,
// (B,H,T,P) with D = H*P and channel d = h*P + p, and the backward reads dy in
// whichever layout the forward wrote. Both are one base plus one stride:
//
//   token-major  base = b*T*D + d,               stride = D
//   head-major   base = b*T*D + h*T*P + p,       stride = P
//
// and stride = D is head-major at H = 1, so the head-major expression covers both
// and neither the prologue nor the loop branches on the layout. Only the store
// and the dy load move; x, the incoming state, the trailing window, dx, and the
// partials are token-major either way, so no byte count above changes.
//
// Coalescing. A warp still moves one contiguous run, but the run is P elements
// rather than D, because one (b,h,t) row is P contiguous elements. P is a
// multiple of 16, so at every reachable dtype the run is a whole number of
// 32-byte sectors and a warp's request splits into aligned sectors exactly as it
// did token-major; the sector count and the DRAM byte count are unchanged in
// both directions, measured, not argued.
//
// What P changes is the request count. A warp covers 32 channels and the block's
// channel tile is 64 wide, so a warp's run crosses a head boundary unless P is a
// multiple of 64: never at P = 64, one warp in three at P = 48, every warp at
// P = 16. Rows of two heads are seqlen*P apart, so a crossing run costs two
// requests instead of one. The forward absorbs that in the store pipe and is
// unaffected. The backward reads dy on the critical path and is slower for it,
// by single-digit percent at P = 48 and P = 16 and not at all at P = 64. Removing
// that would take a thread mapping over (t, p) inside one head rather than over
// channels, which is a different decomposition, not a store index.
//
// Partial reduction. Class: SERIAL-tiny. conv1d_reduce_parts_kernel reads the
// stack the backward leaves, (W+1)*S*D floats with S = ceil(T/kBwdTileT), and
// writes D*(W+1) elements: 1.475 MB read at the standard shape against the 29.8 MB
// the backward moves. Its kernel time there is 2.42 us with the bias slice and
// 2.17 us without, against the 359 us step scripts/bench/bench_conv.py reports at
// that shape, so 0.67% of the step against that class's 2% ceiling. Measured on an
// RTX A6000, clocks unlocked, device shared at 9% utilization, ten launches under
// nsys.
//
// A bandwidth is the wrong bar for it. The stack is 5.898 MB at the long shape, the
// largest of the five, against a 6 MB L2 the backward has just written it into:
// back-to-back launches on that stack run 4.76 us each, 1239 GB/s, 1.8x the
// measured DRAM ceiling, so a resident stack is served from L2 and no DRAM figure
// describes the kernel.
//
// One launch writes both parameter gradients. The grid is (channel tiles, W + 1)
// and the last slice of the second axis is the bias, so an absent bias is one slice
// fewer rather than a second launch. A block sums its slice over S with kReduceRows
// threads striding it, combines the rows in ascending row order through shared
// memory, and casts in the store, which transposes to the weight's own (D,W). Fixed
// order and no float atomic, so two runs agree bitwise. The order is not ATen's,
// whose .sum(0) carries four mod-4 accumulators, so the result differs from the
// expression it replaces in the last bits; matching those would cap the S axis at
// four threads and pin the kernel to an ATen internal no contract holds fixed.
//
// That expression is five launches with a bias and three without, 13.5 us of kernel
// time and 43.4 us of wall against one launch, 2.42 us, and 3.32 us. What a step
// sees of that depends on the caller. A paired step driving the backends directly
// reads 9.664 us, interval [-14.400, -5.120] us over 40 pairs, because there the
// launches queue behind the backward kernel and only their kernel time is exposed.
// Through autograd, whose engine leaves host gaps between them, bench_conv reads
// 419 and 430 us before against 357 and 359 us after, unpaired medians at 17% to
// 72% spread that resolve nothing on their own.
//
// Shared memory is rows[kReduceRows][kReduceChannels] floats, and the block is
// kReduceChannels wide, so a warp's store covers two rows, 32 consecutive floats,
// 32 distinct banks. The combine reads one row across kReduceChannels lanes, 16
// distinct banks. Neither conflicts.

#include "causal_conv1d.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <type_traits>

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

template <typename input_t, int kWidth>
__global__ void conv1d_fwd_kernel(const input_t *__restrict__ x,
                                  const input_t *__restrict__ weight,
                                  const input_t *__restrict__ bias,
                                  const input_t *__restrict__ initial_state,
                                  input_t *__restrict__ y,
                                  input_t *__restrict__ final_state, int seqlen,
                                  int channels, int y_rows, bool activation) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }
  const int t0 = blockIdx.y * kTileT;
  const int t1 = min(t0 + kTileT, seqlen);
  const long xbase = static_cast<long>(blockIdx.z) * seqlen * channels + channel;
  const long sbase =
      static_cast<long>(blockIdx.z) * (kWidth - 1) * channels + channel;
  // y is (B, D/y_rows, T, y_rows), which is token-major at y_rows == channels.
  // The whole layout is this base and the stride y_rows, so the store below is
  // one expression and the head-major case costs it no branch and no copy.
  const int head = channel / y_rows;
  const long ybase = static_cast<long>(blockIdx.z) * seqlen * channels +
                     static_cast<long>(head) * seqlen * y_rows +
                     (channel - head * y_rows);

  // wr[j] is the tap that multiplies lag j, so tap kWidth-1 is the current
  // token.
  float wr[kWidth];
  float xw[kWidth];
#pragma unroll
  for (int j = 0; j < kWidth; ++j) {
    wr[j] = static_cast<float>(
        weight[static_cast<long>(channel) * kWidth + kWidth - 1 - j]);
    // Pre-shift state: the tile loop shifts before it loads, so slot j must
    // hold the sample that lands at lag j+1, i.e. x[t0-1-j]. Slot kWidth-1
    // shifts into lag kWidth, which no tap multiplies.
    xw[j] = j < kWidth - 1
                ? read_extended(x, initial_state, xbase, sbase, channels, seqlen,
                                kWidth, t0 - 1 - j)
                : 0.0f;
  }
  const float bias_of_channel =
      bias == nullptr ? 0.0f : static_cast<float>(bias[channel]);

  // The group's loads are all issued before any of them is consumed, so the
  // group's misses overlap instead of serializing on one global latency per step.
  // The index is clamped rather than predicated, so every load is in bounds; only
  // the store is held back on the tail, and the window a clamped load pollutes is
  // dead because the tile ends with the group.
  for (int t = t0; t < t1; t += kFwdPrefetch) {
    float xc[kFwdPrefetch];
#pragma unroll
    for (int p = 0; p < kFwdPrefetch; ++p) {
      const int tp = min(t + p, t1 - 1);
      xc[p] = static_cast<float>(x[xbase + static_cast<long>(tp) * channels]);
    }
#pragma unroll
    for (int p = 0; p < kFwdPrefetch; ++p) {
#pragma unroll
      for (int j = kWidth - 1; j > 0; --j) {
        xw[j] = xw[j - 1];
      }
      xw[0] = xc[p];
      // Oldest tap first, which is the order the reference sums in.
      float acc = 0.0f;
#pragma unroll
      for (int j = kWidth - 1; j >= 0; --j) {
        acc = fmaf(wr[j], xw[j], acc);
      }
      acc += bias_of_channel;
      if (t + p < t1) {
        y[ybase + static_cast<long>(t + p) * y_rows] =
            static_cast<input_t>(activation ? silu_of(acc) : acc);
      }
    }
  }

  // The next call's window: the W-1 timesteps that precede its first token.
  // Below T = W-1 that window straddles the incoming state, which the extended
  // read handles without a second path.
  if (final_state != nullptr && t1 == seqlen) {
#pragma unroll
    for (int i = 0; i < kWidth - 1; ++i) {
      final_state[sbase + static_cast<long>(i) * channels] =
          static_cast<input_t>(read_extended(x, initial_state, xbase, sbase,
                                             channels, seqlen, kWidth,
                                             seqlen - (kWidth - 1) + i));
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
                  int channels, int dy_rows, bool activation) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }
  // dy carries the layout the forward's y was written in, so it is read through
  // its own base and stride; see the header comment. dx, x, and both windows are
  // token-major whatever dy is.
  const int head = channel / dy_rows;
  const int dyrow = channel - head * dy_rows;
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
  // index the same values and nvcc allocates them as one bank. Dropping wf and
  // indexing wr backwards instead costs 17 registers at kWidth = 8.
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

  // Batch entries are independent and are walked one after another. Putting the
  // batch axis on the grid instead costs the block's whole prologue, the tap bank
  // and the W-1 window, once per entry rather than once per block, and multiplies
  // the partial count by the batch.
  for (int b = 0; b < batch; ++b) {
    const long xbase = static_cast<long>(b) * seqlen * channels + channel;
    const long sbase = static_cast<long>(b) * (kWidth - 1) * channels + channel;
    const long dybase = static_cast<long>(b) * seqlen * channels +
                        static_cast<long>(head) * seqlen * dy_rows + dyrow;
    float xw[kWidth];
    float dsw[kWidth];
#pragma unroll
    for (int j = 0; j < kWidth; ++j) {
      // Pre-shift state, as in the forward: slot j lands at lag j+1.
      xw[j] = j < kWidth - 1
                  ? read_extended(x, initial_state, xbase, sbase, channels,
                                  seqlen, kWidth, t0 - 1 - j)
                  : 0.0f;
      // ds before the tile is zero at t0 = 0 because there is no output there,
      // and is never read at t0 > 0 because u_min holds dx back until the
      // window is full of ds values this tile computed.
      dsw[j] = 0.0f;
    }

    for (int t = t0; t <= tend; ++t) {
      const float xc = t < seqlen ? static_cast<float>(
                                       x[xbase + static_cast<long>(t) * channels])
                                  : 0.0f;
      const float dyc =
          (dy != nullptr && t < seqlen)
              ? static_cast<float>(dy[dybase + static_cast<long>(t) * dy_rows])
              : 0.0f;

#pragma unroll
      for (int j = kWidth - 1; j > 0; --j) {
        xw[j] = xw[j - 1];
      }
      xw[0] = xc;

      float ds = dyc;
      if (activation && dy != nullptr) {
        float acc = 0.0f;
#pragma unroll
        for (int j = kWidth - 1; j >= 0; --j) {
          acc = fmaf(wr[j], xw[j], acc);
        }
        ds *= silu_grad_of(acc + bias_of_channel);
      }

      // The tile owns the parameter gradient over [t0, t1); the overhang past
      // t1 belongs to the next tile and must not be counted twice.
      if (t < t1) {
#pragma unroll
        for (int j = 0; j < kWidth; ++j) {
          dwacc[j] = fmaf(ds, xw[j], dwacc[j]);
        }
        dbacc += ds;
      }

#pragma unroll
      for (int j = kWidth - 1; j > 0; --j) {
        dsw[j] = dsw[j - 1];
      }
      dsw[0] = ds;

      const int u = t - (kWidth - 1);
      if (u >= u_min) {
        float acc = 0.0f;
#pragma unroll
        for (int j = kWidth - 1; j >= 0; --j) {
          acc = fmaf(wf[j], dsw[j], acc);
        }
        // The trailing window is returned as the next call's state, so its
        // cotangent lands on the extended index it was sliced from. Below
        // T = W-1 that index is negative and the contribution belongs to the
        // gradient of the incoming state, which the same test covers.
        if (dfinal_state != nullptr) {
          const int i = u - (seqlen - (kWidth - 1));
          if (i >= 0 && i < kWidth - 1) {
            acc += static_cast<float>(
                dfinal_state[sbase + static_cast<long>(i) * channels]);
          }
        }
        if (u >= 0) {
          dx[xbase + static_cast<long>(u) * channels] =
              static_cast<input_t>(acc);
        } else if (dinitial_state != nullptr) {
          dinitial_state[sbase + static_cast<long>(u + kWidth - 1) * channels] =
              static_cast<input_t>(acc);
        }
      }
    }
  }

  // Plain stores, one slice per time tile, so no output is read back and
  // nothing needs zeroing before the launch. dwacc is indexed by lag; tap k is
  // lag kWidth-1-k. The bound is the template parameter so that the register
  // array is never indexed dynamically, which would put it in local memory.
  //
  // The partial buffer is tap-major, (P,W,D), so consecutive channels land in
  // consecutive floats and each tap is one coalesced store. Channel-major would
  // put a stride of W between neighbouring threads, which is W separate sectors
  // per store and W times the L1 wavefronts for the same bytes.
  float *dw = dweight_parts + static_cast<long>(part) * kWidth * channels +
              channel;
#pragma unroll
  for (int j = 0; j < kWidth; ++j) {
    dw[static_cast<long>(kWidth - 1 - j) * channels] = dwacc[j];
  }
  if (dbias_parts != nullptr) {
    dbias_parts[static_cast<long>(part) * channels + channel] = dbacc;
  }
}

// Reduce the backward's per-time-tile partials into the parameter gradients.
//
// blockIdx.y selects the slice: tap k at k < width, the bias at k == width. Both
// are a stack of `parts` rows of `channels` floats and differ only in the row
// stride, so one instantiation covers the taps and the bias, and the launch count
// is one whether or not a bias is present.
//
// The block is (kReduceChannels, kReduceRows). Threads split the slice stack
// along rows, each holding one accumulator, and the per-row partials meet in
// shared memory. Every reduction here is over a compile-time count in a
// compile-time order, so the sum does not depend on how the blocks were
// scheduled.
template <typename output_t>
__global__ void
conv1d_reduce_parts_kernel(const float *__restrict__ dweight_parts,
                           const float *__restrict__ dbias_parts,
                           output_t *__restrict__ dweight,
                           output_t *__restrict__ dbias, int parts, int channels,
                           int width) {
  const int channel = blockIdx.x * kReduceChannels + threadIdx.x;
  const bool bias_slice = static_cast<int>(blockIdx.y) == width;
  const long stride = bias_slice ? channels : static_cast<long>(width) * channels;
  const float *src =
      bias_slice ? dbias_parts
                 : dweight_parts + static_cast<long>(blockIdx.y) * channels;

  // Four in flight per thread, which is the whole stack at the standard shape:
  // the reduction is one add per four bytes and has no arithmetic to hide a load
  // behind, so loads in flight is the only lever it has.
  float acc = 0.0f;
  if (channel < channels) {
#pragma unroll 4
    for (int i = threadIdx.y; i < parts; i += kReduceRows) {
      acc += src[static_cast<long>(i) * stride + channel];
    }
  }

  // A warp spans two rows of the tile and 16 consecutive floats of each, so its
  // 32 addresses are 32 consecutive floats: conflict-free without a pad.
  __shared__ float rows[kReduceRows][kReduceChannels];
  rows[threadIdx.y][threadIdx.x] = acc;
  __syncthreads();
  if (threadIdx.y != 0 || channel >= channels) {
    return;
  }
  // Ascending rather than a tree: one order, fixed here, is what makes the
  // result reproducible.
  float total = rows[0][threadIdx.x];
#pragma unroll
  for (int r = 1; r < kReduceRows; ++r) {
    total += rows[r][threadIdx.x];
  }
  if (bias_slice) {
    dbias[channel] = static_cast<output_t>(total);
  } else {
    // The weight's own layout, reached by the store index. Consecutive channels
    // land width elements apart, which is the one uncoalesced access in the
    // kernel and covers D*W elements: it does not scale with the sequence, where
    // every load above does.
    dweight[static_cast<long>(channel) * width + blockIdx.y] =
        static_cast<output_t>(total);
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

// Turn a runtime width into a compile-time one, once for both directions.
//
// Width is a template parameter, not an argument: it sizes every register array
// the walk carries, so a runtime width would size all of them at kMaxWidth and
// cap occupancy at the widest case for every call. The assertion is what keeps
// the case list from falling behind the bound; with it the fallthrough is
// unreachable, because the host already refused every width outside
// [1, kMaxWidth]. Duplicating the switch per direction is how one arm falls
// behind the other, so there is exactly one.
template <typename Fn> void dispatch_width(int width, Fn &&launch) {
  static_assert(kMaxWidth == 8, "the width dispatch below enumerates 1 .. 8");
  switch (width) {
  case 1:
    launch(std::integral_constant<int, 1>{});
    return;
  case 2:
    launch(std::integral_constant<int, 2>{});
    return;
  case 3:
    launch(std::integral_constant<int, 3>{});
    return;
  case 4:
    launch(std::integral_constant<int, 4>{});
    return;
  case 5:
    launch(std::integral_constant<int, 5>{});
    return;
  case 6:
    launch(std::integral_constant<int, 6>{});
    return;
  case 7:
    launch(std::integral_constant<int, 7>{});
    return;
  case 8:
    launch(std::integral_constant<int, 8>{});
    return;
  default:
    break;
  }
  TORCH_CHECK(false, "causal_conv1d: width ", width,
              " has no instantiation; the bound is ", kMaxWidth);
}

// Forward operands, gathered so the width dispatch names them once instead of
// once per instantiated width.
template <typename input_t> struct FwdArgs {
  const input_t *x;
  const input_t *weight;
  const input_t *bias;
  const input_t *initial_state;
  input_t *y;
  input_t *final_state;
  int batch;
  int seqlen;
  int channels;
  int y_rows;
  bool activation;
};

template <typename input_t, int kWidth>
void launch_fwd_width(const FwdArgs<input_t> &a) {
  const int threads = block_width(a.channels);
  const dim3 grid((a.channels + threads - 1) / threads, time_tiles(a.seqlen),
                  a.batch);
  conv1d_fwd_kernel<input_t, kWidth>
      <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          a.x, a.weight, a.bias, a.initial_state, a.y, a.final_state, a.seqlen,
          a.channels, a.y_rows, a.activation);
}

template <typename input_t>
void launch_fwd(const at::Tensor &x, const at::Tensor &weight,
                const std::optional<at::Tensor> &bias,
                const std::optional<at::Tensor> &initial_state,
                const at::Tensor &y,
                const std::optional<at::Tensor> &final_state, int64_t y_rows,
                bool activation) {
  const FwdArgs<input_t> a{
      x.const_data_ptr<input_t>(),
      weight.const_data_ptr<input_t>(),
      data_or_null<input_t>(bias),
      data_or_null<input_t>(initial_state),
      y.data_ptr<input_t>(),
      mutable_or_null<input_t>(final_state),
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1)),
      static_cast<int>(x.size(2)),
      static_cast<int>(y_rows),
      activation,
  };
  dispatch_width(static_cast<int>(weight.size(1)), [&](auto w) {
    launch_fwd_width<input_t, decltype(w)::value>(a);
  });
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
  int dy_rows;
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
          a.channels, a.dy_rows, a.activation);
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
                const std::optional<at::Tensor> &dbias_parts, int64_t dy_rows,
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
      static_cast<int>(dy_rows),
      activation,
  };
  dispatch_width(static_cast<int>(weight.size(1)), [&](auto w) {
    launch_bwd_width<input_t, decltype(w)::value>(a);
  });
}

// No width dispatch: the reduction holds no register array of width, so width is
// an argument and one instantiation per dtype covers every tap count.
template <typename output_t>
void launch_reduce_parts(const at::Tensor &dweight_parts,
                         const std::optional<at::Tensor> &dbias_parts,
                         const at::Tensor &dweight,
                         const std::optional<at::Tensor> &dbias) {
  const int parts = static_cast<int>(dweight_parts.size(0));
  const int width = static_cast<int>(dweight_parts.size(1));
  const int channels = static_cast<int>(dweight_parts.size(2));
  const dim3 block(kReduceChannels, kReduceRows);
  const dim3 grid((channels + kReduceChannels - 1) / kReduceChannels,
                  width + (dbias.has_value() ? 1 : 0));
  conv1d_reduce_parts_kernel<output_t>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          dweight_parts.const_data_ptr<float>(), data_or_null<float>(dbias_parts),
          dweight.data_ptr<output_t>(), mutable_or_null<output_t>(dbias), parts,
          channels, width);
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
                       int64_t y_rows, bool activation) {
  const at::cuda::CUDAGuard guard(x.device());
  switch (x.scalar_type()) {
  case at::ScalarType::BFloat16:
    launch_fwd<at::BFloat16>(x, weight, bias, initial_state, y, final_state,
                             y_rows, activation);
    break;
  case at::ScalarType::Half:
    launch_fwd<at::Half>(x, weight, bias, initial_state, y, final_state, y_rows,
                         activation);
    break;
  case at::ScalarType::Float:
    launch_fwd<float>(x, weight, bias, initial_state, y, final_state, y_rows,
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
                       int64_t dy_rows, bool activation) {
  const at::cuda::CUDAGuard guard(x.device());
  switch (x.scalar_type()) {
  case at::ScalarType::BFloat16:
    launch_bwd<at::BFloat16>(dy, dfinal_state, x, weight, bias, initial_state,
                             dx, dinitial_state, dweight_parts, dbias_parts,
                             dy_rows, activation);
    break;
  case at::ScalarType::Half:
    launch_bwd<at::Half>(dy, dfinal_state, x, weight, bias, initial_state, dx,
                         dinitial_state, dweight_parts, dbias_parts, dy_rows,
                         activation);
    break;
  case at::ScalarType::Float:
    launch_bwd<float>(dy, dfinal_state, x, weight, bias, initial_state, dx,
                      dinitial_state, dweight_parts, dbias_parts, dy_rows,
                      activation);
    break;
  default:
    TORCH_CHECK(false, "causal_conv1d_bwd: unsupported dtype ",
                x.scalar_type());
  }
}

void causal_conv1d_reduce_parts(const at::Tensor &dweight_parts,
                                const std::optional<at::Tensor> &dbias_parts,
                                const at::Tensor &dweight,
                                const std::optional<at::Tensor> &dbias) {
  const at::cuda::CUDAGuard guard(dweight_parts.device());
  switch (dweight.scalar_type()) {
  case at::ScalarType::BFloat16:
    launch_reduce_parts<at::BFloat16>(dweight_parts, dbias_parts, dweight, dbias);
    break;
  case at::ScalarType::Half:
    launch_reduce_parts<at::Half>(dweight_parts, dbias_parts, dweight, dbias);
    break;
  case at::ScalarType::Float:
    launch_reduce_parts<float>(dweight_parts, dbias_parts, dweight, dbias);
    break;
  default:
    TORCH_CHECK(false, "causal_conv1d_reduce_parts: unsupported dtype ",
                dweight.scalar_type());
  }
}

} // namespace slinoss
