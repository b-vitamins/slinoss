// Causal depthwise conv1d kernels.
//
// Class: DRAM-bound, both directions. The arithmetic intensity below is 1.75 and
// 2.85 flop/B against a ridge of 164, and measured DRAM traffic is within 7% of the
// compulsory count for the forward and 1% for the backward, so there are few bytes
// left to recover and the class is not in question. SERIAL-tiny is not the escape
// either: both scale with B*T*D and run 33.2 and 51.9 us a launch at the standard
// shape against the reduction's 2.4, and it is the reduction that class describes.
// That reduction is a third kernel; its own block is at the end of this comment.
//
// Achieved fraction of the measured copy ceiling, on an RTX A6000 (sm_86) with
// clocks unlocked, which the fleet denies locking, and with device 0 holding
// nothing but the MPS daemon before and after each run. One run per shape, three
// launches each:
//
//   scripts/perf/profile_op.py --op conv --shape <name> --mode step
//
//                       standard  wide  long  ragged
//   forward                 87.3  84.1  92.2    87.0
//   backward, scalar walk   67.5  46.0  78.1    66.5
//   backward, staged strip  83.3  65.1  90.1    82.8
//
// OPEN DEFECT: the backward clears the 85% floor at the long shape only, and the
// forward straddles it at wide, 84.1 here against 85.0 in the run before this one.
// Run-to-run spread on identical code at the standard shape is 83.3 to 84.0, so the
// backward's 1.7-point shortfall there is of that order and the wide shape's 20 is
// not.
//
// What held the backward down was request granularity, not traffic and not
// occupancy. The scalar walk asks for one element per lane per step, 64 B a warp at
// two bytes, and the same code at float32 -- four bytes a lane, 128 B a warp --
// measured 96.9% of the ceiling against 67.5% at bfloat16, with DRAM traffic within
// 1% of the compulsory count either way. The strip is what raises the request: the
// block stages both read streams through shared memory in 16 B lane requests, 512 B
// a warp, and the walk then issues no global load at all. That is the staged row
// above, and paired against the scalar walk in one process it is -17.58 us at the
// standard shape, interval [-17.65, -17.53] over 100 pairs at 96.5% coverage, and
// -43.48 us at wide, [-45.37, -41.39].
//
// What binds after it, at W = 8: 77 registers cap theoretical occupancy at 50% and
// achieved at 40.0%, and no one stall dominates -- long_scoreboard 25.7%, wait
// 27.0%, not_selected 11.7%, issue_active 60.7%. The kernel has neither the warps
// to cover the strip's round trip nor a single stall to remove, and its traffic is
// already compulsory: 25.37 MB read a launch against 25.17 compulsory. Clearing the
// floor there needs the per-thread state under five arrays of length W, which means
// splitting the tap axis across threads and combining dx through shared memory, not
// another staging or scheduling change. Every lever measured and reverted is
// recorded at the site it would have touched: the batch axis on grid.z, a
// time-direction prefetch group, dropping wf, __launch_bounds__, an unrolled fill, a
// hoisted per-column fill map, a lagged strip window in place of the register
// window, and splitting the time tile across threads.
//
// Compulsory byte count, per token per channel, at bfloat16 with W = 4:
//
//   forward   read x 2 B, write y 2 B: 4 B for 2W-1 = 7 flop, 1.75 flop/B.
//   backward  read x 2 B, read dy 2 B, write dx 2 B, plus the partials at
//             (W+1)*4/(kBwdTileT*B) = 0.31 B: 6.31 B for 4W+2 = 18 flop,
//             2.85 flop/B.
//
// Both kernels read past their tile: the forward re-reads W-1 activations for the
// prologue, and the backward needs W-1 timesteps of both x and dy on either side of
// its tile, because dx at u needs ds at u .. u+W-1. The staged path pays that once
// per block rather than once per thread -- the strip spans 1.19x the block's owned
// timesteps at W = 4 and 1.44x at W = 8 -- and the scalar path pays it per thread,
// which raises requested loads to 1.4x the compulsory reads at W = 4 and 2.2x at
// W = 8. Neither appears in the count above: L2 absorbs the overlap, so requested
// load bytes at the standard shape are 26.37 MB a launch against 18.92 MB out of
// DRAM. The tap bank is D*W*4 B and is L2-resident for every reachable D. The byte
// counts in this block are analytic and hold no claim about achieved bandwidth.
//
// Decomposition. One thread per channel per time tile, walking the tile with a
// 5-array register window. The layout is channels-last, so a warp at a fixed
// timestep covers 32 consecutive channels: one coalesced transaction. The forward
// walk reads global directly. The backward block stages its two read streams into a
// shared strip first and the walk then reads only shared memory; the fill is 16 B
// lane requests over a flat slot index, so it is one 512 B request a warp whatever
// the channel tile's relation to a head. The strip is
// 2 * (kBwdTilesPerBlock*kBwdTileT + 2*(W-1)) * kMaxChannelsPerBlock elements of
// input_t, 9728 B at W = 4 and 11776 B at W = 8, which caps the block count at 10
// an SM and 8 against the 8 and 6 that 64 and 77 registers allow, so the strip does
// not bind occupancy at either width. bwd_can_stage holds the alignment the
// vector fill needs; the same kernel, instantiated with kStage false, is the
// fallback and the only path at four bytes, where the request is wide already.
//
// The forward grid is (channel tiles, time tiles, batch). The backward grid is
// (channel tiles, time tile pairs), with batch in the block's serial loop and the
// pair on blockDim.y, so a block is 128 threads at every D of 64 or more.
//
// Input layout. x is one column band of a wider tensor, the value band of the
// fused input projection, whose token stride is the projection width and not D.
// The band is its two leading strides, which are arguments:
//
//   xbase = b*x_batch + d,   step = x_pitch
//
// A contiguous x is x_batch = T*D and x_pitch = D, so nothing else in the walk
// branches on the layout and a band costs no staging copy. dx is a band of one
// dproj buffer and carries its own pair; the two are cut from different tensors,
// so neither stride is shared. weight, bias, the incoming and trailing windows,
// and the partials are buffers of their own and stay contiguous.
//
// A warp's run is 32 elements inside one row, so the pitch reaches the request
// count only through alignment: rows start on a 32-byte sector boundary when
// x_pitch*sizeof(input_t) is a multiple of 32, and otherwise every row after the
// first is offset inside its sectors and a warp's run crosses one more boundary
// than it did contiguous. Measured against the contiguous case at the standard
// shape, output bitwise identical, 100 pairs at 96.5% coverage with device 0 holding
// nothing but the MPS daemon: a sector-aligned band costs +0.763% forward and
// +0.476% backward, a band whose base sits 16 B inside its sector +2.290% and
// +0.952%, and a pitch that is not a sector multiple +1.371% and +0.857%.
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
// unaffected. The backward reads dy on the critical path, and on the scalar walk was
// 4.5% slower at P = 48 and 6.9% at P = 16. The strip moves the crossing off the
// walk and into the fill, whose slot index runs flat over (stream, timestep, channel
// vector): 51.64, 51.82 and 52.22 us at P = 64, 48 and 16, so +0.35% and +1.12%,
// measured with another tenant holding 3.7 GB of device 0 at 24% utilization. What
// is left is request count alone. A 16 B vector never straddles a head, because P is
// a multiple of 16, so at P = 16 a warp's fill touches four times the rows for the
// same sectors: 2.468M load sectors against 2.472M at P = 64, and neither the sector
// count nor the DRAM count moves.
//
// Partial reduction. Class: SERIAL-tiny. conv1d_reduce_parts_kernel reads the
// stack the backward leaves, (W+1)*S*D floats with S = ceil(T/kBwdTileT), and
// writes D*(W+1) elements: 1.475 MB read at the standard shape against the 29.5 MB
// the backward moves. Its kernel time there is 2.42 us with the bias slice and
// 2.17 us without.
//
// OPEN DEFECT: it holds SERIAL-tiny at three of the four benchmarked shapes and
// fails at the long one. Fraction of the conv step, from the same runs as the table
// above: 0.80% standard, 1.84% wide, 3.28% long, 0.89% ragged, against that class's
// 2% ceiling. S follows T alone, so the stack does not shrink with batch and costs
// twice per token at B = 2 what it costs at B = 4; the long shape is the one that
// pairs the longest T with the smallest B, and there the kernel reads 5.898 MB and
// runs 12.30 us. Closing it takes a partial count that follows B*T, or a last-block
// reduction inside the backward, and the declared class is not in this file.
//
// A bandwidth is the wrong bar for it. That 5.898 MB stack is the largest of the
// five, against a 6 MB L2 the backward has just written it into: back-to-back
// launches on a resident stack run 4.76 us each, 1239 GB/s, 1.8x the measured DRAM
// ceiling. Under the step it reads 12.30 us instead, because the backward's own dx
// stream evicts most of the stack before the reduction reaches it, so the resident
// figure is a floor and no DRAM figure describes the kernel.
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
// Through autograd, whose engine leaves host gaps between them, the bench driver
// read 419 and 430 us before against 357 and 359 us after, unpaired medians at 17%
// to 72% spread that resolve nothing on their own.
//
// Shared memory is rows[kReduceRows][kReduceChannels] floats, and the block is
// kReduceChannels wide, so a warp's store covers two rows, 32 consecutive floats,
// 32 distinct banks. The combine reads one row across kReduceChannels lanes, 16
// distinct banks. Neither conflicts.

#include "causal_conv1d.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
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
// and past the end, the stream is zero. x steps by its own row pitch and the
// state by channels: the state is a buffer of its own and is never a band.
template <typename input_t>
__device__ __forceinline__ float
read_extended(const input_t *__restrict__ x, const input_t *__restrict__ state,
              long xbase, long sbase, int x_pitch, int channels, int seqlen,
              int width, int u) {
  if (u >= 0) {
    return u < seqlen ? static_cast<float>(x[xbase + static_cast<long>(u) * x_pitch])
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
                                  int channels, int x_batch, int x_pitch,
                                  int y_rows, bool activation) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }
  const int t0 = blockIdx.y * kTileT;
  const int t1 = min(t0 + kTileT, seqlen);
  // x is one column band of a wider tensor, so its two leading strides are
  // arguments; see the header comment. A contiguous x is x_batch = T*D and
  // x_pitch = D, which is what the arithmetic below reduces to.
  const long xbase = static_cast<long>(blockIdx.z) * x_batch + channel;
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
                ? read_extended(x, initial_state, xbase, sbase, x_pitch,
                                channels, seqlen, kWidth, t0 - 1 - j)
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
    // One 64-bit index per group, then 32-bit offsets inside it. The addresses
    // stay independent of each other, which a pointer bumped per group does not:
    // that form measured 35.11 us against 33.23 us at the standard shape.
    const long xg = xbase + static_cast<long>(t) * x_pitch;
    float xc[kFwdPrefetch];
#pragma unroll
    for (int p = 0; p < kFwdPrefetch; ++p) {
      xc[p] = static_cast<float>(x[xg + min(p, t1 - 1 - t) * x_pitch]);
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
                                             x_pitch, channels, seqlen, kWidth,
                                             seqlen - (kWidth - 1) + i));
    }
  }
}

template <typename input_t, int kWidth, bool kStage>
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
                  int channels, int x_batch, int x_pitch, int dx_batch,
                  int dx_pitch, int dy_rows, bool activation) {
  // Elements one lane asks for per staging instruction: kAlignBytes at the
  // operand's width. The walk consumes one element per lane per step, so its
  // warp request is 64 B at two bytes an element, where a staging warp's is
  // 512 B. Raising that request is the whole point of the strip.
  constexpr int kVec =
      kStage ? kAlignBytes / static_cast<int>(sizeof(input_t)) : 1;
  // Timesteps the strip holds. A block's tiles are adjacent in time and each
  // walks kWidth-1 steps past its own end, so one strip covers every tile in the
  // block and each step in it is fetched once for the block. The kWidth-1 steps
  // before the block are in it too: they are the first tile's incoming window,
  // and reading them from the strip is what leaves the walk with no global load
  // at all.
  constexpr int kSpan = kBwdTilesPerBlock * kBwdTileT + 2 * (kWidth - 1);
  constexpr int kVecCols = kMaxChannelsPerBlock / kVec;
  // x first, then dy, both at the operand's own width: widening to float would
  // double the strip, and the strip is what bounds blocks per SM.
  __shared__ __align__(kAlignBytes) input_t
      strip[kStage ? 2 * kSpan * kMaxChannelsPerBlock : 1];

  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  // dy carries the layout the forward's y was written in, so it is read through
  // its own base and stride; see the header comment. dx, x, and both windows are
  // token-major whatever dy is.
  const int head = channel / dy_rows;
  const int dyrow = channel - head * dy_rows;
  // blockDim.y time tiles per block, so the grid rounds up and the last block of
  // the axis can hold a tile past the end. That tile owns no partial slice and
  // walks nothing. It still stages: the strip's barriers are block-wide, and the
  // columns a lane fills are not the column it walks.
  const int part = blockIdx.y * kBwdTilesPerBlock + threadIdx.y;
  const int t0 = part * kBwdTileT;
  const bool owns = channel < channels && t0 < seqlen;
  if (!kStage && !owns) {
    return;
  }
  const int t1 = min(t0 + kBwdTileT, seqlen);
  // dx at index u needs ds at u .. u+W-1, so the walk runs W-1 steps past the
  // tile and recomputes those ds. The tile owns dx over [t0, t1) and, at t0 = 0,
  // the W-1 negative indices that are the gradient of the incoming state.
  //
  // One thread per tile, not several. Splitting the tile across kBwdSubTiles
  // threads on blockDim.z, with their parameter-gradient accumulators meeting in
  // shared memory so the partial count and the reduction stay fixed, doubles the
  // thread count and raises achieved occupancy from 52.7 to 57.5 percent at the
  // standard shape and from 40.0 to 46.4 at the wide one, and it measured 54.45 us
  // against 51.69 standard, 107.51 against 92.81 wide, 53.40 against 51.15 ragged,
  // and 101.82 against 102.25 long. Each half walks its own kWidth-1 overhang, so
  // steps per owned timestep rise from 19/16 to 11/8 and issue_active with them,
  // 61.8 percent against 55.0: at compulsory traffic the extra instructions cost
  // more than the extra warps buy. The wave count is not the binding resource.
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
    const long tap = static_cast<long>(channel) * kWidth;
    wr[j] = owns ? static_cast<float>(weight[tap + kWidth - 1 - j]) : 0.0f;
    wf[j] = owns ? static_cast<float>(weight[tap + j]) : 0.0f;
    dwacc[j] = 0.0f;
  }
  const float bias_of_channel =
      (bias == nullptr || !owns) ? 0.0f : static_cast<float>(bias[channel]);
  float dbacc = 0.0f;

  // Batch entries are independent and are walked one after another. Putting the
  // batch axis on the grid instead costs the block's whole prologue, the tap bank
  // and the W-1 window, once per entry rather than once per block, and multiplies
  // the partial count by the batch.
  // Strip geometry, fixed for the block: its first timestep, its column origin,
  // and how many of its columns exist. The first timestep is negative in the
  // first block of the axis, where the incoming state stands in for x.
  const int tstrip =
      blockIdx.y * kBwdTilesPerBlock * kBwdTileT - (kWidth - 1);
  const int col0 = blockIdx.x * blockDim.x;
  const int cols = min(static_cast<int>(blockDim.x), channels - col0);
  const int lanes = blockDim.x * blockDim.y;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  for (int b = 0; b < batch; ++b) {
    // x and dx are each one column band of a wider tensor and carry their own
    // two leading strides; see the header comment. The two bands are cut from
    // different buffers, so neither stride is shared.
    const long xbase = static_cast<long>(b) * x_batch + channel;
    const long dxbase = static_cast<long>(b) * dx_batch + channel;
    const long sbase = static_cast<long>(b) * (kWidth - 1) * channels + channel;
    const long dybase = static_cast<long>(b) * seqlen * channels +
                        static_cast<long>(head) * seqlen * dy_rows + dyrow;

    if constexpr (kStage) {
      // One strip serves every batch entry, so the fill waits for the previous
      // entry's readers. A slot is kVec channels at one timestep of one stream,
      // and the divisors that unpack it are compile-time, so the unpacking is
      // shifts. Giving each lane a fixed column group and hoisting its address
      // arithmetic out of the batch loop instead measured 53.17 us against 51.46
      // at the standard shape and 104.35 against 101.80 at the long one, at two
      // more registers: the arithmetic this loop repeats is cheaper than the
      // registers holding it across the walk.
      __syncthreads();
      const long xrow = static_cast<long>(b) * x_batch + col0;
      const long srow = static_cast<long>(b) * (kWidth - 1) * channels + col0;
      const long dyb = static_cast<long>(b) * seqlen * channels;
      constexpr int kSlots = 2 * kSpan * kVecCols;
      // Not unrolled. An unrolled fill holds one 16 B value per copy in flight,
      // and those registers are the walk's too: the fill is already issuing a
      // 512 B request per warp, so the parallelism it would add is worth less
      // than the occupancy it costs. Unrolled it cost 24 registers at kWidth = 4
      // and measured 76.98 us at the standard shape against 56.26.
#pragma unroll 1
      for (int slot = tid; slot < kSlots; slot += lanes) {
        const int stream = slot / (kSpan * kVecCols);
        const int rem = slot - stream * (kSpan * kVecCols);
        const int step = rem / kVecCols;
        const int col = (rem - step * kVecCols) * kVec;
        if (col >= cols) {
          continue;
        }
        const int t = tstrip + step;
        // Past the sequence end the strip holds zero rather than the last token:
        // the overhang's ds is a value the dx contraction needs, not a dead slot,
        // so a clamped fill would fold that token in twice. Before the sequence
        // the x stream is the incoming state, which is the scalar path's extended
        // read in vector form, and the dy stream holds nothing: those steps
        // belong to the previous block's tiles and nothing reads them here.
        int4 v = make_int4(0, 0, 0, 0);
        if (stream == 0) {
          if (t >= 0) {
            if (t < seqlen) {
              v = *reinterpret_cast<const int4 *>(
                  x + xrow + static_cast<long>(t) * x_pitch + col);
            }
          } else if (initial_state != nullptr) {
            v = *reinterpret_cast<const int4 *>(
                initial_state + srow +
                static_cast<long>(t + kWidth - 1) * channels + col);
          }
        } else if (step >= kWidth - 1) {
          if (dy != nullptr && t < seqlen) {
            const int ch = col0 + col;
            const int hd = ch / dy_rows;
            v = *reinterpret_cast<const int4 *>(
                dy + dyb + static_cast<long>(hd) * seqlen * dy_rows +
                (ch - hd * dy_rows) + static_cast<long>(t) * dy_rows);
          }
        } else {
          continue;
        }
        *reinterpret_cast<int4 *>(
            strip + (static_cast<long>(stream) * kSpan + step) *
                        kMaxChannelsPerBlock +
            col) = v;
      }
      __syncthreads();
      if (!owns) {
        continue;
      }
    }

    // The activation window is a register array in both paths. Reading it out of
    // the strip at a lag instead saves the array and its kWidth-1 moves per step,
    // and it costs kWidth shared loads a step: at kWidth = 8 that took the
    // register count from 76 to 67 and L1 utilization from 16 to 60 percent, and
    // measured 106.01 us against 91.81 at the wide shape and 55.88 against 51.46
    // at the standard one. The window stays in registers.
    //
    // Pre-shift state, as in the forward: slot j lands at lag j+1. In the staged
    // path the strip holds those steps, incoming state included, and index
    // t0-1-j is at or above tstrip for every j the window uses.
    float xw[kWidth];
    float dsw[kWidth];
#pragma unroll
    for (int j = 0; j < kWidth; ++j) {
      if (j < kWidth - 1) {
        if constexpr (kStage) {
          xw[j] = static_cast<float>(
              strip[(t0 - 1 - j - tstrip) * kMaxChannelsPerBlock +
                    threadIdx.x]);
        } else {
          xw[j] = read_extended(x, initial_state, xbase, sbase, x_pitch,
                                channels, seqlen, kWidth, t0 - 1 - j);
        }
      } else {
        xw[j] = 0.0f;
      }
      // ds before the tile is zero at t0 = 0 because there is no output there,
      // and is never read at t0 > 0 because u_min holds dx back until the
      // window is full of ds values this tile computed.
      dsw[j] = 0.0f;
    }

    // One step per iteration, both streams loaded by index. Loading a group of
    // steps ahead of their use is the forward's shape and it does not carry
    // here: the group needs two register arrays as long as it is, which cost 22
    // registers at kWidth = 4 and 49 at kWidth = 8, and the occupancy that buys
    // is worth more than the extra bytes in flight. The numbers are in the
    // header comment. Indexed and not a walked pointer for the same reason a
    // group is not held: a pointer bumped per step is a dependence on the
    // address itself, and measured 160.87 us against 133.46 us at the wide
    // shape.
    for (int t = t0; t <= tend; ++t) {
      float xc;
      float dyc;
      if constexpr (kStage) {
        // A warp reads 32 consecutive elements of one strip row, which share 16
        // four-byte banks two lanes to a word: one transaction, no pad needed.
        const int s = (t - tstrip) * kMaxChannelsPerBlock + threadIdx.x;
        xc = static_cast<float>(strip[s]);
        dyc = dy != nullptr
                  ? static_cast<float>(strip[kSpan * kMaxChannelsPerBlock + s])
                  : 0.0f;
      } else {
        // Past the sequence end both streams are zero rather than clamped: the
        // overhang's ds is a real value the dx contraction needs, not a dead
        // slot, so a clamped load would fold the last token in twice.
        const bool live = t < seqlen;
        xc = live
                 ? static_cast<float>(x[xbase + static_cast<long>(t) * x_pitch])
                 : 0.0f;
        dyc =
            (dy != nullptr && live)
                ? static_cast<float>(dy[dybase + static_cast<long>(t) * dy_rows])
                : 0.0f;
      }

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

      // The tile owns the parameter gradient over [t0, t1); the overhang past t1
      // belongs to the next tile and must not be counted twice.
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
          dx[dxbase + static_cast<long>(u) * dx_pitch] =
              static_cast<input_t>(acc);
        } else if (dinitial_state != nullptr) {
          dinitial_state[sbase + static_cast<long>(u + kWidth - 1) * channels] =
              static_cast<input_t>(acc);
        }
      }
    }
  }

  // A lane that owns no tile has no partial slice to write; in the staged path it
  // reached here only to fill the strip.
  if (!owns) {
    return;
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
  int x_batch;
  int x_pitch;
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
          a.channels, a.x_batch, a.x_pitch, a.y_rows, a.activation);
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
      static_cast<int>(x.stride(0)),
      static_cast<int>(x.stride(1)),
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
  int x_batch;
  int x_pitch;
  int dx_batch;
  int dx_pitch;
  int dy_rows;
  bool activation;
};

// Whether the backward can stage its two read streams through shared memory.
//
// The staged fill asks for kAlignBytes per lane, so every address it forms must
// be that aligned: the two bases, and every stride a slot index multiplies, in
// elements. Only float32 fails the class the strip is there to fix: at four bytes an
// element a lane's request is wide already and the scalar walk measured 96.9 percent
// of this host's copy ceiling against 67.5 at two, the pair in the header comment,
// so the strip would buy it nothing and cost it a round trip through shared memory.
template <typename input_t> bool bwd_can_stage(const BwdArgs<input_t> &a) {
  if constexpr (sizeof(input_t) != 2) {
    return false;
  } else {
    constexpr int vec = kAlignBytes / static_cast<int>(sizeof(input_t));
    const auto aligned = [](const void *p) {
      return reinterpret_cast<uintptr_t>(p) % kAlignBytes == 0;
    };
    return a.channels % vec == 0 && a.x_batch % vec == 0 &&
           a.x_pitch % vec == 0 && aligned(a.x) &&
           (a.dy == nullptr || (a.dy_rows % vec == 0 && aligned(a.dy))) &&
           (a.initial_state == nullptr || aligned(a.initial_state));
  }
}

template <typename input_t, int kWidth>
void launch_bwd_width(const BwdArgs<input_t> &a) {
  const int threads = block_width(a.channels);
  const int tiles = bwd_time_tiles(a.seqlen);
  const dim3 block(threads, kBwdTilesPerBlock);
  const dim3 grid((a.channels + threads - 1) / threads,
                  (tiles + kBwdTilesPerBlock - 1) / kBwdTilesPerBlock, 1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (bwd_can_stage<input_t>(a)) {
    conv1d_bwd_kernel<input_t, kWidth, true><<<grid, block, 0, stream>>>(
        a.dy, a.dfinal_state, a.x, a.weight, a.bias, a.initial_state, a.dx,
        a.dinitial_state, a.dweight_parts, a.dbias_parts, a.batch, a.seqlen,
        a.channels, a.x_batch, a.x_pitch, a.dx_batch, a.dx_pitch, a.dy_rows,
        a.activation);
    return;
  }
  conv1d_bwd_kernel<input_t, kWidth, false><<<grid, block, 0, stream>>>(
      a.dy, a.dfinal_state, a.x, a.weight, a.bias, a.initial_state, a.dx,
      a.dinitial_state, a.dweight_parts, a.dbias_parts, a.batch, a.seqlen,
      a.channels, a.x_batch, a.x_pitch, a.dx_batch, a.dx_pitch, a.dy_rows,
      a.activation);
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
      static_cast<int>(x.stride(0)),
      static_cast<int>(x.stride(1)),
      static_cast<int>(dx.stride(0)),
      static_cast<int>(dx.stride(1)),
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
