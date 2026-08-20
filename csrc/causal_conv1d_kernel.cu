// Causal depthwise conv1d kernels.
//
// Class: DRAM-bound, both directions. The arithmetic intensity below is 1.75 and
// 2.85 flop/B against a ridge of 164, and measured DRAM traffic is within 7% of the
// compulsory count for the forward and 1% for the backward, so there are few bytes
// left to recover and the class is not in question. SERIAL-tiny is not the escape
// either: both scale with B*T*D and run 32.991 and 51.103 us a launch at the standard
// shape against the reduction's 2.752, and it is the reduction that class describes.
// That reduction is a third kernel; its own block is at the end of this comment.
//
// Achieved fraction of the measured copy ceiling, on an RTX A6000 (sm_86) with
// clocks unlocked, which the fleet denies locking, and with device 0 holding
// nothing but the MPS daemon before and after each run. One run per shape, three
// launches each:
//
//   scripts/perf/profile_op.py --op conv --shape <name> --mode step
//
//                        standard  wide  long  ragged
//   bf16  forward            87.6  84.3  92.4    87.0
//         backward, staged   81.6  62.5  88.1    80.5
//   fp32  forward            94.1  91.3  96.9    93.4
//         backward, scalar   96.8  83.1  95.5    96.6
//
// bwd_can_stage is false at four bytes, so bf16 takes the staged path here and fp32
// the scalar one. The harness verdict is not this ratio: it scores against a fitted
// floor carrying a 4.483 us latency constant, so it reads higher, and it is the gate.
// On it the forward passes at every shape in both dtypes, 94.3 the lowest, and the
// backward passes at standard, long, and ragged -- 90.6, 92.3, 89.3 bf16 and 100.3,
// 97.3, 100.2 fp32 -- and fails at wide alone, 67.4 and 84.9.
//
// OPEN DEFECT: wide, W = 8, both dtypes. Nothing else here is under its floor; the
// tiny shape's forward and backward carry no class. The staged row above reads 1.7 to
// 2.6 points under the same code before kBwdTargetBlocks bounded the grid, because
// the kernel now writes a smaller partial stack in the same time. The ratio charges
// it for the byte it stopped writing, so duration is the invariant across that change
// and duration fell: 102.25 to 96.446 us at long.
//
// What held the backward down was request granularity, not traffic and not
// occupancy. The scalar walk asks for one element per lane per step, 64 B a warp at
// two bytes, and the same code at float32 -- four bytes a lane, 128 B a warp --
// measures 96.8% of the ceiling at the standard shape against 67.5% for the bf16
// scalar walk, with DRAM traffic within 1% of the compulsory count either way. That
// 67.5 is from the build before kBwdTargetBlocks and has no row above, because
// bwd_can_stage falls back to the scalar walk at two bytes only when alignment fails
// and no bench shape reaches it. The strip is what raises the request: the
// block stages both read streams through shared memory in 16 B lane requests, 512 B
// a warp, and the walk then issues no global load at all. That is the staged row
// above, and paired against the scalar walk in one process it is -17.58 us at the
// standard shape, interval [-17.65, -17.53] over 100 pairs at 96.5% coverage, and
// -43.48 us at wide, [-45.37, -41.39].
//
// What binds after it, at W = 8: 74 registers cap theoretical occupancy at 50% and
// achieved at 40.1%, and no one stall dominates -- wait 26.7%, long_scoreboard
// 25.3%, issue_active 61.7%. The kernel has neither the warps to cover the strip's
// round trip nor a single stall to remove, and its traffic is already compulsory:
// 25.37 MB read a launch against 25.17 compulsory.
//
// Occupancy is not the missing warps, though, which is measurable and measured: a
// __launch_bounds__ of 8 blocks an SM holds the staged walk to 64 registers with
// nothing spilled and takes achieved occupancy to 49.2%, and the kernel does not move,
// 93.086 to 93.662 us against a 0.10% run-to-run spread. Clearing the floor there
// needs the per-thread state under five arrays of length W, which means splitting the
// tap axis across threads and combining dx through shared memory, not another
// staging, scheduling, or occupancy change. Every lever measured and reverted is
// recorded at the site it would have touched: the batch axis on grid.z, a
// time-direction prefetch group, dropping wf, a launch bound of 8, an unrolled fill,
// a hoisted per-column fill map, a lagged strip window in place of the register
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
// an SM and 8 against the 8 and 6 that 60 and 74 registers allow, so the strip does
// not bind occupancy at either width.
//
// The window arrays reach registers and not local memory, which a register count
// alone does not show: every tap loop between an array's declaration and its uses is
// unrolled at a compile-time W, so no index is dynamic. Measured, all three kernels
// at the standard and wide shapes in both dtypes, local load sectors 0 and local
// store sectors 0, ptxas reporting no spill either way.
//
// No extent here follows a sequence length. The strip spans
// kBwdTilesPerBlock*kBwdTileT = 32 timesteps plus 2*(W-1) of halo, the reduction's
// shared array is kReduceRows floats, and both are tuning constants; T enters only
// through the grid, which kBwdTargetBlocks bounds. P enters only as the dy and y row
// stride, so no P caps a caller's tile or chunk length either.
//
// bwd_can_stage holds the alignment the
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
// What P changes is the request count, and only that. A warp covers 32 channels and
// the block's channel tile is 64 wide, so a warp's run crosses a head boundary
// unless P is a multiple of 64: never at P = 64, one warp in three at P = 48, every
// warp at P = 16. Rows of two heads are seqlen*P apart, so a crossing run costs two
// requests instead of one.
//
// The requests do not reach the sectors, because HEAD_MULTIPLE holds P at 16 or
// above: the smallest piece a crossing can cut a run into is 16 elements, 32 B in
// bf16 and 64 B in fp32, and both are whole sectors. Two 64 B requests carry the
// four sectors one 128 B request carries. Measured on the staged fill, whose slot
// index runs flat over (stream, timestep, channel vector): 2.468M load sectors at
// P = 16 against 2.472M at P = 64.
//
// Nor do they reach the duration, on either walk. Backward kernel alone, standard
// shape, 40 reps, device holding nothing but the MPS daemon before and after,
// min/med/max in us:
//
//   staged  token 56.224/56.320/59.392   P=64 54.272/56.320/57.344
//           P=48  55.296/57.184/57.344   P=16 56.320/56.320/63.488
//   scalar  token 106.496/108.384/113.664  P=64 103.424/107.520/110.688
//           P=48  106.496/107.520/114.688  P=16 105.472/107.504/118.784
//
// Three of the four staged medians are one number and the fourth is 0.864 us over it,
// under the event timer's own 1.024 us tick. Every scalar median is at or under the
// token-major one, by 0.8% at all three P. The minima order the other way, so the two
// statistics disagree on sign at a magnitude of one tick: no P is slower, including
// P = 16, where every warp crosses. Nothing here is a crossing cost; the split is
// free at every P the mixer can be built at.
//
// The reason is that a request count is not this kernel's limiter. The standard-shape
// backward issues on 52.5% of its cycles and stalls 40.9% on long_scoreboard, so it
// waits on sectors, and the crossing does not add any. The strip is therefore not
// what pays for the crossing, and the scalar walk does not pay for it either.
//
// Partial reduction. Class: SERIAL-tiny. conv1d_reduce_parts_kernel reads the stack
// the backward leaves, (W+1)*S*D floats, and writes D*(W+1) elements. S is the
// backward's block count along time, which kBwdTargetBlocks bounds, so the stack
// stops growing once the grid is full and the reduction's work stops following T:
//
//   shape     S    stack       reduce us med [min, max]   share of step
//   tiny      8    0.003 MB    --                         0.563%
//   standard  64   0.737 MB    2.752 [2.688, 2.752]       1.082%
//   wide      64   1.769 MB    5.728 [5.664, 5.760]       1.539%
//   long      128  1.475 MB    4.128 [4.096, 4.224]       1.141%
//   ragged    63   0.726 MB    2.752 [2.688, 2.752]       0.853%
//
// against that class's 2% ceiling, every shape inside it. The long shape is what set
// the bound: S there was ceil(T/kBwdTileT) = 512, a 5.898 MB stack read in 12.30 us
// for 3.28% of the step, over the ceiling. S is 128 now and the kernel runs 4.128 us.
// The backward pays the same bound in the other direction and gains from it, because
// the stack is a store it no longer makes: 102.25 us to 96.446 us at long, where it
// is the shape's DRAM-bound kernel at 92.3% of the floor.
//
// Two other closures were available and both lose. A last-block reduction inside the
// backward removes the launch and not the bytes: the last block still reads every
// slice, and the resident figure below bounds what removing the launch is worth at
// about 2 us. Holding the reduction's work fixed while the stack still grows is not
// a thing that exists, because the reduction must read every slice the backward
// wrote; the only lever on its work is the slice count, which is this one.
//
// A bandwidth is the wrong bar for the kernel whichever bound it runs under. The
// stack is small enough to sit in a 6 MB L2 the backward has just written it into:
// back-to-back launches on a resident stack run 4.76 us each at the largest stack,
// 1239 GB/s, 1.8x the measured DRAM ceiling. Under the step it reads the DRAM figure
// instead, because the backward's own dx stream evicts most of the stack before the
// reduction reaches it, so the resident figure is a floor and no DRAM figure
// describes the kernel. What is left is a latency ladder: at S = 64 a thread walks
// two of the kReduceRows rows it strides, so the wide shape, whose W = 8 makes the
// widest stack of the five out of the smallest S, is the one nearest the ceiling and
// the one whose time follows its bytes least. Its whole margin to the DRAM ceiling
// is 3.1 us of a 368 us step.
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

// The activation runs in float at every operand width. kFast picks the hardware
// exponential and reciprocal over the correctly-rounded ones, and it is set from
// the operand width, not from a build flag.
//
// At two bytes an element the result is rounded to 8 mantissa bits before it is
// stored or before it multiplies a two-byte cotangent. MUFU.EX2 carries at most
// 2 + floor(1.16*|x|) ulp of float and MUFU.RCP under 2 ulp, so the pair is
// bounded by about 2e-6 relative over the range a sigmoid argument can take,
// which is 2000 times finer than the 2^-8 ulp it is rounded into: the stored
// value is the same one the correctly-rounded pair would store, save where it
// already sat on a rounding boundary, and the parity bound in the tests is one
// ulp for exactly that reason. At four bytes the correctly-rounded pair stays.
// The float32 bounds in the tests are counts of 2^-24 roundings with no room for
// a second source of error, and the fast pair costs 5 instructions against 24.
template <bool kFast>
__device__ __forceinline__ float sigmoid_of(float s) {
  // exp(-s) overflows to inf for very negative s, and 1/inf is zero, so the
  // saturated ends are exact rather than NaN. Both intrinsics saturate the same
  // way: EX2 returns inf on overflow and RCP returns zero on inf.
  if constexpr (kFast) {
    return __fdividef(1.0f, 1.0f + __expf(-s));
  } else {
    return 1.0f / (1.0f + expf(-s));
  }
}

template <bool kFast>
__device__ __forceinline__ float silu_of(float s) {
  return s * sigmoid_of<kFast>(s);
}

template <bool kFast>
__device__ __forceinline__ float silu_grad_of(float s) {
  const float g = sigmoid_of<kFast>(s);
  return g * (1.0f + s * (1.0f - g));
}

// Operand widths that round the activation to fewer mantissa bits than the fast
// pair's error, which is every width this kernel is instantiated at except float.
template <typename input_t>
constexpr bool kFastActivation = sizeof(input_t) == 2;

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
            static_cast<input_t>(
                activation ? silu_of<kFastActivation<input_t>>(acc) : acc);
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
__global__ void __launch_bounds__(kMaxChannelsPerBlock *kBwdTilesPerBlock,
                                  kBwdMinBlocksPerSm)
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
                  int dx_pitch, int dy_rows, int groups, bool activation) {
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
  // The strip and the tap-gradient combine share one allocation. The combine runs
  // once, after the last walk, so the strip is dead by then, and the strip is the
  // larger of the two at every width the kernel is instantiated at: sharing them
  // makes the combine free of the resource that bounds blocks per SM. Declared as
  // float so the combine gets the language's alignment; the strip reinterprets it
  // at the operand's own width, because widening the strip to float would double
  // it.
  constexpr int kStripFloats =
      kStage ? (2 * kSpan * kMaxChannelsPerBlock *
                    static_cast<int>(sizeof(input_t)) +
                3) /
                   4
             : 0;
  constexpr int kCombineFloats =
      (kWidth + 1) * kBwdTilesPerBlock * kMaxChannelsPerBlock;
  __shared__ __align__(kAlignBytes) float
      shared[kStripFloats > kCombineFloats ? kStripFloats : kCombineFloats];
  // x first, then dy.
  input_t *const strip = reinterpret_cast<input_t *>(shared);

  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  // dy carries the layout the forward's y was written in, so it is read through
  // its own base and stride; see the header comment. dx, x, and both windows are
  // token-major whatever dy is.
  const int head = channel / dy_rows;
  const int dyrow = channel - head * dy_rows;
  const bool live_channel = channel < channels;
  const int col0 = blockIdx.x * blockDim.x;
  const int cols = min(static_cast<int>(blockDim.x), channels - col0);
  const int lanes = blockDim.x * blockDim.y;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

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
    wr[j] = live_channel ? static_cast<float>(weight[tap + kWidth - 1 - j]) : 0.0f;
    wf[j] = live_channel ? static_cast<float>(weight[tap + j]) : 0.0f;
    dwacc[j] = 0.0f;
  }
  const float bias_of_channel =
      (bias == nullptr || !live_channel) ? 0.0f
                                        : static_cast<float>(bias[channel]);
  float dbacc = 0.0f;

  // Tile groups the block walks, ascending, one slice written for all of them.
  // The stride is gridDim.y, which the host caps at a block count rather than
  // letting it follow T; see kBwdTargetBlocks. The parameter-gradient
  // accumulators live outside this loop, so the group count reaches the reduction
  // as nothing at all.
  for (int group = blockIdx.y; group < groups; group += gridDim.y) {
    // blockDim.y time tiles per group, so the last group of the sequence can hold
    // a tile past the end. That tile walks nothing. It still stages: the strip's
    // barriers are block-wide, and the columns a lane fills are not the column it
    // walks.
    const int t0 = (group * kBwdTilesPerBlock + threadIdx.y) * kBwdTileT;
    const bool owns = live_channel && t0 < seqlen;
    const int t1 = min(t0 + kBwdTileT, seqlen);
    // dx at index u needs ds at u .. u+W-1, so the walk runs W-1 steps past the
    // tile and recomputes those ds. The tile owns dx over [t0, t1) and, at
    // t0 = 0, the W-1 negative indices that are the gradient of the incoming
    // state.
    //
    // One thread per tile, not several. Splitting the tile across kBwdSubTiles
    // threads on blockDim.z, with their parameter-gradient accumulators meeting
    // in shared memory, doubles the thread count and raises achieved occupancy
    // from 52.7 to 57.5 percent at the standard shape and from 40.0 to 46.4 at
    // the wide one, and it measured 54.45 us against 51.69 standard, 107.51
    // against 92.81 wide, 53.40 against 51.15 ragged, and 101.82 against 102.25
    // long. Each half walks its own kWidth-1 overhang, so steps per owned
    // timestep rise from 19/16 to 11/8 and issue_active with them, 61.8 percent
    // against 55.0: at compulsory traffic the extra instructions cost more than
    // the extra warps buy.
    const int u_min = t0 == 0 ? -(kWidth - 1) : t0;
    // Strip geometry: the group's first timestep. Negative in the first group of
    // the sequence, where the incoming state stands in for x.
    const int tstrip = group * kBwdTilesPerBlock * kBwdTileT - (kWidth - 1);
    if constexpr (!kStage) {
      // No barrier on this path, so a lane with no tile can leave the group
      // early. It may not leave the kernel: the combine below is block-wide.
      if (!owns) {
        continue;
      }
    }

    // Batch entries are independent and are walked one after another. Putting the
    // batch axis on the grid instead costs the block's whole prologue, the tap
    // bank and the W-1 window, once per entry rather than once per block, and
    // multiplies the partial count by the batch.
    for (int b = 0; b < batch; ++b) {
      // x and dx are each one column band of a wider tensor and carry their own
      // two leading strides; see the header comment. The two bands are cut from
      // different buffers, so neither stride is shared.
      const long xbase = static_cast<long>(b) * x_batch + channel;
      const long dxbase = static_cast<long>(b) * dx_batch + channel;
      const long sbase =
          static_cast<long>(b) * (kWidth - 1) * channels + channel;
      const long dybase = static_cast<long>(b) * seqlen * channels +
                          static_cast<long>(head) * seqlen * dy_rows + dyrow;

      if constexpr (kStage) {
        // One strip serves every batch entry and every group, so the fill waits
        // for the previous readers. A slot is kVec channels at one timestep of
        // one stream, and the divisors that unpack it are compile-time, so the
        // unpacking is shifts. Giving each lane a fixed column group and hoisting
        // its address arithmetic out of the batch loop instead measured 53.17 us
        // against 51.46 at the standard shape and 104.35 against 101.80 at the
        // long one, at two more registers: the arithmetic this loop repeats is
        // cheaper than the registers holding it across the walk.
        __syncthreads();
        const long xrow = static_cast<long>(b) * x_batch + col0;
        const long srow = static_cast<long>(b) * (kWidth - 1) * channels + col0;
        const long dyb = static_cast<long>(b) * seqlen * channels;
        constexpr int kSlots = 2 * kSpan * kVecCols;
        // Not unrolled. An unrolled fill holds one 16 B value per copy in flight,
        // and those registers are the walk's too: the fill is already issuing a
        // 512 B request per warp, so the parallelism it would add is worth less
        // than the occupancy it costs. Unrolled it cost 24 registers at
        // kWidth = 4 and measured 76.98 us at the standard shape against 56.26.
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
          // Past the sequence end the strip holds zero rather than the last
          // token: the overhang's ds is a value the dx contraction needs, not a
          // dead slot, so a clamped fill would fold that token in twice. Before
          // the sequence the x stream is the incoming state, which is the scalar
          // path's extended read in vector form, and the dy stream holds nothing:
          // those steps belong to the previous group's tiles and nothing reads
          // them here.
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

      // The activation window is a register array in both paths. Reading it out
      // of the strip at a lag instead saves the array and its kWidth-1 moves per
      // step, and it costs kWidth shared loads a step: at kWidth = 8 that took
      // the register count from 76 to 67 and L1 utilization from 16 to 60
      // percent, and measured 106.01 us against 91.81 at the wide shape and 55.88
      // against 51.46 at the standard one. The window stays in registers.
      //
      // Pre-shift state, as in the forward: slot j lands at lag j+1. In the
      // staged path the strip holds those steps, incoming state included, and
      // index t0-1-j is at or above tstrip for every j the window uses.
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
      //
      // The owned steps and the kWidth-1 overhang past them are separate loops.
      // One loop with a test per step charges the owned steps for four things
      // none of them can reach: the tap gradient's ownership test, the bounds
      // test on a global load that cannot run past the end inside a tile, the
      // trailing window's cotangent, whose extended index is past every u this
      // loop produces, and the window shifts themselves. kBwdTileT is a multiple
      // of kMaxWidth, so a full tile unrolls by kWidth with no remainder and the
      // shifts become register renames: 2*(kWidth-1) moves a step leave the loop,
      // which is 14 of 114 executed instructions per step at kWidth = 8.
      //
      // Only the staged path can pay for that unroll. The scalar path carries the
      // two global address chains the strip replaces, and unrolled by 8 at
      // kWidth = 8 it passed the launch bound's register cap and spilled: 98,304
      // local load sectors and 24,576 store, which fails the class whatever the
      // percentage. It keeps its shifts.
      constexpr int kWalkUnroll = kStage ? kWidth : 1;
#pragma unroll kWalkUnroll
      for (int t = t0; t < t1; ++t) {
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
          // t < t1 <= seqlen, so both streams are live: the zero-past-the-end
          // test belongs to the overhang and is not charged here.
          xc = static_cast<float>(x[xbase + static_cast<long>(t) * x_pitch]);
          dyc = dy != nullptr
                    ? static_cast<float>(
                          dy[dybase + static_cast<long>(t) * dy_rows])
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
          ds *= silu_grad_of<kFastActivation<input_t>>(acc + bias_of_channel);
        }

        // The tile owns the parameter gradient over [t0, t1), which is this loop
        // exactly.
#pragma unroll
        for (int j = 0; j < kWidth; ++j) {
          dwacc[j] = fmaf(ds, xw[j], dwacc[j]);
        }
        dbacc += ds;

#pragma unroll
        for (int j = kWidth - 1; j > 0; --j) {
          dsw[j] = dsw[j - 1];
        }
        dsw[0] = ds;

        // u <= t1 - kWidth here, and the trailing window's cotangent lands at
        // u >= seqlen - (kWidth-1), which needs t1 > seqlen: dfinal_state is the
        // overhang's business and its test is not in this loop.
        const int u = t - (kWidth - 1);
        if (u >= u_min) {
          float acc = 0.0f;
#pragma unroll
          for (int j = kWidth - 1; j >= 0; --j) {
            acc = fmaf(wf[j], dsw[j], acc);
          }
          if (u >= 0) {
            dx[dxbase + static_cast<long>(u) * dx_pitch] =
                static_cast<input_t>(acc);
          } else if (dinitial_state != nullptr) {
            dinitial_state[sbase +
                           static_cast<long>(u + kWidth - 1) * channels] =
                static_cast<input_t>(acc);
          }
        }
      }

      // dx at index u needs ds at u .. u+kWidth-1, so the walk runs kWidth-1
      // steps past the tile and recomputes those ds. Their tap gradient belongs
      // to the next tile and is not accumulated here. The trip count is
      // compile-time, so on the staged path the shifts here are renames too; the
      // scalar path holds this loop rolled for the register reason above.
#pragma unroll kWalkUnroll
      for (int k = 0; k < kWidth - 1; ++k) {
        const int t = t1 + k;
        float xc;
        float dyc;
        if constexpr (kStage) {
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
          xc = live ? static_cast<float>(
                          x[xbase + static_cast<long>(t) * x_pitch])
                    : 0.0f;
          dyc = (dy != nullptr && live)
                    ? static_cast<float>(
                          dy[dybase + static_cast<long>(t) * dy_rows])
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
          ds *= silu_grad_of<kFastActivation<input_t>>(acc + bias_of_channel);
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
            dinitial_state[sbase +
                           static_cast<long>(u + kWidth - 1) * channels] =
                static_cast<input_t>(acc);
          }
        }
      }
    }
  }

  // The block's tiles meet here. One slice per block and not one per tile, so
  // the stack the reduction reads is bounded by the grid; and the grid is capped
  // above, so the stack is bounded by the machine rather than by T.
  //
  // Every lane reaches this point with its accumulators, whether or not it ever
  // owned a tile: nothing zeroes the partial buffers and the contract says they
  // are written in full, so a block that runs must write its whole slice, and a
  // lane that owned nothing contributes the zero it started with. Only a lane
  // past the channel count drops out, and only at the store.
  //
  // The combine reuses the strip's allocation, so the barrier below is also what
  // guarantees the last strip reader is done.
  __syncthreads();
#pragma unroll
  for (int j = 0; j < kWidth; ++j) {
    shared[(j * kBwdTilesPerBlock + threadIdx.y) * kMaxChannelsPerBlock +
           threadIdx.x] = dwacc[j];
  }
  if (dbias_parts != nullptr) {
    shared[(kWidth * kBwdTilesPerBlock + threadIdx.y) * kMaxChannelsPerBlock +
           threadIdx.x] = dbacc;
  }
  __syncthreads();
  if (threadIdx.y != 0 || !live_channel) {
    return;
  }

  // Plain stores, so no output is read back and nothing needs zeroing before the
  // launch. dwacc is indexed by lag; tap k is lag kWidth-1-k. The bound is the
  // template parameter so that the register array is never indexed dynamically,
  // which would put it in local memory.
  //
  // The partial buffer is tap-major, (S,W,D), so consecutive channels land in
  // consecutive floats and each tap is one coalesced store. Channel-major would
  // put a stride of W between neighbouring threads, which is W separate sectors
  // per store and W times the L1 wavefronts for the same bytes.
  //
  // Ascending in threadIdx.y, as the group loop was ascending in group: the
  // partition and the order in it are fixed, so two runs agree bitwise. The
  // partition follows kBwdTargetBlocks and the channel count, so it is a
  // property of the launch geometry and not of the schedule.
  float *dw =
      dweight_parts + static_cast<long>(blockIdx.y) * kWidth * channels +
      channel;
#pragma unroll
  for (int j = 0; j < kWidth; ++j) {
    float total = shared[j * kBwdTilesPerBlock * kMaxChannelsPerBlock +
                         threadIdx.x];
#pragma unroll
    for (int r = 1; r < kBwdTilesPerBlock; ++r) {
      total += shared[(j * kBwdTilesPerBlock + r) * kMaxChannelsPerBlock +
                      threadIdx.x];
    }
    dw[static_cast<long>(kWidth - 1 - j) * channels] = total;
  }
  if (dbias_parts != nullptr) {
    float total = shared[kWidth * kBwdTilesPerBlock * kMaxChannelsPerBlock +
                         threadIdx.x];
#pragma unroll
    for (int r = 1; r < kBwdTilesPerBlock; ++r) {
      total += shared[(kWidth * kBwdTilesPerBlock + r) * kMaxChannelsPerBlock +
                      threadIdx.x];
    }
    dbias_parts[static_cast<long>(blockIdx.y) * channels + channel] = total;
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

// The backward's time axis: the tile groups a sequence holds, and the blocks that
// walk them, which is also the partial count. One function so the launch and the
// host's buffer allocation cannot disagree.
struct BwdTimeAxis {
  int groups;
  int blocks;
};

BwdTimeAxis bwd_time_axis(int seqlen, int channels) {
  const int tiles = (seqlen + kBwdTileT - 1) / kBwdTileT;
  const int groups = (tiles + kBwdTilesPerBlock - 1) / kBwdTilesPerBlock;
  const int width = block_width(channels);
  const int rows = (channels + width - 1) / width;
  const int cap = (kBwdTargetBlocks + rows - 1) / rows;
  // Floor: a block takes on a second group only when a whole second group is
  // there for it. Rounding up instead would halve the grid just past the cap,
  // which costs the backward more than the shorter stack saves the reduction.
  const int iters = groups / cap < 1 ? 1 : groups / cap;
  return BwdTimeAxis{groups, (groups + iters - 1) / iters};
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
  const BwdTimeAxis axis = bwd_time_axis(a.seqlen, a.channels);
  const dim3 block(threads, kBwdTilesPerBlock);
  const dim3 grid((a.channels + threads - 1) / threads, axis.blocks, 1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (bwd_can_stage<input_t>(a)) {
    conv1d_bwd_kernel<input_t, kWidth, true><<<grid, block, 0, stream>>>(
        a.dy, a.dfinal_state, a.x, a.weight, a.bias, a.initial_state, a.dx,
        a.dinitial_state, a.dweight_parts, a.dbias_parts, a.batch, a.seqlen,
        a.channels, a.x_batch, a.x_pitch, a.dx_batch, a.dx_pitch, a.dy_rows,
        axis.groups, a.activation);
    return;
  }
  conv1d_bwd_kernel<input_t, kWidth, false><<<grid, block, 0, stream>>>(
      a.dy, a.dfinal_state, a.x, a.weight, a.bias, a.initial_state, a.dx,
      a.dinitial_state, a.dweight_parts, a.dbias_parts, a.batch, a.seqlen,
      a.channels, a.x_batch, a.x_pitch, a.dx_batch, a.dx_pitch, a.dy_rows,
      axis.groups, a.activation);
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

int64_t causal_conv1d_bwd_parts(int64_t seqlen, int64_t channels) {
  return bwd_time_axis(static_cast<int>(seqlen), static_cast<int>(channels))
      .blocks;
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
