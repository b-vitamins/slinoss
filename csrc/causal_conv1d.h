// Causal depthwise conv1d: launch surface shared by the bindings and the kernels.
//
// One entry point per direction. Both take the already-validated tensors; every
// shape, dtype, layout, and width check lives in the Python host code so an
// invalid call raises the same error whichever backend is selected.
#pragma once

#include <ATen/ATen.h>
#include <optional>

namespace slinoss {

// Largest tap count the kernels accept. The window, the tap bank, and the
// per-lag gradient accumulators are register arrays of this length, so the
// bound is a register budget rather than a padding choice. Above it the host
// refuses the call.
constexpr int kMaxWidth = 8;

// Timesteps one forward block walks. The prologue re-reads W-1 activations per
// block, so the tile is what amortizes that re-read; the tile is also what
// supplies the grid its blocks along time, so it trades that re-read against
// how much of the machine one launch fills.
constexpr int kTileT = 32;

// Timesteps one backward block walks at the narrow widths, and the base
// bwd_tile_t scales from. Smaller than the forward tile because the backward folds
// batch into the block's serial loop, so the time tile is the only axis that
// supplies it blocks: the grid holds D*ceil(T/tile) threads whatever the batch is,
// where the forward gets a factor of B on top. Against that, the tile is what
// amortizes the backward's overhang: a block walks tile + kWidth - 1 steps and
// loads both x and dy at each, so the loads per owned timestep are
// (kWidth-1 + 2*(tile+kWidth-1)) / tile, and the tile count is ceil(T/tile). The
// tile count is not the partial count: see kBwdTargetBlocks.
constexpr int kBwdTileT = 16;

// The tile the backward walks, which scales with the width. The overhang is
// kWidth-1 steps, so at a fixed tile its share of the walk,
// (kWidth-1)/(tile+kWidth-1), grows with the width: 30 percent at kWidth = 8
// against 16 at kWidth = 4, which is the wide instantiation paying more
// instructions for the same output. Scaling the tile with the width holds that
// share fixed.
//
// It is also what takes the wide grid off a second wave. The walk holds
// D*ceil(T/tile) threads, so at D = 768 and T = 2048 a 16-step tile is 3072 warps
// of work against 2016 resident at six blocks an SM: 1.52 waves, and the tail
// leaves a quarter of the machine idle for a whole second round. A 32-step tile is
// 1536 warps. Its longer strip costs a resident block -- measured occupancy limit
// five against six at the shorter tile, so 1680 warps resident and 0.91 waves --
// and one round of 39 steps beats two rounds of 23. Measured at the wide shape in
// bfloat16, 77.567 us against 80.447, and 150.589 against 170.973 on the scalar
// path, which carries no strip and keeps six blocks.
//
// The strip is what sets that block count, and it is a threshold and not a slope:
// the shared allocation is rounded up before it is divided into the SM's 102,400 B,
// so 19,968 B a block left four resident and trimming it to 19,072 left five. The
// trim is the dy stream's leading halo, which the fill never writes; see kDySpan.
//
// One step, not a ramp: the widths above 4 all carry an overhang the 16-step tile
// cannot amortize, and a second step would cost the narrow arms their block count
// for a share the wide arm already pays. Both values are a multiple of the vector
// the staged fill forms, so no tile needs the fill's tail predicate.
constexpr int bwd_tile_t(int width) {
  return width <= 4 ? kBwdTileT : 2 * kBwdTileT;
}

// Time tiles one backward block covers, which is blockDim.y. The block is the
// unit an SM's occupancy limits are quantized in: at 64 threads a block is two
// warps, and sm_86's cap of 16 blocks per SM then holds the SM to 32 of its 48
// warp slots, 66.7 percent, whatever the register count. Two tiles per block
// make the block four warps, so the warp slots bind instead of the block slots.
// The tiles in a block meet once, at the end, to combine their parameter-gradient
// accumulators, so the walk itself needs no synchronization.
constexpr int kBwdTilesPerBlock = 2;

// Blocks along time the backward aims for, and with it the size of the partial
// stack: the stack holds one slice per block along time, so what bounds the grid
// bounds the stack. A grid that follows the sequence gives a stack that follows
// it too, and a stack that follows the sequence is what put the reduction over
// its class ceiling at T = 8192.
//
// Above this count a block walks several tile groups in a grid-stride loop and
// still writes one slice. The division that sets the stride is a floor, so a
// block takes on another group only when a whole extra group is there for it and
// the grid never drops below the target; the slice count is then at most twice
// ceil(kBwdTargetBlocks / grid.x) whatever T is.
//
// The number is a residency and not a wave: at 8 blocks an SM it covers 96 SMs,
// and sm_86 holds 6 blocks an SM at W = 8 and 10 at W = 1, so the grid stays at
// or above one full wave on every part these kernels are built for.
constexpr int kBwdTargetBlocks = 768;

// Blocks per SM the backward is compiled to hold, which is a register ceiling:
// this many blocks of blockDim.x*kBwdTilesPerBlock/32 warps, against sm_86's
// 65536 registers per SM and its 256-register-per-warp allocation granularity.
// At 6 blocks of 4 warps the ceiling is 80 registers a thread.
//
// The bound is here because the tile groups are a loop rather than a grid axis:
// left to itself nvcc holds more of the walk across that loop than the occupancy
// is worth, 124 registers on the scalar path at kWidth = 8 for 4 blocks an SM
// where 80 gives 6, and it spills nothing at 80.
//
// 8 is available on the staged walk and buys nothing. That walk reads its window
// from shared memory rather than registers, so it reaches 64 registers, 8 blocks,
// and 32 of the 48 warp slots at every width with nothing spilled, where 6 blocks
// leave it 74 registers and 24 slots at kWidth = 8. Measured at the wide shape,
// which is the only benched shape at that width: achieved occupancy 40.16 to
// 49.17 percent, and the kernel 93.311 to 93.662 us against a run-to-run spread of
// 0.10 to 2.44 percent, so the duration did not move. The kernel is not occupancy
// limited and the tighter ceiling is a constraint on every future edit to the walk
// in exchange for nothing, so it is not taken. The scalar walk cannot take 8 in any
// case: it spills there, 20 B of stores and 20 B of loads at kWidth = 8.
constexpr int kBwdMinBlocksPerSm = 6;

// Timesteps one forward thread loads before it consumes any of them. The window
// carries a serial dependence across time but the loads do not, so this is the
// forward's bytes-in-flight-per-thread knob. It divides kTileT, so a full tile is
// a whole number of groups and only the ragged tail tile needs the predicate.
constexpr int kFwdPrefetch = 8;
static_assert(kTileT % kFwdPrefetch == 0, "the prefetch group must divide kTileT");

// Channels per block. A warp reads 32 consecutive channels at a fixed timestep,
// which is one coalesced transaction because the layout is channels-last. Two
// warps rather than four: the block is the unit the grid is quantized in, so a
// narrower block divides a channel count more often and leaves fewer idle lanes
// in the last block of a row, and it splits the tail wave finer.
constexpr int kMaxChannelsPerBlock = 64;

// Channels one partial-reduction block covers, and partial slices one of its
// threads walks. The product is the block, 512 threads, and the split is what
// supplies the grid its blocks: the grid is ceil(D/kReduceChannels) * (W+1), so
// at D = 576 and W = 4 a 32-channel block would leave 90 blocks, under twice the
// SM count of every part this runs on, while 16 leaves 180. A 16-lane run of
// float32 is 64 B, two whole sectors, so narrowing the block costs the load
// nothing.
constexpr int kReduceChannels = 16;
constexpr int kReduceRows = 32;

// Byte alignment the staged fill forms addresses at, and the only alignment these
// kernels need. The same number as slinoss._guard.ALIGN_BYTES; the host rule holds a
// strict band to slinoss._guard.SECTOR_BYTES instead, which is a bandwidth rule about
// the sector a row overhangs and not an address the kernel could fault on, so the
// binding's backstop checks this number and the host checks the stricter one.
constexpr int kAlignBytes = 16;

// y = act(conv(x)), with the activation in the kernel epilogue.
//
// A pitched operand is one column band of a wider tensor: unit stride on the
// trailing axis, a row pitch at or above D, and a base and pitch on a multiple of
// kAlignBytes. A contiguous tensor is the case pitch == D.
//
//   x             (B,T,D)   pitched, bf16/fp16/fp32
//   weight        (D,W)     contiguous, same dtype as x
//   bias          (D,)      contiguous, same dtype as x, or nullopt
//   initial_state (B,W-1,D) contiguous, same dtype as x, or nullopt
//   y             (B,T,D) or (B,D/P,T,P), contiguous, same dtype as x, written
//                           in full
//   final_state   (B,W-1,D) contiguous, same dtype as x, written in full,
//                           or nullopt
//
// y_rows is P, the elements of y that one (b,h,t) row holds: D for the
// token-major shape and P for the head-major one, where channel d = h*P + p
// lands at (b,h,t,p). The store is base plus t*y_rows either way, so the
// head-major output is a store address and never a repack.
void causal_conv1d_fwd(const at::Tensor &x, const at::Tensor &weight,
                       const std::optional<at::Tensor> &bias,
                       const std::optional<at::Tensor> &initial_state,
                       const at::Tensor &y,
                       const std::optional<at::Tensor> &final_state,
                       int64_t y_rows, bool activation);

// Pullback of causal_conv1d_fwd.
//
//   dy             (B,T,D) or (B,D/P,T,P), contiguous, same dtype as x, or
//                            nullopt for a cotangent that is identically zero
//   dfinal_state   (B,W-1,D) contiguous, same dtype as x, or nullopt
//   dx             (B,T,D)   pitched, same dtype as x, written in full
//   dinitial_state (B,W-1,D) contiguous, same dtype as x, written in full,
//                            or nullopt
//   dweight_parts  (S,W,D)   contiguous float32, written in full
//   dbias_parts    (S,D)     contiguous float32, written in full, or nullopt
//
// dy_rows is dy's own y_rows: the cotangent carries the layout the forward's y
// was written in, and dy is the only operand the layout reaches. dx is
// token-major because x is, and pitched for the same reason, at its own pitch:
// the two bands are cut from different buffers. The value is unread when dy is
// nullopt.
//
// S is causal_conv1d_bwd_parts(T, D), the backward's block count along time and
// not its tile count: a block reduces every tile group it walks into one slice.
// The parameter gradients are per-lag float32 accumulators reduced along time
// inside the block and stored, never accumulated, so no output needs zeroing
// before the launch. Both partial buffers are channel-minor, so a warp's store is
// one coalesced transaction; causal_conv1d_reduce_parts sums the S slices and
// transposes the tap block on the way out.
void causal_conv1d_bwd(const std::optional<at::Tensor> &dy,
                       const std::optional<at::Tensor> &dfinal_state,
                       const at::Tensor &x, const at::Tensor &weight,
                       const std::optional<at::Tensor> &bias,
                       const std::optional<at::Tensor> &initial_state,
                       const at::Tensor &dx,
                       const std::optional<at::Tensor> &dinitial_state,
                       const at::Tensor &dweight_parts,
                       const std::optional<at::Tensor> &dbias_parts,
                       int64_t dy_rows, bool activation);

// Number of partial slices the backward writes, which is its block count along
// time. Bounded by kBwdTargetBlocks and the channel count, so it stops growing
// with the sequence length; the channel count enters because the grid's channel
// axis is what is left of the target, and the width because it sets the tile.
int64_t causal_conv1d_bwd_parts(int64_t seqlen, int64_t channels, int64_t width);

// Reduce the backward's partials into the parameter gradients.
//
//   dweight_parts (S,W,D) contiguous float32
//   dbias_parts   (S,D)   contiguous float32, or nullopt
//   dweight       (D,W)   contiguous, bf16/fp16/fp32, written in full
//   dbias         (D,)    contiguous, same dtype as dweight, written in full,
//                         or nullopt
//
// One launch covers both gradients: the grid's second axis holds the W tap slices
// and, when a bias is present, the bias as one more slice. Each slice is a stack
// of S rows of D floats and differs only in its row stride, so the taps and the
// bias are one kernel and an absent bias is one fewer slice rather than a second
// launch that is skipped.
//
// The transpose from the partials' tap-major (W,D) to the weight's own (D,W) is
// the store index and the narrowing to the weight's dtype is the store, so
// neither is a pass over anything.
//
// The reduction order is fixed at compile time: thread r sums slice rows r,
// r + kReduceRows, ... ascending, and the combine across the block's rows runs
// ascending too. No atomics reach either output, so two runs on identical
// partials agree bitwise.
void causal_conv1d_reduce_parts(const at::Tensor &dweight_parts,
                                const std::optional<at::Tensor> &dbias_parts,
                                const at::Tensor &dweight,
                                const std::optional<at::Tensor> &dbias);

} // namespace slinoss
