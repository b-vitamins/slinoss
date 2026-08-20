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

// Timesteps one backward block walks. Smaller than the forward tile because the
// backward folds batch into the block's serial loop, so the time tile is the
// only axis that supplies it blocks: the grid holds D*ceil(T/kBwdTileT) threads
// whatever the batch is, where the forward gets a factor of B on top. Against
// that, the tile is what amortizes the backward's overhang: a block walks
// kBwdTileT + kWidth - 1 steps and loads both x and dy at each, so the loads per
// owned timestep are (kWidth-1 + 2*(kBwdTileT+kWidth-1)) / kBwdTileT, and the
// partial count is ceil(T/kBwdTileT), which is also what the host sums.
constexpr int kBwdTileT = 16;

// Time tiles one backward block covers, which is blockDim.y. The block is the
// unit an SM's occupancy limits are quantized in: at 64 threads a block is two
// warps, and sm_86's cap of 16 blocks per SM then holds the SM to 32 of its 48
// warp slots, 66.7 percent, whatever the register count. Two tiles per block
// make the block four warps, so the warp slots bind instead of the block slots.
// The tiles in a block share no state, so this costs no synchronization.
constexpr int kBwdTilesPerBlock = 2;

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
// S is causal_conv1d_bwd_parts(T). The parameter gradients are per-lag float32
// accumulators reduced along time inside the block and stored, never
// accumulated, so no output needs zeroing before the launch. Both partial
// buffers are channel-minor, so a warp's store is one coalesced transaction;
// causal_conv1d_reduce_parts sums the S slices and transposes the tap block on
// the way out.
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

// Number of time-tile partials the backward writes for a sequence length.
int64_t causal_conv1d_bwd_parts(int64_t seqlen);

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
