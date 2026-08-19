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
// whatever the batch is, where the forward gets a factor of B on top. The
// backward pays a wider re-read for it, W-1 timesteps of x and dy per block
// instead of W-1 of x. Halving it again is not free: the partial count is
// ceil(T/kBwdTileT), so it doubles the partials the host has to sum.
constexpr int kBwdTileT = 8;

// Batch entries one backward thread walks at once. The walk over time is serial
// and each step depends on the one before it, so this is the axis that supplies
// the thread independent loads to overlap: bytes in flight per thread scale with
// it, and the window registers do too.
constexpr int kBwdStreams = 2;

// Channels per block. A warp reads 32 consecutive channels at a fixed timestep,
// which is one coalesced transaction because the layout is channels-last. Two
// warps rather than four: the block is the unit the grid is quantized in, so a
// narrower block divides a channel count more often and leaves fewer idle lanes
// in the last block of a row, and it splits the tail wave finer.
constexpr int kMaxChannelsPerBlock = 64;

// y = act(conv(x)), with the activation in the kernel epilogue.
//
//   x             (B,T,D)   contiguous, bf16/fp16/fp32
//   weight        (D,W)     contiguous, same dtype as x
//   bias          (D,)      contiguous, same dtype as x, or nullopt
//   initial_state (B,W-1,D) contiguous, same dtype as x, or nullopt
//   y             (B,T,D)   contiguous, same dtype as x, written in full
//   final_state   (B,W-1,D) contiguous, same dtype as x, written in full,
//                           or nullopt
void causal_conv1d_fwd(const at::Tensor &x, const at::Tensor &weight,
                       const std::optional<at::Tensor> &bias,
                       const std::optional<at::Tensor> &initial_state,
                       const at::Tensor &y,
                       const std::optional<at::Tensor> &final_state,
                       bool activation);

// Pullback of causal_conv1d_fwd.
//
//   dy             (B,T,D)   contiguous, same dtype as x, or nullopt for a
//                            cotangent that is identically zero
//   dfinal_state   (B,W-1,D) contiguous, same dtype as x, or nullopt
//   dx             (B,T,D)   contiguous, same dtype as x, written in full
//   dinitial_state (B,W-1,D) contiguous, same dtype as x, written in full,
//                            or nullopt
//   dweight_parts  (P,D,W)   contiguous float32, written in full
//   dbias_parts    (P,D)     contiguous float32, written in full, or nullopt
//
// P is causal_conv1d_bwd_parts(T). The parameter gradients are per-lag float32
// accumulators reduced along time inside the block and stored, never
// accumulated, so no output needs zeroing before the launch. The caller sums
// the P slices.
void causal_conv1d_bwd(const std::optional<at::Tensor> &dy,
                       const std::optional<at::Tensor> &dfinal_state,
                       const at::Tensor &x, const at::Tensor &weight,
                       const std::optional<at::Tensor> &bias,
                       const std::optional<at::Tensor> &initial_state,
                       const at::Tensor &dx,
                       const std::optional<at::Tensor> &dinitial_state,
                       const at::Tensor &dweight_parts,
                       const std::optional<at::Tensor> &dbias_parts,
                       bool activation);

// Number of time-tile partials the backward writes for a sequence length.
int64_t causal_conv1d_bwd_parts(int64_t seqlen);

} // namespace slinoss
