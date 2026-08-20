// Python bindings for the causal depthwise conv1d kernels.
//
// The checks here are a backstop, not the contract. Shape, dtype, layout, and
// width validation lives in the Python host code so that an invalid call raises
// the same error whichever backend is selected; a failure that reaches this file
// is a bug in that host code.

#include "causal_conv1d.h"

#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace {

void expect(const at::Tensor &t, const std::string &name,
            const std::vector<int64_t> &shape, at::ScalarType dtype,
            const at::Device &device) {
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.device() == device, name, " must be on ", device, ", got ",
              t.device());
  TORCH_CHECK(t.scalar_type() == dtype, name, " must be ", dtype, ", got ",
              t.scalar_type());
  TORCH_CHECK(t.sizes() == at::IntArrayRef(shape), name, " has the wrong shape");
}

// Same as expect but for a pitched operand: one column band of a wider tensor.
// slinoss._guard.check_pitched is where the contract is stated and where a caller's
// violation is reported; a failure here is a bug in that host code. The conditions
// are the subset the kernels themselves need, so a strict band's sector rule is not
// among them: it is about the sector an overhanging row fetches and discards, which
// no kernel can detect.
//
// The alignment clause is skipped for a contiguous operand. It is the producer's
// obligation on a band it cuts out of a projection, and D is an operator shape
// that no producer chose, so holding a contiguous D to it would refuse channel
// counts this operator has always accepted.
void expect_pitched(const at::Tensor &t, const std::string &name,
                    const std::vector<int64_t> &shape, at::ScalarType dtype,
                    const at::Device &device) {
  TORCH_CHECK(t.device() == device, name, " must be on ", device, ", got ",
              t.device());
  TORCH_CHECK(t.scalar_type() == dtype, name, " must be ", dtype, ", got ",
              t.scalar_type());
  TORCH_CHECK(t.sizes() == at::IntArrayRef(shape), name, " has the wrong shape");
  TORCH_CHECK(t.stride(-1) == 1, name,
              " must have unit stride on its trailing axis, got ", t.stride(-1));
  TORCH_CHECK(t.stride(-2) >= t.size(-1), name, " rows overlap: pitch ",
              t.stride(-2), " is below the row width ", t.size(-1));
  if (t.is_contiguous()) {
    return;
  }
  const int64_t multiple = slinoss::kAlignBytes / t.element_size();
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.const_data_ptr()) %
                          slinoss::kAlignBytes ==
                      0 &&
                  t.stride(-2) % multiple == 0,
              name, " must start and step on a multiple of ", multiple,
              " elements; got pitch ", t.stride(-2));
}

void expect_optional(const std::optional<at::Tensor> &t,
                     const std::string &name,
                     const std::vector<int64_t> &shape, at::ScalarType dtype,
                     const at::Device &device) {
  if (t.has_value()) {
    expect(*t, name, shape, dtype, device);
  }
}

struct Dims {
  int64_t batch;
  int64_t seqlen;
  int64_t channels;
  int64_t width;
  at::ScalarType dtype;
  at::Device device;
};

Dims check_common(const at::Tensor &x, const at::Tensor &weight,
                  const std::optional<at::Tensor> &bias,
                  const std::optional<at::Tensor> &initial_state) {
  TORCH_CHECK(x.is_cuda(), "x must be on a CUDA device");
  TORCH_CHECK(x.dim() == 3, "x must be (B,T,D)");
  const Dims d{x.size(0),      x.size(1),       x.size(2),
               weight.size(1), x.scalar_type(), x.device()};
  expect_pitched(x, "x", {d.batch, d.seqlen, d.channels}, d.dtype, d.device);
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == d.channels,
              "weight must be (D,W)");
  TORCH_CHECK(d.width >= 1 && d.width <= slinoss::kMaxWidth,
              "width must lie in [1, ", slinoss::kMaxWidth, "], got ", d.width);
  expect(weight, "weight", {d.channels, d.width}, d.dtype, d.device);
  expect_optional(bias, "bias", {d.channels}, d.dtype, d.device);
  expect_optional(initial_state, "initial_state",
                  {d.batch, d.width - 1, d.channels}, d.dtype, d.device);
  return d;
}

// Rows one (b,h,t) of an output holds, read off its rank rather than passed
// alongside it: (B,T,D) is token-major and gives D, (B,D/P,T,P) is head-major and
// gives P. The rank is the whole layout, so a separate flag would be a second
// source of truth for something the shape already states.
int64_t output_rows(const at::Tensor &t, const std::string &name,
                    const Dims &d) {
  TORCH_CHECK(t.dim() == 3 || t.dim() == 4, name,
              " must be (B,T,D) or (B,D/P,T,P)");
  if (t.dim() == 3) {
    expect(t, name, {d.batch, d.seqlen, d.channels}, d.dtype, d.device);
    return d.channels;
  }
  const int64_t rows = t.size(3);
  TORCH_CHECK(rows >= 1 && d.channels % rows == 0, name,
              " has trailing extent ", rows, ", which does not divide D=",
              d.channels);
  expect(t, name, {d.batch, d.channels / rows, d.seqlen, rows}, d.dtype,
         d.device);
  return rows;
}

void fwd(const at::Tensor &x, const at::Tensor &weight,
         const std::optional<at::Tensor> &bias,
         const std::optional<at::Tensor> &initial_state, const at::Tensor &y,
         const std::optional<at::Tensor> &final_state, bool activation) {
  const Dims d = check_common(x, weight, bias, initial_state);
  const int64_t y_rows = output_rows(y, "y", d);
  expect_optional(final_state, "final_state",
                  {d.batch, d.width - 1, d.channels}, d.dtype, d.device);
  slinoss::causal_conv1d_fwd(x, weight, bias, initial_state, y, final_state,
                             y_rows, activation);
}

void bwd(const std::optional<at::Tensor> &dy,
         const std::optional<at::Tensor> &dfinal_state,
         const at::Tensor &x, const at::Tensor &weight,
         const std::optional<at::Tensor> &bias,
         const std::optional<at::Tensor> &initial_state, const at::Tensor &dx,
         const std::optional<at::Tensor> &dinitial_state,
         const at::Tensor &dweight_parts,
         const std::optional<at::Tensor> &dbias_parts, bool activation) {
  const Dims d = check_common(x, weight, bias, initial_state);
  const at::ScalarType dtype = d.dtype;
  const at::Device device = d.device;
  const std::vector<int64_t> window{d.batch, d.width - 1, d.channels};
  // Absent dy leaves the layout unread, so the token-major row count stands in.
  const int64_t dy_rows =
      dy.has_value() ? output_rows(*dy, "dy", d) : d.channels;
  expect_pitched(dx, "dx", {d.batch, d.seqlen, d.channels}, dtype, device);
  expect_optional(dfinal_state, "dfinal_state", window, dtype, device);
  expect_optional(dinitial_state, "dinitial_state", window, dtype, device);
  const int64_t parts =
      slinoss::causal_conv1d_bwd_parts(d.seqlen, d.channels, d.width);
  expect(dweight_parts, "dweight_parts", {parts, d.width, d.channels},
         at::kFloat, device);
  expect_optional(dbias_parts, "dbias_parts", {parts, d.channels}, at::kFloat,
                  device);
  slinoss::causal_conv1d_bwd(dy, dfinal_state, x, weight, bias, initial_state,
                             dx, dinitial_state, dweight_parts, dbias_parts,
                             dy_rows, activation);
}

void bwd_reduce(const at::Tensor &dweight_parts,
                const std::optional<at::Tensor> &dbias_parts,
                const at::Tensor &dweight,
                const std::optional<at::Tensor> &dbias) {
  TORCH_CHECK(dweight_parts.is_cuda(), "dweight_parts must be on a CUDA device");
  TORCH_CHECK(dweight_parts.dim() == 3, "dweight_parts must be (S,W,D)");
  const int64_t parts = dweight_parts.size(0);
  const int64_t width = dweight_parts.size(1);
  const int64_t channels = dweight_parts.size(2);
  const at::Device device = dweight_parts.device();
  const at::ScalarType dtype = dweight.scalar_type();
  expect(dweight_parts, "dweight_parts", {parts, width, channels}, at::kFloat,
         device);
  expect(dweight, "dweight", {channels, width}, dtype, device);
  expect_optional(dbias_parts, "dbias_parts", {parts, channels}, at::kFloat,
                  device);
  expect_optional(dbias, "dbias", {channels}, dtype, device);
  // Paired, because the bias slice of the grid is what writes dbias: a dbias
  // without its partials would be left unwritten rather than reduced.
  TORCH_CHECK(dbias.has_value() == dbias_parts.has_value(),
              "dbias and dbias_parts must be present together");
  slinoss::causal_conv1d_reduce_parts(dweight_parts, dbias_parts, dweight, dbias);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Causal depthwise conv1d CUDA kernels.";
  m.attr("MAX_WIDTH") = py::int_(slinoss::kMaxWidth);
  m.attr("TILE_T") = py::int_(slinoss::kTileT);
  m.attr("BWD_TILE_T") = py::int_(slinoss::kBwdTileT);
  m.attr("BWD_TILES_PER_BLOCK") = py::int_(slinoss::kBwdTilesPerBlock);
  m.attr("BWD_TARGET_BLOCKS") = py::int_(slinoss::kBwdTargetBlocks);
  m.attr("CHANNELS_PER_BLOCK") = py::int_(slinoss::kMaxChannelsPerBlock);
  m.def("fwd", &fwd, py::arg("x"), py::arg("weight"), py::arg("bias"),
        py::arg("initial_state"), py::arg("y"), py::arg("final_state"),
        py::arg("activation"),
        "Write y = act(conv(x)) and, when asked, the trailing window.");
  m.def("bwd", &bwd, py::arg("dy"), py::arg("dfinal_state"), py::arg("x"),
        py::arg("weight"), py::arg("bias"), py::arg("initial_state"),
        py::arg("dx"), py::arg("dinitial_state"), py::arg("dweight_parts"),
        py::arg("dbias_parts"), py::arg("activation"),
        "Write dx, the incoming-state gradient, and the parameter-gradient "
        "partials, one slice per block along time.");
  m.def("bwd_tile_t", &slinoss::bwd_tile_t, py::arg("width"),
        "Timesteps one backward tile covers at a filter width. Scales with the "
        "width so the overhang keeps a fixed share of the walk; BWD_TILE_T is "
        "its value at the narrow widths.");
  m.def("bwd_parts", &slinoss::causal_conv1d_bwd_parts, py::arg("seqlen"),
        py::arg("channels"), py::arg("width"),
        "Number of partial slices the backward writes for a sequence length, "
        "channel count, and filter width. Bounded independently of the sequence "
        "length; the width enters because it sets the time tile.");
  m.def("bwd_reduce", &bwd_reduce, py::arg("dweight_parts"),
        py::arg("dbias_parts"), py::arg("dweight"), py::arg("dbias"),
        "Reduce the parameter-gradient partials into dweight, in the weight's "
        "layout and dtype, and dbias, in one launch.");
}
