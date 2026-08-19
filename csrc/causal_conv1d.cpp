// Python bindings for the causal depthwise conv1d kernels.
//
// The checks here are a backstop, not the contract. Shape, dtype, layout, and
// width validation lives in the Python host code so that an invalid call raises
// the same error whichever backend is selected; a failure that reaches this file
// is a bug in that host code.

#include "causal_conv1d.h"

#include <torch/extension.h>

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
};

Dims check_common(const at::Tensor &x, const at::Tensor &weight,
                  const std::optional<at::Tensor> &bias,
                  const std::optional<at::Tensor> &initial_state) {
  TORCH_CHECK(x.is_cuda(), "x must be on a CUDA device");
  TORCH_CHECK(x.dim() == 3, "x must be (B,T,D)");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  const Dims d{x.size(0), x.size(1), x.size(2), weight.size(1)};
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == d.channels,
              "weight must be (D,W)");
  TORCH_CHECK(d.width >= 1 && d.width <= slinoss::kMaxWidth,
              "width must lie in [1, ", slinoss::kMaxWidth, "], got ", d.width);
  const at::ScalarType dtype = x.scalar_type();
  const at::Device device = x.device();
  expect(weight, "weight", {d.channels, d.width}, dtype, device);
  expect_optional(bias, "bias", {d.channels}, dtype, device);
  expect_optional(initial_state, "initial_state",
                  {d.batch, d.width - 1, d.channels}, dtype, device);
  return d;
}

void fwd(const at::Tensor &x, const at::Tensor &weight,
         const std::optional<at::Tensor> &bias,
         const std::optional<at::Tensor> &initial_state, const at::Tensor &y,
         const std::optional<at::Tensor> &final_state, bool activation) {
  const Dims d = check_common(x, weight, bias, initial_state);
  expect(y, "y", {d.batch, d.seqlen, d.channels}, x.scalar_type(), x.device());
  expect_optional(final_state, "final_state",
                  {d.batch, d.width - 1, d.channels}, x.scalar_type(),
                  x.device());
  slinoss::causal_conv1d_fwd(x, weight, bias, initial_state, y, final_state,
                             activation);
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
  const at::ScalarType dtype = x.scalar_type();
  const at::Device device = x.device();
  const std::vector<int64_t> window{d.batch, d.width - 1, d.channels};
  expect_optional(dy, "dy", {d.batch, d.seqlen, d.channels}, dtype, device);
  expect(dx, "dx", {d.batch, d.seqlen, d.channels}, dtype, device);
  expect_optional(dfinal_state, "dfinal_state", window, dtype, device);
  expect_optional(dinitial_state, "dinitial_state", window, dtype, device);
  const int64_t parts = slinoss::causal_conv1d_bwd_parts(d.seqlen);
  expect(dweight_parts, "dweight_parts", {parts, d.channels, d.width},
         at::kFloat, device);
  expect_optional(dbias_parts, "dbias_parts", {parts, d.channels}, at::kFloat,
                  device);
  slinoss::causal_conv1d_bwd(dy, dfinal_state, x, weight, bias, initial_state,
                             dx, dinitial_state, dweight_parts, dbias_parts,
                             activation);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Causal depthwise conv1d CUDA kernels.";
  m.attr("MAX_WIDTH") = py::int_(slinoss::kMaxWidth);
  m.attr("TILE_T") = py::int_(slinoss::kTileT);
  m.attr("BWD_TILE_T") = py::int_(slinoss::kBwdTileT);
  m.def("fwd", &fwd, py::arg("x"), py::arg("weight"), py::arg("bias"),
        py::arg("initial_state"), py::arg("y"), py::arg("final_state"),
        py::arg("activation"),
        "Write y = act(conv(x)) and, when asked, the trailing window.");
  m.def("bwd", &bwd, py::arg("dy"), py::arg("dfinal_state"), py::arg("x"),
        py::arg("weight"), py::arg("bias"), py::arg("initial_state"),
        py::arg("dx"), py::arg("dinitial_state"), py::arg("dweight_parts"),
        py::arg("dbias_parts"), py::arg("activation"),
        "Write dx, the incoming-state gradient, and the per-time-tile "
        "parameter-gradient partials.");
  m.def("bwd_parts", &slinoss::causal_conv1d_bwd_parts, py::arg("seqlen"),
        "Number of partial slices the backward writes for a sequence length.");
}
