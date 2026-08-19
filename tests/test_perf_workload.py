"""The benchmarked workloads: shapes, inputs, and the timed callables.

Every test runs on the CPU reference at a shape small enough to be cheap, because
what is under test is the workload definition and not the operator. The standard
sizes in :data:`slinoss.perf.workload.SHAPES` and
:data:`slinoss.perf.workload.CONV_SHAPES` are checked against the shape
constraints they have to satisfy, since a bench that runs at an illegal shape
reports a number for a configuration the operator does not support.
"""

from __future__ import annotations

import pytest
import torch

from slinoss import _C
from slinoss.perf.budget import assert_closed, budget
from slinoss.perf.timing import measure, measure_paired
from slinoss.perf.workload import (
    CONV_SHAPES,
    SHAPE_NAMES,
    SHAPES,
    W_MAX,
    ConvShape,
    OpShape,
    conv_forward_only,
    conv_shape_by_name,
    conv_step,
    forward_only,
    make_conv_inputs,
    make_inputs,
    shape_by_name,
    step,
)

CPU = torch.device("cpu")

SMALL = OpShape("small", bsz=1, heads=1, seq=8, rows=16, lanes=16, chunk=4)
"""Two whole chunks at the smallest legal row and lane counts."""

SMALL_CONV = ConvShape("small", bsz=1, seq=8, channels=4, width=4)
"""One tap bank wider than one token, so the streaming carry is not degenerate."""


# ---------------------------------------------------------------------------
# OpShape and SHAPES
# ---------------------------------------------------------------------------


def test_op_shape_derives_the_state_width_and_the_token_count() -> None:
    assert SMALL.d_state == 48
    assert SMALL.token_count == 8
    assert SMALL.describe() == "small: B=1 H=1 T=8 P=16 N=16 3N=48 L=4"


def test_standard_shapes_satisfy_the_operator_constraints() -> None:
    # N a multiple of 16 makes 3N a multiple of 48 and every contraction MMA-k
    # friendly with no padding path. A bench at an unsupported shape is a number
    # for a configuration that cannot ship.
    for shape in SHAPES:
        assert shape.lanes % 16 == 0
        assert shape.rows % 8 == 0
        assert shape.d_state % 48 == 0
        assert shape.chunk > 0
        assert shape.seq > 0
    names = [s.name for s in SHAPES]
    assert len(set(names)) == len(names)
    assert shape_by_name("tiny").name == "tiny"
    ragged = shape_by_name("ragged")
    # A sequence length that is not a multiple of the chunk, so a tail-handling
    # regression shows up in the bench and not only in the tests.
    assert ragged.seq % ragged.chunk != 0
    assert all(s.seq % s.chunk == 0 for s in SHAPES if s.name != "ragged")
    with pytest.raises(KeyError, match="no shape 'huge'"):
        shape_by_name("huge")


# ---------------------------------------------------------------------------
# make_inputs
# ---------------------------------------------------------------------------


def test_make_inputs_matches_the_tensor_contract() -> None:
    got = make_inputs(SMALL, CPU, dtype=torch.float32)
    lead = (SMALL.bsz, SMALL.heads, SMALL.seq)
    assert tuple(got.U.shape) == (*lead, SMALL.rows)
    assert tuple(got.trans.shape) == (*lead, 4)
    assert tuple(got.K.shape) == (*lead, 2, 4)
    assert tuple(got.B.shape) == (*lead, SMALL.d_state)
    assert tuple(got.C.shape) == (*lead, SMALL.d_state)
    assert tuple(got.dy.shape) == (*lead, SMALL.rows)
    for t in got:
        assert t.is_contiguous()
    # I4: trans and K are float32 whatever U, B, C, and Y are.
    low = make_inputs(SMALL, CPU, dtype=torch.bfloat16)
    assert low.U.dtype == torch.bfloat16
    assert low.B.dtype == torch.bfloat16
    assert low.C.dtype == torch.bfloat16
    assert low.dy.dtype == torch.bfloat16
    assert low.trans.dtype == torch.float32
    assert low.K.dtype == torch.float32


def test_make_inputs_holds_the_numerical_invariants() -> None:
    # I1 and I2 on the benchmarked tensors, not only on the trained ones. Built by
    # the real parameter maps rather than from randn, which would put ls > 0 into a
    # decay prefix and measure a kernel that cannot run in training.
    got = make_inputs(SMALL, CPU)
    assert torch.all(got.trans[..., 3] <= 0.0)
    assert torch.all(torch.linalg.vector_norm(got.trans[..., :3], dim=-1) <= W_MAX)
    # I2 is a bound below pi, not merely a bound: at pi the quaternion polynomial
    # leaves the domain its minimax fit covers.
    assert torch.pi > W_MAX
    # Lane 3 of each tap is a hard zero, present for float4 alignment.
    assert torch.all(got.K[..., 3] == 0.0)


def test_make_inputs_carries_gradients_on_the_five_differentiable_inputs() -> None:
    got = make_inputs(SMALL, CPU, requires_grad=True)
    assert got.differentiable == (got.U, got.trans, got.K, got.B, got.C)
    assert all(t.requires_grad for t in got.differentiable)
    # The output-gradient seed is preallocated and is not a graph input, so the
    # backward measurement contains no allocation of its own.
    assert not got.dy.requires_grad
    plain = make_inputs(SMALL, CPU, requires_grad=False)
    assert not any(t.requires_grad for t in plain.differentiable)


def test_make_inputs_is_reproducible_from_the_seed() -> None:
    # Two runs of a bench must compare the same numbers, or the delta includes the
    # inputs.
    first = make_inputs(SMALL, CPU, seed=7)
    same = make_inputs(SMALL, CPU, seed=7)
    other = make_inputs(SMALL, CPU, seed=8)
    for a, b in zip(first, same):
        assert torch.equal(a, b)
    assert not torch.equal(first.U, other.U)


# ---------------------------------------------------------------------------
# forward_only and step
# ---------------------------------------------------------------------------


def test_forward_only_records_one_region_and_builds_no_graph() -> None:
    inputs = make_inputs(SMALL, CPU, dtype=torch.float32, requires_grad=True)
    timed = measure(
        forward_only(inputs, SMALL.chunk),
        label="op",
        iters=2,
        warmup=1,
        device=CPU,
    )
    assert [t.label for t in timed.regions] == ["op.forward"]
    assert timed.region("op.forward").spread.sample_count == 2
    # Under no_grad the inputs still require grad and the output does not, so a
    # forward-only bench cannot be paying for a graph it never uses.
    assert all(t.requires_grad for t in inputs.differentiable)
    # A prefix renames the region, so one loop can hold two arms.
    prefixed = measure(
        forward_only(inputs, SMALL.chunk, prefix="arm-b"),
        label="op",
        iters=1,
        warmup=0,
        device=CPU,
    )
    assert [t.label for t in prefixed.regions] == ["arm-b.forward"]


def test_step_records_the_forward_and_the_backward() -> None:
    inputs = make_inputs(SMALL, CPU, dtype=torch.float32, requires_grad=True)
    timed = measure(
        step(inputs, SMALL.chunk), label="op", iters=2, warmup=1, device=CPU
    )
    assert [t.label for t in timed.regions] == ["op.forward", "op.backward"]
    assert timed.region("op.backward").spread.sample_count == 2
    # torch.autograd.grad, so nothing accumulates into a .grad buffer and no
    # aten::fill_ can contaminate the backward bucket.
    assert all(t.grad is None for t in inputs.differentiable)
    # A named subset is differentiated the same way, through the same regions.
    subset = measure(
        step(inputs, SMALL.chunk, wrt=(inputs.U,)),
        label="op",
        iters=1,
        warmup=0,
        device=CPU,
    )
    assert [t.label for t in subset.regions] == ["op.forward", "op.backward"]
    assert subset.region("op.backward").spread.sample_count == 1


def test_step_rejects_inputs_that_take_no_gradient() -> None:
    # Otherwise it would time a forward under grad mode and report it as a step.
    with pytest.raises(ValueError, match="at least one input requiring grad"):
        step(make_inputs(SMALL, CPU, requires_grad=False), SMALL.chunk)
    # A wrt naming only the output-gradient seed is the same defect.
    grads = make_inputs(SMALL, CPU, requires_grad=True)
    with pytest.raises(ValueError, match="at least one input requiring grad"):
        step(grads, SMALL.chunk, wrt=(grads.dy,))


def test_two_prefixes_keep_two_arms_apart_in_one_loop() -> None:
    inputs = make_inputs(SMALL, CPU, dtype=torch.float32, requires_grad=True)
    out = measure_paired(
        "arm-a",
        step(inputs, SMALL.chunk, prefix="arm-a"),
        "arm-b",
        step(inputs, SMALL.chunk, prefix="arm-b"),
        label="paired",
        iters=2,
        warmup=0,
        device=CPU,
    )
    # The default prefix in both arms would sum the two forwards into one region
    # and one backward into another, and the tree would describe neither arm.
    assert {t.label for t in out.timed.regions} == {
        "arm-a",
        "arm-a.forward",
        "arm-a.backward",
        "arm-b",
        "arm-b.forward",
        "arm-b.backward",
    }
    assert out.timed.region("arm-a.forward").parent == "arm-a"
    assert out.comparison.sample_count == 2
    # Each arm's children sit under it, so the budget closes over both arms.
    assert_closed(budget(out.timed))


# ---------------------------------------------------------------------------
# ConvShape and CONV_SHAPES
# ---------------------------------------------------------------------------


def test_conv_shapes_carry_the_scan_shape_names_and_the_kernel_bounds() -> None:
    # One name table for both operators: a driver offers one --shape list whatever
    # --op it was handed, so a name that resolves under one operator and not the
    # other is an argparse choice that fails after the inputs are allocated.
    assert tuple(s.name for s in CONV_SHAPES) == SHAPE_NAMES
    for shape in CONV_SHAPES:
        assert shape.bsz > 0
        assert shape.seq > 0
        assert shape.channels > 0
        assert shape.state_shape == (shape.bsz, shape.width - 1, shape.channels)
    assert SMALL_CONV.token_count == 8
    assert SMALL_CONV.describe() == "small: B=1 T=8 D=4 W=4"
    assert conv_shape_by_name("tiny").name == "tiny"
    with pytest.raises(KeyError, match="no conv shape 'huge'"):
        conv_shape_by_name("huge")
    ragged = conv_shape_by_name("ragged")
    # The tap bound and the time tile belong to the kernel, so they are read off
    # the extension where one is built rather than restated here.
    if _C.is_available():
        module = _C.extension()
        assert all(1 <= s.width <= int(module.MAX_WIDTH) for s in CONV_SHAPES)
        assert max(s.width for s in CONV_SHAPES) == int(module.MAX_WIDTH)
        # A sequence length that is not a whole number of time tiles, so a
        # tail-handling regression shows up in the bench and not only in the tests.
        # Both tiles, because the two directions tile time differently: a length
        # ragged against one and exact against the other leaves that direction's
        # tail unmeasured, which is how the property silently lapses when a tile is
        # retuned.
        for tile in (int(module.TILE_T), int(module.BWD_TILE_T)):
            assert ragged.seq % tile != 0
            assert all(s.seq % tile == 0 for s in CONV_SHAPES if s.name != "ragged")


# ---------------------------------------------------------------------------
# make_conv_inputs
# ---------------------------------------------------------------------------


def test_make_conv_inputs_matches_the_tensor_contract() -> None:
    got = make_conv_inputs(SMALL_CONV, CPU, dtype=torch.float32)
    lead = (SMALL_CONV.bsz, SMALL_CONV.seq, SMALL_CONV.channels)
    assert tuple(got.x.shape) == lead
    assert tuple(got.weight.shape) == (SMALL_CONV.channels, SMALL_CONV.width)
    assert tuple(got.bias.shape) == (SMALL_CONV.channels,)
    assert tuple(got.initial_state.shape) == SMALL_CONV.state_shape
    assert tuple(got.dy.shape) == lead
    for t in got:
        assert t.is_contiguous()
    # One dtype throughout: the native backend is one template instantiation per
    # dtype and refuses a mixed-dtype call rather than promoting an operand.
    low = make_conv_inputs(SMALL_CONV, CPU, dtype=torch.bfloat16)
    assert {t.dtype for t in low} == {torch.bfloat16}


def test_make_conv_inputs_carries_gradients_on_the_four_differentiable_inputs() -> None:
    got = make_conv_inputs(SMALL_CONV, CPU, requires_grad=True)
    assert got.differentiable == (got.x, got.weight, got.bias, got.initial_state)
    assert all(t.requires_grad for t in got.differentiable)
    # The output-gradient seed is preallocated and is not a graph input, so the
    # backward measurement contains no allocation of its own.
    assert not got.dy.requires_grad
    plain = make_conv_inputs(SMALL_CONV, CPU, requires_grad=False)
    assert not any(t.requires_grad for t in plain.differentiable)


def test_make_conv_inputs_is_reproducible_from_the_seed() -> None:
    # Two runs of a bench must compare the same numbers, or the delta includes the
    # inputs.
    first = make_conv_inputs(SMALL_CONV, CPU, seed=7)
    same = make_conv_inputs(SMALL_CONV, CPU, seed=7)
    other = make_conv_inputs(SMALL_CONV, CPU, seed=8)
    for a, b in zip(first, same):
        assert torch.equal(a, b)
    assert not torch.equal(first.x, other.x)


# ---------------------------------------------------------------------------
# conv_forward_only and conv_step
# ---------------------------------------------------------------------------


def test_the_conv_runners_record_their_own_regions() -> None:
    inputs = make_conv_inputs(SMALL_CONV, CPU, dtype=torch.float32, requires_grad=True)
    forward = measure(
        conv_forward_only(inputs), label="conv", iters=2, warmup=1, device=CPU
    )
    assert [t.label for t in forward.regions] == ["conv.forward"]
    # Under no_grad the inputs still require grad and the output does not, so a
    # forward-only bench cannot be paying for a graph it never uses.
    assert all(t.requires_grad for t in inputs.differentiable)
    timed = measure(conv_step(inputs), label="conv", iters=2, warmup=1, device=CPU)
    assert [t.label for t in timed.regions] == ["conv.forward", "conv.backward"]
    assert timed.region("conv.backward").spread.sample_count == 2
    # torch.autograd.grad, so nothing accumulates into a .grad buffer and no
    # aten::fill_ can contaminate the backward bucket.
    assert all(t.grad is None for t in inputs.differentiable)
    # A prefix renames the regions, so one loop can hold two arms.
    prefixed = measure(
        conv_step(inputs, prefix="arm-b"), label="conv", iters=1, warmup=0, device=CPU
    )
    assert [t.label for t in prefixed.regions] == ["arm-b.forward", "arm-b.backward"]


def test_conv_step_rejects_inputs_that_take_no_gradient() -> None:
    # Otherwise it would time a forward under grad mode and report it as a step.
    with pytest.raises(ValueError, match="at least one input requiring grad"):
        conv_step(make_conv_inputs(SMALL_CONV, CPU, requires_grad=False))
    # A wrt naming only the output-gradient seed is the same defect.
    grads = make_conv_inputs(SMALL_CONV, CPU, requires_grad=True)
    with pytest.raises(ValueError, match="at least one input requiring grad"):
        conv_step(grads, wrt=(grads.dy,))
