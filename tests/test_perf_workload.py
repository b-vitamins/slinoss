"""The benchmarked workloads: shapes, inputs, and the timed callables.

Every test runs on the CPU reference at a shape small enough to be cheap, because
what is under test is the workload definition and not the operator. The four
standard size tables are checked against the shape constraints they have to
satisfy, since a bench that runs at an illegal shape reports a number for a
configuration the operator does not support.
"""

from __future__ import annotations

import pytest
import torch

from slinoss import _C
from slinoss.perf.budget import assert_closed, budget
from slinoss.perf.timing import measure, measure_paired
from slinoss.perf.workload import (
    BLOCK_SHAPES,
    CONV_SHAPES,
    PREP_SHAPES,
    PROJ_ALIGN,
    SHAPE_NAMES,
    SHAPES,
    W_MAX,
    BlockShape,
    ConvShape,
    OpShape,
    PrepShape,
    block_forward_only,
    block_shape_by_name,
    block_step,
    conv_forward_only,
    conv_shape_by_name,
    conv_step,
    forward_only,
    layer_config,
    make_block_inputs,
    make_conv_inputs,
    make_inputs,
    make_prep_inputs,
    prep_forward_only,
    prep_shape_by_name,
    prep_step,
    shape_by_name,
    step,
)

CPU = torch.device("cpu")

SMALL = OpShape("small", bsz=1, heads=1, seq=8, rows=16, lanes=16, chunk=4)
"""Two whole chunks at the smallest legal row and lane counts."""

SMALL_CONV = ConvShape("small", bsz=1, seq=8, channels=4, width=4)
"""One tap bank wider than one token, so the streaming carry is not degenerate."""

SMALL_LAYER = OpShape("small", bsz=1, heads=1, seq=8, rows=16, lanes=16, chunk=16)
""":data:`SMALL` at the shortest legal layer chunk.

:func:`slinoss.perf.workload.layer_config` builds a real
:class:`slinoss.config.SLinOSSConfig`, which bounds the chunk below at 16;
:data:`SMALL` is a chunk of 4, legal for the scan callables and not for a layer.
"""

SMALL_PREP = PrepShape(SMALL_LAYER, groups=1)
"""``H = 1``, so ``10*H`` is 10 and the projection width takes the padding path."""

SMALL_BLOCK = BlockShape(SMALL_LAYER)


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


# ---------------------------------------------------------------------------
# layer_config
# ---------------------------------------------------------------------------


def test_layer_config_derives_the_layer_the_scan_shape_belongs_to() -> None:
    # Everything the frontier and the block are measured at is read off this
    # config, so a d_model that does not expand back to H*P benches both operators
    # at widths no layer has.
    config = layer_config(SMALL_LAYER)
    assert config.d_inner == SMALL_LAYER.heads * SMALL_LAYER.rows
    assert config.n_heads == SMALL_LAYER.heads
    assert config.d_head == SMALL_LAYER.rows
    assert config.d_state == SMALL_LAYER.d_state
    assert config.chunk_size == SMALL_LAYER.chunk
    assert config.n_groups == 1
    for shape in SHAPES:
        # The standard shapes are all layers, so no bench resolves to a config the
        # mixer would reject.
        assert layer_config(shape).d_inner == shape.heads * shape.rows
    assert layer_config(SMALL_LAYER, groups=1).n_groups == 1
    with pytest.raises(ValueError, match="H\\*P must be even"):
        layer_config(OpShape("odd", bsz=1, heads=3, seq=8, rows=17, lanes=16, chunk=16))


# ---------------------------------------------------------------------------
# PrepShape and PREP_SHAPES
# ---------------------------------------------------------------------------


def test_prep_shapes_align_every_band_and_carry_the_scan_shape_names() -> None:
    assert tuple(s.name for s in PREP_SHAPES) == SHAPE_NAMES
    # A band is handed over whole and pitched, so its base and the projection width
    # both have to clear the alignment the kernels require. 10*H clears it only for
    # H a multiple of 4, which is why the width is padded and not merely summed.
    assert SMALL_PREP.params_width == 10
    assert SMALL_PREP.bc_offset == 32
    assert SMALL_PREP.bc_width == 96
    assert SMALL_PREP.params_offset == 128
    assert SMALL_PREP.proj_width == 144
    for shape in PREP_SHAPES:
        assert shape.groups >= 1
        assert shape.scan.heads % shape.groups == 0
        assert shape.bc_offset % PROJ_ALIGN == 0
        assert shape.params_offset % PROJ_ALIGN == 0
        assert shape.proj_width % PROJ_ALIGN == 0
        assert shape.proj_width >= shape.params_offset + shape.params_width
    # Neither G = 1 nor G = H exercises the general h // (H//G) address.
    wide = prep_shape_by_name("wide")
    assert 1 < wide.groups < wide.scan.heads
    assert SMALL_PREP.token_count == 8
    assert SMALL_PREP.describe() == "small: B=1 T=8 H=1 3N=48 G=1 W=144"
    with pytest.raises(KeyError, match="no prep shape 'huge'"):
        prep_shape_by_name("huge")


# ---------------------------------------------------------------------------
# make_prep_inputs
# ---------------------------------------------------------------------------


def test_make_prep_inputs_keeps_the_projection_pitch_on_both_operands() -> None:
    got = make_prep_inputs(SMALL_PREP, CPU, dtype=torch.float32)
    scan = SMALL_PREP.scan
    assert tuple(got.proj.shape) == (scan.bsz, scan.seq, SMALL_PREP.proj_width)
    assert tuple(got.params.shape) == (scan.bsz, scan.seq, SMALL_PREP.params_width)
    assert tuple(got.bc.shape) == (scan.bsz, scan.seq, SMALL_PREP.bc_width)
    # The row pitch is the whole projection and only the trailing axis is unit
    # stride. A contiguous copy here would measure an access pattern the mixer
    # never produces.
    pitch = SMALL_PREP.proj_width
    for band in (got.params, got.bc):
        assert band.stride() == (scan.seq * pitch, pitch, 1)
        assert not band.is_contiguous()
    # Leaves, so the backward measures the frontier and not a pullback into a
    # zeroed projection buffer.
    assert got.differentiable == (got.params, got.bc, got.param_bias)
    assert all(t.requires_grad and t.is_leaf for t in got.differentiable)
    assert not any(t.requires_grad for t in got.cotangents)
    plain = make_prep_inputs(SMALL_PREP, CPU, requires_grad=False)
    assert not any(t.requires_grad for t in plain.differentiable)
    # I4: the bias is float32 whatever the projection width, and so are the two
    # packed outputs' cotangents.
    low = make_prep_inputs(SMALL_PREP, CPU, dtype=torch.bfloat16)
    assert low.params.dtype == torch.bfloat16
    assert low.bc.dtype == torch.bfloat16
    assert low.param_bias.dtype == torch.float32
    assert low.dtrans.dtype == torch.float32
    assert low.dK.dtype == torch.float32
    assert low.dB.dtype == torch.bfloat16
    lead = (scan.bsz, scan.heads, scan.seq)
    assert tuple(low.dtrans.shape) == (*lead, 4)
    assert tuple(low.dK.shape) == (*lead, 2, 4)
    assert tuple(low.dB.shape) == (
        scan.bsz,
        SMALL_PREP.groups,
        scan.seq,
        scan.d_state,
    )
    # Two runs of a bench must compare the same numbers.
    assert torch.equal(
        make_prep_inputs(SMALL_PREP, CPU, seed=7).proj,
        make_prep_inputs(SMALL_PREP, CPU, seed=7).proj,
    )
    assert not torch.equal(
        make_prep_inputs(SMALL_PREP, CPU, seed=7).proj,
        make_prep_inputs(SMALL_PREP, CPU, seed=8).proj,
    )


# ---------------------------------------------------------------------------
# prep_forward_only and prep_step
# ---------------------------------------------------------------------------


def test_the_prep_runners_record_their_own_regions() -> None:
    inputs = make_prep_inputs(SMALL_PREP, CPU, dtype=torch.float32, requires_grad=True)
    forward = measure(
        prep_forward_only(inputs, SMALL_PREP),
        label="prep",
        iters=2,
        warmup=1,
        device=CPU,
    )
    assert [t.label for t in forward.regions] == ["prep.forward"]
    assert all(t.requires_grad for t in inputs.differentiable)
    timed = measure(
        prep_step(inputs, SMALL_PREP), label="prep", iters=2, warmup=1, device=CPU
    )
    assert [t.label for t in timed.regions] == ["prep.forward", "prep.backward"]
    assert timed.region("prep.backward").spread.sample_count == 2
    # torch.autograd.grad, so nothing accumulates into a .grad buffer.
    assert all(t.grad is None for t in inputs.differentiable)
    prefixed = measure(
        prep_step(inputs, SMALL_PREP, prefix="arm-b"),
        label="prep",
        iters=1,
        warmup=0,
        device=CPU,
    )
    assert [t.label for t in prefixed.regions] == ["arm-b.forward", "arm-b.backward"]


def test_prep_step_rejects_inputs_that_take_no_gradient() -> None:
    # Otherwise it would time a forward under grad mode and report it as a step.
    with pytest.raises(ValueError, match="at least one input requiring grad"):
        prep_step(make_prep_inputs(SMALL_PREP, CPU, requires_grad=False), SMALL_PREP)
    grads = make_prep_inputs(SMALL_PREP, CPU, requires_grad=True)
    with pytest.raises(ValueError, match="at least one input requiring grad"):
        prep_step(grads, SMALL_PREP, wrt=(grads.dtrans,))


# ---------------------------------------------------------------------------
# BlockShape and BLOCK_SHAPES
# ---------------------------------------------------------------------------


def test_block_shapes_read_both_widths_off_the_layer_config() -> None:
    assert tuple(s.name for s in BLOCK_SHAPES) == SHAPE_NAMES
    # Two kernels, two widths: the norm runs on the residual stream and the
    # activation on the FFN hidden. Benching either at the other's width reports a
    # rate for a row length the block never has.
    for shape in BLOCK_SHAPES:
        config = layer_config(shape.scan)
        assert shape.width == config.d_model
        assert shape.hidden == config.d_ffn
        assert shape.eps == config.norm_eps
        assert shape.hidden > shape.width
    assert SMALL_BLOCK.width == 8
    assert SMALL_BLOCK.hidden == 32
    assert SMALL_BLOCK.token_count == 8
    assert SMALL_BLOCK.describe() == "small: B=1 T=8 d_model=8 d_ffn=32"
    with pytest.raises(KeyError, match="no block shape 'huge'"):
        block_shape_by_name("huge")


# ---------------------------------------------------------------------------
# make_block_inputs
# ---------------------------------------------------------------------------


def test_make_block_inputs_widens_the_residual_stream_and_the_weight() -> None:
    got = make_block_inputs(SMALL_BLOCK, CPU, dtype=torch.bfloat16)
    stream = (SMALL_BLOCK.scan.bsz, SMALL_BLOCK.scan.seq, SMALL_BLOCK.width)
    ffn = (SMALL_BLOCK.scan.bsz, SMALL_BLOCK.scan.seq, SMALL_BLOCK.hidden)
    assert tuple(got.x.shape) == stream
    assert tuple(got.residual.shape) == stream
    assert tuple(got.weight.shape) == (SMALL_BLOCK.width,)
    assert tuple(got.gate.shape) == ffn
    assert tuple(got.up.shape) == ffn
    assert tuple(got.dnormed.shape) == stream
    assert tuple(got.dresidual.shape) == stream
    assert tuple(got.dout.shape) == ffn
    for t in got:
        assert t.is_contiguous()
    # Every block of a stack but the first: the stream has been widened once and is
    # never narrowed again, so its cotangent is float32 too. Benching a
    # low-precision stream would measure a kernel arc the stack does not run.
    assert got.residual.dtype == torch.float32
    assert got.weight.dtype == torch.float32
    assert got.dresidual.dtype == torch.float32
    assert {got.x.dtype, got.gate.dtype, got.up.dtype, got.dnormed.dtype} == {
        torch.bfloat16
    }
    assert got.differentiable == (got.x, got.residual, got.weight, got.gate, got.up)
    assert all(t.requires_grad for t in got.differentiable)
    assert not (got.dnormed.requires_grad or got.dout.requires_grad)
    plain = make_block_inputs(SMALL_BLOCK, CPU, requires_grad=False)
    assert not any(t.requires_grad for t in plain.differentiable)
    # Two runs of a bench must compare the same numbers.
    assert torch.equal(
        make_block_inputs(SMALL_BLOCK, CPU, seed=7).x,
        make_block_inputs(SMALL_BLOCK, CPU, seed=7).x,
    )
    assert not torch.equal(
        make_block_inputs(SMALL_BLOCK, CPU, seed=7).x,
        make_block_inputs(SMALL_BLOCK, CPU, seed=8).x,
    )


# ---------------------------------------------------------------------------
# block_forward_only and block_step
# ---------------------------------------------------------------------------


def test_the_block_runners_record_their_own_regions() -> None:
    inputs = make_block_inputs(
        SMALL_BLOCK, CPU, dtype=torch.float32, requires_grad=True
    )
    forward = measure(
        block_forward_only(inputs, SMALL_BLOCK),
        label="block",
        iters=2,
        warmup=1,
        device=CPU,
    )
    assert [t.label for t in forward.regions] == ["block.forward"]
    assert all(t.requires_grad for t in inputs.differentiable)
    timed = measure(
        block_step(inputs, SMALL_BLOCK), label="block", iters=2, warmup=1, device=CPU
    )
    assert [t.label for t in timed.regions] == ["block.forward", "block.backward"]
    assert timed.region("block.backward").spread.sample_count == 2
    # torch.autograd.grad, so nothing accumulates into a .grad buffer.
    assert all(t.grad is None for t in inputs.differentiable)
    prefixed = measure(
        block_step(inputs, SMALL_BLOCK, prefix="arm-b"),
        label="block",
        iters=1,
        warmup=0,
        device=CPU,
    )
    assert [t.label for t in prefixed.regions] == ["arm-b.forward", "arm-b.backward"]


def test_block_step_rejects_inputs_that_take_no_gradient() -> None:
    # Otherwise it would time a forward under grad mode and report it as a step.
    with pytest.raises(ValueError, match="at least one input requiring grad"):
        block_step(
            make_block_inputs(SMALL_BLOCK, CPU, requires_grad=False), SMALL_BLOCK
        )
    grads = make_block_inputs(SMALL_BLOCK, CPU, requires_grad=True)
    with pytest.raises(ValueError, match="at least one input requiring grad"):
        block_step(grads, SMALL_BLOCK, wrt=(grads.dout,))
