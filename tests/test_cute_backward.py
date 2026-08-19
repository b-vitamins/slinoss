"""Cotangent contract of the backward driver.

The driver checks all four cotangents before it launches anything, so a caller's
mistake names the cotangent rather than surfacing from whichever stage happens to
read it. Two of the rules cannot be checked anywhere else: the dtype agreement
spans two stages, and the all-absent call reaches no stage at all.

The check reads shape and dtype and nothing else, so the operands are empty CPU
tensors.
"""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from collections.abc import Callable

from slinoss.ops.so3ssd.cute.guard import check_cotangents

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# (B, H, G, T, P, 3N), as check_shapes returns it. G divides H and 3N is a multiple
# of 48, so the record is a legal one and every rejection below is the mutation's
# own fault.
SHAPE = (2, 4, 2, 8, 16, 48)

Cotangents = dict[str, torch.Tensor | None]


def _ok() -> Cotangents:
    """All four cotangents, at the shapes ``SHAPE`` implies."""
    bsz, heads, groups, seqlen, rows, dim = SHAPE
    return {
        "dy": torch.empty(bsz, heads, seqlen, rows, dtype=torch.bfloat16),
        "dstate": torch.empty(bsz, heads, rows, dim, dtype=torch.float32),
        "db_last": torch.empty(bsz, groups, dim, dtype=torch.bfloat16),
        "du_last": torch.empty(bsz, heads, rows, dtype=torch.bfloat16),
    }


def _check(args: Cotangents) -> None:
    check_cotangents(
        args["dy"], args["dstate"], args["db_last"], args["du_last"], SHAPE
    )


# One case per expected shape rather than one per branch: the four share one loop,
# but the shape each is compared against is hand-written, and a wrong entry in that
# table is only reachable through its own cotangent.
WRONG_SHAPE = [
    pytest.param(
        "dy", lambda t: t[:, :, 0].contiguous(), r"dy must be \(2, 4, 8, 16\)"
    ),
    pytest.param("dstate", lambda t: t[..., :16].contiguous(), "dstate must be"),
    pytest.param("db_last", lambda t: t[:, :1].contiguous(), "db_last must be"),
    pytest.param("du_last", lambda t: t[:, :2].contiguous(), "du_last must be"),
]


@pytest.mark.parametrize(("name", "mutate", "match"), WRONG_SHAPE)
def test_rejects_a_cotangent_of_the_wrong_shape(
    name: str,
    mutate: Callable[[torch.Tensor], torch.Tensor],
    match: str,
) -> None:
    """A cotangent that does not match the forward output it belongs to."""
    args = _ok()
    current = args[name]
    assert current is not None
    args[name] = mutate(current)
    with pytest.raises(ValueError, match=match):
        _check(args)


def test_rejects_activation_cotangents_that_disagree_about_dtype() -> None:
    """The rule this check exists for.

    ``dy`` is read by the chunk-start stage and ``du_last`` by the boundary stage,
    so neither kernel sees both and a mixed pair would launch twice at two dtypes
    and return gradients in two dtypes with no error.
    """
    args = _ok()
    du_last = args["du_last"]
    assert du_last is not None
    args["du_last"] = du_last.half()
    with pytest.raises(TypeError, match="one dtype per call"):
        _check(args)


def test_rejects_a_call_with_no_cotangent() -> None:
    """Every cotangent absent asks the driver to run to produce zeros."""
    with pytest.raises(ValueError, match="at least one cotangent"):
        check_cotangents(None, None, None, None, SHAPE)


def test_accepts_every_cotangent() -> None:
    """The baseline the rejections mutate.

    Without it every rejection above would also pass against a check that refuses
    everything.
    """
    _check(_ok())


def test_accepts_a_state_cotangent_alone() -> None:
    """A state-only backward carries no activation cotangent.

    The dtype group is then empty, and a group of one dtype has nothing to agree on.
    """
    args = _ok()
    args["dy"] = None
    args["db_last"] = None
    args["du_last"] = None
    _check(args)
