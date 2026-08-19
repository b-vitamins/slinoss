"""The pitched-layout rule in :mod:`slinoss._guard`.

An operand that is one column band of a wider tensor is pitched rather than
contiguous. Every operator that takes such an operand shares the rule, so it is
pinned once here, against fixtures this file owns, rather than once per caller;
each caller's rejection table keeps one row proving it makes the call.

The contiguous rule needs no file of its own: it has two conditions and every
operator's rejection table already triggers both.
"""

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from slinoss._guard import ALIGN_BYTES, check_pitched

pytestmark = [pytest.mark.cuda]

WIDTH = 64


def _band(pad: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """A ``(2,4,WIDTH)`` band starting ``pad`` columns into a wider buffer."""
    wide = torch.empty(2, 4, WIDTH + 2 * pad, dtype=dtype, device="cuda")
    return wide[..., pad : pad + WIDTH]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_accepts_a_contiguous_tensor_and_an_aligned_band(dtype: torch.dtype) -> None:
    """The ``pitch == width`` case and a real band of a wider buffer.

    The alignment multiple is :data:`slinoss._guard.ALIGN_BYTES` in elements, so it
    is twice as many columns at bfloat16 as at float32; the padding is read from
    the dtype rather than written down, which is the arithmetic the producer has to
    do when it hands a band out.
    """
    multiple = ALIGN_BYTES // torch.empty(0, dtype=dtype).element_size()
    check_pitched(((torch.empty(2, 4, WIDTH, dtype=dtype, device="cuda"), "t"),))
    check_pitched(((_band(multiple, dtype), "t"),))


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: torch.empty(2, 4, WIDTH), r"t must be on a CUDA device"),
        (lambda: torch.empty(WIDTH, device="cuda"), r"t must have a row axis"),
        (
            lambda: torch.empty(2, WIDTH, 4, device="cuda").transpose(1, 2),
            r"t must have unit stride on its trailing axis",
        ),
        (
            lambda: torch.empty(2, 1, WIDTH, device="cuda").expand(2, 4, WIDTH),
            r"t rows overlap",
        ),
        (lambda: _band(1), r"t must start and step on a multiple of 4 elements"),
    ],
)
def test_rejects(make: Callable[[], Tensor], match: str) -> None:
    """Every condition, each reached by the layout it describes.

    The order is the reporting order: the transposed view also has a pitch below
    its row width, and the unaligned band also has an unaligned pitch, so each is
    reported under the first condition it violates.
    """
    with pytest.raises(ValueError, match=match):
        check_pitched(((make(), "t"),))
