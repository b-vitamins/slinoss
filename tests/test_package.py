"""The package's public surface.

`slinoss.__all__` is what a stack that depends on this repo imports, so a name
that dangles or a symbol that stops being exported is a break in the only
interface anyone outside the repo sees. Neither failure shows up in a test of the
thing itself: a renamed class keeps passing its own tests while
``from slinoss import *`` stops handing it over.

Same contract `tests/test_perf_units.py` holds for that module, one level up.
"""

import slinoss

EXPECTED = {
    "BlockOutput",
    "DecodeOutput",
    "GraphedStep",
    "MixerState",
    "SLinOSSBlock",
    "SLinOSSConfig",
    "SLinOSSMixerConfig",
    "SLinOSSMixer",
    "SLinOSSStack",
    "StackState",
    "__version__",
    "capture",
    "capture_decode",
    "generate",
}


def test_every_exported_name_resolves() -> None:
    # A dangling entry raises only at star-import time, which no other test does.
    dangling = [name for name in slinoss.__all__ if not hasattr(slinoss, name)]
    assert dangling == []


def test_the_exported_set_is_the_documented_one() -> None:
    # Pinned rather than derived, so adding or dropping a public name is a
    # deliberate edit to this list and not a silent change of surface. The mixer
    # is named separately because it is the operator's entry point and the one
    # name the repo's contract promises by name.
    assert set(slinoss.__all__) == EXPECTED
    assert len(slinoss.__all__) == len(set(slinoss.__all__))
    assert "SLinOSSMixer" in slinoss.__all__


def test_the_version_is_a_release_number() -> None:
    # A version that is not dotted digits breaks any dependency resolver reading
    # it, and nothing else in the repo reads this string.
    parts = slinoss.__version__.split(".")
    assert len(parts) >= 2
    assert all(part.isdigit() for part in parts), slinoss.__version__
