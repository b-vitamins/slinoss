"""Partial stub for the CuTe DSL.

The DSL ships no ``py.typed``, and inferring types from its source is worse than
having none. It rewrites the AST of every decorated function, so
``range_constexpr`` is declared only to raise and reads as ``NoReturn``;
``Constexpr`` is a runtime marker with no static meaning, so arithmetic on a
compile-time parameter and any ``int`` passed into one look like errors; float32
arithmetic widens to ``Numeric``; a tensor read is ``IntTuple | Tensor``; and a
``@cute.kernel`` call returns a launch builder the decorator does not describe.
Every one of those is a false positive on correct code.

The DSL also lives on a search-path root that is not the site-packages directory,
so ``useLibraryCodeForTypes`` does not reach it. This states the position
explicitly instead: the DSL surface is dynamically typed. Everything else --
slinoss, torch, pytest, numpy -- stays fully checked.

``__getattr__`` keeps the stub the size of the claim. Do not enumerate the API
here: a hand-maintained stub for a moving DSL would rot, and a rotten stub is a
worse lie than no stub.
"""

from typing import Any

def __getattr__(name: str) -> Any: ...
