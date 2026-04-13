"""Private validation helpers shared across the layer stack."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


__all__ = ["_require"]
