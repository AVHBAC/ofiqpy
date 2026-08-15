"""Internal value contract shared by every canonical measure producer."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MeasureValue:
    """One named raw/scalar result before it becomes a public component result."""

    name: str
    raw: float | None
    scalar: float | None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("measure name must not be empty")
        if (self.raw is None) != (self.scalar is None):
            raise ValueError(f"{self.name} must provide both raw and scalar values, or neither")
        if self.raw is None:
            return
        if self.scalar is None:
            raise ValueError(f"{self.name} must provide both raw and scalar values, or neither")
        raw = float(self.raw)
        scalar = float(self.scalar)
        if not math.isfinite(raw) or not math.isfinite(scalar):
            raise ValueError(f"{self.name} returned a non-finite value")
        if not 0.0 <= scalar <= 100.0:
            raise ValueError(f"{self.name} scalar is outside [0, 100]: {scalar}")
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "scalar", scalar)

    @classmethod
    def success(cls, name: str, raw: float, scalar: float) -> "MeasureValue":
        return cls(name=name, raw=raw, scalar=scalar)

    @classmethod
    def unavailable(cls, name: str) -> "MeasureValue":
        return cls(name=name, raw=None, scalar=None)

    @property
    def available(self) -> bool:
        return self.raw is not None
