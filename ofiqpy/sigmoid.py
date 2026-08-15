"""OFIQ generic ScalarConversion sigmoid (Measure.h:203-285).

quality = h * (a + s * sigmoid(raw; x0, w)), optionally std::round, clamp [0,100].
Struct defaults: h=100, a=0, s=1, x0=4, w=0.7, round=True (Measure.h:121-129).
"""

from __future__ import annotations

import math


def _sigmoid(x: float, x0: float, w: float) -> float:
    z = (x0 - x) / w
    if z > 700:  # exp overflow guard
        return 0.0
    if z < -700:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def _round_half_away(q: float) -> float:
    return math.copysign(math.floor(abs(q) + 0.5), q)


# OFIQ SigmoidParameters::Reset() defaults (Measure.h:121-129)
STRUCT_DEFAULTS = {"h": 100.0, "a": 0.0, "s": 1.0, "x0": 4.0, "w": 0.7, "round": True}


def scalar_conversion(raw: float, **params) -> float:
    p = {**STRUCT_DEFAULTS, **params}
    q = p["h"] * (p["a"] + p["s"] * _sigmoid(raw, p["x0"], p["w"]))
    if p["round"]:
        q = _round_half_away(q)
    return max(0.0, min(100.0, q))
