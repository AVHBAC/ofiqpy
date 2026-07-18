"""Shared measure helpers — faithful to FaceMeasures.cpp / adnet_FaceMap.h."""

from __future__ import annotations

import math

# ADNet LM_98 index maps (adnet_FaceMap.h)
LEFT_EYE_CORNERS = (60, 64)
RIGHT_EYE_CORNERS = (68, 72)
CHIN = 16
LEFT_EYE_LID_PAIRS = [(61, 67), (62, 66), (63, 65)]
RIGHT_EYE_LID_PAIRS = [(69, 75), (70, 74), (71, 73)]
MOUTH_INNER_PAIRS = [(89, 95), (90, 94), (91, 93)]


def c_round(x: float) -> float:
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def get_distance(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def get_middle(points):
    """Integer (int16) rounded midpoint — FaceMeasures.cpp GetMiddle."""
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    return (int(c_round(sx / n)), int(c_round(sy / n)))


def max_pair_distance(lm, pairs) -> float:
    return max(get_distance(lm[i], lm[j]) for i, j in pairs)
