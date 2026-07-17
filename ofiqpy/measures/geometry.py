"""Geometry/presentation measures: C11, C12, C13, C19, C24-C27.

Faithful to InterEyeDistance/CropOfTheFaceImage/SingleFacePresent/EyesOpen/
MouthClosed.cpp. IED yaw = pose[1] (== our s.pitch, confirmed by exact HeadPose match).
"""
from __future__ import annotations

import math

from ..align import tmetric
from ..sigmoid import scalar_conversion
from .helpers import (
    LEFT_EYE_CORNERS,
    LEFT_EYE_LID_PAIRS,
    MOUTH_INNER_PAIRS,
    RIGHT_EYE_CORNERS,
    RIGHT_EYE_LID_PAIRS,
    c_round,
    get_distance,
    get_middle,
    max_pair_distance,
)

EPS = 1e-6


def inter_eye_distance(s):
    """C19 — ORIGINAL landmarks, IED*(1/cos(yaw)); yaw=pose[1]=s.pitch."""
    lm = s.landmarks
    L = get_middle([lm[LEFT_EYE_CORNERS[0]], lm[LEFT_EYE_CORNERS[1]]])
    R = get_middle([lm[RIGHT_EYE_CORNERS[0]], lm[RIGHT_EYE_CORNERS[1]]])
    cos_yaw = math.cos(math.radians(s.pitch))
    if abs(cos_yaw) < EPS:
        return "InterEyeDistance", None, -1  # FailureToAssess
    raw = get_distance(L, R) * (1.0 / cos_yaw)
    scalar = scalar_conversion(raw, h=100, a=0, s=1, x0=70.0, w=20.0, round=True)
    return "InterEyeDistance", raw, scalar


def crop_of_face(s):
    """C24/C25/C26/C27 — ORIGINAL landmarks + original image size. Own t / plain IED."""
    lm = s.landmarks
    H, W = s.image.shape[:2]
    L = get_middle([lm[LEFT_EYE_CORNERS[0]], lm[LEFT_EYE_CORNERS[1]]])
    R = get_middle([lm[RIGHT_EYE_CORNERS[0]], lm[RIGHT_EYE_CORNERS[1]]])
    eye_mid = get_middle([L, R])                # integer-rounded midpoint
    chin = lm[16]
    t = get_distance(eye_mid, chin)
    ied = get_distance(L, R)
    raw_left = R[0] / ied                        # C24 Leftward
    raw_right = (W - L[0]) / ied                 # C25 Rightward
    raw_above = eye_mid[1] / t                    # C26 MarginAbove
    raw_below = (H - eye_mid[1]) / t              # C27 MarginBelow
    return {
        "LeftwardCropOfTheFaceImage": (raw_left, scalar_conversion(raw_left, h=100, x0=0.9, w=0.1, round=True)),
        "RightwardCropOfTheFaceImage": (raw_right, scalar_conversion(raw_right, h=100, x0=0.9, w=0.1, round=True)),
        "MarginAboveOfTheFaceImage": (raw_above, scalar_conversion(raw_above, h=100, x0=1.4, w=0.1, round=True)),
        "MarginBelowOfTheFaceImage": (raw_below, scalar_conversion(raw_below, h=100, x0=1.8, w=0.1, round=True)),
    }


def single_face_present(s):
    """C11 — f = 2nd-largest/largest face area; qc = round(100*(1-f)). No sigmoid."""
    areas = s.face_areas
    if not areas:
        return "SingleFacePresent", None, -1
    f = 0.0 if len(areas) == 1 else areas[1] / areas[0]
    qc = c_round(100.0 * (1.0 - f))
    return "SingleFacePresent", f, max(0.0, min(100.0, qc))


def eyes_open(s):
    """C12 — ALIGNED landmarks, min(eye openings)/tmetric(aligned)."""
    al = s.aligned_landmarks
    left = max_pair_distance(al, LEFT_EYE_LID_PAIRS)
    right = max_pair_distance(al, RIGHT_EYE_LID_PAIRS)
    raw = min(left, right) / tmetric(al)
    scalar = scalar_conversion(raw, h=100, a=0, s=1, x0=0.02, w=0.01, round=True)
    return "EyesOpen", raw, scalar


def mouth_closed(s):
    """C13 — ORIGINAL landmarks, max inner-lip opening / tmetric; t==0 -> NaN."""
    lm = s.landmarks
    t = tmetric(lm)
    if t == 0.0:
        return "MouthClosed", None, -1
    raw = max_pair_distance(lm, MOUTH_INNER_PAIRS) / t
    scalar = scalar_conversion(raw, h=100, a=1, s=-1, x0=0.2, w=0.06, round=True)
    return "MouthClosed", raw, scalar
