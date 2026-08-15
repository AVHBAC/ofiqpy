"""Alignment (616x616), landmarked-region mask, tmetric, luminance.

Derived from utils.cpp:236-331, FaceMeasures.cpp:98-228, and image_utils.cpp:43-112.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# ADNet FaceMap indices (adnet_FaceMap.h)
LEFT_EYE_CORNERS = (60, 64)
RIGHT_EYE_CORNERS = (68, 72)
NOSE = 54
RIGHT_MOUTH = 76
LEFT_MOUTH = 82
CHIN = 16

REF_POINTS = np.array(
    [
        [251, 272],
        [364, 272],
        [308, 336],
        [262, 402],
        [355, 402],
    ],
    np.float32,
)


def _round(x):
    return math.copysign(math.floor(abs(x) + 0.5), x)


def eye_centers(P: np.ndarray):
    """Midpoints of eye-corner pairs, each coord rounded (utils.cpp:302-317)."""
    l = (
        _round(P[LEFT_EYE_CORNERS[0], 0] + 0.5 * (P[LEFT_EYE_CORNERS[1], 0] - P[LEFT_EYE_CORNERS[0], 0])),
        _round(P[LEFT_EYE_CORNERS[0], 1] + 0.5 * (P[LEFT_EYE_CORNERS[1], 1] - P[LEFT_EYE_CORNERS[0], 1])),
    )
    r = (
        _round(P[RIGHT_EYE_CORNERS[0], 0] + 0.5 * (P[RIGHT_EYE_CORNERS[1], 0] - P[RIGHT_EYE_CORNERS[0], 0])),
        _round(P[RIGHT_EYE_CORNERS[0], 1] + 0.5 * (P[RIGHT_EYE_CORNERS[1], 1] - P[RIGHT_EYE_CORNERS[0], 1])),
    )
    return l, r


def align(img: np.ndarray, P: np.ndarray):
    """Return (aligned 616x616 BGR, aligned_landmarks (98,2), affine 2x3)."""
    Lc, Rc = eye_centers(P)
    src = np.array(
        [
            [Lc[0], Lc[1]],
            [Rc[0], Rc[1]],
            [P[NOSE, 0], P[NOSE, 1]],
            [P[RIGHT_MOUTH, 0], P[RIGHT_MOUTH, 1]],
            [P[LEFT_MOUTH, 0], P[LEFT_MOUTH, 1]],
        ],
        np.float32,
    )
    M, _ = cv2.estimateAffinePartial2D(src, REF_POINTS, method=cv2.LMEDS)
    aligned = cv2.warpAffine(img, M, (616, 616))  # default INTER_LINEAR, BORDER_CONSTANT 0
    ap = cv2.transform(P.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
    aligned_lms = np.array([[_round(x), _round(y)] for x, y in ap], np.float64)
    return aligned, aligned_lms, M


def tmetric(P: np.ndarray) -> float:
    """||chin - eye_midpoint|| (utils.cpp:319-331). Space-agnostic; pass the right landmarks."""
    Lc, Rc = eye_centers(P)
    eye_mid = ((Lc[0] + Rc[0]) / 2.0, (Lc[1] + Rc[1]) / 2.0)
    chin = P[CHIN]
    return float(math.hypot(chin[0] - eye_mid[0], chin[1] - eye_mid[1]))


def landmarked_region(aligned_lms: np.ndarray, height=616, width=616, alpha=0.0) -> np.ndarray:
    """GetFaceMask at alpha=0 (FaceMeasures.cpp:173-226): convex hull -> 224 -> resize."""
    pts = aligned_lms.astype(np.int32)
    hull = cv2.convexHull(pts)
    rx, ry, rw, rh = cv2.boundingRect(hull)
    b = int(ry - rh * 0.05)
    d = int(ry + rh * 1.05)
    a = int(rx + rw / 2.0 - (d - b) / 2.0)
    c = int(rx + rw / 2.0 + (d - b) / 2.0)
    S = 224
    hull_s = ((hull.reshape(-1, 2).astype(np.float64) - (a, b)) / (d - b) * S).astype(np.int32)
    mask = np.zeros((S, S), np.uint8)
    cv2.fillConvexPoly(mask, hull_s, 1)
    mrs = cv2.resize(mask, (c - a, d - b), interpolation=cv2.INTER_NEAREST)
    region = np.zeros((height, width), np.uint8)
    left, top = 0, 0
    right, bottom = mrs.shape[1], mrs.shape[0]
    an, bn, cn, dn = a, b, c, d
    if a < 0:
        left = -a
        an = 0
    if c > width:
        right = mrs.shape[1] - (c - width)
        cn = width
    if b < 0:
        top = -b
        bn = 0
    if d > height:
        bottom = mrs.shape[0] - (d - height)
        dn = height
    region[bn:dn, an:cn] = mrs[top:bottom, left:right]
    return region


# --- luminance (image_utils.cpp:43-112) ---
_SRGB_LUT = np.array(
    [((v / 255.0) / 12.92) if (v / 255.0) <= 0.04045 else (((v / 255.0) + 0.055) / 1.055) ** 2.4 for v in range(256)], np.float64
)


def luminance(img_bgr: np.ndarray) -> np.ndarray:
    """sRGB-linearized Rec.709 luma -> floor(y*255+0.5) uint8 (BGR input)."""
    B = _SRGB_LUT[img_bgr[:, :, 0]]
    G = _SRGB_LUT[img_bgr[:, :, 1]]
    R = _SRGB_LUT[img_bgr[:, :, 2]]
    y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return np.floor(y * 255.0 + 0.5).astype(np.uint8)
