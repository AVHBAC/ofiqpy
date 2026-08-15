"""Pixel/exposure measures: C01, C02, C04(var), C05, C06, C07, C10.

Derived from BackgroundUniformity/IlluminationUniformity/Luminance/Under/Over
Exposure/DynamicRange/NaturalColour.cpp. Note the non-sigmoid mappings:
LuminanceVariance=sin, OverExposure=1/(v+0.01), DynamicRange=12.5*entropy,
IlluminationUniformity=100*D^0.3.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..align import luminance
from ..sigmoid import _round_half_away, scalar_conversion
from .helpers import get_distance, get_middle
from .types import MeasureValue

MOUTH_CENTER_PAIR = (90, 94)


def _norm_hist(L, mask=None):
    if mask is None:
        h = np.bincount(L.ravel(), minlength=256).astype(np.float64)
    else:
        h = np.bincount(L[mask != 0].ravel(), minlength=256).astype(np.float64)
    tot = h.sum()
    return h / tot if tot > 0 else h, tot


def _cheek_rois(al):
    """CalculateReferencePoints + RegionOfInterest on ALIGNED landmarks (image_utils.cpp:114-150)."""
    L = get_middle([al[60], al[64]])
    R = get_middle([al[68], al[72]])
    inter_eye = get_distance(L, R)
    eye_mid = get_middle([L, R])
    mouth = get_middle([al[MOUTH_CENTER_PAIR[0]], al[MOUTH_CENTER_PAIR[1]]])
    eye_mouth = get_distance(eye_mid, mouth)
    zone = int(inter_eye * 0.3)
    dy = int(eye_mouth / 2)
    right_roi = (R[0] - zone, R[1] + dy, zone, zone)  # x,y,w,h
    left_roi = (L[0], L[1] + dy, zone, zone)
    return left_roi, right_roi, zone


def _slice_roi(img, roi):
    x, y, w, h = roi
    return img[y : y + h, x : x + w]


# --- C01 BackgroundUniformity ---
def background_uniformity(s):
    A = np.zeros(s.image.shape[:2], np.uint8)
    P = cv2.warpAffine(A, s.affine, (616, 616), flags=cv2.INTER_NEAREST, borderValue=255)
    I = s.aligned_face[0:406, 62:554]
    Pc = P[0:406, 62:554]
    I = cv2.resize(I, (354, 292), interpolation=cv2.INTER_LINEAR)
    Pc = cv2.resize(Pc, (354, 292), interpolation=cv2.INTER_NEAREST)
    S = s.parsing[0:292, 23:377]  # 400->crop marginX=23
    B = ((Pc == 0) & (S == 0)).astype(np.uint8)
    B = cv2.erode(B, np.ones((4, 4), np.uint8), iterations=1)
    if B.sum() == 0:
        return MeasureValue.unavailable("BackgroundUniformity")
    L = luminance(I).astype(np.float32)
    sx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=-1)  # Scharr
    sy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=-1)
    G = np.sqrt(sx.astype(np.float64) ** 2 + sy.astype(np.float64) ** 2)
    raw = float(G[B != 0].mean())
    return MeasureValue.success("BackgroundUniformity", raw, scalar_conversion(raw, h=190, a=1, s=-1, x0=10, w=100, round=True))


# --- C02 IlluminationUniformity ---
def illumination_uniformity(s):
    faceMask = (s.landmarked_region * 255).astype(np.uint8)
    L = luminance(s.aligned_face)
    maskedL = cv2.bitwise_and(L, L, mask=faceMask)
    left_roi, right_roi, _ = _cheek_rois(s.aligned_landmarks)
    lr = _slice_roi(maskedL, left_roi)
    rr = _slice_roi(maskedL, right_roi)
    if lr.size == 0 or rr.size == 0:
        return MeasureValue.unavailable("IlluminationUniformity")
    hL, _ = _norm_hist(lr)  # empty mask -> counts black pixels too
    hR, _ = _norm_hist(rr)
    D = float(np.minimum(hL, hR).sum())
    scalar = _round_half_away(100.0 * (D**0.3))
    return MeasureValue.success("IlluminationUniformity", D, max(0.0, min(100.0, scalar)))


# --- C04 LuminanceVariance (sin mapping) ---
def luminance_variance(s):
    L = luminance(s.aligned_face)
    hist, _ = _norm_hist(L, s.landmarked_region)
    idx = np.arange(256) / 255.0
    mean = float((hist * idx).sum())
    var = float((hist * (idx - mean) ** 2).sum())
    scalar = _round_half_away(100.0 * math.sin((60 * var) / (60 * var + 1) * math.pi))
    return MeasureValue.success("LuminanceVariance", var, max(0.0, min(100.0, scalar)))


def _exposure(s, lo, hi):
    combined = cv2.bitwise_and(s.landmarked_region, s.occlusion_mask)
    L = luminance(s.aligned_face)
    hist = np.bincount(L[combined != 0].ravel(), minlength=256).astype(np.float64)
    tot = hist.sum()
    if tot == 0:
        return None
    return float(hist[lo : hi + 1].sum() / tot)


# --- C05 UnderExposure ---
def under_exposure(s):
    raw = _exposure(s, 0, 25)
    if raw is None:
        return MeasureValue.unavailable("UnderExposurePrevention")
    return MeasureValue.success(
        "UnderExposurePrevention", raw, scalar_conversion(raw, h=120, a=0.832, s=-1, x0=0.92, w=0.05, round=True)
    )


# --- C06 OverExposure (reciprocal mapping) ---
def over_exposure(s):
    raw = _exposure(s, 247, 255)
    if raw is None:
        return MeasureValue.unavailable("OverExposurePrevention")
    scalar = _round_half_away(1.0 / (raw + 0.01))
    return MeasureValue.success("OverExposurePrevention", raw, max(0.0, min(100.0, scalar)))


# --- C07 DynamicRange (12.5*entropy) ---
def dynamic_range(s):
    L = luminance(s.aligned_face)
    hist = np.bincount(L[s.landmarked_region != 0].ravel(), minlength=256).astype(np.float64)
    tot = hist.sum()
    if tot == 0:
        return MeasureValue.unavailable("DynamicRange")
    p = hist / tot
    nz = p[p != 0]
    ent = float(-(nz * np.log2(nz)).sum())
    scalar = _round_half_away(12.5 * ent)
    return MeasureValue.success("DynamicRange", ent, max(0.0, min(100.0, scalar)))


# --- C10 NaturalColour ---
_D50 = np.array([[0.43605, 0.38508, 0.14309], [0.22249, 0.71689, 0.06062], [0.01393, 0.09710, 0.71419]])
_WHITE = (0.964221, 1.0, 0.825211)
LAB_K = 24289 / 27.0


def _srgb_eotf(x):
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def natural_colour(s):
    region = s.landmarked_region
    seg = cv2.bitwise_and(s.aligned_face, s.aligned_face, mask=region)
    left_roi, right_roi, _ = _cheek_rois(s.aligned_landmarks)
    reduced = cv2.hconcat([_slice_roi(seg, right_roi), _slice_roi(seg, left_roi)])
    if reduced.size == 0:
        return MeasureValue.success(
            "NaturalColour", 100.0, scalar_conversion(100.0, h=200, a=1, s=-1, x0=0.0, w=10.0, round=True)
        )
    Rm = reduced[:, :, 2].mean() / 255.0
    Gm = reduced[:, :, 1].mean() / 255.0
    Bm = reduced[:, :, 0].mean() / 255.0
    r, g, b = _srgb_eotf(Rm), _srgb_eotf(Gm), _srgb_eotf(Bm)
    X, Y, Z = _D50 @ np.array([r, g, b])
    Xr, Yr, Zr = X / _WHITE[0], Y / _WHITE[1], Z / _WHITE[2]
    k, eps = LAB_K, 216 / 24389.0

    def f(t):
        return ((k * t) + 16) / 116 if t <= eps else t ** (1 / 3)

    a_ = 500.0 * (f(Xr) - f(Yr))
    b_ = 200.0 * (f(Yr) - f(Zr))
    if a_ >= 0 and b_ >= 0:
        da = max(0.0, 5 - a_, a_ - 25)
        db = max(0.0, 5 - b_, b_ - 35)
        raw = math.sqrt(da * da + db * db)
    else:
        raw = 100.0
    return MeasureValue.success("NaturalColour", raw, scalar_conversion(raw, h=200, a=1, s=-1, x0=0.0, w=10.0, round=True))
