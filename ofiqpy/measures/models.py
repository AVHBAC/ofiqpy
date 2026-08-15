"""Model-based measures: C08 Sharpness, C14/C15/C16 occlusion, C18 Expression.

Sharpness runs on the ORIGINAL image + original landmarks (RTrees via cv2.ml).
Occlusion measures use the 616x616 occlusion mask + aligned regions, linear 100*(1-a).
Expression: crop -> normalize-then-resize -> enet_b0/b2 ONNX -> AdaBoost PREDICT_SUM.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..align import landmarked_region
from ..sigmoid import _round_half_away, scalar_conversion
from .types import MeasureValue


# --- C08 Sharpness ---
def _sharpness_features(gray, mask):
    blur3 = cv2.GaussianBlur(gray, (3, 3), 0)
    feats = []
    for k in (1, 3, 5, 7, 9):
        lap = cv2.Laplacian(blur3, cv2.CV_64F, ksize=k)
        m, sd = cv2.meanStdDev(np.abs(lap), mask=mask)
        feats += [m[0, 0], sd[0, 0]]
    for k in (3, 5, 7):
        box = cv2.blur(gray, (k, k))
        ad = cv2.absdiff(gray, box)
        m, sd = cv2.meanStdDev(ad, mask=mask)
        feats += [m[0, 0], sd[0, 0]]
    for k in (1, 3, 5, 7, 9):
        so = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=k)
        m, sd = cv2.meanStdDev(np.abs(so), mask=mask)
        feats += [m[0, 0], sd[0, 0]]
    return np.array(feats, np.float32).reshape(1, 26)


def sharpness(s, rtree, num_trees):
    H, W = s.image.shape[:2]
    face_mask = (landmarked_region(s.landmarks, H, W) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(face_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return MeasureValue.unavailable("Sharpness")
    x, y, w, h = cv2.boundingRect(contours[0])
    gray = cv2.cvtColor(s.image[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    mask_crop = face_mask[y : y + h, x : x + w]
    feats = _sharpness_features(gray, mask_crop)
    _, raw_out = rtree.predict(feats, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
    native = num_trees - float(raw_out[0, 0])
    return MeasureValue.success(
        "Sharpness", native, scalar_conversion(native, h=1, a=-14.0, s=115.0, x0=-20.0, w=15.0, round=True)
    )


# --- occlusion measures (linear 100*(1-alpha)) ---
def _linear(alpha):
    return max(0.0, min(100.0, _round_half_away(100.0 * (1.0 - alpha))))


def eyes_visible(s):
    al = s.aligned_landmarks.astype(np.int32)
    occ = s.occlusion_mask
    import math

    from .helpers import get_distance, get_middle

    lc = get_middle([al[60], al[64]])
    rc = get_middle([al[68], al[72]])
    cosy = math.cos(math.radians(s.pitch))
    if abs(cosy) < 1e-6:
        return MeasureValue.unavailable("EyesVisible")
    ied = get_distance(lc, rc) / cosy
    V = int(math.floor(ied / 20.0))

    def evz_rect(idx):
        pts = al[idx[0] : idx[1]]
        x, y, w, h = cv2.boundingRect(pts)
        return np.array([[x - V, y - V], [x + w + V, y - V], [x + w + V, y + h + V], [x - V, y + h + V]], np.int32)

    EVZ = np.zeros(occ.shape, np.uint8)
    cv2.drawContours(EVZ, [evz_rect((60, 68)), evz_rect((68, 76))], -1, 1, -1)
    tot = int(EVZ.sum())
    if tot == 0:
        return MeasureValue.unavailable("EyesVisible")
    occluded = int((EVZ * (1 - occ)).sum())
    alpha = occluded / tot
    return MeasureValue.success("EyesVisible", alpha, _linear(alpha))


def mouth_occlusion(s):
    al = s.aligned_landmarks.astype(np.int32)
    poly = al[76:88]
    mask = np.zeros((616, 616), np.uint8)
    cv2.fillConvexPoly(mask, poly, 1)
    tot = int(mask.sum())
    if tot == 0:
        return MeasureValue.unavailable("MouthOcclusionPrevention")
    occluded = int((mask * (1 - s.occlusion_mask)).sum())
    alpha = occluded / tot
    return MeasureValue.success("MouthOcclusionPrevention", alpha, _linear(alpha))


def face_occlusion(s):
    region = s.landmarked_region
    G = int(np.count_nonzero(region))
    if G == 0:
        return MeasureValue.unavailable("FaceOcclusionPrevention")
    occluded = int(np.count_nonzero(region * (1 - s.occlusion_mask)))
    alpha = occluded / G
    return MeasureValue.success("FaceOcclusionPrevention", alpha, _linear(alpha))


# --- C18 ExpressionNeutrality ---
_IMAGENET_MEAN = (0.485, 0.456, 0.406, 0.0)
_IMAGENET_STD = (0.229, 0.224, 0.225, 0.0)


def expression_neutrality(s, enet1, enet2, boost):
    crop = s.aligned_face[148 : 148 + 340, 144 : 144 + 328]
    t = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    t = cv2.divide(t, (255.0, 255.0, 255.0, 0.0))
    t = cv2.subtract(t, _IMAGENET_MEAN)
    t = cv2.divide(t, _IMAGENET_STD)
    r1 = cv2.resize(t, (224, 224), interpolation=cv2.INTER_LINEAR)
    r2 = cv2.resize(t, (260, 260), interpolation=cv2.INTER_LINEAR)
    b1 = np.ascontiguousarray(r1.transpose(2, 0, 1)[None])
    b2 = np.ascontiguousarray(r2.transpose(2, 0, 1)[None])
    f1 = enet1.run(None, {"input": b1})[0].reshape(1, 1280).astype(np.float32)
    f2 = enet2.run(None, {"input": b2})[0].reshape(1, 1408).astype(np.float32)
    feats = np.hstack([f1, f2]).astype(np.float32)
    _, pred = boost.predict(feats, flags=cv2.ml.DTREES_PREDICT_SUM)
    native = float(pred[0, 0])
    return MeasureValue.success(
        "ExpressionNeutrality", native, scalar_conversion(native, h=100, a=0, s=1, x0=-5000.0, w=5000.0, round=True)
    )
