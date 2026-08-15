"""ADNet 98-point landmarks derived from adnet_landmarks.cpp and utils.cpp.

ONNX input 'input' [1,3,256,256] BGR planes, norm 2p/255-1; output '3209' [1,98,2];
denorm (v+1)/2*255; back-map to original px via square-box scale.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import onnxruntime as ort


def _floor(x):
    return int(math.floor(x))


def _ceil(x):
    return int(math.ceil(x))


def _round(x):
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def _adnet_input(crop: np.ndarray) -> np.ndarray:
    """Match OpenCV ``convertTo(CV_32F, 2./255, -1.)`` with one final rounding."""
    alpha = float(np.float32(2.0 / 255.0))
    beta = float(np.float32(-1.0))
    normalized = (crop.astype(np.float64) * alpha + beta).astype(np.float32)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def make_square_bbox(left, top, w, h):
    """utils.cpp:109-167 — extend the shorter side symmetrically to a square."""
    xtl, ytl, xbr, ybr = left, top, left + w, top + h
    if w < h:
        xl = _floor((xbr + xtl - h) / 2.0)
        xr = _ceil((xbr + xtl + h) / 2.0)
        return xl, ytl, xr - xl, ybr - ytl  # (x, y, side, side)
    else:
        yt = _floor((ybr + ytl - w) / 2.0)
        yb = _ceil((ybr + ytl + w) / 2.0)
        return xtl, yt, xbr - xtl, yb - yt


def make_square_with_padding(sq, img):
    """utils.cpp:50-107 — pad image if the square box escapes bounds."""
    x, y, sw, sh = sq
    H, W = img.shape[:2]
    xbr, ybr = x + sw, y + sh
    top = max(0, -y)
    bottom = max(0, ybr - H + 1)
    left = max(0, -x)
    right = max(0, xbr - W + 1)
    if top or bottom or left or right:
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        padded = img
    tx, ty = left, top
    sq_pad = (x + tx, y + ty, sw, sh)
    return padded, sq_pad, (tx, ty)


class ADNetLandmarker:
    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.out_name = "3209"

    def extract(self, img: np.ndarray, primary_box) -> np.ndarray:
        """img BGR uint8, primary_box=(left,top,w,h). Returns (98,2) original-px coords."""
        sq = make_square_bbox(*primary_box)
        padded, sq_pad, (tx, ty) = make_square_with_padding(sq, img)
        x, y, sw, sh = sq_pad
        crop = padded[y : y + sh, x : x + sw]
        if crop.shape[:2] != (256, 256):
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
        blob = _adnet_input(crop)
        out = self.sess.run([self.out_name], {"input": blob})[0].reshape(-1)
        out = (out + np.float32(1.0)) / np.float32(2.0) * np.float32(255.0)
        scale = np.float32(sq[3]) / np.float32(256.0)  # OFIQ uses square HEIGHT/256 (adnet_landmarks.cpp:313),
        #                          not width — the floor/ceil square can be 1px non-square.
        ox = np.float32(sq_pad[0] - tx)  # original square xleft
        oy = np.float32(sq_pad[1] - ty)
        pts = np.empty((98, 2), np.float64)
        for i in range(98):
            pts[i, 0] = _round(float(np.float32(out[2 * i] * scale + ox)))
            pts[i, 1] = _round(float(np.float32(out[2 * i + 1] * scale + oy)))
        return pts
