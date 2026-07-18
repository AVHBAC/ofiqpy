"""SSD face detector — faithful port of opencv_ssd_face_detector.cpp.

Caffe model via OpenCV DNN (NOT onnxruntime): 300x300, BGR mean (104,117,123),
pad 20%, conf>=0.4, min rel face size 0.05, primary = largest area.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def _round(x: float) -> int:
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


class SSDDetector:
    def __init__(self, caffemodel, prototxt, confidence_thr=0.4, min_rel_face_size=0.05, padding=0.2):
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        self.thr = confidence_thr
        self.min_rel = min_rel_face_size
        self.pad = padding

    def detect(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return face boxes [(left, top, w, h), ...] in ORIGINAL px, largest first."""
        H, W = img.shape[:2]
        padH = int(W * self.pad)  # static_cast truncation
        padV = int(H * self.pad)
        padded = cv2.copyMakeBorder(img, padV, padV, padH, padH, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        Ph, Pw = padded.shape[:2]
        blob = cv2.dnn.blobFromImage(padded, 1.0, (300, 300), (104, 117, 123), swapRB=False, crop=False)
        self.net.setInput(blob)
        det = self.net.forward()  # (1,1,N,7)
        faces = []
        for row in det[0, 0]:
            conf, l, t, r, b = float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])
            if conf >= self.thr and l > 0 and t > 0 and r < 1 and b < 1 and (r - l) > self.min_rel:
                left = _round(l * Pw) - padH
                top = _round(t * Ph) - padV
                width = _round((r - l) * Pw)
                height = _round((b - t) * Ph)
                faces.append((left, top, width, height))
        faces.sort(key=lambda f: f[2] * f[3], reverse=True)
        return faces
