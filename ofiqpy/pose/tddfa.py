"""3DDFA-V2 head pose — faithful port of HeadPose3DDFAV2.cpp.

Crop from ORIGINAL image + SSD box (cy-0.44h .. cy+0.51h, squared), 120x120,
(pix-127.5)/128 BGR NCHW, mb1 ONNX -> 62 params, first 7 -> rotation -> angles.
Returns geometric (yaw, pitch, roll) in degrees; the HeadPose measure applies the swap.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import onnxruntime as ort

from ..landmarks.adnet import make_square_with_padding

PARAM_MEAN = np.array([3.4926363e-04, 2.5279013e-07, -6.8751979e-07, 6.0167957e+01,
                       -6.2955132e-07, 5.7572004e-04, -5.0853912e-05], np.float64)
PARAM_STD = np.array([1.76321526e-04, 6.73794348e-05, 4.47084894e-04, 2.65502319e+01,
                      1.23137695e-04, 4.49302170e-05, 7.92367064e-05], np.float64)
THRES = 0.9975


class TDDFAPose:
    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def estimate(self, image: np.ndarray, box) -> tuple[float, float, float]:
        left, top, w, h = box
        cx = left + w / 2.0
        cy = top + h / 2.0
        b = int(cy - 0.44 * h)   # int16 truncation toward zero
        d = int(cy + 0.51 * h)
        a = int(cx - (d - b) / 2.0)
        c = a + (d - b)
        sq = (a, b, c - a, d - b)
        padded, sq_pad, _ = make_square_with_padding(sq, image)
        x, y, sw, sh = sq_pad
        crop = padded[y:y + sh, x:x + sw]
        crop = cv2.resize(crop, (120, 120), interpolation=cv2.INTER_LINEAR)
        arr = (crop.astype(np.float32) - 127.5) / 128.0        # BGR
        blob = np.transpose(arr, (2, 0, 1))[None]              # (1,3,120,120)
        out = self.sess.run(["output"], {"input": blob})[0].reshape(-1)
        p = out[:7] * PARAM_STD + PARAM_MEAN
        r0 = p[[0, 1, 2]].astype(np.float64)
        r1 = p[[4, 5, 6]].astype(np.float64)
        r0 /= np.linalg.norm(r0)
        r1 /= np.linalg.norm(r1)
        r2 = np.cross(r0, r1)
        R = np.stack([r0, r1, r2], axis=0).T
        r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
        r21 = R[1, 0]
        r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]
        if -THRES <= r31 <= THRES:
            pitch = math.asin(r31)
            s = 1.0 / math.cos(pitch)
            yaw = -math.atan2(s * r32, s * r33)
            roll = -math.atan2(s * r21, s * r11)
        elif r31 < -THRES:
            pitch = -math.pi / 2
            yaw = -math.atan2(r12, r13)
            roll = 0.0
        else:
            pitch = math.pi / 2
            yaw = math.atan2(r12, r13)
            roll = 0.0
        deg = 180.0 / math.pi
        return yaw * deg, pitch * deg, roll * deg   # geometric (yaw, pitch, roll)
