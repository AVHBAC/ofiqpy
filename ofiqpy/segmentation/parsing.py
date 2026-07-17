"""BiSeNet face parsing — faithful port of FaceParsing.cpp.

Aligned 616x616 -> crop [0:556, 30:586] -> RGB, ImageNet norm -> resize 400 ->
bisenet_400 ONNX (output[0], [1,19,400,400]) -> argmax -> 400x400 class map.
"""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

MEAN255 = np.array([123.675, 116.28, 103.8], np.float32)   # R,G,B
STD255 = np.array([58.395, 57.12, 57.375], np.float32)

# class labels (segmentations.h): 16=cloth, 18=hat, 1=skin, 17=hair, 6=eyeglasses ...
CLOTH, HAT, SKIN, HAIR = 16, 18, 1, 17


class FaceParser:
    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.out0 = self.sess.get_outputs()[0].name

    def parse(self, aligned: np.ndarray) -> np.ndarray:
        crop = aligned[0:616 - 60, 30:616 - 30]          # 556 x 556
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        t = (rgb - MEAN255) / STD255
        t = cv2.resize(t, (400, 400), interpolation=cv2.INTER_LINEAR)
        blob = np.transpose(t, (2, 0, 1))[None]          # (1,3,400,400)
        out = self.sess.run([self.out0], {"input": blob})[0]  # (1,19,400,400)
        return np.argmax(out[0], axis=0).astype(np.uint8)     # 400x400 class map
