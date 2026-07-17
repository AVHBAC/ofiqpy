"""Face occlusion segmentation — faithful port of FaceOcclusionSegmentation.cpp.

Aligned 616x616 -> crop [96:520, 96:520] (424) -> resize 224 -> RGB /255 ONNX
-> logit >= 0 => 1 (non-occluded) -> NEAREST resize 424 -> place into 616 (96px border=0).
"""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort


class OcclusionSeg:
    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.out0 = self.sess.get_outputs()[0].name

    def segment(self, aligned: np.ndarray) -> np.ndarray:
        crop = aligned[96:616 - 96, 96:616 - 96]         # 424 x 424
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(resized, 1.0 / 255.0, (224, 224), (0, 0, 0),
                                     swapRB=True, crop=False)   # BGR->RGB /255
        logit = self.sess.run([self.out0], {"input": blob})[0].reshape(224, 224)
        mask224 = (logit >= 0).astype(np.uint8)          # 1 = non-occluded
        mask424 = cv2.resize(mask224, (424, 424), interpolation=cv2.INTER_NEAREST)
        full = np.zeros((616, 616), np.uint8)
        full[96:520, 96:520] = mask424
        return full
