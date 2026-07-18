"""Vertical-slice measures — faithful ports, each gated ±1 vs OFIQ.

LuminanceMean (C03), HeadSize (C20), NoHeadCoverings (C17),
CompressionArtifacts (C09), UnifiedQualityScore, HeadPose Yaw/Pitch/Roll (slot-swap).
Returns OFIQ component names -> scalar (0-100), matching the OFIQ CSV columns.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ..align import luminance, tmetric
from ..segmentation.parsing import CLOTH, HAT
from ..sigmoid import _round_half_away, _sigmoid, scalar_conversion


class Measures:
    def __init__(self, cfg):
        self.cfg = cfg
        self._ssim = None
        self._magface = None
        self._rtree = None
        self._num_trees = None
        self._enet1 = self._enet2 = self._boost = None

    def _load_ml_gz(self, path, loader):
        import gzip
        import tempfile

        data = gzip.decompress(Path(path).read_bytes())
        suffix = ".xml" if str(path).endswith(".xml.gz") else ".yml"
        with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as f:
            f.write(data)
            tmp = f.name
        return loader(tmp)

    def _rtree_model(self):
        if self._rtree is None:
            p = self.cfg.resolve(self.cfg.measure("Sharpness")["model_path"])
            self._rtree = self._load_ml_gz(p, cv2.ml.RTrees_load)
            self._num_trees = int(self._rtree.getTermCriteria()[1])
        return self._rtree, self._num_trees

    def _expression_models(self):
        if self._enet1 is None:
            m = self.cfg.measure("ExpressionNeutrality")
            self._enet1 = ort.InferenceSession(str(self.cfg.resolve(m["cnn1_model_path"])), providers=["CPUExecutionProvider"])
            self._enet2 = ort.InferenceSession(str(self.cfg.resolve(m["cnn2_model_path"])), providers=["CPUExecutionProvider"])
            self._boost = self._load_ml_gz(self.cfg.resolve(m["adaboost_model_path"]), cv2.ml.Boost_load)
        return self._enet1, self._enet2, self._boost

    def _ssim_sess(self):
        if self._ssim is None:
            p = self.cfg.resolve(self.cfg.measure("CompressionArtifacts")["model_path"])
            self._ssim = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        return self._ssim

    def _magface_sess(self):
        if self._magface is None:
            p = self.cfg.resolve(self.cfg.measure("UnifiedQualityScore")["model_path"])
            self._magface = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        return self._magface

    # --- C03 LuminanceMean (hardcoded double-sigmoid mapping) ---
    def luminance_mean(self, s):
        L = luminance(s.aligned_face)
        hist = cv2.calcHist([L], [0], s.landmarked_region, [256], [0, 256]).flatten()
        hist = hist / hist.sum()
        mean = float(hist @ np.arange(256)) / 255.0
        scalar = _round_half_away(100.0 * _sigmoid(mean, 0.2, 0.05) * (1.0 - _sigmoid(mean, 0.8, 0.05)))
        return "LuminanceMean", max(0.0, min(100.0, scalar)), mean

    # --- C20 HeadSize ---
    def head_size(self, s):
        T = tmetric(s.landmarks)  # ORIGINAL landmarks
        raw = T / s.image.shape[0]  # original height
        cs = abs(raw - 0.45)
        scalar = scalar_conversion(cs, h=200, a=1.0, s=-1.0, x0=0.0, w=0.05, round=True)
        return "HeadSize", scalar, raw

    # --- C17 NoHeadCoverings (custom piecewise mapping) ---
    def no_head_coverings(self, s):
        M = s.parsing  # 400x400
        crop = M[0 : 400 - 204, :]  # top 196 rows
        n = int((crop == CLOTH).sum() + (crop == HAT).sum())
        raw = n / (400 * 196)
        T0, T1, w, x0 = 0.0, 0.95, 0.1, 0.02
        if raw <= T0:
            scalar = 100.0
        elif raw >= T1:
            scalar = 0.0
        else:
            sv = _sigmoid(raw, x0, w)
            s0 = _sigmoid(T0, x0, w)
            s1 = _sigmoid(T1, x0, w)
            scalar = _round_half_away(100.0 * (s1 - sv) / (s1 - s0))
        return "NoHeadCoverings", max(0.0, min(100.0, scalar)), raw

    # --- C09 CompressionArtifacts (reuses OFIQ ssim_248 ONNX) ---
    def compression(self, s):
        crop = s.aligned_face[184 : 616 - 184, 184 : 616 - 184]  # 248x248
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.array([123.7, 116.3, 103.5], np.float32)
        std = np.array([58.4, 57.1, 57.4], np.float32)
        t = (rgb - mean) / std
        blob = np.transpose(t, (2, 0, 1))[None]
        raw = float(self._ssim_sess().run(None, {"input": blob})[0].reshape(-1)[0])
        scalar = scalar_conversion(raw, h=1, a=-0.0278, s=103.0, x0=0.3308, w=0.092, round=True)
        return "CompressionArtifacts", scalar, raw

    # --- UnifiedQualityScore (MagFace magnitude -> sigmoid) ---
    def unified(self, s):
        resized = cv2.resize(s.aligned_face, (192, 192), interpolation=cv2.INTER_LINEAR)
        crop = resized[33 : 192 - 47, 40 : 192 - 40]  # 112x112, BGR
        conv = crop.astype(np.float32) / 255.0
        blob = np.transpose(conv, (2, 0, 1))[None]
        raw = float(self._magface_sess().run(None, {"input": blob})[0].reshape(-1)[0])
        scalar = scalar_conversion(raw, h=100, a=0.0, s=1.0, x0=23.0, w=2.6, round=True)
        return "UnifiedQualityScore", scalar, raw

    # --- HeadPose Yaw/Pitch/Roll (with the OFIQ slot swap) ---
    @staticmethod
    def _cos2(angle_deg):
        c = max(0.0, math.cos(math.radians(angle_deg)))
        return _round_half_away(100.0 * c * c)

    def head_pose(self, s):
        yaw, pitch, roll = s.yaw, s.pitch, s.roll  # geometric
        # slot swap: HeadPoseYaw<-pitch, HeadPosePitch<-yaw, HeadPoseRoll<-roll
        # native value = the (swapped) angle in degrees
        return {
            "HeadPoseYaw": (pitch, self._cos2(pitch)),
            "HeadPosePitch": (yaw, self._cos2(yaw)),
            "HeadPoseRoll": (roll, self._cos2(roll)),
        }

    def compute(self, s) -> dict:
        """Return {component_name: (raw, scalar)} for all implemented measures."""
        from . import geometry as G
        from . import models as M
        from . import pixel as P

        out = {}
        for fn in (self.luminance_mean, self.head_size, self.no_head_coverings, self.compression, self.unified):
            name, scalar, raw = fn(s)
            out[name] = (raw, scalar)
        for fn in (G.inter_eye_distance, G.single_face_present, G.eyes_open, G.mouth_closed):
            name, raw, scalar = fn(s)
            out[name] = (raw, scalar)
        out.update(G.crop_of_face(s))
        # pixel/exposure batch
        for fn in (
            P.background_uniformity,
            P.illumination_uniformity,
            P.luminance_variance,
            P.under_exposure,
            P.over_exposure,
            P.dynamic_range,
            P.natural_colour,
        ):
            name, raw, scalar = fn(s)
            out[name] = (raw, scalar)
        # occlusion measures
        for fn in (M.eyes_visible, M.mouth_occlusion, M.face_occlusion):
            name, raw, scalar = fn(s)
            out[name] = (raw, scalar)
        # model measures
        rtree, nt = self._rtree_model()
        out["Sharpness"] = tuple(M.sharpness(s, rtree, nt)[1:])
        e1, e2, bo = self._expression_models()
        out["ExpressionNeutrality"] = tuple(M.expression_neutrality(s, e1, e2, bo)[1:])
        if s.yaw is not None:
            out.update(self.head_pose(s))
        return out

    def compute_scalars(self, s) -> dict:
        """Return {component_name: scalar} (for the ±1 gate)."""
        return {k: v[1] for k, v in self.compute(s).items()}
