"""Canonical 28-component executor and model lifecycle.

LuminanceMean (C03), HeadSize (C20), NoHeadCoverings (C17),
CompressionArtifacts (C09), UnifiedQualityScore, HeadPose Yaw/Pitch/Roll (slot-swap).
Returns OFIQ component names -> scalar (0-100), matching the OFIQ CSV columns.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ..align import luminance, tmetric
from ..results import OFIQ_COMPONENTS, AssessmentResult, ComponentResult, FailureCode, FailureDetail
from ..segmentation.parsing import CLOTH, HAT
from ..sigmoid import _round_half_away, _sigmoid, scalar_conversion
from .types import MeasureValue


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
        try:
            return loader(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)

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

    def preload(self) -> None:
        """Load every canonical measure model before output creation."""
        self._ssim_sess()
        self._magface_sess()
        self._rtree_model()
        self._expression_models()

    # --- C03 LuminanceMean (hardcoded double-sigmoid mapping) ---
    def luminance_mean(self, s):
        L = luminance(s.aligned_face)
        hist = cv2.calcHist([L], [0], s.landmarked_region, [256], [0, 256]).flatten()
        hist = hist / hist.sum()
        mean = float(hist @ np.arange(256)) / 255.0
        scalar = _round_half_away(100.0 * _sigmoid(mean, 0.2, 0.05) * (1.0 - _sigmoid(mean, 0.8, 0.05)))
        return MeasureValue.success("LuminanceMean", mean, max(0.0, min(100.0, scalar)))

    # --- C20 HeadSize ---
    def head_size(self, s):
        T = tmetric(s.landmarks)  # ORIGINAL landmarks
        raw = T / s.image.shape[0]  # original height
        cs = abs(raw - 0.45)
        scalar = scalar_conversion(cs, h=200, a=1.0, s=-1.0, x0=0.0, w=0.05, round=True)
        return MeasureValue.success("HeadSize", raw, scalar)

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
        return MeasureValue.success("NoHeadCoverings", raw, max(0.0, min(100.0, scalar)))

    # --- C09 CompressionArtifacts (reuses OFIQ ssim_248 ONNX) ---
    def compression(self, s):
        crop = s.aligned_face[184 : 616 - 184, 184 : 616 - 184]  # 248x248
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        transformed = cv2.subtract(rgb, (123.7, 116.3, 103.5, 0.0))
        transformed = cv2.divide(transformed, (58.4, 57.1, 57.4, 0.0))
        blob = np.ascontiguousarray(transformed.transpose(2, 0, 1)[None])
        raw = float(self._ssim_sess().run(None, {"input": blob})[0].reshape(-1)[0])
        scalar = scalar_conversion(raw, h=1, a=-0.0278, s=103.0, x0=0.3308, w=0.092, round=True)
        return MeasureValue.success("CompressionArtifacts", raw, scalar)

    # --- UnifiedQualityScore (MagFace magnitude -> sigmoid) ---
    def unified(self, s):
        resized = cv2.resize(s.aligned_face, (192, 192), interpolation=cv2.INTER_LINEAR)
        crop = resized[33 : 192 - 47, 40 : 192 - 40]  # 112x112, BGR
        conv = crop.astype(np.float32) / 255.0
        blob = np.transpose(conv, (2, 0, 1))[None]
        raw = float(self._magface_sess().run(None, {"input": blob})[0].reshape(-1)[0])
        scalar = scalar_conversion(raw, h=100, a=0.0, s=1.0, x0=23.0, w=2.6, round=True)
        return MeasureValue.success("UnifiedQualityScore", raw, scalar)

    # --- HeadPose Yaw/Pitch/Roll (with the OFIQ slot swap) ---
    @staticmethod
    def _cos2(angle_deg):
        c = max(0.0, math.cos(math.radians(angle_deg)))
        return _round_half_away(100.0 * c * c)

    def head_pose(self, s):
        yaw, pitch, roll = s.yaw, s.pitch, s.roll  # geometric
        names = ("HeadPoseYaw", "HeadPosePitch", "HeadPoseRoll")
        if yaw is None or pitch is None or roll is None:
            return tuple(MeasureValue.unavailable(name) for name in names)
        # slot swap: HeadPoseYaw<-pitch, HeadPosePitch<-yaw, HeadPoseRoll<-roll
        # native value = the (swapped) angle in degrees
        return (
            MeasureValue.success("HeadPoseYaw", pitch, self._cos2(pitch)),
            MeasureValue.success("HeadPosePitch", yaw, self._cos2(yaw)),
            MeasureValue.success("HeadPoseRoll", roll, self._cos2(roll)),
        )

    @staticmethod
    def _execute_group(
        out: dict[str, ComponentResult],
        expected_names: tuple[str, ...],
        producer: Callable[[], MeasureValue | tuple[MeasureValue, ...]],
    ) -> None:
        """Run and validate one native OFIQ component group."""
        try:
            produced = producer()
            values = (produced,) if isinstance(produced, MeasureValue) else tuple(produced)
            if not all(isinstance(value, MeasureValue) for value in values):
                raise TypeError("measure producer returned a value outside the MeasureValue contract")
            actual_names = tuple(value.name for value in values)
            if actual_names != expected_names:
                raise ValueError(f"measure producer names {actual_names} do not match expected names {expected_names}")
            for value in values:
                if value.raw is not None and value.scalar is not None:
                    out[value.name] = ComponentResult.success(value.raw, value.scalar)
                else:
                    out[value.name] = ComponentResult.failed(
                        FailureDetail(FailureCode.COMPONENT_UNAVAILABLE, f"{value.name} returned FailureToAssess")
                    )
        except Exception as exc:
            detail = FailureDetail(FailureCode.COMPONENT_ERROR, f"{type(exc).__name__}: {exc}")
            for name in expected_names:
                out[name] = ComponentResult.failed(detail)

    def compute_typed(self, s) -> AssessmentResult:
        """Compute all components, isolating a failure to its owning component group."""
        from . import geometry as G
        from . import models as M
        from . import pixel as P

        out: dict[str, ComponentResult] = {}
        groups: list[tuple[tuple[str, ...], Callable[[], MeasureValue | tuple[MeasureValue, ...]]]] = [
            (("LuminanceMean",), lambda: self.luminance_mean(s)),
            (("HeadSize",), lambda: self.head_size(s)),
            (("NoHeadCoverings",), lambda: self.no_head_coverings(s)),
            (("CompressionArtifacts",), lambda: self.compression(s)),
            (("UnifiedQualityScore",), lambda: self.unified(s)),
            (("InterEyeDistance",), lambda: G.inter_eye_distance(s)),
            (("SingleFacePresent",), lambda: G.single_face_present(s)),
            (("EyesOpen",), lambda: G.eyes_open(s)),
            (("MouthClosed",), lambda: G.mouth_closed(s)),
        ]
        crop_names = (
            "LeftwardCropOfTheFaceImage",
            "RightwardCropOfTheFaceImage",
            "MarginAboveOfTheFaceImage",
            "MarginBelowOfTheFaceImage",
        )
        groups.extend(
            [
                (crop_names, lambda: G.crop_of_face(s)),
                (("BackgroundUniformity",), lambda: P.background_uniformity(s)),
                (("IlluminationUniformity",), lambda: P.illumination_uniformity(s)),
                (("LuminanceVariance",), lambda: P.luminance_variance(s)),
                (("UnderExposurePrevention",), lambda: P.under_exposure(s)),
                (("OverExposurePrevention",), lambda: P.over_exposure(s)),
                (("DynamicRange",), lambda: P.dynamic_range(s)),
                (("NaturalColour",), lambda: P.natural_colour(s)),
                (("EyesVisible",), lambda: M.eyes_visible(s)),
                (("MouthOcclusionPrevention",), lambda: M.mouth_occlusion(s)),
                (("FaceOcclusionPrevention",), lambda: M.face_occlusion(s)),
            ]
        )

        def sharpness() -> MeasureValue:
            rtree, nt = self._rtree_model()
            return M.sharpness(s, rtree, nt)

        def expression() -> MeasureValue:
            e1, e2, bo = self._expression_models()
            return M.expression_neutrality(s, e1, e2, bo)

        pose_names = ("HeadPoseYaw", "HeadPosePitch", "HeadPoseRoll")
        groups.extend(
            [
                (("Sharpness",), sharpness),
                (("ExpressionNeutrality",), expression),
                (pose_names, lambda: self.head_pose(s)),
            ]
        )

        for expected_names, producer in groups:
            self._execute_group(out, expected_names, producer)

        missing = [name for name in OFIQ_COMPONENTS if name not in out]
        if missing:
            detail = FailureDetail(FailureCode.COMPONENT_UNAVAILABLE, "component did not produce a result")
            for name in missing:
                out[name] = ComponentResult.failed(detail)
        return AssessmentResult.from_components(out)

    def compute(self, s) -> dict:
        """Compatibility mapping of component name to ``(raw, scalar)``."""
        return self.compute_typed(s).as_legacy_dict()

    def compute_scalars(self, s) -> dict:
        """Return the compatibility mapping of component name to scalar."""
        return {k: v[1] for k, v in self.compute(s).items()}
