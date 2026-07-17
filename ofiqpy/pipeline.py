"""OFIQ pipeline orchestrator — faithful port (OFIQImpl.cpp preprocess order).

detect -> pose -> landmarks -> align -> parse -> occlusion -> region -> luminance,
each product stored on Session and reused by measures.
"""
from __future__ import annotations

import numpy as np

from .align import align, landmarked_region
from .config import OFIQConfig
from .detectors.ssd import SSDDetector
from .landmarks.adnet import ADNetLandmarker
from .session import Session


class OFIQPipeline:
    def __init__(self, cfg: OFIQConfig | None = None, enable_pose=True,
                 enable_parsing=True, enable_occlusion=True):
        self.cfg = cfg or OFIQConfig()
        d = self.cfg.detector()
        self.detector = SSDDetector(
            self.cfg.resolve(d["model_path"]), self.cfg.resolve(d["prototxt_path"]),
            d["confidence_thr"], d["min_rel_face_size"], d["padding"],
        )
        self.landmarker = ADNetLandmarker(self.cfg.landmarks_model())
        self._pose = self._parser = self._occ = None
        self.enable_pose = enable_pose
        self.enable_parsing = enable_parsing
        self.enable_occlusion = enable_occlusion

    def _get_pose(self):
        if self._pose is None:
            from .pose.tddfa import TDDFAPose
            self._pose = TDDFAPose(self.cfg.resolve(self.cfg.measure("HeadPose")["model_path"]))
        return self._pose

    def _get_parser(self):
        if self._parser is None:
            from .segmentation.parsing import FaceParser
            self._parser = FaceParser(self.cfg.resolve(self.cfg.measure("FaceParsing")["model_path"]))
        return self._parser

    def _get_occ(self):
        if self._occ is None:
            from .segmentation.occlusion import OcclusionSeg
            self._occ = OcclusionSeg(self.cfg.resolve(self.cfg.measure("FaceOcclusionSegmentation")["model_path"]))
        return self._occ

    def process(self, image: np.ndarray) -> Session:
        s = Session(image=image)
        faces = self.detector.detect(image)
        s.n_faces = len(faces)
        s.face_areas = [f[2] * f[3] for f in faces]
        if not faces:
            return s
        primary = faces[0]
        s.bbox = np.array([primary[0], primary[1], primary[0] + primary[2], primary[1] + primary[3]])
        if self.enable_pose:
            yaw, pitch, roll = self._get_pose().estimate(image, primary)
            s.yaw, s.pitch, s.roll = yaw, pitch, roll
        s.landmarks = self.landmarker.extract(image, primary)
        s.aligned_face, s.aligned_landmarks, s.affine = align(image, s.landmarks)
        if self.enable_parsing:
            s.parsing = self._get_parser().parse(s.aligned_face)
        if self.enable_occlusion:
            s.occlusion_mask = self._get_occ().segment(s.aligned_face)
        s.landmarked_region = landmarked_region(s.aligned_landmarks)
        return s
