"""Thread-safe, preflighted lifecycle for canonical OFIQ assessment."""

from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from .config import OFIQConfig
from .measures.core import Measures
from .pipeline import OFIQPipeline
from .results import AssessmentResult, FailureCode, FailureDetail


class Assessor:
    """Own one verified model graph and serialize access to mutable inference sessions."""

    def __init__(self, config: OFIQConfig | None = None):
        self.config = config or OFIQConfig()
        self.pipeline = OFIQPipeline(self.config)
        self.measures = Measures(self.config)
        self._lock = threading.RLock()
        self.pipeline.preload()
        self.measures.preload()

    def assess(self, image: str | os.PathLike[str] | np.ndarray) -> AssessmentResult:
        if isinstance(image, np.ndarray):
            bgr = image
        else:
            bgr = cv2.imread(os.fspath(image))
            if bgr is None:
                return AssessmentResult.failed(FailureDetail(FailureCode.IMAGE_READ_ERROR, f"could not read image: {image}"))
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.shape[0] == 0 or bgr.shape[1] == 0:
            return AssessmentResult.failed(
                FailureDetail(
                    FailureCode.INVALID_IMAGE,
                    f"expected a non-empty BGR uint8 array with shape (height, width, 3); got {bgr.dtype} {bgr.shape}",
                )
            )

        with self._lock:
            try:
                session = self.pipeline.process(bgr)
            except Exception as exc:
                return AssessmentResult.failed(FailureDetail(FailureCode.PIPELINE_ERROR, f"{type(exc).__name__}: {exc}"))
            if session.bbox is None:
                return AssessmentResult.failed(FailureDetail(FailureCode.NO_FACE, "no face detected"))
            return self.measures.compute_typed(session)
