"""Typed OFIQ assessment results and failure states."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

OFIQ_COMPONENTS = (
    "UnifiedQualityScore",
    "BackgroundUniformity",
    "IlluminationUniformity",
    "LuminanceMean",
    "LuminanceVariance",
    "UnderExposurePrevention",
    "OverExposurePrevention",
    "DynamicRange",
    "Sharpness",
    "CompressionArtifacts",
    "NaturalColour",
    "SingleFacePresent",
    "EyesOpen",
    "MouthClosed",
    "EyesVisible",
    "MouthOcclusionPrevention",
    "FaceOcclusionPrevention",
    "InterEyeDistance",
    "HeadSize",
    "LeftwardCropOfTheFaceImage",
    "RightwardCropOfTheFaceImage",
    "MarginAboveOfTheFaceImage",
    "MarginBelowOfTheFaceImage",
    "HeadPoseYaw",
    "HeadPosePitch",
    "HeadPoseRoll",
    "ExpressionNeutrality",
    "NoHeadCoverings",
)


class ComponentStatus(str, Enum):
    SUCCESS = "success"
    FAILURE_TO_ASSESS = "failure_to_assess"


class AssessmentStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE_TO_ASSESS = "failure_to_assess"


class FailureCode(str, Enum):
    NO_FACE = "no_face"
    IMAGE_READ_ERROR = "image_read_error"
    INVALID_IMAGE = "invalid_image"
    PIPELINE_ERROR = "pipeline_error"
    COMPONENT_ERROR = "component_error"
    COMPONENT_UNAVAILABLE = "component_unavailable"


@dataclass(frozen=True)
class FailureDetail:
    code: FailureCode
    message: str


@dataclass(frozen=True)
class ComponentResult:
    raw: float
    scalar: float
    status: ComponentStatus
    failure: FailureDetail | None = None

    def __post_init__(self) -> None:
        if self.status is ComponentStatus.SUCCESS:
            if self.failure is not None:
                raise ValueError("a successful component cannot carry a failure detail")
            if not math.isfinite(self.raw) or not math.isfinite(self.scalar):
                raise ValueError("successful component values must be finite")
            if not 0.0 <= self.scalar <= 100.0:
                raise ValueError(f"successful component scalar is outside [0, 100]: {self.scalar}")
        else:
            if self.failure is None:
                raise ValueError("a failed component requires a failure detail")
            if self.raw != 0.0 or self.scalar != -1.0:
                raise ValueError("a failed component must use the canonical (0, -1) sentinel")

    @classmethod
    def success(cls, raw: float, scalar: float) -> "ComponentResult":
        return cls(float(raw), float(scalar), ComponentStatus.SUCCESS)

    @classmethod
    def failed(cls, failure: FailureDetail) -> "ComponentResult":
        return cls(0.0, -1.0, ComponentStatus.FAILURE_TO_ASSESS, failure)


@dataclass(frozen=True)
class AssessmentResult:
    components: Mapping[str, ComponentResult]
    status: AssessmentStatus
    failure: FailureDetail | None = None

    def __post_init__(self) -> None:
        supplied = set(self.components)
        expected = set(OFIQ_COMPONENTS)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(f"assessment component set mismatch; missing={missing}, extra={extra}")
        ordered = {name: self.components[name] for name in OFIQ_COMPONENTS}
        successful = sum(result.status is ComponentStatus.SUCCESS for result in ordered.values())
        if successful == len(ordered):
            expected_status = AssessmentStatus.SUCCESS
        elif successful:
            expected_status = AssessmentStatus.PARTIAL
        else:
            expected_status = AssessmentStatus.FAILURE_TO_ASSESS
        if self.status is not expected_status:
            raise ValueError(f"assessment status {self.status.value} does not match component results")
        if self.failure is not None and self.status is not AssessmentStatus.FAILURE_TO_ASSESS:
            raise ValueError("an aggregate failure is valid only when the assessment failed")
        object.__setattr__(self, "components", MappingProxyType(ordered))

    @classmethod
    def from_components(cls, components: Mapping[str, ComponentResult]) -> "AssessmentResult":
        supplied = set(components)
        expected = set(OFIQ_COMPONENTS)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(f"assessment component set mismatch; missing={missing}, extra={extra}")
        ordered = {name: components[name] for name in OFIQ_COMPONENTS}
        successful = sum(result.status is ComponentStatus.SUCCESS for result in ordered.values())
        if successful == len(ordered):
            status = AssessmentStatus.SUCCESS
        elif successful:
            status = AssessmentStatus.PARTIAL
        else:
            status = AssessmentStatus.FAILURE_TO_ASSESS
        return cls(components=ordered, status=status)

    @classmethod
    def failed(cls, failure: FailureDetail) -> "AssessmentResult":
        components = {name: ComponentResult.failed(failure) for name in OFIQ_COMPONENTS}
        return cls(components=components, status=AssessmentStatus.FAILURE_TO_ASSESS, failure=failure)

    def as_legacy_dict(self) -> dict[str, tuple[float, float]]:
        return {name: (result.raw, result.scalar) for name, result in self.components.items()}
