"""Per-component raw-value compatibility bounds for the canonical CPU profile."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .results import OFIQ_COMPONENTS


@dataclass(frozen=True, slots=True)
class RawTolerance:
    absolute_tolerance: float
    rationale: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.absolute_tolerance) or self.absolute_tolerance < 0:
            raise ValueError("raw tolerance must be finite and non-negative")
        if not self.rationale.strip():
            raise ValueError("raw tolerance requires a rationale")


@dataclass(frozen=True, slots=True)
class RawToleranceProfile:
    profile_id: str
    decimal_places: int
    components: Mapping[str, RawTolerance]

    def __post_init__(self) -> None:
        if self.decimal_places < 0:
            raise ValueError("raw comparison precision must be non-negative")
        if tuple(self.components) != OFIQ_COMPONENTS:
            raise ValueError("raw tolerance profile must cover the canonical component order exactly")
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))


_EXACT = "Exact OFIQSampleApp six-decimal output is required for this deterministic component."

CANONICAL_RAW_TOLERANCE_PROFILE = RawToleranceProfile(
    profile_id="bsi-ofiq-v1.1.0-cpu-raw-csv-v1",
    decimal_places=6,
    components={
        "UnifiedQualityScore": RawTolerance(
            0.00001,
            "Ten OFIQSampleApp last-decimal units cover the reviewed CPU ONNX magnitude output.",
        ),
        "BackgroundUniformity": RawTolerance(
            1.5,
            "Face-parsing boundary and gradient envelope; 1.5 is 1.5 percent of the canonical sigmoid width.",
        ),
        "IlluminationUniformity": RawTolerance(
            0.000002,
            "Two OFIQSampleApp last-decimal units cover aligned-mask histogram intersection.",
        ),
        "LuminanceMean": RawTolerance(
            0.00005,
            "Fifty OFIQSampleApp last-decimal units cover normalized aligned-mask luminance histograms.",
        ),
        "LuminanceVariance": RawTolerance(
            0.00001,
            "Ten OFIQSampleApp last-decimal units cover normalized aligned-mask luminance variance.",
        ),
        "UnderExposurePrevention": RawTolerance(
            0.00015,
            "Masked under-exposure proportion envelope for segmentation boundary differences.",
        ),
        "OverExposurePrevention": RawTolerance(0.0, _EXACT),
        "DynamicRange": RawTolerance(
            0.00025,
            "Luminance-entropy envelope for aligned-mask boundary differences.",
        ),
        "Sharpness": RawTolerance(0.0, _EXACT),
        "CompressionArtifacts": RawTolerance(
            0.000002,
            "Two OFIQSampleApp last-decimal units cover the CPU ONNX regression output.",
        ),
        "NaturalColour": RawTolerance(0.0, _EXACT),
        "SingleFacePresent": RawTolerance(
            0.000001,
            "One OFIQSampleApp last-decimal unit covers area-ratio serialization rounding.",
        ),
        "EyesOpen": RawTolerance(0.0, _EXACT),
        "MouthClosed": RawTolerance(
            0.000001,
            "One OFIQSampleApp last-decimal unit covers landmark-distance serialization rounding.",
        ),
        "EyesVisible": RawTolerance(0.0, _EXACT),
        "MouthOcclusionPrevention": RawTolerance(0.0, _EXACT),
        "FaceOcclusionPrevention": RawTolerance(
            0.000005,
            "Occlusion proportion envelope for aligned segmentation boundary differences.",
        ),
        "InterEyeDistance": RawTolerance(
            0.00003,
            "Thirty OFIQSampleApp last-decimal units cover pose propagation into pixel distance.",
        ),
        "HeadSize": RawTolerance(
            0.000001,
            "One OFIQSampleApp last-decimal unit covers normalized landmark-distance serialization rounding.",
        ),
        "LeftwardCropOfTheFaceImage": RawTolerance(0.0, _EXACT),
        "RightwardCropOfTheFaceImage": RawTolerance(0.0, _EXACT),
        "MarginAboveOfTheFaceImage": RawTolerance(0.0, _EXACT),
        "MarginBelowOfTheFaceImage": RawTolerance(0.0, _EXACT),
        "HeadPoseYaw": RawTolerance(
            0.000012,
            "Twelve OFIQSampleApp last-decimal units cover CPU rotation-to-angle evaluation in degrees.",
        ),
        "HeadPosePitch": RawTolerance(
            0.000005,
            "Five OFIQSampleApp last-decimal units cover CPU rotation-to-angle evaluation in degrees.",
        ),
        "HeadPoseRoll": RawTolerance(
            0.000004,
            "Four OFIQSampleApp last-decimal units cover CPU rotation-to-angle evaluation in degrees.",
        ),
        "ExpressionNeutrality": RawTolerance(
            75.0,
            "CPU dual-ONNX embedding and AdaBoost-sum envelope; 75 is 1.5 percent of the canonical sigmoid width.",
        ),
        "NoHeadCoverings": RawTolerance(
            0.003,
            "Face-parsing class-boundary envelope for the normalized top-crop pixel proportion.",
        ),
    },
)
