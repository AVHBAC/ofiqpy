"""OFIQ-compatible CSV output (mirrors OFIQSampleApp).

Semicolon-delimited: Filename; <raw per component in OFIQ order>; <.scalar per
component>; assessment_time_in_ms. Unimplemented components emit the OFIQ
FailureToAssess sentinel (raw 0, scalar -1) until ported.
"""
from __future__ import annotations

from pathlib import Path

# OFIQ CSV component order (from OFIQSampleApp header).
OFIQ_ORDER = [
    "UnifiedQualityScore", "BackgroundUniformity", "IlluminationUniformity", "LuminanceMean",
    "LuminanceVariance", "UnderExposurePrevention", "OverExposurePrevention", "DynamicRange",
    "Sharpness", "CompressionArtifacts", "NaturalColour", "SingleFacePresent", "EyesOpen",
    "MouthClosed", "EyesVisible", "MouthOcclusionPrevention", "FaceOcclusionPrevention",
    "InterEyeDistance", "HeadSize", "LeftwardCropOfTheFaceImage", "RightwardCropOfTheFaceImage",
    "MarginAboveOfTheFaceImage", "MarginBelowOfTheFaceImage", "HeadPoseYaw", "HeadPosePitch",
    "HeadPoseRoll", "ExpressionNeutrality", "NoHeadCoverings",
]

FAILURE = (0.0, -1.0)  # OFIQ FailureToAssess / not-implemented sentinel


def header() -> str:
    cols = ["Filename"] + OFIQ_ORDER + [f"{c}.scalar" for c in OFIQ_ORDER] + ["assessment_time_in_ms"]
    return ";".join(cols)


def row(filename: str, results: dict, time_ms: float = 0.0) -> str:
    """results: {component: (raw, scalar)}. Missing components -> FailureToAssess."""
    raws, scalars = [], []
    for c in OFIQ_ORDER:
        raw, scalar = results.get(c, FAILURE)
        raws.append(f"{raw:.6f}" if raw is not None else "nan")
        scalars.append(str(int(scalar)) if scalar is not None else "-1")
    parts = [Path(filename).name] + raws + scalars + [f"{time_ms:.0f}"]
    return ";".join(parts)


def write_csv(path, rows: list[str]) -> None:
    Path(path).write_text(header() + "\n" + "\n".join(rows) + "\n")
