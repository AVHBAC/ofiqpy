"""Typed assessment and component-failure contracts on real face captures."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import pytest
from conftest import bsi_image, official_root

from ofiqpy import Assessor, assess, native_cv
from ofiqpy.config import OFIQConfig
from ofiqpy.measures.core import Measures
from ofiqpy.pipeline import OFIQPipeline
from ofiqpy.results import OFIQ_COMPONENTS, AssessmentStatus, ComponentStatus, FailureCode


def test_real_bsi_image_returns_complete_typed_assessment() -> None:
    root = official_root()
    assessor = Assessor(OFIQConfig(data_root=root / "data"))

    result = assessor.assess(bsi_image("r-01-frontal.png"))

    assert result.status is AssessmentStatus.SUCCESS
    assert tuple(result.components) == OFIQ_COMPONENTS
    assert len(result.components) == 28
    assert all(component.status is ComponentStatus.SUCCESS for component in result.components.values())


def test_real_assessment_components_are_finite_and_immutable() -> None:
    root = official_root()
    result = Assessor(OFIQConfig(data_root=root / "data")).assess(bsi_image("r-01-frontal.png"))

    assert all(math.isfinite(component.raw) for component in result.components.values())
    assert all(math.isfinite(component.scalar) for component in result.components.values())
    assert all(0.0 <= component.scalar <= 100.0 for component in result.components.values())
    with pytest.raises(TypeError):
        result.components["UnifiedQualityScore"] = result.components["UnifiedQualityScore"]  # type: ignore[index]


def test_real_bsi_background_subsample_returns_typed_no_face_failure() -> None:
    assessor = Assessor(OFIQConfig(data_root=official_root() / "data"))
    capture = cv2.imread(str(bsi_image("r-01-frontal.png")))
    background_only = capture[:512, :512].copy()

    result = assessor.assess(background_only)

    assert result.status is AssessmentStatus.FAILURE_TO_ASSESS
    assert result.failure is not None
    assert result.failure.code is FailureCode.NO_FACE
    assert tuple(result.components) == OFIQ_COMPONENTS
    assert all(component.status is ComponentStatus.FAILURE_TO_ASSESS for component in result.components.values())


def test_real_partial_pipeline_preserves_unaffected_component_results() -> None:
    root = official_root()
    config = OFIQConfig(data_root=root / "data")
    image = cv2.imread(str(bsi_image("r-01-frontal.png")))
    pipeline = OFIQPipeline(config, enable_parsing=False)
    session = pipeline.process(image)

    result = Measures(config).compute_typed(session)

    assert result.status is AssessmentStatus.PARTIAL
    assert result.components["UnifiedQualityScore"].status is ComponentStatus.SUCCESS
    assert result.components["LuminanceMean"].status is ComponentStatus.SUCCESS
    assert result.components["NoHeadCoverings"].status is ComponentStatus.FAILURE_TO_ASSESS
    assert result.components["BackgroundUniformity"].status is ComponentStatus.FAILURE_TO_ASSESS


def test_real_bsi_grayscale_pixels_are_rejected_at_the_input_boundary() -> None:
    assessor = Assessor(OFIQConfig(data_root=official_root() / "data"))
    grayscale = cv2.imread(str(bsi_image("r-01-frontal.png")), cv2.IMREAD_GRAYSCALE)

    result = assessor.assess(grayscale)

    assert result.status is AssessmentStatus.FAILURE_TO_ASSESS
    assert result.failure is not None
    assert result.failure.code is FailureCode.INVALID_IMAGE


def test_real_opencv_models_leave_no_decompressed_temp_artifacts(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_path)
    env["OFIQPY_OFIQ_DATA"] = str(official_root() / "data")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    completed = subprocess.run(
        [sys.executable, "-c", "from ofiqpy import Assessor; Assessor()"],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_legacy_adapter_preserves_unreadable_path_error(tmp_path: Path) -> None:
    missing_real_capture = tmp_path / "absent-bsi-capture.png"

    with pytest.raises(FileNotFoundError, match="could not read image"):
        assess(missing_real_capture)


def test_native_diagnostic_has_defined_available_or_unavailable_contract() -> None:
    capture = cv2.imread(str(bsi_image("r-01-frontal.png")))

    if native_cv.AVAILABLE:
        resized = native_cv.resize_linear(capture, 616, 616)
        assert resized.shape == (616, 616, 3)
        assert resized.dtype == capture.dtype
    else:
        with pytest.raises(native_cv.NativeBridgeUnavailableError, match="native/ofiq_cv.so"):
            native_cv.resize_linear(capture, 616, 616)
