"""Cardinality-enforcing live OFIQ conformance contract on all real BSI images."""

from __future__ import annotations

import inspect
import platform
from pathlib import Path

import cv2
import pytest
from conftest import bsi_image, historical_celeba_image_000189, official_root

from ofiqpy.conformance import ConformanceContractError, _assert_stable_artifacts, run_conformance
from ofiqpy.provenance import sha256_file


def test_artifact_stability_guard_rejects_two_different_real_bsi_bindings() -> None:
    before = {"input_aggregate_sha256": sha256_file(bsi_image("r-01-frontal.png"))}
    after = {"input_aggregate_sha256": sha256_file(bsi_image("r-02-rolled.png"))}

    with pytest.raises(ConformanceContractError, match="changed during conformance execution"):
        _assert_stable_artifacts(before, after)


def test_real_celeba_sharpness_boundary_is_exact(tmp_path: Path) -> None:
    root = official_root()
    image = historical_celeba_image_000189()
    cohort = tmp_path / "celeba-sharpness-boundary"
    cohort.mkdir()
    (cohort / image.name).symlink_to(image)

    report = run_conformance(
        ofiq_root=root,
        image_dir=cohort,
        expected_count=1,
        scalar_tolerance=0.0,
        source_root=Path(__file__).resolve().parents[1],
    )

    sharpness = report.components["Sharpness"]
    assert sharpness.scalar_exact == 1
    assert sharpness.raw_exact_at_6_decimals == 1
    assert sharpness.max_scalar_delta == 0.0


def test_real_bsi_pose_uses_reference_float32_path(tmp_path: Path) -> None:
    root = official_root()
    image = bsi_image("r-07-scarf.png")
    cohort = tmp_path / "bsi-pose-float32"
    cohort.mkdir()
    (cohort / image.name).symlink_to(image)

    report = run_conformance(
        ofiq_root=root,
        image_dir=cohort,
        expected_count=1,
        scalar_tolerance=0.0,
        source_root=Path(__file__).resolve().parents[1],
    )

    yaw = report.components["HeadPoseYaw"]
    assert yaw.raw_exact_at_6_decimals == 1


def test_real_bsi_expression_uses_reference_opencv_normalization(tmp_path: Path) -> None:
    root = official_root()
    image = bsi_image("r-01-frontal.png")
    cohort = tmp_path / "bsi-expression-normalization"
    cohort.mkdir()
    (cohort / image.name).symlink_to(image)

    report = run_conformance(
        ofiq_root=root,
        image_dir=cohort,
        expected_count=1,
        scalar_tolerance=0.0,
        source_root=Path(__file__).resolve().parents[1],
    )

    expression = report.components["ExpressionNeutrality"]
    assert expression.max_raw_delta <= 0.2


def test_real_bsi_compression_uses_reference_opencv_normalization(tmp_path: Path) -> None:
    root = official_root()
    image = bsi_image("c-10-background.png")
    cohort = tmp_path / "bsi-compression-normalization"
    cohort.mkdir()
    (cohort / image.name).symlink_to(image)

    report = run_conformance(
        ofiq_root=root,
        image_dir=cohort,
        expected_count=1,
        scalar_tolerance=0.0,
        source_root=Path(__file__).resolve().parents[1],
    )

    compression = report.components["CompressionArtifacts"]
    assert compression.raw_exact_at_6_decimals == 1


def test_real_bsi_expression_uses_reference_float_resize_path(tmp_path: Path) -> None:
    root = official_root()
    image = bsi_image("c-02-headcovered-mouthopen.png")
    cohort = tmp_path / "bsi-expression-float-resize"
    cohort.mkdir()
    (cohort / image.name).symlink_to(image)

    report = run_conformance(
        ofiq_root=root,
        image_dir=cohort,
        expected_count=1,
        scalar_tolerance=0.0,
        source_root=Path(__file__).resolve().parents[1],
    )

    expression = report.components["ExpressionNeutrality"]
    assert expression.raw_exact_at_6_decimals == 1
    assert report.bindings["opencv_optimized"] is False


def test_real_bsi_derived_no_face_excludes_undefined_raw_values(tmp_path: Path) -> None:
    root = official_root()
    capture = cv2.imread(str(root / "data" / "tests" / "images" / "r-01-frontal.png"))
    no_face = capture[:512, :512].copy()
    input_dir = tmp_path / "real-bsi-background"
    input_dir.mkdir()
    assert cv2.imwrite(str(input_dir / "background.png"), no_face)

    report = run_conformance(
        ofiq_root=root,
        image_dir=input_dir,
        expected_count=1,
        scalar_tolerance=0.0,
        source_root=Path(__file__).resolve().parents[1],
    )

    assert report.passed
    assert report.status_exact == 28
    assert report.raw_observations == 0
    assert report.raw_excluded == 28


def test_gate_validates_port_raw_and_scalar_finiteness() -> None:
    source = inspect.getsource(run_conformance)

    assert "_number(port_result.raw" in source
    assert "_number(port_result.scalar" in source


def test_all_28_bsi_images_are_observed_and_conformant() -> None:
    root = official_root()

    report = run_conformance(
        ofiq_root=root,
        image_dir=root / "data" / "tests" / "images",
        expected_count=28,
        scalar_tolerance=0.0,
        source_root=Path(__file__).resolve().parents[1],
    )

    assert report.passed
    assert report.reference_rows == 28
    assert report.port_rows == 28
    assert report.component_observations == 784
    assert report.scalar_exact == 784
    assert report.scalar_within_tolerance == 784
    assert report.status_exact == 784
    assert report.raw_observations == 784
    assert report.raw_within_tolerance == 784
    assert report.raw_passed
    assert report.raw_exact_at_6_decimals == 702
    assert all(comparison.observations == 28 for comparison in report.components.values())
    assert all(comparison.raw_within_tolerance == comparison.raw_observations for comparison in report.components.values())
    assert all(comparison.raw_passed for comparison in report.components.values())
    assert all(0.0 <= comparison.raw_mean_delta <= comparison.max_raw_delta for comparison in report.components.values())
    assert all(0.0 <= comparison.raw_p95_delta <= comparison.max_raw_delta for comparison in report.components.values())
    assert report.bindings["input_aggregate_sha256"] == "f9dab8561ac543e44e93e12c254b0a7af6b6cdd12d3156f1174b8b984a6c4be5"
    assert report.bindings["model_aggregate_sha256"] == "9a7b6b7943f4c17000bf32eed5be66ad5259f20c6166be4e83bf63db695d8831"
    assert len(str(report.bindings["ofiq_binary_sha256"])) == 64
    assert report.bindings["ofiq_commit"] == "bb5dc91d00477e02ce53d2530d28e35021484393"
    assert len(str(report.bindings["source_commit"])) == 40
    assert len(str(report.bindings["source_tree_sha256"])) == 64
    assert report.bindings["python_version"] == platform.python_version()
    assert report.bindings["python_implementation"] == "CPython"
    assert report.bindings["machine"]
    assert report.bindings["numpy_version"] == "1.26.4"
    assert report.bindings["opencv_version"] == "4.5.5"
    assert report.bindings["onnxruntime_version"] == "1.18.1"
    assert report.bindings["onnxruntime_requested_providers"] == ["CPUExecutionProvider"]
