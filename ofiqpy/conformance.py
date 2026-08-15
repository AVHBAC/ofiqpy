"""Cardinality-enforcing conformance gate against live BSI OFIQ v1.1.0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import onnxruntime as ort

from ._version import __version__
from .assessor import Assessor
from .batch import discover
from .config import OFIQConfig
from .output import header_fields
from .profile import CANONICAL_PROFILE, ProfileVerificationError
from .provenance import build_input_provenance, sha256_file
from .raw_policy import CANONICAL_RAW_TOLERANCE_PROFILE
from .results import OFIQ_COMPONENTS, ComponentResult, ComponentStatus


class ConformanceContractError(RuntimeError):
    """The reference or port failed to produce a complete comparable result set."""


@dataclass(frozen=True)
class ComponentComparison:
    observations: int
    scalar_exact: int
    scalar_within_tolerance: int
    status_exact: int
    raw_observations: int
    raw_exact_at_6_decimals: int
    raw_within_tolerance: int
    raw_tolerance: float
    raw_mean_delta: float
    raw_p95_delta: float
    max_scalar_delta: float
    max_raw_delta: float
    raw_passed: bool
    passed: bool


@dataclass(frozen=True)
class ConformanceReport:
    passed: bool
    expected_rows: int
    reference_rows: int
    port_rows: int
    scalar_tolerance: float
    component_observations: int
    scalar_exact: int
    scalar_within_tolerance: int
    status_exact: int
    raw_observations: int
    raw_excluded: int
    raw_exact_at_6_decimals: int
    raw_tolerance_profile_id: str
    raw_within_tolerance: int
    raw_passed: bool
    components: Mapping[str, ComponentComparison]
    bindings: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _aggregate_hash(paths: list[Path], base: Path) -> str:
    return build_input_provenance(paths, base).aggregate_sha256


def _model_aggregate_hash() -> str:
    digest = hashlib.sha256()
    for artifact in sorted(CANONICAL_PROFILE.artifacts, key=lambda item: item.relative_path):
        digest.update(f"{artifact.sha256}  {artifact.relative_path}\n".encode("utf-8"))
    return digest.hexdigest()


def _assert_stable_artifacts(before: Mapping[str, object], after: Mapping[str, object]) -> None:
    """Reject a run if any bound source, input, native, model, or distribution bytes changed."""
    changed = sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
    if changed:
        raise ConformanceContractError(f"bound artifacts changed during conformance execution: {changed}")


def _git_value(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_tree_hash(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        check=True,
    )
    relative_paths = sorted(item for item in completed.stdout.split(b"\0") if item)
    digest = hashlib.sha256()
    for encoded_path in relative_paths:
        relative = encoded_path.decode("utf-8")
        path = root / relative
        if not path.is_file():
            continue
        digest.update(encoded_path)
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _absolute_identity(value: str | Path, cwd: Path | None = None) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = (cwd or Path.cwd()) / path
    return os.path.abspath(path)


def _run_reference(ofiq_root: Path, image_dir: Path) -> list[dict[str, str]]:
    binary = ofiq_root / "install_x86_64_linux" / "Release" / "bin" / "OFIQSampleApp"
    library_dir = ofiq_root / "install_x86_64_linux" / "Release" / "lib"
    if not binary.is_file():
        raise ConformanceContractError(f"OFIQSampleApp is unavailable: {binary}")
    env = os.environ.copy()
    library_path = f"{library_dir}:{binary.parent}"
    if env.get("LD_LIBRARY_PATH"):
        library_path = f"{library_path}:{env['LD_LIBRARY_PATH']}"
    env["LD_LIBRARY_PATH"] = library_path

    with tempfile.TemporaryDirectory(prefix="ofiqpy-conformance-") as temp_dir:
        output = Path(temp_dir) / "ofiq.csv"
        completed = subprocess.run(
            [
                str(binary),
                "-c",
                str(ofiq_root / "data"),
                "-cf",
                "ofiq_config.jaxn",
                "-i",
                str(image_dir),
                "-o",
                str(output),
            ],
            cwd=ofiq_root,
            capture_output=True,
            check=False,
            env=env,
            text=True,
        )
        if completed.returncode != 0:
            raise ConformanceContractError(f"OFIQSampleApp exited {completed.returncode}: {completed.stderr.strip()}")
        with output.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter=";")
            if reader.fieldnames != header_fields():
                raise ConformanceContractError("OFIQSampleApp emitted an unexpected CSV header")
            return [dict(row) for row in reader]


def _index_reference(rows: list[dict[str, str]], cwd: Path) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        filename = row.get("Filename")
        if not filename:
            raise ConformanceContractError("OFIQSampleApp emitted a row without an image identity")
        identity = _absolute_identity(filename, cwd)
        if identity in indexed:
            raise ConformanceContractError(f"OFIQSampleApp emitted duplicate image identity: {identity}")
        indexed[identity] = row
    return indexed


def _run_port(images: list[Path], data_root: Path) -> dict[str, Mapping[str, ComponentResult]]:
    assessor = Assessor(OFIQConfig(data_root=data_root))
    results: dict[str, Mapping[str, ComponentResult]] = {}
    for image in images:
        identity = _absolute_identity(image)
        assessment = assessor.assess(image)
        if tuple(assessment.components) != OFIQ_COMPONENTS:
            raise ConformanceContractError(f"port component cardinality/order mismatch for {identity}")
        if identity in results:
            raise ConformanceContractError(f"port produced duplicate image identity: {identity}")
        results[identity] = assessment.components
    return results


def _number(value: str | float, identity: str, component: str, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConformanceContractError(f"non-numeric {field} for {identity} / {component}: {value!r}") from exc
    if not math.isfinite(number):
        raise ConformanceContractError(f"non-finite {field} for {identity} / {component}: {value!r}")
    return number


def _raw_ticks(value: str, decimal_places: int) -> int:
    scale = 10**decimal_places
    scaled = Decimal(value) * scale
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise ConformanceContractError(f"raw value exceeds the declared {decimal_places}-decimal comparison precision: {value}")
    return int(integral)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _artifact_bindings(
    ofiq_root: Path,
    image_dir: Path,
    images: list[Path],
    source_root: Path,
    distribution: Path | None,
) -> dict[str, object]:
    binary_dir = ofiq_root / "install_x86_64_linux" / "Release"
    try:
        CANONICAL_PROFILE.verify(ofiq_root / "data")
    except (OSError, ProfileVerificationError) as exc:
        raise ConformanceContractError(f"canonical OFIQ profile verification failed: {exc}") from exc
    bindings: dict[str, object] = {
        "profile_id": CANONICAL_PROFILE.profile_id,
        "ofiq_commit": _git_value(ofiq_root, "rev-parse", "HEAD"),
        "ofiq_binary_sha256": _sha256(binary_dir / "bin" / "OFIQSampleApp"),
        "ofiq_library_sha256": _sha256(binary_dir / "lib" / "libofiq_lib.so"),
        "onnxruntime_library_sha256": _sha256(binary_dir / "lib" / "libonnxruntime.so.1.18.1"),
        "config_sha256": _sha256(ofiq_root / "data" / "ofiq_config.jaxn"),
        "model_aggregate_sha256": _model_aggregate_hash(),
        "input_aggregate_sha256": _aggregate_hash(images, image_dir),
        "input_count": len(images),
        "source_commit": _git_value(source_root, "rev-parse", "HEAD"),
        "source_dirty": bool(_git_value(source_root, "status", "--porcelain")),
        "source_tree_sha256": _source_tree_hash(source_root),
    }
    if distribution is not None:
        if not distribution.is_file():
            raise ConformanceContractError(f"distribution artifact is unavailable: {distribution}")
        bindings["distribution_sha256"] = _sha256(distribution)
    return bindings


def _runtime_bindings() -> dict[str, object]:
    thread_environment = {
        name: os.environ[name]
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FOR_THREADS_NUM")
        if name in os.environ
    }
    return {
        "ofiqpy_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "opencv_optimized": cv2.useOptimized(),
        "onnxruntime_version": ort.__version__,
        "onnxruntime_available_providers": ort.get_available_providers(),
        "onnxruntime_requested_providers": ["CPUExecutionProvider"],
        "thread_environment": thread_environment,
        "raw_comparison_policy": (
            f"{CANONICAL_RAW_TOLERANCE_PROFILE.profile_id}; port and reference compared as "
            f"OFIQSampleApp values at {CANONICAL_RAW_TOLERANCE_PROFILE.decimal_places} decimal places"
        ),
        "reference_status_source": "OFIQSampleApp FailureToAssess scalar sentinel (-1)",
    }


def run_conformance(
    *,
    ofiq_root: Path,
    image_dir: Path,
    expected_count: int,
    scalar_tolerance: float,
    source_root: Path | None = None,
    distribution: Path | None = None,
) -> ConformanceReport:
    """Run live OFIQ and ofiqpy with strict row/component/status cardinality."""
    if expected_count < 1:
        raise ValueError("expected_count must be greater than zero")
    if scalar_tolerance < 0:
        raise ValueError("scalar_tolerance must not be negative")
    ofiq_root = Path(ofiq_root).absolute()
    image_dir = Path(image_dir).absolute()
    source_root = Path(source_root or Path(__file__).resolve().parents[1]).absolute()
    distribution = Path(distribution).absolute() if distribution is not None else None
    images = discover(image_dir)
    if len(images) != expected_count:
        raise ConformanceContractError(f"input cardinality mismatch: expected {expected_count}, discovered {len(images)}")
    expected_identities = {_absolute_identity(image) for image in images}
    if len(expected_identities) != expected_count:
        raise ConformanceContractError("input identities are not unique")

    artifact_bindings = _artifact_bindings(ofiq_root, image_dir, images, source_root, distribution)

    reference_rows = _run_reference(ofiq_root, image_dir)
    reference = _index_reference(reference_rows, ofiq_root)
    port = _run_port(images, ofiq_root / "data")
    if set(reference) != expected_identities:
        missing = sorted(expected_identities - set(reference))
        unexpected = sorted(set(reference) - expected_identities)
        raise ConformanceContractError(f"reference identity mismatch: missing={missing}, unexpected={unexpected}")
    if set(port) != expected_identities:
        missing = sorted(expected_identities - set(port))
        unexpected = sorted(set(port) - expected_identities)
        raise ConformanceContractError(f"port identity mismatch: missing={missing}, unexpected={unexpected}")

    comparisons: dict[str, ComponentComparison] = {}
    raw_places = CANONICAL_RAW_TOLERANCE_PROFILE.decimal_places
    raw_scale = 10**raw_places
    for component in OFIQ_COMPONENTS:
        policy = CANONICAL_RAW_TOLERANCE_PROFILE.components[component]
        tolerance_ticks = _raw_ticks(f"{policy.absolute_tolerance:.{raw_places}f}", raw_places)
        scalar_exact = scalar_within = status_exact = raw_observations = raw_exact = raw_within = 0
        raw_deltas: list[float] = []
        max_scalar_delta = max_raw_delta = 0.0
        for identity in sorted(expected_identities):
            reference_scalar = _number(reference[identity][f"{component}.scalar"], identity, component, "scalar")
            port_result = port[identity][component]
            port_scalar = _number(port_result.scalar, identity, component, "port scalar")
            port_success = port_result.status is ComponentStatus.SUCCESS
            reference_success = reference_scalar != -1.0
            reference_raw_text = reference[identity][component] if reference_success else None
            reference_raw = _number(reference_raw_text, identity, component, "raw") if reference_raw_text is not None else None
            port_raw = _number(port_result.raw, identity, component, "port raw") if port_success else None
            status_exact += port_success == reference_success
            scalar_delta = abs(port_scalar - reference_scalar)
            scalar_exact += scalar_delta == 0.0
            scalar_within += scalar_delta <= scalar_tolerance
            max_scalar_delta = max(max_scalar_delta, scalar_delta)
            if reference_raw is not None and reference_raw_text is not None and port_raw is not None:
                port_reported = f"{port_raw:.{raw_places}f}"
                reference_ticks = _raw_ticks(reference_raw_text, raw_places)
                port_ticks = _raw_ticks(port_reported, raw_places)
                delta_ticks = abs(port_ticks - reference_ticks)
                raw_delta = delta_ticks / raw_scale
                raw_observations += 1
                raw_exact += delta_ticks == 0
                raw_within += delta_ticks <= tolerance_ticks
                raw_deltas.append(raw_delta)
                max_raw_delta = max(max_raw_delta, raw_delta)
        raw_passed = raw_within == raw_observations
        passed = scalar_within == expected_count and status_exact == expected_count and raw_passed
        comparisons[component] = ComponentComparison(
            observations=expected_count,
            scalar_exact=scalar_exact,
            scalar_within_tolerance=scalar_within,
            status_exact=status_exact,
            raw_observations=raw_observations,
            raw_exact_at_6_decimals=raw_exact,
            raw_within_tolerance=raw_within,
            raw_tolerance=policy.absolute_tolerance,
            raw_mean_delta=sum(raw_deltas) / len(raw_deltas) if raw_deltas else 0.0,
            raw_p95_delta=_p95(raw_deltas),
            max_scalar_delta=max_scalar_delta,
            max_raw_delta=max_raw_delta,
            raw_passed=raw_passed,
            passed=passed,
        )

    component_observations = expected_count * len(OFIQ_COMPONENTS)
    raw_observations = sum(item.raw_observations for item in comparisons.values())
    final_artifact_bindings = _artifact_bindings(ofiq_root, image_dir, images, source_root, distribution)
    _assert_stable_artifacts(artifact_bindings, final_artifact_bindings)
    return ConformanceReport(
        passed=all(comparison.passed for comparison in comparisons.values()),
        expected_rows=expected_count,
        reference_rows=len(reference),
        port_rows=len(port),
        scalar_tolerance=scalar_tolerance,
        component_observations=component_observations,
        scalar_exact=sum(item.scalar_exact for item in comparisons.values()),
        scalar_within_tolerance=sum(item.scalar_within_tolerance for item in comparisons.values()),
        status_exact=sum(item.status_exact for item in comparisons.values()),
        raw_observations=raw_observations,
        raw_excluded=component_observations - raw_observations,
        raw_exact_at_6_decimals=sum(item.raw_exact_at_6_decimals for item in comparisons.values()),
        raw_tolerance_profile_id=CANONICAL_RAW_TOLERANCE_PROFILE.profile_id,
        raw_within_tolerance=sum(item.raw_within_tolerance for item in comparisons.values()),
        raw_passed=all(item.raw_passed for item in comparisons.values()),
        components=comparisons,
        bindings={**artifact_bindings, **_runtime_bindings()},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="strict live-OFIQ conformance gate")
    parser.add_argument("--ofiq-root", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--scalar-tolerance", type=float, default=1.0)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--distribution", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = run_conformance(
        ofiq_root=args.ofiq_root,
        image_dir=args.images,
        expected_count=args.expected_count,
        scalar_tolerance=args.scalar_tolerance,
        source_root=args.source_root,
        distribution=args.distribution,
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    print(rendered)
    if args.report is not None:
        with args.report.open("w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.write("\n")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
