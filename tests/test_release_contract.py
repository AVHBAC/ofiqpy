"""Release identity and source-level OFIQ v1.1.0 contracts."""

from __future__ import annotations

import re
from pathlib import Path

from conftest import official_root

import ofiqpy
from ofiqpy.measures import pixel
from ofiqpy.results import AssessmentStatus


def test_runtime_version_matches_reviewed_release() -> None:
    assert ofiqpy.__version__ == "0.2.0"


def test_citation_version_matches_runtime_release() -> None:
    citation = (Path(__file__).resolve().parents[1] / "CITATION.cff").read_text(encoding="utf-8")

    assert "version: 0.2.0" in citation


def test_top_level_typed_status_export_is_available() -> None:
    assert ofiqpy.AssessmentStatus is AssessmentStatus


def test_natural_colour_lab_coefficient_matches_ofiq_v1_1_0_source() -> None:
    source = (official_root() / "OFIQlib" / "modules" / "utils" / "src" / "image_utils.cpp").read_text(encoding="utf-8")
    match = re.search(r"double k = (\d+) / 27\.0;", source)
    assert match is not None
    assert pixel.LAB_K == float(match.group(1)) / 27.0


def test_pyproject_reads_version_from_runtime_source() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in text
    assert 'version = {attr = "ofiqpy._version.__version__"}' in text


def test_pyproject_uses_current_spdx_license_metadata() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert 'license = "MIT"' in text
    assert "License :: OSI Approved :: MIT License" not in text


def test_source_distribution_manifest_includes_the_dependency_lock() -> None:
    manifest = Path(__file__).resolve().parents[1] / "MANIFEST.in"

    assert "include uv.lock" in manifest.read_text(encoding="utf-8").splitlines()
    assert "recursive-include docs *.md *.json" in manifest.read_text(encoding="utf-8").splitlines()
