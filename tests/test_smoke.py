"""Installed-style smoke contract using a hash-bound real BSI capture."""

from __future__ import annotations

import csv
from pathlib import Path

from conftest import bsi_image, official_root

from ofiqpy import assess
from ofiqpy.results import OFIQ_COMPONENTS


def test_legacy_adapter_matches_real_bsi_reference_scalars() -> None:
    expected_path = official_root() / "data" / "tests" / "expected_results" / "expected_results.csv"
    with expected_path.open(newline="", encoding="utf-8") as stream:
        expected_rows = list(csv.DictReader(stream, delimiter=";"))
    expected = next(row for row in expected_rows if Path(row["Filename"]).name == "r-01-frontal.png")

    result = assess(bsi_image("r-01-frontal.png"))

    assert tuple(result) == OFIQ_COMPONENTS
    assert all(result[name][1] == float(expected[f"{name}.scalar"]) for name in OFIQ_COMPONENTS)
