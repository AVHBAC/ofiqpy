"""CSV, recursive identity, resume, and CLI contracts using real BSI results."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import bsi_image, official_root

from ofiqpy.batch import BatchAssessmentError, BatchProgress, BatchReport, _already_done, discover, run_batch
from ofiqpy.output import header, row, write_csv
from ofiqpy.results import OFIQ_COMPONENTS


def _official_result(image_name: str) -> tuple[dict[str, tuple[float, float]], float]:
    expected = official_root() / "data" / "tests" / "expected_results" / "expected_results.csv"
    started = time.perf_counter_ns()
    with expected.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    selected = next(item for item in rows if Path(item["Filename"]).name == image_name)
    results = {component: (float(selected[component]), float(selected[f"{component}.scalar"])) for component in OFIQ_COMPONENTS}
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return results, elapsed_ms


def test_csv_round_trip_preserves_full_semicolon_path_and_real_scores(tmp_path: Path) -> None:
    source = bsi_image("r-01-frontal.png")
    cohort = tmp_path / "cohort;alpha"
    cohort.mkdir()
    capture = cohort / source.name
    capture.symlink_to(source)
    results, elapsed_ms = _official_result(source.name)

    encoded = row(str(capture), results, elapsed_ms)
    parsed = next(csv.reader([encoded], delimiter=";"))

    assert parsed[0] == str(capture)
    assert len(parsed) == 1 + 2 * len(OFIQ_COMPONENTS) + 1
    assert header().endswith(";")


def test_recursive_duplicate_basenames_keep_distinct_resume_identities(tmp_path: Path) -> None:
    first = tmp_path / "first" / "capture.png"
    second = tmp_path / "second" / "capture.png"
    first.parent.mkdir()
    second.parent.mkdir()
    first.symlink_to(bsi_image("r-01-frontal.png"))
    second.symlink_to(bsi_image("r-02-rolled.png"))
    discovered = discover(tmp_path)
    first_results, first_elapsed_ms = _official_result("r-01-frontal.png")
    second_results, second_elapsed_ms = _official_result("r-02-rolled.png")
    output = tmp_path / "assessments.csv"
    write_csv(
        output,
        [
            row(str(first), first_results, first_elapsed_ms),
            row(str(second), second_results, second_elapsed_ms),
        ],
    )

    assert discovered == [first, second]
    assert _already_done(output) == {str(first), str(second)}


def test_resume_rejects_real_conformance_table_with_wrong_contract() -> None:
    conformance_table = official_root() / "data" / "tests" / "expected_results" / "expected_results.csv"

    with pytest.raises(ValueError, match="header"):
        _already_done(conformance_table)


def test_nonexistent_cli_input_fails_before_output_creation(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["OFIQPY_OFIQ_DATA"] = str(official_root() / "data")
    completed = subprocess.run(
        [sys.executable, "-m", "ofiqpy.cli", "-i", str(tmp_path / "absent.png"), "-o", str(output)],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert completed.returncode != 0
    assert not output.exists()


def test_batch_defaults_to_one_worker() -> None:
    assert run_batch.__defaults__ is not None
    assert run_batch.__defaults__[0] == 1


def test_library_batch_is_silent_and_reports_real_aggregate_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = bsi_image("r-01-frontal.png")
    output = tmp_path / "single-real-capture.csv"
    progress_events = []

    report = run_batch(source, output, progress=progress_events.append)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert isinstance(report, BatchReport)
    assert report.output_csv == output
    assert report.discovered == 1
    assert report.assessed == 1
    assert report.skipped == 0
    assert report.provenance.input_count == 1
    assert report.provenance.total_bytes == source.stat().st_size
    assert len(report.provenance.aggregate_sha256) == 64
    assert report.provenance.records is None
    assert len(progress_events) == 1
    assert progress_events[0].processed == 1
    assert progress_events[0].total == 1


def test_per_image_provenance_is_explicit_for_a_real_capture(tmp_path: Path) -> None:
    source = bsi_image("r-01-frontal.png")
    output = tmp_path / "explicit-per-image.csv"

    report = run_batch(source, output, progress=None, include_per_image_provenance=True)

    assert report.provenance.records is not None
    assert len(report.provenance.records) == 1
    record = report.provenance.records[0]
    assert record.identity == source.name
    assert record.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()


def test_resume_rejects_truncated_row_derived_from_real_bsi_result(tmp_path: Path) -> None:
    source = bsi_image("r-01-frontal.png")
    results, elapsed_ms = _official_result(source.name)
    complete_fields = next(csv.reader([row(str(source), results, elapsed_ms)], delimiter=";"))
    output = tmp_path / "truncated-real-result.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";", lineterminator="\n")
        writer.writerow(next(csv.reader([header()], delimiter=";")))
        writer.writerow(complete_fields[:-1])

    with pytest.raises(ValueError, match="field count"):
        _already_done(output)


def test_resume_report_counts_only_discovered_real_images_as_skipped(tmp_path: Path) -> None:
    prior = bsi_image("r-01-frontal.png")
    current = bsi_image("r-02-rolled.png")
    prior_results, elapsed_ms = _official_result(prior.name)
    output = tmp_path / "different-prior-input.csv"
    write_csv(output, [row(str(prior), prior_results, elapsed_ms)])

    report = run_batch(current, output, resume=True, progress=None)

    assert report.discovered == 1
    assert report.assessed == 1
    assert report.skipped == 0


def test_cli_rejects_real_non_image_source_with_image_suffix(tmp_path: Path) -> None:
    mislabeled_source = tmp_path / "image_utils.png"
    mislabeled_source.symlink_to(official_root() / "OFIQlib" / "modules" / "utils" / "src" / "image_utils.cpp")
    output = tmp_path / "must-not-exist.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["OFIQPY_OFIQ_DATA"] = str(official_root() / "data")

    completed = subprocess.run(
        [sys.executable, "-m", "ofiqpy.cli", "-i", str(mislabeled_source), "-o", str(output)],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert completed.returncode != 0
    assert "image_read_error" in completed.stderr
    assert not output.exists()


def test_cli_writes_aggregate_only_report_for_real_capture(tmp_path: Path) -> None:
    source = bsi_image("r-01-frontal.png")
    output = tmp_path / "cli.csv"
    report_path = tmp_path / "cli.provenance.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["OFIQPY_OFIQ_DATA"] = str(official_root() / "data")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ofiqpy.cli",
            "-i",
            str(source),
            "-o",
            str(output),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["input_count"] == 1
    assert payload["provenance"]["records"] is None
    assert len(payload["provenance"]["aggregate_sha256"]) == 64


def test_batch_rejects_real_non_image_source_instead_of_writing_sentinel(tmp_path: Path) -> None:
    mislabeled_source = tmp_path / "image_utils.png"
    mislabeled_source.symlink_to(official_root() / "OFIQlib" / "modules" / "utils" / "src" / "image_utils.cpp")
    output = tmp_path / "must-not-exist.csv"

    with pytest.raises(BatchAssessmentError, match="image_read_error"):
        run_batch(mislabeled_source, output, progress=False)

    assert not output.exists()


def test_spawn_batch_processes_two_real_bsi_captures(tmp_path: Path) -> None:
    source = inspect.getsource(run_batch)
    assert 'get_context("spawn")' in source

    cohort = tmp_path / "real-bsi-cohort"
    cohort.mkdir()
    for name in ("r-01-frontal.png", "r-02-rolled.png"):
        (cohort / name).symlink_to(bsi_image(name))
    output = tmp_path / "multiprocess.csv"

    run_batch(cohort, output, workers=2, progress=False)

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    assert [Path(item["Filename"]).name for item in rows] == ["r-01-frontal.png", "r-02-rolled.png"]
    assert all(float(item["UnifiedQualityScore.scalar"]) >= 0 for item in rows)


def test_single_worker_batch_is_reentrant_on_real_bsi_captures(tmp_path: Path) -> None:
    outer_input = tmp_path / "outer-real-bsi"
    outer_input.mkdir()
    for name in ("r-01-frontal.png", "r-02-rolled.png"):
        (outer_input / name).symlink_to(bsi_image(name))
    outer_output = tmp_path / "outer.csv"
    inner_output = tmp_path / "inner.csv"
    nested_reports: list[BatchReport] = []

    def assess_nested_batch(progress: BatchProgress) -> None:
        if progress.processed == 1:
            nested_reports.append(run_batch(bsi_image("r-01-frontal.png"), inner_output, progress=None))

    outer_report = run_batch(outer_input, outer_output, progress=assess_nested_batch)

    assert outer_report.assessed == 2
    assert len(nested_reports) == 1
    assert nested_reports[0].assessed == 1
