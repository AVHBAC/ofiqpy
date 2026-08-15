"""Parallel batch runner — assess a directory of images to an OFIQ-format CSV.

Each worker process builds its own pipeline (ONNX/cv2.ml models are not shared
across processes). Resumable: rows already present in the output CSV are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import chain
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Callable

from .assessor import Assessor
from .config import OFIQConfig
from .output import header, header_fields, row
from .provenance import InputProvenance, build_input_provenance
from .results import AssessmentStatus, FailureCode

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# per-worker singletons
_ASSESSOR = None


class BatchAssessmentError(RuntimeError):
    """A batch item failed before it could produce a valid OFIQ assessment row."""


@dataclass(frozen=True, slots=True)
class BatchProgress:
    processed: int
    total: int
    skipped: int
    elapsed_seconds: float
    images_per_second: float


@dataclass(frozen=True, slots=True)
class BatchReport:
    schema: str
    output_csv: Path
    discovered: int
    assessed: int
    skipped: int
    successful: int
    partial: int
    failure_to_assess: int
    elapsed_seconds: float
    provenance: InputProvenance

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_csv"] = str(self.output_csv)
        return payload


@dataclass(frozen=True, slots=True)
class _EncodedAssessment:
    line: str
    status: AssessmentStatus


def _init_worker():
    global _ASSESSOR
    _ASSESSOR = Assessor()


def _assess_with(assessor: Assessor, path_str: str) -> _EncodedAssessment:
    started = perf_counter()
    result = assessor.assess(path_str)
    if result.failure is not None and result.failure.code is not FailureCode.NO_FACE:
        raise BatchAssessmentError(f"{path_str}: {result.failure.code.value}: {result.failure.message}")
    return _EncodedAssessment(row(path_str, result, (perf_counter() - started) * 1000), result.status)


def _assess_one(path_str: str) -> _EncodedAssessment:
    if _ASSESSOR is None:
        raise RuntimeError("batch worker assessor was not initialized")
    return _assess_with(_ASSESSOR, path_str)


def discover(input_path: Path) -> list[Path]:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"input does not exist: {p}")
    if p.is_file():
        if p.suffix.lower() not in EXTS:
            raise ValueError(f"unsupported image extension: {p}")
        return [p]
    images = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in EXTS)
    if not images:
        raise ValueError(f"input directory contains no supported images: {p}")
    return images


def _already_done(out_csv: Path) -> set[str]:
    if not out_csv.exists():
        return set()
    done = set()
    expected_header = header_fields()
    with open(out_csv, newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter=";")
        actual_header = next(r, None)
        if actual_header != expected_header:
            raise ValueError(f"resume CSV header does not match the canonical output contract: {out_csv}")
        for line in r:
            if line:
                expected_fields = len(expected_header) - 1
                if len(line) != expected_fields:
                    raise ValueError(
                        f"resume CSV row field count does not match the canonical output contract: "
                        f"expected {expected_fields}, got {len(line)}"
                    )
                identity = line[0]
                if not identity:
                    raise ValueError("resume CSV contains an empty image identity")
                if identity in done:
                    raise ValueError(f"resume CSV contains a duplicate image identity: {identity}")
                done.add(identity)
    return done


def run_batch(
    input_path,
    output_csv,
    workers=1,
    resume=False,
    progress: Callable[[BatchProgress], None] | bool | None = None,
    include_per_image_provenance: bool = False,
) -> BatchReport:
    if workers is None or workers < 1:
        raise ValueError("workers must be an integer greater than or equal to one")
    source = Path(input_path)
    images = discover(source)
    progress_callback: Callable[[BatchProgress], None] | None
    if progress is True:
        progress_callback = _console_progress
    elif progress is False:
        progress_callback = None
    else:
        progress_callback = progress
    out = Path(output_csv)
    done = _already_done(out) if resume else set()
    todo = [im for im in images if str(im) not in done]
    skipped = len(images) - len(todo)
    provenance_base = source if source.is_dir() else source.parent
    provenance = build_input_provenance(images, provenance_base, include_records=include_per_image_provenance)

    write_header = not (resume and out.exists())
    mode = "a" if (resume and out.exists()) else "w"
    assessed = successful = partial = failure_to_assess = 0
    started = perf_counter()
    if not todo:
        return BatchReport(
            schema="ofiqpy.batch-report.v1",
            output_csv=out,
            discovered=len(images),
            assessed=0,
            skipped=skipped,
            successful=0,
            partial=0,
            failure_to_assess=0,
            elapsed_seconds=perf_counter() - started,
            provenance=provenance,
        )

    executor = None
    encoded_rows: Iterator[_EncodedAssessment]
    if workers == 1:
        assessor = Assessor()
        encoded_rows = (_assess_with(assessor, str(image)) for image in todo)
    else:
        OFIQConfig()
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            mp_context=get_context("spawn"),
        )
        encoded_rows = executor.map(_assess_one, [str(im) for im in todo], chunksize=1)

    try:
        first_result = next(encoded_rows)
        with open(out, mode, encoding="utf-8", newline="") as fh:
            if write_header:
                fh.write(header() + "\n")
            for encoded in chain((first_result,), encoded_rows):
                fh.write(encoded.line + "\n")
                fh.flush()
                assessed += 1
                successful += encoded.status is AssessmentStatus.SUCCESS
                partial += encoded.status is AssessmentStatus.PARTIAL
                failure_to_assess += encoded.status is AssessmentStatus.FAILURE_TO_ASSESS
                if progress_callback is not None:
                    elapsed = perf_counter() - started
                    progress_callback(
                        BatchProgress(
                            processed=assessed,
                            total=len(todo),
                            skipped=skipped,
                            elapsed_seconds=elapsed,
                            images_per_second=assessed / max(elapsed, 1e-9),
                        )
                    )
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    return BatchReport(
        schema="ofiqpy.batch-report.v1",
        output_csv=out,
        discovered=len(images),
        assessed=assessed,
        skipped=skipped,
        successful=successful,
        partial=partial,
        failure_to_assess=failure_to_assess,
        elapsed_seconds=perf_counter() - started,
        provenance=provenance,
    )


def write_report(path: Path, report: BatchReport) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(report.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _console_progress(progress: BatchProgress) -> None:
    if progress.processed % 25 == 0 or progress.processed == progress.total:
        print(
            f"  {progress.processed}/{progress.total}  {progress.images_per_second:.1f} img/s",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser(description="ofiqpy parallel batch runner")
    ap.add_argument("-i", "--input", required=True, help="image directory (recursive) or file")
    ap.add_argument("-o", "--output", required=True, help="output CSV")
    ap.add_argument("-w", "--workers", type=int, default=1, help="worker processes (default: 1; override explicitly)")
    ap.add_argument("--resume", action="store_true", help="skip images already in the output CSV")
    ap.add_argument("--report", type=Path, help="write an aggregate provenance and execution report")
    ap.add_argument(
        "--per-image-provenance",
        action="store_true",
        help="include per-image identities and hashes in --report (explicit privacy opt-in)",
    )
    args = ap.parse_args()
    if args.per_image_provenance and args.report is None:
        ap.error("--per-image-provenance requires --report")
    report = run_batch(
        args.input,
        args.output,
        args.workers,
        args.resume,
        progress=_console_progress,
        include_per_image_provenance=args.per_image_provenance,
    )
    if args.report is not None:
        write_report(args.report, report)
    print(f"done: {report.assessed} images -> {report.output_csv} ({report.skipped} pre-existing skipped)")


if __name__ == "__main__":
    main()
