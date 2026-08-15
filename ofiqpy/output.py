"""Canonical-profile CSV output with OFIQSampleApp's field order.

Semicolon-delimited: Filename; <raw per component in OFIQ order>; <.scalar per
component>; assessment_time_in_ms. Failed components emit the OFIQ
FailureToAssess sentinel (raw 0, scalar -1).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from .results import OFIQ_COMPONENTS, AssessmentResult

# OFIQ CSV component order (from OFIQSampleApp header).
OFIQ_ORDER = list(OFIQ_COMPONENTS)

FAILURE = (0.0, -1.0)  # OFIQ FailureToAssess sentinel


def _encode(fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    csv.writer(stream, delimiter=";", lineterminator="").writerow(fields)
    return stream.getvalue()


def header_fields() -> list[str]:
    return ["Filename"] + OFIQ_ORDER + [f"{c}.scalar" for c in OFIQ_ORDER] + ["assessment_time_in_ms", ""]


def header() -> str:
    return _encode(header_fields())


def row(filename: str, results: dict | AssessmentResult, time_ms: float) -> str:
    """results: {component: (raw, scalar)}. Missing components -> FailureToAssess."""
    values = results.as_legacy_dict() if isinstance(results, AssessmentResult) else results
    raws, scalars = [], []
    for c in OFIQ_ORDER:
        raw, scalar = values.get(c, FAILURE)
        raws.append(f"{raw:.6f}" if raw is not None else "nan")
        scalars.append(str(int(scalar)) if scalar is not None else "-1")
    parts = [filename] + raws + scalars + [f"{time_ms:.0f}"]
    return _encode(parts)


def write_csv(path, rows: list[str]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        stream.write(header())
        stream.write("\n")
        for encoded in rows:
            stream.write(encoded)
            stream.write("\n")
