"""Canonical-profile CLI with OFIQSampleApp-style input/output flags."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .batch import BatchAssessmentError, BatchProgress, run_batch, write_report

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _console_progress(progress: BatchProgress) -> None:
    if progress.processed % 25 == 0:
        print(f"  {progress.processed}/{progress.total}  {progress.images_per_second:.1f} img/s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="ofiqpy canonical OFIQ v1.1.0 profile")
    ap.add_argument("-i", "--input", required=True, help="image file or directory")
    ap.add_argument("-o", "--output", required=True, help="output CSV")
    ap.add_argument("--report", type=Path, help="write an aggregate provenance and execution report")
    ap.add_argument(
        "--per-image-provenance",
        action="store_true",
        help="include per-image identities and hashes in --report (explicit privacy opt-in)",
    )
    args = ap.parse_args()
    if args.per_image_provenance and args.report is None:
        ap.error("--per-image-provenance requires --report")

    try:
        report = run_batch(
            args.input,
            args.output,
            workers=1,
            resume=False,
            progress=_console_progress,
            include_per_image_provenance=args.per_image_provenance,
        )
    except (FileNotFoundError, ValueError) as exc:
        ap.error(str(exc))
    except BatchAssessmentError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.report is not None:
        write_report(args.report, report)
    print(f"wrote {args.output} ({report.assessed} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
