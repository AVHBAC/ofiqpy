"""Assess a real image directory with the resumable canonical batch runner.

Run after setting OFIQPY_OFIQ_DATA:
    python examples/batch_directory.py "$OFIQPY_OFIQ_DATA/tests/images" assessments.csv
"""

from __future__ import annotations

import argparse

from ofiqpy.batch import run_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_batch(args.input, args.output, workers=args.workers, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
