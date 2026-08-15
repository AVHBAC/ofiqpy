"""Assess one real image and print typed component results.

Run after setting OFIQPY_OFIQ_DATA:
    python examples/assess_single.py "$OFIQPY_OFIQ_DATA/tests/images/r-01-frontal.png"
"""

from __future__ import annotations

import argparse

from ofiqpy import Assessor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()
    result = Assessor().assess(args.image)
    print(result.status.value)
    if result.failure is not None:
        print(result.failure.code.value, result.failure.message)
    for name, component in result.components.items():
        print(f"{name:30} {component.status.value:18} {component.raw:12.6f} {component.scalar:3.0f}")
    return 0 if result.failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
