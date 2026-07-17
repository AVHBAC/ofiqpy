"""Assess a single image and print all component scores.

Usage:
    export OFIQPY_OFIQ_DATA=/path/to/OFIQ-Project/data
    python examples/assess_single.py face.jpg
"""
import sys

from ofiqpy import assess


def main(path):
    scores = assess(path)
    if not scores:
        print("no face detected")
        return
    print(f"{'component':30} {'raw':>12} {'scalar':>7}")
    for name, (raw, scalar) in sorted(scores.items()):
        print(f"{name:30} {raw:12.4f} {scalar:7.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "face.jpg")
