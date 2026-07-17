"""Batch-assess a directory of images to an OFIQ-format CSV (parallel, resumable).

Usage:
    export OFIQPY_OFIQ_DATA=/path/to/OFIQ-Project/data
    python examples/batch_directory.py /path/to/images out.csv
"""
import sys

from ofiqpy.batch import run_batch


def main(input_dir, out_csv):
    run_batch(input_dir, out_csv, workers=None, resume=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "out.csv")
