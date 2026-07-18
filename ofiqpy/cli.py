"""ofiqpy CLI — OFIQSampleApp-compatible: -i <file|dir> -o <out.csv>."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .config import OFIQConfig
from .measures.core import Measures
from .output import row, write_csv
from .pipeline import OFIQPipeline

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser(description="ofiqpy — faithful OFIQ Python port")
    ap.add_argument("-i", "--input", required=True, help="image file or directory")
    ap.add_argument("-o", "--output", required=True, help="output CSV")
    args = ap.parse_args()

    inp = Path(args.input)
    imgs = [inp] if inp.is_file() else sorted(p for p in inp.rglob("*") if p.suffix.lower() in EXTS)

    cfg = OFIQConfig()
    pipe = OFIQPipeline(cfg)
    meas = Measures(cfg)

    rows = []
    for img in imgs:
        t0 = time.time()
        bgr = cv2.imread(str(img))
        try:
            s = pipe.process(bgr)
            res = meas.compute(s) if s.bbox is not None else {}
        except Exception:
            res = {}
        rows.append(row(str(img), res, (time.time() - t0) * 1000))
    write_csv(args.output, rows)
    print(f"wrote {args.output} ({len(rows)} images)")


if __name__ == "__main__":
    main()
