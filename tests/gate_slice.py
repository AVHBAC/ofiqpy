"""Gate all 28 components against live OFIQ at the ISO Annex A ±1 criterion.

Usage:
    export OFIQPY_OFIQ_ROOT=/path/to/OFIQ-Project    # built OFIQSampleApp
    export OFIQPY_TEST_IMAGES=/path/to/face/images   # directory of .jpg
    python tests/gate_slice.py 100
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_ofiq import gate, print_report, run_ofiq

from ofiqpy.config import OFIQConfig
from ofiqpy.measures.core import Measures
from ofiqpy.output import OFIQ_ORDER
from ofiqpy.pipeline import OFIQPipeline

SRC = Path(os.environ.get("OFIQPY_TEST_IMAGES", "test_images"))
SLICE = OFIQ_ORDER  # all 28 components


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    imgs = sorted(SRC.glob("*.jpg"))[:n]
    cfg = OFIQConfig()
    pipe = OFIQPipeline(cfg)
    meas = Measures(cfg)

    port = {}
    t0 = time.time()
    for i, img in enumerate(imgs):
        bgr = cv2.imread(str(img))
        try:
            s = pipe.process(bgr)
            if s.bbox is None:
                continue
            port[img.name] = meas.compute_scalars(s)
        except Exception as e:
            print(f"  PORT FAIL {img.name}: {type(e).__name__}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  port {i + 1}/{len(imgs)}  {(i + 1) / (time.time() - t0):.1f} img/s", flush=True)
    print(f"port done: {len(port)} images in {time.time() - t0:.0f}s")

    ofiq_df = run_ofiq(imgs)
    report = gate(port, ofiq_df, SLICE, tol=1.0)
    print()
    print_report(report)
    # also show a few raw pairs for the first image for diagnosis
    if port:
        bn = next(iter(port))
        print(f"\nsample {bn}:")
        for c in SLICE:
            pv = port[bn].get(c)
            ov = float(ofiq_df.loc[bn, f"{c}.scalar"]) if bn in ofiq_df.index else None
            print(f"  {c:26} port={pv}  ofiq={ov}  d={abs(pv - ov) if pv is not None and ov is not None else '-'}")


if __name__ == "__main__":
    main()
