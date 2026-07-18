"""Parallel batch runner — assess a directory of images to an OFIQ-format CSV.

Each worker process builds its own pipeline (ONNX/cv2.ml models are not shared
across processes). Resumable: rows already present in the output CSV are skipped.
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from .output import header, row

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# per-worker singletons
_PIPE = None
_MEAS = None


def _init_worker():
    global _PIPE, _MEAS
    from .config import OFIQConfig
    from .measures.core import Measures
    from .pipeline import OFIQPipeline

    cfg = OFIQConfig()
    _PIPE = OFIQPipeline(cfg)
    _MEAS = Measures(cfg)


def _assess_one(path_str: str) -> str:
    assert _PIPE is not None and _MEAS is not None  # set by _init_worker in each process
    t0 = time.time()
    bgr = cv2.imread(path_str)
    try:
        s = _PIPE.process(bgr)
        res = _MEAS.compute(s) if s.bbox is not None else {}
    except Exception:
        res = {}
    return row(path_str, res, (time.time() - t0) * 1000)


def discover(input_path: Path) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS)


def _already_done(out_csv: Path) -> set[str]:
    if not out_csv.exists():
        return set()
    done = set()
    with open(out_csv, newline="") as f:
        r = csv.reader(f, delimiter=";")
        next(r, None)  # header
        for line in r:
            if line:
                done.add(line[0])
    return done


def run_batch(input_path, output_csv, workers=None, resume=False, progress=True):
    images = discover(Path(input_path))
    out = Path(output_csv)
    done = _already_done(out) if resume else set()
    todo = [im for im in images if im.name not in done]

    write_header = not (resume and out.exists())
    mode = "a" if (resume and out.exists()) else "w"
    n_done = 0
    t0 = time.time()
    with open(out, mode) as fh, ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        if write_header:
            fh.write(header() + "\n")
        for line in ex.map(_assess_one, [str(im) for im in todo], chunksize=1):
            fh.write(line + "\n")
            fh.flush()
            n_done += 1
            if progress and n_done % 25 == 0:
                rate = n_done / max(time.time() - t0, 1e-6)
                print(f"  {n_done}/{len(todo)}  {rate:.1f} img/s", flush=True)
    print(f"done: {n_done} images -> {out} ({len(done)} pre-existing skipped)")
    return out


def main():
    ap = argparse.ArgumentParser(description="ofiqpy parallel batch runner")
    ap.add_argument("-i", "--input", required=True, help="image directory (recursive) or file")
    ap.add_argument("-o", "--output", required=True, help="output CSV")
    ap.add_argument("-w", "--workers", type=int, default=None, help="worker processes (default: CPUs)")
    ap.add_argument("--resume", action="store_true", help="skip images already in the output CSV")
    args = ap.parse_args()
    run_batch(args.input, args.output, args.workers, args.resume)


if __name__ == "__main__":
    main()
