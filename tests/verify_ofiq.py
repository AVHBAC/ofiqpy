"""Verification harness: run live OFIQ v1.1.0 and gate the port at ISO Annex A ±1.

Runs OFIQSampleApp on a set of real images, parses its semicolon CSV into raw +
scalar per component, and compares against port output. Pass = |port_scalar -
ofiq_scalar| <= 1 for every image (the ISO/IEC 29794-5 Annex A.2 criterion).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Reference OFIQ install (for the ±1 gate). Override via OFIQPY_OFIQ_ROOT.
OFIQ_ROOT = Path(os.environ.get("OFIQPY_OFIQ_ROOT", "OFIQ-Project"))
OFIQ_BIN = OFIQ_ROOT / "install_x86_64_linux/Release/bin/OFIQSampleApp"
OFIQ_LIB = OFIQ_ROOT / "install_x86_64_linux/Release/lib"
OFIQ_DATA = OFIQ_ROOT / "data"

# 27 quality components (raw column names in OFIQ CSV; each also has a .scalar column).
COMPONENTS = [
    "BackgroundUniformity", "IlluminationUniformity", "LuminanceMean", "LuminanceVariance",
    "UnderExposurePrevention", "OverExposurePrevention", "DynamicRange", "Sharpness",
    "CompressionArtifacts", "NaturalColour", "SingleFacePresent", "EyesOpen", "MouthClosed",
    "EyesVisible", "MouthOcclusionPrevention", "FaceOcclusionPrevention", "InterEyeDistance",
    "HeadSize", "LeftwardCropOfTheFaceImage", "RightwardCropOfTheFaceImage",
    "MarginAboveOfTheFaceImage", "MarginBelowOfTheFaceImage", "HeadPoseYaw", "HeadPosePitch",
    "HeadPoseRoll", "ExpressionNeutrality", "NoHeadCoverings", "UnifiedQualityScore",
]


def run_ofiq(images: list[Path], config_file: str = "ofiq_config.jaxn") -> pd.DataFrame:
    """Run OFIQSampleApp on the given images; return DataFrame indexed by basename."""
    with tempfile.TemporaryDirectory() as td:
        indir = Path(td) / "in"
        indir.mkdir()
        for img in images:
            (indir / Path(img).name).symlink_to(Path(img).resolve())
        out_csv = Path(td) / "out.csv"
        env = {"LD_LIBRARY_PATH": f"{OFIQ_LIB}:{OFIQ_BIN.parent}"}
        subprocess.run(
            [str(OFIQ_BIN), "-c", str(OFIQ_DATA), "-cf", config_file,
             "-i", str(indir), "-o", str(out_csv)],
            check=True, capture_output=True, env=env,
        )
        df = pd.read_csv(out_csv, sep=";")
    df["basename"] = df["Filename"].apply(lambda x: Path(str(x)).name)
    return df.set_index("basename")


def gate(port_scalars: dict[str, dict[str, float]], ofiq_df: pd.DataFrame,
         components: list[str], tol: float = 1.0) -> dict:
    """port_scalars: {basename: {component: scalar}}. Returns per-component ±tol report."""
    report = {}
    for comp in components:
        scalar_col = f"{comp}.scalar"
        diffs, n_pass, n = [], 0, 0
        for bn, ports in port_scalars.items():
            if comp not in ports or bn not in ofiq_df.index:
                continue
            pv = ports[comp]
            ov = float(ofiq_df.loc[bn, scalar_col])
            if pv is None or (isinstance(pv, float) and np.isnan(pv)):
                continue
            d = abs(float(pv) - ov)
            diffs.append(d)
            n += 1
            if d <= tol:
                n_pass += 1
        if n == 0:
            report[comp] = {"n": 0, "pass": 0, "max_diff": None, "verdict": "NODATA"}
        else:
            report[comp] = {
                "n": n, "pass": n_pass, "max_diff": round(max(diffs), 3),
                "mean_diff": round(float(np.mean(diffs)), 3),
                "verdict": "CONFORMANT" if n_pass == n else "FAIL",
            }
    return report


def print_report(report: dict) -> None:
    print(f"{'component':30} {'n':>4} {'pass':>5} {'maxΔ':>8} {'meanΔ':>8}  verdict")
    for comp, r in report.items():
        md = f"{r['max_diff']}" if r["max_diff"] is not None else "  -"
        mn = f"{r.get('mean_diff','-')}"
        print(f"{comp:30} {r['n']:>4} {r['pass']:>5} {md:>8} {mn:>8}  {r['verdict']}")


if __name__ == "__main__":
    import sys
    imgs = [Path(p) for p in sys.argv[1:]]
    df = run_ofiq(imgs)
    print("OFIQ ran on", len(df), "images. Columns:", len(df.columns))
    print(df[["UnifiedQualityScore", "UnifiedQualityScore.scalar", "HeadSize.scalar"]].head())
