# ofiqpy

**A faithful Python port of [OFIQ](https://github.com/BSI-OFIQ/OFIQ-Project) v1.1.0** — the
ISO/IEC 29794-5 face image quality reference implementation from the German Federal Office
for Information Security (BSI).

Unlike an ISO-*inspired* reimplementation, ofiqpy **reuses OFIQ's own model files and
reproduces its exact `.cpp` algorithms**, targeting per-component agreement with the OFIQ
reference to within **±1 quality point** — the ISO/IEC 29794-5 Annex A conformance criterion.

## Conformance at a glance

Validated on **1,000+ real CelebA images** against a live `OFIQSampleApp` v1.1.0 run:

- **27 of 28 components fully conformant** (±1 on every image); most are **bit-exact** (Δ=0).
- The handful of per-image residuals are numerical boundaries of discrete/learned models
  (the RTrees vote count, an AdaBoost score, a sigmoid round) — not algorithm gaps.
- **~99.99% of all component-image pairs within ISO ±1.**

See [Conformance](conformance.md) for the full methodology and per-component figures.

## What it computes

All 27 ISO/IEC 29794-5 quality components plus the unified quality score:

- **Capture / pixel**: background & illumination uniformity, luminance mean/variance,
  under/over-exposure, dynamic range, sharpness, compression artifacts, natural colour.
- **Subject / geometry**: single-face, eyes open/visible, mouth closed/occluded, face
  occlusion, no head coverings, expression neutrality, inter-eye distance, head size,
  crop margins (leftward/rightward/above/below), head pose (yaw/pitch/roll).
- **Unified**: MagFace-based unified quality score.

## Design

- **Same weights** — models are loaded directly from an OFIQ install (see
  [Installation](installation.md)); nothing is re-trained.
- **Same math** — detection, ADNet landmarks, 5-point similarity alignment, the
  landmarked-region mask, and every measure are ported line-faithfully from OFIQ's C++.
- **Same versions** — pinned to OpenCV 4.5.5 + onnxruntime 1.18.1 (OFIQ's toolchain) for
  bit-faithful model inference.
- **Gated, not asserted** — a harness runs live OFIQ and checks `|port − OFIQ| ≤ 1` per image.

```python
from ofiqpy import assess
scores = assess("face.jpg")          # {component: (raw, scalar)}
print(scores["UnifiedQualityScore"]) # (raw_magnitude, 0-100 scalar)
```
