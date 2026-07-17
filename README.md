# ofiqpy — a faithful Python port of OFIQ v1.1.0

[![CI & Release](https://github.com/AVHBAC/ofiqpy/actions/workflows/workflow.yml/badge.svg)](https://github.com/AVHBAC/ofiqpy/actions/workflows/workflow.yml)
[![Docs](https://github.com/AVHBAC/ofiqpy/actions/workflows/docs.yml/badge.svg)](https://avhbac.github.io/ofiqpy/)
[![PyPI](https://img.shields.io/pypi/v/ofiqpy.svg)](https://pypi.org/project/ofiqpy/)
[![Python](https://img.shields.io/pypi/pyversions/ofiqpy.svg)](https://pypi.org/project/ofiqpy/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A behavior-preserving Python reimplementation of the BSI **OFIQ** ISO/IEC 29794-5
face image quality library. `ofiqpy` **reuses OFIQ's own model files and reproduces
its exact `.cpp` algorithms**, matching the live OFIQ reference to within **±1** per
component (the ISO/IEC 29794-5 Annex A.2 conformance criterion).

📖 **Docs: <https://avhbac.github.io/ofiqpy/>**

> Status: **full coverage + near-exact parity.** All 27 ISO components + UnifiedQualityScore,
> an OFIQ-compatible CSV writer, a CLI, and a parallel batch runner — all gated against
> live `OFIQSampleApp`. Validated on **1,000+ real CelebA images**: 27/28 components fully
> conformant (±1 on every image, most bit-exact); ~99.99% of all component-image pairs
> within ±1. The rare per-image residuals are numerical boundaries of discrete/learned
> models (RTrees vote, AdaBoost score, round tie), not algorithm gaps.

## Install & run

```bash
pip install ofiqpy
export OFIQPY_OFIQ_DATA=/path/to/OFIQ-Project/data   # OFIQ's models (not bundled)

ofiqpy -i face.jpg -o out.csv                        # CLI
```
```python
from ofiqpy import assess
scores = assess("face.jpg")            # {component: (raw, scalar)}
print(scores["UnifiedQualityScore"])   # (magnitude, 0-100)
```

See [Installation](https://avhbac.github.io/ofiqpy/installation/) and
[Quickstart](https://avhbac.github.io/ofiqpy/quickstart/).

## Design

- **Same weights.** Models are loaded directly from the reference checkout
  `OFIQ-Project/data/models/` via `config.py` (SSD caffemodel, ADNet-98 ONNX,
  3DDFA-V2 ONNX, BiSeNet parsing ONNX, occlusion-seg ONNX, ssim-248 ONNX, MagFace ONNX).
  No re-training, no substitute backbones.
- **Same math.** Detection, ADNet landmark crop/denorm, the 5-point LMEDS similarity
  alignment to 616×616, the landmarked-region mask, `tmetric`, the generic
  `h·(a+s·sigmoid(x;x0,w))` scalar mapping, and each measure are ported line-faithfully
  from OFIQ's C++ (see per-module docstrings for `file:line` provenance).
- **Same versions.** The isolated `.venv` pins **OpenCV 4.5.5** and **onnxruntime 1.18.1**
  to match OFIQ's build (`libofiq_lib.so` links OpenCV 4.5.5; bundles onnxruntime 1.18.1).
- **Gated, not asserted.** `tests/verify_ofiq.py` runs live OFIQ and checks
  `|port_scalar − ofiq_scalar| ≤ 1` per image; `tests/gate_slice.py` reports it.

## Conformance (1,000+ real CelebA images, port vs live OFIQ, ISO Annex A ±1)

Validated on **1,197 real CelebA images**: **27 of 28 components fully conformant** (±1 on
every image), 24 of them **bit-exact** (maxΔ=0). **~99.99% of all component-image pairs are
within ISO ±1** (mean |Δ| ≤ 0.02 for every component).

The rare per-image residuals:
- **Sharpness — 1 image at Δ=2**: OFIQ's RTrees vote count differs by exactly 2 trees at a
  split-threshold knife-edge (a sub-LSB feature difference flips 2 borderline votes through
  the step-function forest). Bit-exact on the rest.
- **BackgroundUniformity / ExpressionNeutrality / NoHeadCoverings — one ±1 image each**, a
  single sigmoid/round boundary.

Sharpness (RTrees) and ExpressionNeutrality (dual EfficientNet + AdaBoost) run OFIQ's own
`cv2.ml` / ONNX models; UnifiedQualityScore runs OFIQ's MagFace ONNX. These four residuals
are numerical boundaries of discrete/learned models, not algorithm gaps.

### How parity was reached (the residual was a bug, not a build limit)

An earlier version of this port was only ~96% conformant, and the residual was *wrongly*
attributed to a build-level OpenCV float difference. To test that, ctypes bridges were
built against OFIQ's own conan OpenCV static libs (`native/ofiq_cv.cpp`, `ofiq_ssd.cpp`)
and used to compare OFIQ's compiled `estimateAffinePartial2D`, `warpAffine`, `resize`, and
the SSD dnn forward pass against the pip `opencv-python` wheel. **Every OpenCV operation
was bit-identical** — which disproved the build-level theory and localized the divergence
to the **ADNet landmark back-projection**: OFIQ scales landmarks back with
`squareBox.height / 256` (`adnet_landmarks.cpp:313`), but `makeSquareBoundingBox`'s
`floor`/`ceil` can leave the box 1px non-square, and the port had used the *width*. On
exactly-square detector boxes it matched; otherwise it drifted ~1px, propagating (via the
alignment source points → affine → whole aligned face) into every landmark-sensitive
measure. One-character fix (width→height); all 27 non-model components went bit-exact.

The bridges in `native/` are diagnostic only — the runtime uses pip `cv2`, which is
bit-identical to OFIQ's OpenCV. No source build of OpenCV was needed.

## Layout

```
ofiqpy/
  config.py            JAXN loader + OFIQ model resolver + sigmoid params
  sigmoid.py           OFIQ ScalarConversion (Measure.h:271-285)
  session.py           shared preprocessing products
  pipeline.py          detect -> pose -> landmarks -> align -> parse -> occlusion -> region
  detectors/ssd.py     SSD (OpenCV DNN, Caffe)
  landmarks/adnet.py   ADNet-98 + square-crop helpers
  align.py             616x616 alignment, landmarked region (GetFaceMask), tmetric, luminance
  pose/tddfa.py        3DDFA-V2 pose
  segmentation/        BiSeNet parsing, face-occlusion seg
  measures/
    core.py            model cache + dispatch; C03/C09/C17/C20, unified, HeadPose (slot swap)
    geometry.py        C11,C12,C13,C19,C24-C27
    pixel.py           C01,C02,C04(var),C05,C06,C07,C10
    models.py          C08 Sharpness (RTrees), C14/C15/C16 occlusion, C18 Expression
    helpers.py         get_distance/get_middle, c_round, landmark index maps
  output.py            OFIQ-format CSV (named cols, raw + .scalar)
  cli.py               OFIQSampleApp-compatible CLI
tests/
  verify_ofiq.py       runs live OFIQ, ±1 gate
  gate_slice.py        full-coverage conformance runner
```

## Run

```bash
export OFIQPY_OFIQ_DATA=/path/to/OFIQ-Project/data

ofiqpy -i <image|dir> -o out.csv               # single / small runs (OFIQ-format CSV)
python -m ofiqpy.batch -i <dir> -o out.csv -w 8 --resume   # parallel batch

# reproduce the conformance gate (needs a built OFIQSampleApp)
export OFIQPY_OFIQ_ROOT=/path/to/OFIQ-Project
export OFIQPY_TEST_IMAGES=/path/to/images
python tests/gate_slice.py 1000
```

Full documentation: <https://avhbac.github.io/ofiqpy/>.

## License & attribution

`ofiqpy` is released under the [MIT License](LICENSE).

It is a faithful port of **OFIQ** (Open Source Face Image Quality), developed by the German
Federal Office for Information Security (BSI), Copyright © 2024, MIT-licensed
(<https://github.com/BSI-OFIQ/OFIQ-Project>). Please acknowledge OFIQ when using ofiqpy.

**Models are not bundled.** ofiqpy loads OFIQ's own model files at runtime; they ship with
OFIQ and may be licensed separately (see OFIQ's `LICENSE.md`). Obtain them from an OFIQ
install and set `OFIQPY_OFIQ_DATA`. See [`NOTICE`](NOTICE) and the
[licensing docs](https://avhbac.github.io/ofiqpy/licensing/).
