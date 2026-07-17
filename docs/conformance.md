# Conformance

## Methodology

Conformance is measured against a **live `OFIQSampleApp` v1.1.0** run, not against a cached
baseline. For each image the port scalar is compared to OFIQ's scalar per component; a
component passes if `|port − OFIQ| ≤ 1` on every image — the ISO/IEC 29794-5 **Annex A.2**
per-image ±1 criterion. Images are real (unaligned) CelebA photographs; both OFIQ and the
port run their full pipelines end-to-end.

The reference OFIQ output is regenerated live by `tests/verify_ofiq.py`; the runner is
`tests/gate_slice.py`.

```bash
export OFIQPY_OFIQ_ROOT=/path/to/OFIQ-Project
python tests/gate_slice.py 1000
```

## Results

Validated on **1,000+ real CelebA images**:

- **27 of 28 components fully conformant** (±1 on every image), most **bit-exact** (Δ=0).
- Every component's mean absolute deviation is well under 1 quality point.
- **~99.99% of all component-image pairs within ISO ±1.**

The rare per-image residuals are numerical boundaries of discrete/learned models — the
Sharpness random-forest vote count landing on a split threshold, an AdaBoost expression
score on a sigmoid boundary, or a round tie — not algorithm differences. The underlying
feature/embedding computations are bit-exact.

## How bit-exact parity was reached

An early version was only ~96% conformant. The residual was **not** a toolchain limit: it
was a real 1px bug in the ADNet landmark back-projection (it used the square box's *width*
where OFIQ uses `height / 256`, and `floor`/`ceil` squaring can leave the box 1px
non-square). Because OFIQ's alignment source points derive from the landmarks, that 1px
drift cascaded into every landmark-sensitive measure.

The bug was isolated by building ctypes bridges (`native/`) against OFIQ's *own* conan
OpenCV static libraries and confirming that `estimateAffinePartial2D`, `warpAffine`,
`resize`, and the SSD dnn forward pass are **bit-identical** between the pip `opencv-python`
wheel and OFIQ's build — which ruled out an OpenCV difference and localized the defect. The
bridges remain in the repo as verification tools; the runtime uses pip `cv2`.

## Reproducing

`tests/gate_slice.py N` runs the port on the first `N` CelebA images, runs live OFIQ on the
same set, and prints a per-component pass/`maxΔ`/`meanΔ` table with a CONFORMANT/FAIL verdict.
