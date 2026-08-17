# Installation

## Package

Python 3.11 or later is required. The complete release gate runs on Python 3.11 and 3.12.

```bash
python -m pip install ofiqpy
```

The 0.2.0 numerical profile uses these runtime dependency versions:

| Package | Version |
|---|---|
| numpy | `>=1.23,<2` |
| opencv-python-headless | `==4.5.5.64` |
| onnxruntime | `==1.18.1` |

The supported numerical profile uses CPU execution. Initializing the pipeline disables
OpenCV's process-wide optimized kernels to match the tested OFIQ OpenCV 4.5.5 CPU path;
see [Architecture](architecture.md) before sharing a process with another OpenCV workload.

## Exact OFIQ v1.1.0 data profile

Models are not bundled. Obtain the pinned OFIQ source and let its build retrieve the real
model and test-image artifacts:

```bash
git clone --branch v1.1.0 https://github.com/BSI-OFIQ/OFIQ-Project.git ../OFIQ-Project
(
  cd ../OFIQ-Project/scripts
  sh build.sh
)

export OFIQPY_OFIQ_ROOT="$(cd ../OFIQ-Project && pwd)"
export OFIQPY_OFIQ_DATA="$OFIQPY_OFIQ_ROOT/data"
```

Initialization verifies the canonical config SHA-256 and twelve model SHA-256/size pairs.
It fails closed for a newer OFIQ checkout, a locally edited JAXN file, a partial download,
or substituted weights.

## Runtime check on a real BSI image

```bash
python -c "import ofiqpy; print(ofiqpy.__version__)"
ofiqpy \
  -i "$OFIQPY_OFIQ_DATA/tests/images/r-01-frontal.png" \
  -o assessment.csv
```

To execute the strict reference comparison, follow [Conformance](conformance.md). No
additional pandas dependency is required.
