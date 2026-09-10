# Installation

## Package

Use Python 3.11 or 3.12 on Linux x86-64 or Linux inside Windows WSL2. These are
the Python versions exercised by the existing complete release gate. Keep a
separate environment from OpenFIQA and QualSight because the canonical
runtime pins intentionally differ.

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

Models are not bundled. For an already authorized local profile, no C++
toolchain or download is required:

```bash
export OFIQPY_OFIQ_DATA="$AUTHORIZED_OFIQ_DATA"
python -m ofiqpy.config
```

To acquire the upstream profile explicitly, review its model licenses first.
The following model-only route uses the URL and layout in
[the pinned OFIQ build](https://github.com/BSI-OFIQ/OFIQ-Project/blob/bb5dc91d00477e02ce53d2530d28e35021484393/CMakeLists.txt)
without compiling the native reference. It needs `git`, `curl`, and `unzip`.

```bash
git clone --branch v1.1.0 https://github.com/BSI-OFIQ/OFIQ-Project.git ../OFIQ-Project
curl --fail --location --output OFIQ-MODELS.zip \
  https://standards.iso.org/iso-iec/29794/-5/ed-1/en/OFIQ-MODELS.zip
unzip -n OFIQ-MODELS.zip -d ../OFIQ-Project/data
export OFIQPY_OFIQ_ROOT="$(cd ../OFIQ-Project && pwd)"
export OFIQPY_OFIQ_DATA="$OFIQPY_OFIQ_ROOT/data"
python -m ofiqpy.config
```

Initialization verifies the canonical config SHA-256 and twelve model SHA-256/size pairs.
It fails closed for a newer OFIQ checkout, a locally edited JAXN file, a partial download,
or substituted weights.

## Runtime check on a real BSI image

Obtain the BSI images separately under their terms before this example:

```bash
curl --fail --location --output OFIQ-IMAGES.zip \
  https://standards.iso.org/iso-iec/29794/-5/ed-1/en/OFIQ-IMAGES.zip
unzip -n OFIQ-IMAGES.zip -d "$OFIQPY_OFIQ_DATA/tests"
```

```bash
python -c "import ofiqpy; print(ofiqpy.__version__)"
ofiqpy \
  -i "$OFIQPY_OFIQ_DATA/tests/images/r-01-frontal.png" \
  -o assessment.csv
```

Only the strict live-C++ comparison needs the native reference build:
`(cd "$OFIQPY_OFIQ_ROOT/scripts" && sh build.sh)`. Follow
[Conformance](conformance.md) for that separate gate and upstream build
prerequisites. No pandas dependency is required. Model configuration checks
are not new conformance evidence. The prepared 0.2.1 source changes the
setuptools build floor, not this runtime profile; publication is not claimed.
