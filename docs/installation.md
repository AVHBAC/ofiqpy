# Installation

## Install the package

```bash
pip install ofiqpy
```

or from source:

```bash
git clone https://github.com/AVHBAC/ofiqpy
cd ofiqpy
pip install -e .
```

Dependencies are pinned to OFIQ v1.1.0's toolchain for bit-faithful model inference:

| Package | Version |
|---|---|
| numpy | `>=1.23,<2` |
| opencv-python-headless | `==4.5.5.64` |
| onnxruntime | `==1.18.1` |

## Provide OFIQ's models

ofiqpy reproduces OFIQ's algorithms but **does not bundle OFIQ's model files** (they ship
with OFIQ and may be licensed separately). Obtain them from an OFIQ install and point
ofiqpy at the `data/` directory:

```bash
git clone https://github.com/BSI-OFIQ/OFIQ-Project
# download OFIQ's models per its instructions, then:
export OFIQPY_OFIQ_DATA=/path/to/OFIQ-Project/data
```

The `data/` directory must contain `ofiq_config.jaxn` and the `models/` tree
(`face_detection/`, `face_landmark_estimation/`, `head_pose_estimation/`, `face_parsing/`,
`face_occlusion_segmentation/`, `sharpness/`, `no_compression_artifacts/`,
`expression_neutrality/`, `unified_quality_score/`).

## (Optional) The conformance gate

To reproduce the ±1 conformance validation you also need a built `OFIQSampleApp` and pandas:

```bash
pip install "ofiqpy[verify]"
export OFIQPY_OFIQ_ROOT=/path/to/OFIQ-Project   # dir containing install_x86_64_linux/
python tests/gate_slice.py 100
```

## Verify the install

```bash
python -c "import ofiqpy; print(ofiqpy.__version__)"
ofiqpy -i face.jpg -o out.csv
```
