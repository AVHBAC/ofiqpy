# ofiqpy

`ofiqpy` is a Python reimplementation of the fixed, canonical BSI OFIQ v1.1.0
quality-assessment profile. It computes the 27 ISO/IEC 29794-5 components exposed by
OFIQ v1.1.0 plus `UnifiedQualityScore` using OFIQ's model files.

ofiqpy supports one exact contract:

- OFIQ source tag `v1.1.0`, commit `bb5dc91d00477e02ce53d2530d28e35021484393`.
- The canonical `ofiq_config.jaxn` with SHA-256
  `e117286706d799a1e23130db01cfdff7ef36d157b22ecf68dd196052a8a8d0b3`.
- Twelve exact model artifacts totaling 453,497,937 bytes. `OFIQConfig` hashes every
  required artifact before creating inference sessions.
- All 28 canonical output components. Arbitrary JAXN measure lists, scalar overrides,
  and the full C++ API are not implemented.

## Confirmed parity evidence

The strict gate runs the live C++ `OFIQSampleApp` and this package on all 28 real BSI
conformance images. It rejects missing images, duplicate identities, missing components,
non-numeric values, and status mismatches.

At the 0.2.0 source state on 2026-08-16:

| Observation | Result |
|---|---:|
| Official C++ conformance suite | 787 / 787 passed |
| Image/component observations | 784 / 784 present |
| Scalar values exactly equal | 784 / 784 |
| Scalar values within tolerance 0 | 784 / 784 |
| Success/FailureToAssess status equal | 784 / 784 |
| Raw values equal at six-decimal CSV precision | 702 / 784 |
| Raw values within the named component policy | 784 / 784 |

Raw values determine the verdict under the immutable
`bsi-ofiq-v1.1.0-cpu-raw-csv-v1` component policy. The policy compares the six-decimal
values emitted by `OFIQSampleApp`; ten deterministic components require equality and each
remaining component has a unit-specific bound and rationale. Raw values are excluded only
when either implementation reports `FailureToAssess`, where OFIQ does not define a raw
result. The earlier unbound 1,197-image and universal bit-exactness claims are not used as
release evidence.

A separate, non-redistributed diagnostic reran the recovered 1,197-image CelebA selection
with aggregate provenance. All 33,516 statuses matched; 33,483 scalars were exact and all
were within one point; all 33,430 defined raw comparisons met the component policy. This
binds the currently available bytes but does not retroactively prove that the original
unhashed run used identical bytes. See [Conformance](docs/conformance.md).

## Install and configure

```bash
python -m pip install ofiqpy

git clone --branch v1.1.0 https://github.com/BSI-OFIQ/OFIQ-Project.git ../OFIQ-Project
(
  cd ../OFIQ-Project/scripts
  sh build.sh
)

export OFIQPY_OFIQ_ROOT="$(cd ../OFIQ-Project && pwd)"
export OFIQPY_OFIQ_DATA="$OFIQPY_OFIQ_ROOT/data"
```

The OFIQ build downloads the separately licensed models and BSI test images. `ofiqpy`
does not bundle or redistribute those files. Initialization stops with an integrity error
if the downloaded configuration or any required model differs from the verified profile.

## Python API

Use `Assessor` when status and failure information matters:

```python
import os
from pathlib import Path

from ofiqpy import Assessor
from ofiqpy.config import OFIQConfig

data_root = Path(os.environ["OFIQPY_OFIQ_DATA"])
assessor = Assessor(OFIQConfig(data_root=data_root))
result = assessor.assess(data_root / "tests" / "images" / "r-01-frontal.png")

print(result.status)
print(result.components["UnifiedQualityScore"])
```

`Assessor` loads and validates the complete model graph before assessment and serializes
access to mutable OpenCV/ONNX sessions. A component exception produces a typed
`FailureToAssess` only for that component or compound component group; unaffected results
are retained.

The original mapping API remains as a compatibility adapter:

```python
import os
from pathlib import Path

from ofiqpy import assess

image = Path(os.environ["OFIQPY_OFIQ_DATA"]) / "tests" / "images" / "r-01-frontal.png"
scores = assess(image)
print(scores["UnifiedQualityScore"])
```

It returns `{component: (raw, scalar)}` and returns an empty mapping only when no face is
detected. Unreadable paths, invalid arrays, and preprocessing failures raise instead of
being collapsed into a no-face result. New integrations should use `Assessor.assess` or
`assess_typed` when the typed failure detail is required.

## CLI and batch

```bash
ofiqpy \
  -i "$OFIQPY_OFIQ_DATA/tests/images/r-01-frontal.png" \
  -o assessment.csv

python -m ofiqpy.batch \
  -i "$OFIQPY_OFIQ_DATA/tests/images" \
  -o assessments.csv \
  --resume
```

The semicolon CSV uses the canonical 28-component OFIQ column order and preserves the
full supplied/discovered image path. CSV quoting protects delimiters in paths. Resume
validates the exact header and row width and uses the full path identity, so recursive
duplicate basenames do not collide. Batch execution defaults to one worker because each
worker owns a complete model graph; additional workers require an explicit `-w` value and
start with Python's clean `spawn` process context. An unreadable image or preprocessing
failure exits nonzero rather than silently writing a whole-image sentinel row.

On the tested 64-image real-data workload, version 0.2.0 processed 2.649 images/s
at one worker, 2.391 at two, and 2.140 at four; median process-tree RSS rose from 1.292 GiB
to 2.450 and 4.723 GiB. One worker is therefore the measured default for that host, not a
universal optimum. See [Runtime performance](docs/performance.md).

## Reproduce the strict gate

```bash
python -m ofiqpy.conformance \
  --ofiq-root "$OFIQPY_OFIQ_ROOT" \
  --images "$OFIQPY_OFIQ_DATA/tests/images" \
  --expected-count 28 \
  --scalar-tolerance 0 \
  --source-root "$PWD" \
  --report conformance-report.json
```

The command exits nonzero for any conformance failure or incomplete comparison. The JSON
report binds the source tree, OFIQ and ofiqpy commits, official binary and library hashes,
canonical config/model hashes, and the complete input set.

## License and attribution

`ofiqpy` is MIT-licensed. OFIQ is developed by the German Federal Office for Information
Security (BSI) and is also MIT-licensed. OFIQ's models have their own license terms; review
the license files downloaded into the OFIQ data directory before redistribution.

This project is independent and is not endorsed by ISO, IEC, or BSI.
