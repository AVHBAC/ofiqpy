# Quickstart

Set `OFIQPY_OFIQ_DATA` to your OFIQ `data/` directory first (see
[Installation](installation.md)).

## Python API

```python
from ofiqpy import assess

scores = assess("face.jpg")
# scores: {component_name: (raw_native_value, scalar_0_100)}

raw, scalar = scores["UnifiedQualityScore"]
print(f"unified quality: {scalar}/100 (magnitude {raw:.2f})")

for name, (raw, scalar) in sorted(scores.items()):
    print(f"{name:28} {scalar:3.0f}")
```

`assess` also accepts a BGR uint8 numpy array (as returned by `cv2.imread`):

```python
import cv2
from ofiqpy import assess
scores = assess(cv2.imread("face.jpg"))
```

## Lower-level API

For access to the shared pipeline products (aligned face, landmarks, masks, pose):

```python
import cv2
from ofiqpy.config import OFIQConfig
from ofiqpy.pipeline import OFIQPipeline
from ofiqpy.measures.core import Measures

cfg = OFIQConfig()
pipe = OFIQPipeline(cfg)
meas = Measures(cfg)

session = pipe.process(cv2.imread("face.jpg"))
# session.aligned_face, session.landmarks, session.aligned_landmarks,
# session.parsing, session.occlusion_mask, session.yaw/pitch/roll ...
scores = meas.compute(session)          # {name: (raw, scalar)}
scalars = meas.compute_scalars(session) # {name: scalar}
```

## Command line

```bash
# single image or directory -> OFIQ-format CSV
ofiqpy -i face.jpg -o out.csv
ofiqpy -i /path/to/images/ -o out.csv

# parallel batch over a large directory (resumable)
python -m ofiqpy.batch -i /path/to/images/ -o out.csv -w 8 --resume
```

The CSV mirrors `OFIQSampleApp`: semicolon-delimited, named columns, raw value + `.scalar`
per component, plus `assessment_time_in_ms`. See [CLI & Batch](cli.md).
