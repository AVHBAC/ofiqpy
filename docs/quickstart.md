# Quickstart

Set `OFIQPY_OFIQ_DATA` to the verified OFIQ v1.1.0 data directory first.

## Typed API

```python
import os
from pathlib import Path

from ofiqpy import Assessor, AssessmentStatus
from ofiqpy.config import OFIQConfig

data_root = Path(os.environ["OFIQPY_OFIQ_DATA"])
image = data_root / "tests" / "images" / "r-01-frontal.png"
assessor = Assessor(OFIQConfig(data_root=data_root))
assessment = assessor.assess(image)

if assessment.status is AssessmentStatus.FAILURE_TO_ASSESS:
    print(assessment.failure)
else:
    for name, component in assessment.components.items():
        print(name, component.status, component.raw, component.scalar)
```

The same `Assessor` can process multiple images. Its internal lock makes calls safe when an
application shares one instance across threads; calls are serialized because the model
objects themselves are mutable.

## Compatibility mapping

```python
import os
from pathlib import Path

from ofiqpy import assess

data_root = Path(os.environ["OFIQPY_OFIQ_DATA"])
scores = assess(data_root / "tests" / "images" / "r-01-frontal.png")
raw, scalar = scores["UnifiedQualityScore"]
print(raw, scalar)
```

The mapping adapter returns an empty dictionary only for a valid image in which no face is
detected. It raises for unreadable paths, invalid arrays, and preprocessing errors. Use the
typed API when the failure reason or partial component results must be retained as data.

## Command line

```bash
ofiqpy \
  -i "$OFIQPY_OFIQ_DATA/tests/images/r-01-frontal.png" \
  -o assessment.csv

python -m ofiqpy.batch \
  -i "$OFIQPY_OFIQ_DATA/tests/images" \
  -o assessments.csv \
  --resume
```
