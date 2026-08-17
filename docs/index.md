# ofiqpy

`ofiqpy` implements the fixed BSI OFIQ v1.1.0 profile in Python. The profile contains
28 outputs: 27 ISO/IEC 29794-5 quality components and `UnifiedQualityScore`.

## Supported contract

- OFIQ `v1.1.0` at commit `bb5dc91d00477e02ce53d2530d28e35021484393`.
- One canonical configuration and twelve hash-verified model artifacts.
- The complete 28-component output set.
- Typed image and component success/`FailureToAssess` results.
- Canonical semicolon CSV output with stable full-path identities.

This release does not implement arbitrary JAXN measure selection or parameter overrides,
nor does it reproduce the full C++ API surface.

## Current evidence

The reproducible gate uses all 28 real BSI conformance images and observes every one of the
784 image/component pairs. At the 0.2.0 source state, all 784 scalar values and all 784
statuses matched live `OFIQSampleApp` exactly. At six-decimal CSV precision, 702 raw values
were exact and all 784 met the named, component-specific raw policy. See
[Conformance](conformance.md) for the artifact hashes and verdict policy, and
[Architecture review](review.md) for the full C++ crosswalk and implementation record.

The [runtime performance review](performance.md) uses real BSI and CelebA inputs and
confirms one worker as the measured default on the benchmark host; its aggregate evidence
contains no licensed images or per-image identities.

A non-redistributed 1,197-image CelebA diagnostic is also aggregate-bound in the review.
It is broader compatibility evidence, not a substitute for the strict BSI release gate or
independent validation of the raw tolerance policy.

```python
import os
from pathlib import Path

from ofiqpy import Assessor
from ofiqpy.config import OFIQConfig

data_root = Path(os.environ["OFIQPY_OFIQ_DATA"])
result = Assessor(OFIQConfig(data_root=data_root)).assess(
    data_root / "tests" / "images" / "r-01-frontal.png"
)
print(result.components["UnifiedQualityScore"])
```
