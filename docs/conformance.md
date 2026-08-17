# Conformance

## Verdict contract

`python -m ofiqpy.conformance` executes the live BSI `OFIQSampleApp` and `ofiqpy` against
the same input directory. The gate requires:

1. Exactly the declared number of supported images and unique path identities.
2. Exactly one reference row and one port result for every image.
3. Exactly 28 named components for every image.
4. Finite scalar values for every observation and finite raw values whenever both
   implementations report success.
5. Matching success/`FailureToAssess` status for every observation.
6. Scalar difference no greater than the declared tolerance for every observation.
7. Raw difference no greater than the named component tolerance for every jointly
   successful observation.

There are no skip paths for no-face images, absent components, NaN values, or missing
reference rows. A contract violation raises an error; a numerical/status failure returns a
nonzero verdict.

Raw values are compared at the six-decimal precision emitted by `OFIQSampleApp` and do
determine the verdict. The immutable policy
`bsi-ofiq-v1.1.0-cpu-raw-csv-v1` covers all 28 components. Ten deterministic components
require exact raw equality. The others use absolute tolerances in their native units, with
an executable rationale for every value in `ofiqpy/raw_policy.py`. A raw value is undefined
and explicitly excluded only when either side reports `FailureToAssess`; status comparison
still applies to that observation.

## Release result: 2026-08-16

| Observation | Result |
|---|---:|
| BSI images required / observed by each implementation | 28 / 28 |
| Components required / observed per image | 28 / 28 |
| Scalar exact | 784 / 784 |
| Scalar within tolerance 0 | 784 / 784 |
| Status exact | 784 / 784 |
| Raw exact at six decimals | 698 to 702 / 784 across verified Linux runs |
| Raw within component policy | 784 / 784 |
| Undefined raw values excluded | 0 / 784 |

The official C++ conformance executable independently passed 787 / 787 tests in the same
review environment. The raw exact count was 698 on the GitHub Python 3.11.15 runner and 702
on the local Python 3.11.14 host. It is an environment-sensitive diagnostic; the declared
component policy, which accepted all 784 observations, determines the raw-value verdict.

## Bound artifacts

The release run recorded these stable profile/input bindings:

| Artifact | SHA-256 |
|---|---|
| Canonical config | `e117286706d799a1e23130db01cfdff7ef36d157b22ecf68dd196052a8a8d0b3` |
| Canonical 12-model manifest aggregate | `9a7b6b7943f4c17000bf32eed5be66ad5259f20c6166be4e83bf63db695d8831` |
| 28-image BSI input aggregate | `f9dab8561ac543e44e93e12c254b0a7af6b6cdd12d3156f1174b8b984a6c4be5` |
| Tested local `OFIQSampleApp` binary | `1c36ce0a1099e3df4c406a53a3cfd70c75a00cfb97299ebe1aa50cdec235e98c` |

The generated report also records the source-tree hash, both git commits, OFIQ shared
library hashes, ONNX Runtime library hash, and an optional wheel/sdist hash. Rebuilt C++
binaries can have a different binary hash; the report binds the executable actually used.
The gate captures these source, input, model, native, and distribution bindings before and
after execution and rejects the run if any bound artifact changes between the two
snapshots.

## Reproduce

```bash
export OFIQPY_OFIQ_ROOT="$(cd ../OFIQ-Project && pwd)"
export OFIQPY_OFIQ_DATA="$OFIQPY_OFIQ_ROOT/data"

python -m ofiqpy.conformance \
  --ofiq-root "$OFIQPY_OFIQ_ROOT" \
  --images "$OFIQPY_OFIQ_DATA/tests/images" \
  --expected-count 28 \
  --scalar-tolerance 0 \
  --source-root "$PWD" \
  --report conformance-report.json
```

## Licensed larger-corpus diagnostic

The historical selection was recovered as the first 1,197 sorted CelebA image identities
from the original project dataset. The current, non-redistributed bytes were bound as:

| Binding | Value |
|---|---:|
| Images / bytes | 1,197 / 125,934,071 |
| Aggregate SHA-256 | `6a586d0a42e868a05fb817804a284bbd0a42135a3a50b41fce9539689536cc18` |
| Image/component observations | 33,516 |
| Status exact | 33,516 / 33,516 |
| Scalar exact | 33,483 / 33,516 |
| Scalar within tolerance 1 | 33,516 / 33,516 |
| Defined raw exact at six decimals | 29,938 / 33,430 |
| Defined raw within component policy | 33,430 / 33,430 |
| Undefined raw values excluded | 86 / 33,516 |

The 33 non-exact scalars differed by one point: 12 `BackgroundUniformity`, 14
`ExpressionNeutrality`, and 7 `NoHeadCoverings` observations. The generated aggregate-only
report binds the same native/profile/source fields as the BSI gate and contains no
per-image records.

CelebA permits non-commercial research use and prohibits redistribution under its
[official project agreement](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), so neither
images nor a per-image manifest are shipped. No historical input hash was recorded by the
old reporter; this new result binds the currently available bytes but cannot prove that the
earlier run used byte-identical files. The raw policy was finalized through diagnosis on
the BSI set and this recovered cohort, so this result is in-sample compatibility evidence,
not independent external validation of the tolerance envelope.
