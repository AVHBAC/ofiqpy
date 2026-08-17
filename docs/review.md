# Architecture review

## Review scope

This review compares:

- `ofiqpy` v0.1.1 at `c80fb388fe1ccecd17f60f296a1ef31bd95a1ee3`.
- BSI OFIQ v1.1.0 at `bb5dc91d00477e02ce53d2530d28e35021484393`.
- The exact canonical config and twelve-model profile verified by `ofiqpy.profile`.

The supported target is the canonical OFIQ v1.1.0 profile. Arbitrary JAXN configuration
and the full C++ library API remain outside this package's scope. Implementation and
verification ran in an isolated worktree, leaving the original Python and C++ worktrees
untouched.

## Review result

At the baseline commit, `ofiqpy` was a close translation of the default successful-image
algorithms, but it was not faithful across configuration, lifecycle, errors, result status,
CSV identity, batch resume, or release verification. Its committed comparison program
could skip observations and never failed the process, so historical large-corpus claims
were not reproducibly established by the repository.

Version 0.2.0 uses a smaller contract that can be checked directly:

- a hash-verified OFIQ v1.1.0 config/model profile;
- a preflighted, locked `Assessor` lifecycle;
- typed image and per-component results;
- fixed canonical 28-component output;
- exact full-path CSV identity and strict resume validation;
- cardinality-enforcing live-C++ conformance;
- installed-wheel, real-image, notebook, and release gates.

The 28 real BSI images produced 784/784 exact scalars and 784/784 exact statuses. Raw
values were exact for 702/784 observations at the C++ CSV's six-decimal precision, and all
784 met the immutable component-specific raw policy. Raw comparison is part of the verdict;
undefined raw values are excluded only for matched `FailureToAssess` observations.

## Implementation and verification record

The work followed this dependency order:

```text
isolated worktree
  -> baseline and artifact bindings
  -> canonical-profile scope
  -> typed runtime and component failure isolation
  -> CSV, input, resume, and worker lifecycle
  -> strict live conformance and provenance
  -> real-data tests and release documentation
  -> lint, types, docs, and complete source tests
  -> fresh wheel installation and real-image test
  -> independent diff review and repairs
  -> uncontended-host benchmark and aggregate evidence
  -> final release gate
```

| Area | Result | Confirming evidence |
|---|---|---|
| Work isolation | Sibling worktree used; source repositories left untouched | Git worktree and status inspection |
| Baseline | C++ `787/787`; Python `784/784` scalar/status exact; raw `696/784`; mypy `0`; ruff baseline `23` | Live official suite and 28-image comparison |
| Artifact binding | Binary, shared libraries, config, models, inputs, source, and distribution bound by SHA-256 | `ofiqpy.conformance` report |
| Scope | Canonical v1.1.0 profile; `NaturalColour` coefficient `24289/27`; one version source | Profile and release contracts |
| Runtime | Preflighted `Assessor`, lock, typed statuses, and isolated component failures | Real BSI assessment contracts |
| Input and output | Quoted full paths, canonical header, resume validation, one-worker default, and `spawn` for explicit multiprocessing | Real CSV and two-process BSI contracts |
| Conformance | Exact cardinality and finite-value checks, status/scalar/raw-policy verdict, and nonzero failure | Live conformance contract |
| Test data | Invented smoke values removed; real BSI captures and real derived subsamples used | Complete pytest suite |
| Release workflow | CI builds C++, runs real data and notebooks, tests the wheel, and checks the release tag | Public documentation and workflow contracts |
| Static checks | Repository-wide lint, format, types, docs, YAML, and marker checks | Command exit codes |
| Source checks | Every source test executed, including live C++ | Complete pytest exit code |
| Package checks | Wheel inspected, installed in a fresh environment, and used on a live BSI input | Installed-wheel conformance report |
| Independent review | Three diff reviews drove repairs to runtime, resume, spawn, CLI, finite-value, notebook, native, release, source-cleanliness, and matrix behavior | Review findings and regression tests |
| Final evidence | Generated from the release tree and emitted as JSON, avoiding a self-referential source/distribution hash | Release handoff and generated report |
| Benchmark controls | Host-load guard passed; source, runtime, and real inputs were bound | Process guard and source/input aggregates |
| Single-process benchmark | Three alternating measured repetitions per version after one warm-up | Identical scalar digests and latency/RSS medians |
| Batch benchmark | Three measured repetitions per version at workers 1, 2, and 4 | Exact 64-row cardinality and process-tree RSS/throughput |
| Published benchmark evidence | Aggregate-only record with no licensed image identities | `docs/evidence/runtime-benchmark-20260815.json` |

## Runtime architecture comparison

| Concern | OFIQ C++ v1.1.0 | Baseline Python | ofiqpy 0.2.0 |
|---|---|---|---|
| Initialization | Explicit library initialization | Implicit lazy globals | Explicit `Assessor`; lazy adapter retained |
| Configuration | Selectable measures and parameter overrides | Parsed config but fixed hard-coded executor | Exact canonical config hash; other config rejected |
| Models | C++ runtime objects | Lazy mixed OpenCV/ONNX objects | All required objects preload before assessment |
| Artifact integrity | Build/download provenance | Existence checks only | Exact config/model SHA-256 and size checks |
| Concurrency | Native synchronization around mutable detector state | Shared state without a public contract | One re-entrant lock per `Assessor` |
| Multiprocessing | Not applicable to library | CPU-count default, one model graph per fork | Default one; explicit workers use `spawn` |
| Image failure | Call status plus configured FTA results | No-face `{}`; CLI/batch swallowed exceptions | Typed image failure; CLI/batch hard failures exit nonzero |
| Component failure | Per-measure return code; other results survive | One exception aborted the whole compute call | Single/compound component failure isolation |
| Result structure | Raw, scalar, return code | Dict of raw/scalar pairs | Typed result plus compatibility dict adapter |
| Preprocessing output API | Selectable original-coordinate outputs | Internal `Session` only | Internal advanced API remains; full C++ surface is out of scope |
| CSV columns | Active configured map | Fixed 28 columns | Fixed canonical 28 columns by declared scope |
| CSV identity | Supplied path | Basename only | Full supplied/discovered path with CSV quoting |
| Resume | Sample app has no Python resume layer | Basename-only, header unchecked | Exact header/row width/full identity/duplicate validation |
| Verification | Official native conformance suite | Reporter with skips and no failing verdict | Live native suite plus strict installed-wheel gate |

## Preprocessing crosswalk

| Stage | Python implementation | C++ reference behavior | 0.2.0 status |
|---|---|---|---|
| SSD face detection | `detectors/ssd.py` | Caffe SSD, BGR mean, canonical thresholds/padding, largest-area ordering | Equivalent for pinned profile |
| 3DDFA pose | `pose/tddfa.py` | Canonical crop, float32 normalization, 62-parameter output, Euler conversion | Float32 path matches the tested CPU profile |
| ADNet landmarks | `landmarks/adnet.py` | Square/pad, single-rounded `[-1,1]`, height-based float32 back-projection | Matches the tested C++ conversion path |
| Alignment | `align.py` | Five reference points, LMEDS partial affine, 616-square warp | Equivalent for pinned runtime |
| Face parsing | `segmentation/parsing.py` | Canonical crop, RGB/ImageNet normalization, BiSeNet argmax | Equivalent for pinned model |
| Occlusion | `segmentation/occlusion.py` | Canonical crop/scale/threshold/nearest resize/border | Equivalent for single-output pinned model |
| Face-region mask | `align.landmarked_region` | Convex-hull mask with canonical `alpha=0` | Equivalent canonical behavior |
| Luminance | `align.luminance` | sRGB EOTF, Rec.709 weights, C++ rounding | Equivalent values; Python recomputes on demand |

## Complete component crosswalk

Every row below had 28/28 exact BSI scalar values and 28/28 matching statuses in the
tested environment.

| Output | Python owner | C++ measure | Important contract note |
|---|---|---|---|
| UnifiedQualityScore | `measures/core.py` | `UnifiedQualityScore` | Canonical MagFace crop/model/mapping |
| BackgroundUniformity | `measures/pixel.py` | `BackgroundUniformity` | Canonical parsing/background masks |
| IlluminationUniformity | `measures/pixel.py` | `IlluminationUniformity` | Canonical cheek histograms |
| LuminanceMean | `measures/core.py` | `Luminance` | Canonical double-sigmoid mapping |
| LuminanceVariance | `measures/pixel.py` | `Luminance` | Canonical sine mapping |
| UnderExposurePrevention | `measures/pixel.py` | `UnderExposurePrevention` | Region/occlusion intersection |
| OverExposurePrevention | `measures/pixel.py` | `OverExposurePrevention` | Canonical reciprocal mapping |
| DynamicRange | `measures/pixel.py` | `DynamicRange` | Canonical entropy mapping |
| Sharpness | `measures/models.py` | `Sharpness` | Exact canonical RTrees profile only |
| CompressionArtifacts | `measures/core.py` | `CompressionArtifacts` | Exact canonical crop/model only |
| NaturalColour | `measures/pixel.py` | `NaturalColour` | Corrected C++ coefficient `24289/27` |
| SingleFacePresent | `measures/geometry.py` | `SingleFacePresent` | Second/largest detected area ratio |
| EyesOpen | `measures/geometry.py` | `EyesOpen` | Aligned eyelid geometry |
| MouthClosed | `measures/geometry.py` | `MouthClosed` | Original-landmark mouth opening |
| EyesVisible | `measures/models.py` | `EyesVisible` | Canonical occlusion zones |
| MouthOcclusionPrevention | `measures/models.py` | `MouthOcclusionPrevention` | Canonical mouth polygon |
| FaceOcclusionPrevention | `measures/models.py` | `FaceOcclusionPrevention` | Canonical landmarked region |
| InterEyeDistance | `measures/geometry.py` | `InterEyeDistance` | Includes OFIQ pose-slot convention |
| HeadSize | `measures/core.py` | `HeadSize` | Original-image height normalization |
| LeftwardCropOfTheFaceImage | `measures/geometry.py` | `CropOfTheFaceImage` | Compound four-output failure group |
| RightwardCropOfTheFaceImage | `measures/geometry.py` | `CropOfTheFaceImage` | Compound four-output failure group |
| MarginAboveOfTheFaceImage | `measures/geometry.py` | `CropOfTheFaceImage` | Compound four-output failure group |
| MarginBelowOfTheFaceImage | `measures/geometry.py` | `CropOfTheFaceImage` | Compound four-output failure group |
| HeadPoseYaw | `measures/core.py` | `HeadPose` | Reproduces OFIQ yaw/pitch slot swap |
| HeadPosePitch | `measures/core.py` | `HeadPose` | Reproduces OFIQ yaw/pitch slot swap |
| HeadPoseRoll | `measures/core.py` | `HeadPose` | Canonical cosine-square mapping |
| ExpressionNeutrality | `measures/models.py` | `ExpressionNeutrality` | Dual EfficientNet plus AdaBoost |
| NoHeadCoverings | `measures/core.py` | `NoHeadCoverings` | Exact canonical thresholds only |

## Baseline findings and disposition

| Baseline finding | Disposition |
|---|---|
| Historical reporter skipped missing/no-face/NaN observations and never failed | Replaced by strict cardinality and nonzero verdict |
| Historical parity prose exceeded committed evidence | Removed; large historical run excluded from release evidence |
| Config measure selection and overrides were ignored | Scope narrowed; noncanonical config/model bytes rejected |
| One component exception discarded all results in CLI/batch | Per-component isolation plus typed image boundary |
| CLI/batch encoded systemic errors as valid-looking whole-image FTA | Hard image failures now exit nonzero; no sentinel row for failed item |
| Basename CSV identity corrupted recursive resume | Full path identity, quoting, row/header/duplicate validation |
| Default process count could multiply a 453,497,937-byte model profile plus runtime state by CPU count | Default one worker; explicit clean-spawn workers only |
| `NaturalColour` used `24389/27` rather than C++ `24289/27` | Corrected and source-gated |
| NumPy's two-step ADNet normalization changed a boundary landmark and two Sharpness tree votes on a real CelebA image | Reproduced against a native preprocessing bridge; changed to the C++ single-rounding float32 path |
| Pose constants and intermediates used float64 where OFIQ uses float32 | Changed to the C++ float32 evaluation path and real-image raw regression-gated |
| NumPy scalar normalization and optimized wheel resize kernels diverged from OFIQ's OpenCV CPU path | Changed affected model preprocessing to OpenCV scalar operations and the tested generic CPU kernel path |
| Config/models/version/distribution were not cryptographically tied to evidence | Canonical manifest, one version source, and report bindings |
| Decompressed OpenCV model files leaked | Temporary artifacts removed in `finally` |
| Optional native diagnostic failed with undefined names when unbuilt | Defined availability error; real bridge still executes when present |
| Release could publish source-tested but untested wheel or mismatched tag | Fresh wheel install, real wheel gate, tag/version check |

## Runtime performance review

The performance check compared the pinned 0.1.1 source with the 0.2.0 Python
source on all 28 real BSI conformance images and a fixed 64-image real CelebA subsample.
One warm-up preceded three alternating measured repetitions per variant and scenario.
Every case passed the source/input hash and output-cardinality contracts.

Single-process scalar digests were identical. Version 0.2.0 improved median warm
throughput from 2.004 to 2.060 images/s and reduced median peak process-tree RSS from
1.467 to 1.211 GiB; cold first assessment increased from 2.601 to 2.693 seconds. In batch
mode, version 0.2.0 throughput was 2.649, 2.391, and 2.140 images/s at one, two, and four
workers, while RSS was 1.292, 2.450, and 4.723 GiB. Those measurements support one worker
as the default on the benchmark host. The full bounded interpretation is in
[Runtime performance](performance.md).

## Scope limits

- This is canonical-profile parity, not general JAXN or C++ API parity.
- Raw conformance is bounded by the named CPU component policy; the bounds are not a claim
  about unreviewed accelerators, operating systems, or dependency versions.
- Models remain separately licensed and are not included in the wheel.
- The complete live release matrix is Linux/Python 3.11 and 3.12. Later Python versions
  and other operating systems do not yet receive the complete live release gate.
- The original 1,197-image claim remains historically unbound. A new aggregate-bound run
  verifies the recovered current bytes, but no old input hash exists to prove the two byte
  sets identical; the licensed images and per-image manifest are not redistributed.
- The recovered CelebA run is in-sample evidence for the raw policy because it contributed
  to diagnosis and bound selection; an independent real-data cohort remains necessary for
  external validation of those bounds.
- `native_cv` is an operator-built diagnostic surface, not part of canonical execution.

These limits define the public contract.
