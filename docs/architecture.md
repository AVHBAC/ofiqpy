# Architecture

## Runtime lifecycle

```text
canonical data root
  -> hash config + 12 model artifacts
  -> load detector, landmark, pose, parsing, occlusion, and measure models
  -> Assessor (one re-entrant lock around mutable inference sessions)
       -> read and validate BGR uint8 image
       -> preprocessing Session
       -> 28 isolated component results
       -> typed AssessmentResult
            -> Python typed API
            -> legacy mapping adapter
            -> canonical CSV adapter
```

`Assessor` is the public lifecycle boundary. Construction fails before assessment output is
created if the config/model profile is missing or does not hash to the reviewed OFIQ v1.1.0
artifacts. Model sessions are preloaded, so corrupt model serialization also fails at
initialization rather than halfway through a batch.

OpenCV DNN networks and several ONNX/`cv2.ml` objects are mutable during inference. One
`Assessor` therefore serializes assessment with an `RLock`. Batch workers each own a
separate `Assessor`; the default is one worker because a model graph consumes substantial
memory. Multi-worker batches use a clean `spawn` context, never a fork of already-created
ONNX/OpenCV threads.

The default is also measured rather than precautionary alone. On the reviewed 64-image
real-data workload, candidate throughput was 2.649 images/s with one worker, 2.391 with
two, and 2.140 with four, while median process-tree RSS rose from 1.292 GiB to 2.450 and
4.723 GiB. See [Runtime performance](performance.md) for the complete host, input,
alternation, and idle-guard contract.

Construction also disables OpenCV's process-wide optimized kernels with
`cv2.setUseOptimized(False)`. The `opencv-python-headless` 4.5.5 wheel's optimized
float-resize path differs numerically from OFIQ's reviewed Conan OpenCV 4.5.5 CPU build;
the generic path produced matching reviewed tensors. This setting is idempotent but global:
applications that share a process with other OpenCV workloads must account for the
performance and last-bit effects. Separate worker processes contain that setting.

## Preprocessing graph

```text
BGR uint8 image
  -> SSD face detection
  -> 3DDFA-V2 pose
  -> ADNet-98 landmarks
  -> five-point LMEDS alignment (616 x 616)
  -> BiSeNet parsing (400 x 400)
  -> face-occlusion segmentation (616 x 616)
  -> landmarked face-region mask
  -> component executor
```

Each product is stored once on `Session` and reused by dependent measures. The component
executor catches exceptions at the same single/compound-measure granularity used by OFIQ:
crop produces four results, head pose produces three, and the remaining measures produce
one or two defined results. A failed group becomes typed `FailureToAssess`; other groups
remain available.

## C++ comparison

| Concern | OFIQ C++ v1.1.0 | ofiqpy supported profile |
|---|---|---|
| Configuration | Selectable measures and parameter overrides | Exact canonical config hash only |
| Executor | Builds configured measure list | Fixed canonical 28 outputs |
| Result status | `QualityMeasureReturnCode` per component | Typed component status plus image status |
| Measure failure | Isolated in `Executor` | Isolated per single/compound group |
| Preprocessing failure | All configured measures FTA | All 28 components FTA |
| Thread safety | Internal synchronization around mutable runtime | `Assessor`-level `RLock` |
| CSV identity | Supplied image path | Supplied/discovered image path |
| CSV columns | Active configured map | Fixed canonical 28-component map |

This is algorithm/profile parity, not full API/config parity. Passing a modified JAXN file
is rejected even if it would be accepted by OFIQ C++.

## Package map

```text
ofiqpy/
  profile.py       canonical artifact manifest and integrity verification
  config.py        JAXN reader and verified model resolver
  assessor.py      preflight, locking, input validation, typed lifecycle
  results.py       component/image statuses and result types
  pipeline.py      OFIQ preprocessing graph
  session.py       shared preprocessing products
  measures/        canonical component executor and algorithms
  output.py        canonical semicolon CSV encoding
  batch.py         recursive/resumable execution
  conformance.py   live reference runner and strict comparison report
```
