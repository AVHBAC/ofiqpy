# Changelog

All notable changes to ofiqpy are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semantic versioning.

## [0.2.0] — 2026-08-14

### Changed

- Defined support as the exact canonical OFIQ v1.1.0 config/model profile; initialization
  now verifies the config and all twelve required model artifacts by SHA-256 and size.
- Added preflighted, thread-safe `Assessor` lifecycle and typed image/component results.
- Isolated component failures so unaffected component results are preserved.
- Corrected the `NaturalColour` CIELAB coefficient to OFIQ v1.1.0's `24289/27`.
- Made package version metadata read from one runtime source.
- Preserved full CSV path identities, added standards-compliant quoting, validated resume
  headers/rows/duplicates, changed the batch default to one worker, and made explicit
  multi-worker runs use a clean `spawn` process context.
- Preserved legacy mapping exceptions for unreadable/invalid inputs, made CLI/batch hard
  image failures exit nonzero, and defined the unavailable-native-diagnostic error.
- Replaced the non-asserting reporter and fabricated smoke values with hash-bound real-data
  tests and a cardinality-enforcing live-OFIQ gate.
- Added executed real-data notebook and release-tag/package-version gates.
- Raised the declared and CI-tested Python support floor from 3.9 to 3.11; the static
  and complete live matrices now cover Python 3.11 and 3.12.
- Restricted release evidence to reproducibly bound runs. The current 28-image BSI gate is
  784/784 scalar exact and 784/784 status exact; raw values are 702/784 exact at six-decimal
  CSV precision and 784/784 within the named component policy.
- Standardized every internal measure producer on one immutable named raw/scalar contract
  and made public assessment mappings immutable with finite/range validation.
- Made batch execution re-entrant in the parent process, monotonic-timed, streaming, and
  silent by default; added typed progress/report contracts and aggregate input provenance
  with explicit opt-in for per-image disclosure.
- Defined raw comparison only for pairs where both implementations successfully assessed
  the component; undefined FailureToAssess raw values are counted and excluded explicitly.
- Reproduced and corrected C++/Python preprocessing differences in ADNet normalization and
  back-projection, 3DDFA float32 evaluation, CompressionArtifacts/ExpressionNeutrality
  normalization, and the reviewed OpenCV 4.5.5 CPU resize path.
- Replaced mutable CI action tags and pip resolution with commit-pinned actions and a
  hash-bearing `uv.lock`; release artifacts now use locked build/runtime dependencies.
- Made conformance capture source, input, model, native, and optional distribution
  bindings before and after execution and reject any mid-run artifact change.
- Recovered and aggregate-bound the current 1,197-image CelebA diagnostic without
  redistributing licensed images or per-image records: 33,516/33,516 statuses matched,
  33,483 scalars were exact and all were within one point, and all 33,430 defined raw
  comparisons met the named policy. The original unhashed run remains historically
  unproven, and this cohort is not independent validation of the raw bounds.
- Added an aggregate-only runtime benchmark over real BSI and CelebA inputs. On the
  reviewed host, one worker was faster and materially smaller than two or four workers,
  confirming the bounded one-worker batch default.

## [0.1.1] — 2026-07-17

### Added
- Source distribution now ships the full project (CHANGELOG, CITATION, docs sources,
  examples, notebooks, native bridge sources, and the test suite) via `MANIFEST.in`.
- CI: full lint + type pipeline (`ruff check` + `ruff format --check` + `mypy`).

### Fixed
- Removed local development paths from `config.py` and the original comparison reporter in
  favour of environment-driven data/reference roots.
- Minor typing fixes (array dtypes, worker None-guards).

## [0.1.0] — 2026-07-17

Initial Python implementation of OFIQ v1.1.0 algorithms.

### Added
- Full pipeline: SSD face detection, ADNet-98 landmarks, 5-point LMEDS alignment to
  616×616, 3DDFA-V2 head pose, BiSeNet face parsing, face-occlusion segmentation, and the
  landmarked-region mask.
- All 27 ISO/IEC 29794-5 quality components plus the MagFace unified quality score.
- `assess()` Python API, a canonical-profile CSV CLI, and a parallel, resumable
  batch runner.
- OFIQ-format CSV output (named columns, raw + `.scalar` per component).
- Initial live-OFIQ comparison reporter and unit tests.
- Environment-driven model path (`OFIQPY_OFIQ_DATA`); models are not bundled.
- Documentation site (mkdocs-material), example notebooks, and CI / PyPI-publish workflows.

The initial release notes reported a 1,197-image CelebA comparison. That run was not bound
to exact source, executable, model, and input manifests by the committed reporter, so the
current release process does not treat it as reproducible evidence.
