# Changelog

All notable changes to ofiqpy are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semantic versioning.

## [0.1.0] — 2026-07-17

Initial release — a faithful Python port of OFIQ v1.1.0 (ISO/IEC 29794-5).

### Added
- Full pipeline: SSD face detection, ADNet-98 landmarks, 5-point LMEDS alignment to
  616×616, 3DDFA-V2 head pose, BiSeNet face parsing, face-occlusion segmentation, and the
  landmarked-region mask.
- All 27 ISO/IEC 29794-5 quality components plus the MagFace unified quality score,
  ported line-faithfully from OFIQ's C++.
- `assess()` Python API, an `OFIQSampleApp`-compatible CLI, and a parallel, resumable
  batch runner.
- OFIQ-format CSV output (named columns, raw + `.scalar` per component).
- Live-OFIQ ±1 conformance gate harness and pure-function unit tests.
- Environment-driven model path (`OFIQPY_OFIQ_DATA`); models are not bundled.
- Documentation site (mkdocs-material), example notebooks, and CI / PyPI-publish workflows.

### Validated
- 1,197 real CelebA images vs live OFIQ: 27/28 components conformant to ISO Annex A ±1
  (24 bit-exact); ~99.99% of component-image pairs within ±1.
