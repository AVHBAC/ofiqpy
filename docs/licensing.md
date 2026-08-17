# Licensing and attribution

## ofiqpy

ofiqpy is released under the **MIT License** (see [`LICENSE`](https://github.com/AVHBAC/ofiqpy/blob/main/LICENSE)).

## OFIQ

ofiqpy is a Python reimplementation of the canonical **OFIQ** v1.1.0 profile. OFIQ is developed by the
**German Federal Office for Information Security (BSI)**, Copyright © 2024, and MIT-licensed:

> <https://github.com/BSI-OFIQ/OFIQ-Project>

The implementation derives its algorithms and model contract from OFIQ. Please cite or
acknowledge OFIQ when using ofiqpy.

## Models are not bundled

ofiqpy **does not redistribute OFIQ's model files.** They ship with OFIQ and may be
licensed separately from the OFIQ source (see OFIQ's `LICENSE.md`, which documents the
license situation of individual dependencies and model files). ofiqpy loads them at runtime
from a directory you provide:

```bash
export OFIQPY_OFIQ_ROOT="$(cd ../OFIQ-Project && pwd)"
export OFIQPY_OFIQ_DATA="$OFIQPY_OFIQ_ROOT/data"
```

The models include third-party components (e.g. ADNet, 3DDFA-V2, BiSeNet, HSEmotion,
MagFace) under their own licenses. Consult OFIQ's licensing documentation before
redistributing any model file.

## CelebA diagnostic data

The optional large-corpus diagnostic uses a locally held CelebA cohort under the
[official CelebA agreement](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). It is used
only for non-commercial research verification. The images and per-image identity/hash
manifest are not included in this repository, distributions, or default reports.

## Standard

ISO/IEC 29794-5 is a standard of ISO/IEC. ofiqpy is an independent implementation and is not
endorsed by ISO, IEC, or BSI.
