# Licensing & Attribution

## ofiqpy

ofiqpy is released under the **MIT License** (see [`LICENSE`](https://github.com/AVHBAC/ofiqpy/blob/main/LICENSE)).

## OFIQ

ofiqpy is a faithful port of **OFIQ** (Open Source Face Image Quality), developed by the
**German Federal Office for Information Security (BSI)**, Copyright © 2024, and MIT-licensed:

> <https://github.com/BSI-OFIQ/OFIQ-Project>

The port reproduces OFIQ's ISO/IEC 29794-5 algorithms in Python. Please cite / acknowledge
OFIQ when using ofiqpy.

## Models are not bundled

ofiqpy **does not redistribute OFIQ's model files.** They ship with OFIQ and may be
licensed separately from the OFIQ source (see OFIQ's `LICENSE.md`, which documents the
license situation of individual dependencies and model files). ofiqpy loads them at runtime
from a directory you provide:

```bash
export OFIQPY_OFIQ_DATA=/path/to/OFIQ-Project/data
```

The models include third-party components (e.g. ADNet, 3DDFA-V2, BiSeNet, HSEmotion,
MagFace) under their own licenses. Consult OFIQ's licensing documentation before
redistributing any model file.

## Standard

ISO/IEC 29794-5 is a standard of ISO/IEC. ofiqpy is an independent implementation and is not
endorsed by ISO, IEC, or BSI.
