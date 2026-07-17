"""ofiqpy — a faithful Python port of OFIQ v1.1.0 (ISO/IEC 29794-5 face image quality).

Reuses OFIQ's own models and reproduces its exact algorithms, matching the reference
within +/- 1 quality point per component. Point ofiqpy at an OFIQ install's data/ dir:

    export OFIQPY_OFIQ_DATA=/path/to/OFIQ-Project/data

Then:

    from ofiqpy import assess
    scores = assess("face.jpg")   # {component_name: (raw, scalar)}
"""
from __future__ import annotations

__version__ = "0.1.0"

_PIPE = None
_MEAS = None


def _lazy():
    """Build (once) the shared pipeline + measures. Models load on first use."""
    global _PIPE, _MEAS
    if _PIPE is None:
        from .config import OFIQConfig
        from .measures.core import Measures
        from .pipeline import OFIQPipeline
        cfg = OFIQConfig()
        _PIPE = OFIQPipeline(cfg)
        _MEAS = Measures(cfg)
    return _PIPE, _MEAS


def assess(image: "str | object") -> dict:
    """Assess one image and return ``{component_name: (raw, scalar)}``.

    Args:
        image: path to an image file (str or PathLike), or a BGR uint8 numpy
            array of shape (H, W, 3).

    Returns:
        Mapping of OFIQ component name to ``(raw native value, 0-100 scalar)``. Empty
        if no face is detected.
    """
    import numpy as np

    pipe, meas = _lazy()
    if isinstance(image, np.ndarray):
        bgr = image
    else:
        import cv2
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(f"could not read image: {image}")
    session = pipe.process(bgr)
    if session.bbox is None:
        return {}
    return meas.compute(session)


__all__ = ["assess", "__version__"]
