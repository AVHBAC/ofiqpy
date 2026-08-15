"""Python implementation of the hash-verified canonical OFIQ v1.1.0 profile."""

from __future__ import annotations

import threading

from ._version import __version__
from .assessor import Assessor
from .results import AssessmentResult, AssessmentStatus, ComponentResult, ComponentStatus, FailureCode

_ASSESSOR: Assessor | None = None
_ASSESSOR_LOCK = threading.Lock()


def _lazy():
    """Build one preflighted, internally locked assessor."""
    global _ASSESSOR
    if _ASSESSOR is None:
        with _ASSESSOR_LOCK:
            if _ASSESSOR is None:
                _ASSESSOR = Assessor()
    return _ASSESSOR


def assess_typed(image: "str | object") -> AssessmentResult:
    """Assess one image and retain typed image/component failure states."""
    return _lazy().assess(image)


def assess(image: "str | object") -> dict:
    """Assess one image and return ``{component_name: (raw, scalar)}``.

    Args:
        image: path to an image file (str or PathLike), or a BGR uint8 numpy
            array of shape (H, W, 3).

    Returns:
        Mapping of OFIQ component name to ``(raw native value, 0-100 scalar)``. Empty
        only if no face is detected.

    Raises:
        FileNotFoundError: The supplied path cannot be decoded as an image.
        ValueError: The supplied array is not a non-empty BGR uint8 image.
        RuntimeError: Canonical preprocessing fails before component assessment.
    """
    result = assess_typed(image)
    if result.failure is not None:
        if result.failure.code is FailureCode.NO_FACE:
            return {}
        if result.failure.code is FailureCode.IMAGE_READ_ERROR:
            raise FileNotFoundError(result.failure.message)
        if result.failure.code is FailureCode.INVALID_IMAGE:
            raise ValueError(result.failure.message)
        raise RuntimeError(f"{result.failure.code.value}: {result.failure.message}")
    return result.as_legacy_dict()


__all__ = [
    "AssessmentResult",
    "AssessmentStatus",
    "Assessor",
    "ComponentResult",
    "ComponentStatus",
    "__version__",
    "assess",
    "assess_typed",
]
