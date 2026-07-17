"""ctypes bridge to OFIQ's own conan-built OpenCV 4.5.5 (native/ofiq_cv.so).

Using OFIQ's exact compiled estimateAffinePartial2D/warpAffine/resize makes the
alignment bit-identical to OFIQ, closing the periphery/landmark parity gap.
Falls back to pip cv2 if the .so is absent.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_SO = Path(__file__).parent.parent / "native" / "ofiq_cv.so"
AVAILABLE = _SO.exists()

if AVAILABLE:
    _lib = ctypes.CDLL(str(_SO))
    _dp = ctypes.POINTER(ctypes.c_double)
    _up = ctypes.POINTER(ctypes.c_ubyte)
    _lib.estimate_affine_partial2d.restype = ctypes.c_int
    _lib.estimate_affine_partial2d.argtypes = [_dp, _dp, ctypes.c_int, _dp]
    _lib.warp_affine.restype = None
    _lib.warp_affine.argtypes = [_up, ctypes.c_int, ctypes.c_int, ctypes.c_int, _dp,
                                 _up, ctypes.c_int, ctypes.c_int]
    _lib.resize_linear.restype = None
    _lib.resize_linear.argtypes = [_up, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   _up, ctypes.c_int, ctypes.c_int]


def _d(a):
    return np.ascontiguousarray(a, np.float64).ctypes.data_as(_dp)


def _u(a):
    return np.ascontiguousarray(a, np.uint8).ctypes.data_as(_up)


def estimate_affine_partial2d(src, dst):
    src = np.ascontiguousarray(src, np.float64)
    dst = np.ascontiguousarray(dst, np.float64)
    out = np.zeros(6, np.float64)
    r = _lib.estimate_affine_partial2d(_d(src), _d(dst), len(src),
                                       out.ctypes.data_as(_dp))
    return None if r != 0 else out.reshape(2, 3)


def warp_affine(img, M, oh, ow):
    img = np.ascontiguousarray(img, np.uint8)
    h, w = img.shape[:2]
    ch = img.shape[2] if img.ndim == 3 else 1
    out = np.zeros((oh, ow, ch) if ch == 3 else (oh, ow), np.uint8)
    _lib.warp_affine(_u(img), h, w, ch, _d(M), _u(out), oh, ow)
    return out


def resize_linear(img, oh, ow):
    img = np.ascontiguousarray(img, np.uint8)
    h, w = img.shape[:2]
    ch = img.shape[2] if img.ndim == 3 else 1
    out = np.zeros((oh, ow, ch) if ch == 3 else (oh, ow), np.uint8)
    _lib.resize_linear(_u(img), h, w, ch, _u(out), oh, ow)
    return out
