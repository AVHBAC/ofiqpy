"""Real OFIQ v1.1.0 test-artifact discovery."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

_BSI_IMAGE_SHA256 = {
    "c-02-headcovered-mouthopen.png": "7538c9cd0052f9eb653f1f57a3cf5ccbc666a8f7d3a6bdc8742aa86b9b199121",
    "c-10-background.png": "d99984e3195469630d2357992d993f160e145aad2aaba48692660905ceab37e7",
    "r-01-frontal.png": "3c3655e6ac9f16bc6909b6f71d7c5ec385008aa6d2352167bdb77acd61d8c947",
    "r-02-rolled.png": "10fbd729419c626354af5180704b07f8a9782b6c061d0fc11ce38a0de044cbd2",
    "r-07-scarf.png": "94d1916b3ce7d9c3a87a52884f4a82c28219926696863e8541b111be3906a2ac",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_root() -> Path:
    configured = os.environ.get("OFIQPY_OFIQ_ROOT")
    root = Path(configured) if configured else Path(__file__).resolve().parents[2] / "OFIQ-Project"
    required = (
        root / "data" / "ofiq_config.jaxn",
        root / "data" / "tests" / "images" / "r-01-frontal.png",
        root / "OFIQlib" / "modules" / "utils" / "src" / "image_utils.cpp",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"real OFIQ v1.1.0 test artifacts are unavailable: {missing}")
    return root


def bsi_image(name: str) -> Path:
    expected = _BSI_IMAGE_SHA256[name]
    path = official_root() / "data" / "tests" / "images" / name
    digest = _sha256(path)
    if digest != expected:
        raise RuntimeError(f"BSI contract image hash mismatch for {name}: expected {expected}, got {digest}")
    return path


def historical_celeba_image_000189() -> Path:
    """Locate the licensed, non-redistributed Sharpness boundary capture."""
    configured = os.environ.get("OFIQPY_CELEBA_ROOT")
    if not configured:
        pytest.skip("OFIQPY_CELEBA_ROOT is required for the licensed CelebA regression")
    path = Path(configured) / "000189.jpg"
    if not path.is_file():
        raise RuntimeError(f"real CelebA regression image is unavailable: {path}")
    expected = "751959876229d462612ee446257be0a69c7a5aeb96c2a04db473b75126594120"
    digest = _sha256(path)
    if digest != expected:
        raise RuntimeError(f"CelebA 000189 hash mismatch: expected {expected}, got {digest}")
    return path
