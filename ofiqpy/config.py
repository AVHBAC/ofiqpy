"""Load and resolve the hash-verified canonical OFIQ v1.1.0 data profile."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .profile import CANONICAL_PROFILE, CanonicalProfile, ProfileVerification

_DEFAULT_DATA = Path("OFIQ-Project/data")


def default_data_root() -> Path:
    """Resolve the OFIQ data directory at call time, not module-import time."""
    return Path(os.environ.get("OFIQPY_OFIQ_DATA", _DEFAULT_DATA))


def _strip_jaxn(text: str) -> str:
    """Strip // and /* */ comments and trailing commas from JAXN, respecting strings."""
    out = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:  # escape
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":  # line comment
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":  # block comment
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    s = "".join(out)
    # remove trailing commas before } or ]
    import re

    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s


def load_config(path: Path) -> dict:
    raw = Path(path).read_text()
    data = json.loads(_strip_jaxn(raw))
    return data["config"]


class OFIQConfig:
    """Verified canonical OFIQ v1.1.0 config and model-path resolver."""

    def __init__(self, path: Path | None = None, data_root: Path | None = None):
        if data_root is None:
            data_root = Path(path).parent if path is not None else default_data_root()
        self.data_root = Path(data_root)
        canonical_path = self.data_root / "ofiq_config.jaxn"
        self.path = Path(path) if path is not None else canonical_path
        if self.path != canonical_path:
            raise ValueError(
                f"the canonical profile requires config and models from one data root: expected {canonical_path}, got {self.path}"
            )
        self.profile: CanonicalProfile = CANONICAL_PROFILE
        self.verification: ProfileVerification = self.profile.verify(self.data_root)
        self.cfg = load_config(self.path)
        self.params = self.cfg["params"]

    def resolve(self, rel_path: str) -> Path:
        p = self.data_root / rel_path
        if not p.exists():
            raise FileNotFoundError(f"OFIQ model not found: {p}")
        return p

    def measure(self, name: str) -> dict:
        return self.params["measures"].get(name, {})

    def sigmoid_params(self, measure: str) -> dict:
        """Return canonical JAXN values for inspection; runtime overrides are unsupported."""
        return self.measure(measure).get("Sigmoid", {})

    def detector(self) -> dict:
        return self.params["detector"]["ssd"]

    def landmarks_model(self) -> Path:
        return self.resolve(self.params["landmarks"]["ADNet"]["model_path"])


if __name__ == "__main__":
    c = OFIQConfig()
    print("detector:", c.detector())
    print("ADNet:", c.landmarks_model())
    print("UnifiedQualityScore sigmoid:", c.sigmoid_params("UnifiedQualityScore"))
    print("measures listed:", len(c.cfg["measures"]))
