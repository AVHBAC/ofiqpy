"""OFIQ config loader — parses ofiq_config.jaxn and resolves model paths.

Faithful port of OFIQ v1.1.0. Model files and config are reused directly from the
reference OFIQ-Project checkout so the port loads the *same* weights OFIQ uses.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# OFIQ's models + config are reused directly (they are separately licensed and NOT bundled).
# Point ofiqpy at an OFIQ checkout's data/ directory via the OFIQPY_OFIQ_DATA env var, e.g.:
#     export OFIQPY_OFIQ_DATA=/path/to/OFIQ-Project/data
# Falls back to a local development checkout if the env var is unset.
_DEFAULT_DATA = Path("/mnt/projects/02_perception_biometrics/OFIQ-Project/data")
OFIQ_DATA = Path(os.environ.get("OFIQPY_OFIQ_DATA", _DEFAULT_DATA))
OFIQ_CONFIG = OFIQ_DATA / "ofiq_config.jaxn"


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


def load_config(path: Path = OFIQ_CONFIG) -> dict:
    raw = Path(path).read_text()
    data = json.loads(_strip_jaxn(raw))
    return data["config"]


class OFIQConfig:
    """Parsed OFIQ config with model-path resolution against the OFIQ data root."""

    def __init__(self, path: Path = OFIQ_CONFIG, data_root: Path = OFIQ_DATA):
        self.cfg = load_config(path)
        self.data_root = Path(data_root)
        self.params = self.cfg["params"]

    def resolve(self, rel_path: str) -> Path:
        p = self.data_root / rel_path
        if not p.exists():
            raise FileNotFoundError(f"OFIQ model not found: {p}")
        return p

    def measure(self, name: str) -> dict:
        return self.params["measures"].get(name, {})

    def sigmoid_params(self, measure: str) -> dict:
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
