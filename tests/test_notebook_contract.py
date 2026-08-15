"""Execute the pipeline notebook against the hash-bound real BSI profile."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from conftest import official_root

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_notebook_executes_on_real_bsi_capture(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["OFIQPY_OFIQ_DATA"] = str(official_root() / "data")
    env["PYTHONPATH"] = str(ROOT)
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}"
    output = tmp_path / "02_pipeline_internals.executed.ipynb"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(ROOT / "notebooks" / "02_pipeline_internals.ipynb"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
