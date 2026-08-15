"""Public-claim and release-governance contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "CITATION.cff", *sorted((ROOT / "docs").glob("*.md"))]


def test_public_claims_exclude_unbound_parity_language() -> None:
    current_docs = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_FILES if path.name != "CHANGELOG.md")

    for unsupported in (
        "behavior-preserving",
        "full coverage + near-exact parity",
        "~99.99%",
        "1,000+ real CelebA",
        "feature/embedding computations are bit-exact",
    ):
        assert unsupported not in current_docs


def test_public_artifacts_contain_no_unresolved_markers() -> None:
    public_text = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_FILES)

    markers = ("/path" + "/to", "[" + "TBD" + "]", "FIX" + "ME", "TO" + "DO", "<- " + "change")
    for marker in markers:
        assert marker not in public_text


def test_publish_and_pages_depend_on_the_real_installed_wheel_gate() -> None:
    workflow = (ROOT / ".github" / "workflows" / "workflow.yml").read_text(encoding="utf-8")

    assert "ref: bb5dc91d00477e02ce53d2530d28e35021484393" in workflow
    assert "Run official OFIQ conformance suite" in workflow
    assert "Run all real-data ofiqpy contracts" in workflow
    assert "Execute real-data notebooks" in workflow
    assert ("LD_LIBRARY_PATH: ${{ github.workspace }}/_reference/OFIQ-Project/install_x86_64_linux/Release/lib") in workflow
    assert 'python-version: ["3.11", "3.12"]' in workflow
    assert "python-version: ${{ matrix.python-version }}" in workflow
    assert "$RUNNER_TEMP/ofiqpy-notebooks" in workflow
    assert "Verify source checkout remains clean" in workflow
    assert "Run installed-wheel strict gate" in workflow
    assert "Verify release tag matches package version" in workflow
    assert workflow.count("needs: real-gate") == 2
    assert "workflow_dispatch:" in workflow
    assert "group: pages" in workflow


def test_ci_actions_and_dependency_resolution_are_immutable() -> None:
    workflow = (ROOT / ".github" / "workflows" / "workflow.yml").read_text(encoding="utf-8")
    uses = [line.strip() for line in workflow.splitlines() if line.strip().startswith("uses:")]

    assert uses
    assert all(re.fullmatch(r"uses: [^@\s]+@[0-9a-f]{40}(?:\s+#\s+v?[^\s]+)?", line) for line in uses)
    assert "@latest" not in workflow
    assert "@release/" not in workflow
    assert "python -m pip install" not in workflow
    assert "astral-sh/setup-uv@" in workflow
    assert 'version: "0.9.26"' in workflow
    assert workflow.count("uv sync --locked") >= 3
    assert "--scalar-tolerance 0" in workflow


def test_live_gate_runs_on_supported_python_minors() -> None:
    workflow = (ROOT / ".github" / "workflows" / "workflow.yml").read_text(encoding="utf-8")
    real_gate = workflow.split("  real-gate:", 1)[1].split("\n  docs:", 1)[0]

    assert 'python-version: ["3.11", "3.12"]' in real_gate
    assert "python-version: ${{ matrix.python-version }}" in real_gate


def test_python_support_starts_at_3_11() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.11"' in pyproject
    assert 'target-version = "py311"' in pyproject
    assert 'python_version = "3.11"' in pyproject


def test_runtime_benchmark_is_real_aggregate_only_evidence() -> None:
    evidence_path = ROOT / "docs" / "evidence" / "runtime-benchmark-20260815.json"
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert payload["schema"] == "ofiqpy.runtime-benchmark-aggregate.v1"
    assert payload["real_inputs"]["single"]["dataset"] == "BSI OFIQ v1.1.0 conformance images"
    assert payload["real_inputs"]["single"]["count"] == 28
    assert payload["real_inputs"]["batch"]["dataset"] == "CelebA original wild images"
    assert payload["real_inputs"]["batch"]["count"] == 64
    assert payload["method"]["measured_repetitions"] == 3
    assert payload["method"]["workers"] == [1, 2, 4]
    assert (
        payload["single_image_medians"]["baseline"]["scalar_digest"]
        == (payload["single_image_medians"]["candidate"]["scalar_digest"])
    )
    for variant in ("baseline", "candidate"):
        for workers in ("1", "2", "4"):
            result = payload["batch_medians"][variant][workers]
            assert result["rows"] == 64
            assert result["all_failure_to_assess_rows"] == 1

    serialized = evidence_path.read_text(encoding="utf-8")
    assert "/mnt/" not in serialized
    assert "image_name" not in serialized
