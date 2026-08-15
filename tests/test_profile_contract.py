"""Canonical OFIQ v1.1.0 profile contracts against the real BSI artifacts."""

from __future__ import annotations

from conftest import official_root

from ofiqpy.config import OFIQConfig
from ofiqpy.profile import CANONICAL_PROFILE
from ofiqpy.raw_policy import CANONICAL_RAW_TOLERANCE_PROFILE
from ofiqpy.results import OFIQ_COMPONENTS


def test_canonical_profile_verifies_real_bsi_artifacts() -> None:
    data_root = official_root() / "data"
    report = CANONICAL_PROFILE.verify(data_root)

    assert report.profile_id == "bsi-ofiq-v1.1.0-canonical"
    assert report.config_sha256 == "e117286706d799a1e23130db01cfdff7ef36d157b22ecf68dd196052a8a8d0b3"
    assert report.verified_artifacts == 12
    assert report.verified_bytes == 453_497_937


def test_data_root_selects_config_and_models_as_one_profile() -> None:
    data_root = official_root() / "data"
    config = OFIQConfig(data_root=data_root)

    assert config.data_root == data_root
    assert config.path == data_root / "ofiq_config.jaxn"
    assert config.profile.profile_id == "bsi-ofiq-v1.1.0-canonical"


def test_raw_tolerance_profile_is_complete_immutable_and_explained() -> None:
    profile = CANONICAL_RAW_TOLERANCE_PROFILE

    assert profile.profile_id == "bsi-ofiq-v1.1.0-cpu-raw-csv-v1"
    assert profile.decimal_places == 6
    assert tuple(profile.components) == OFIQ_COMPONENTS
    assert all(policy.absolute_tolerance >= 0 for policy in profile.components.values())
    assert all(policy.rationale.strip() for policy in profile.components.values())

    try:
        profile.components["Sharpness"] = profile.components["Sharpness"]  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("raw tolerance mapping must be immutable")
