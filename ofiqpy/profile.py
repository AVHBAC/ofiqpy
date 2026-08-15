"""Integrity contract for the supported OFIQ v1.1.0 data profile."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class ProfileVerificationError(RuntimeError):
    """The configured OFIQ data directory is not the supported canonical profile."""


@dataclass(frozen=True)
class ArtifactSpec:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ProfileVerification:
    profile_id: str
    data_root: Path
    config_sha256: str
    verified_artifacts: int
    verified_bytes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CanonicalProfile:
    profile_id: str
    config_sha256: str
    artifacts: tuple[ArtifactSpec, ...]

    def verify(self, data_root: Path) -> ProfileVerification:
        root = Path(data_root)
        config_path = root / "ofiq_config.jaxn"
        if not config_path.is_file():
            raise ProfileVerificationError(f"canonical OFIQ config is missing: {config_path}")
        actual_config_hash = _sha256(config_path)
        if actual_config_hash != self.config_sha256:
            raise ProfileVerificationError(
                f"OFIQ config hash mismatch for {config_path}: expected {self.config_sha256}, got {actual_config_hash}"
            )

        verified_bytes = 0
        for artifact in self.artifacts:
            path = root / artifact.relative_path
            if not path.is_file():
                raise ProfileVerificationError(f"canonical OFIQ artifact is missing: {path}")
            actual_size = path.stat().st_size
            if actual_size != artifact.size_bytes:
                raise ProfileVerificationError(
                    f"OFIQ artifact size mismatch for {path}: expected {artifact.size_bytes}, got {actual_size}"
                )
            actual_hash = _sha256(path)
            if actual_hash != artifact.sha256:
                raise ProfileVerificationError(
                    f"OFIQ artifact hash mismatch for {path}: expected {artifact.sha256}, got {actual_hash}"
                )
            verified_bytes += actual_size

        return ProfileVerification(
            profile_id=self.profile_id,
            data_root=root,
            config_sha256=actual_config_hash,
            verified_artifacts=len(self.artifacts),
            verified_bytes=verified_bytes,
        )


CANONICAL_PROFILE = CanonicalProfile(
    profile_id="bsi-ofiq-v1.1.0-canonical",
    config_sha256="e117286706d799a1e23130db01cfdff7ef36d157b22ecf68dd196052a8a8d0b3",
    artifacts=(
        ArtifactSpec(
            "models/face_detection/ssd_facedetect.caffemodel",
            "2a56a11a57a4a295956b0660b4a3d76bbdca2206c4961cea8efe7d95c7cb2f2d",
            10_666_211,
        ),
        ArtifactSpec(
            "models/face_detection/ssd_facedetect.prototxt.txt",
            "34d850436faf158ca4e690533ba27c1decf29653559299d80db9274146332ca7",
            29_881,
        ),
        ArtifactSpec(
            "models/face_landmark_estimation/ADNet.onnx",
            "5aed3eb5b6ecd3504e5216078a3db064f6c93f2a420f12d0e82f2184c64c9654",
            54_747_293,
        ),
        ArtifactSpec(
            "models/head_pose_estimation/mb1_120x120.onnx",
            "efeb77517d067c197a314bc3d72aa04f03ceb6468f121d6c9f185dd44736e4ae",
            13_046_240,
        ),
        ArtifactSpec(
            "models/face_occlusion_segmentation/face_occlusion_segmentation_ort.onnx",
            "e847464bd8c0c596e8921c18d303fc6bf0ddb4a127ba4d1213c19701ed8790d0",
            57_307_779,
        ),
        ArtifactSpec(
            "models/face_parsing/bisenet_400.onnx",
            "750ae55d5c440074a085ddd31a3fbd19a9e922ffb22d25eeb2bdf4f26eb3250a",
            53_193_813,
        ),
        ArtifactSpec(
            "models/unified_quality_score/magface_iresnet50_norm.onnx",
            "bc03f082a59ccb3189c6c9aec429f376523fe6586e207c401ab8d1694888babb",
            174_394_035,
        ),
        ArtifactSpec(
            "models/expression_neutrality/hsemotion/enet_b0_8_best_vgaf_embed_zeroed.onnx",
            "467e6c31152cd6db94f50c5d0b8e2ce4d1280f2fbd7b826609aefb92779a9e15",
            16_052_940,
        ),
        ArtifactSpec(
            "models/expression_neutrality/hsemotion/enet_b2_8_embed_zeroed.onnx",
            "9a62be277daf63e553900ff15fe2124fb4ba9b4d759364e21f07464f351e8b44",
            30_800_917,
        ),
        ArtifactSpec(
            "models/expression_neutrality/grimmer/hse_1_2_C_adaboost.yml.gz",
            "6555645c47dd2d3ff290a6e881890fa8ca8adeaf31d4678a4bd2a3524d25ad87",
            19_368_922,
        ),
        ArtifactSpec(
            "models/no_compression_artifacts/ssim_248_model.onnx",
            "001fe8e73803a3ffb4dbfc7794d9e20232eb925606e8efe68e54696516d04b41",
            23_382_575,
        ),
        ArtifactSpec(
            "models/sharpness/face_sharpness_rtree.xml.gz",
            "0d7decdad2b635e4d811f4de30a4bb451038d0698e72d452ba4b231de7710b02",
            507_331,
        ),
    ),
)
