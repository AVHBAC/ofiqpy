"""Privacy-conscious aggregate identities for real input files."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class InputRecord:
    """Per-file provenance emitted only after explicit caller opt-in."""

    identity: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class InputProvenance:
    """Aggregate input binding with optional per-file disclosure."""

    schema: str
    input_count: int
    total_bytes: int
    aggregate_sha256: str
    records: tuple[InputRecord, ...] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_input_provenance(
    paths: list[Path],
    base: Path,
    *,
    include_records: bool = False,
) -> InputProvenance:
    """Hash file contents and stable identities without disclosing records by default."""
    entries: list[InputRecord] = []
    for path in paths:
        try:
            identity = path.relative_to(base).as_posix()
        except ValueError as exc:
            raise ValueError(f"input path is outside the provenance base: {path}") from exc
        entries.append(InputRecord(identity=identity, sha256=sha256_file(path), size_bytes=path.stat().st_size))
    entries.sort(key=lambda item: item.identity)
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry.sha256}  {entry.identity}\n".encode("utf-8"))
    return InputProvenance(
        schema="ofiqpy.input-provenance.v1",
        input_count=len(entries),
        total_bytes=sum(entry.size_bytes for entry in entries),
        aggregate_sha256=digest.hexdigest(),
        records=tuple(entries) if include_records else None,
    )
