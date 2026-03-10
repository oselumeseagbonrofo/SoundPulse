from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ContractBundle, load_contract_bundle


VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class ClipRecord:
    dataset_name: str
    clip_id: str
    class_slug: str
    relative_path: str
    split: str
    duration_ms: int
    sample_rate_hz: int
    source_group_id: str


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset_manifest(path: Path) -> dict[str, Any]:
    return _load_json(path)


def iter_clips(manifest: dict[str, Any]) -> list[ClipRecord]:
    clips: list[ClipRecord] = []
    for dataset in manifest.get("datasets", []):
        dataset_name = dataset.get("name", "unknown")
        for clip in dataset.get("clips", []):
            clips.append(
                ClipRecord(
                    dataset_name=dataset_name,
                    clip_id=clip["clip_id"],
                    class_slug=clip["class_slug"],
                    relative_path=clip["relative_path"],
                    split=clip["split"],
                    duration_ms=int(clip["duration_ms"]),
                    sample_rate_hz=int(clip["sample_rate_hz"]),
                    source_group_id=clip["source_group_id"],
                )
            )
    return clips


def validate_dataset_manifest(
    manifest: dict[str, Any],
    contracts: ContractBundle | None = None,
) -> None:
    contract_bundle = contracts or load_contract_bundle()
    known_classes = set(contract_bundle.class_slugs)
    expected_sample_rate = contract_bundle.preprocessing["audio"]["sample_rate_hz"]

    if not manifest.get("datasets"):
        raise ValueError("Dataset manifest must define at least one dataset")

    clip_records = iter_clips(manifest)
    seen_clip_ids: set[str] = set()
    group_to_split: dict[str, str] = {}

    for record in clip_records:
        if record.clip_id in seen_clip_ids:
            raise ValueError(f"Duplicate clip_id detected: {record.clip_id}")
        seen_clip_ids.add(record.clip_id)

        if record.class_slug not in known_classes:
            raise ValueError(f"Unknown class_slug in manifest: {record.class_slug}")

        if record.split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{record.split}' for clip {record.clip_id}; expected one of {sorted(VALID_SPLITS)}"
            )

        if record.duration_ms <= 0:
            raise ValueError(f"duration_ms must be positive for clip {record.clip_id}")

        if record.sample_rate_hz != expected_sample_rate:
            raise ValueError(
                f"sample_rate_hz mismatch for clip {record.clip_id}: "
                f"expected {expected_sample_rate}, found {record.sample_rate_hz}"
            )

        existing_split = group_to_split.get(record.source_group_id)
        if existing_split and existing_split != record.split:
            raise ValueError(
                "Split leakage detected for source_group_id "
                f"{record.source_group_id}: {existing_split} vs {record.split}"
            )
        group_to_split[record.source_group_id] = record.split


def summarize_dataset_manifest(path: Path) -> str:
    manifest = load_dataset_manifest(path)
    validate_dataset_manifest(manifest)
    clip_records = iter_clips(manifest)

    split_counts = {split: 0 for split in sorted(VALID_SPLITS)}
    class_counts: dict[str, int] = {}
    for record in clip_records:
        split_counts[record.split] += 1
        class_counts[record.class_slug] = class_counts.get(record.class_slug, 0) + 1

    class_summary = ",".join(
        f"{slug}:{count}" for slug, count in sorted(class_counts.items())
    )
    return (
        "SoundPulse dataset manifest loaded successfully\n"
        f"datasets={len(manifest.get('datasets', []))}\n"
        f"clips={len(clip_records)}\n"
        f"split_counts={split_counts}\n"
        f"class_counts={class_summary}"
    )
