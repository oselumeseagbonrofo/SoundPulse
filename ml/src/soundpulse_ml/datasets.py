from __future__ import annotations

import json
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf

from .contracts import ContractBundle, load_contract_bundle


ROOT = Path(__file__).resolve().parents[3]
ML_ROOT = ROOT / "ml"
RAW_DATA_DIR = ML_ROOT / "data" / "raw"
CURATED_DATA_DIR = ML_ROOT / "data" / "curated"
DEFAULT_MANIFEST_PATH = CURATED_DATA_DIR / "soundpulse_manifest.v1.json"
DEFAULT_SUMMARY_PATH = CURATED_DATA_DIR / "soundpulse_manifest_summary.csv"
DEFAULT_SOURCE_MAPPING_PATH = CURATED_DATA_DIR / "soundpulse_class_to_source_mapping.json"

VALID_SPLITS = {"train", "validation", "test"}
CURATED_EXCLUDED_CLASSES = {"background"}

ESC50_PHASE1_LABEL_MAP = {
    "fireworks": "fire_alarm",
    "clock_alarm": "clock_alarm",
    "door_wood_knock": "door_knock",
    "crying_baby": "crying_baby",
    "dog": "dog_barking",
    "door_wood_creaks": "door_wood_creaking",
}

FSD50K_PHASE1_MID_TO_CLASS = {
    "/m/07pp_mv": "fire_alarm",
    "/m/07r4wb8": "door_knock",
    "/m/03wwcy": "doorbell",
    "/m/0463cq4": "crying_baby",
    "/m/02jz0l": "water_running",
    "/m/05tny_": "dog_barking",
    "/m/07q6cd_": "door_wood_creaking",
    "/m/09x0r": "speech",
    "/m/04brg2": "background",
    "/m/0dxrf": "background",
    "/m/04rlf": "background",
    "/m/02x984l": "background",
}

FSD50K_PHASE1_LABEL_TO_CLASS = {
    "Alarm": "fire_alarm",
    "Knock": "door_knock",
    "Doorbell": "doorbell",
    "Crying_and_sobbing": "crying_baby",
    "Water_tap_and_faucet": "water_running",
    "Bark": "dog_barking",
    "Squeak": "door_wood_creaking",
    "Speech": "speech",
    "Dishes_and_pots_and_pans": "background",
    "Frying_(food)": "background",
    "Music": "background",
    "Mechanical_fan": "background",
}

FSD50K_SPECIFIC_PHASE1_CLASSES = {
    "door_knock",
    "doorbell",
    "crying_baby",
    "water_running",
    "dog_barking",
    "door_wood_creaking",
    "speech",
    "background",
}

PHASE1_CLASS_TO_SOURCE_MAPPING = {
    "fire_alarm": {
        "esc50_categories": ["fireworks"],
        "fsd50k_mids": ["/m/07pp_mv"],
    },
    "clock_alarm": {
        "esc50_categories": ["clock_alarm"],
        "fsd50k_mids": [],
    },
    "door_knock": {
        "esc50_categories": ["door_wood_knock"],
        "fsd50k_mids": ["/m/07r4wb8"],
    },
    "doorbell": {
        "esc50_categories": [],
        "fsd50k_mids": ["/m/03wwcy"],
    },
    "crying_baby": {
        "esc50_categories": ["crying_baby"],
        "fsd50k_mids": ["/m/0463cq4"],
    },
    "water_running": {
        "esc50_categories": [],
        "fsd50k_mids": ["/m/02jz0l"],
    },
    "dog_barking": {
        "esc50_categories": ["dog"],
        "fsd50k_mids": ["/m/05tny_"],
    },
    "door_wood_creaking": {
        "esc50_categories": ["door_wood_creaks"],
        "fsd50k_mids": ["/m/07q6cd_"],
    },
    "speech": {
        "esc50_categories": [],
        "fsd50k_mids": ["/m/09x0r"],
    },
    "background": {
        "esc50_categories": [],
        "fsd50k_mids": ["/m/04brg2", "/m/0dxrf", "/m/04rlf", "/m/02x984l"],
        "note": "Excluded from the curated phase-1 dataset; retained only as a runtime/background concept.",
    },
}


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


def resolve_audio_path(relative_path: str, project_root: Path | None = None) -> Path:
    base_dir = project_root or ROOT
    return (base_dir / relative_path).resolve()


def manifest_to_dataframe(
    manifest: dict[str, Any],
    project_root: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in manifest.get("datasets", []):
        dataset_name = dataset.get("name", "unknown")
        for clip in dataset.get("clips", []):
            audio_path = resolve_audio_path(clip["relative_path"], project_root=project_root)
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "source_name": dataset.get("source", dataset_name),
                    "audio_path": str(audio_path),
                    "class_slug": clip["class_slug"],
                    "split": clip["split"],
                    "clip_id": clip["clip_id"],
                    "source_group_id": clip["source_group_id"],
                    "duration_ms": int(clip["duration_ms"]),
                    "sample_rate_hz": int(clip["sample_rate_hz"]),
                    "relative_path": clip["relative_path"],
                }
            )
    return pd.DataFrame(rows)


def build_phase1_manifest(
    output_path: Path | None = None,
    contracts: ContractBundle | None = None,
) -> dict[str, Any]:
    contract_bundle = contracts or load_contract_bundle()
    known_classes = set(contract_bundle.class_slugs)

    records = _build_esc50_records() + _build_fsd50k_records()
    class_counts = defaultdict(int)
    dataset_clips: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        if record["class_slug"] not in known_classes:
            raise ValueError(f"Unexpected class_slug generated for phase 1: {record['class_slug']}")
        class_counts[record["class_slug"]] += 1
        dataset_clips[record["dataset_name"]].append(
            {
                "clip_id": record["clip_id"],
                "class_slug": record["class_slug"],
                "relative_path": record["relative_path"],
                "split": record["split"],
                "duration_ms": record["duration_ms"],
                "sample_rate_hz": record["sample_rate_hz"],
                "source_group_id": record["source_group_id"],
            }
        )

    manifest = {
        "version": "1.0.0",
        "description": (
            "Phase-1 curated manifest generated from the local ESC-50 and FSD50K copies. "
            "AudioSet is intentionally excluded until the local download is complete."
        ),
        "classes_version": contract_bundle.class_labels["version"],
        "disabled_datasets": [
            {
                "name": "audioset_subset",
                "reason": "AudioSet has not been downloaded completely and is intentionally disabled for phase 1.",
            }
        ],
        "datasets": [
            {
                "name": "esc50",
                "source": "ESC-50",
                "clips": sorted(dataset_clips.get("esc50", []), key=lambda clip: clip["clip_id"]),
            },
            {
                "name": "fsd50k",
                "source": "FSD50K",
                "clips": sorted(dataset_clips.get("fsd50k", []), key=lambda clip: clip["clip_id"]),
            },
        ],
        "class_counts": dict(sorted(class_counts.items())),
    }

    validate_dataset_manifest(manifest, contracts=contract_bundle)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    return manifest


def write_phase1_dataset_artifacts(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    source_mapping_path: Path = DEFAULT_SOURCE_MAPPING_PATH,
) -> dict[str, Path]:
    manifest = build_phase1_manifest(output_path=manifest_path)
    summary_frame = summarize_dataset_manifest_frame(manifest)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(summary_path, index=False)

    source_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with source_mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(PHASE1_CLASS_TO_SOURCE_MAPPING, handle, indent=2)

    return {
        "manifest": manifest_path,
        "summary": summary_path,
        "source_mapping": source_mapping_path,
    }


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

        if record.sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive for clip {record.clip_id}")

        if not record.source_group_id:
            raise ValueError(f"source_group_id must be populated for clip {record.clip_id}")

        existing_split = group_to_split.get(record.source_group_id)
        if existing_split and existing_split != record.split:
            raise ValueError(
                "Split leakage detected for source_group_id "
                f"{record.source_group_id}: {existing_split} vs {record.split}"
            )
        group_to_split[record.source_group_id] = record.split


def summarize_dataset_manifest_frame(manifest: dict[str, Any]) -> pd.DataFrame:
    frame = manifest_to_dataframe(manifest)
    if frame.empty:
        return pd.DataFrame(columns=["dataset_name", "split", "class_slug", "clip_count"])

    summary = (
        frame.groupby(["dataset_name", "split", "class_slug"], as_index=False)
        .size()
        .rename(columns={"size": "clip_count"})
        .sort_values(["dataset_name", "split", "class_slug"], ignore_index=True)
    )
    return summary


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


def _audio_metadata(audio_path: Path) -> tuple[int, int]:
    if audio_path.suffix.lower() == ".wav":
        with wave.open(str(audio_path), "rb") as wav_file:
            sample_rate_hz = wav_file.getframerate()
            frame_count = wav_file.getnframes()
        duration_ms = int(round((frame_count / sample_rate_hz) * 1000))
        return duration_ms, int(sample_rate_hz)

    info = sf.info(str(audio_path))
    return int(round(info.duration * 1000)), int(info.samplerate)


def _manifest_relative_path(audio_path: Path) -> str:
    return audio_path.resolve().relative_to(ROOT).as_posix()


def _build_esc50_records() -> list[dict[str, Any]]:
    metadata_path = RAW_DATA_DIR / "esc_50_dataset" / "meta" / "esc50.csv"
    audio_root = RAW_DATA_DIR / "esc_50_dataset" / "audio"
    frame = pd.read_csv(metadata_path)
    frame = frame[frame["category"].isin(ESC50_PHASE1_LABEL_MAP)].copy()

    fold_to_split = {1: "validation", 5: "test"}
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        audio_path = audio_root / row["filename"]
        if not audio_path.exists():
            continue

        duration_ms, sample_rate_hz = _audio_metadata(audio_path)
        records.append(
            {
                "dataset_name": "esc50",
                "clip_id": f"esc50-{audio_path.stem}",
                "class_slug": ESC50_PHASE1_LABEL_MAP[row["category"]],
                "relative_path": _manifest_relative_path(audio_path),
                "split": fold_to_split.get(int(row["fold"]), "train"),
                "duration_ms": duration_ms,
                "sample_rate_hz": sample_rate_hz,
                "source_group_id": f"esc50::{row['src_file']}",
            }
        )

    return records


def _build_fsd50k_records() -> list[dict[str, Any]]:
    dev_csv_path = RAW_DATA_DIR / "FSDK50_dataset" / "FSD50K.ground_truth" / "dev.csv"
    eval_csv_path = RAW_DATA_DIR / "FSDK50_dataset" / "FSD50K.ground_truth" / "eval.csv"
    dev_audio_root = RAW_DATA_DIR / "FSDK50_dataset" / "FSD50K.dev_audio"
    eval_audio_root = RAW_DATA_DIR / "FSDK50_dataset" / "FSD50K.eval_audio"

    records = _build_fsd50k_split_records(
        csv_path=dev_csv_path,
        audio_root=dev_audio_root,
        split_mapping={"train": "train", "val": "validation"},
    )
    records.extend(
        _build_fsd50k_split_records(
            csv_path=eval_csv_path,
            audio_root=eval_audio_root,
            split_mapping={None: "test"},
        )
    )
    return records


def _build_fsd50k_split_records(
    csv_path: Path,
    audio_root: Path,
    split_mapping: dict[str | None, str],
) -> list[dict[str, Any]]:
    frame = pd.read_csv(csv_path)
    records: list[dict[str, Any]] = []

    for row in frame.to_dict(orient="records"):
        class_slug = _resolve_fsd50k_class_slug(row["mids"], row["labels"])
        if class_slug is None:
            continue
        if class_slug in CURATED_EXCLUDED_CLASSES:
            continue

        audio_path = audio_root / f"{row['fname']}.wav"
        if not audio_path.exists():
            continue

        split_key = row.get("split")
        split = split_mapping.get(split_key)
        if split is None:
            continue

        duration_ms, sample_rate_hz = _audio_metadata(audio_path)
        records.append(
            {
                "dataset_name": "fsd50k",
                "clip_id": f"fsd50k-{row['fname']}",
                "class_slug": class_slug,
                "relative_path": _manifest_relative_path(audio_path),
                "split": split,
                "duration_ms": duration_ms,
                "sample_rate_hz": sample_rate_hz,
                "source_group_id": f"fsd50k::{row['fname']}",
            }
        )

    return records


def _resolve_fsd50k_class_slug(mids: str, labels: str) -> str | None:
    matched_label_classes = {
        FSD50K_PHASE1_LABEL_TO_CLASS[label]
        for label in labels.split(",")
        if label in FSD50K_PHASE1_LABEL_TO_CLASS
    }
    specific_matches = sorted(
        class_slug for class_slug in matched_label_classes if class_slug in FSD50K_SPECIFIC_PHASE1_CLASSES
    )
    if len(specific_matches) == 1:
        return specific_matches[0]
    if len(specific_matches) > 1:
        return None

    matched_mid_classes = sorted(
        {
            FSD50K_PHASE1_MID_TO_CLASS[mid]
            for mid in mids.split(",")
            if mid in FSD50K_PHASE1_MID_TO_CLASS
        }
    )
    if len(matched_mid_classes) != 1:
        return None
    return matched_mid_classes[0]
