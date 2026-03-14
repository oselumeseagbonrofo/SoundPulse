from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from soundpulse_ml.datasets import load_dataset_manifest, manifest_to_dataframe
from soundpulse_ml.preprocessing import (
    hash_feature_tensor,
    load_preprocessing_config,
    preprocess_audio_path,
    summarize_preprocessing_config,
)


@dataclass(frozen=True)
class GenerationJobConfig:
    manifest_path: Path
    output_dir: Path
    preprocessing_config_path: Path
    split: str | None
    max_clips: int | None
    include_partial_last_window: bool


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_job_config(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic log-mel preprocessing features from manifest rows.",
    )
    parser.add_argument(
        "--job-config",
        type=Path,
        default=None,
        help="Optional JSON file with job settings.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("ml/data/curated/soundpulse_manifest.v1.json"),
        help="Path to dataset manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/export/artifacts/generated/preprocessing_features"),
        help="Directory where generated features and metadata are written.",
    )
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=Path("packages/shared_contracts/v1/preprocessing_config.json"),
        help="Path to shared preprocessing configuration JSON.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "validation", "test"],
        help="Optional split filter.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Optional cap on number of clips to process.",
    )
    parser.add_argument(
        "--drop-partial-last-window",
        action="store_true",
        help="If set, does not include a padded tail window when audio length is not hop-aligned.",
    )
    return parser.parse_args()


def resolve_job_config(args: argparse.Namespace) -> GenerationJobConfig:
    config_payload: dict[str, Any] = {}
    if args.job_config is not None:
        config_payload = _load_job_config(args.job_config)

    manifest_path = Path(config_payload.get("manifest", args.manifest))
    output_dir = Path(config_payload.get("output_dir", args.output_dir))
    preprocessing_config_path = Path(
        config_payload.get("preprocessing_config", args.preprocessing_config)
    )
    split = config_payload.get("split", args.split)
    max_clips = config_payload.get("max_clips", args.max_clips)
    include_partial_last_window = bool(
        config_payload.get("include_partial_last_window", not args.drop_partial_last_window)
    )

    if max_clips is not None and int(max_clips) <= 0:
        raise ValueError("max_clips must be > 0 when provided")

    return GenerationJobConfig(
        manifest_path=manifest_path,
        output_dir=output_dir,
        preprocessing_config_path=preprocessing_config_path,
        split=split,
        max_clips=int(max_clips) if max_clips is not None else None,
        include_partial_last_window=include_partial_last_window,
    )


def select_manifest_rows(manifest_path: Path, split: str | None, max_clips: int | None) -> pd.DataFrame:
    manifest = load_dataset_manifest(manifest_path)
    frame = manifest_to_dataframe(manifest)

    if split is not None:
        frame = frame[frame["split"] == split]

    frame = frame.sort_values(["split", "class_slug", "clip_id"], ignore_index=True)

    if max_clips is not None:
        frame = frame.head(max_clips)

    return frame


def main() -> int:
    args = parse_args()
    job = resolve_job_config(args)

    config = load_preprocessing_config(job.preprocessing_config_path)
    rows = select_manifest_rows(job.manifest_path, split=job.split, max_clips=job.max_clips)

    if rows.empty:
        raise ValueError("No manifest rows matched the requested filters")

    output_dir = job.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    feature_blocks: list[np.ndarray] = []
    missing_files = 0

    for row in rows.to_dict(orient="records"):
        audio_path = Path(row["audio_path"])
        if not audio_path.exists():
            missing_files += 1
            continue

        clip_features = preprocess_audio_path(
            audio_path,
            config=config,
            include_partial_last_window=job.include_partial_last_window,
        )

        for window_idx, feature in enumerate(clip_features):
            feature_index = len(feature_blocks)
            feature_blocks.append(feature)
            index_rows.append(
                {
                    "feature_index": feature_index,
                    "clip_id": row["clip_id"],
                    "class_slug": row["class_slug"],
                    "split": row["split"],
                    "dataset_name": row["dataset_name"],
                    "audio_path": str(audio_path),
                    "window_index": window_idx,
                    "window_count": int(clip_features.shape[0]),
                    "feature_sha256": hash_feature_tensor(feature),
                }
            )

    if not feature_blocks:
        raise ValueError(
            "No features were generated. Ensure manifest rows point to existing audio files."
        )

    feature_tensor = np.stack(feature_blocks, axis=0).astype(np.float32)
    features_path = output_dir / "log_mel_features.npy"
    np.save(features_path, feature_tensor)

    index_frame = pd.DataFrame(index_rows)
    index_path = output_dir / "feature_index.csv"
    index_frame.to_csv(index_path, index=False)

    summary = {
        "manifest_path": str(job.manifest_path),
        "preprocessing_config_path": str(job.preprocessing_config_path),
        "output_dir": str(output_dir),
        "clip_rows_selected": int(len(rows)),
        "clip_rows_with_audio": int(rows.shape[0] - missing_files),
        "missing_audio_rows": int(missing_files),
        "feature_windows_generated": int(feature_tensor.shape[0]),
        "feature_shape": list(feature_tensor.shape[1:]),
        "include_partial_last_window": job.include_partial_last_window,
    }
    summary_path = output_dir / "generation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Preprocessing feature generation complete")
    print(f"- features: {features_path}")
    print(f"- index: {index_path}")
    print(f"- summary: {summary_path}")
    print()
    print(summarize_preprocessing_config(config))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
