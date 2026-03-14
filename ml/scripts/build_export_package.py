from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
ML_ROOT = CURRENT_DIR.parent
REPO_ROOT = ML_ROOT.parent
CONTRACTS_DIR = REPO_ROOT / "packages" / "shared_contracts" / "v1"

TINYCNN_SOURCE_MODEL = ML_ROOT / "export" / "artifacts" / "generated" / "tinycnn_baseline" / "tinycnn_embedding.tflite"
YAMNET_SUMMARY_PATH = ML_ROOT / "export" / "artifacts" / "generated" / "yamnet_kaggle" / "artifact_summary.json"

TINYCNN_METADATA_PATH = ML_ROOT / "export" / "artifacts" / "generated" / "tinycnn_baseline" / "export_metadata.json"
TINYCNN_SUMMARY_PATH = ML_ROOT / "evaluation" / "tinycnn_baseline_summary.json"
YAMNET_EVAL_SUMMARY_PATH = ML_ROOT / "evaluation" / "yamnet_kaggle_summary.json"
SAME_CAP_SUMMARY_PATH = ML_ROOT / "evaluation" / "same_cap_model_comparison_summary.json"

DEFAULT_ARTIFACT_VERSION = "m8_v1.0.0"
DEFAULT_RELEASE_DIR = ML_ROOT / "export" / "artifacts" / "generated" / "release_v1.0.0"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_yamnet_artifact() -> Path:
    summary = _load_json(YAMNET_SUMMARY_PATH)
    artifact = summary.get("yamnet_artifact")
    if not isinstance(artifact, str) or not artifact:
        raise ValueError(f"Invalid yamnet_artifact in {YAMNET_SUMMARY_PATH}")

    yamnet_pb = Path(artifact)
    if not yamnet_pb.exists():
        raise FileNotFoundError(f"YAMNet artifact not found: {yamnet_pb}")

    yamnet_dir = yamnet_pb.parent
    required = [yamnet_dir / "saved_model.pb", yamnet_dir / "variables"]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Incomplete YAMNet SavedModel directory: {missing_text}")
    return yamnet_dir


def _build_label_map(class_labels: dict[str, Any]) -> dict[str, Any]:
    classes = class_labels.get("classes", [])
    label_entries = []
    for item in classes:
        label_entries.append(
            {
                "id": item["id"],
                "slug": item["slug"],
                "display_name": item["display_name"],
                "urgent": bool(item.get("urgent", False)),
                "suppressed": bool(item.get("suppressed", False)),
            }
        )
    return {
        "version": class_labels.get("version", "unknown"),
        "labels": label_entries,
    }


def _build_embedding_metadata(preprocessing: dict[str, Any]) -> dict[str, Any]:
    audio = preprocessing.get("audio", {})
    embedding_contract = preprocessing.get("embedding_contract", {})
    return {
        "version": preprocessing.get("version", "unknown"),
        "window_contract": {
            "sample_rate_hz": audio.get("sample_rate_hz"),
            "window_duration_ms": audio.get("window_duration_ms"),
            "hop_duration_ms": audio.get("hop_duration_ms"),
            "channels": audio.get("channels"),
        },
        "backbones": {
            "tiny_cnn": {
                "embedding_dim": embedding_contract.get("tiny_cnn_dim"),
                "output_tensor": "embedding",
            },
            "yamnet": {
                "embedding_dim": embedding_contract.get("yamnet_dim"),
                "output_tensor": "embedding",
            },
        },
    }


def _build_model_cards(
    tinycnn_eval: dict[str, Any],
    yamnet_eval: dict[str, Any],
    same_cap_summary: dict[str, Any],
) -> dict[str, Any]:
    winner = same_cap_summary.get("winner_by_macro_f1", "unknown")
    return {
        "version": "1.0.0",
        "recommended_backbone": winner,
        "models": [
            {
                "name": "tiny_cnn",
                "source": tinycnn_eval.get("source_notebook"),
                "embedding_dim": tinycnn_eval.get("embedding_dim"),
                "aggregate_metrics": tinycnn_eval.get("aggregate_metrics", {}),
                "notes": "TinyCNN baseline export from notebook training run.",
            },
            {
                "name": "yamnet",
                "source": yamnet_eval.get("source_notebook"),
                "embedding_dim": yamnet_eval.get("embedding_dim"),
                "aggregate_metrics": yamnet_eval.get("aggregate_metrics", {}),
                "notes": "YAMNet prototype path using Kaggle-hosted SavedModel weights.",
            },
        ],
    }


def _collect_checksums(release_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for file_path in sorted(release_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative_path = file_path.relative_to(release_dir).as_posix()
        if relative_path == "checksums.sha256.json":
            continue
        entries.append(
            {
                "path": relative_path,
                "sha256": _sha256(file_path),
                "size_bytes": file_path.stat().st_size,
            }
        )
    return entries


def build_export_package(release_dir: Path, artifact_version: str) -> dict[str, Any]:
    class_labels = _load_json(CONTRACTS_DIR / "class_labels.json")
    preprocessing = _load_json(CONTRACTS_DIR / "preprocessing_config.json")
    threshold_defaults = _load_json(CONTRACTS_DIR / "threshold_defaults.json")

    tinycnn_metadata = _load_json(TINYCNN_METADATA_PATH)
    tinycnn_eval = _load_json(TINYCNN_SUMMARY_PATH)
    yamnet_eval = _load_json(YAMNET_EVAL_SUMMARY_PATH)
    same_cap_summary = _load_json(SAME_CAP_SUMMARY_PATH)

    yamnet_saved_model_dir = _resolve_yamnet_artifact()

    if release_dir.exists():
        shutil.rmtree(release_dir)

    models_dir = release_dir / "models"
    metadata_dir = release_dir / "metadata"
    models_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    tinycnn_target = models_dir / "tinycnn_embedding.tflite"
    yamnet_target_dir = models_dir / "yamnet_saved_model"

    shutil.copy2(TINYCNN_SOURCE_MODEL, tinycnn_target)
    shutil.copytree(yamnet_saved_model_dir, yamnet_target_dir)

    _write_json(metadata_dir / "label_map.json", _build_label_map(class_labels))
    _write_json(metadata_dir / "threshold_defaults.json", threshold_defaults)
    _write_json(metadata_dir / "embedding_metadata.json", _build_embedding_metadata(preprocessing))
    _write_json(
        metadata_dir / "preprocessing_spec.reference.json",
        {
            "contracts_version": preprocessing.get("version", "unknown"),
            "contract_path": "packages/shared_contracts/v1/preprocessing_config.json",
            "audio": preprocessing.get("audio", {}),
            "stft": preprocessing.get("stft", {}),
            "mel": preprocessing.get("mel", {}),
            "normalization": preprocessing.get("normalization", {}),
        },
    )
    _write_json(
        metadata_dir / "model_cards.json",
        _build_model_cards(tinycnn_eval=tinycnn_eval, yamnet_eval=yamnet_eval, same_cap_summary=same_cap_summary),
    )

    manifest = {
        "version": "1.0.0",
        "artifact_version": artifact_version,
        "contracts_version": class_labels.get("version", "unknown"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backbones": [
            {
                "name": "tiny_cnn",
                "model_path": "models/tinycnn_embedding.tflite",
                "embedding_dim": tinycnn_metadata.get("embedding_dim", 128),
                "sha256": _sha256(tinycnn_target),
            },
            {
                "name": "yamnet",
                "model_path": "models/yamnet_saved_model/saved_model.pb",
                "embedding_dim": int(preprocessing.get("embedding_contract", {}).get("yamnet_dim", 1024)),
                "sha256": _sha256(yamnet_target_dir / "saved_model.pb"),
            },
        ],
        "benchmarks": {
            "tinycnn_summary": "ml/evaluation/tinycnn_baseline_summary.json",
            "yamnet_summary": "ml/evaluation/yamnet_kaggle_summary.json",
            "same_cap_summary": "ml/evaluation/same_cap_model_comparison_summary.json",
            "winner_by_macro_f1": same_cap_summary.get("winner_by_macro_f1"),
        },
    }
    _write_json(release_dir / "export_manifest.v1.json", manifest)

    checksums = {
        "version": "1.0.0",
        "artifact_version": artifact_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": _collect_checksums(release_dir),
    }
    _write_json(release_dir / "checksums.sha256.json", checksums)

    return {
        "release_dir": str(release_dir),
        "manifest_path": str(release_dir / "export_manifest.v1.json"),
        "checksums_path": str(release_dir / "checksums.sha256.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the M8 export package for SoundPulse.")
    parser.add_argument(
        "--artifact-version",
        default=DEFAULT_ARTIFACT_VERSION,
        help="Artifact version string to embed in export metadata.",
    )
    parser.add_argument(
        "--release-dir",
        default=str(DEFAULT_RELEASE_DIR),
        help="Output directory for the generated runtime-ready release package.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_export_package(
        release_dir=Path(args.release_dir).resolve(),
        artifact_version=str(args.artifact_version),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
