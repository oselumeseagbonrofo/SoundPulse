from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
ML_ROOT = CURRENT_DIR.parent
REPO_ROOT = ML_ROOT.parent

import validate_export_artifacts as export_validator


DEFAULT_RELEASE_DIR = ML_ROOT / "export" / "artifacts" / "generated" / "release_v1.0.0"
DEFAULT_JSON_REPORT = ML_ROOT / "evaluation" / "release_gate_summary.json"
DEFAULT_MARKDOWN_REPORT = ML_ROOT / "evaluation" / "release_gate_report.md"
DEFAULT_DATASET_MANIFEST = ML_ROOT / "data" / "curated" / "soundpulse_manifest.v1.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _check_leakage(manifest_path: Path) -> dict[str, Any]:
    result = {"name": "leakage_check_state", "status": "pass", "details": []}
    if not manifest_path.exists():
        result["status"] = "fail"
        result["details"].append(f"Missing dataset manifest: {manifest_path}")
        return result

    try:
        manifest = _load_json(manifest_path)
    except Exception as exc:  # noqa: BLE001
        result["status"] = "fail"
        result["details"].append(f"Failed to parse dataset manifest JSON: {exc}")
        return result

    datasets = manifest.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        result["status"] = "fail"
        result["details"].append("Dataset manifest has no datasets array.")
        return result

    seen_clip_ids: set[str] = set()
    source_group_split: dict[str, str] = {}
    clip_count = 0

    for dataset in datasets:
        clips = dataset.get("clips", []) if isinstance(dataset, dict) else []
        for clip in clips:
            clip_count += 1
            clip_id = clip.get("clip_id")
            split = clip.get("split")
            source_group_id = clip.get("source_group_id")

            if not isinstance(clip_id, str) or not clip_id:
                result["status"] = "fail"
                result["details"].append("Found clip without valid clip_id.")
                continue

            if clip_id in seen_clip_ids:
                result["status"] = "fail"
                result["details"].append(f"Duplicate clip_id detected: {clip_id}")
            seen_clip_ids.add(clip_id)

            if not isinstance(split, str) or split not in {"train", "validation", "test"}:
                result["status"] = "fail"
                result["details"].append(f"Invalid split '{split}' for clip_id={clip_id}")

            if not isinstance(source_group_id, str) or not source_group_id:
                result["status"] = "fail"
                result["details"].append(f"Missing source_group_id for clip_id={clip_id}")
                continue

            existing_split = source_group_split.get(source_group_id)
            if existing_split and split and existing_split != split:
                result["status"] = "fail"
                result["details"].append(
                    f"Leakage detected for source_group_id={source_group_id}: {existing_split} vs {split}"
                )
            elif split:
                source_group_split[source_group_id] = split

    if result["status"] == "pass":
        result["details"].append(
            f"Dataset split-integrity validation passed for {clip_count} clips and {len(source_group_split)} source groups."
        )

    return result


def _check_model_loadability(release_dir: Path) -> dict[str, Any]:
    result = {"name": "artifact_loadability", "status": "pass", "details": []}
    tiny_model = release_dir / "models" / "tinycnn_embedding.tflite"
    yamnet_model_dir = release_dir / "models" / "yamnet_saved_model"

    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # noqa: BLE001
        result["status"] = "fail"
        result["details"].append(f"TensorFlow import failed: {exc}")
        return result

    if not tiny_model.exists():
        result["status"] = "fail"
        result["details"].append(f"Missing TinyCNN TFLite model: {tiny_model}")
    else:
        try:
            interpreter = tf.lite.Interpreter(model_path=str(tiny_model))
            interpreter.allocate_tensors()
            result["details"].append("TinyCNN TFLite model loaded successfully.")
        except Exception as exc:  # noqa: BLE001
            result["status"] = "fail"
            result["details"].append(f"TinyCNN TFLite load failed: {exc}")

    if not yamnet_model_dir.exists():
        result["status"] = "fail"
        result["details"].append(f"Missing YAMNet SavedModel directory: {yamnet_model_dir}")
    else:
        try:
            tf.saved_model.load(str(yamnet_model_dir))
            result["details"].append("YAMNet SavedModel loaded successfully.")
        except Exception as exc:  # noqa: BLE001
            result["status"] = "fail"
            result["details"].append(f"YAMNet SavedModel load failed: {exc}")

    return result


def _check_preprocessing_parity(status_input: str) -> dict[str, Any]:
    status_map = {
        "pass": "pass",
        "fail": "fail",
        "not_run": "warn",
        "unknown": "warn",
    }
    mapped_status = status_map[status_input]

    details_map = {
        "pass": "Preprocessing parity tests reported as passing by operator input.",
        "fail": "Preprocessing parity tests reported as failing by operator input.",
        "not_run": "Preprocessing parity tests have not been run yet.",
        "unknown": "Preprocessing parity status is unknown.",
    }
    return {
        "name": "preprocessing_parity_status_input",
        "status": mapped_status,
        "details": [details_map[status_input]],
    }


def _check_benchmark_baselines() -> dict[str, Any]:
    result = {"name": "benchmark_baseline_presence", "status": "pass", "details": []}

    required_files = {
        "tinycnn": ML_ROOT / "evaluation" / "tinycnn_baseline_summary.json",
        "yamnet": ML_ROOT / "evaluation" / "yamnet_kaggle_summary.json",
        "same_cap": ML_ROOT / "evaluation" / "same_cap_model_comparison_summary.json",
    }

    payloads: dict[str, dict[str, Any]] = {}
    for key, path in required_files.items():
        if not path.exists():
            result["status"] = "fail"
            result["details"].append(f"Missing benchmark summary: {path}")
            continue
        payloads[key] = _load_json(path)

    if result["status"] == "fail":
        return result

    tiny_f1 = payloads["tinycnn"].get("aggregate_metrics", {}).get("macro_f1")
    yamnet_f1 = payloads["yamnet"].get("aggregate_metrics", {}).get("macro_f1")
    winner = payloads["same_cap"].get("winner_by_macro_f1")

    if tiny_f1 is None:
        result["status"] = "fail"
        result["details"].append("tinycnn_baseline_summary.json missing aggregate_metrics.macro_f1")
    if yamnet_f1 is None:
        result["status"] = "fail"
        result["details"].append("yamnet_kaggle_summary.json missing aggregate_metrics.macro_f1")

    if winner is None:
        result["status"] = "fail"
        result["details"].append("same_cap_model_comparison_summary.json missing winner_by_macro_f1")

    if result["status"] != "fail":
        result["details"].append(
            f"Benchmark summaries present. tinycnn_macro_f1={tiny_f1}, yamnet_macro_f1={yamnet_f1}, winner={winner}."
        )

    return result


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# SoundPulse M9 Release Gate Report",
        "",
        f"- generated_at_utc: {summary['generated_at_utc']}",
        f"- release_dir: {summary['release_dir']}",
        f"- overall_status: {summary['overall_status']}",
        "",
        "## Gate Checks",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]

    for check in summary["checks"]:
        details = " ".join(check.get("details", [])) if check.get("details") else "-"
        lines.append(f"| {check['name']} | {check['status']} | {details} |")

    lines.append("")
    lines.append("## Export Validation")
    lines.append("")
    lines.append(f"- overall_status: {summary['export_validation']['overall_status']}")

    for check in summary["export_validation"]["checks"]:
        details = "; ".join(check.get("details", [])) if check.get("details") else "no issues"
        lines.append(f"- {check['name']}: {check['status']} ({details})")

    return "\n".join(lines) + "\n"


def run_release_gate(release_dir: Path, manifest_path: Path, preprocessing_parity_status: str) -> dict[str, Any]:
    export_validation = export_validator.validate_export_artifacts(release_dir)

    checks = [
        _check_leakage(manifest_path),
        {
            "name": "artifact_integrity",
            "status": export_validation["overall_status"],
            "details": [
                f"Export artifact validation overall_status={export_validation['overall_status']}.",
            ],
        },
        _check_model_loadability(release_dir),
        _check_preprocessing_parity(preprocessing_parity_status),
        _check_benchmark_baselines(),
    ]

    statuses = [item["status"] for item in checks]
    if "fail" in statuses:
        overall_status = "fail"
    elif "warn" in statuses:
        overall_status = "warn"
    else:
        overall_status = "pass"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_dir": str(release_dir),
        "dataset_manifest": str(manifest_path),
        "preprocessing_parity_status_input": preprocessing_parity_status,
        "overall_status": overall_status,
        "checks": checks,
        "export_validation": export_validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SoundPulse M9 release-gate checks.")
    parser.add_argument(
        "--release-dir",
        default=str(DEFAULT_RELEASE_DIR),
        help="Path to the generated export release package.",
    )
    parser.add_argument(
        "--dataset-manifest",
        default=str(DEFAULT_DATASET_MANIFEST),
        help="Path to the curated dataset manifest used for leakage integrity checks.",
    )
    parser.add_argument(
        "--preprocessing-parity-status",
        default="unknown",
        choices=["pass", "fail", "not_run", "unknown"],
        help="External input status for preprocessing parity checks.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_JSON_REPORT),
        help="Output path for machine-readable release-gate summary.",
    )
    parser.add_argument(
        "--output-markdown",
        default=str(DEFAULT_MARKDOWN_REPORT),
        help="Output path for markdown release-gate report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    release_dir = Path(args.release_dir).resolve()
    manifest_path = Path(args.dataset_manifest).resolve()

    summary = run_release_gate(
        release_dir=release_dir,
        manifest_path=manifest_path,
        preprocessing_parity_status=args.preprocessing_parity_status,
    )

    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_markdown).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    with output_md.open("w", encoding="utf-8") as handle:
        handle.write(_markdown_report(summary))

    print(json.dumps(summary, indent=2))
    return 0 if summary["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
