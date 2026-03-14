from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
ML_ROOT = CURRENT_DIR.parent
REPO_ROOT = ML_ROOT.parent
CONTRACTS_DIR = REPO_ROOT / "packages" / "shared_contracts" / "v1"

DEFAULT_RELEASE_DIR = ML_ROOT / "export" / "artifacts" / "generated" / "release_v1.0.0"
DEFAULT_OUTPUT_PATH = ML_ROOT / "evaluation" / "release_artifact_validation_summary.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _new_check(name: str) -> dict[str, Any]:
    return {"name": name, "status": "pass", "details": []}


def _fail(check: dict[str, Any], detail: str) -> None:
    check["status"] = "fail"
    check["details"].append(detail)


def _warn(check: dict[str, Any], detail: str) -> None:
    if check["status"] == "pass":
        check["status"] = "warn"
    check["details"].append(detail)


def _validate_manifest_shape(manifest: dict[str, Any], check: dict[str, Any]) -> None:
    required_top = ["version", "artifact_version", "contracts_version", "created_at_utc", "backbones"]
    for key in required_top:
        if key not in manifest:
            _fail(check, f"Missing required manifest field: {key}")

    backbones = manifest.get("backbones")
    if not isinstance(backbones, list):
        _fail(check, "Manifest backbones must be an array")
        return

    seen_names: set[str] = set()
    for index, backbone in enumerate(backbones):
        if not isinstance(backbone, dict):
            _fail(check, f"backbones[{index}] must be an object")
            continue

        for key in ["name", "model_path", "embedding_dim", "sha256"]:
            if key not in backbone:
                _fail(check, f"backbones[{index}] missing field: {key}")

        name = backbone.get("name")
        if name not in {"tiny_cnn", "yamnet"}:
            _fail(check, f"backbones[{index}].name must be tiny_cnn|yamnet, found: {name}")
        elif name in seen_names:
            _fail(check, f"Duplicate backbone entry: {name}")
        else:
            seen_names.add(name)

        if not isinstance(backbone.get("model_path"), str):
            _fail(check, f"backbones[{index}].model_path must be a string")
        if not isinstance(backbone.get("embedding_dim"), int):
            _fail(check, f"backbones[{index}].embedding_dim must be an integer")

        sha = backbone.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            _fail(check, f"backbones[{index}].sha256 must be a 64-character hex digest")


def _validate_model_files(release_dir: Path, manifest: dict[str, Any], check: dict[str, Any]) -> None:
    backbones = manifest.get("backbones", [])
    for backbone in backbones:
        model_path = backbone.get("model_path")
        expected_sha = backbone.get("sha256")
        if not isinstance(model_path, str):
            continue

        model_file = release_dir / model_path
        if not model_file.exists():
            _fail(check, f"Model file missing: {model_path}")
            continue

        if expected_sha and _sha256(model_file) != expected_sha:
            _fail(check, f"Checksum mismatch for model file: {model_path}")


def _validate_checksums(release_dir: Path, checksums: dict[str, Any], check: dict[str, Any]) -> None:
    files = checksums.get("files")
    if not isinstance(files, list):
        _fail(check, "checksums.sha256.json must contain a files array")
        return

    if not files:
        _warn(check, "checksums.sha256.json files array is empty")

    for entry in files:
        path_text = entry.get("path")
        expected_sha = entry.get("sha256")
        if not isinstance(path_text, str) or not path_text:
            _fail(check, "Checksum entry has invalid path")
            continue
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            _fail(check, f"Invalid checksum format for {path_text}")
            continue

        file_path = release_dir / path_text
        if not file_path.exists():
            _fail(check, f"Checksum target missing: {path_text}")
            continue

        actual_sha = _sha256(file_path)
        if actual_sha != expected_sha:
            _fail(check, f"Checksum mismatch for {path_text}")


def _validate_threshold_consistency(release_dir: Path, manifest: dict[str, Any], check: dict[str, Any]) -> None:
    threshold_path = release_dir / "metadata" / "threshold_defaults.json"
    schema_path = CONTRACTS_DIR / "thresholds.schema.json"

    if not threshold_path.exists():
        _fail(check, f"Missing threshold payload: {threshold_path}")
        return

    threshold_payload = _load_json(threshold_path)
    schema = _load_json(schema_path)

    required_fields = schema.get("required", [])
    for key in required_fields:
        if key not in threshold_payload:
            _fail(check, f"Threshold payload missing required field: {key}")

    contracts_version = manifest.get("contracts_version")
    threshold_version = threshold_payload.get("version")
    if contracts_version != threshold_version:
        _fail(
            check,
            "Threshold version mismatch: "
            f"contracts_version={contracts_version}, threshold_version={threshold_version}",
        )

    confidence = threshold_payload.get("confidence_tiers", {})
    for key in ["high", "medium", "low"]:
        if key not in confidence:
            _fail(check, f"Threshold confidence_tiers missing key: {key}")

    modes = threshold_payload.get("modes", {})
    expected_modes = {"general", "focus", "sleep"}
    if set(modes.keys()) != expected_modes:
        _fail(check, f"Threshold modes mismatch: expected {sorted(expected_modes)}, got {sorted(modes.keys())}")


def _validate_metadata_consistency(release_dir: Path, manifest: dict[str, Any], check: dict[str, Any]) -> None:
    preprocessing_ref_path = release_dir / "metadata" / "preprocessing_spec.reference.json"
    embedding_meta_path = release_dir / "metadata" / "embedding_metadata.json"
    label_map_path = release_dir / "metadata" / "label_map.json"

    for required in [preprocessing_ref_path, embedding_meta_path, label_map_path]:
        if not required.exists():
            _fail(check, f"Missing metadata file: {required.relative_to(release_dir).as_posix()}")

    if preprocessing_ref_path.exists():
        preprocessing_ref = _load_json(preprocessing_ref_path)
        contracts_version = manifest.get("contracts_version")
        ref_version = preprocessing_ref.get("contracts_version")
        if contracts_version != ref_version:
            _fail(
                check,
                "Preprocessing contracts version mismatch: "
                f"manifest={contracts_version}, preprocessing_ref={ref_version}",
            )

    if embedding_meta_path.exists():
        embedding_meta = _load_json(embedding_meta_path)
        dims = {
            backbone.get("name"): backbone.get("embedding_dim") for backbone in manifest.get("backbones", [])
        }
        tiny_dim = embedding_meta.get("backbones", {}).get("tiny_cnn", {}).get("embedding_dim")
        yam_dim = embedding_meta.get("backbones", {}).get("yamnet", {}).get("embedding_dim")
        if dims.get("tiny_cnn") != tiny_dim:
            _fail(check, "TinyCNN embedding_dim mismatch between manifest and embedding metadata")
        if dims.get("yamnet") != yam_dim:
            _fail(check, "YAMNet embedding_dim mismatch between manifest and embedding metadata")

    if label_map_path.exists():
        label_map = _load_json(label_map_path)
        labels = label_map.get("labels", [])
        ids = [item.get("id") for item in labels]
        if ids != list(range(len(ids))):
            _fail(check, "Label map IDs must be contiguous and zero-based")


def validate_export_artifacts(release_dir: Path) -> dict[str, Any]:
    manifest_path = release_dir / "export_manifest.v1.json"
    checksums_path = release_dir / "checksums.sha256.json"

    checks: list[dict[str, Any]] = []

    file_presence = _new_check("file_presence")
    checks.append(file_presence)
    for required in [manifest_path, checksums_path]:
        if not required.exists():
            _fail(file_presence, f"Missing required file: {required.relative_to(release_dir).as_posix()}")

    if file_presence["status"] == "fail":
        result = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "release_dir": str(release_dir),
            "overall_status": "fail",
            "checks": checks,
        }
        return result

    manifest = _load_json(manifest_path)
    checksums = _load_json(checksums_path)

    manifest_shape = _new_check("manifest_schema_conformance")
    _validate_manifest_shape(manifest, manifest_shape)
    checks.append(manifest_shape)

    model_files = _new_check("model_files_and_manifest_checksums")
    _validate_model_files(release_dir, manifest, model_files)
    checks.append(model_files)

    checksum_integrity = _new_check("checksum_manifest_integrity")
    _validate_checksums(release_dir, checksums, checksum_integrity)
    checks.append(checksum_integrity)

    threshold_consistency = _new_check("threshold_payload_version_consistency")
    _validate_threshold_consistency(release_dir, manifest, threshold_consistency)
    checks.append(threshold_consistency)

    metadata_consistency = _new_check("metadata_consistency")
    _validate_metadata_consistency(release_dir, manifest, metadata_consistency)
    checks.append(metadata_consistency)

    statuses = [check["status"] for check in checks]
    if "fail" in statuses:
        overall_status = "fail"
    elif "warn" in statuses:
        overall_status = "warn"
    else:
        overall_status = "pass"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_dir": str(release_dir),
        "manifest_path": str(manifest_path),
        "checksums_path": str(checksums_path),
        "overall_status": overall_status,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SoundPulse export package integrity and consistency.")
    parser.add_argument(
        "--release-dir",
        default=str(DEFAULT_RELEASE_DIR),
        help="Path to the release package directory.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write the machine-readable validation summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    release_dir = Path(args.release_dir).resolve()
    summary = validate_export_artifacts(release_dir)

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(json.dumps(summary, indent=2))
    return 0 if summary["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
