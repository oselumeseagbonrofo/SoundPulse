from __future__ import annotations

import json
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "ml"
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import soundpulse_ml.datasets
# Temporarily allow "background" class to be processed
soundpulse_ml.datasets.CURATED_EXCLUDED_CLASSES = set()

from soundpulse_ml.datasets import (
    load_dataset_manifest,
    _build_fsd50k_records,
    validate_dataset_manifest,
    CURATED_DATA_DIR
)

def main() -> int:
    curated_path = CURATED_DATA_DIR / "soundpulse_manifest.v1.json"
    if not curated_path.exists():
        print(f"Error: Curated manifest not found at {curated_path}")
        return 1
        
    manifest = load_dataset_manifest(curated_path)
    
    # Generate fsd50k background clips for all splits (train, validation, test)
    fsd50k_records = _build_fsd50k_records()
    background_test_records = [
        r for r in fsd50k_records
        if r["class_slug"] == "background"
    ]
    
    print(f"Found {len(background_test_records)} total background records in FSD50K.")
    
    # Format them into clips structure
    bg_clips = []
    for r in background_test_records:
        bg_clips.append({
            "clip_id": r["clip_id"],
            "class_slug": r["class_slug"],
            "relative_path": r["relative_path"],
            "split": r["split"],
            "duration_ms": r["duration_ms"],
            "sample_rate_hz": r["sample_rate_hz"],
            "source_group_id": r["source_group_id"],
        })
        
    # Append to FSD50K dataset in manifest
    fsd50k_dataset = next(d for d in manifest["datasets"] if d["name"] == "fsd50k")
    fsd50k_dataset["clips"].extend(bg_clips)
    fsd50k_dataset["clips"] = sorted(fsd50k_dataset["clips"], key=lambda c: c["clip_id"])
    
    # Update manifest's class_counts
    manifest["class_counts"]["background"] = len(background_test_records)
    manifest["class_counts"] = dict(sorted(manifest["class_counts"].items()))
    
    # Validate the manifest
    validate_dataset_manifest(manifest)
    
    # Write new evaluation manifest to disk
    eval_path = CURATED_DATA_DIR / "soundpulse_evaluation_manifest.v1.json"
    with eval_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        
    print(f"Successfully generated evaluation manifest at {eval_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
