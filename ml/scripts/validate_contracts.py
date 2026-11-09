from __future__ import annotations

from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from soundpulse_ml.contracts import (
    CONTRACTS_DIR,
    load_contract_bundle,
    summarize_contracts,
)


REQUIRED_FILES = [
    "class_labels.json",
    "preprocessing_config.json",
    "threshold_defaults.json",
    "thresholds.schema.json",
    "context_aware_alert_policy.json",
    "inference_result.schema.json",
    "export_manifest.schema.json",
]


def main() -> int:
    missing = [name for name in REQUIRED_FILES if not (CONTRACTS_DIR / name).exists()]
    if missing:
        print("Missing contract files:")
        for name in missing:
            print(f"- {name}")
        return 1

    load_contract_bundle()
    print(summarize_contracts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
