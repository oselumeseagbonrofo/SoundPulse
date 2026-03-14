from __future__ import annotations

from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from soundpulse_ml.datasets import (
    DEFAULT_MANIFEST_PATH,
    summarize_dataset_manifest,
    write_phase1_dataset_artifacts,
)


def main() -> int:
    outputs = write_phase1_dataset_artifacts()
    print("Generated phase-1 dataset artifacts:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print()
    print(summarize_dataset_manifest(DEFAULT_MANIFEST_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())