from __future__ import annotations

from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from soundpulse_ml.datasets import summarize_dataset_manifest


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_dataset_manifest.py <manifest-path>")
        return 1

    manifest_path = Path(sys.argv[1]).resolve()
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    print(summarize_dataset_manifest(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
