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
)


@dataclass(frozen=True)
class DeterminismJobConfig:
    manifest_path: Path
    preprocessing_config_path: Path
    report_path: Path
    split: str | None
    max_clips: int
    repeats: int
    atol: float
    rtol: float
    include_partial_last_window: bool


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate deterministic preprocessing output across repeated runs.",
    )
    parser.add_argument("--job-config", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("ml/data/curated/soundpulse_manifest.v1.json"),
    )
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=Path("packages/shared_contracts/v1/preprocessing_config.json"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/evaluation/preprocessing_determinism_report.json"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--max-clips", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--drop-partial-last-window", action="store_true")
    return parser.parse_args()


def resolve_job_config(args: argparse.Namespace) -> DeterminismJobConfig:
    payload: dict[str, Any] = {}
    if args.job_config is not None:
        payload = _load_json(args.job_config)

    max_clips = int(payload.get("max_clips", args.max_clips))
    repeats = int(payload.get("repeats", args.repeats))
    if max_clips <= 0:
        raise ValueError("max_clips must be > 0")
    if repeats < 2:
        raise ValueError("repeats must be >= 2")

    return DeterminismJobConfig(
        manifest_path=Path(payload.get("manifest", args.manifest)),
        preprocessing_config_path=Path(
            payload.get("preprocessing_config", args.preprocessing_config)
        ),
        report_path=Path(payload.get("report_path", args.report_path)),
        split=payload.get("split", args.split),
        max_clips=max_clips,
        repeats=repeats,
        atol=float(payload.get("atol", args.atol)),
        rtol=float(payload.get("rtol", args.rtol)),
        include_partial_last_window=bool(
            payload.get("include_partial_last_window", not args.drop_partial_last_window)
        ),
    )


def select_rows(manifest_path: Path, split: str | None, max_clips: int) -> pd.DataFrame:
    manifest = load_dataset_manifest(manifest_path)
    frame = manifest_to_dataframe(manifest)
    if split is not None:
        frame = frame[frame["split"] == split]
    frame = frame.sort_values(["split", "class_slug", "clip_id"], ignore_index=True)
    frame = frame[frame["audio_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    return frame.head(max_clips)


def run_determinism_check(job: DeterminismJobConfig) -> dict[str, Any]:
    config = load_preprocessing_config(job.preprocessing_config_path)
    rows = select_rows(job.manifest_path, split=job.split, max_clips=job.max_clips)
    if rows.empty:
        raise ValueError("No audio rows available for determinism validation")

    comparisons: list[dict[str, Any]] = []
    passed = True

    for row in rows.to_dict(orient="records"):
        audio_path = Path(row["audio_path"])
        runs = [
            preprocess_audio_path(
                audio_path,
                config=config,
                include_partial_last_window=job.include_partial_last_window,
            )
            for _ in range(job.repeats)
        ]
        baseline = runs[0]
        baseline_hash = hash_feature_tensor(baseline)

        run_checks: list[dict[str, Any]] = []
        for idx, candidate in enumerate(runs[1:], start=2):
            close = bool(np.allclose(baseline, candidate, atol=job.atol, rtol=job.rtol))
            max_abs_diff = float(np.max(np.abs(baseline - candidate)))
            same_hash = hash_feature_tensor(candidate) == baseline_hash
            run_checks.append(
                {
                    "run_index": idx,
                    "allclose": close,
                    "max_abs_diff": max_abs_diff,
                    "same_sha256": same_hash,
                }
            )
            passed = passed and close

        comparisons.append(
            {
                "clip_id": row["clip_id"],
                "audio_path": str(audio_path),
                "feature_shape": list(baseline.shape),
                "baseline_sha256": baseline_hash,
                "checks": run_checks,
            }
        )

    report = {
        "status": "pass" if passed else "fail",
        "manifest_path": str(job.manifest_path),
        "preprocessing_config_path": str(job.preprocessing_config_path),
        "split": job.split,
        "max_clips": job.max_clips,
        "repeats": job.repeats,
        "tolerance": {"atol": job.atol, "rtol": job.rtol},
        "include_partial_last_window": job.include_partial_last_window,
        "clip_count": len(comparisons),
        "comparisons": comparisons,
    }
    return report


def main() -> int:
    args = parse_args()
    job = resolve_job_config(args)
    report = run_determinism_check(job)

    job.report_path.parent.mkdir(parents=True, exist_ok=True)
    with job.report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Determinism report written: {job.report_path}")
    print(f"Status: {report['status']}")

    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
