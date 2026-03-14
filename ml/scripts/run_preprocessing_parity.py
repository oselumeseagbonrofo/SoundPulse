from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from soundpulse_ml.preprocessing import hash_feature_tensor, load_preprocessing_config, preprocess_audio_path


@dataclass(frozen=True)
class ParityJobConfig:
    preprocessing_config: Path
    baseline: Path
    report_path: Path
    atol: float
    rtol: float
    write_baseline: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run preprocessing parity checks against committed reference fixtures. "
            "Default tolerance is atol=1e-6, rtol=1e-5."
        ),
    )
    parser.add_argument("--job-config", type=Path, default=None)
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=Path("packages/shared_contracts/v1/preprocessing_config.json"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("ml/evaluation/fixtures/preprocessing_parity_baseline.v1.json"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/evaluation/preprocessing_parity_report.json"),
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Regenerate baseline from the current implementation.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_job_config(args: argparse.Namespace) -> ParityJobConfig:
    payload: dict[str, Any] = {}
    if args.job_config is not None:
        payload = _load_json(args.job_config)

    return ParityJobConfig(
        preprocessing_config=Path(payload.get("preprocessing_config", args.preprocessing_config)),
        baseline=Path(payload.get("baseline", args.baseline)),
        report_path=Path(payload.get("report_path", args.report_path)),
        atol=float(payload.get("atol", args.atol)),
        rtol=float(payload.get("rtol", args.rtol)),
        write_baseline=bool(payload.get("write_baseline", args.write_baseline)),
    )


def fixture_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "sine_440_mono_16k",
            "sample_rate_hz": 16000,
            "channels": 1,
            "duration_seconds": 1.0,
        },
        {
            "name": "dual_tone_stereo_22k05",
            "sample_rate_hz": 22050,
            "channels": 2,
            "duration_seconds": 1.0,
        },
        {
            "name": "chirp_mono_24k",
            "sample_rate_hz": 24000,
            "channels": 1,
            "duration_seconds": 1.35,
        },
    ]


def synthesize_waveform(spec: dict[str, Any]) -> np.ndarray:
    sr = int(spec["sample_rate_hz"])
    duration = float(spec["duration_seconds"])
    samples = int(sr * duration)
    t = np.arange(samples, dtype=np.float32) / sr

    if spec["name"].startswith("sine_440"):
        waveform = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    elif spec["name"].startswith("dual_tone"):
        base = 0.35 * np.sin(2.0 * np.pi * 220.0 * t) + 0.25 * np.sin(2.0 * np.pi * 880.0 * t)
        envelope = np.ones_like(base)
        edge = int(0.1 * sr)
        envelope[:edge] = np.linspace(0.0, 1.0, num=edge, dtype=np.float32)
        envelope[-edge:] = np.linspace(1.0, 0.0, num=edge, dtype=np.float32)
        waveform = base * envelope
    else:
        f0, f1 = 200.0, 3000.0
        k = (f1 - f0) / duration
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
        waveform = 0.4 * np.sin(phase)

    waveform = waveform.astype(np.float32)

    channels = int(spec["channels"])
    if channels == 1:
        return waveform
    if channels == 2:
        right = 0.8 * waveform
        return np.stack([waveform, right], axis=1).astype(np.float32)

    raise ValueError(f"Unsupported channel count in fixture: {channels}")


def collect_fixture_observations(config_path: Path) -> list[dict[str, Any]]:
    config = load_preprocessing_config(config_path)

    observations: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="soundpulse-parity-") as tmpdir:
        tmp_root = Path(tmpdir)

        for spec in fixture_specs():
            waveform = synthesize_waveform(spec)
            wav_path = tmp_root / f"{spec['name']}.wav"
            sf.write(
                wav_path,
                waveform,
                samplerate=int(spec["sample_rate_hz"]),
                subtype="PCM_16",
            )

            feature_tensor = preprocess_audio_path(wav_path, config=config, include_partial_last_window=True)
            flat = feature_tensor.reshape(-1)

            # Use stable anchor positions to catch subtle numerical drift.
            anchor_indices = np.linspace(0, flat.shape[0] - 1, num=16, dtype=np.int64)
            anchors = flat[anchor_indices]

            observations.append(
                {
                    "name": spec["name"],
                    "source_sample_rate_hz": int(spec["sample_rate_hz"]),
                    "source_channels": int(spec["channels"]),
                    "duration_seconds": float(spec["duration_seconds"]),
                    "feature_shape": list(feature_tensor.shape),
                    "feature_sha256": hash_feature_tensor(feature_tensor),
                    "feature_stats": {
                        "mean": float(np.mean(feature_tensor)),
                        "std": float(np.std(feature_tensor)),
                        "min": float(np.min(feature_tensor)),
                        "max": float(np.max(feature_tensor)),
                    },
                    "anchor_indices": anchor_indices.tolist(),
                    "anchor_values": [float(v) for v in anchors.tolist()],
                }
            )

    return observations


def write_baseline(path: Path, config_path: Path) -> None:
    observations = collect_fixture_observations(config_path)
    payload = {
        "baseline_version": "1.0.0",
        "description": (
            "Reference preprocessing parity fixtures for M4. "
            "Fail CI if anchors/stats drift beyond tolerance."
        ),
        "preprocessing_config_path": str(config_path),
        "fixtures": observations,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _allclose_scalar(lhs: float, rhs: float, atol: float, rtol: float) -> bool:
    return bool(np.isclose(np.float64(lhs), np.float64(rhs), atol=atol, rtol=rtol))


def compare_with_baseline(
    baseline_path: Path,
    config_path: Path,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    with baseline_path.open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)

    observed = {item["name"]: item for item in collect_fixture_observations(config_path)}
    comparisons: list[dict[str, Any]] = []
    passed = True

    for expected in baseline.get("fixtures", []):
        fixture_name = expected["name"]
        current = observed.get(fixture_name)
        if current is None:
            comparisons.append({"name": fixture_name, "status": "missing"})
            passed = False
            continue

        expected_anchors = np.array(expected["anchor_values"], dtype=np.float64)
        current_anchors = np.array(current["anchor_values"], dtype=np.float64)
        anchor_ok = bool(np.allclose(expected_anchors, current_anchors, atol=atol, rtol=rtol))
        max_anchor_abs_diff = float(np.max(np.abs(expected_anchors - current_anchors)))

        expected_stats = expected["feature_stats"]
        current_stats = current["feature_stats"]
        stat_checks = {
            key: _allclose_scalar(
                float(expected_stats[key]),
                float(current_stats[key]),
                atol=atol,
                rtol=rtol,
            )
            for key in ["mean", "std", "min", "max"]
        }

        shape_ok = expected["feature_shape"] == current["feature_shape"]
        fixture_pass = shape_ok and anchor_ok and all(stat_checks.values())
        passed = passed and fixture_pass

        comparisons.append(
            {
                "name": fixture_name,
                "status": "pass" if fixture_pass else "fail",
                "shape_ok": shape_ok,
                "anchor_allclose": anchor_ok,
                "max_anchor_abs_diff": max_anchor_abs_diff,
                "stats_ok": stat_checks,
                "expected_sha256": expected["feature_sha256"],
                "observed_sha256": current["feature_sha256"],
            }
        )

    return {
        "status": "pass" if passed else "fail",
        "baseline_path": str(baseline_path),
        "preprocessing_config_path": str(config_path),
        "tolerance": {"atol": atol, "rtol": rtol},
        "comparisons": comparisons,
    }


def main() -> int:
    args = parse_args()
    job = resolve_job_config(args)

    if job.write_baseline:
        write_baseline(job.baseline, job.preprocessing_config)
        print(f"Wrote parity baseline: {job.baseline}")
        return 0

    if not job.baseline.exists():
        raise FileNotFoundError(
            f"Baseline file does not exist: {job.baseline}. "
            "Run with --write-baseline once to generate committed fixtures."
        )

    report = compare_with_baseline(
        baseline_path=job.baseline,
        config_path=job.preprocessing_config,
        atol=job.atol,
        rtol=job.rtol,
    )

    job.report_path.parent.mkdir(parents=True, exist_ok=True)
    with job.report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Parity report written: {job.report_path}")
    print(f"Status: {report['status']}")

    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
