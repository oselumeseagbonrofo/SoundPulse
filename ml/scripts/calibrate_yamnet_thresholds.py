from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "ml"
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soundpulse_ml.datasets import load_dataset_manifest, manifest_to_dataframe


CONTRACTS_DIR = PROJECT_ROOT / "packages" / "shared_contracts" / "v1"
CURATED_MANIFEST_PATH = ML_ROOT / "data" / "curated" / "soundpulse_manifest.v1.json"
FSD50K_DEV_METADATA_PATH = ML_ROOT / "data" / "raw" / "FSDK50_dataset" / "FSD50K.ground_truth" / "dev.csv"
FSD50K_EVAL_METADATA_PATH = ML_ROOT / "data" / "raw" / "FSDK50_dataset" / "FSD50K.ground_truth" / "eval.csv"
FSD50K_DEV_AUDIO_DIR = ML_ROOT / "data" / "raw" / "FSDK50_dataset" / "FSD50K.dev_audio"
FSD50K_EVAL_AUDIO_DIR = ML_ROOT / "data" / "raw" / "FSDK50_dataset" / "FSD50K.eval_audio"
YAMNET_ARTIFACT_SUMMARY_PATH = ML_ROOT / "export" / "artifacts" / "generated" / "yamnet_kaggle" / "artifact_summary.json"
THRESHOLD_DEFAULTS_PATH = CONTRACTS_DIR / "threshold_defaults.json"
EVALUATION_DIR = ML_ROOT / "evaluation"

KNOWN_CAPS = {"train": 128, "validation": 64, "test": 64}
UNKNOWN_CALIBRATION_CAP = 1024
UNKNOWN_HOLDOUT_CAP = 1024
UNKNOWN_MAX_PER_CATEGORY = 12
URGENT_CLASS_SLUGS = ["fire_alarm", "clock_alarm", "door_knock", "doorbell"]
# Hard absolute floors: the selector never picks a candidate that falls below these
# values regardless of how many tolerance brackets are relaxed. This prevents the
# unconstrained (1.0, 1.0) fallback from choosing a pure-rejection degenerate point.
SELECTOR_URGENT_RECALL_FLOOR = 0.10   # never drop urgent recall below this
SELECTOR_MACRO_F1_FLOOR_FRAC = 0.40   # never drop macro F1 below this fraction of baseline
FSD50K_IN_SCOPE_LABELS = {
    "Alarm",
    "Bark",
    "Child_speech_and_kid_speaking",
    "Crying_and_sobbing",
    "Dog",
    "Door",
    "Doorbell",
    "Female_speech_and_woman_speaking",
    "Knock",
    "Male_speech_and_man_speaking",
    "Siren",
    "Slam",
    "Sliding_door",
    "Speech",
    "Speech_synthesizer",
    "Squeak",
    "Water_tap_and_faucet",
}
FSD50K_IN_SCOPE_MIDS = {
    "/m/07pp_mv",
    "/m/05tny_",
    "/m/0ytgt",
    "/m/0463cq4",
    "/m/0bt9lr",
    "/m/02dgv",
    "/m/03wwcy",
    "/m/02zsn",
    "/m/07r4wb8",
    "/m/05zppz",
    "/m/02y_763",
    "/m/09x0r",
    "/m/0brhx",
    "/m/07q6cd_",
    "/m/02jz0l",
    "/m/03kmc9",
    "/m/07rjzl8",
}
FSD50K_IN_SCOPE_KEYWORDS = (
    "alarm",
    "bark",
    "cry",
    "dog",
    "door",
    "knock",
    "siren",
    "slam",
    "sobb",
    "speech",
    "squeak",
    "water_tap",
    "faucet",
)


with (CONTRACTS_DIR / "class_labels.json").open("r", encoding="utf-8") as handle:
    CLASS_CONTRACT = json.load(handle)
with THRESHOLD_DEFAULTS_PATH.open("r", encoding="utf-8") as handle:
    THRESHOLD_DEFAULTS = json.load(handle)


CLASS_ORDER = [item["slug"] for item in CLASS_CONTRACT["classes"]]
SAMPLE_RATE = 16000
YAMNET_EMBEDDING_DIM = 1024


def cap_examples_per_class(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    split_frame = frame[frame["split"] == split_name].copy()
    max_examples = KNOWN_CAPS.get(split_name)
    if max_examples is None:
        return split_frame.reset_index(drop=True)
    return (
        split_frame.sort_values(["class_slug", "clip_id"])
        .groupby("class_slug", group_keys=False)
        .head(max_examples)
        .reset_index(drop=True)
    )


def load_known_manifest_frame() -> pd.DataFrame:
    manifest = load_dataset_manifest(CURATED_MANIFEST_PATH)
    manifest_df = manifest_to_dataframe(manifest)
    manifest_df = manifest_df[manifest_df["audio_path"].map(lambda value: Path(value).exists())].reset_index(drop=True)
    return pd.concat(
        [cap_examples_per_class(manifest_df, split_name) for split_name in ["train", "validation", "test"]],
        ignore_index=True,
    )


def parse_multi_value_field(value: str) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def is_out_of_scope_unknown(labels: list[str], mids: list[str]) -> bool:
    label_set = set(labels)
    mid_set = set(mids)
    if FSD50K_IN_SCOPE_MIDS & mid_set:
        return False
    if FSD50K_IN_SCOPE_LABELS & label_set:
        return False
    normalized_labels = [label.lower() for label in labels]
    for label in normalized_labels:
        if any(keyword in label for keyword in FSD50K_IN_SCOPE_KEYWORDS):
            return False
    return True


def cap_unknown_examples(frame: pd.DataFrame, max_total: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    capped = (
        frame.sort_values(["source_category", "clip_id"])
        .groupby("source_category", group_keys=False)
        .head(UNKNOWN_MAX_PER_CATEGORY)
        .reset_index(drop=True)
    )
    return capped.head(max_total).reset_index(drop=True)


def load_fsd50k_unknown_frame(
    metadata_path: Path,
    audio_dir: Path,
    dataset_name: str,
    max_total: int,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    rows: list[dict[str, Any]] = []
    for row in metadata.to_dict(orient="records"):
        labels = parse_multi_value_field(str(row.get("labels", "")))
        mids = parse_multi_value_field(str(row.get("mids", "")))
        if not labels or not is_out_of_scope_unknown(labels, mids):
            continue
        clip_stem = str(row["fname"])
        audio_path = audio_dir / f"{clip_stem}.wav"
        if not audio_path.exists():
            continue
        rows.append(
            {
                "clip_id": f"{dataset_name}-{clip_stem}",
                "audio_path": str(audio_path),
                "dataset": dataset_name,
                "source_category": labels[0],
                "source_labels": "|".join(labels),
                "source_mids": "|".join(mids),
                "true_label": "unknown",
            }
        )
    frame = pd.DataFrame(rows)
    return cap_unknown_examples(frame, max_total)


def load_audio_file(audio_path: str) -> np.ndarray:
    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if sample_rate != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return waveform.astype(np.float32)


def loudness_dbfs(waveform: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
    return 20.0 * math.log10(max(rms, 1e-12))


def resolve_embedding_tensor(output: object) -> np.ndarray:
    candidates: list[np.ndarray] = []
    if isinstance(output, dict):
        candidates.extend(np.array(value) for value in output.values())
    elif isinstance(output, (list, tuple)):
        candidates.extend(np.array(value) for value in output)
    else:
        candidates.append(np.array(output))
    for value in candidates:
        if value.ndim >= 1 and value.shape[-1] == YAMNET_EMBEDDING_DIM:
            return value
    raise ValueError("Unable to resolve a 1024-d embedding tensor from the YAMNet output")


def load_yamnet_model() -> Any:
    with YAMNET_ARTIFACT_SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        artifact_summary = json.load(handle)
    artifact_path = Path(artifact_summary["yamnet_artifact"])
    if artifact_path.name == "saved_model.pb":
        return tf.saved_model.load(str(artifact_path.parent))
    return tf.keras.models.load_model(str(artifact_path))


def extract_yamnet_embedding(model: Any, waveform: np.ndarray) -> np.ndarray:
    input_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    if hasattr(model, "signatures") and model.signatures:
        signature = next(iter(model.signatures.values()))
        output = signature(input_tensor)
    else:
        output = model(input_tensor)
    embedding_frames = resolve_embedding_tensor(output)
    if embedding_frames.ndim == 1:
        return embedding_frames.astype(np.float32)
    return np.array(embedding_frames).mean(axis=0).astype(np.float32)


def build_known_embeddings(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        waveform = load_audio_file(row["audio_path"])
        rows.append(
            {
                "clip_id": row["clip_id"],
                "split": row["split"],
                "true_label": row["class_slug"],
                "loudness_dbfs": loudness_dbfs(waveform),
                "embedding": extract_yamnet_embedding(model, waveform),
            }
        )
    return pd.DataFrame(rows)


def build_unknown_embeddings(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        waveform = load_audio_file(row["audio_path"])
        rows.append(
            {
                "clip_id": row["clip_id"],
                "dataset": row["dataset"],
                "source_category": row["source_category"],
                "source_labels": row["source_labels"],
                "source_mids": row["source_mids"],
                "true_label": "unknown",
                "loudness_dbfs": loudness_dbfs(waveform),
                "embedding": extract_yamnet_embedding(model, waveform),
            }
        )
    return pd.DataFrame(rows)


def build_prototypes(train_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    prototype_store: dict[str, np.ndarray] = {}
    for class_slug, class_frame in train_frame.groupby("true_label"):
        prototype_store[class_slug] = np.stack(class_frame["embedding"].to_list(), axis=0).mean(axis=0).astype(np.float32)
    return prototype_store


def score_embedding(embedding: np.ndarray, prototypes: dict[str, np.ndarray]) -> dict[str, float | str]:
    ordered = sorted(
        ((class_slug, float(np.sum((embedding - prototype) ** 2))) for class_slug, prototype in prototypes.items()),
        key=lambda item: item[1],
    )
    best_label, best_distance = ordered[0]
    second_distance = ordered[1][1] if len(ordered) > 1 else ordered[0][1]
    ratio = best_distance / max(second_distance, 1e-12)
    return {
        "top_label": best_label,
        "best_distance": best_distance,
        "second_distance": second_distance,
        "distance_ratio": ratio,
    }


def apply_thresholds(score: dict[str, float | str], loudness_value: float, thresholds: dict[str, float]) -> tuple[str, str]:
    if loudness_value < thresholds["loudness_floor_dbfs"]:
        return "background", "suppressed_by_loudness"
    if float(score["best_distance"]) > thresholds["distance_threshold"]:
        return "unknown", "rejected_by_distance"
    if float(score["distance_ratio"]) > thresholds["unknown_ratio_threshold"]:
        return "unknown", "rejected_by_ratio"
    return str(score["top_label"]), "accepted"


def classify_frame(
    frame: pd.DataFrame,
    prototypes: dict[str, np.ndarray],
    thresholds: dict[str, float] | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        score = score_embedding(np.array(row["embedding"], dtype=np.float32), prototypes)
        if thresholds is None:
            final_label = str(score["top_label"])
            decision = "nearest_prototype"
        else:
            final_label, decision = apply_thresholds(score, float(row["loudness_dbfs"]), thresholds)
        rows.append(
            {
                "clip_id": row["clip_id"],
                "true_label": row["true_label"],
                "final_label": final_label,
                "decision": decision,
                "top_label": score["top_label"],
                "best_distance": round(float(score["best_distance"]), 6),
                "second_distance": round(float(score["second_distance"]), 6),
                "distance_ratio": round(float(score["distance_ratio"]), 6),
                "loudness_dbfs": round(float(row["loudness_dbfs"]), 4),
                **({"split": row["split"]} if "split" in row else {}),
                **({"dataset": row["dataset"]} if "dataset" in row else {}),
                **({"source_category": row["source_category"]} if "source_category" in row else {}),
                **({"source_labels": row["source_labels"]} if "source_labels" in row else {}),
            }
        )
    return pd.DataFrame(rows)


def known_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    report = classification_report(
        predictions["true_label"],
        predictions["final_label"],
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    urgent_recalls = [float(report[class_slug]["recall"]) for class_slug in URGENT_CLASS_SLUGS]
    background_rows = predictions[predictions["true_label"] == "background"]
    background_non_alert_rate = float(background_rows["final_label"].isin(["background", "unknown"]).mean()) if not background_rows.empty else 0.0
    return {
        "accuracy": round(float(accuracy_score(predictions["true_label"], predictions["final_label"])), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        "urgent_recall": round(float(np.mean(urgent_recalls)), 4),
        "background_non_alert_rate": round(background_non_alert_rate, 4),
    }


def open_set_metrics(known_predictions: pd.DataFrame, unknown_predictions: pd.DataFrame) -> dict[str, float]:
    unknown_recall = float((unknown_predictions["final_label"] == "unknown").mean())
    unknown_non_alert_rate = float((unknown_predictions["final_label"].isin(["unknown", "background"])).mean())
    false_alert_rate = 1.0 - unknown_non_alert_rate

    combined = pd.concat(
        [
            known_predictions.assign(is_unknown_truth=False),
            unknown_predictions.assign(is_unknown_truth=True),
        ],
        ignore_index=True,
    )
    predicted_unknown = combined["final_label"] == "unknown"
    unknown_precision = float(combined[predicted_unknown]["is_unknown_truth"].mean()) if predicted_unknown.any() else 0.0

    return {
        "unknown_precision": round(unknown_precision, 4),
        "unknown_recall": round(unknown_recall, 4),
        "unknown_non_alert_rate": round(unknown_non_alert_rate, 4),
        "false_alert_rate": round(false_alert_rate, 4),
    }


def balanced_score(known_summary: dict[str, float], open_set_summary: dict[str, float]) -> float:
    return (
        0.4 * known_summary["macro_f1"]
        + 0.25 * known_summary["urgent_recall"]
        + 0.2 * open_set_summary["unknown_recall"]
        + 0.1 * open_set_summary["unknown_precision"]
        + 0.05 * open_set_summary["unknown_non_alert_rate"]
    )


def candidate_grids(
    calibration_known: pd.DataFrame,
    calibration_unknown: pd.DataFrame,
    prototypes: dict[str, np.ndarray],
) -> tuple[list[float], list[float], list[float]]:
    scored_known = classify_frame(calibration_known, prototypes, thresholds=None)
    scored_unknown = classify_frame(calibration_unknown, prototypes, thresholds=None)

    distance_values = np.concatenate(
        [scored_known["best_distance"].to_numpy(dtype=float), scored_unknown["best_distance"].to_numpy(dtype=float)]
    )
    ratio_values = np.concatenate(
        [scored_known["distance_ratio"].to_numpy(dtype=float), scored_unknown["distance_ratio"].to_numpy(dtype=float)]
    )
    loudness_values = np.concatenate(
        [calibration_known["loudness_dbfs"].to_numpy(dtype=float), calibration_unknown["loudness_dbfs"].to_numpy(dtype=float)]
    )

    distance_grid = sorted(
        {
            round(float(value), 4)
            for value in np.quantile(distance_values, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        }
    )
    ratio_grid = sorted(
        {
            round(float(value), 4)
            for value in np.quantile(ratio_values, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        }
        | {0.85, 0.9, 0.92, 0.95, 0.98}
    )
    loudness_grid = sorted(
        {
            round(float(value), 1)
            for value in np.quantile(loudness_values, [0.01, 0.05, 0.1, 0.2])
        }
        | {-60.0, -55.0, -52.0, -50.0, -48.0, -45.0}
    )
    return distance_grid, ratio_grid, loudness_grid


def build_candidate_frame(
    calibration_known: pd.DataFrame,
    calibration_unknown: pd.DataFrame,
    prototypes: dict[str, np.ndarray],
) -> tuple[dict[str, float], pd.DataFrame]:
    distance_grid, ratio_grid, loudness_grid = candidate_grids(calibration_known, calibration_unknown, prototypes)
    baseline_known_predictions = classify_frame(calibration_known, prototypes, thresholds=None)
    baseline_known_summary = known_metrics(baseline_known_predictions)
    rows: list[dict[str, Any]] = []

    for distance_threshold, unknown_ratio_threshold, loudness_floor_dbfs in itertools.product(
        distance_grid,
        ratio_grid,
        loudness_grid,
    ):
        thresholds = {
            "distance_threshold": float(distance_threshold),
            "unknown_ratio_threshold": float(unknown_ratio_threshold),
            "loudness_floor_dbfs": float(loudness_floor_dbfs),
        }
        known_predictions = classify_frame(calibration_known, prototypes, thresholds)
        unknown_predictions = classify_frame(calibration_unknown, prototypes, thresholds)
        known_summary = known_metrics(known_predictions)
        open_set_summary = open_set_metrics(known_predictions, unknown_predictions)
        rows.append(
            {
                "distance_threshold": round(float(distance_threshold), 4),
                "unknown_ratio_threshold": round(float(unknown_ratio_threshold), 4),
                "loudness_floor_dbfs": round(float(loudness_floor_dbfs), 1),
                "balanced_score": round(balanced_score(known_summary, open_set_summary), 4),
                "known_validation_accuracy": known_summary["accuracy"],
                "known_validation_macro_f1": known_summary["macro_f1"],
                "known_validation_macro_recall": known_summary["macro_recall"],
                "known_validation_urgent_recall": known_summary["urgent_recall"],
                "known_validation_background_non_alert_rate": known_summary["background_non_alert_rate"],
                "unknown_calibration_unknown_precision": open_set_summary["unknown_precision"],
                "unknown_calibration_unknown_recall": open_set_summary["unknown_recall"],
                "unknown_calibration_unknown_non_alert_rate": open_set_summary["unknown_non_alert_rate"],
                "unknown_calibration_false_alert_rate": open_set_summary["false_alert_rate"],
            }
        )

    candidate_frame = pd.DataFrame(rows)
    candidate_frame.insert(0, "candidate_id", range(1, len(candidate_frame) + 1))
    return baseline_known_summary, candidate_frame


def select_candidate_row(candidate_frame: pd.DataFrame, baseline_known_summary: dict[str, float]) -> tuple[pd.Series, dict[str, Any]]:
    # Hard floors — computed once and applied to every tolerance level. These prevent
    # the unconstrained (1.0, 1.0) fallback from selecting a degenerate point that
    # rejects nearly all known-set sounds by maximising raw unknown rejection.
    urgent_recall_floor = max(
        SELECTOR_URGENT_RECALL_FLOOR,
        baseline_known_summary["urgent_recall"] * 0.30,
    )
    macro_f1_floor = max(
        0.25,
        baseline_known_summary["macro_f1"] * SELECTOR_MACRO_F1_FLOOR_FRAC,
    )
    hard_feasible = candidate_frame[
        (candidate_frame["known_validation_urgent_recall"] >= urgent_recall_floor)
        & (candidate_frame["known_validation_macro_f1"] >= macro_f1_floor)
    ]
    # Use the hard-floored frame when possible, fall back to the full frame only
    # if the hard floor would leave us with nothing to pick from.
    search_frame = hard_feasible if not hard_feasible.empty else candidate_frame

    # Tolerance ladder — progressively relaxed relative drops.
    # Primary sort is balanced_score so we pick an overall compromise rather than
    # always maximising raw unknown rejection first.
    tolerances = [(0.02, 0.02), (0.03, 0.03), (0.05, 0.05), (0.10, 0.15), (0.15, 0.25), (1.0, 1.0)]
    for macro_f1_drop, urgent_drop in tolerances:
        feasible = search_frame[
            (search_frame["known_validation_macro_f1"] >= baseline_known_summary["macro_f1"] - macro_f1_drop)
            & (search_frame["known_validation_urgent_recall"] >= baseline_known_summary["urgent_recall"] - urgent_drop)
        ]
        if feasible.empty:
            continue
        selected = feasible.sort_values(
            [
                "balanced_score",
                "unknown_calibration_unknown_non_alert_rate",
                "unknown_calibration_unknown_recall",
                "known_validation_macro_f1",
                "known_validation_urgent_recall",
            ],
            ascending=[False, False, False, False, False],
        ).iloc[0]
        return selected, {
            "baseline_known_validation": baseline_known_summary,
            "selection_tolerance": {
                "macro_f1_drop": macro_f1_drop,
                "urgent_recall_drop": urgent_drop,
            },
            "hard_floor_urgent_recall": round(urgent_recall_floor, 4),
            "hard_floor_macro_f1": round(macro_f1_floor, 4),
            "selected_candidate_id": int(selected["candidate_id"]),
            "candidate_count": int(len(candidate_frame)),
        }
    raise RuntimeError("Threshold search did not yield any candidate")


def candidate_row_to_thresholds(candidate_row: pd.Series) -> dict[str, float]:
    return {
        "distance_threshold": float(candidate_row["distance_threshold"]),
        "unknown_ratio_threshold": float(candidate_row["unknown_ratio_threshold"]),
        "loudness_floor_dbfs": float(candidate_row["loudness_floor_dbfs"]),
    }


def pick_operating_points(
    candidate_frame: pd.DataFrame,
    baseline_known_summary: dict[str, float],
    selected_candidate_id: int,
) -> list[dict[str, Any]]:
    operating_points: list[dict[str, Any]] = []
    seen_candidate_ids: set[int] = set()

    def add_point(name: str, description: str, row: pd.Series) -> None:
        candidate_id = int(row["candidate_id"])
        if candidate_id in seen_candidate_ids:
            return
        operating_points.append(
            {
            "name": name,
            "description": description,
            "candidate_id": candidate_id,
            "thresholds": candidate_row_to_thresholds(row),
            "calibration_validation": {
                "accuracy": float(row["known_validation_accuracy"]),
                "macro_f1": float(row["known_validation_macro_f1"]),
                "macro_recall": float(row["known_validation_macro_recall"]),
                "urgent_recall": float(row["known_validation_urgent_recall"]),
                "background_non_alert_rate": float(row["known_validation_background_non_alert_rate"]),
            },
            "unknown_calibration": {
                "unknown_precision": float(row["unknown_calibration_unknown_precision"]),
                "unknown_recall": float(row["unknown_calibration_unknown_recall"]),
                "unknown_non_alert_rate": float(row["unknown_calibration_unknown_non_alert_rate"]),
                "false_alert_rate": float(row["unknown_calibration_false_alert_rate"]),
            },
            "balanced_score": float(row["balanced_score"]),
            }
        )
        seen_candidate_ids.add(candidate_id)

    selected_row = candidate_frame[candidate_frame["candidate_id"] == selected_candidate_id].iloc[0]
    add_point("selected_release_candidate", "Toleranced candidate that preserves known validation quality while improving rejection.", selected_row)

    best_balanced = candidate_frame.sort_values("balanced_score", ascending=False).iloc[0]
    add_point("best_balanced_score", "Highest overall calibration-stage balanced score.", best_balanced)

    best_known = candidate_frame.sort_values(
        ["known_validation_macro_f1", "known_validation_urgent_recall", "balanced_score"],
        ascending=[False, False, False],
    ).iloc[0]
    add_point("best_known_retention", "Strongest known-set retention on validation before considering open-set gains.", best_known)

    unknown_feasible = candidate_frame[
        (candidate_frame["known_validation_macro_f1"] >= baseline_known_summary["macro_f1"] - 0.05)
        & (candidate_frame["known_validation_urgent_recall"] >= baseline_known_summary["urgent_recall"] - 0.05)
    ]
    best_unknown = (unknown_feasible if not unknown_feasible.empty else candidate_frame).sort_values(
        [
            "unknown_calibration_unknown_non_alert_rate",
            "unknown_calibration_unknown_recall",
            "unknown_calibration_unknown_precision",
            "balanced_score",
        ],
        ascending=[False, False, False, False],
    ).iloc[0]
    add_point("best_unknown_rejection", "Most aggressive unknown rejection while staying near the known-set baseline when possible.", best_unknown)

    return operating_points


def predicted_label_breakdown(predictions: pd.DataFrame) -> dict[str, int]:
    counts = predictions["final_label"].value_counts().to_dict()
    return {str(key): int(value) for key, value in sorted(counts.items())}


def source_category_false_alerts(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    false_alerts = predictions[~predictions["final_label"].isin(["unknown", "background"])].copy()
    if false_alerts.empty:
        return []
    summary = (
        false_alerts.groupby(["source_category", "final_label"]).size().reset_index(name="count").sort_values("count", ascending=False)
    )
    return summary.head(15).to_dict(orient="records")


def evaluate_operating_points(
    operating_points: list[dict[str, Any]],
    test_known: pd.DataFrame,
    unknown_holdout: pd.DataFrame,
    prototypes: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for point in operating_points:
        thresholds = point["thresholds"]
        known_predictions = classify_frame(test_known, prototypes, thresholds)
        unknown_predictions = classify_frame(unknown_holdout, prototypes, thresholds)
        evaluated.append(
            {
                **point,
                "holdout_known_test": known_metrics(known_predictions),
                "holdout_open_set": open_set_metrics(known_predictions, unknown_predictions),
                "known_test_label_breakdown": predicted_label_breakdown(known_predictions),
                "unknown_holdout_label_breakdown": predicted_label_breakdown(unknown_predictions),
                "top_unknown_false_alert_categories": source_category_false_alerts(unknown_predictions),
            }
        )
    return evaluated


def write_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    baseline = summary["baseline_nearest_prototype"]
    selected = summary["holdout_evaluation"]
    operating_points = summary["operating_points"]
    top_candidates = summary["top_sweep_candidates"]
    calibrated = summary["calibrated_thresholds"]
    unknown_proxy = summary["unknown_proxy"]

    lines = [
        "# YAMNet Threshold Calibration Report",
        "",
        "## Protocol",
        "",
        "Threshold calibration was rerun on the YAMNet prototype path using out-of-scope FSD50K clips rather than the earlier ESC-50-only proxy set.",
        "",
        f"- Known caps: train={KNOWN_CAPS['train']}, validation={KNOWN_CAPS['validation']}, test={KNOWN_CAPS['test']}",
        f"- Unknown calibration proxy: FSD50K dev out-of-scope, support={unknown_proxy['calibration_support']}, categories={unknown_proxy['calibration_unique_source_categories']}",
        f"- Unknown holdout proxy: FSD50K eval out-of-scope, support={unknown_proxy['holdout_support']}, categories={unknown_proxy['holdout_unique_source_categories']}",
        f"- Unknown cap policy: max_total={UNKNOWN_CALIBRATION_CAP}/{UNKNOWN_HOLDOUT_CAP}, max_per_category={UNKNOWN_MAX_PER_CATEGORY}",
        f"- Urgent-release metric classes: {', '.join(URGENT_CLASS_SLUGS)}",
        "",
        "## Selected thresholds",
        "",
        f"- Distance threshold: {calibrated['distance_threshold']:.4f}",
        f"- Unknown ratio threshold: {calibrated['unknown_ratio_threshold']:.4f}",
        f"- Loudness floor dBFS: {calibrated['loudness_floor_dbfs']:.1f}",
        "",
        "## Operating-point comparison on holdout",
        "",
        "| Operating point | Accuracy | Macro F1 | Urgent recall | Unknown non-alert | Unknown recall | Unknown precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in operating_points:
        lines.append(
            f"| {point['name']} | {point['holdout_known_test']['accuracy']:.4f} | {point['holdout_known_test']['macro_f1']:.4f} | {point['holdout_known_test']['urgent_recall']:.4f} | {point['holdout_open_set']['unknown_non_alert_rate']:.4f} | {point['holdout_open_set']['unknown_recall']:.4f} | {point['holdout_open_set']['unknown_precision']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Baseline vs selected on known test",
            "",
            "| Metric | Baseline | Selected | Delta |",
            "| --- | ---: | ---: | ---: |",
            f"| Accuracy | {baseline['known_test']['accuracy']:.4f} | {selected['known_test']['accuracy']:.4f} | {selected['known_test']['accuracy'] - baseline['known_test']['accuracy']:+.4f} |",
            f"| Macro F1 | {baseline['known_test']['macro_f1']:.4f} | {selected['known_test']['macro_f1']:.4f} | {selected['known_test']['macro_f1'] - baseline['known_test']['macro_f1']:+.4f} |",
            f"| Macro recall | {baseline['known_test']['macro_recall']:.4f} | {selected['known_test']['macro_recall']:.4f} | {selected['known_test']['macro_recall'] - baseline['known_test']['macro_recall']:+.4f} |",
            f"| Urgent recall | {baseline['known_test']['urgent_recall']:.4f} | {selected['known_test']['urgent_recall']:.4f} | {selected['known_test']['urgent_recall'] - baseline['known_test']['urgent_recall']:+.4f} |",
            f"| Background non-alert rate | {baseline['known_test']['background_non_alert_rate']:.4f} | {selected['known_test']['background_non_alert_rate']:.4f} | {selected['known_test']['background_non_alert_rate'] - baseline['known_test']['background_non_alert_rate']:+.4f} |",
            "",
            "## Selected open-set holdout results",
            "",
            f"- Unknown precision: {selected['open_set_holdout']['unknown_precision']:.4f}",
            f"- Unknown recall: {selected['open_set_holdout']['unknown_recall']:.4f}",
            f"- Unknown non-alert rate: {selected['open_set_holdout']['unknown_non_alert_rate']:.4f}",
            f"- Unknown false-alert rate: {selected['open_set_holdout']['false_alert_rate']:.4f}",
            "",
            "## Top sweep candidates on calibration",
            "",
            "| Candidate ID | Distance | Ratio | Loudness | Balanced | Val macro F1 | Val urgent recall | Unknown non-alert |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate in top_candidates:
        lines.append(
            f"| {candidate['candidate_id']} | {candidate['distance_threshold']:.4f} | {candidate['unknown_ratio_threshold']:.4f} | {candidate['loudness_floor_dbfs']:.1f} | {candidate['balanced_score']:.4f} | {candidate['known_validation_macro_f1']:.4f} | {candidate['known_validation_urgent_recall']:.4f} | {candidate['unknown_calibration_unknown_non_alert_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Holdout routing breakdown",
            "",
            f"- Selected known-test routed labels: {json.dumps(selected['known_test_label_breakdown'], sort_keys=True)}",
            f"- Selected unknown-holdout routed labels: {json.dumps(selected['unknown_holdout_label_breakdown'], sort_keys=True)}",
            "",
            "## Assessment",
            "",
            "This remains an M7 calibration artifact rather than a ship-default threshold package. The difference now is that the proxy unknown pool is materially harder and more realistic, and the sweep export makes the tradeoff between known retention and unknown rejection explicit rather than hiding it behind one selected point.",
            "",
            f"Hard floors applied during selection: urgent recall ≥ {summary['calibration_selection'].get('hard_floor_urgent_recall', 'N/A')}, macro F1 ≥ {summary['calibration_selection'].get('hard_floor_macro_f1', 'N/A')}.",
        ]
    )

    if selected["known_test"]["urgent_recall"] < 0.9:
        lines.extend(
            [
                "",
                f"The selected operating point does not yet meet the PRD urgent-recall target of 0.90. Current urgent recall is {selected['known_test']['urgent_recall']:.4f}.",
            ]
        )

    false_alerts = selected["top_unknown_false_alert_categories"]
    if false_alerts:
        lines.extend(["", "## Top holdout false alerts", ""])
        for item in false_alerts:
            lines.append(f"- {item['source_category']} -> {item['final_label']}: {item['count']}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    model = load_yamnet_model()
    known_frame = load_known_manifest_frame()
    unknown_calibration_frame = load_fsd50k_unknown_frame(
        FSD50K_DEV_METADATA_PATH,
        FSD50K_DEV_AUDIO_DIR,
        "fsd50k_dev_out_of_scope",
        UNKNOWN_CALIBRATION_CAP,
    )
    unknown_holdout_frame = load_fsd50k_unknown_frame(
        FSD50K_EVAL_METADATA_PATH,
        FSD50K_EVAL_AUDIO_DIR,
        "fsd50k_eval_out_of_scope",
        UNKNOWN_HOLDOUT_CAP,
    )
    if unknown_calibration_frame.empty or unknown_holdout_frame.empty:
        raise RuntimeError("FSD50K unknown proxy selection returned no local audio clips")

    known_embeddings = build_known_embeddings(model, known_frame)
    unknown_calibration_embeddings = build_unknown_embeddings(model, unknown_calibration_frame)
    unknown_holdout_embeddings = build_unknown_embeddings(model, unknown_holdout_frame)

    prototypes = build_prototypes(known_embeddings[known_embeddings["split"] == "train"])
    validation_known = known_embeddings[known_embeddings["split"] == "validation"].reset_index(drop=True)
    test_known = known_embeddings[known_embeddings["split"] == "test"].reset_index(drop=True)

    baseline_known_validation_summary, candidate_frame = build_candidate_frame(
        validation_known,
        unknown_calibration_embeddings,
        prototypes,
    )
    selected_candidate_row, calibration_selection = select_candidate_row(candidate_frame, baseline_known_validation_summary)
    operating_points = pick_operating_points(
        candidate_frame,
        baseline_known_validation_summary,
        int(selected_candidate_row["candidate_id"]),
    )
    evaluated_operating_points = evaluate_operating_points(
        operating_points,
        test_known,
        unknown_holdout_embeddings,
        prototypes,
    )

    operating_point_ids = {point["candidate_id"]: point["name"] for point in evaluated_operating_points}
    candidate_frame = candidate_frame.copy()
    candidate_frame["operating_point"] = candidate_frame["candidate_id"].map(operating_point_ids).fillna("")
    candidate_frame["selected_release_candidate"] = candidate_frame["candidate_id"] == int(selected_candidate_row["candidate_id"])

    selected_operating_point = next(point for point in evaluated_operating_points if point["name"] == "selected_release_candidate")
    selected_thresholds = selected_operating_point["thresholds"]
    baseline_known_test = classify_frame(test_known, prototypes, thresholds=None)
    selected_known_test = classify_frame(test_known, prototypes, thresholds=selected_thresholds)
    selected_unknown_holdout = classify_frame(unknown_holdout_embeddings, prototypes, thresholds=selected_thresholds)
    selected_unknown_calibration = classify_frame(unknown_calibration_embeddings, prototypes, thresholds=selected_thresholds)

    baseline_summary = {
        "known_test": known_metrics(baseline_known_test),
    }
    holdout_summary = {
        "known_test": selected_operating_point["holdout_known_test"],
        "open_set_holdout": selected_operating_point["holdout_open_set"],
        "open_set_calibration": open_set_metrics(
            classify_frame(validation_known, prototypes, thresholds=selected_thresholds),
            selected_unknown_calibration,
        ),
        "known_test_label_breakdown": predicted_label_breakdown(selected_known_test),
        "unknown_holdout_label_breakdown": predicted_label_breakdown(selected_unknown_holdout),
        "top_unknown_false_alert_categories": source_category_false_alerts(selected_unknown_holdout),
    }

    candidate_thresholds = {
        "version": "1.0.0-yamnet-calibrated-fsd50k-2026-03-11",
        "distance_threshold": round(selected_thresholds["distance_threshold"], 4),
        "loudness_floor_dbfs": round(selected_thresholds["loudness_floor_dbfs"], 1),
        "unknown_ratio_threshold": round(selected_thresholds["unknown_ratio_threshold"], 4),
        "confidence_tiers": THRESHOLD_DEFAULTS["confidence_tiers"],
        "modes": THRESHOLD_DEFAULTS["modes"],
    }

    summary = {
        "protocol": "yamnet_threshold_calibration_with_fsd50k_out_of_scope_unknowns",
        "unknown_proxy": {
            "calibration_dataset": "fsd50k_dev_out_of_scope",
            "holdout_dataset": "fsd50k_eval_out_of_scope",
            "calibration_support": int(len(unknown_calibration_frame)),
            "holdout_support": int(len(unknown_holdout_frame)),
            "calibration_unique_source_categories": int(unknown_calibration_frame["source_category"].nunique()),
            "holdout_unique_source_categories": int(unknown_holdout_frame["source_category"].nunique()),
        },
        "baseline_nearest_prototype": baseline_summary,
        "calibrated_thresholds": candidate_thresholds,
        "calibration_selection": calibration_selection,
        "operating_points": evaluated_operating_points,
        "selected_operating_point": "selected_release_candidate",
        "holdout_evaluation": holdout_summary,
        "top_sweep_candidates": candidate_frame.sort_values(
            ["balanced_score", "known_validation_macro_f1", "unknown_calibration_unknown_non_alert_rate"],
            ascending=[False, False, False],
        ).head(10).to_dict(orient="records"),
        "artifacts": {
            "known_test_predictions": "ml/evaluation/yamnet_known_test_calibrated_predictions.csv",
            "unknown_holdout_predictions": "ml/evaluation/yamnet_unknown_holdout_calibrated_predictions.csv",
            "candidate_thresholds": "ml/evaluation/yamnet_threshold_candidate.json",
            "threshold_sweep": "ml/evaluation/yamnet_threshold_sweep.csv",
            "operating_points": "ml/evaluation/yamnet_threshold_operating_points.json",
            "report": "ml/evaluation/yamnet_threshold_calibration_report.md",
        },
    }

    (EVALUATION_DIR / "yamnet_known_test_calibrated_predictions.csv").write_text(
        selected_known_test.to_csv(index=False),
        encoding="utf-8",
    )
    (EVALUATION_DIR / "yamnet_unknown_holdout_calibrated_predictions.csv").write_text(
        selected_unknown_holdout.to_csv(index=False),
        encoding="utf-8",
    )
    (EVALUATION_DIR / "yamnet_threshold_candidate.json").write_text(
        json.dumps(candidate_thresholds, indent=2),
        encoding="utf-8",
    )
    (CONTRACTS_DIR / "yamnet_threshold_defaults.json").write_text(
        json.dumps(candidate_thresholds, indent=2),
        encoding="utf-8",
    )
    (EVALUATION_DIR / "yamnet_threshold_sweep.csv").write_text(
        candidate_frame.to_csv(index=False),
        encoding="utf-8",
    )
    (EVALUATION_DIR / "yamnet_threshold_operating_points.json").write_text(
        json.dumps(evaluated_operating_points, indent=2),
        encoding="utf-8",
    )
    (EVALUATION_DIR / "yamnet_threshold_calibration_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(summary, EVALUATION_DIR / "yamnet_threshold_calibration_report.md")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()