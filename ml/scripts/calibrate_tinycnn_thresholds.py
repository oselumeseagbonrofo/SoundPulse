from __future__ import annotations

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
TINYCNN_ARTIFACT_DIR = ML_ROOT / "export" / "artifacts" / "generated" / "tinycnn_baseline"
THRESHOLD_DEFAULTS_PATH = CONTRACTS_DIR / "threshold_defaults.json"
EVALUATION_DIR = ML_ROOT / "evaluation"

KNOWN_CAPS = {"train": 128, "validation": 64, "test": 64}
UNKNOWN_CALIBRATION_CAP = 1024
UNKNOWN_HOLDOUT_CAP = 1024
UNKNOWN_MAX_PER_CATEGORY = 12
URGENT_CLASS_SLUGS = ["fire_alarm", "clock_alarm", "door_knock", "doorbell"]

SELECTOR_URGENT_RECALL_FLOOR = 0.10
SELECTOR_MACRO_F1_FLOOR_FRAC = 0.40

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
with (CONTRACTS_DIR / "preprocessing_config.json").open("r", encoding="utf-8") as handle:
    PREPROCESSING_CONTRACT = json.load(handle)

CLASS_ORDER = [item["slug"] for item in CLASS_CONTRACT["classes"]]
SAMPLE_RATE = PREPROCESSING_CONTRACT["audio"]["sample_rate_hz"]
WINDOW_DURATION_MS = PREPROCESSING_CONTRACT["audio"]["window_duration_ms"]
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION_MS / 1000)
TINYCNN_EMBEDDING_DIM = PREPROCESSING_CONTRACT["embedding_contract"]["tiny_cnn_dim"]
IMAGE_SIZE = (128, 128)


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
    # Pad or trim to WINDOW_SAMPLES
    if waveform.shape[0] < WINDOW_SAMPLES:
        waveform = np.pad(waveform, (0, WINDOW_SAMPLES - waveform.shape[0]))
    else:
        waveform = waveform[:WINDOW_SAMPLES]
    return waveform.astype(np.float32)


def loudness_dbfs(waveform: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
    return 20.0 * math.log10(max(rms, 1e-12))


def waveform_to_rgb_log_mel(waveform: np.ndarray) -> tf.Tensor:
    waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    stft = tf.signal.stft(
        waveform_tensor,
        frame_length=PREPROCESSING_CONTRACT["stft"]["win_length"],
        frame_step=PREPROCESSING_CONTRACT["stft"]["hop_length"],
        fft_length=PREPROCESSING_CONTRACT["stft"]["n_fft"],
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    spectrogram = tf.abs(stft) ** PREPROCESSING_CONTRACT["mel"]["power"]
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=PREPROCESSING_CONTRACT["mel"]["n_mels"],
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=PREPROCESSING_CONTRACT["mel"]["f_min_hz"],
        upper_edge_hertz=PREPROCESSING_CONTRACT["mel"]["f_max_hz"],
    )
    mel_spectrogram = tf.matmul(spectrogram, mel_matrix)
    log_mel = tf.math.log(mel_spectrogram + PREPROCESSING_CONTRACT["mel"]["log_offset"])
    log_mel = tf.clip_by_value(
        log_mel,
        PREPROCESSING_CONTRACT["normalization"]["clip_min_db"],
        PREPROCESSING_CONTRACT["normalization"]["clip_max_db"],
    )
    log_mel = tf.expand_dims(log_mel, axis=-1)
    log_mel = tf.image.resize(log_mel, IMAGE_SIZE)
    log_mel = tf.repeat(log_mel, repeats=3, axis=-1)
    log_mel = tf.image.per_image_standardization(log_mel)
    return tf.expand_dims(log_mel, axis=0)


def load_tinycnn_model() -> Any:
    saved_model_path = TINYCNN_ARTIFACT_DIR / "saved_model"
    model = tf.saved_model.load(str(saved_model_path))
    if hasattr(model, "signatures") and model.signatures:
        return next(iter(model.signatures.values()))
    return model


def extract_tinycnn_embedding(model: Any, waveform: np.ndarray) -> np.ndarray:
    input_tensor = waveform_to_rgb_log_mel(waveform)
    output = model(input_tensor)
    if isinstance(output, dict):
        output_tensor = list(output.values())[0]
    elif isinstance(output, (list, tuple)):
        output_tensor = output[0]
    else:
        output_tensor = output
    embedding = np.array(output_tensor)
    if embedding.ndim == 1:
        return embedding.astype(np.float32)
    return embedding[0].astype(np.float32)


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
                "embedding": extract_tinycnn_embedding(model, waveform),
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
                "embedding": extract_tinycnn_embedding(model, waveform),
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
        "predicted_label": best_label,
        "best_distance": best_distance,
        "second_distance": second_distance,
        "ratio": ratio,
    }


def classify_frame(
    frame: pd.DataFrame,
    prototypes: dict[str, np.ndarray],
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    scored: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        score = score_embedding(row["embedding"], prototypes)
        label = score["predicted_label"]
        ratio = score["ratio"]

        if thresholds is not None:
            if row["loudness_dbfs"] < thresholds["loudness_floor_dbfs"]:
                label = "unknown"
            elif ratio > thresholds["unknown_ratio_threshold"]:
                label = "unknown"
            elif score["best_distance"] > thresholds["distance_threshold"]:
                label = "unknown"

        scored.append(
            {
                "clip_id": row["clip_id"],
                "true_label": row["true_label"],
                "predicted_label": label,
                "best_distance": score["best_distance"],
                "ratio": ratio,
                "loudness_dbfs": row["loudness_dbfs"],
            }
        )
    return pd.DataFrame(scored)


def known_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y_true = predictions["true_label"].tolist()
    y_pred = predictions["predicted_label"].tolist()
    accuracy = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = float(report["macro avg"]["f1-score"])
    macro_recall = float(report["macro avg"]["recall"])
    urgent_recalls = [
        float(report[class_slug]["recall"]) for class_slug in URGENT_CLASS_SLUGS if class_slug in report
    ]
    urgent_recall = float(np.mean(urgent_recalls)) if urgent_recalls else 0.0
    background_non_alert_rate = (
        float(report["background"]["recall"]) if "background" in report else 0.0
    )
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_recall": round(macro_recall, 4),
        "urgent_recall": round(urgent_recall, 4),
        "background_non_alert_rate": round(background_non_alert_rate, 4),
    }


def open_set_metrics(known_predictions: pd.DataFrame, unknown_predictions: pd.DataFrame) -> dict[str, float]:
    known_clean = known_predictions[known_predictions["true_label"] != "background"].copy()
    unknowns = unknown_predictions.copy()

    total_known = len(known_clean)
    total_unknowns = len(unknowns)

    false_alerts_known = int((known_clean["predicted_label"] == "unknown").sum())
    correct_rejects_unknown = int((unknowns["predicted_label"] == "unknown").sum())

    unknown_recall = correct_rejects_unknown / max(total_unknowns, 1)
    unknown_precision = correct_rejects_unknown / max(correct_rejects_unknown + false_alerts_known, 1)
    unknown_non_alert_rate = unknown_recall
    false_alert_rate = 1.0 - unknown_recall

    return {
        "unknown_precision": round(unknown_precision, 4),
        "unknown_recall": round(unknown_recall, 4),
        "unknown_non_alert_rate": round(unknown_non_alert_rate, 4),
        "false_alert_rate": round(false_alert_rate, 4),
    }


def build_candidate_frame(
    validation_known: pd.DataFrame,
    unknown_calibration: pd.DataFrame,
    prototypes: dict[str, np.ndarray],
) -> tuple[dict[str, float], pd.DataFrame]:
    baseline_predictions = classify_frame(validation_known, prototypes, thresholds=None)
    baseline_summary = known_metrics(baseline_predictions)

    distance_min = float(validation_known["embedding"].map(lambda e: float(np.min([np.sum((e - p) ** 2) for p in prototypes.values()]))).min())
    distance_max = float(validation_known["embedding"].map(lambda e: float(np.max([np.sum((e - p) ** 2) for p in prototypes.values()]))).max())

    distance_sweep = np.linspace(distance_min, distance_max, 30)
    ratio_sweep = np.linspace(0.85, 1.0, 8)
    loudness_sweep = np.linspace(-60.0, -35.0, 11)

    rows: list[dict[str, Any]] = []
    candidate_id = 0
    for d_thresh in distance_sweep:
        for r_thresh in ratio_sweep:
            for l_thresh in loudness_sweep:
                thresholds = {
                    "distance_threshold": d_thresh,
                    "unknown_ratio_threshold": r_thresh,
                    "loudness_floor_dbfs": l_thresh,
                }
                k_pred = classify_frame(validation_known, prototypes, thresholds=thresholds)
                u_pred = classify_frame(unknown_calibration, prototypes, thresholds=thresholds)

                k_metrics = known_metrics(k_pred)
                o_metrics = open_set_metrics(k_pred, u_pred)

                f1_score_ratio = k_metrics["macro_f1"] / max(baseline_summary["macro_f1"], 1e-12)
                alert_penalty = o_metrics["false_alert_rate"]

                balanced_score = 0.5 * f1_score_ratio + 0.5 * (1.0 - alert_penalty)
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "distance_threshold": round(d_thresh, 4),
                        "unknown_ratio_threshold": round(r_thresh, 4),
                        "loudness_floor_dbfs": round(l_thresh, 1),
                        "balanced_score": round(balanced_score, 4),
                        "known_validation_accuracy": k_metrics["accuracy"],
                        "known_validation_macro_f1": k_metrics["macro_f1"],
                        "known_validation_macro_recall": k_metrics["macro_recall"],
                        "known_validation_urgent_recall": k_metrics["urgent_recall"],
                        "known_validation_background_non_alert_rate": k_metrics["background_non_alert_rate"],
                        "unknown_calibration_unknown_precision": o_metrics["unknown_precision"],
                        "unknown_calibration_unknown_recall": o_metrics["unknown_recall"],
                        "unknown_calibration_unknown_non_alert_rate": o_metrics["unknown_non_alert_rate"],
                        "unknown_calibration_false_alert_rate": o_metrics["false_alert_rate"],
                    }
                )
                candidate_id += 1

    return baseline_summary, pd.DataFrame(rows)


def select_candidate_row(candidate_frame: pd.DataFrame, baseline_summary: dict[str, float]) -> tuple[pd.Series, dict[str, Any]]:
    f1_min = baseline_summary["macro_f1"] - 0.02
    recall_min = baseline_summary["urgent_recall"] - 0.02

    urgent_recall_floor = max(recall_min, SELECTOR_URGENT_RECALL_FLOOR)
    macro_f1_floor = max(f1_min, baseline_summary["macro_f1"] * SELECTOR_MACRO_F1_FLOOR_FRAC)

    filtered = candidate_frame[
        (candidate_frame["known_validation_macro_f1"] >= macro_f1_floor)
        & (candidate_frame["known_validation_urgent_recall"] >= urgent_recall_floor)
    ]

    if filtered.empty:
        filtered = candidate_frame[
            (candidate_frame["known_validation_macro_f1"] >= 0.25)
            & (candidate_frame["known_validation_urgent_recall"] >= 0.10)
        ]

    if filtered.empty:
        best_row = candidate_frame.sort_values("balanced_score", ascending=False).iloc[0]
    else:
        best_row = filtered.sort_values(
            ["balanced_score", "known_validation_macro_f1", "unknown_calibration_unknown_non_alert_rate"],
            ascending=[False, False, False],
        ).iloc[0]

    selection_info = {
        "baseline_known_validation": baseline_summary,
        "selection_tolerance": {"macro_f1_drop": 0.02, "urgent_recall_drop": 0.02},
        "hard_floor_urgent_recall": 0.10,
        "hard_floor_macro_f1": 0.25,
        "selected_candidate_id": int(best_row["candidate_id"]),
        "candidate_count": len(candidate_frame),
    }
    return best_row, selection_info


def pick_operating_points(
    candidate_frame: pd.DataFrame,
    baseline_summary: dict[str, float],
    selected_id: int,
) -> list[dict[str, Any]]:
    selected_row = candidate_frame[candidate_frame["candidate_id"] == selected_id].iloc[0]
    best_balanced_row = candidate_frame.sort_values("balanced_score", ascending=False).iloc[0]

    rejection_candidates = candidate_frame[
        (candidate_frame["known_validation_macro_f1"] >= baseline_summary["macro_f1"] - 0.06)
        & (candidate_frame["known_validation_urgent_recall"] >= baseline_summary["urgent_recall"] - 0.06)
    ]
    if rejection_candidates.empty:
        rejection_candidates = candidate_frame
    best_rejection_row = rejection_candidates.sort_values(
        ["unknown_calibration_unknown_non_alert_rate", "balanced_score"],
        ascending=[False, False],
    ).iloc[0]

    return [
        {
            "name": "selected_release_candidate",
            "description": "Toleranced candidate that preserves known validation quality while improving rejection.",
            "candidate_id": int(selected_row["candidate_id"]),
            "thresholds": {
                "distance_threshold": float(selected_row["distance_threshold"]),
                "unknown_ratio_threshold": float(selected_row["unknown_ratio_threshold"]),
                "loudness_floor_dbfs": float(selected_row["loudness_floor_dbfs"]),
            },
        },
        {
            "name": "best_balanced_score",
            "description": "Highest overall calibration-stage balanced score.",
            "candidate_id": int(best_balanced_row["candidate_id"]),
            "thresholds": {
                "distance_threshold": float(best_balanced_row["distance_threshold"]),
                "unknown_ratio_threshold": float(best_balanced_row["unknown_ratio_threshold"]),
                "loudness_floor_dbfs": float(best_balanced_row["loudness_floor_dbfs"]),
            },
        },
        {
            "name": "best_unknown_rejection",
            "description": "Most aggressive unknown rejection while staying near the known-set baseline when possible.",
            "candidate_id": int(best_rejection_row["candidate_id"]),
            "thresholds": {
                "distance_threshold": float(best_rejection_row["distance_threshold"]),
                "unknown_ratio_threshold": float(best_rejection_row["unknown_ratio_threshold"]),
                "loudness_floor_dbfs": float(best_rejection_row["loudness_floor_dbfs"]),
            },
        },
    ]


def evaluate_operating_points(
    operating_points: list[dict[str, Any]],
    test_known: pd.DataFrame,
    unknown_holdout: pd.DataFrame,
    prototypes: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    evaluated = []
    for point in operating_points:
        t_pred = classify_frame(test_known, prototypes, thresholds=point["thresholds"])
        u_pred = classify_frame(unknown_holdout, prototypes, thresholds=point["thresholds"])

        k_metrics = known_metrics(t_pred)
        o_metrics = open_set_metrics(t_pred, u_pred)

        f1_score_ratio = k_metrics["macro_f1"] / max(0.1, 1e-12)
        alert_penalty = o_metrics["false_alert_rate"]
        balanced = 0.5 * f1_score_ratio + 0.5 * (1.0 - alert_penalty)

        evaluated.append(
            {
                "name": point["name"],
                "description": point["description"],
                "candidate_id": point["candidate_id"],
                "thresholds": point["thresholds"],
                "calibration_validation": k_metrics,
                "unknown_calibration": o_metrics,
                "balanced_score": round(float(balanced), 4),
                "holdout_known_test": k_metrics,
                "holdout_open_set": o_metrics,
                "known_test_label_breakdown": predicted_label_breakdown(t_pred),
                "unknown_holdout_label_breakdown": predicted_label_breakdown(u_pred),
                "top_unknown_false_alert_categories": source_category_false_alerts(u_pred),
            }
        )
    return evaluated


def predicted_label_breakdown(predictions: pd.DataFrame) -> dict[str, int]:
    counts = predictions["predicted_label"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def source_category_false_alerts(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    false_alerts = predictions[predictions["predicted_label"] != "unknown"].copy()
    if false_alerts.empty:
        return []
    grouped = (
        false_alerts.groupby(["source_category", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    entries = []
    for row in grouped.head(15).to_dict(orient="records"):
        entries.append(
            {
                "source_category": str(row["source_category"]),
                "final_label": str(row["predicted_label"]),
                "count": int(row["count"]),
            }
        )
    return entries


def write_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# TinyCNN / MobileNetV4 Decision Threshold Calibration",
        "",
        "## Calibration summary",
        "",
        "- Unknown calibration proxy dataset: fsd50k_dev_out_of_scope",
        "- Unknown holdout proxy dataset: fsd50k_eval_out_of_scope",
        f"- Target candidate ID selected: {summary['calibration_selection']['selected_candidate_id']}",
        "",
        "## Selected Operating Point Defaults",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| distance_threshold | {summary['calibrated_thresholds']['distance_threshold']:.4f} |",
        f"| unknown_ratio_threshold | {summary['calibrated_thresholds']['unknown_ratio_threshold']:.4f} |",
        f"| loudness_floor_dbfs | {summary['calibrated_thresholds']['loudness_floor_dbfs']:.1f} |",
        "",
        "## Performance comparison",
        "",
        "| Phase | Accuracy | Macro F1 | Macro Recall | Urgent Recall | Background Non-Alert Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    base = summary["baseline_nearest_prototype"]["known_test"]
    lines.append(
        f"| Baseline Known Test | {base['accuracy']:.4f} | {base['macro_f1']:.4f} | {base['macro_recall']:.4f} | {base['urgent_recall']:.4f} | {base['background_non_alert_rate']:.4f} |"
    )

    release = summary["holdout_evaluation"]["known_test"]
    lines.append(
        f"| Calibrated Known Test | {release['accuracy']:.4f} | {release['macro_f1']:.4f} | {release['macro_recall']:.4f} | {release['urgent_recall']:.4f} | {release['background_non_alert_rate']:.4f} |"
    )

    lines.extend(
        [
            "",
            "## Open-set Rejection Metrics",
            "",
            "| Phase | Unknown Precision | Unknown Recall (Rejection Rate) | False Alert Rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    cal = summary["holdout_evaluation"]["open_set_calibration"]
    lines.append(
        f"| Calibration | {cal['unknown_precision']:.4f} | {cal['unknown_recall']:.4f} | {cal['false_alert_rate']:.4f} |"
    )

    hold = summary["holdout_evaluation"]["open_set_holdout"]
    lines.append(
        f"| Holdout | {hold['unknown_precision']:.4f} | {hold['unknown_recall']:.4f} | {hold['false_alert_rate']:.4f} |"
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading TinyCNN SavedModel...")
    model = load_tinycnn_model()
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

    print("Extracting known subset embeddings...")
    known_embeddings = build_known_embeddings(model, known_frame)
    print("Extracting unknown calibration embeddings...")
    unknown_calibration_embeddings = build_unknown_embeddings(model, unknown_calibration_frame)
    print("Extracting unknown holdout embeddings...")
    unknown_holdout_embeddings = build_unknown_embeddings(model, unknown_holdout_frame)

    prototypes = build_prototypes(known_embeddings[known_embeddings["split"] == "train"])
    validation_known = known_embeddings[known_embeddings["split"] == "validation"].reset_index(drop=True)
    test_known = known_embeddings[known_embeddings["split"] == "test"].reset_index(drop=True)

    print("Running threshold grid search sweep...")
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
        "version": f"1.0.0-mobilenetv4-calibrated-fsd50k-2026-06-10",
        "distance_threshold": round(selected_thresholds["distance_threshold"], 4),
        "loudness_floor_dbfs": round(selected_thresholds["loudness_floor_dbfs"], 1),
        "unknown_ratio_threshold": round(selected_thresholds["unknown_ratio_threshold"], 4),
        "confidence_tiers": THRESHOLD_DEFAULTS["confidence_tiers"],
        "modes": THRESHOLD_DEFAULTS["modes"],
    }

    summary = {
        "protocol": "mobilenetv4_threshold_calibration_with_fsd50k_out_of_scope_unknowns",
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
            "known_test_predictions": "ml/evaluation/mobilenetv4_known_test_calibrated_predictions.csv",
            "unknown_holdout_predictions": "ml/evaluation/mobilenetv4_unknown_holdout_calibrated_predictions.csv",
            "candidate_thresholds": "packages/shared_contracts/v1/tinycnn_threshold_defaults.json",
            "threshold_sweep": "ml/evaluation/mobilenetv4_threshold_sweep.csv",
            "operating_points": "ml/evaluation/mobilenetv4_threshold_operating_points.json",
            "report": "ml/evaluation/mobilenetv4_threshold_calibration_report.md",
        },
    }

    print("Writing evaluation predictions and sweep reports...")
    (EVALUATION_DIR / "mobilenetv4_known_test_calibrated_predictions.csv").write_text(
        selected_known_test.to_csv(index=False),
        encoding="utf-8",
    )
    (EVALUATION_DIR / "mobilenetv4_unknown_holdout_calibrated_predictions.csv").write_text(
        selected_unknown_holdout.to_csv(index=False),
        encoding="utf-8",
    )
    (CONTRACTS_DIR / "tinycnn_threshold_defaults.json").write_text(
        json.dumps(candidate_thresholds, indent=2),
        encoding="utf-8",
    )
    (EVALUATION_DIR / "mobilenetv4_threshold_sweep.csv").write_text(
        candidate_frame.to_csv(index=False),
        encoding="utf-8",
    )
    (EVALUATION_DIR / "mobilenetv4_threshold_operating_points.json").write_text(
        json.dumps(evaluated_operating_points, indent=2),
        encoding="utf-8",
    )
    (EVALUATION_DIR / "mobilenetv4_threshold_calibration_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(summary, EVALUATION_DIR / "mobilenetv4_threshold_calibration_report.md")
    print("Done!")


if __name__ == "__main__":
    main()
