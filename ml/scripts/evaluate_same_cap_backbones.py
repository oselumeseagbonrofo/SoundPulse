from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import classification_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "ml"
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soundpulse_ml.datasets import load_dataset_manifest, manifest_to_dataframe


CONTRACTS_DIR = PROJECT_ROOT / "packages" / "shared_contracts" / "v1"
CURATED_MANIFEST_PATH = ML_ROOT / "data" / "curated" / "soundpulse_manifest.v1.json"
TINYCNN_ARTIFACT_DIR = ML_ROOT / "export" / "artifacts" / "generated" / "tinycnn_baseline"
YAMNET_ARTIFACT_DIR = ML_ROOT / "export" / "artifacts" / "generated" / "yamnet_kaggle"
EVALUATION_DIR = ML_ROOT / "evaluation"
COMMON_CAPS = {"train": 128, "validation": 64, "test": 64}
IMAGE_SIZE = (128, 128)


with (CONTRACTS_DIR / "class_labels.json").open("r", encoding="utf-8") as handle:
    CLASS_CONTRACT = json.load(handle)
with (CONTRACTS_DIR / "preprocessing_config.json").open("r", encoding="utf-8") as handle:
    PREPROCESSING_CONTRACT = json.load(handle)


CLASS_ORDER = [item["slug"] for item in CLASS_CONTRACT["classes"]]
SAMPLE_RATE = PREPROCESSING_CONTRACT["audio"]["sample_rate_hz"]
WINDOW_DURATION_MS = PREPROCESSING_CONTRACT["audio"]["window_duration_ms"]
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION_MS / 1000)
TINYCNN_EMBEDDING_DIM = PREPROCESSING_CONTRACT["embedding_contract"]["tiny_cnn_dim"]
YAMNET_EMBEDDING_DIM = PREPROCESSING_CONTRACT["embedding_contract"]["yamnet_dim"]


def cap_examples_per_class(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    split_frame = frame[frame["split"] == split_name].copy()
    max_examples = COMMON_CAPS.get(split_name)
    if max_examples is None:
        return split_frame.reset_index(drop=True)
    return (
        split_frame.sort_values(["class_slug", "clip_id"])
        .groupby("class_slug", group_keys=False)
        .head(max_examples)
        .reset_index(drop=True)
    )


def load_capped_manifest_frame() -> pd.DataFrame:
    manifest = load_dataset_manifest(CURATED_MANIFEST_PATH)
    manifest_df = manifest_to_dataframe(manifest)
    manifest_df = manifest_df[manifest_df["audio_path"].map(lambda value: Path(value).exists())].reset_index(drop=True)
    return pd.concat(
        [cap_examples_per_class(manifest_df, split_name) for split_name in ["train", "validation", "test"]],
        ignore_index=True,
    )


def load_audio_file(audio_path: str, *, pad_or_trim: bool) -> np.ndarray:
    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if sample_rate != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if pad_or_trim:
        if waveform.shape[0] < WINDOW_SAMPLES:
            waveform = np.pad(waveform, (0, WINDOW_SAMPLES - waveform.shape[0]))
        else:
            waveform = waveform[:WINDOW_SAMPLES]
    return waveform.astype(np.float32)


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


def resolve_embedding_tensor(output: object, expected_dim: int) -> np.ndarray:
    candidates: list[np.ndarray] = []
    if isinstance(output, dict):
        candidates.extend(np.array(value) for value in output.values())
    elif isinstance(output, (list, tuple)):
        candidates.extend(np.array(value) for value in output)
    else:
        candidates.append(np.array(output))

    for value in candidates:
        if value.ndim >= 1 and value.shape[-1] == expected_dim:
            return value
    raise ValueError(f"Unable to resolve an embedding tensor with last dim {expected_dim}")


def load_tinycnn_embedding_model() -> Any:
    model = tf.saved_model.load(str(TINYCNN_ARTIFACT_DIR / "saved_model"))
    if hasattr(model, "signatures") and model.signatures:
        return next(iter(model.signatures.values()))
    return model


def extract_tinycnn_embedding(model: Any, audio_path: str) -> np.ndarray:
    waveform = load_audio_file(audio_path, pad_or_trim=True)
    input_tensor = waveform_to_rgb_log_mel(waveform)
    output = model(input_tensor)
    embedding = resolve_embedding_tensor(output, TINYCNN_EMBEDDING_DIM)
    if embedding.ndim == 1:
        return embedding.astype(np.float32)
    return embedding[0].astype(np.float32)


def find_yamnet_artifact() -> Path:
    summary_path = YAMNET_ARTIFACT_DIR / "artifact_summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        artifact_summary = json.load(handle)
    artifact_path = Path(artifact_summary["yamnet_artifact"])
    if not artifact_path.exists():
        raise FileNotFoundError(f"YAMNet artifact does not exist: {artifact_path}")
    return artifact_path


def load_yamnet_model(artifact_path: Path) -> Any:
    if artifact_path.name == "saved_model.pb":
        model_root = artifact_path.parent
        return tf.saved_model.load(str(model_root))
    return tf.keras.models.load_model(str(artifact_path))


def extract_yamnet_embedding(model: Any, audio_path: str) -> np.ndarray:
    waveform = load_audio_file(audio_path, pad_or_trim=False)
    input_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    if hasattr(model, "signatures") and model.signatures:
        signature = next(iter(model.signatures.values()))
        output = signature(input_tensor)
    else:
        output = model(input_tensor)
    embedding_frames = resolve_embedding_tensor(output, YAMNET_EMBEDDING_DIM)
    if embedding_frames.ndim == 1:
        return embedding_frames.astype(np.float32)
    return np.array(embedding_frames).mean(axis=0).astype(np.float32)


def build_prototypes(frame: pd.DataFrame) -> dict[str, list[float]]:
    prototype_store: dict[str, list[float]] = {}
    for class_slug, class_frame in frame.groupby("class_slug"):
        prototype_store[class_slug] = np.stack(class_frame["embedding"].to_list(), axis=0).mean(axis=0).tolist()
    return prototype_store


def prototype_predict(embedding: np.ndarray, prototypes: dict[str, list[float]]) -> tuple[str, float]:
    distances = {
        class_slug: float(np.sum((embedding - np.array(prototype, dtype=np.float32)) ** 2))
        for class_slug, prototype in prototypes.items()
    }
    predicted_label = min(distances, key=distances.get)
    return predicted_label, distances[predicted_label]


def compute_metrics(true_labels: list[str], predicted_labels: list[str]) -> dict[str, Any]:
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(float(report["accuracy"]), 4),
        "macro_precision": round(float(report["macro avg"]["precision"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_precision": round(float(report["weighted avg"]["precision"]), 4),
        "weighted_recall": round(float(report["weighted avg"]["recall"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
    }


def evaluate_backbone(name: str, frame: pd.DataFrame, extractor) -> tuple[pd.DataFrame, dict[str, Any]]:
    embedding_rows = []
    for row in frame.to_dict(orient="records"):
        embedding_rows.append(
            {
                "clip_id": row["clip_id"],
                "class_slug": row["class_slug"],
                "split": row["split"],
                "embedding": extractor(row["audio_path"]),
            }
        )

    embedding_df = pd.DataFrame(embedding_rows)
    prototypes = build_prototypes(embedding_df[embedding_df["split"] == "train"])

    predictions = []
    for row in embedding_df[embedding_df["split"] == "test"].to_dict(orient="records"):
        predicted_label, distance = prototype_predict(row["embedding"], prototypes)
        predictions.append(
            {
                "clip_id": row["clip_id"],
                "true_label": row["class_slug"],
                "predicted_label": predicted_label,
                "distance": distance,
            }
        )

    predictions_df = pd.DataFrame(predictions)
    metrics = compute_metrics(
        predictions_df["true_label"].tolist(),
        predictions_df["predicted_label"].tolist(),
    )
    summary = {
        "model_name": name,
        "protocol": "shared_prototype_eval_same_cap",
        "shared_caps": COMMON_CAPS,
        "test_support": int(len(predictions_df)),
        "aggregate_metrics": metrics,
        "per_class_recall": {
            class_slug: round(
                float(
                    classification_report(
                        predictions_df["true_label"],
                        predictions_df["predicted_label"],
                        labels=CLASS_ORDER,
                        output_dict=True,
                        zero_division=0,
                    )[class_slug]["recall"]
                ),
                4,
            )
            for class_slug in CLASS_ORDER
        },
    }
    predictions_path = EVALUATION_DIR / f"{name.lower()}_same_cap_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    summary["predictions_path"] = str(predictions_path.relative_to(PROJECT_ROOT))
    return predictions_df, summary


def write_markdown_report(tinycnn_summary: dict[str, Any], yamnet_summary: dict[str, Any], output_path: Path) -> None:
    tiny = tinycnn_summary["aggregate_metrics"]
    yam = yamnet_summary["aggregate_metrics"]
    lines = [
        "# Same-Cap Backbone Comparison",
        "",
        "## Protocol",
        "",
        "Both backbones were re-evaluated with prototype classification on the same capped manifest protocol.",
        "",
        f"- Shared caps: train={COMMON_CAPS['train']}, validation={COMMON_CAPS['validation']}, test={COMMON_CAPS['test']}",
        f"- Test support per model: {tinycnn_summary['test_support']}",
        "- Scoring method: nearest-prototype inference using train-split centroids",
        "",
        "## Aggregate comparison",
        "",
        "| Metric | TinyCNN | YAMNet | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Accuracy | {tiny['accuracy']:.4f} | {yam['accuracy']:.4f} | {yam['accuracy'] - tiny['accuracy']:+.4f} |",
        f"| Macro precision | {tiny['macro_precision']:.4f} | {yam['macro_precision']:.4f} | {yam['macro_precision'] - tiny['macro_precision']:+.4f} |",
        f"| Macro recall | {tiny['macro_recall']:.4f} | {yam['macro_recall']:.4f} | {yam['macro_recall'] - tiny['macro_recall']:+.4f} |",
        f"| Macro F1 | {tiny['macro_f1']:.4f} | {yam['macro_f1']:.4f} | {yam['macro_f1'] - tiny['macro_f1']:+.4f} |",
        f"| Weighted precision | {tiny['weighted_precision']:.4f} | {yam['weighted_precision']:.4f} | {yam['weighted_precision'] - tiny['weighted_precision']:+.4f} |",
        f"| Weighted recall | {tiny['weighted_recall']:.4f} | {yam['weighted_recall']:.4f} | {yam['weighted_recall'] - tiny['weighted_recall']:+.4f} |",
        f"| Weighted F1 | {tiny['weighted_f1']:.4f} | {yam['weighted_f1']:.4f} | {yam['weighted_f1'] - tiny['weighted_f1']:+.4f} |",
        "",
        "## Per-class recall",
        "",
        "| Class | TinyCNN | YAMNet | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]

    for class_slug in CLASS_ORDER:
        tiny_recall = tinycnn_summary["per_class_recall"][class_slug]
        yam_recall = yamnet_summary["per_class_recall"][class_slug]
        lines.append(f"| {class_slug} | {tiny_recall:.4f} | {yam_recall:.4f} | {yam_recall - tiny_recall:+.4f} |")

    winner = "YAMNet" if yam["macro_f1"] > tiny["macro_f1"] else "TinyCNN"
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"{winner} remains the stronger backbone under the stricter same-cap prototype-evaluation protocol.",
            "",
            f"- TinyCNN predictions: {tinycnn_summary['predictions_path']}",
            f"- YAMNet predictions: {yamnet_summary['predictions_path']}",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    manifest_frame = load_capped_manifest_frame()

    tinycnn_model = load_tinycnn_embedding_model()
    _, tinycnn_summary = evaluate_backbone(
        "tinycnn",
        manifest_frame,
        lambda audio_path: extract_tinycnn_embedding(tinycnn_model, audio_path),
    )

    yamnet_artifact = find_yamnet_artifact()
    yamnet_model = load_yamnet_model(yamnet_artifact)
    _, yamnet_summary = evaluate_backbone(
        "yamnet",
        manifest_frame,
        lambda audio_path: extract_yamnet_embedding(yamnet_model, audio_path),
    )

    comparison_summary = {
        "protocol": "shared_prototype_eval_same_cap",
        "shared_caps": COMMON_CAPS,
        "tinycnn": tinycnn_summary,
        "yamnet": yamnet_summary,
        "winner_by_macro_f1": "YAMNet"
        if yamnet_summary["aggregate_metrics"]["macro_f1"] > tinycnn_summary["aggregate_metrics"]["macro_f1"]
        else "TinyCNN",
    }

    tinycnn_summary_path = EVALUATION_DIR / "tinycnn_same_cap_summary.json"
    yamnet_summary_path = EVALUATION_DIR / "yamnet_same_cap_summary.json"
    comparison_summary_path = EVALUATION_DIR / "same_cap_model_comparison_summary.json"
    comparison_report_path = EVALUATION_DIR / "same_cap_model_comparison_report.md"

    tinycnn_summary_path.write_text(json.dumps(tinycnn_summary, indent=2), encoding="utf-8")
    yamnet_summary_path.write_text(json.dumps(yamnet_summary, indent=2), encoding="utf-8")
    comparison_summary_path.write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")
    write_markdown_report(tinycnn_summary, yamnet_summary, comparison_report_path)

    print(json.dumps(comparison_summary, indent=2))


if __name__ == "__main__":
    main()