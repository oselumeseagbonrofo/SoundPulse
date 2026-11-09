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
EVALUATION_MANIFEST_PATH = ML_ROOT / "data" / "curated" / "soundpulse_evaluation_manifest.v1.json"
TINYCNN_ARTIFACT_DIR = ML_ROOT / "export" / "artifacts" / "generated" / "tinycnn_baseline"
YAMNET_ARTIFACT_DIR = ML_ROOT / "export" / "artifacts" / "generated" / "yamnet_kaggle"
EVALUATION_DIR = ML_ROOT / "evaluation"
COMMON_CAPS = {"train": 128, "validation": 64, "test": 64}
IMAGE_SIZE = (96, 101)
RANDOM_SEED = 42


with (CONTRACTS_DIR / "class_labels.json").open("r", encoding="utf-8") as handle:
    CLASS_CONTRACT = json.load(handle)
with (CONTRACTS_DIR / "preprocessing_config.json").open("r", encoding="utf-8") as handle:
    PREPROCESSING_CONTRACT = json.load(handle)


CLASS_ORDER = [item["slug"] for item in CLASS_CONTRACT["classes"]]
SAMPLE_RATE = PREPROCESSING_CONTRACT["audio"]["sample_rate_hz"]
WINDOW_DURATION_MS = PREPROCESSING_CONTRACT["audio"]["window_duration_ms"]
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION_MS / 1000)
HOP_SAMPLES = WINDOW_SAMPLES // 2
TINYCNN_EMBEDDING_DIM = PREPROCESSING_CONTRACT["embedding_contract"]["tiny_cnn_dim"]
YAMNET_EMBEDDING_DIM = PREPROCESSING_CONTRACT["embedding_contract"]["yamnet_dim"]


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return (arr / norm).astype(np.float32)


def cap_examples_per_class(frame: pd.DataFrame, split_name: str, seed: int = RANDOM_SEED) -> pd.DataFrame:
    split_frame = frame[frame["split"] == split_name].copy()
    max_examples = COMMON_CAPS.get(split_name)
    if max_examples is None or split_frame.empty:
        return split_frame.reset_index(drop=True)

    sampled_groups: list[pd.DataFrame] = []
    for _, class_frame in split_frame.sort_values(["class_slug", "clip_id"]).groupby("class_slug", sort=False):
        if len(class_frame) > max_examples:
            sampled_groups.append(class_frame.sample(n=max_examples, random_state=seed))
        else:
            sampled_groups.append(class_frame)
    return pd.concat(sampled_groups, ignore_index=True)

def remap_audio_path(relative_path: str, external_root: Path | None) -> Path:
    if external_root is None:
        return (PROJECT_ROOT / relative_path).resolve()
    
    parts = Path(relative_path).parts
    # Strip everything up to and including 'raw' if present:
    # 'ml/data/raw/esc_50_dataset/...' -> 'esc_50_dataset/...'
    if "raw" in parts:
        sub_path = Path(*parts[parts.index("raw") + 1 :])
    else:
        sub_path = Path(relative_path)
        
    return (external_root / sub_path).resolve()

def load_capped_manifest_frame(manifest_path: Path, audio_dir: Path | None = None) -> pd.DataFrame:
    manifest = load_dataset_manifest(manifest_path)
    manifest_df = manifest_to_dataframe(manifest)
    
    if audio_dir is not None:
        manifest_df["audio_path"] = manifest_df["relative_path"].map(
            lambda rel: str(remap_audio_path(rel, audio_dir))
        )
        
    manifest_df = manifest_df[manifest_df["audio_path"].map(lambda value: Path(value).exists())].reset_index(drop=True)
    if manifest_df.empty:
        raise FileNotFoundError(
            f"No existing audio files found on disk for manifest: {manifest_path}. "
            "Check that your external audio path contains the required datasets."
        )
    return pd.concat(
        [cap_examples_per_class(manifest_df, split_name) for split_name in ["train", "test"]],
        ignore_index=True,
    )


def replicate_patch_if_needed(waveform: np.ndarray) -> np.ndarray:
    if waveform.shape[0] == 0:
        return np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    if waveform.shape[0] < WINDOW_SAMPLES:
        repeats = int(np.ceil(WINDOW_SAMPLES / waveform.shape[0]))
        return np.tile(waveform, repeats)[:WINDOW_SAMPLES].astype(np.float32)
    return waveform.astype(np.float32)


def iter_waveform_patches(waveform: np.ndarray) -> list[np.ndarray]:
    if waveform.shape[0] <= WINDOW_SAMPLES:
        return [replicate_patch_if_needed(waveform)]

    patches: list[np.ndarray] = []
    for start in range(0, waveform.shape[0] - WINDOW_SAMPLES + 1, HOP_SAMPLES):
        patches.append(waveform[start : start + WINDOW_SAMPLES].astype(np.float32))

    last_start = (len(patches) - 1) * HOP_SAMPLES
    if waveform.shape[0] - (last_start + WINDOW_SAMPLES) > WINDOW_SAMPLES // 2:
        tail = waveform[last_start + HOP_SAMPLES :]
        patches.append(replicate_patch_if_needed(tail))
    return patches


def load_audio_file(audio_path: str, *, pad_or_trim: bool = False) -> np.ndarray:
    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if sample_rate != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if pad_or_trim:
        waveform = replicate_patch_if_needed(waveform)[:WINDOW_SAMPLES]
    return waveform.astype(np.float32)


def patch_to_log_mel_image(patch: np.ndarray) -> tf.Tensor:
    mel_spec = librosa.feature.melspectrogram(
        y=patch,
        sr=SAMPLE_RATE,
        n_fft=PREPROCESSING_CONTRACT["stft"]["n_fft"],
        hop_length=PREPROCESSING_CONTRACT["stft"]["hop_length"],
        win_length=PREPROCESSING_CONTRACT["stft"]["win_length"],
        n_mels=PREPROCESSING_CONTRACT["mel"]["n_mels"],
        fmin=PREPROCESSING_CONTRACT["mel"]["f_min_hz"],
        fmax=PREPROCESSING_CONTRACT["mel"]["f_max_hz"],
        power=PREPROCESSING_CONTRACT["mel"]["power"],
    )
    log_mel_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
    log_mel_db = np.clip(
        log_mel_db,
        PREPROCESSING_CONTRACT["normalization"]["clip_min_db"],
        PREPROCESSING_CONTRACT["normalization"]["clip_max_db"],
    )
    log_mel = tf.convert_to_tensor(log_mel_db, dtype=tf.float32)
    log_mel = tf.expand_dims(log_mel, axis=-1)
    if tuple(log_mel.shape[:2]) != IMAGE_SIZE:
        log_mel = tf.image.resize(log_mel, IMAGE_SIZE)
    return tf.image.per_image_standardization(log_mel)


def waveform_to_log_mel(waveform: np.ndarray) -> tf.Tensor:
    patches = iter_waveform_patches(np.asarray(waveform, dtype=np.float32))
    return tf.stack([patch_to_log_mel_image(patch) for patch in patches], axis=0)


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
    waveform = load_audio_file(audio_path, pad_or_trim=False)
    input_tensor = waveform_to_log_mel(waveform)
    output = model(input_tensor)
    embedding = resolve_embedding_tensor(output, TINYCNN_EMBEDDING_DIM)
    if embedding.ndim == 1:
        return l2_normalize(embedding)
    patch_embeddings = np.stack([l2_normalize(row) for row in np.asarray(embedding, dtype=np.float32)], axis=0)
    return l2_normalize(patch_embeddings.mean(axis=0))


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
        return l2_normalize(embedding_frames)
    normalized_frames = np.stack(
        [l2_normalize(row) for row in np.asarray(embedding_frames, dtype=np.float32)],
        axis=0,
    )
    return l2_normalize(normalized_frames.mean(axis=0))


def build_prototypes(frame: pd.DataFrame) -> dict[str, list[float]]:
    prototype_store: dict[str, list[float]] = {}
    for class_slug, class_frame in frame.groupby("class_slug"):
        normalized_rows = np.stack([l2_normalize(vec) for vec in class_frame["embedding"].to_list()], axis=0)
        centroid = l2_normalize(normalized_rows.mean(axis=0))
        prototype_store[class_slug] = centroid.tolist()
    return prototype_store


def prototype_predict(embedding: np.ndarray, prototypes: dict[str, list[float]]) -> tuple[str, float]:
    normalized_embedding = l2_normalize(embedding)
    distances = {
        class_slug: float(np.sum((normalized_embedding - l2_normalize(np.array(prototype, dtype=np.float32))) ** 2))
        for class_slug, prototype in prototypes.items()
    }
    predicted_label = min(distances, key=distances.get)
    return predicted_label, distances[predicted_label]


def compute_metrics_and_per_class_recall(
    true_labels: list[str],
    predicted_labels: list[str],
) -> tuple[dict[str, Any], dict[str, float]]:
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": round(float(report["accuracy"]), 4),
        "macro_precision": round(float(report["macro avg"]["precision"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_precision": round(float(report["weighted avg"]["precision"]), 4),
        "weighted_recall": round(float(report["weighted avg"]["recall"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
    }
    per_class_recall = {
        class_slug: round(float(report[class_slug]["recall"]), 4)
        for class_slug in CLASS_ORDER
    }
    return metrics, per_class_recall


def compute_metrics(true_labels: list[str], predicted_labels: list[str]) -> dict[str, Any]:
    metrics, _ = compute_metrics_and_per_class_recall(true_labels, predicted_labels)
    return metrics


def evaluate_backbone(name: str, frame: pd.DataFrame, extractor) -> tuple[pd.DataFrame, dict[str, Any]]:
    eval_frame = frame[frame["split"].isin(["train", "test"])].reset_index(drop=True)
    embedding_rows = []
    for row in eval_frame.to_dict(orient="records"):
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
    metrics, per_class_recall = compute_metrics_and_per_class_recall(
        predictions_df["true_label"].tolist(),
        predictions_df["predicted_label"].tolist(),
    )
    summary = {
        "model_name": name,
        "protocol": "shared_prototype_eval_same_cap",
        "shared_caps": COMMON_CAPS,
        "test_support": int(len(predictions_df)),
        "aggregate_metrics": metrics,
        "per_class_recall": per_class_recall,
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
        "- Scoring method: nearest-prototype inference using L2-normalized train-split centroids",
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EVALUATION_MANIFEST_PATH if EVALUATION_MANIFEST_PATH.exists() else CURATED_MANIFEST_PATH,
        help="Path to manifest JSON used for evaluation.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("G:/My Drive/SoundPulse/data/raw"),  # or Path("D:/your/external/path")
        help="Optional external root directory where raw audio datasets are stored."
    )
    args, _ = parser.parse_known_args()

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    manifest_frame = load_capped_manifest_frame(args.manifest, audio_dir=args.audio_dir)

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
