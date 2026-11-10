from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

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

from soundpulse_ml.backbones import (
    extract_ced_tiny_embedding,
    extract_efficientat_mn10_embedding,
    extract_panns_mobilenetv1_embedding,
    load_ced_tiny_model,
    load_efficientat_mn10_model,
    load_panns_mobilenetv1_model,
)
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

MODEL_DISPLAY_NAMES = {
    "tinycnn": "TinyCNN",
    "yamnet": "YAMNet",
    "efficientat_mn10": "EfficientAT (mn10)",
    "panns_mobilenetv1": "PANNs MobileNetV1",
    "ced_tiny": "CED-Tiny",
}

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
    if "raw" in parts:
        sub_path = Path(*parts[parts.index("raw") + 1 :])
    else:
        sub_path = Path(relative_path)

    return (external_root / sub_path).resolve()


def _build_drivefs_cache_map() -> dict[str, str]:
    drivefs_root = Path.home() / "AppData" / "Local" / "Google" / "DriveFS"
    if not drivefs_root.exists():
        return {}

    filename_to_cache_path: dict[str, str] = {}
    for db_path in drivefs_root.glob("*/metadata_sqlite_db"):
        cache_dir = db_path.parent / "content_cache"
        if not cache_dir.exists():
            continue
        cid_to_file = {p.name: str(p) for p in cache_dir.rglob("*") if p.is_file()}
        if not cid_to_file:
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            conn.text_factory = bytes
            cursor = conn.cursor()
            cursor.execute("SELECT local_title, proto FROM items WHERE is_folder = 0")
            for raw_title, proto in cursor.fetchall():
                if not raw_title or not proto:
                    continue
                title = raw_title.decode("utf-8", errors="ignore")
                if not title.endswith(".wav"):
                    continue
                idx = proto.find(b"content-entry")
                if idx == -1:
                    continue
                pos = idx + 15
                val = 0
                shift = 0
                while pos < len(proto):
                    byte = proto[pos]
                    val |= (byte & 0x7F) << shift
                    pos += 1
                    if not (byte & 0x80):
                        break
                    shift += 7
                cid_str = str(val)
                if cid_str in cid_to_file:
                    filename_to_cache_path[title] = cid_to_file[cid_str]
            conn.close()
        except Exception:
            continue
    return filename_to_cache_path


def load_capped_manifest_frame(manifest_path: Path, audio_dir: Path | None = None) -> pd.DataFrame:
    manifest = load_dataset_manifest(manifest_path)
    manifest_df = manifest_to_dataframe(manifest)

    if audio_dir is not None:
        manifest_df["audio_path"] = manifest_df["relative_path"].map(
            lambda rel: str(remap_audio_path(rel, audio_dir))
        )

    existing_mask = manifest_df["audio_path"].map(lambda value: Path(value).exists())
    if not existing_mask.any():
        cache_map = _build_drivefs_cache_map()
        if cache_map:
            manifest_df["audio_path"] = manifest_df["relative_path"].map(
                lambda rel: cache_map.get(Path(rel).name, str(remap_audio_path(rel, audio_dir)))
            )
            existing_mask = manifest_df["audio_path"].map(lambda value: Path(value).exists())

    manifest_df = manifest_df[existing_mask].reset_index(drop=True)
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


def evaluate_backbone(
    name: str,
    frame: pd.DataFrame,
    extractor: Callable[[str], np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
        "display_name": MODEL_DISPLAY_NAMES.get(name, name),
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


def write_markdown_report(summaries: dict[str, dict[str, Any]], output_path: Path) -> None:
    model_keys = list(summaries.keys())
    headers = [MODEL_DISPLAY_NAMES.get(k, k) for k in model_keys]
    first_summary = summaries[model_keys[0]]

    metric_rows = [
        ("Accuracy", "accuracy"),
        ("Macro precision", "macro_precision"),
        ("Macro recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
        ("Weighted precision", "weighted_precision"),
        ("Weighted recall", "weighted_recall"),
        ("Weighted F1", "weighted_f1"),
    ]

    lines = [
        "# Same-Cap Backbone Comparison",
        "",
        "## Protocol",
        "",
        "All 5 backbones were evaluated with prototype classification on the same capped manifest protocol.",
        "",
        f"- Shared caps: train={COMMON_CAPS['train']}, validation={COMMON_CAPS['validation']}, test={COMMON_CAPS['test']}",
        f"- Test support per model: {first_summary['test_support']}",
        "- Scoring method: nearest-prototype inference using L2-normalized train-split centroids",
        "",
        "## Aggregate comparison",
        "",
        "| Metric | " + " | ".join(headers) + " | Best Model |",
        "| --- | " + " | ".join(["---:"] * len(headers)) + " | --- |",
    ]

    for label, metric_key in metric_rows:
        values = [summaries[k]["aggregate_metrics"][metric_key] for k in model_keys]
        best_val = max(values)
        best_idx = values.index(best_val)
        formatted_vals = [
            f"**{val:.4f}**" if val == best_val else f"{val:.4f}"
            for val in values
        ]
        lines.append(f"| {label} | " + " | ".join(formatted_vals) + f" | {headers[best_idx]} |")

    lines.extend(
        [
            "",
            "## Per-class recall",
            "",
            "| Class | " + " | ".join(headers) + " | Best Model |",
            "| --- | " + " | ".join(["---:"] * len(headers)) + " | --- |",
        ]
    )

    for class_slug in CLASS_ORDER:
        recalls = [summaries[k]["per_class_recall"][class_slug] for k in model_keys]
        best_rec = max(recalls)
        best_models = [headers[i] for i, r in enumerate(recalls) if r == best_rec]
        formatted_recs = [
            f"**{r:.4f}**" if r == best_rec else f"{r:.4f}"
            for r in recalls
        ]
        lines.append(
            f"| {class_slug} | " + " | ".join(formatted_recs) + f" | {', '.join(best_models)} |"
        )

    ranked_keys = sorted(
        model_keys,
        key=lambda k: summaries[k]["aggregate_metrics"]["macro_f1"],
        reverse=True,
    )
    winner_key = ranked_keys[0]
    winner_name = MODEL_DISPLAY_NAMES.get(winner_key, winner_key)
    winner_f1 = summaries[winner_key]["aggregate_metrics"]["macro_f1"]

    lines.extend(
        [
            "",
            "## Leaderboard by Macro F1",
            "",
            "| Rank | Backbone | Embedding Dim | Macro F1 | Accuracy | Macro Recall |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    emb_dims = {
        "tinycnn": TINYCNN_EMBEDDING_DIM,
        "yamnet": YAMNET_EMBEDDING_DIM,
        "efficientat_mn10": 960,
        "panns_mobilenetv1": 1024,
        "ced_tiny": 192,
    }
    for rank, k in enumerate(ranked_keys, start=1):
        m = summaries[k]["aggregate_metrics"]
        disp = MODEL_DISPLAY_NAMES.get(k, k)
        lines.append(
            f"| {rank} | {disp} | {emb_dims.get(k, '-')} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} | {m['macro_recall']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"**{winner_name}** achieves the highest Macro F1 (`{winner_f1:.4f}`) under the same-cap prototype-evaluation protocol.",
            "",
        ]
    )
    for k in model_keys:
        disp = MODEL_DISPLAY_NAMES.get(k, k)
        lines.append(f"- {disp} predictions: `{summaries[k]['predictions_path']}`")

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
        default=Path("G:/My Drive/SoundPulse/data/raw"),
        help="Optional external root directory where raw audio datasets are stored.",
    )
    args, _ = parser.parse_known_args()

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    manifest_frame = load_capped_manifest_frame(args.manifest, audio_dir=args.audio_dir)

    summaries: dict[str, dict[str, Any]] = {}

    print("[1/5] Evaluating TinyCNN...", flush=True)
    tinycnn_model = load_tinycnn_embedding_model()
    _, summaries["tinycnn"] = evaluate_backbone(
        "tinycnn",
        manifest_frame,
        lambda audio_path: extract_tinycnn_embedding(tinycnn_model, audio_path),
    )

    print("[2/5] Evaluating YAMNet...", flush=True)
    yamnet_artifact = find_yamnet_artifact()
    yamnet_model = load_yamnet_model(yamnet_artifact)
    _, summaries["yamnet"] = evaluate_backbone(
        "yamnet",
        manifest_frame,
        lambda audio_path: extract_yamnet_embedding(yamnet_model, audio_path),
    )

    print("[3/5] Evaluating EfficientAT (mn10)...", flush=True)
    efficientat_model = load_efficientat_mn10_model()
    _, summaries["efficientat_mn10"] = evaluate_backbone(
        "efficientat_mn10",
        manifest_frame,
        lambda audio_path: extract_efficientat_mn10_embedding(efficientat_model, audio_path),
    )

    print("[4/5] Evaluating PANNs MobileNetV1...", flush=True)
    panns_model = load_panns_mobilenetv1_model()
    _, summaries["panns_mobilenetv1"] = evaluate_backbone(
        "panns_mobilenetv1",
        manifest_frame,
        lambda audio_path: extract_panns_mobilenetv1_embedding(panns_model, audio_path),
    )

    print("[5/5] Evaluating CED-Tiny...", flush=True)
    ced_model = load_ced_tiny_model()
    _, summaries["ced_tiny"] = evaluate_backbone(
        "ced_tiny",
        manifest_frame,
        lambda audio_path: extract_ced_tiny_embedding(ced_model, audio_path),
    )

    ranked_keys = sorted(
        summaries.keys(),
        key=lambda k: summaries[k]["aggregate_metrics"]["macro_f1"],
        reverse=True,
    )
    winner_key = ranked_keys[0]
    winner_display = MODEL_DISPLAY_NAMES.get(winner_key, winner_key)

    comparison_summary: dict[str, Any] = {
        "protocol": "shared_prototype_eval_same_cap",
        "shared_caps": COMMON_CAPS,
        **summaries,
        "ranking_by_macro_f1": [
            {
                "rank": idx + 1,
                "model_name": k,
                "display_name": MODEL_DISPLAY_NAMES.get(k, k),
                "macro_f1": summaries[k]["aggregate_metrics"]["macro_f1"],
                "accuracy": summaries[k]["aggregate_metrics"]["accuracy"],
            }
            for idx, k in enumerate(ranked_keys)
        ],
        "winner_by_macro_f1": winner_display,
    }

    for model_key, model_summary in summaries.items():
        summary_path = EVALUATION_DIR / f"{model_key}_same_cap_summary.json"
        summary_path.write_text(json.dumps(model_summary, indent=2), encoding="utf-8")

    comparison_summary_path = EVALUATION_DIR / "same_cap_model_comparison_summary.json"
    comparison_report_path = EVALUATION_DIR / "same_cap_model_comparison_report.md"

    comparison_summary_path.write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")
    write_markdown_report(summaries, comparison_report_path)

    print(json.dumps(comparison_summary, indent=2))


if __name__ == "__main__":
    main()
