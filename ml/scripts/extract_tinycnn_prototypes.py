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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "ml"
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soundpulse_ml.datasets import load_dataset_manifest, manifest_to_dataframe

CONTRACTS_DIR = PROJECT_ROOT / "packages" / "shared_contracts" / "v1"
CURATED_MANIFEST_PATH = ML_ROOT / "data" / "curated" / "soundpulse_manifest.v1.json"
TINYCNN_ARTIFACT_DIR = ML_ROOT / "export" / "artifacts" / "generated" / "mobilenetv4_convsmall_10"
IMAGE_SIZE = (96, 101)

# Load contracts
with (CONTRACTS_DIR / "class_labels.json").open("r", encoding="utf-8") as handle:
    CLASS_CONTRACT = json.load(handle)
with (CONTRACTS_DIR / "preprocessing_config.json").open("r", encoding="utf-8") as handle:
    PREPROCESSING_CONTRACT = json.load(handle)

CLASS_ORDER = [item["slug"] for item in CLASS_CONTRACT["classes"]]
SAMPLE_RATE = PREPROCESSING_CONTRACT["audio"]["sample_rate_hz"]
WINDOW_DURATION_MS = PREPROCESSING_CONTRACT["audio"]["window_duration_ms"]
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION_MS / 1000)
TINYCNN_EMBEDDING_DIM = PREPROCESSING_CONTRACT["embedding_contract"]["tiny_cnn_dim"]


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
    log_mel = tf.image.per_image_standardization(log_mel)
    return tf.expand_dims(log_mel, axis=0)


def main() -> None:
    saved_model_path = TINYCNN_ARTIFACT_DIR / "saved_model"
    output_path = TINYCNN_ARTIFACT_DIR / "prototype_store.json"

    print(f"Loading SavedModel from {saved_model_path}...")
    model = tf.saved_model.load(str(saved_model_path))
    if hasattr(model, "signatures") and model.signatures:
        serving_fn = next(iter(model.signatures.values()))
    else:
        serving_fn = model

    print(f"Loading manifest from {CURATED_MANIFEST_PATH}...")
    manifest = load_dataset_manifest(CURATED_MANIFEST_PATH)
    manifest_df = manifest_to_dataframe(manifest)

    # Filter to only get the train split
    train_df = manifest_df[manifest_df["split"] == "train"].copy()
    print(f"Found {len(train_df)} train clips in manifest.")

    embeddings_by_class: dict[str, list[np.ndarray]] = {class_slug: [] for class_slug in CLASS_ORDER}

    for row in tqdm(train_df.to_dict(orient="records"), desc="Extracting embeddings"):
        audio_path = row["audio_path"]
        class_slug = row["class_slug"]
        if not Path(audio_path).exists():
            continue

        try:
            waveform = load_audio_file(audio_path, pad_or_trim=True)
            input_tensor = waveform_to_rgb_log_mel(waveform)

            # Run inference
            output = serving_fn(input_tensor)

            # Resolve embedding tensor (last dimension is TINYCNN_EMBEDDING_DIM)
            if isinstance(output, dict):
                output_tensor = list(output.values())[0]
            elif isinstance(output, (list, tuple)):
                output_tensor = output[0]
            else:
                output_tensor = output
            embedding = np.array(output_tensor)

            # Save the 1024-dimensional embedding
            if embedding.ndim == 1:
                embeddings_by_class[class_slug].append(embedding.astype(np.float32))
            else:
                embeddings_by_class[class_slug].append(embedding[0].astype(np.float32))
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    # Compute prototypes by averaging the embeddings per class
    prototype_store = {}
    for class_slug, embs in embeddings_by_class.items():
        if not embs:
            print(f"Warning: No embeddings found for class {class_slug}")
            continue
        prototype_store[class_slug] = np.mean(np.stack(embs, axis=0), axis=0).tolist()

    # Write prototype store
    print(f"Writing prototype store to {output_path}...")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(prototype_store, handle, indent=2)

    print(f"Successfully generated prototypes for {len(prototype_store)} classes!")


if __name__ == "__main__":
    main()
