"""Extract prototype embeddings from a TinyCNN TFLite model.

Usage examples:
python extract_tflite_prototype_embeddings.py \
  --tflite_model path/to/tinycnn.tflite \
  --input_wav_dir path/to/wavs \
    --output_json prototype_store.json

The script supports:
- Loading a TFLite model (tensorflow.lite or tflite_runtime)
- Computing mel spectrograms consistent with repository preprocessing
- Running each 1s patch through the model
- Aggregating a per-file prototype (mean across patches)
- Saving prototypes to a JSON file keyed by label

Requires: numpy, librosa, tqdm, either tensorflow or tflite_runtime
"""
import argparse
import importlib
import json
import os
import glob
import time
import numpy as np
import librosa
from tqdm import tqdm

# Preprocessing defaults matching ml/data/convert_data_to_mel.py
TARGET_SR = 16000
N_FFT = 1024
HOP_LENGTH = 512
WIN_LENGTH = 1024
N_MELS = 96
PATCH_SECONDS = 1.0
MIN_CLIP_DURATION = int(TARGET_SR * PATCH_SECONDS)
HOP_SAMPLES = MIN_CLIP_DURATION // 2


def replicate_patch_if_needed(x, min_clip_duration=MIN_CLIP_DURATION):
    if len(x) < min_clip_duration:
        tile_size = (min_clip_duration // x.shape[0]) + 1
        x = np.tile(x, tile_size)[:min_clip_duration]
    return x


def iter_patches(y, min_clip_duration=MIN_CLIP_DURATION, hop_samples=HOP_SAMPLES):
    if len(y) < min_clip_duration:
        y = replicate_patch_if_needed(y, min_clip_duration)
    for start in range(0, max(len(y) - min_clip_duration + 1, 1), hop_samples):
        patch = y[start:start + min_clip_duration]
        patch = replicate_patch_if_needed(patch, min_clip_duration)
        yield start, patch


def compute_mel_db(patch):
    mel_spec = librosa.feature.melspectrogram(
        y=patch,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db.astype(np.float32)


class TFLiteEmbedder:
    def __init__(self, model_path, output_index=0):
        self.model_path = model_path
        self.output_index = output_index
        Interpreter = self._load_interpreter_class()
        self.interpreter = Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _load_interpreter_class(self):
        candidates = [
            "tflite_runtime.interpreter",
            "tensorflow.lite.python.interpreter",
            "tensorflow.lite",
        ]
        for module_name in candidates:
            try:
                module = importlib.import_module(module_name)
                interpreter = getattr(module, "Interpreter", None)
                if interpreter is not None:
                    return interpreter
            except Exception:
                continue
        raise ImportError(
            "Could not import a TFLite interpreter. Install tflite-runtime or tensorflow."
        )

    def _try_set_input(self, mel):
        # mel: (n_mels, frames)
        n_mels, frames = mel.shape[0], mel.shape[1]

        # Candidate shapes and corresponding transforms
        candidates = [
            ((1, frames, n_mels, 1), lambda m: m.T[None, ..., None]),
            ((1, n_mels, frames, 1), lambda m: m[None, ..., None]),
            ((1, frames, n_mels), lambda m: m.T[None, ...]),
            ((1, n_mels, frames), lambda m: m[None, ...]),
        ]

        input_index = self.input_details[0]['index']
        for shape, transform in candidates:
            try:
                # Resize input if model supports dynamic resize
                try:
                    self.interpreter.resize_tensor_input(input_index, shape)
                    self.interpreter.allocate_tensors()
                except Exception:
                    # If resizing is not supported, skip this candidate
                    pass

                tensor_in = transform(mel).astype(self.input_details[0]['dtype'])
                if tensor_in.size != np.prod(self.interpreter.get_input_details()[0]['shape']):
                    # Shapes don't match; try next
                    continue

                self.interpreter.set_tensor(input_index, tensor_in)
                return True
            except Exception:
                continue

        return False

    def infer(self, mel):
        if not self._try_set_input(mel):
            raise RuntimeError("Failed to map mel spectrogram to model input shape")

        self.interpreter.invoke()

        # Choose the requested output
        out_details = self.interpreter.get_output_details()
        if self.output_index >= len(out_details):
            raise IndexError("Requested output_index is out of range for the model outputs")

        out = self.interpreter.get_tensor(out_details[self.output_index]['index'])
        return out.squeeze()


def infer_label(file_path, root_dir=None):
    if root_dir:
        relative_path = os.path.relpath(file_path, root_dir)
        parts = relative_path.split(os.sep)
        if len(parts) > 1 and parts[0] not in (".", ".."):
            return parts[0]
    return os.path.splitext(os.path.basename(file_path))[0]


def gather_audio_files(input_wav_dir=None, input_npy_dir=None):
    items = []
    if input_wav_dir:
        for file_path in sorted(glob.glob(os.path.join(input_wav_dir, "**", "*.wav"), recursive=True)):
            items.append((file_path, infer_label(file_path, input_wav_dir)))
    if input_npy_dir:
        for file_path in sorted(glob.glob(os.path.join(input_npy_dir, "**", "*.npy"), recursive=True)):
            items.append((file_path, infer_label(file_path, input_npy_dir)))
    return items


def process_files(model_path, files, output_json, aggregate='mean', output_index=0):
    embedder = TFLiteEmbedder(model_path, output_index=output_index)
    results = {}

    if aggregate != 'mean':
        raise NotImplementedError("JSON prototype_store output currently supports mean aggregation only")

    for fpath, label in tqdm(files, desc="Processing files"):
        try:
            if fpath.lower().endswith('.npy'):
                mel = np.load(fpath)
                # Expect mel shape already (n_mels, frames)
                if mel.ndim == 3:
                    # (patches, n_mels, frames) -> iterate patches
                    patch_mels = mel
                else:
                    patch_mels = np.expand_dims(mel, 0)
            else:
                y, sr = librosa.load(fpath, sr=TARGET_SR, mono=True)
                patch_mels = []
                for _, patch in iter_patches(y):
                    patch_mels.append(compute_mel_db(patch))
                patch_mels = np.stack(patch_mels, axis=0)

            embeddings = []
            for mel in patch_mels:
                emb = embedder.infer(mel)
                embeddings.append(np.asarray(emb, dtype=np.float32))
            embeddings = np.stack(embeddings, axis=0)

            proto = np.mean(embeddings, axis=0)
            results[label] = np.asarray(proto, dtype=np.float32).tolist()
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    with open(output_json, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract prototype embeddings from TFLite model")
    parser.add_argument('--tflite_model', required=True)
    parser.add_argument('--input_wav_dir', default=None)
    parser.add_argument('--input_npy_dir', default=None,
                        help='If you already produced mel .npy files, point here')
    parser.add_argument('--output_json', default='prototype_store.json')
    parser.add_argument('--aggregate', choices=['mean'], default='mean',
                        help='How to aggregate per-file patch embeddings into a prototype')
    parser.add_argument('--output_index', type=int, default=0,
                        help='Index of the model output to use as embedding')
    args = parser.parse_args()

    files = gather_audio_files(args.input_wav_dir, args.input_npy_dir)
    if not files:
        raise SystemExit('No input files found. Provide --input_wav_dir or --input_npy_dir')

    t0 = time.time()
    results = process_files(args.tflite_model, files, args.output_json,
                            aggregate=args.aggregate, output_index=args.output_index)
    dt = time.time() - t0
    print(f"Saved {len(results)} prototypes to {args.output_json} in {dt:.1f}s")


if __name__ == '__main__':
    main()
