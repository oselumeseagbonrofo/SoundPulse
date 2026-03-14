from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from .contracts import CONTRACTS_DIR


@dataclass(frozen=True)
class AudioConfig:
    sample_rate_hz: int
    channels: int
    window_duration_ms: int
    hop_duration_ms: int
    pcm_dtype: str


@dataclass(frozen=True)
class StftConfig:
    n_fft: int
    hop_length: int
    win_length: int
    window: str
    center: bool


@dataclass(frozen=True)
class MelConfig:
    n_mels: int
    f_min_hz: float
    f_max_hz: float
    power: float
    log_offset: float


@dataclass(frozen=True)
class NormalizationConfig:
    normalization_type: str
    clip_min_db: float
    clip_max_db: float


@dataclass(frozen=True)
class PreprocessingConfig:
    version: str
    audio: AudioConfig
    stft: StftConfig
    mel: MelConfig
    normalization: NormalizationConfig

    @property
    def window_samples(self) -> int:
        return int(self.audio.sample_rate_hz * self.audio.window_duration_ms / 1000)

    @property
    def hop_samples(self) -> int:
        return int(self.audio.sample_rate_hz * self.audio.hop_duration_ms / 1000)

    @property
    def expected_time_frames(self) -> int:
        if self.stft.center:
            raise ValueError("center=True is unsupported for fixed-shape SoundPulse preprocessing")
        if self.window_samples < self.stft.n_fft:
            raise ValueError(
                f"window_samples ({self.window_samples}) must be >= n_fft ({self.stft.n_fft})"
            )
        return 1 + (self.window_samples - self.stft.n_fft) // self.stft.hop_length

    @property
    def expected_feature_shape(self) -> tuple[int, int]:
        return (self.mel.n_mels, self.expected_time_frames)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_preprocessing_config(config_path: Path | None = None) -> PreprocessingConfig:
    path = config_path or (CONTRACTS_DIR / "preprocessing_config.json")
    payload = _load_json(path)

    config = PreprocessingConfig(
        version=str(payload["version"]),
        audio=AudioConfig(
            sample_rate_hz=int(payload["audio"]["sample_rate_hz"]),
            channels=int(payload["audio"]["channels"]),
            window_duration_ms=int(payload["audio"]["window_duration_ms"]),
            hop_duration_ms=int(payload["audio"]["hop_duration_ms"]),
            pcm_dtype=str(payload["audio"]["pcm_dtype"]),
        ),
        stft=StftConfig(
            n_fft=int(payload["stft"]["n_fft"]),
            hop_length=int(payload["stft"]["hop_length"]),
            win_length=int(payload["stft"]["win_length"]),
            window=str(payload["stft"]["window"]),
            center=bool(payload["stft"]["center"]),
        ),
        mel=MelConfig(
            n_mels=int(payload["mel"]["n_mels"]),
            f_min_hz=float(payload["mel"]["f_min_hz"]),
            f_max_hz=float(payload["mel"]["f_max_hz"]),
            power=float(payload["mel"]["power"]),
            log_offset=float(payload["mel"]["log_offset"]),
        ),
        normalization=NormalizationConfig(
            normalization_type=str(payload["normalization"]["type"]),
            clip_min_db=float(payload["normalization"]["clip_min_db"]),
            clip_max_db=float(payload["normalization"]["clip_max_db"]),
        ),
    )
    validate_preprocessing_config(config)
    return config


def validate_preprocessing_config(config: PreprocessingConfig) -> None:
    if config.audio.channels != 1:
        raise ValueError(f"Expected mono audio (channels=1), found {config.audio.channels}")
    if config.audio.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if config.audio.window_duration_ms <= 0 or config.audio.hop_duration_ms <= 0:
        raise ValueError("window_duration_ms and hop_duration_ms must be positive")
    if config.audio.pcm_dtype != "float32":
        raise ValueError(f"Expected pcm_dtype='float32', found {config.audio.pcm_dtype}")

    if config.stft.window.lower() != "hann":
        raise ValueError("Only hann STFT window is supported")
    if config.stft.center:
        raise ValueError("center=True is unsupported for deterministic fixed-shape features")
    if config.stft.hop_length <= 0 or config.stft.win_length <= 0 or config.stft.n_fft <= 0:
        raise ValueError("STFT n_fft, hop_length, and win_length must be positive")
    if config.stft.win_length > config.stft.n_fft:
        raise ValueError("STFT win_length must be <= n_fft")

    if config.mel.n_mels <= 0:
        raise ValueError("mel.n_mels must be positive")
    if config.mel.f_min_hz < 0 or config.mel.f_max_hz <= config.mel.f_min_hz:
        raise ValueError("Mel bounds are invalid")
    if config.mel.log_offset <= 0:
        raise ValueError("mel.log_offset must be > 0")

    if config.normalization.clip_min_db > config.normalization.clip_max_db:
        raise ValueError("clip_min_db must be <= clip_max_db")

    _ = config.expected_feature_shape


def load_waveform(audio_path: Path, config: PreprocessingConfig) -> tuple[np.ndarray, int]:
    waveform, sample_rate_hz = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    return waveform.astype(np.float32), int(sample_rate_hz)


def resample_waveform(
    waveform: np.ndarray,
    source_sample_rate_hz: int,
    target_sample_rate_hz: int,
) -> np.ndarray:
    if source_sample_rate_hz == target_sample_rate_hz:
        return waveform.astype(np.float32)
    resampled = librosa.resample(
        waveform,
        orig_sr=source_sample_rate_hz,
        target_sr=target_sample_rate_hz,
        res_type="soxr_hq",
    )
    return resampled.astype(np.float32)


def fixed_window_waveform(
    waveform: np.ndarray,
    config: PreprocessingConfig,
    include_partial_last_window: bool = True,
) -> np.ndarray:
    window_samples = config.window_samples
    hop_samples = config.hop_samples

    if waveform.size == 0:
        return np.zeros((1, window_samples), dtype=np.float32)

    if waveform.shape[0] < window_samples:
        padded = np.pad(waveform, (0, window_samples - waveform.shape[0]))
        return np.expand_dims(padded.astype(np.float32), axis=0)

    max_start = waveform.shape[0] - window_samples
    starts = list(range(0, max_start + 1, hop_samples))

    if include_partial_last_window:
        tail_start = max(waveform.shape[0] - window_samples, 0)
        if not starts or starts[-1] != tail_start:
            starts.append(tail_start)

    windows: list[np.ndarray] = []
    for start in starts:
        segment = waveform[start : start + window_samples]
        if segment.shape[0] < window_samples:
            segment = np.pad(segment, (0, window_samples - segment.shape[0]))
        windows.append(segment.astype(np.float32))

    if not windows:
        windows = [np.zeros((window_samples,), dtype=np.float32)]

    return np.stack(windows, axis=0)


def waveform_window_to_log_mel(
    waveform_window: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    if waveform_window.ndim != 1:
        raise ValueError(f"Expected 1D waveform window, found shape {waveform_window.shape}")
    if waveform_window.shape[0] != config.window_samples:
        raise ValueError(
            f"Expected exactly {config.window_samples} samples per window, found {waveform_window.shape[0]}"
        )

    stft_matrix = librosa.stft(
        y=waveform_window.astype(np.float32),
        n_fft=config.stft.n_fft,
        hop_length=config.stft.hop_length,
        win_length=config.stft.win_length,
        window=config.stft.window,
        center=config.stft.center,
    )
    power_spectrogram = np.abs(stft_matrix) ** config.mel.power

    mel_filter = librosa.filters.mel(
        sr=config.audio.sample_rate_hz,
        n_fft=config.stft.n_fft,
        n_mels=config.mel.n_mels,
        fmin=config.mel.f_min_hz,
        fmax=config.mel.f_max_hz,
    ).astype(np.float32)

    mel_spectrogram = mel_filter @ power_spectrogram
    log_mel = np.log(mel_spectrogram + config.mel.log_offset)
    log_mel = np.clip(
        log_mel,
        config.normalization.clip_min_db,
        config.normalization.clip_max_db,
    ).astype(np.float32)

    validate_feature_shape(log_mel, config)
    return log_mel


def validate_feature_shape(feature: np.ndarray, config: PreprocessingConfig) -> None:
    expected_shape = config.expected_feature_shape
    if feature.shape != expected_shape:
        raise ValueError(
            f"Invalid feature shape {feature.shape}; expected {expected_shape}. "
            "Preprocessing contract drift detected."
        )


def preprocess_audio_path(
    audio_path: Path,
    config: PreprocessingConfig,
    include_partial_last_window: bool = True,
) -> np.ndarray:
    waveform, source_sr = load_waveform(audio_path, config)
    waveform = resample_waveform(waveform, source_sr, config.audio.sample_rate_hz)
    windows = fixed_window_waveform(
        waveform,
        config=config,
        include_partial_last_window=include_partial_last_window,
    )
    features = np.stack(
        [waveform_window_to_log_mel(window, config) for window in windows],
        axis=0,
    ).astype(np.float32)

    if features.ndim != 3:
        raise ValueError(f"Expected features to have 3 dimensions, found shape {features.shape}")
    if features.shape[1:] != config.expected_feature_shape:
        raise ValueError(
            f"Invalid feature tensor shape {features.shape}; "
            f"expected (_, {config.expected_feature_shape[0]}, {config.expected_feature_shape[1]})"
        )

    return features


def hash_feature_tensor(feature_tensor: np.ndarray) -> str:
    digest = hashlib.sha256(np.ascontiguousarray(feature_tensor).tobytes()).hexdigest()
    return digest


def summarize_preprocessing_config(config: PreprocessingConfig) -> str:
    return (
        f"preprocessing_version={config.version}\n"
        f"sample_rate_hz={config.audio.sample_rate_hz}\n"
        f"window_samples={config.window_samples}\n"
        f"hop_samples={config.hop_samples}\n"
        f"stft_n_fft={config.stft.n_fft}\n"
        f"stft_hop_length={config.stft.hop_length}\n"
        f"n_mels={config.mel.n_mels}\n"
        f"expected_feature_shape={config.expected_feature_shape}"
    )
