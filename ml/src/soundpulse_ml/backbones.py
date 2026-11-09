from __future__ import annotations

import collections.abc
import math
import sys
import urllib.request
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import librosa
import numpy as np
import soundfile as sf

# Ensure PyTorch is importable from local shared environment if not installed in .venv
_SHARED_SITE_PACKAGES = Path(
    r"C:\Users\ZBOOK STUDIO G5\Documents\GitHub\waxal-asr-challenge\.venv\Lib\site-packages"
)
if _SHARED_SITE_PACKAGES.exists() and str(_SHARED_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(_SHARED_SITE_PACKAGES))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from safetensors.torch import load_file
from torchvision.ops.misc import Conv2dNormActivation


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRETRAINED_DIR = PROJECT_ROOT / "ml" / "export" / "artifacts" / "generated" / "pretrained_backbones"

CHECKPOINT_URLS = {
    "efficientat_mn10": (
        "mn10_as_mAP_471.pt",
        "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt",
    ),
    "panns_mobilenetv1": (
        "MobileNetV1_mAP=0.389.pth",
        "https://zenodo.org/api/records/3987831/files/MobileNetV1_mAP=0.389.pth/content",
    ),
    "ced_tiny": (
        "ced_tiny_model.safetensors",
        "https://huggingface.co/mispeech/ced-tiny/resolve/main/model.safetensors",
    ),
}

EFFICIENTAT_MN10_EMBEDDING_DIM = 960
PANNS_MOBILENETV1_EMBEDDING_DIM = 1024
CED_TINY_EMBEDDING_DIM = 192


def ensure_checkpoint(backbone_key: str, checkpoint_dir: Path = PRETRAINED_DIR) -> Path:
    filename, url = CHECKPOINT_URLS[backbone_key]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_path = checkpoint_dir / filename
    if not target_path.exists():
        urllib.request.urlretrieve(url, str(target_path))
    return target_path


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return (arr / norm).astype(np.float32)


def _load_waveform(audio_path: str, target_sr: int, min_samples: int) -> np.ndarray:
    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
    if waveform.shape[0] == 0:
        return np.zeros(min_samples, dtype=np.float32)
    if waveform.shape[0] < min_samples:
        repeats = int(np.ceil(min_samples / waveform.shape[0]))
        waveform = np.tile(waveform, repeats)[:min_samples]
    return waveform.astype(np.float32)


# ============================================================================
# 1. EfficientAT (mn10_as)
# ============================================================================


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _cnn_out_size(in_size: int, padding: int, dilation: int, kernel: int, stride: int) -> int:
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


class _SqueezeExcitation(nn.Module):
    def __init__(self, input_dim: int, squeeze_dim: int, se_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, squeeze_dim)
        self.fc2 = nn.Linear(squeeze_dim, input_dim)
        self.se_dim = [1, 2, 3]
        self.se_dim.remove(se_dim)
        self.activation = nn.ReLU()
        self.scale_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.mean(x, self.se_dim, keepdim=True)
        shape = scale.size()
        scale = self.fc1(scale.squeeze(2).squeeze(2))
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale).view(shape) * x


class _ConcurrentSEBlock(nn.Module):
    def __init__(self, c_dim: int, f_dim: int, t_dim: int, se_cnf: dict[str, Any]) -> None:
        super().__init__()
        dims = [c_dim, f_dim, t_dim]
        self.conc_se_layers = nn.ModuleList()
        for d in se_cnf["se_dims"]:
            input_dim = dims[d - 1]
            squeeze_dim = _make_divisible(input_dim // se_cnf["se_r"], 8)
            self.conc_se_layers.append(_SqueezeExcitation(input_dim, squeeze_dim, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        se_outs = [layer(x) for layer in self.conc_se_layers]
        return torch.max(torch.stack(se_outs, dim=0), dim=0)[0]


class _InvertedResidualConfig:
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float = 1.0,
    ) -> None:
        self.input_channels = _make_divisible(input_channels * width_mult, 8)
        self.kernel = kernel
        self.expanded_channels = _make_divisible(expanded_channels * width_mult, 8)
        self.out_channels = _make_divisible(out_channels * width_mult, 8)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation
        self.f_dim: Optional[int] = None
        self.t_dim: Optional[int] = None

    def out_size(self, in_size: int) -> int:
        padding = (self.kernel - 1) // 2 * self.dilation
        return _cnn_out_size(in_size, padding, self.dilation, self.kernel, self.stride)


class _EfficientATInvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: _InvertedResidualConfig,
        se_cnf: dict[str, Any],
        norm_layer: Callable[..., nn.Module],
        depthwise_norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        layers: list[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=depthwise_norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se and se_cnf["se_dims"] is not None:
            layers.append(_ConcurrentSEBlock(cnf.expanded_channels, cnf.f_dim or 1, cnf.t_dim or 1, se_cnf))

        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )
        self.block = nn.Sequential(*layers)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        result = self.block(inp)
        if self.use_res_connect:
            result = result + inp
        return result


class EfficientATMN10(nn.Module):
    def __init__(self, width_mult: float = 1.0, num_classes: int = 527) -> None:
        super().__init__()
        bneck_conf = partial(_InvertedResidualConfig, width_mult=width_mult)
        strides = (2, 2, 2, 2)
        inverted_residual_setting: Sequence[_InvertedResidualConfig] = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", strides[0], 1),
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", strides[1], 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", strides[2], 1),
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160, True, "HS", strides[3], 1),
            bneck_conf(160, 5, 960, 160, True, "HS", 1, 1),
            bneck_conf(160, 5, 960, 160, True, "HS", 1, 1),
        ]
        last_channel = _make_divisible(1280 * width_mult, 8)
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        se_cnf = {"se_dims": [1], "se_agg": "max", "se_r": 4}

        layers: list[nn.Module] = [
            Conv2dNormActivation(
                1,
                inverted_residual_setting[0].input_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        ]

        f_dim = _cnn_out_size(128, 1, 1, 3, 2)
        t_dim = _cnn_out_size(1000, 1, 1, 3, 2)
        for cnf in inverted_residual_setting:
            f_dim = cnf.out_size(f_dim)
            t_dim = cnf.out_size(t_dim)
            cnf.f_dim, cnf.t_dim = f_dim, t_dim
            layers.append(_EfficientATInvertedResidual(cnf, se_cnf, norm_layer, norm_layer))

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.features:
            x = layer(x)
        features = F.adaptive_avg_pool2d(x, (1, 1)).flatten(start_dim=1)
        logits = self.classifier(x)
        return logits, features


class AugmentMelSTFT(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        sr: int = 32000,
        win_length: int = 800,
        hopsize: int = 320,
        n_fft: int = 1024,
        fmin: float = 0.0,
        fmax: float = 15000.0,
    ) -> None:
        super().__init__()
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.hopsize = hopsize
        self.register_buffer("window", torch.hann_window(win_length, periodic=False), persistent=False)
        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-0.97, 1.0]]]), persistent=False)
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels,
            self.n_fft,
            self.sr,
            self.fmin,
            self.fmax,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_basis = F.pad(mel_basis, (0, 1), mode="constant", value=0.0)
        self.register_buffer("mel_basis", mel_basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        stft_complex = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hopsize,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=self.window,
            return_complex=True,
        )
        power_mag = torch.view_as_real(stft_complex).pow(2).sum(dim=-1)
        melspec = torch.matmul(self.mel_basis, power_mag)
        melspec = (melspec + 1e-5).log()
        return (melspec + 4.5) / 5.0


def load_efficientat_mn10_model(checkpoint_path: Optional[Path] = None) -> tuple[AugmentMelSTFT, EfficientATMN10]:
    ckpt_path = checkpoint_path or ensure_checkpoint("efficientat_mn10")
    mel_extractor = AugmentMelSTFT()
    mel_extractor.eval()
    model = EfficientATMN10(width_mult=1.0, num_classes=527)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return mel_extractor, model


def extract_efficientat_mn10_embedding(
    model_bundle: tuple[AugmentMelSTFT, EfficientATMN10],
    audio_path: str,
) -> np.ndarray:
    mel_extractor, model = model_bundle
    waveform = _load_waveform(audio_path, target_sr=32000, min_samples=32000)
    with torch.inference_mode():
        wave_tensor = torch.from_numpy(waveform).unsqueeze(0)
        spec = mel_extractor(wave_tensor)  # [1, 128, T]
        _, features = model(spec.unsqueeze(1))  # [1, 960]
        embedding = features.squeeze(0).cpu().numpy()
    return _l2_normalize(embedding)


# ============================================================================
# 2. PANNs MobileNetV1
# ============================================================================


class _PANNsSTFT(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 320, center: bool = True, pad_mode: str = "reflect") -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        out_channels = n_fft // 2 + 1
        self.conv_real = nn.Conv1d(1, out_channels, kernel_size=n_fft, stride=hop_length, padding=0, bias=False)
        self.conv_imag = nn.Conv1d(1, out_channels, kernel_size=n_fft, stride=hop_length, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x[:, None, :]
        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)
        real = self.conv_real(x)[:, None, :, :].transpose(2, 3)
        imag = self.conv_imag(x)[:, None, :, :].transpose(2, 3)
        return real, imag


class _PANNsSpectrogram(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 320) -> None:
        super().__init__()
        self.stft = _PANNsSTFT(n_fft=n_fft, hop_length=hop_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = self.stft(x)
        return real**2 + imag**2


class _PANNsLogmelFilterBank(nn.Module):
    def __init__(self, n_fft: int = 1024, n_mels: int = 64, ref: float = 1.0, amin: float = 1e-10) -> None:
        super().__init__()
        self.ref = ref
        self.amin = amin
        self.melW = nn.Parameter(torch.zeros(n_fft // 2 + 1, n_mels), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel_spectrogram = torch.matmul(x, self.melW)
        log_spec = 10.0 * torch.log10(torch.clamp(mel_spectrogram, min=self.amin))
        log_spec -= 10.0 * math.log10(max(self.amin, self.ref))
        return log_spec


class PANNsMobileNetV1(nn.Module):
    def __init__(self, classes_num: int = 527) -> None:
        super().__init__()
        self.spectrogram_extractor = _PANNsSpectrogram(n_fft=1024, hop_length=320)
        self.logmel_extractor = _PANNsLogmelFilterBank(n_fft=1024, n_mels=64)
        self.bn0 = nn.BatchNorm2d(64)

        def conv_bn(inp: int, oup: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp: int, oup: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn(1, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.features(x)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.relu_(self.fc1(x))
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        return {"clipwise_output": clipwise_output, "embedding": x}


def load_panns_mobilenetv1_model(checkpoint_path: Optional[Path] = None) -> PANNsMobileNetV1:
    ckpt_path = checkpoint_path or ensure_checkpoint("panns_mobilenetv1")
    model = PANNsMobileNetV1(classes_num=527)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def extract_panns_mobilenetv1_embedding(model: PANNsMobileNetV1, audio_path: str) -> np.ndarray:
    waveform = _load_waveform(audio_path, target_sr=32000, min_samples=32000)
    with torch.inference_mode():
        wave_tensor = torch.from_numpy(waveform).unsqueeze(0)
        out = model(wave_tensor)
        embedding = out["embedding"].squeeze(0).cpu().numpy()
    return _l2_normalize(embedding)


# ============================================================================
# 3. CED-Tiny (mispeech/ced-tiny)
# ============================================================================


def _to_2tuple(x: Any) -> tuple[Any, Any]:
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)  # type: ignore[return-value]
    return (x, x)


class _CedAudioPatchEmbed(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int] = (64, 1012),
        patch_size: int = 16,
        patch_stride: int = 16,
        in_chans: int = 1,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.input_size = _to_2tuple(input_size)
        self.patch_size = _to_2tuple(patch_size)
        self.patch_stride = _to_2tuple(patch_stride)
        self.grid_size = (
            self.input_size[0] // self.patch_stride[0],
            self.input_size[1] // self.patch_stride[1],
        )
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_stride)
        self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class _CedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 3, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class _CedMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _CedBlock(nn.Module):
    def __init__(self, dim: int = 192, num_heads: int = 3, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.attn = _CedAttention(dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = norm_layer(dim)
        self.mlp = _CedMlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _CedEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 64,
        target_length: int = 1012,
        embed_dim: int = 192,
        patch_size: int = 16,
        patch_stride: int = 16,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        pad_last: bool = True,
    ) -> None:
        super().__init__()
        self.maximal_allowed_length = target_length
        self.pad_last = pad_last
        self.init_bn = nn.BatchNorm2d(n_mels, momentum=0.01)
        self.patch_embed = _CedAudioPatchEmbed(
            input_size=(n_mels, target_length),
            embed_dim=embed_dim,
            patch_size=patch_size,
            patch_stride=patch_stride,
        )
        self.time_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 1, self.patch_embed.grid_size[1]))
        self.freq_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_embed.grid_size[0], 1))
        self.blocks = nn.Sequential(*[_CedBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        _, _, _, t = x.shape
        x = x + self.time_pos_embed[:, :, :, :t]
        x = x + self.freq_pos_embed[:, :, :, :]
        x = torch.permute(torch.flatten(x, 2, 3), (0, 2, 1))
        x = self.blocks(x)
        return self.norm(x)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        x = torch.unsqueeze(input_values, 1)
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.init_bn(x)
        x = torch.permute(x, (0, 2, 1, 3))

        if x.shape[-1] > self.maximal_allowed_length:
            splits = x.split(self.maximal_allowed_length, -1)
            if splits[-1].shape[-1] < self.maximal_allowed_length:
                if self.pad_last:
                    pad = torch.zeros(*x.shape[:-1], self.maximal_allowed_length, device=x.device)
                    pad[..., : splits[-1].shape[-1]] = splits[-1]
                    splits = torch.stack((*splits[:-1], pad), dim=0)
                else:
                    splits = torch.stack(splits[:-1], dim=0)
            else:
                splits = torch.stack(splits[:-1], dim=0)
            n_splits = len(splits)
            x = torch.flatten(splits, 0, 1)
        else:
            n_splits = 1

        x = self.forward_features(x)
        return torch.reshape(x, (x.shape[0] // n_splits, -1, x.shape[-1]))


class CedTinyForAudioClassification(nn.Module):
    def __init__(self, embed_dim: int = 192, outputdim: int = 527) -> None:
        super().__init__()
        self.encoder = _CedEncoder(embed_dim=embed_dim)
        self.outputlayer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, outputdim),
        )
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=512,
            hop_length=160,
            f_min=0,
            f_max=8000,
            n_mels=64,
            center=True,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=120)

    def extract_features(self, waveform_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_values = self.amplitude_to_db(self.mel_spectrogram(waveform_batch))
        hidden_states = self.encoder(input_values)  # [B, N_tokens, 192]
        pooled = hidden_states.mean(dim=1)  # [B, 192]
        logits = self.outputlayer(pooled).sigmoid()
        return logits, pooled


def load_ced_tiny_model(checkpoint_path: Optional[Path] = None) -> CedTinyForAudioClassification:
    ckpt_path = checkpoint_path or ensure_checkpoint("ced_tiny")
    model = CedTinyForAudioClassification()
    state_dict = load_file(str(ckpt_path))
    model.load_state_dict(state_dict, strict=False)
    # Verify all encoder and outputlayer parameters matched
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    non_transform_missing = [k for k in missing if not k.startswith("mel_spectrogram.")]
    if non_transform_missing or unexpected:
        raise RuntimeError(f"CED-Tiny state_dict mismatch: missing={non_transform_missing}, unexpected={unexpected}")
    model.eval()
    return model


def extract_ced_tiny_embedding(model: CedTinyForAudioClassification, audio_path: str) -> np.ndarray:
    waveform = _load_waveform(audio_path, target_sr=16000, min_samples=16000)
    with torch.inference_mode():
        wave_tensor = torch.from_numpy(waveform).unsqueeze(0)
        _, pooled = model.extract_features(wave_tensor)
        embedding = pooled.squeeze(0).cpu().numpy()
    return _l2_normalize(embedding)
