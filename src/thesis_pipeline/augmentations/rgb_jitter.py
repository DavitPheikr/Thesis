"""RGB brightness/contrast jitter for point-feature batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RGBJitterConfig:
    """Configuration for train-only RGB jitter.

    The expected model feature layout is:

    ``[x, y, z, intensity, r, g, b, rgb_valid]``.
    """

    enabled: bool = False
    train_only: bool = True
    brightness_min: float = 1.0
    brightness_max: float = 1.0
    contrast_min: float = 1.0
    contrast_max: float = 1.0
    rgb_valid_threshold: float = 0.5
    rgb_start_idx: int = 4
    rgb_end_idx: int = 7
    rgb_valid_idx: int = 7

    @classmethod
    def from_mapping(cls, cfg: Any) -> "RGBJitterConfig":
        if not cfg:
            return cls(enabled=False)
        getter = cfg.get if hasattr(cfg, "get") else lambda key, default=None: default
        out = cls(
            enabled=bool(getter("enabled", False)),
            train_only=bool(getter("train_only", True)),
            brightness_min=float(getter("brightness_min", 1.0)),
            brightness_max=float(getter("brightness_max", 1.0)),
            contrast_min=float(getter("contrast_min", 1.0)),
            contrast_max=float(getter("contrast_max", 1.0)),
            rgb_valid_threshold=float(getter("rgb_valid_threshold", 0.5)),
            rgb_start_idx=int(getter("rgb_start_idx", 4)),
            rgb_end_idx=int(getter("rgb_end_idx", 7)),
            rgb_valid_idx=int(getter("rgb_valid_idx", 7)),
        )
        out.validate()
        return out

    def validate(self) -> None:
        if self.brightness_min <= 0 or self.brightness_max <= 0:
            raise ValueError("rgb_jitter brightness factors must be positive")
        if self.contrast_min <= 0 or self.contrast_max <= 0:
            raise ValueError("rgb_jitter contrast factors must be positive")
        if self.brightness_min > self.brightness_max:
            raise ValueError("rgb_jitter brightness_min must be <= brightness_max")
        if self.contrast_min > self.contrast_max:
            raise ValueError("rgb_jitter contrast_min must be <= contrast_max")
        if self.rgb_start_idx < 0 or self.rgb_end_idx <= self.rgb_start_idx:
            raise ValueError("rgb_jitter RGB indices are invalid")
        if self.rgb_end_idx - self.rgb_start_idx != 3:
            raise ValueError("rgb_jitter expects exactly three RGB columns")
        if self.rgb_valid_idx < self.rgb_end_idx:
            raise ValueError("rgb_jitter rgb_valid_idx must come after RGB columns")


def _uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if low == high:
        return torch.full(shape, low, device=device, dtype=dtype)
    return torch.empty(shape, device=device, dtype=dtype).uniform_(low, high)


def apply_rgb_jitter_to_features(
    features: torch.Tensor,
    cfg: RGBJitterConfig,
) -> torch.Tensor:
    """Apply train-time RGB jitter in-place and return ``features``.

    One brightness factor and one contrast factor are sampled per batch sample.
    RGB-invalid points are left unchanged.
    """

    if not cfg.enabled:
        return features
    if not torch.is_floating_point(features):
        raise TypeError("rgb_jitter expects floating-point features")
    if features.ndim not in (2, 3):
        raise ValueError(
            "rgb_jitter expects features shaped [N, C] or [B, N, C], "
            f"got {tuple(features.shape)}"
        )
    if features.shape[-1] <= cfg.rgb_valid_idx:
        raise ValueError(
            "rgb_jitter expected feature layout [x,y,z,intensity,r,g,b,rgb_valid]; "
            f"got last dimension {features.shape[-1]}"
        )

    batched = features.unsqueeze(0) if features.ndim == 2 else features
    batch_size = batched.shape[0]
    brightness = _uniform(
        (batch_size,),
        cfg.brightness_min,
        cfg.brightness_max,
        device=features.device,
        dtype=features.dtype,
    )
    contrast = _uniform(
        (batch_size,),
        cfg.contrast_min,
        cfg.contrast_max,
        device=features.device,
        dtype=features.dtype,
    )

    for sample_idx in range(batch_size):
        sample = batched[sample_idx]
        valid = sample[:, cfg.rgb_valid_idx] >= cfg.rgb_valid_threshold
        if not bool(valid.any()):
            continue

        rgb = sample[:, cfg.rgb_start_idx : cfg.rgb_end_idx]
        valid_rgb = rgb[valid]
        mean_rgb = valid_rgb.mean(dim=0, keepdim=True)
        jittered = (valid_rgb - mean_rgb) * contrast[sample_idx] + mean_rgb
        jittered = jittered * brightness[sample_idx]
        rgb[valid] = torch.clamp(jittered, 0.0, 1.0)

    return features
