"""Training-time augmentation helpers."""

from .rgb_jitter import RGBJitterConfig, apply_rgb_jitter_to_features

__all__ = ["RGBJitterConfig", "apply_rgb_jitter_to_features"]
