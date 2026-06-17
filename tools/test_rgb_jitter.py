#!/usr/bin/env python
"""Synthetic checks for train-only RGB jitter feature mutation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from thesis_pipeline.augmentations import RGBJitterConfig, apply_rgb_jitter_to_features


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    features = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 0.5, 0.10, 0.20, 0.30, 1.0],
                [4.0, 5.0, 6.0, 0.6, 0.40, 0.50, 0.60, 0.0],
                [7.0, 8.0, 9.0, 0.7, 0.90, 0.95, 1.00, 1.0],
            ],
            [
                [1.0, 1.0, 1.0, 0.1, 0.20, 0.30, 0.40, 0.4],
                [2.0, 2.0, 2.0, 0.2, 0.30, 0.40, 0.50, 0.5],
                [3.0, 3.0, 3.0, 0.3, 0.40, 0.50, 0.60, 0.9],
            ],
        ],
        dtype=torch.float32,
    )
    original = features.clone()
    cfg = RGBJitterConfig(
        enabled=True,
        brightness_min=2.0,
        brightness_max=2.0,
        contrast_min=1.0,
        contrast_max=1.0,
        rgb_valid_threshold=0.5,
    )
    out = apply_rgb_jitter_to_features(features, cfg)

    _check(out is features, "jitter should mutate and return the same tensor")
    _check(torch.equal(features[..., :4], original[..., :4]), "xyz/intensity changed")
    _check(torch.equal(features[..., 7], original[..., 7]), "rgb_valid changed")

    valid = original[..., 7] >= 0.5
    invalid = ~valid
    expected_valid_rgb = torch.clamp(original[..., 4:7][valid] * 2.0, 0.0, 1.0)
    _check(
        torch.allclose(features[..., 4:7][valid], expected_valid_rgb),
        "valid RGB was not brightness-scaled/clipped as expected",
    )
    _check(
        torch.allclose(features[..., 4:7][invalid], original[..., 4:7][invalid]),
        "invalid RGB points changed",
    )
    _check(
        torch.all((features[..., 4:7] >= 0.0) & (features[..., 4:7] <= 1.0)),
        "RGB values are outside [0, 1]",
    )

    disabled = original.clone()
    disabled_out = apply_rgb_jitter_to_features(
        disabled,
        RGBJitterConfig(enabled=False),
    )
    _check(disabled_out is disabled, "disabled jitter should return same tensor")
    _check(torch.equal(disabled, original), "disabled jitter changed features")

    contrast_features = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.00, 0.20, 0.40, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.00, 0.80, 0.60, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    contrast_original = contrast_features.clone()
    apply_rgb_jitter_to_features(
        contrast_features,
        RGBJitterConfig(
            enabled=True,
            brightness_min=1.0,
            brightness_max=1.0,
            contrast_min=0.5,
            contrast_max=0.5,
            rgb_valid_threshold=0.5,
        ),
    )
    valid_rgb = contrast_original[..., 4:7][contrast_original[..., 7] >= 0.5]
    mean_rgb = valid_rgb.mean(dim=0, keepdim=True)
    expected_contrast = (valid_rgb - mean_rgb) * 0.5 + mean_rgb
    _check(
        torch.allclose(
            contrast_features[..., 4:7][contrast_original[..., 7] >= 0.5],
            expected_contrast,
        ),
        "contrast jitter did not match expected per-sample RGB mean formula",
    )
    _check(
        torch.allclose(
            contrast_features[..., 4:7][contrast_original[..., 7] < 0.5],
            contrast_original[..., 4:7][contrast_original[..., 7] < 0.5],
        ),
        "contrast jitter changed invalid RGB points",
    )

    print("rgb_jitter_synthetic_checks passed")


if __name__ == "__main__":
    main()
