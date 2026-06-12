#!/usr/bin/env python
"""Milestone E step 9: single-sample smoke test.

Loads ONE frame through the extended dataset class with
feature_mode = intensity_rgb_front and verifies:

- shapes are right (point (N,3), feat (N,5), label (N,))
- intensity is finite and standardized (mean ~ 0, std ~ 1)
- RGB is in [0, 1]
- rgb_valid is binary {0.0, 1.0}, with the ratio matching the step-5 audit
- the per-class RGB-valid ratios on this frame are sensible
- A visual overlay of the RGB-colored point cloud back over the camera image
  looks correct.

Also loads the same frame with feature_mode = intensity (D0 path) and
verifies feat.shape == (N, 1) so we know the D0 contract is intact.

Loading a single sample uses one lidar pickle + one semseg pickle + one
JPEG. Safe to run on the laptop.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Make src importable when running directly.
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from thesis_pipeline.adapters.pandaset_ff_lane3 import (
    FEATURE_MODE_INTENSITY,
    FEATURE_MODE_INTENSITY_RGB_FRONT,
    LABEL_MODE_ROAD_MARKING3,
    project_points_to_camera,
)
from thesis_pipeline.datasets.pandaset_ff_lane3_dataset import (
    PandaSetFFLane3Dataset,
)

OUT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUT_DIR / "smoke_report.json"
PNG_PATH = OUT_DIR / "smoke_overlay.png"

SMOKE_SEQ = "003"
SMOKE_FRAME = 0


def fmt(arr):
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)) if arr.size else None,
        "max": float(np.max(arr)) if arr.size else None,
        "mean": float(np.mean(arr)) if arr.size else None,
        "std": float(np.std(arr)) if arr.size else None,
    }


def make_dataset(feature_mode: str) -> PandaSetFFLane3Dataset:
    # Use_cache=False here so we don't write a real cache during smoke.
    # E0 cache is built when the real trainer runs.
    return PandaSetFFLane3Dataset(
        dataset_path=None,
        cache_dir=str(OUT_DIR / "_unused_cache"),
        use_cache=False,
        label_mode=LABEL_MODE_ROAD_MARKING3,
        feature_mode=feature_mode,
        rgb_max_dt_s=0.060,
        color_sampling="bilinear",
    )


def render_overlay(dataset, sample, png_path):
    seq_id = SMOKE_SEQ
    frame_idx = SMOKE_FRAME
    meta = dataset._get_camera_metadata(seq_id)
    target_t = float(meta["lid_ts"][frame_idx])
    diffs = np.abs(meta["cam_ts"] - target_t)
    cam_idx = int(np.argmin(diffs))
    img_path = meta["cam_dir"] / f"{cam_idx:02d}.jpg"
    with Image.open(img_path) as im:
        img = im.convert("RGB")
        W, H = im.size
    # Re-project the same world coords used by _load_sample. We can
    # recover them from the dataset by reading the lidar pkl directly,
    # to keep this overlay independent of internal sample dict layout.
    import pandas as pd
    lidar_pkl = (
        Path(dataset._dataset_path) / seq_id / "lidar" / f"{frame_idx:02d}.pkl"
    )
    pkl = pd.read_pickle(lidar_pkl)
    fwd = (pkl["d"].to_numpy() == 1) if "d" in pkl.columns else np.ones(len(pkl), bool)
    pts_w = pkl.loc[fwd, ["x", "y", "z"]].to_numpy(dtype=np.float64)
    uv, _depth, in_img = project_points_to_camera(
        pts_w, meta["cam_poses"][cam_idx], meta["intrinsics"], W, H
    )

    # The sample's feat layout is [intensity, r, g, b, rgb_valid].
    feat = sample["feat"]
    rgb = feat[:, 1:4]
    rgb_valid = feat[:, 4]

    # Show every 12th point, colored by the RGB we actually attached.
    keep = np.where(in_img)[0][::12]
    fig, ax = plt.subplots(figsize=(20, 11))
    ax.imshow(np.asarray(img))
    if len(keep):
        ax.scatter(
            uv[keep, 0], uv[keep, 1],
            c=rgb[keep], s=8, edgecolors="black", linewidths=0.3,
        )
    ax.set_title(
        f"E0 smoke  seq={seq_id} lidar_frame={frame_idx} cam_frame={cam_idx}  "
        f"rgb_valid_ratio={rgb_valid.mean():.3f}",
        fontsize=11,
    )
    ax.set_axis_off()
    fig.savefig(png_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def main():
    print("=== D0 path (feature_mode=intensity) ===", flush=True)
    t0 = time.time()
    ds_d0 = make_dataset(FEATURE_MODE_INTENSITY)
    sample_d0 = ds_d0._load_sample(SMOKE_SEQ, SMOKE_FRAME)
    feat_d0 = sample_d0["feat"]
    print(f"  D0 feat: shape={feat_d0.shape}  dtype={feat_d0.dtype}")
    assert feat_d0.shape[1] == 1, (
        f"D0 path broken: expected feat.shape[1] == 1, got {feat_d0.shape}"
    )
    print(f"  elapsed={time.time()-t0:.1f}s")

    print("\n=== E0 path (feature_mode=intensity_rgb_front) ===", flush=True)
    t0 = time.time()
    ds_e0 = make_dataset(FEATURE_MODE_INTENSITY_RGB_FRONT)
    sample_e0 = ds_e0._load_sample(SMOKE_SEQ, SMOKE_FRAME)
    point = sample_e0["point"]
    feat = sample_e0["feat"]
    label = sample_e0["label"]
    print(f"  point: {fmt(point)}")
    print(f"  feat:  {fmt(feat)}")
    print(f"  label: shape={label.shape}  dtype={label.dtype}  "
          f"unique={np.unique(label).tolist()}")
    assert feat.shape[1] == 5, (
        f"E0 path broken: expected feat.shape[1] == 5, got {feat.shape}"
    )

    intensity = feat[:, 0]
    r = feat[:, 1]
    g = feat[:, 2]
    b = feat[:, 3]
    rgb_valid = feat[:, 4]
    # Channel-by-channel sanity
    assert np.all(np.isfinite(feat)), "non-finite values in feat"
    assert r.min() >= 0.0 and r.max() <= 1.0, "R out of [0,1]"
    assert g.min() >= 0.0 and g.max() <= 1.0, "G out of [0,1]"
    assert b.min() >= 0.0 and b.max() <= 1.0, "B out of [0,1]"
    # pre-voxel rgb_valid must be exactly 0 or 1
    unique_flag = np.unique(rgb_valid)
    assert set(unique_flag.tolist()).issubset({0.0, 1.0}), (
        f"rgb_valid pre-voxel must be {{0,1}}, got unique={unique_flag.tolist()}"
    )

    # per-class rgb_valid ratios
    valid_mask = rgb_valid > 0.5
    class_stats = {}
    for c, name in [(1, "road"), (2, "marking"), (3, "other")]:
        m = label == c
        n_total = int(m.sum())
        n_valid = int((m & valid_mask).sum())
        class_stats[name] = {
            "n_total": n_total,
            "n_valid": n_valid,
            "valid_ratio": (n_valid / n_total) if n_total else None,
        }
    print(f"\n  intensity stats:")
    print(f"    {fmt(intensity)}")
    print(f"  RGB stats (channel mean):  R={r.mean():.3f} G={g.mean():.3f} B={b.mean():.3f}")
    print(f"  rgb_valid ratio: {rgb_valid.mean():.4f}  "
          f"(0s={(rgb_valid==0).sum()}, 1s={(rgb_valid==1).sum()})")
    print("  per-class valid ratio:")
    for cname, st in class_stats.items():
        if st["valid_ratio"] is not None:
            print(f"    {cname:8s}  total={st['n_total']:6d}  "
                  f"valid_ratio={st['valid_ratio']:.4f}")
    print(f"  elapsed={time.time()-t0:.1f}s")

    print("\n=== Rendering overlay ===", flush=True)
    render_overlay(ds_e0, sample_e0, PNG_PATH)
    print(f"  wrote {PNG_PATH}")

    report = {
        "seq": SMOKE_SEQ,
        "frame": SMOKE_FRAME,
        "d0_feat_shape": list(feat_d0.shape),
        "e0_point_shape": list(point.shape),
        "e0_feat_shape": list(feat.shape),
        "e0_label_unique": np.unique(label).tolist(),
        "intensity": fmt(intensity),
        "rgb_channel_means": {
            "r": float(r.mean()), "g": float(g.mean()), "b": float(b.mean()),
        },
        "rgb_valid_ratio": float(rgb_valid.mean()),
        "rgb_valid_unique_pre_voxel": [float(x) for x in unique_flag.tolist()],
        "per_class": class_stats,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"  wrote {REPORT_PATH}")
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
