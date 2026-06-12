#!/usr/bin/env python
"""Milestone E step 3: class-colored projection overlays.

For each of the 8 selected frames, render a 3-panel figure:
- left: original front-camera image
- middle: projected forward-LiDAR points colored by road_marking3 class
- right: ONLY the projected marking-class points (raw 8/9/10), drawn large.

The right panel is the key E0 sanity check. Marking points should land on
white road paint in the photo. If they consistently land on asphalt next to
the paint, projection is too imprecise for clean per-point RGB.

road_marking3 remap (matches Milestone D):
- ignore   = raw 1, 2, 3, 4    (hidden in the overlay)
- road     = raw 7
- marking  = raw 8 + 9 + 10
- other    = everything else not ignored
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[5]
DATASET_ROOT = Path((REPO_ROOT / "logs" / "dataset_root.txt").read_text().strip())
OUT_DIR = Path(__file__).resolve().parent.parent
OVERLAY_DIR = OUT_DIR / "overlays_class"
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "overlays_class.csv"
MD_PATH = OUT_DIR / "summary_class.md"

FRAMES = [
    {"split": "train", "seq": "003", "lidar_frame": 0,  "mode": "same_index"},
    {"split": "train", "seq": "017", "lidar_frame": 0,  "mode": "same_index"},
    {"split": "train", "seq": "017", "lidar_frame": 40, "mode": "same_index"},
    {"split": "train", "seq": "037", "lidar_frame": 52, "mode": "same_index"},
    {"split": "val",   "seq": "054", "lidar_frame": 0,  "mode": "same_index"},
    {"split": "val",   "seq": "054", "lidar_frame": 0,  "mode": "nearest_ts"},
    {"split": "val",   "seq": "054", "lidar_frame": 79, "mode": "same_index"},
    {"split": "val",   "seq": "106", "lidar_frame": 20, "mode": "same_index"},
]

SUBSAMPLE_EVERY = 8

# raw_id -> remapped label
IGNORE_RAW = {1, 2, 3, 4}
ROAD_RAW = {7}
MARKING_RAW = {8, 9, 10}
# anything else (not in the above sets) -> other

LABEL_IGNORE = 0
LABEL_ROAD = 1
LABEL_MARKING = 2
LABEL_OTHER = 3


def remap_classes(raw: np.ndarray) -> np.ndarray:
    out = np.full(raw.shape, LABEL_OTHER, dtype=np.int8)
    out[np.isin(raw, list(IGNORE_RAW))] = LABEL_IGNORE
    out[np.isin(raw, list(ROAD_RAW))] = LABEL_ROAD
    out[np.isin(raw, list(MARKING_RAW))] = LABEL_MARKING
    return out


def quat_to_rot(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def pose_to_mat(pose):
    h = pose["heading"]
    p = pose["position"]
    R = quat_to_rot(h["w"], h["x"], h["y"], h["z"])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [p["x"], p["y"], p["z"]]
    return M


def project_with_index(points_world, camera_pose, intrinsics, image_w, image_h):
    """Return (uv (M,2), depth (M,), original_idx (M,)) for survivors."""
    M = pose_to_mat(camera_pose)
    Tinv = np.linalg.inv(M)
    pcam = Tinv[:3, :3] @ points_world.T + Tinv[:3, 3:4]
    idx_all = np.arange(points_world.shape[0])
    front = pcam[2, :] > 0
    pcam = pcam[:, front]
    idx = idx_all[front]
    K = np.array([
        [intrinsics["fx"], 0.0,               intrinsics["cx"]],
        [0.0,              intrinsics["fy"],  intrinsics["cy"]],
        [0.0,              0.0,               1.0],
    ])
    uvw = K @ pcam
    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]
    in_img = (u > 0) & (u < image_w) & (v > 0) & (v < image_h)
    return (
        np.stack([u[in_img], v[in_img]], axis=1),
        pcam[2, in_img],
        idx[in_img],
    )


def load_json(p):
    with p.open() as f:
        return json.load(f)


def load_forward_lidar_with_labels(seq_dir, frame_idx):
    """Returns (points_world (N,3), remapped_labels (N,)) for forward sensor 1."""
    lidar_pkl = seq_dir / "lidar" / f"{frame_idx:02d}.pkl"
    semseg_pkl = seq_dir / "annotations" / "semseg" / f"{frame_idx:02d}.pkl"
    df = pd.read_pickle(lidar_pkl)
    sem = pd.read_pickle(semseg_pkl)
    if len(df) != len(sem):
        raise ValueError(
            f"lidar/semseg row count mismatch in {seq_dir.name}/{frame_idx}: "
            f"{len(df)} vs {len(sem)}"
        )
    raw_class = sem["class"].to_numpy()
    mask = (df["d"].to_numpy() == 1) if "d" in df.columns else np.ones(len(df), bool)
    pts = df.loc[mask, ["x", "y", "z"]].to_numpy(dtype=np.float64)
    cls = remap_classes(raw_class[mask])
    return pts, cls


def pick_camera(mode, lidar_frame, cam_ts, lid_ts):
    target = lid_ts[lidar_frame]
    if mode == "same_index":
        ci = lidar_frame
    else:
        ci = int(np.argmin([abs(c - target) for c in cam_ts]))
    return ci, cam_ts[ci] - target


def render_one(spec, i):
    seq_dir = DATASET_ROOT / spec["seq"]
    fc_dir = seq_dir / "camera" / "front_camera"
    intr = load_json(fc_dir / "intrinsics.json")
    cam_poses = load_json(fc_dir / "poses.json")
    cam_ts = load_json(fc_dir / "timestamps.json")
    lid_ts = load_json(seq_dir / "lidar" / "timestamps.json")

    ci, dt = pick_camera(spec["mode"], spec["lidar_frame"], cam_ts, lid_ts)
    img_path = fc_dir / f"{ci:02d}.jpg"

    pts_world, labels = load_forward_lidar_with_labels(
        seq_dir, spec["lidar_frame"]
    )

    with Image.open(img_path) as im:
        img = im.convert("RGB")
        W, H = img.size

    uv, depth, idx = project_with_index(pts_world, cam_poses[ci], intr, W, H)
    proj_labels = labels[idx]
    n_in = pts_world.shape[0]
    n_proj = uv.shape[0]
    valid = n_proj / max(n_in, 1)

    # class counts (forward sensor, full lidar set)
    n_road_total = int((labels == LABEL_ROAD).sum())
    n_marking_total = int((labels == LABEL_MARKING).sum())
    n_other_total = int((labels == LABEL_OTHER).sum())
    n_ignore_total = int((labels == LABEL_IGNORE).sum())
    # projected counts
    n_road_proj = int((proj_labels == LABEL_ROAD).sum())
    n_marking_proj = int((proj_labels == LABEL_MARKING).sum())
    n_other_proj = int((proj_labels == LABEL_OTHER).sum())

    # ---- panels ----
    fig, axes = plt.subplots(1, 3, figsize=(30, 6))
    img_np = np.asarray(img)

    axes[0].imshow(img_np)
    axes[0].set_title(
        f"original  {spec['split']}/{spec['seq']} cam={ci:02d}", fontsize=11
    )
    axes[0].set_axis_off()

    # Middle: class overlay (subsampled, ignore hidden)
    axes[1].imshow(img_np)
    keep_idx = np.where(proj_labels != LABEL_IGNORE)[0][::SUBSAMPLE_EVERY]
    cls_palette = {
        LABEL_ROAD:    ("#bbbbbb", 6),    # light gray, small
        LABEL_OTHER:   ("#1f77b4", 6),    # blue
        LABEL_MARKING: ("#ffd400", 10),   # bright yellow
    }
    # Plot road and other first, marking on top so it isn't covered.
    for lab in (LABEL_ROAD, LABEL_OTHER, LABEL_MARKING):
        sel = keep_idx[proj_labels[keep_idx] == lab]
        if not len(sel):
            continue
        color, size = cls_palette[lab]
        axes[1].scatter(
            uv[sel, 0], uv[sel, 1],
            c=color, s=size, alpha=0.7,
            edgecolors="black", linewidths=0.3,
            label={
                LABEL_ROAD: "road",
                LABEL_OTHER: "other",
                LABEL_MARKING: "marking",
            }[lab],
        )
    axes[1].legend(loc="upper right", fontsize=10, framealpha=0.7)
    axes[1].set_title(
        f"class overlay  dt={dt:+.4f}s  valid={valid:.3f}  "
        f"R={n_road_proj} M={n_marking_proj} O={n_other_proj}",
        fontsize=10,
    )
    axes[1].set_axis_off()

    # Right: marking-only, ALL marking points, drawn big.
    axes[2].imshow(img_np)
    sel_mark = np.where(proj_labels == LABEL_MARKING)[0]
    if len(sel_mark):
        axes[2].scatter(
            uv[sel_mark, 0], uv[sel_mark, 1],
            c="#ffd400", s=22, alpha=0.9,
            edgecolors="red", linewidths=0.6,
        )
    axes[2].set_title(
        f"marking only  projected={n_marking_proj} / total={n_marking_total}",
        fontsize=10,
    )
    axes[2].set_axis_off()

    out_name = (
        f"{i+1:02d}__{spec['split']}_{spec['seq']}_"
        f"f{spec['lidar_frame']:02d}_{spec['mode']}_class.png"
    )
    fig.savefig(OVERLAY_DIR / out_name, bbox_inches="tight", dpi=110)
    plt.close(fig)

    return {
        "idx": i + 1,
        "split": spec["split"],
        "sequence": spec["seq"],
        "lidar_frame": spec["lidar_frame"],
        "mode": spec["mode"],
        "camera_frame": ci,
        "dt_seconds": dt,
        "points_in": n_in,
        "points_projected": n_proj,
        "valid_ratio": valid,
        "n_road_total": n_road_total,
        "n_marking_total": n_marking_total,
        "n_other_total": n_other_total,
        "n_ignore_total": n_ignore_total,
        "n_road_projected": n_road_proj,
        "n_marking_projected": n_marking_proj,
        "n_other_projected": n_other_proj,
        "marking_proj_ratio": (
            n_marking_proj / n_marking_total if n_marking_total else 0.0
        ),
        "road_proj_ratio": (
            n_road_proj / n_road_total if n_road_total else 0.0
        ),
        "output": out_name,
    }


def main():
    rows = []
    for i, spec in enumerate(FRAMES):
        print(f"[{i+1}/{len(FRAMES)}] {spec['split']}/{spec['seq']} "
              f"lid={spec['lidar_frame']:02d} mode={spec['mode']} ...",
              flush=True)
        rows.append(render_one(spec, i))

    fields = list(rows[0].keys())
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = ["# Class-Colored Projection Overlays", ""]
    lines.append(
        "Three panels per frame: original | class overlay (subsampled) | "
        "marking-only (all marking points, drawn large)."
    )
    lines.append("")
    lines.append("Color key in the class overlay:")
    lines.append("")
    lines.append("- light gray = road (raw 7)")
    lines.append("- blue = other")
    lines.append("- bright yellow = marking (raw 8 + 9 + 10)")
    lines.append("- ignore class (raw 1-4) is hidden.")
    lines.append("")
    lines.append(
        "Key check: in the right panel, do the yellow dots sit on visible "
        "white road paint in the photo, or do they sit on plain asphalt next "
        "to it? If they consistently land on the paint, the projection is "
        "good enough to attach per-point RGB for E0."
    )
    lines.append("")
    lines.append("| # | split/seq | lid | cam | mode | dt(s) | valid | "
                 "marking proj/total | road proj/total | file |")
    lines.append("| ---: | --- | ---: | ---: | --- | ---: | ---: | "
                 "--- | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['idx']} | {r['split']}/{r['sequence']} | "
            f"{r['lidar_frame']:02d} | {r['camera_frame']:02d} | "
            f"{r['mode']} | {r['dt_seconds']:+.4f} | "
            f"{r['valid_ratio']:.3f} | "
            f"{r['n_marking_projected']} / {r['n_marking_total']} "
            f"({r['marking_proj_ratio']:.3f}) | "
            f"{r['n_road_projected']} / {r['n_road_total']} "
            f"({r['road_proj_ratio']:.3f}) | "
            f"`overlays_class/{r['output']}` |"
        )
    lines.append("")
    MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")
    print(f"Wrote {len(rows)} PNGs under {OVERLAY_DIR}")


if __name__ == "__main__":
    main()
