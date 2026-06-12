#!/usr/bin/env python
"""Milestone E step 3: render projected-LiDAR overlays for 8 selected frames.

This is a visual sanity check. For each (sequence, lidar_frame) pair we:
- load the forward-sensor LiDAR points (sensor id 1) for that frame
- choose a front-camera frame either by same index or nearest timestamp
- project the LiDAR points onto the camera image using the devkit's pinhole
  projection (no distortion, no occlusion, no motion compensation)
- save a PNG of the camera image with the projected points scatter-overlaid,
  colored by camera-frame depth

The script does NOT decode any image it does not render, and it loads exactly
one LiDAR pickle per frame requested. Memory footprint stays bounded.
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


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[5]
DATASET_ROOT = Path((REPO_ROOT / "logs" / "dataset_root.txt").read_text().strip())
OUT_DIR = Path(__file__).resolve().parent.parent
OVERLAY_DIR = OUT_DIR / "overlays"
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "overlays.csv"
MD_PATH = OUT_DIR / "summary.md"


# ----------------------------------------------------------------------------
# Frame spec
# ----------------------------------------------------------------------------
# mode: "same_index"  -> camera frame == lidar frame
#       "nearest_ts"  -> camera frame = argmin |cam_ts - lid_ts|
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


# ----------------------------------------------------------------------------
# Projection math (inlined from pandaset.geometry to avoid devkit imports)
# ----------------------------------------------------------------------------
def quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def pose_to_mat(pose: dict) -> np.ndarray:
    h = pose["heading"]
    p = pose["position"]
    R = quat_to_rot(h["w"], h["x"], h["y"], h["z"])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [p["x"], p["y"], p["z"]]
    return M


def project_world_to_camera(
    points_world: np.ndarray,  # (N, 3) world coords
    camera_pose: dict,
    intrinsics: dict,           # {"fx","fy","cx","cy"}
    image_w: int,
    image_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (uv (M,2), depth (M,), idx (M,)) for points that fall inside
    the image and in front of the camera."""
    M = pose_to_mat(camera_pose)
    T_world_to_cam = np.linalg.inv(M)

    pts_cam = T_world_to_cam[:3, :3] @ points_world.T + T_world_to_cam[:3, 3:4]
    idx_all = np.arange(points_world.shape[0])

    # z > 0 (in front of camera)
    z = pts_cam[2, :]
    front = z > 0
    pts_cam = pts_cam[:, front]
    idx = idx_all[front]
    z = pts_cam[2, :]

    K = np.array([
        [intrinsics["fx"], 0.0,               intrinsics["cx"]],
        [0.0,              intrinsics["fy"],  intrinsics["cy"]],
        [0.0,              0.0,               1.0],
    ])
    uvw = K @ pts_cam
    u = uvw[0, :] / uvw[2, :]
    v = uvw[1, :] / uvw[2, :]

    in_img = (u > 0) & (u < image_w) & (v > 0) & (v < image_h)
    u = u[in_img]
    v = v[in_img]
    z = z[in_img]
    idx = idx[in_img]
    return np.stack([u, v], axis=1), z, idx


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_json(p: Path):
    with p.open() as f:
        return json.load(f)


def load_forward_lidar(seq_dir: Path, frame_idx: int) -> np.ndarray:
    """Return forward-sensor (d==1) LiDAR points in world coords, shape (N,3)."""
    pkl = seq_dir / "lidar" / f"{frame_idx:02d}.pkl"
    df = pd.read_pickle(pkl)
    if "d" in df.columns:
        df = df[df["d"] == 1]
    return df[["x", "y", "z"]].to_numpy(dtype=np.float64)


def pick_camera_frame(mode: str, lidar_frame: int,
                      cam_ts: list[float], lid_ts: list[float]) -> tuple[int, float]:
    """Return (camera_frame_index, dt = cam_ts - lid_ts)."""
    target = lid_ts[lidar_frame]
    if mode == "same_index":
        ci = lidar_frame
    elif mode == "nearest_ts":
        diffs = [abs(c - target) for c in cam_ts]
        ci = int(np.argmin(diffs))
    else:
        raise ValueError(mode)
    return ci, cam_ts[ci] - target


# ----------------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------------
def render_one(spec: dict, idx_in_list: int) -> dict:
    seq_dir = DATASET_ROOT / spec["seq"]
    fc_dir = seq_dir / "camera" / "front_camera"

    intr = load_json(fc_dir / "intrinsics.json")
    cam_poses = load_json(fc_dir / "poses.json")
    cam_ts = load_json(fc_dir / "timestamps.json")
    lid_ts = load_json(seq_dir / "lidar" / "timestamps.json")

    cam_idx, dt = pick_camera_frame(
        spec["mode"], spec["lidar_frame"], cam_ts, lid_ts
    )

    img_path = fc_dir / f"{cam_idx:02d}.jpg"
    pts_world = load_forward_lidar(seq_dir, spec["lidar_frame"])

    with Image.open(img_path) as img_lazy:
        img = img_lazy.convert("RGB")
        image_w, image_h = img.size

    uv, depth, kept_idx = project_world_to_camera(
        pts_world, cam_poses[cam_idx], intr, image_w, image_h
    )
    n_in = pts_world.shape[0]
    n_out = uv.shape[0]
    valid_ratio = (n_out / n_in) if n_in else 0.0

    # Plot: image as background, points scattered on top, colored by depth.
    fig, ax = plt.subplots(figsize=(image_w / 100, image_h / 100), dpi=100)
    ax.imshow(np.asarray(img))
    if n_out:
        sc = ax.scatter(
            uv[:, 0], uv[:, 1],
            c=np.clip(depth, 0, 60),
            cmap="viridis", s=2, alpha=0.7, edgecolors="none",
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label("depth (m, clipped 0-60)")
    ax.set_xlim(0, image_w)
    ax.set_ylim(image_h, 0)
    ax.set_axis_off()
    title = (
        f"{spec['split']}/{spec['seq']}  lid={spec['lidar_frame']:02d}  "
        f"cam={cam_idx:02d}  mode={spec['mode']}  dt={dt:+.4f}s  "
        f"valid={n_out}/{n_in} ({valid_ratio:.3f})"
    )
    ax.set_title(title, fontsize=10)

    out_name = (
        f"{idx_in_list+1:02d}__{spec['split']}_{spec['seq']}_"
        f"f{spec['lidar_frame']:02d}_{spec['mode']}.png"
    )
    out_path = OVERLAY_DIR / out_name
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "idx": idx_in_list + 1,
        "split": spec["split"],
        "sequence": spec["seq"],
        "lidar_frame": spec["lidar_frame"],
        "mode": spec["mode"],
        "camera_frame": cam_idx,
        "dt_seconds": dt,
        "points_in": n_in,
        "points_projected": n_out,
        "valid_ratio": valid_ratio,
        "image_w": image_w,
        "image_h": image_h,
        "output": out_name,
    }


def main() -> None:
    rows = []
    for i, spec in enumerate(FRAMES):
        print(
            f"[{i+1}/{len(FRAMES)}] {spec['split']}/{spec['seq']} "
            f"lid={spec['lidar_frame']:02d} mode={spec['mode']} ...",
            flush=True,
        )
        rows.append(render_one(spec, i))

    fields = list(rows[0].keys())
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = ["# Projection Overlay Quick Look", ""]
    lines.append("| # | split/seq | lid | cam | mode | dt(s) | valid | pts proj / in | file |")
    lines.append("| ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['idx']} | {r['split']}/{r['sequence']} | "
            f"{r['lidar_frame']:02d} | {r['camera_frame']:02d} | {r['mode']} | "
            f"{r['dt_seconds']:+.4f} | {r['valid_ratio']:.3f} | "
            f"{r['points_projected']} / {r['points_in']} | "
            f"`overlays/{r['output']}` |"
        )
    lines.append("")
    lines.append("Open each PNG and check that LiDAR points land on the correct")
    lines.append("pixels (road on road, car-silhouette points on the car body, etc.).")
    lines.append("Frame 5 (val/054/0 same_index) is expected to look misaligned on")
    lines.append("moving objects; frame 6 (nearest_ts) should look much better.")
    lines.append("")
    MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")
    print(f"Wrote {len(rows)} PNGs under {OVERLAY_DIR}")


if __name__ == "__main__":
    main()
