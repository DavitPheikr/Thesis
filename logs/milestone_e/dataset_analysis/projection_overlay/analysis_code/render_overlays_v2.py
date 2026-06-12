#!/usr/bin/env python
"""Milestone E step 3 (v2): sparser, more diagnostic overlay.

Same 8 frames as v1, but the visualization is built for readability:
- two panels side by side: original image | overlay
- the overlay is heavily subsampled so you can actually see individual dots
- three "anchor" dots at known camera-frame depths (5 m, 10 m, 20 m straight
  ahead) drawn as large red circles with labels. If the projection is
  correct, these sit on the road at plausible positions in the image.

Outputs to overlays_v2/. The v1 outputs remain in overlays/ untouched.
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
OVERLAY_DIR = OUT_DIR / "overlays_v2"
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "overlays_v2.csv"
MD_PATH = OUT_DIR / "summary_v2.md"

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

SUBSAMPLE_EVERY = 30   # keep one in every N projected points
ANCHOR_DEPTHS_M = [5.0, 10.0, 20.0]


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


def project(points_world, camera_pose, intrinsics, image_w, image_h):
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
    return np.stack([u[in_img], v[in_img]], axis=1), pcam.T[in_img], idx[in_img]


def load_json(p):
    with p.open() as f:
        return json.load(f)


def load_forward_lidar(seq_dir, frame_idx):
    pkl = seq_dir / "lidar" / f"{frame_idx:02d}.pkl"
    df = pd.read_pickle(pkl)
    if "d" in df.columns:
        df = df[df["d"] == 1]
    return df[["x", "y", "z"]].to_numpy(dtype=np.float64)


def pick_camera(mode, lidar_frame, cam_ts, lid_ts):
    target = lid_ts[lidar_frame]
    if mode == "same_index":
        ci = lidar_frame
    else:
        ci = int(np.argmin([abs(c - target) for c in cam_ts]))
    return ci, cam_ts[ci] - target


def find_anchor(pcam, target_depth_m):
    """Among projected (front-of-camera, in-image) points, pick the one closest
    to (x=0, y arbitrary, z=target_depth). Returns the row index into pcam."""
    if pcam.shape[0] == 0:
        return None
    # Prefer points roughly at ground level: y_cam ~ +1 .. +2 m below camera
    # (in camera frame, +y is down). We rank by abs(z - target) + abs(x).
    cost = np.abs(pcam[:, 2] - target_depth_m) + np.abs(pcam[:, 0])
    return int(np.argmin(cost))


def render_one(spec, i):
    seq_dir = DATASET_ROOT / spec["seq"]
    fc_dir = seq_dir / "camera" / "front_camera"
    intr = load_json(fc_dir / "intrinsics.json")
    cam_poses = load_json(fc_dir / "poses.json")
    cam_ts = load_json(fc_dir / "timestamps.json")
    lid_ts = load_json(seq_dir / "lidar" / "timestamps.json")

    ci, dt = pick_camera(spec["mode"], spec["lidar_frame"], cam_ts, lid_ts)
    img_path = fc_dir / f"{ci:02d}.jpg"
    pts_world = load_forward_lidar(seq_dir, spec["lidar_frame"])

    with Image.open(img_path) as im:
        img = im.convert("RGB")
        W, H = img.size

    uv, pcam, _ = project(pts_world, cam_poses[ci], intr, W, H)
    n_proj = uv.shape[0]
    n_in = pts_world.shape[0]
    valid = n_proj / max(n_in, 1)

    # subsample for display
    keep = np.arange(0, n_proj, SUBSAMPLE_EVERY)
    uv_s = uv[keep]
    depth_s = pcam[keep, 2]

    # anchors (computed from full projected set, not subsample)
    anchors = []
    for d in ANCHOR_DEPTHS_M:
        idx = find_anchor(pcam, d)
        if idx is not None:
            anchors.append((d, uv[idx, 0], uv[idx, 1], pcam[idx, 2]))

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    img_np = np.asarray(img)

    axes[0].imshow(img_np)
    axes[0].set_title(f"original  {spec['split']}/{spec['seq']} cam={ci:02d}",
                      fontsize=11)
    axes[0].set_axis_off()

    axes[1].imshow(img_np)
    if len(uv_s):
        sc = axes[1].scatter(
            uv_s[:, 0], uv_s[:, 1],
            c=np.clip(depth_s, 0, 60), cmap="viridis",
            s=12, edgecolors="black", linewidths=0.4,
        )
        cbar = fig.colorbar(sc, ax=axes[1], fraction=0.03, pad=0.01)
        cbar.set_label("depth (m, 0-60)")

    for (d, u, v, real_z) in anchors:
        axes[1].plot(u, v, "o", markersize=18, mfc="none",
                     mec="red", mew=2.5)
        axes[1].annotate(
            f"~{d:.0f} m (actual {real_z:.1f} m)",
            (u, v), xytext=(10, -10), textcoords="offset points",
            color="white", fontsize=11, weight="bold",
            bbox=dict(facecolor="red", alpha=0.7, pad=2,
                      edgecolor="none"),
        )

    title = (
        f"overlay  lid={spec['lidar_frame']:02d}  cam={ci:02d}  "
        f"mode={spec['mode']}  dt={dt:+.4f}s  "
        f"valid={n_proj}/{n_in} ({valid:.3f})  "
        f"shown={len(uv_s)} of {n_proj}"
    )
    axes[1].set_title(title, fontsize=10)
    axes[1].set_axis_off()

    out_name = (
        f"{i+1:02d}__{spec['split']}_{spec['seq']}_"
        f"f{spec['lidar_frame']:02d}_{spec['mode']}.png"
    )
    fig.savefig(OVERLAY_DIR / out_name, bbox_inches="tight", dpi=120)
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
        "points_shown": len(uv_s),
        "valid_ratio": valid,
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

    lines = ["# Projection Overlay (v2) Quick Look", ""]
    lines.append("Sparser overlays (every 30th point) with 5 m / 10 m / 20 m "
                 "anchor circles on the right panel.")
    lines.append("")
    lines.append("| # | split/seq | lid | cam | mode | dt(s) | valid | "
                 "shown / proj | file |")
    lines.append("| ---: | --- | ---: | ---: | --- | ---: | ---: | "
                 "--- | --- |")
    for r in rows:
        lines.append(
            f"| {r['idx']} | {r['split']}/{r['sequence']} | "
            f"{r['lidar_frame']:02d} | {r['camera_frame']:02d} | "
            f"{r['mode']} | {r['dt_seconds']:+.4f} | "
            f"{r['valid_ratio']:.3f} | "
            f"{r['points_shown']} / {r['points_projected']} | "
            f"`overlays_v2/{r['output']}` |"
        )
    lines.append("")
    lines.append("## How to read these")
    lines.append("")
    lines.append("Each red circle is a single LiDAR point chosen to be roughly")
    lines.append("at 5 m / 10 m / 20 m directly in front of the camera. If the")
    lines.append("projection is correct, the 5 m circle sits on the road just")
    lines.append("ahead of the car; 10 m further down the road; 20 m further")
    lines.append("still. If they land in the sky, on a building, or off the road,")
    lines.append("the projection is wrong.")
    lines.append("")
    lines.append("The colored dots are a 1-in-30 sample. They should sit on the")
    lines.append("road and on the silhouettes of cars / poles / signs, not float")
    lines.append("above the road or appear in the sky.")
    lines.append("")
    MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")
    print(f"Wrote {len(rows)} PNGs under {OVERLAY_DIR}")


if __name__ == "__main__":
    main()
