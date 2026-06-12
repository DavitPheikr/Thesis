#!/usr/bin/env python
"""Milestone E step 7: voxel-mixing audit.

For 8 step-3 frames, measure how often a 4 cm voxel contains both
RGB-valid and RGB-invalid contributors. The answer drives the dataset
preprocessor's rgb_valid handling policy.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]
DATASET_ROOT = Path((REPO_ROOT / "logs" / "dataset_root.txt").read_text().strip())
OUT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

RGB_MAX_DT_S = 0.060
GRID_SIZE_M = 0.04
IMAGE_W, IMAGE_H = 1920, 1080

IGNORE_RAW = {1, 2, 3, 4}
ROAD_RAW = {7}
MARKING_RAW = {8, 9, 10}
LABEL_IGNORE, LABEL_ROAD, LABEL_MARKING, LABEL_OTHER = 0, 1, 2, 3

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


def remap_classes(raw):
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
    h, p = pose["heading"], pose["position"]
    R = quat_to_rot(h["w"], h["x"], h["y"], h["z"])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [p["x"], p["y"], p["z"]]
    return M


def in_image_mask(points_world, camera_pose, intrinsics, W, H):
    M = pose_to_mat(camera_pose)
    Tinv = np.linalg.inv(M)
    pcam = Tinv[:3, :3] @ points_world.T + Tinv[:3, 3:4]
    n = points_world.shape[0]
    z = pcam[2, :]
    K = np.array([
        [intrinsics["fx"], 0.0, intrinsics["cx"]],
        [0.0, intrinsics["fy"], intrinsics["cy"]],
        [0.0, 0.0, 1.0],
    ])
    safe_z = np.where(z > 1e-6, z, 1.0)
    u = (K @ pcam)[0] / safe_z
    v = (K @ pcam)[1] / safe_z
    return (z > 0) & (u > 0) & (u < W) & (v > 0) & (v < H)


def load_json(p):
    with p.open() as f:
        return json.load(f)


def pick_camera(mode, lf, cam_ts, lid_ts):
    target = lid_ts[lf]
    if mode == "same_index":
        ci = lf
    else:
        ci = int(np.argmin(np.abs(np.array(cam_ts) - target)))
    return ci, cam_ts[ci] - target


def voxel_index_3d(pts_world):
    """Discretize each point to its 4 cm voxel index in world coordinates."""
    q = np.floor(pts_world / GRID_SIZE_M).astype(np.int64)
    # Pack (i, j, k) into a single 64-bit hash.
    base = 1 << 20
    return (q[:, 0] + base) * (base * base) + \
           (q[:, 1] + base) * base + \
           (q[:, 2] + base)


def process_frame(spec):
    seq_dir = DATASET_ROOT / spec["seq"]
    fc_dir = seq_dir / "camera" / "front_camera"
    intr = load_json(fc_dir / "intrinsics.json")
    cam_poses = load_json(fc_dir / "poses.json")
    cam_ts = load_json(fc_dir / "timestamps.json")
    lid_ts = load_json(seq_dir / "lidar" / "timestamps.json")

    lf = spec["lidar_frame"]
    ci, dt = pick_camera(spec["mode"], lf, cam_ts, lid_ts)
    frame_valid = abs(dt) <= RGB_MAX_DT_S

    lid_pkl = seq_dir / "lidar" / f"{lf:02d}.pkl"
    sem_pkl = seq_dir / "annotations" / "semseg" / f"{lf:02d}.pkl"
    df = pd.read_pickle(lid_pkl)
    sem = pd.read_pickle(sem_pkl)
    raw_full = sem["class"].to_numpy()
    fwd = (df["d"].to_numpy() == 1) if "d" in df.columns else np.ones(len(df), bool)
    pts_w = df.loc[fwd, ["x", "y", "z"]].to_numpy(dtype=np.float64)
    cls = remap_classes(raw_full[fwd])

    if frame_valid:
        in_img = in_image_mask(pts_w, cam_poses[ci], intr, IMAGE_W, IMAGE_H)
    else:
        in_img = np.zeros(pts_w.shape[0], dtype=bool)
    rgb_valid = in_img.astype(np.int8)

    vox = voxel_index_3d(pts_w)
    is_marking = cls == LABEL_MARKING

    # group by voxel
    order = np.argsort(vox, kind="stable")
    vox_s = vox[order]
    valid_s = rgb_valid[order]
    mark_s = is_marking[order]

    # boundaries between voxels
    boundaries = np.empty(vox_s.shape, dtype=bool)
    boundaries[0] = True
    boundaries[1:] = vox_s[1:] != vox_s[:-1]
    starts = np.where(boundaries)[0]
    ends = np.r_[starts[1:], len(vox_s)]

    # per-voxel stats
    n_voxels = len(starts)
    n_singletons = 0
    n_multi = 0
    n_pure_valid = 0
    n_pure_invalid = 0
    n_mixed = 0
    # voxels containing marking
    n_vox_with_marking = 0
    n_vox_marking_pure_valid = 0
    n_vox_marking_pure_invalid = 0
    n_vox_marking_mixed = 0

    for s, e in zip(starts, ends):
        size = e - s
        valid_sum = int(valid_s[s:e].sum())
        marking_in_voxel = bool(mark_s[s:e].any())
        if size == 1:
            n_singletons += 1
            if valid_sum == 1:
                n_pure_valid += 1
            else:
                n_pure_invalid += 1
            if marking_in_voxel:
                n_vox_with_marking += 1
                if valid_sum == size:
                    n_vox_marking_pure_valid += 1
                else:
                    n_vox_marking_pure_invalid += 1
        else:
            n_multi += 1
            if valid_sum == 0:
                n_pure_invalid += 1
                if marking_in_voxel:
                    n_vox_with_marking += 1
                    n_vox_marking_pure_invalid += 1
            elif valid_sum == size:
                n_pure_valid += 1
                if marking_in_voxel:
                    n_vox_with_marking += 1
                    n_vox_marking_pure_valid += 1
            else:
                n_mixed += 1
                if marking_in_voxel:
                    n_vox_with_marking += 1
                    n_vox_marking_mixed += 1

    return {
        "split": spec["split"],
        "sequence": spec["seq"],
        "lidar_frame": lf,
        "mode": spec["mode"],
        "cam_frame": ci,
        "dt_seconds": dt,
        "frame_valid": int(frame_valid),
        "n_points": int(pts_w.shape[0]),
        "n_voxels": n_voxels,
        "n_voxels_single": n_singletons,
        "n_voxels_multi": n_multi,
        "n_voxels_pure_valid": n_pure_valid,
        "n_voxels_pure_invalid": n_pure_invalid,
        "n_voxels_mixed": n_mixed,
        "frac_voxels_mixed": (n_mixed / n_voxels) if n_voxels else 0.0,
        "frac_multi_mixed": (n_mixed / n_multi) if n_multi else 0.0,
        "n_voxels_with_marking": n_vox_with_marking,
        "n_voxels_marking_pure_valid": n_vox_marking_pure_valid,
        "n_voxels_marking_pure_invalid": n_vox_marking_pure_invalid,
        "n_voxels_marking_mixed": n_vox_marking_mixed,
        "frac_marking_voxels_mixed": (n_vox_marking_mixed / n_vox_with_marking)
                                     if n_vox_with_marking else 0.0,
    }


def main():
    rows = []
    for i, spec in enumerate(FRAMES):
        print(f"[{i+1}/{len(FRAMES)}] {spec['split']}/{spec['seq']} "
              f"f{spec['lidar_frame']:02d} {spec['mode']} ...", flush=True)
        r = process_frame(spec)
        print(f"  voxels={r['n_voxels']}  with_marking={r['n_voxels_with_marking']}  "
              f"marking_mixed={r['n_voxels_marking_mixed']} "
              f"({r['frac_marking_voxels_mixed']*100:.2f}%)")
        rows.append(r)

    fields = list(rows[0].keys())
    with (OUT_DIR / "per_frame.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    total_mark_voxels = sum(r["n_voxels_with_marking"] for r in rows)
    total_mark_mixed = sum(r["n_voxels_marking_mixed"] for r in rows)
    agg_frac = (total_mark_mixed / total_mark_voxels) if total_mark_voxels else 0.0

    lines = ["# Voxel-Mixing Audit", ""]
    lines.append(f"Grid size: {GRID_SIZE_M} m. {len(rows)} frames sampled.")
    lines.append("")
    lines.append("## Per-frame mixing of marking voxels")
    lines.append("")
    lines.append("| split/seq | lid | mode | voxels w/ marking | mixed | mixed % |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            f"| {r['split']}/{r['sequence']} | {r['lidar_frame']:02d} | "
            f"{r['mode']} | {r['n_voxels_with_marking']} | "
            f"{r['n_voxels_marking_mixed']} | "
            f"{r['frac_marking_voxels_mixed']*100:.2f}% |"
        )
    lines.append("")
    lines.append(f"## Aggregate across the 8 frames")
    lines.append("")
    lines.append(f"- total voxels containing marking: {total_mark_voxels}")
    lines.append(f"- of those, mixed valid/invalid contributors: {total_mark_mixed}")
    lines.append(f"- aggregate mixed fraction: **{agg_frac*100:.2f}%**")
    lines.append("")

    lines.append("## E0 decision")
    lines.append("")
    if agg_frac < 0.05:
        decision = ("**SKIP** — pass-through the fractional rgb_valid flag. "
                    "Mixed voxels are rare; not worth the implementation "
                    "complexity. Document as a known limitation.")
    elif agg_frac > 0.15:
        decision = ("**MASK-AND-THRESHOLD** — at preprocessing, average RGB "
                    "only over valid contributors, then snap rgb_valid to "
                    "{0, 1}.")
    else:
        decision = ("**JUDGMENT CALL** — falls in 5-15% band. Weigh "
                    "implementation cost vs expected E0 benefit.")
    lines.append(decision)
    lines.append("")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_DIR / 'per_frame.csv'}")
    print(f"Wrote {OUT_DIR / 'summary.md'}")
    print(f"\nAggregate marking-voxel mixed fraction: {agg_frac*100:.2f}%")


if __name__ == "__main__":
    main()
