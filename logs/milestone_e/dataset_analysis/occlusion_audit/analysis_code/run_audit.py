#!/usr/bin/env python
"""Milestone E step 6: occlusion audit via z-buffer.

For every frame in train/val/test, project forward-LiDAR points through the
step-4-chosen camera frame, run a per-pixel z-buffer, and count points
that would receive RGB from a closer occluding object. Counts are broken
down by class, distance bucket, and marking subtype.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]
DATASET_ROOT = Path((REPO_ROOT / "logs" / "dataset_root.txt").read_text().strip())
OUT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": REPO_ROOT / "configs" / "splits" / "train.txt",
    "val":   REPO_ROOT / "configs" / "splits" / "val.txt",
    "test":  REPO_ROOT / "configs" / "splits" / "test.txt",
}

RGB_MAX_DT_S = 0.060
IMAGE_W, IMAGE_H = 1920, 1080
Z_TOLERANCE_M = 0.5

IGNORE_RAW = {1, 2, 3, 4}
ROAD_RAW = {7}
MARKING_RAW = {8, 9, 10}
LABEL_IGNORE, LABEL_ROAD, LABEL_MARKING, LABEL_OTHER = 0, 1, 2, 3
CLASS_NAMES = {LABEL_ROAD: "road", LABEL_MARKING: "marking", LABEL_OTHER: "other"}

BUCKETS = [
    ("0_10m",     0.0,  10.0),
    ("10_20m",   10.0,  20.0),
    ("20_30m",   20.0,  30.0),
    ("30_40m",   30.0,  40.0),
    ("40_60m",   40.0,  60.0),
    ("60m_plus", 60.0,  float("inf")),
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


def project_full(points_world, camera_pose, intrinsics, W, H):
    """Return (in_img_mask (N,), u (N,), v (N,), depth (N,)).
    Values where in_img_mask is False are not meaningful."""
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
    uvw = K @ pcam
    safe_z = np.where(z > 1e-6, z, 1.0)
    u = uvw[0] / safe_z
    v = uvw[1] / safe_z
    in_img = (z > 0) & (u > 0) & (u < W) & (v > 0) & (v < H)
    return in_img, u, v, z


def zbuffer_occluded(u, v, depth, mask):
    """Return a boolean array (same shape as mask) True where the point is
    inside the image AND another point at the same integer pixel is closer
    by more than Z_TOLERANCE_M."""
    n = u.shape[0]
    occluded = np.zeros(n, dtype=bool)
    if not mask.any():
        return occluded
    idx = np.where(mask)[0]
    ui = u[idx].astype(np.int32)
    vi = v[idx].astype(np.int32)
    di = depth[idx]
    # pixel key
    pix = vi.astype(np.int64) * IMAGE_W + ui.astype(np.int64)
    # Sort by pixel, then by depth ascending
    order = np.lexsort((di, pix))
    pix_s = pix[order]
    di_s = di[order]
    idx_s = idx[order]
    # min depth per pixel: first occurrence of each pix in sorted order
    is_first = np.empty(pix_s.shape, dtype=bool)
    is_first[0] = True
    is_first[1:] = pix_s[1:] != pix_s[:-1]
    # propagate min depth forward
    min_depth_per_row = np.empty_like(di_s)
    cur_min = np.inf
    for i in range(pix_s.shape[0]):
        if is_first[i]:
            cur_min = di_s[i]
        min_depth_per_row[i] = cur_min
    occluded_s = (di_s - min_depth_per_row) > Z_TOLERANCE_M
    occluded[idx_s] = occluded_s
    return occluded


def load_json(p):
    with p.open() as f:
        return json.load(f)


def planar_range_ego(points_world, lidar_pose):
    M = pose_to_mat(lidar_pose)
    Tinv = np.linalg.inv(M)
    pe = Tinv[:3, :3] @ points_world.T + Tinv[:3, 3:4]
    return np.sqrt(pe[0] ** 2 + pe[1] ** 2)


def bucket_assign(dist):
    out = np.full(dist.shape, len(BUCKETS) - 1, dtype=np.int8)
    for bi, (_, lo, hi) in enumerate(BUCKETS):
        m = (dist >= lo) & (dist < hi)
        out[m] = bi
    return out


def empty_counts():
    return {
        "n_rgb_valid_inimg": 0,
        "n_occluded": 0,
        "by_class": {LABEL_ROAD: [0, 0],
                     LABEL_MARKING: [0, 0],
                     LABEL_OTHER: [0, 0]},
        "by_bucket_class": {b[0]: {LABEL_ROAD: [0, 0],
                                   LABEL_MARKING: [0, 0],
                                   LABEL_OTHER: [0, 0]}
                            for b in BUCKETS},
        "by_raw_marking": {8: [0, 0], 9: [0, 0], 10: [0, 0]},
    }


def process_sequence(split, seq):
    seq_dir = DATASET_ROOT / seq
    fc_dir = seq_dir / "camera" / "front_camera"
    intr = load_json(fc_dir / "intrinsics.json")
    cam_poses = load_json(fc_dir / "poses.json")
    cam_ts = load_json(fc_dir / "timestamps.json")
    lid_ts = load_json(seq_dir / "lidar" / "timestamps.json")
    lid_poses = load_json(seq_dir / "lidar" / "poses.json")

    seq_agg = empty_counts()
    frame_rows = []
    n_frames = min(len(lid_ts), len(cam_poses), len(lid_poses))
    for lf in range(n_frames):
        target = lid_ts[lf]
        diffs = np.abs(np.array(cam_ts) - target)
        ci = int(diffs.argmin())
        dt = cam_ts[ci] - target
        frame_valid = abs(dt) <= RGB_MAX_DT_S

        lid_pkl = seq_dir / "lidar" / f"{lf:02d}.pkl"
        sem_pkl = seq_dir / "annotations" / "semseg" / f"{lf:02d}.pkl"
        df = pd.read_pickle(lid_pkl)
        sem = pd.read_pickle(sem_pkl)
        raw_full = sem["class"].to_numpy()
        fwd = (df["d"].to_numpy() == 1) if "d" in df.columns else np.ones(len(df), bool)
        pts_w = df.loc[fwd, ["x", "y", "z"]].to_numpy(dtype=np.float64)
        raw_class = raw_full[fwd]
        cls = remap_classes(raw_class)

        dist = planar_range_ego(pts_w, lid_poses[lf])
        bidx = bucket_assign(dist)

        if frame_valid:
            in_img, u, v, depth = project_full(pts_w, cam_poses[ci], intr,
                                               IMAGE_W, IMAGE_H)
            occluded = zbuffer_occluded(u, v, depth, in_img)
        else:
            in_img = np.zeros(pts_w.shape[0], dtype=bool)
            occluded = np.zeros(pts_w.shape[0], dtype=bool)

        # counts
        n_in = int(in_img.sum())
        n_occ = int(occluded.sum())
        seq_agg["n_rgb_valid_inimg"] += n_in
        seq_agg["n_occluded"] += n_occ

        for c in (LABEL_ROAD, LABEL_MARKING, LABEL_OTHER):
            m = cls == c
            seq_agg["by_class"][c][0] += int((m & in_img).sum())
            seq_agg["by_class"][c][1] += int((m & occluded).sum())
        for bi, (bname, _, _) in enumerate(BUCKETS):
            bmask = bidx == bi
            for c in (LABEL_ROAD, LABEL_MARKING, LABEL_OTHER):
                m = (cls == c) & bmask
                seq_agg["by_bucket_class"][bname][c][0] += int((m & in_img).sum())
                seq_agg["by_bucket_class"][bname][c][1] += int((m & occluded).sum())
        for raw_id in (8, 9, 10):
            m = raw_class == raw_id
            seq_agg["by_raw_marking"][raw_id][0] += int((m & in_img).sum())
            seq_agg["by_raw_marking"][raw_id][1] += int((m & occluded).sum())

        row = {
            "split": split, "sequence": seq, "lidar_frame": lf,
            "cam_frame": ci, "dt_seconds": dt,
            "frame_valid": int(frame_valid),
            "n_in_image": n_in, "n_occluded": n_occ,
        }
        for c, name in CLASS_NAMES.items():
            m = cls == c
            row[f"n_{name}_in_image"] = int((m & in_img).sum())
            row[f"n_{name}_occluded"] = int((m & occluded).sum())
        for raw_id in (8, 9, 10):
            m = raw_class == raw_id
            row[f"n_raw{raw_id}_in_image"] = int((m & in_img).sum())
            row[f"n_raw{raw_id}_occluded"] = int((m & occluded).sum())
        frame_rows.append(row)

    return seq_agg, frame_rows


def main():
    t0 = time.time()
    all_frame_rows = []
    splits_aggregate = {s: empty_counts() for s in SPLITS}

    for split, sp in SPLITS.items():
        seqs = [s.strip() for s in sp.read_text().splitlines() if s.strip()]
        for i, seq in enumerate(seqs):
            t_seq = time.time()
            seq_agg, frame_rows = process_sequence(split, seq)
            all_frame_rows.extend(frame_rows)
            elapsed = time.time() - t_seq
            tot, occ = (seq_agg["by_class"][LABEL_MARKING][0],
                        seq_agg["by_class"][LABEL_MARKING][1])
            ratio = (occ / tot) if tot else 0.0
            print(f"[{split} {i+1}/{len(seqs)}] {seq}  "
                  f"frames={len(frame_rows)}  "
                  f"marking_occ_ratio={ratio:.4f}  "
                  f"elapsed={elapsed:.1f}s", flush=True)

            sa = splits_aggregate[split]
            sa["n_rgb_valid_inimg"] += seq_agg["n_rgb_valid_inimg"]
            sa["n_occluded"] += seq_agg["n_occluded"]
            for c in (LABEL_ROAD, LABEL_MARKING, LABEL_OTHER):
                sa["by_class"][c][0] += seq_agg["by_class"][c][0]
                sa["by_class"][c][1] += seq_agg["by_class"][c][1]
            for bname, _, _ in BUCKETS:
                for c in (LABEL_ROAD, LABEL_MARKING, LABEL_OTHER):
                    sa["by_bucket_class"][bname][c][0] += seq_agg["by_bucket_class"][bname][c][0]
                    sa["by_bucket_class"][bname][c][1] += seq_agg["by_bucket_class"][bname][c][1]
            for raw_id in (8, 9, 10):
                sa["by_raw_marking"][raw_id][0] += seq_agg["by_raw_marking"][raw_id][0]
                sa["by_raw_marking"][raw_id][1] += seq_agg["by_raw_marking"][raw_id][1]

    fields = list(all_frame_rows[0].keys())
    with (OUT_DIR / "per_frame.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_frame_rows:
            w.writerow(r)

    with (OUT_DIR / "aggregate.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "bucket", "n_in_image", "n_occluded",
                    "occluded_ratio"])
        for split in SPLITS:
            for c, cname in CLASS_NAMES.items():
                for bname, _, _ in BUCKETS:
                    n_in, n_occ = splits_aggregate[split]["by_bucket_class"][bname][c]
                    w.writerow([split, cname, bname, n_in, n_occ,
                                (n_occ / n_in) if n_in else 0.0])

    with (OUT_DIR / "marking_subtype.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "raw_id", "raw_name", "n_in_image", "n_occluded",
                    "occluded_ratio"])
        names = {8: "lane_line", 9: "stop_line", 10: "other_road_marking"}
        for split in SPLITS:
            for raw_id in (8, 9, 10):
                n_in, n_occ = splits_aggregate[split]["by_raw_marking"][raw_id]
                w.writerow([split, raw_id, names[raw_id], n_in, n_occ,
                            (n_occ / n_in) if n_in else 0.0])

    lines = ["# Occlusion Audit", ""]
    elapsed = time.time() - t0
    lines.append(f"Computed in {elapsed:.1f}s over all 6080 frames "
                 f"with Z_TOLERANCE_M = {Z_TOLERANCE_M}.")
    lines.append("")
    lines.append("Counts are restricted to points that already pass the step-5 "
                 "valid-RGB filter (in front of camera, inside image, frame "
                 "within 60 ms). Occluded fraction = occluded / in_image.")
    lines.append("")

    lines.append("## Overall occlusion by split and class")
    lines.append("")
    lines.append("| split | overall | road occl | marking occl | other occl |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for split in SPLITS:
        a = splits_aggregate[split]
        def _r(n, d):
            return f"{(n/d):.4f}" if d else "n/a"
        lines.append(
            f"| {split} | {_r(a['n_occluded'], a['n_rgb_valid_inimg'])} | "
            f"{_r(a['by_class'][LABEL_ROAD][1], a['by_class'][LABEL_ROAD][0])} | "
            f"{_r(a['by_class'][LABEL_MARKING][1], a['by_class'][LABEL_MARKING][0])} | "
            f"{_r(a['by_class'][LABEL_OTHER][1], a['by_class'][LABEL_OTHER][0])} |"
        )
    lines.append("")

    lines.append("## Marking occlusion by distance bucket")
    lines.append("")
    header = "| split |" + "".join(f" {b[0]} |" for b in BUCKETS)
    sep = "| --- |" + "".join(" ---: |" for _ in BUCKETS)
    lines.append(header)
    lines.append(sep)
    for split in SPLITS:
        a = splits_aggregate[split]
        cells = []
        for bname, _, _ in BUCKETS:
            n_in, n_occ = a["by_bucket_class"][bname][LABEL_MARKING]
            cells.append(f" {(n_occ/n_in):.4f}" if n_in else " n/a")
        lines.append(f"| {split} |" + " |".join(cells) + " |")
    lines.append("")

    lines.append("## Marking occlusion by subtype (raw 8 / 9 / 10)")
    lines.append("")
    lines.append("| split | raw 8 lane | raw 9 stop | raw 10 other |")
    lines.append("| --- | ---: | ---: | ---: |")
    for split in SPLITS:
        a = splits_aggregate[split]
        cells = []
        for raw_id in (8, 9, 10):
            n_in, n_occ = a["by_raw_marking"][raw_id]
            cells.append(f"{(n_occ/n_in):.4f}" if n_in else "n/a")
        lines.append(f"| {split} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")

    lines.append("## E0 decision rule")
    lines.append("")
    train_mark = splits_aggregate["train"]["by_class"][LABEL_MARKING]
    train_mark_ratio = (train_mark[1] / train_mark[0]) if train_mark[0] else 0.0
    lines.append(f"Train marking occluded ratio = {train_mark_ratio:.4f} "
                 f"({100*train_mark_ratio:.2f}%).")
    if train_mark_ratio < 0.02:
        lines.append("")
        lines.append("**Decision: SKIP** occlusion handling for E0. Below 2% "
                     "threshold. Document as a known limitation.")
    elif train_mark_ratio > 0.10:
        lines.append("")
        lines.append("**Decision: ADD z-buffer** to E0 dataset code. Above "
                     "10% threshold.")
    else:
        lines.append("")
        lines.append("**Decision: judgment call.** Falls in the 2-10% band. "
                     "Compare implementation cost vs expected E0 benefit.")
    lines.append("")

    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_DIR / 'per_frame.csv'}")
    print(f"Wrote {OUT_DIR / 'aggregate.csv'}")
    print(f"Wrote {OUT_DIR / 'marking_subtype.csv'}")
    print(f"Wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
