#!/usr/bin/env python
"""Milestone E step 5: stratified valid-RGB audit.

For every (sequence, lidar_frame) in the train/val/test splits, apply the
step-4 timestamp policy (nearest_ts + 60 ms) and project the forward-LiDAR
points through the chosen camera frame. Count, per class and per distance
bucket and per marking subtype, the total points and the valid-RGB points.

Writes:
- per_frame.csv          one row per (split, sequence, lidar_frame)
- aggregate.csv          one row per (split, class, distance bucket)
- marking_subtype.csv    one row per (split, raw_id) for raw 8/9/10
- per_sequence.csv       one row per sequence with summary ratios
- summary.md             headline tables
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
IMAGE_W, IMAGE_H = 1920, 1080  # uniform across the dataset (verified in step 1)

# Class remap (matches Milestone D)
IGNORE_RAW = {1, 2, 3, 4}
ROAD_RAW = {7}
MARKING_RAW = {8, 9, 10}
LABEL_IGNORE, LABEL_ROAD, LABEL_MARKING, LABEL_OTHER = 0, 1, 2, 3
CLASS_NAMES = {
    LABEL_ROAD: "road",
    LABEL_MARKING: "marking",
    LABEL_OTHER: "other",
}

# Distance buckets (planar ego-frame range), matching Milestone D
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


def project_get_inimage_mask(points_world, camera_pose, intrinsics, W, H):
    """Return a boolean mask over points_world that is True for points
    in front of the camera AND inside the image."""
    M = pose_to_mat(camera_pose)
    Tinv = np.linalg.inv(M)
    pcam = Tinv[:3, :3] @ points_world.T + Tinv[:3, 3:4]
    n = points_world.shape[0]
    front = pcam[2, :] > 0
    K = np.array([
        [intrinsics["fx"], 0.0, intrinsics["cx"]],
        [0.0, intrinsics["fy"], intrinsics["cy"]],
        [0.0, 0.0, 1.0],
    ])
    mask = np.zeros(n, dtype=bool)
    if not front.any():
        return mask
    pcam_f = pcam[:, front]
    uvw = K @ pcam_f
    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]
    in_img = (u > 0) & (u < W) & (v > 0) & (v < H)
    idx_full = np.where(front)[0]
    mask[idx_full[in_img]] = True
    return mask


def load_json(p):
    with p.open() as f:
        return json.load(f)


def planar_range_ego(points_world, lidar_pose):
    M = pose_to_mat(lidar_pose)
    Tinv = np.linalg.inv(M)
    pe = Tinv[:3, :3] @ points_world.T + Tinv[:3, 3:4]
    return np.sqrt(pe[0] ** 2 + pe[1] ** 2)


def empty_counts():
    return {
        "n_total": 0,
        "n_valid": 0,
        "by_class": {LABEL_ROAD: {"n_total": 0, "n_valid": 0},
                     LABEL_MARKING: {"n_total": 0, "n_valid": 0},
                     LABEL_OTHER: {"n_total": 0, "n_valid": 0}},
        "by_bucket_class": {b[0]: {LABEL_ROAD: [0, 0],
                                   LABEL_MARKING: [0, 0],
                                   LABEL_OTHER: [0, 0]}
                            for b in BUCKETS},
        "by_raw_marking": {8: [0, 0], 9: [0, 0], 10: [0, 0]},
    }


def add_counts(agg, cls, raw_class, bucket_idx, rgb_valid):
    n_total = len(cls)
    n_valid = int(rgb_valid.sum())
    agg["n_total"] += n_total
    agg["n_valid"] += n_valid
    for c in (LABEL_ROAD, LABEL_MARKING, LABEL_OTHER):
        m = cls == c
        agg["by_class"][c]["n_total"] += int(m.sum())
        agg["by_class"][c]["n_valid"] += int((m & rgb_valid).sum())
    for bi, (bname, _, _) in enumerate(BUCKETS):
        bmask = bucket_idx == bi
        for c in (LABEL_ROAD, LABEL_MARKING, LABEL_OTHER):
            m = (cls == c) & bmask
            agg["by_bucket_class"][bname][c][0] += int(m.sum())
            agg["by_bucket_class"][bname][c][1] += int((m & rgb_valid).sum())
    for raw_id in (8, 9, 10):
        m = raw_class == raw_id
        agg["by_raw_marking"][raw_id][0] += int(m.sum())
        agg["by_raw_marking"][raw_id][1] += int((m & rgb_valid).sum())


def bucket_assign(dist):
    out = np.full(dist.shape, len(BUCKETS) - 1, dtype=np.int8)
    for bi, (_, lo, hi) in enumerate(BUCKETS):
        mask = (dist >= lo) & (dist < hi)
        out[mask] = bi
    return out


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
        if "d" in df.columns:
            fwd = (df["d"].to_numpy() == 1)
        else:
            fwd = np.ones(len(df), dtype=bool)
        pts_w = df.loc[fwd, ["x", "y", "z"]].to_numpy(dtype=np.float64)
        raw_class = raw_full[fwd]
        cls = remap_classes(raw_class)

        dist = planar_range_ego(pts_w, lid_poses[lf])
        bidx = bucket_assign(dist)

        if frame_valid:
            in_img = project_get_inimage_mask(pts_w, cam_poses[ci], intr,
                                              IMAGE_W, IMAGE_H)
        else:
            in_img = np.zeros(pts_w.shape[0], dtype=bool)
        rgb_valid = in_img & frame_valid

        add_counts(seq_agg, cls, raw_class, bidx, rgb_valid)

        # per-frame compact row
        row = {
            "split": split, "sequence": seq, "lidar_frame": lf,
            "cam_frame": ci, "dt_seconds": dt, "frame_valid": int(frame_valid),
            "n_total_fwd": int(pts_w.shape[0]),
        }
        for c, name in CLASS_NAMES.items():
            m = cls == c
            row[f"n_{name}"] = int(m.sum())
            row[f"n_{name}_valid"] = int((m & rgb_valid).sum())
        for raw_id in (8, 9, 10):
            m = raw_class == raw_id
            row[f"n_raw{raw_id}"] = int(m.sum())
            row[f"n_raw{raw_id}_valid"] = int((m & rgb_valid).sum())
        for bi, (bname, _, _) in enumerate(BUCKETS):
            bmask = bidx == bi
            m_mark = (cls == LABEL_MARKING) & bmask
            row[f"n_marking_{bname}"] = int(m_mark.sum())
            row[f"n_marking_{bname}_valid"] = int((m_mark & rgb_valid).sum())
        frame_rows.append(row)

    return seq_agg, frame_rows


def main():
    t0 = time.time()
    all_frame_rows = []
    all_seq_summaries = []
    splits_aggregate = {s: empty_counts() for s in SPLITS}

    for split, sp in SPLITS.items():
        seqs = [s.strip() for s in sp.read_text().splitlines() if s.strip()]
        for i, seq in enumerate(seqs):
            t_seq = time.time()
            seq_agg, frame_rows = process_sequence(split, seq)
            all_frame_rows.extend(frame_rows)
            elapsed = time.time() - t_seq
            mark_total = seq_agg["by_class"][LABEL_MARKING]["n_total"]
            mark_valid = seq_agg["by_class"][LABEL_MARKING]["n_valid"]
            mark_ratio = (mark_valid / mark_total) if mark_total else 0.0
            print(f"[{split} {i+1}/{len(seqs)}] {seq}  "
                  f"frames={len(frame_rows)}  "
                  f"marking_valid={mark_ratio:.3f}  "
                  f"elapsed={elapsed:.1f}s", flush=True)

            # update split aggregate
            sa = splits_aggregate[split]
            sa["n_total"] += seq_agg["n_total"]
            sa["n_valid"] += seq_agg["n_valid"]
            for c in (LABEL_ROAD, LABEL_MARKING, LABEL_OTHER):
                sa["by_class"][c]["n_total"] += seq_agg["by_class"][c]["n_total"]
                sa["by_class"][c]["n_valid"] += seq_agg["by_class"][c]["n_valid"]
            for bname, _, _ in BUCKETS:
                for c in (LABEL_ROAD, LABEL_MARKING, LABEL_OTHER):
                    sa["by_bucket_class"][bname][c][0] += \
                        seq_agg["by_bucket_class"][bname][c][0]
                    sa["by_bucket_class"][bname][c][1] += \
                        seq_agg["by_bucket_class"][bname][c][1]
            for raw_id in (8, 9, 10):
                sa["by_raw_marking"][raw_id][0] += seq_agg["by_raw_marking"][raw_id][0]
                sa["by_raw_marking"][raw_id][1] += seq_agg["by_raw_marking"][raw_id][1]

            all_seq_summaries.append({
                "split": split, "sequence": seq,
                "n_total": seq_agg["n_total"],
                "n_valid": seq_agg["n_valid"],
                "n_marking": seq_agg["by_class"][LABEL_MARKING]["n_total"],
                "n_marking_valid": seq_agg["by_class"][LABEL_MARKING]["n_valid"],
                "n_road": seq_agg["by_class"][LABEL_ROAD]["n_total"],
                "n_road_valid": seq_agg["by_class"][LABEL_ROAD]["n_valid"],
                "n_other": seq_agg["by_class"][LABEL_OTHER]["n_total"],
                "n_other_valid": seq_agg["by_class"][LABEL_OTHER]["n_valid"],
                "valid_ratio_overall": (seq_agg["n_valid"] / seq_agg["n_total"])
                                       if seq_agg["n_total"] else 0.0,
                "valid_ratio_marking": (seq_agg["by_class"][LABEL_MARKING]["n_valid"]
                                        / seq_agg["by_class"][LABEL_MARKING]["n_total"])
                                       if seq_agg["by_class"][LABEL_MARKING]["n_total"]
                                       else 0.0,
                "valid_ratio_road": (seq_agg["by_class"][LABEL_ROAD]["n_valid"]
                                     / seq_agg["by_class"][LABEL_ROAD]["n_total"])
                                    if seq_agg["by_class"][LABEL_ROAD]["n_total"]
                                    else 0.0,
            })

    # --- write per_frame.csv ---
    fields = list(all_frame_rows[0].keys())
    with (OUT_DIR / "per_frame.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_frame_rows:
            w.writerow(r)

    # --- write aggregate.csv (per split x class x bucket) ---
    with (OUT_DIR / "aggregate.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "bucket", "n_total", "n_valid",
                    "valid_ratio"])
        for split in SPLITS:
            for c, cname in CLASS_NAMES.items():
                for bname, _, _ in BUCKETS:
                    n_tot, n_val = splits_aggregate[split]["by_bucket_class"][bname][c]
                    w.writerow([split, cname, bname, n_tot, n_val,
                                (n_val / n_tot) if n_tot else 0.0])

    # --- marking_subtype.csv ---
    with (OUT_DIR / "marking_subtype.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "raw_id", "raw_name", "n_total", "n_valid",
                    "valid_ratio"])
        names = {8: "lane_line", 9: "stop_line", 10: "other_road_marking"}
        for split in SPLITS:
            for raw_id in (8, 9, 10):
                n_tot, n_val = splits_aggregate[split]["by_raw_marking"][raw_id]
                w.writerow([split, raw_id, names[raw_id], n_tot, n_val,
                            (n_val / n_tot) if n_tot else 0.0])

    # --- per_sequence.csv ---
    seq_fields = list(all_seq_summaries[0].keys())
    with (OUT_DIR / "per_sequence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=seq_fields)
        w.writeheader()
        for r in all_seq_summaries:
            w.writerow(r)

    # --- summary.md ---
    lines = ["# Stratified Valid-RGB Audit", ""]
    elapsed = time.time() - t0
    lines.append(f"Computed in {elapsed:.1f}s over all 6080 frames.")
    lines.append("")
    lines.append("## Overall RGB coverage by split and class")
    lines.append("")
    lines.append("| split | overall valid | road valid | marking valid | other valid |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for split in SPLITS:
        a = splits_aggregate[split]
        def _r(n, d):
            return f"{(n/d):.3f}" if d else "n/a"
        lines.append(
            f"| {split} | {_r(a['n_valid'], a['n_total'])} | "
            f"{_r(a['by_class'][LABEL_ROAD]['n_valid'], a['by_class'][LABEL_ROAD]['n_total'])} | "
            f"{_r(a['by_class'][LABEL_MARKING]['n_valid'], a['by_class'][LABEL_MARKING]['n_total'])} | "
            f"{_r(a['by_class'][LABEL_OTHER]['n_valid'], a['by_class'][LABEL_OTHER]['n_total'])} |"
        )
    lines.append("")

    lines.append("## Marking valid ratio by distance bucket")
    lines.append("")
    header = "| split |" + "".join(f" {b[0]} |" for b in BUCKETS)
    sep = "| --- |" + "".join(" ---: |" for _ in BUCKETS)
    lines.append(header)
    lines.append(sep)
    for split in SPLITS:
        a = splits_aggregate[split]
        cells = []
        for bname, _, _ in BUCKETS:
            n_tot, n_val = a["by_bucket_class"][bname][LABEL_MARKING]
            cells.append(f" {(n_val/n_tot):.3f}" if n_tot else " n/a")
        lines.append(f"| {split} |" + " |".join(cells) + " |")
    lines.append("")

    lines.append("## Marking valid by subtype (raw 8 / 9 / 10)")
    lines.append("")
    lines.append("| split | raw 8 lane line | raw 9 stop line | raw 10 other road marking |")
    lines.append("| --- | ---: | ---: | ---: |")
    for split in SPLITS:
        a = splits_aggregate[split]
        cells = []
        for raw_id in (8, 9, 10):
            n_tot, n_val = a["by_raw_marking"][raw_id]
            cells.append(f"{(n_val/n_tot):.3f}" if n_tot else "n/a")
        lines.append(f"| {split} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")

    # Sequence outliers: marking valid < 0.6
    lines.append("## Sequences with marking valid ratio < 0.6")
    lines.append("")
    bad = [s for s in all_seq_summaries
           if s["n_marking"] > 0 and s["valid_ratio_marking"] < 0.6]
    if bad:
        lines.append("| split | sequence | n_marking | marking valid | overall valid |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for s in sorted(bad, key=lambda x: x["valid_ratio_marking"]):
            lines.append(
                f"| {s['split']} | {s['sequence']} | "
                f"{s['n_marking']} | {s['valid_ratio_marking']:.3f} | "
                f"{s['valid_ratio_overall']:.3f} |"
            )
    else:
        lines.append("None.")
    lines.append("")

    # Frame-level cost of the timestamp policy
    n_frames_total = len(all_frame_rows)
    n_frames_invalid = sum(1 for r in all_frame_rows if r["frame_valid"] == 0)
    lines.append("## Cost of the timestamp policy")
    lines.append("")
    lines.append(f"- frames total: {n_frames_total}")
    lines.append(f"- frames invalidated (|dt| > 60 ms): {n_frames_invalid} "
                 f"({100*n_frames_invalid/n_frames_total:.2f}%)")
    lines.append("")

    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_DIR / 'per_frame.csv'}")
    print(f"Wrote {OUT_DIR / 'aggregate.csv'}")
    print(f"Wrote {OUT_DIR / 'marking_subtype.csv'}")
    print(f"Wrote {OUT_DIR / 'per_sequence.csv'}")
    print(f"Wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
