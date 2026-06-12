#!/usr/bin/env python
"""Milestone E step 3: per-sequence projection audit.

For every sequence in the train/val/test splits, project ONE frame (default
frame index 40, which is mid-sequence) and compute:
- overall valid-RGB ratio (projected / forward-LiDAR points)
- valid ratio for the marking class only
- valid ratio for the road class only
- marking-point absolute counts (forward LiDAR vs projected)
- chosen camera frame index, dt seconds

The output is a CSV with one row per sequence and a markdown summary that
flags outliers. Goal: catch any sequence where projection is silently
broken (e.g. pose convention mismatch) before we proceed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[5]
DATASET_ROOT = Path((REPO_ROOT / "logs" / "dataset_root.txt").read_text().strip())
OUT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = OUT_DIR / "per_sequence_audit.csv"
MD_PATH = OUT_DIR / "per_sequence_audit.md"

SPLIT_PATHS = {
    "train": REPO_ROOT / "configs" / "splits" / "train.txt",
    "val":   REPO_ROOT / "configs" / "splits" / "val.txt",
    "test":  REPO_ROOT / "configs" / "splits" / "test.txt",
}
FRAME_IDX = 40
DT_OK_S = 0.06   # |dt| <= 60 ms is considered usable.

IGNORE_RAW = {1, 2, 3, 4}
ROAD_RAW = {7}
MARKING_RAW = {8, 9, 10}
LABEL_IGNORE, LABEL_ROAD, LABEL_MARKING, LABEL_OTHER = 0, 1, 2, 3


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
    return idx[in_img]


def load_json(p):
    with p.open() as f:
        return json.load(f)


def audit_one(seq_id):
    seq_dir = DATASET_ROOT / seq_id
    fc_dir = seq_dir / "camera" / "front_camera"

    intr = load_json(fc_dir / "intrinsics.json")
    cam_poses = load_json(fc_dir / "poses.json")
    cam_ts = load_json(fc_dir / "timestamps.json")
    lid_ts = load_json(seq_dir / "lidar" / "timestamps.json")

    lid_frame = FRAME_IDX
    target = lid_ts[lid_frame]
    diffs = [abs(c - target) for c in cam_ts]
    ci = int(np.argmin(diffs))
    dt = cam_ts[ci] - target

    img_path = fc_dir / f"{ci:02d}.jpg"
    with Image.open(img_path) as im:
        W, H = im.size

    lidar_pkl = seq_dir / "lidar" / f"{lid_frame:02d}.pkl"
    semseg_pkl = seq_dir / "annotations" / "semseg" / f"{lid_frame:02d}.pkl"
    df = pd.read_pickle(lidar_pkl)
    sem = pd.read_pickle(semseg_pkl)
    raw_class = sem["class"].to_numpy()
    fwd = (df["d"].to_numpy() == 1) if "d" in df.columns else np.ones(len(df), bool)
    pts = df.loc[fwd, ["x", "y", "z"]].to_numpy(dtype=np.float64)
    cls = remap_classes(raw_class[fwd])

    proj_idx = project(pts, cam_poses[ci], intr, W, H)
    proj_cls = cls[proj_idx]

    n_fwd = int(fwd.sum())
    n_active = int((cls != LABEL_IGNORE).sum())
    n_road = int((cls == LABEL_ROAD).sum())
    n_mark = int((cls == LABEL_MARKING).sum())
    n_other = int((cls == LABEL_OTHER).sum())
    n_proj = int(proj_idx.shape[0])
    n_road_proj = int((proj_cls == LABEL_ROAD).sum())
    n_mark_proj = int((proj_cls == LABEL_MARKING).sum())
    n_other_proj = int((proj_cls == LABEL_OTHER).sum())

    return {
        "sequence": seq_id,
        "lid_frame": lid_frame,
        "cam_frame": ci,
        "dt_seconds": dt,
        "dt_ok": abs(dt) <= DT_OK_S,
        "n_forward_lidar": n_fwd,
        "n_active": n_active,
        "n_road": n_road,
        "n_marking": n_mark,
        "n_other": n_other,
        "n_projected": n_proj,
        "n_road_projected": n_road_proj,
        "n_marking_projected": n_mark_proj,
        "n_other_projected": n_other_proj,
        "valid_ratio_overall": (n_proj / n_fwd) if n_fwd else 0.0,
        "valid_ratio_road": (n_road_proj / n_road) if n_road else 0.0,
        "valid_ratio_marking": (n_mark_proj / n_mark) if n_mark else 0.0,
        "valid_ratio_other": (n_other_proj / n_other) if n_other else 0.0,
    }


def main():
    rows = []
    for split, sp in SPLIT_PATHS.items():
        seqs = [s.strip() for s in sp.read_text().splitlines() if s.strip()]
        for seq in seqs:
            try:
                r = audit_one(seq)
                r["split"] = split
                r["error"] = ""
            except Exception as exc:  # noqa: BLE001
                r = {"sequence": seq, "split": split, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[{split}] {seq}  "
                  f"overall={r.get('valid_ratio_overall', float('nan')):.3f}  "
                  f"marking={r.get('valid_ratio_marking', float('nan')):.3f}  "
                  f"dt={r.get('dt_seconds', float('nan')):+.4f}  "
                  f"{r.get('error', '')}",
                  flush=True)
            rows.append(r)

    fields = [
        "split", "sequence", "error", "lid_frame", "cam_frame",
        "dt_seconds", "dt_ok",
        "n_forward_lidar", "n_active",
        "n_road", "n_marking", "n_other",
        "n_projected", "n_road_projected", "n_marking_projected",
        "n_other_projected",
        "valid_ratio_overall", "valid_ratio_road", "valid_ratio_marking",
        "valid_ratio_other",
    ]
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    ok_rows = [r for r in rows if not r.get("error")]
    failed = [r for r in rows if r.get("error")]
    overall = np.array([r["valid_ratio_overall"] for r in ok_rows])
    marking = np.array([r["valid_ratio_marking"] for r in ok_rows
                        if r["n_marking"] > 0])
    road = np.array([r["valid_ratio_road"] for r in ok_rows
                     if r["n_road"] > 0])
    n_dt_bad = sum(1 for r in ok_rows if not r["dt_ok"])
    outliers_marking = [r for r in ok_rows
                        if r["n_marking"] > 0 and r["valid_ratio_marking"] < 0.6]

    lines = ["# Per-Sequence Projection Audit", ""]
    lines.append(f"Frame audited per sequence: index `{FRAME_IDX}` "
                 f"(camera frame chosen by nearest timestamp).")
    lines.append("")
    lines.append(f"## Coverage across {len(ok_rows)} ok sequences "
                 f"(of {len(rows)})")
    lines.append("")
    lines.append("| metric | mean | min | p10 | median | p90 | max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    if overall.size:
        lines.append(_stat_row("overall valid", overall))
    if road.size:
        lines.append(_stat_row("road valid", road))
    if marking.size:
        lines.append(_stat_row("marking valid", marking))
    lines.append("")

    lines.append(f"## Timestamp policy outcomes")
    lines.append("")
    lines.append(f"- sequences with |dt| > {DT_OK_S} s at frame {FRAME_IDX}: "
                 f"**{n_dt_bad}**")
    bad_dt = [r for r in ok_rows if not r["dt_ok"]]
    if bad_dt:
        lines.append("")
        lines.append("| split | sequence | dt(s) |")
        lines.append("| --- | --- | ---: |")
        for r in bad_dt:
            lines.append(f"| {r['split']} | {r['sequence']} | "
                         f"{r['dt_seconds']:+.4f} |")
    lines.append("")

    lines.append(f"## Sequences with marking valid ratio < 0.6")
    lines.append("")
    if outliers_marking:
        lines.append("| split | sequence | marking valid | overall valid "
                     "| n_marking |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for r in sorted(outliers_marking, key=lambda x: x["valid_ratio_marking"]):
            lines.append(
                f"| {r['split']} | {r['sequence']} | "
                f"{r['valid_ratio_marking']:.3f} | "
                f"{r['valid_ratio_overall']:.3f} | "
                f"{r['n_marking']} |"
            )
    else:
        lines.append("None.")
    lines.append("")

    if failed:
        lines.append(f"## Failures ({len(failed)})")
        lines.append("")
        lines.append("| split | sequence | error |")
        lines.append("| --- | --- | --- |")
        for r in failed:
            lines.append(f"| {r['split']} | {r['sequence']} | {r['error']} |")
        lines.append("")

    MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")


def _stat_row(name, arr):
    return (
        f"| {name} | {arr.mean():.3f} | {arr.min():.3f} | "
        f"{np.percentile(arr,10):.3f} | {np.median(arr):.3f} | "
        f"{np.percentile(arr,90):.3f} | {arr.max():.3f} |"
    )


if __name__ == "__main__":
    main()
