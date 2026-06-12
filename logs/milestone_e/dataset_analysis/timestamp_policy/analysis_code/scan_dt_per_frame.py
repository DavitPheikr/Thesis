#!/usr/bin/env python
"""Milestone E step 4: scan dt for every lidar frame in every split sequence.

For each (sequence, lidar_frame), compute:
  - same-index dt        = cam_ts[lidar_frame] - lid_ts[lidar_frame]
  - nearest-ts cam frame = argmin |cam_ts - lid_ts[lidar_frame]|
  - nearest-ts dt        = cam_ts[nearest] - lid_ts[lidar_frame]

Then report, per candidate dt threshold T:
  - how many lidar frames pass under T using nearest-ts lookup
  - how many sequences have any failing frame
  - per-sequence frame failure counts

Pure JSON read; runs in seconds, no devkit, no images, no lidar pickles.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
DATASET_ROOT = Path((REPO_ROOT / "logs" / "dataset_root.txt").read_text().strip())
OUT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = OUT_DIR / "dt_per_frame.csv"
SEQ_CSV_PATH = OUT_DIR / "dt_per_sequence_summary.csv"
MD_PATH = OUT_DIR / "scan_summary.md"

SPLITS = {
    "train": REPO_ROOT / "configs" / "splits" / "train.txt",
    "val":   REPO_ROOT / "configs" / "splits" / "val.txt",
    "test":  REPO_ROOT / "configs" / "splits" / "test.txt",
}
THRESHOLDS_MS = [40, 60, 80, 100, 150, 200, 300, 500]


def load_json(p):
    with p.open() as f:
        return json.load(f)


def nearest_index(target, arr):
    best_i = 0
    best_d = abs(arr[0] - target)
    for i in range(1, len(arr)):
        d = abs(arr[i] - target)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def main():
    rows = []
    for split, sp in SPLITS.items():
        for seq in [s.strip() for s in sp.read_text().splitlines() if s.strip()]:
            fc = DATASET_ROOT / seq / "camera" / "front_camera"
            lid = DATASET_ROOT / seq / "lidar"
            cam_ts = load_json(fc / "timestamps.json")
            lid_ts = load_json(lid / "timestamps.json")
            n = min(len(cam_ts), len(lid_ts))
            for li in range(n):
                tgt = lid_ts[li]
                same_dt = cam_ts[li] - tgt if li < len(cam_ts) else float("nan")
                ni = nearest_index(tgt, cam_ts)
                near_dt = cam_ts[ni] - tgt
                rows.append({
                    "split": split,
                    "sequence": seq,
                    "lidar_frame": li,
                    "same_index_cam_frame": li,
                    "same_index_dt_s": same_dt,
                    "nearest_cam_frame": ni,
                    "nearest_dt_s": near_dt,
                })

    with CSV_PATH.open("w", newline="") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Per-threshold pass counts (using nearest-ts lookup)
    threshold_pass = {}
    for T_ms in THRESHOLDS_MS:
        T = T_ms / 1000.0
        passed = sum(1 for r in rows if abs(r["nearest_dt_s"]) <= T)
        threshold_pass[T_ms] = passed
    total = len(rows)

    # Per-sequence failing-frame counts at 60ms (our default candidate)
    T_default = 0.060
    seq_fail_counts = {}
    for r in rows:
        key = (r["split"], r["sequence"])
        seq_fail_counts.setdefault(key, 0)
        if abs(r["nearest_dt_s"]) > T_default:
            seq_fail_counts[key] += 1

    with SEQ_CSV_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "sequence", "frames_failing_at_60ms",
                    "frames_total", "fail_pct"])
        for (split, seq), fail in sorted(seq_fail_counts.items()):
            w.writerow([split, seq, fail, 80, 100.0 * fail / 80])

    lines = ["# Per-Frame dt Scan", ""]
    lines.append(f"Total frames scanned (76 sequences x 80 frames): {total}.")
    lines.append("")
    lines.append("## Frames passing under each candidate threshold "
                 "(nearest-timestamp camera lookup)")
    lines.append("")
    lines.append("| threshold | frames pass | frames fail | pass % |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for T_ms in THRESHOLDS_MS:
        p = threshold_pass[T_ms]
        lines.append(
            f"| {T_ms} ms | {p} | {total - p} | {100.0 * p / total:.2f}% |"
        )
    lines.append("")

    lines.append("## Sequences with any frame failing at the 60 ms candidate")
    lines.append("")
    fail_seqs = [(s, q, c) for (s, q), c in seq_fail_counts.items() if c > 0]
    if fail_seqs:
        lines.append("| split | sequence | failing frames / 80 | % |")
        lines.append("| --- | --- | ---: | ---: |")
        for split, seq, c in sorted(fail_seqs):
            lines.append(f"| {split} | {seq} | {c} | {100*c/80:.1f}% |")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("## Same-index vs nearest-timestamp")
    lines.append("")
    n_same_eq_nearest = sum(1 for r in rows
                            if r["same_index_cam_frame"] == r["nearest_cam_frame"])
    lines.append(f"- frames where same_index == nearest_ts: {n_same_eq_nearest} "
                 f"({100*n_same_eq_nearest/total:.2f}%)")
    lines.append(f"- frames where they differ: {total - n_same_eq_nearest}")
    if total - n_same_eq_nearest > 0:
        diff_rows = [r for r in rows
                     if r["same_index_cam_frame"] != r["nearest_cam_frame"]]
        seqs_diff = sorted({(r["split"], r["sequence"]) for r in diff_rows})
        lines.append("")
        lines.append("Sequences where the two policies disagree on at least one frame:")
        lines.append("")
        lines.append("| split | sequence | frames disagreeing |")
        lines.append("| --- | --- | ---: |")
        for s, q in seqs_diff:
            n_diff = sum(1 for r in diff_rows
                         if r["split"] == s and r["sequence"] == q)
            lines.append(f"| {s} | {q} | {n_diff} |")
    lines.append("")

    MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {SEQ_CSV_PATH}")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
