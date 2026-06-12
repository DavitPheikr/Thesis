#!/usr/bin/env python
"""Milestone E step 1: front-camera inventory check.

For every sequence in configs/splits/{train,val,test}.txt, verify the
front-camera assets are present and consistent with the LiDAR sweep.

This script reads only the JSON sidecars and directory listings. It does NOT
load any JPEG image data or any LiDAR pickle. One JPEG header per sequence is
opened lazily to read the image size; no pixels are decoded. Memory footprint
is tiny.

Outputs:
- inventory.csv  (one row per split sequence)
- summary.md     (counts and any failures)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Locate repo root from this script's path:
# logs/milestone_e/dataset_analysis/front_camera_inventory/analysis_code/<this>
REPO_ROOT = Path(__file__).resolve().parents[5]

DATASET_ROOT_FILE = REPO_ROOT / "logs" / "dataset_root.txt"
SPLITS = {
    "train": REPO_ROOT / "configs" / "splits" / "train.txt",
    "val":   REPO_ROOT / "configs" / "splits" / "val.txt",
    "test":  REPO_ROOT / "configs" / "splits" / "test.txt",
}
OUT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = OUT_DIR / "inventory.csv"
MD_PATH = OUT_DIR / "summary.md"


def read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def read_image_size(jpg_path: Path) -> tuple[int, int]:
    # Lazy header read; does not decode pixels.
    from PIL import Image
    with Image.open(jpg_path) as img:
        return img.size  # (W, H)


def inspect(root: Path, seq_id: str) -> dict:
    row: dict = {"sequence": seq_id, "ok": True, "error": ""}
    seq_dir = root / seq_id
    fc_dir = seq_dir / "camera" / "front_camera"
    lid_dir = seq_dir / "lidar"

    try:
        if not fc_dir.is_dir():
            raise FileNotFoundError(f"missing front_camera dir: {fc_dir}")
        if not lid_dir.is_dir():
            raise FileNotFoundError(f"missing lidar dir: {lid_dir}")

        jpgs = sorted(fc_dir.glob("*.jpg"))
        intrinsics_path = fc_dir / "intrinsics.json"
        cam_poses_path = fc_dir / "poses.json"
        cam_ts_path = fc_dir / "timestamps.json"
        lid_ts_path = lid_dir / "timestamps.json"
        lid_poses_path = lid_dir / "poses.json"
        lid_pkls = sorted(lid_dir.glob("*.pkl"))

        for required in (intrinsics_path, cam_poses_path, cam_ts_path,
                         lid_ts_path, lid_poses_path):
            if not required.is_file():
                raise FileNotFoundError(f"missing {required}")

        with intrinsics_path.open() as f:
            intr = json.load(f)
        with cam_poses_path.open() as f:
            cam_poses = json.load(f)
        with cam_ts_path.open() as f:
            cam_ts = json.load(f)
        with lid_ts_path.open() as f:
            lid_ts = json.load(f)
        with lid_poses_path.open() as f:
            lid_poses = json.load(f)

        n_img = len(jpgs)
        n_pkl = len(lid_pkls)
        n_cam_pose = len(cam_poses)
        n_cam_ts = len(cam_ts)
        n_lid_ts = len(lid_ts)
        n_lid_pose = len(lid_poses)

        if n_img == 0:
            raise RuntimeError("no jpg files in front_camera dir")
        img_w, img_h = read_image_size(jpgs[0])

        row.update({
            "num_images": n_img,
            "num_lidar_pkls": n_pkl,
            "num_cam_poses": n_cam_pose,
            "num_cam_timestamps": n_cam_ts,
            "num_lid_poses": n_lid_pose,
            "num_lid_timestamps": n_lid_ts,
            "image_width": int(img_w),
            "image_height": int(img_h),
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
            "cam_ts_first": float(cam_ts[0]),
            "cam_ts_last": float(cam_ts[-1]),
            "lid_ts_first": float(lid_ts[0]),
            "lid_ts_last": float(lid_ts[-1]),
        })

        # Consistency: image count == cam pose == cam ts == lid pkl == lid pose == lid ts.
        counts = {
            "img": n_img, "pkl": n_pkl,
            "cam_pose": n_cam_pose, "cam_ts": n_cam_ts,
            "lid_pose": n_lid_pose, "lid_ts": n_lid_ts,
        }
        if len(set(counts.values())) != 1:
            raise ValueError(f"count mismatch: {counts}")

    except Exception as exc:  # noqa: BLE001
        row["ok"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"

    return row


def main() -> None:
    if not DATASET_ROOT_FILE.is_file():
        print(f"ERROR: dataset_root file not found: {DATASET_ROOT_FILE}",
              file=sys.stderr)
        sys.exit(2)
    root = Path(DATASET_ROOT_FILE.read_text().strip())
    if not root.is_dir():
        print(f"ERROR: dataset_root is not a directory: {root}",
              file=sys.stderr)
        sys.exit(2)

    rows: list[dict] = []
    per_split: dict[str, dict] = {}
    for split_name, split_path in SPLITS.items():
        if not split_path.is_file():
            print(f"WARN: split file missing: {split_path}")
            continue
        seqs = read_lines(split_path)
        n_ok = n_fail = 0
        for seq_id in seqs:
            print(f"[{split_name}] {seq_id} ...", flush=True)
            row = inspect(root, seq_id)
            row["split"] = split_name
            rows.append(row)
            if row["ok"]:
                n_ok += 1
            else:
                n_fail += 1
                print(f"  FAIL: {row['error']}")
        per_split[split_name] = {
            "total": len(seqs), "ok": n_ok, "fail": n_fail,
        }

    fields = [
        "split", "sequence", "ok", "error",
        "num_images", "num_lidar_pkls",
        "num_cam_poses", "num_cam_timestamps",
        "num_lid_poses", "num_lid_timestamps",
        "image_width", "image_height",
        "fx", "fy", "cx", "cy",
        "cam_ts_first", "cam_ts_last",
        "lid_ts_first", "lid_ts_last",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    lines: list[str] = []
    lines.append("# Front-Camera Inventory Summary")
    lines.append("")
    lines.append("## Per-split counts")
    lines.append("")
    lines.append("| split | total | ok | fail |")
    lines.append("| --- | ---: | ---: | ---: |")
    for s, c in per_split.items():
        lines.append(f"| {s} | {c['total']} | {c['ok']} | {c['fail']} |")
    lines.append("")
    failed = [r for r in rows if not r["ok"]]
    lines.append(f"## Failures: {len(failed)}")
    lines.append("")
    if failed:
        lines.append("| split | sequence | error |")
        lines.append("| --- | --- | --- |")
        for r in failed:
            lines.append(f"| {r['split']} | {r['sequence']} | {r['error']} |")
    else:
        lines.append("None.")
    lines.append("")

    ok_rows = [r for r in rows if r["ok"]]
    if ok_rows:
        widths = sorted({r["image_width"] for r in ok_rows})
        heights = sorted({r["image_height"] for r in ok_rows})
        nimgs = sorted({r["num_images"] for r in ok_rows})
        fxs = sorted({round(r["fx"], 3) for r in ok_rows})
        lines.append("## Consistency across ok sequences")
        lines.append("")
        lines.append(f"- distinct image widths: {widths}")
        lines.append(f"- distinct image heights: {heights}")
        lines.append(f"- distinct num_images per sequence: {nimgs}")
        lines.append(f"- distinct fx values (3dp): {fxs}")
        lines.append("")
        lines.append("## First/last timestamps (camera vs lidar) -- first 8 ok sequences")
        lines.append("")
        lines.append(
            "| split | seq | cam_first | cam_last | lid_first | lid_last "
            "| dt_first(s) | dt_last(s) |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for r in ok_rows[:8]:
            dt_first = r["cam_ts_first"] - r["lid_ts_first"]
            dt_last = r["cam_ts_last"] - r["lid_ts_last"]
            lines.append(
                f"| {r['split']} | {r['sequence']} | "
                f"{r['cam_ts_first']:.3f} | {r['cam_ts_last']:.3f} | "
                f"{r['lid_ts_first']:.3f} | {r['lid_ts_last']:.3f} | "
                f"{dt_first:+.4f} | {dt_last:+.4f} |"
            )
        lines.append("")

    MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
