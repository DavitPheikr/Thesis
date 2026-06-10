"""Dataset-level intensity analysis for the Milestone D road_marking3 remap.

This script is intentionally stored under logs/milestone_d because it is part
of the Milestone D thesis evidence, not a reusable project command. It analyzes
the forward-facing PandaSet LiDAR under the current remap:

  road    = raw 7
  marking = raw 8 + raw 9 + raw 10
  other   = all other non-ignored raw classes

The outputs answer whether road markings are separable from road/other by raw
LiDAR intensity, both globally and by distance bucket.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs/splits").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError("Could not find project root from analysis script path")


PROJECT_ROOT = find_project_root()
PATCHED_DEVKIT = PROJECT_ROOT / "pandaset-devkit/python"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PATCHED_DEVKIT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pandaset import DataSet, geometry as pds_geometry

from thesis_pipeline.adapters.pandaset_ff_lane3 import (
    LABEL_MODE_ROAD_MARKING3,
    RAW_OTHER_ROAD_MARKING_ID,
    RAW_ROAD_ID,
    RAW_ROAD_MARKING_IDS,
    RAW_STOP_LINE_ID,
    remap_raw_pandaset_ids,
)
from thesis_pipeline.core.pandaset_compat import get_frame_count


DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "logs/milestone_d/dataset_analysis/road_marking3_intensity"
)
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "configs/splits"
DEFAULT_DATASET_ROOT_FILE = PROJECT_ROOT / "logs/dataset_root.txt"
DEFAULT_STATS_FILE = PROJECT_ROOT / "logs/milestone_d/road_marking3_training_statistics.json"
DEFAULT_PREFLIGHT_PATTERN_FILE = (
    PROJECT_ROOT / "logs/milestone_b_preflight_sensor_pattern.txt"
)

FORWARD_SENSOR_ID = 1
CLASS_DEFS = {
    1: {
        "name": "road",
        "raw_definition": "raw 7 Road",
        "color": "#5f6368",
    },
    2: {
        "name": "marking",
        "raw_definition": "raw 8 Lane Line Marking + raw 9 Stop Line Marking + raw 10 Other Road Marking",
        "color": "#1b9e77",
    },
    3: {
        "name": "other",
        "raw_definition": "all other non-ignored raw classes",
        "color": "#7570b3",
    },
}
CLASS_ORDER = (1, 2, 3)
SPLIT_ORDER = ("training", "validation", "test")
DISTANCE_BUCKETS = (
    (0.0, 10.0, "0_10m"),
    (10.0, 20.0, "10_20m"),
    (20.0, 30.0, "20_30m"),
    (30.0, 40.0, "30_40m"),
    (40.0, 60.0, "40_60m"),
    (60.0, np.inf, "60m_plus"),
)
HIST_BINS = np.arange(-0.5, 256.5, 1.0)
HIST_VALUES = np.arange(256, dtype=np.float64)
EXPECTED_TRAIN_COUNTS = {
    "road": 119_562_394,
    "marking": 5_129_328,
    "other": 173_473_484,
}


@dataclass
class HistStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float | None = None
    max_value: float | None = None
    hist: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.int64))

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return
        clipped_for_hist = np.clip(values, 0.0, 255.0)
        hist, _ = np.histogram(clipped_for_hist, bins=HIST_BINS)
        self.hist += hist.astype(np.int64, copy=False)
        self.count += int(values.size)
        self.total += float(values.sum())
        self.total_sq += float(np.square(values).sum())
        value_min = float(values.min())
        value_max = float(values.max())
        self.min_value = value_min if self.min_value is None else min(self.min_value, value_min)
        self.max_value = value_max if self.max_value is None else max(self.max_value, value_max)

    def percentile(self, q: float) -> float | None:
        if self.count == 0:
            return None
        target = (q / 100.0) * max(self.count - 1, 0)
        cumulative = np.cumsum(self.hist)
        idx = int(np.searchsorted(cumulative, target + 1, side="left"))
        idx = min(max(idx, 0), len(HIST_VALUES) - 1)
        return float(HIST_VALUES[idx])

    def row(self, extra: dict[str, object]) -> dict[str, object]:
        row = dict(extra)
        if self.count == 0:
            row.update(
                {
                    "count": 0,
                    "share": "",
                    "mean": "",
                    "std": "",
                    "min": "",
                    "p05": "",
                    "p25": "",
                    "median": "",
                    "p75": "",
                    "p95": "",
                    "max": "",
                }
            )
            return row
        mean = self.total / self.count
        var = max(0.0, self.total_sq / self.count - mean * mean)
        row.update(
            {
                "count": self.count,
                "share": "",
                "mean": mean,
                "std": float(np.sqrt(var)),
                "min": self.min_value,
                "p05": self.percentile(5),
                "p25": self.percentile(25),
                "median": self.percentile(50),
                "p75": self.percentile(75),
                "p95": self.percentile(95),
                "max": self.max_value,
            }
        )
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-root-file", type=Path, default=DEFAULT_DATASET_ROOT_FILE)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--stats-file", type=Path, default=DEFAULT_STATS_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--sequence-ids", nargs="*", default=None)
    parser.add_argument("--skip-count-validation", action="store_true")
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required split file: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_dataset_root(dataset_root: Path | None, dataset_root_file: Path) -> Path:
    if dataset_root is not None:
        return dataset_root
    if not dataset_root_file.exists():
        raise FileNotFoundError(f"Missing dataset root file: {dataset_root_file}")
    root = Path(dataset_root_file.read_text().strip())
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def read_split_ids(split_dir: Path) -> dict[str, list[str]]:
    return {
        "training": read_lines(split_dir / "train.txt"),
        "validation": read_lines(split_dir / "val.txt"),
        "test": read_lines(split_dir / "test.txt"),
    }


def sensor_reapply_required() -> bool:
    if not DEFAULT_PREFLIGHT_PATTERN_FILE.exists():
        return False
    return "reapply_per_frame: true" in DEFAULT_PREFLIGHT_PATTERN_FILE.read_text()


def build_sequence_plan(
    split_ids: dict[str, list[str]],
    sequence_ids: list[str] | None,
    max_sequences: int | None,
) -> list[tuple[str, str]]:
    requested = {seq.zfill(3) for seq in sequence_ids} if sequence_ids else None
    plan: list[tuple[str, str]] = []
    for split in SPLIT_ORDER:
        for seq_id in split_ids[split]:
            if requested is not None and seq_id not in requested:
                continue
            plan.append((split, seq_id))
    if max_sequences is not None:
        plan = plan[:max_sequences]
    return plan


def bucket_mask(ranges: np.ndarray, low: float, high: float) -> np.ndarray:
    if np.isinf(high):
        return ranges >= low
    return (ranges >= low) & (ranges < high)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stat_row_fields(prefix: Iterable[str] = ()) -> list[str]:
    return [
        *prefix,
        "count",
        "share",
        "mean",
        "std",
        "min",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "max",
    ]


def add_shares(rows: list[dict[str, object]], group_keys: tuple[str, ...]) -> None:
    totals: dict[tuple[object, ...], int] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        totals[key] = totals.get(key, 0) + int(row["count"] or 0)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        total = totals.get(key, 0)
        row["share"] = "" if total == 0 else int(row["count"]) / total


def plot_class_histograms(stats_by_class: dict[int, HistStats], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for class_id in CLASS_ORDER:
        stats = stats_by_class[class_id]
        if stats.count == 0:
            continue
        density = stats.hist / max(1, int(stats.hist.sum()))
        cls = CLASS_DEFS[class_id]
        ax.plot(
            HIST_VALUES,
            density,
            color=str(cls["color"]),
            linewidth=2.2,
            label=f"{cls['name']} (n={stats.count:,})",
        )
    ax.set_title("Milestone D remapped class intensity distributions")
    ax.set_xlabel("raw LiDAR intensity")
    ax.set_ylabel("density")
    ax.set_xlim(0, 114)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_distance_medians(rows: list[dict[str, object]], out_path: Path) -> None:
    labels = [label for _lo, _hi, label in DISTANCE_BUCKETS]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    for class_id in CLASS_ORDER:
        cls = CLASS_DEFS[class_id]
        medians: list[float] = []
        p25: list[float] = []
        p75: list[float] = []
        for label in labels:
            match = next(
                row for row in rows if row["class_id"] == class_id and row["bucket"] == label
            )
            medians.append(float(match["median"]) if match["median"] != "" else np.nan)
            p25.append(float(match["p25"]) if match["p25"] != "" else np.nan)
            p75.append(float(match["p75"]) if match["p75"] != "" else np.nan)
        y = np.asarray(medians, dtype=np.float64)
        lo = np.asarray(p25, dtype=np.float64)
        hi = np.asarray(p75, dtype=np.float64)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.4,
            color=str(cls["color"]),
            label=str(cls["name"]),
        )
        ax.fill_between(x, lo, hi, color=str(cls["color"]), alpha=0.14)
    ax.set_title("Median raw intensity by distance bucket")
    ax.set_xlabel("range bucket (m)")
    ax.set_ylabel("raw LiDAR intensity")
    ax.set_xticks(x, [label.replace("_", "-").replace("m", "") for label in labels])
    ax.set_xlim(-0.2, len(labels) - 0.8)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="remapped class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def process_frame(
    seq,
    seq_id: str,
    frame_idx: int,
    split: str,
    sensor_reapply: bool,
    stats_by_class: dict[int, HistStats],
    stats_by_split_class: dict[tuple[str, int], HistStats],
    stats_by_bucket: dict[tuple[int, str], HistStats],
) -> dict[str, object]:
    seq.lidar.set_sensor(FORWARD_SENSOR_ID)
    if sensor_reapply:
        seq.lidar.set_sensor(FORWARD_SENSOR_ID)

    pc_df = seq.lidar[frame_idx]
    semseg_df = seq.semseg[frame_idx]
    raw_labels = semseg_df.loc[pc_df.index, "class"].to_numpy(dtype=np.int32)
    remapped = remap_raw_pandaset_ids(raw_labels, label_mode=LABEL_MODE_ROAD_MARKING3)
    unexpected = sorted(set(np.unique(remapped).tolist()) - {0, 1, 2, 3})
    if unexpected:
        raise RuntimeError(f"Unexpected remapped class ids in {seq_id}/{frame_idx}: {unexpected}")

    intensity = pc_df["i"].to_numpy(dtype=np.float32)
    xyz_world = pc_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    xyz_ego = pds_geometry.lidar_points_to_ego(
        xyz_world, seq.lidar.poses[frame_idx]
    ).astype(np.float32, copy=False)
    ranges = np.linalg.norm(xyz_ego[:, :3], axis=1)

    row: dict[str, object] = {
        "split": split,
        "seq_id": seq_id,
        "frame_idx": frame_idx,
        "active_points_forward_sensor": int(raw_labels.size),
    }
    for class_id in CLASS_ORDER:
        class_name = str(CLASS_DEFS[class_id]["name"])
        mask = remapped == class_id
        values = intensity[mask]
        stats_by_class[class_id].update(values)
        stats_by_split_class[(split, class_id)].update(values)
        row[f"{class_name}_count"] = int(mask.sum())
        class_ranges = ranges[mask]
        class_values = intensity[mask]
        for low, high, label in DISTANCE_BUCKETS:
            stats_by_bucket[(class_id, label)].update(class_values[bucket_mask(class_ranges, low, high)])

    row["ignored_count"] = int((remapped == 0).sum())
    row["raw_8_count"] = int((raw_labels == 8).sum())
    row["raw_9_count"] = int((raw_labels == RAW_STOP_LINE_ID).sum())
    row["raw_10_count"] = int((raw_labels == RAW_OTHER_ROAD_MARKING_ID).sum())
    return row


def validate_training_counts(
    rows_by_split: list[dict[str, object]],
    stats_file: Path,
    skip: bool,
) -> dict[str, object]:
    if skip:
        return {"skipped": True}
    if not stats_file.exists():
        raise FileNotFoundError(f"Missing Milestone D statistics file: {stats_file}")
    stats = json.loads(stats_file.read_text())
    if stats.get("label_mode") != LABEL_MODE_ROAD_MARKING3:
        raise RuntimeError(f"Unexpected stats label_mode: {stats.get('label_mode')!r}")
    expected = stats["class_counts"]
    by_name = {
        str(row["class_name"]): int(row["count"])
        for row in rows_by_split
        if row["split"] == "training"
    }
    expected_by_name = {
        "road": int(expected["road"]),
        "marking": int(expected["marking"]),
        "other": int(expected["other"]),
    }
    if by_name != expected_by_name:
        raise RuntimeError(
            "Training class counts do not match road_marking3 statistics file: "
            f"computed={by_name} expected={expected_by_name}"
        )
    if expected_by_name != EXPECTED_TRAIN_COUNTS:
        raise RuntimeError(
            "Training class counts differ from frozen Milestone D expected counts: "
            f"{expected_by_name}"
        )
    return {"skipped": False, "computed": by_name, "expected": expected_by_name}


def overlap_width(row_a: dict[str, object], row_b: dict[str, object]) -> float:
    a_low, a_high = float(row_a["p25"]), float(row_a["p75"])
    b_low, b_high = float(row_b["p25"]), float(row_b["p75"])
    return max(0.0, min(a_high, b_high) - max(a_low, b_low))


def write_readme(
    out_dir: Path,
    dataset_root: Path,
    frames_processed: int,
    class_rows: list[dict[str, object]],
    split_rows: list[dict[str, object]],
    count_validation: dict[str, object],
) -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    by_name = {str(row["class_name"]): row for row in class_rows}
    road = by_name["road"]
    marking = by_name["marking"]
    other = by_name["other"]
    road_marking_overlap = overlap_width(road, marking)
    marking_other_overlap = overlap_width(marking, other)
    lines = [
        "# Road Marking3 Dataset Intensity Analysis",
        "",
        "This analysis describes the dataset under the Milestone D remap before using any D0 model predictions.",
        "",
        "## Scope",
        "",
        f"- generated_at: `{generated_at}`",
        f"- script: `analysis_code/analyze_road_marking3_intensity.py`",
        f"- dataset_root: `{dataset_root}`",
        "- sensor: forward-facing LiDAR only (`sensor_id=1`)",
        "- splits: training, validation, test",
        "- distance buckets: `0-10`, `10-20`, `20-30`, `30-40`, `40-60`, `60+` meters",
        "",
        "## Class Definition",
        "",
        "- `road`: raw `7 Road`",
        "- `marking`: raw `8 Lane Line Marking` + raw `9 Stop Line Marking` + raw `10 Other Road Marking`",
        "- `other`: all other non-ignored raw classes",
        "",
        "## Outputs",
        "",
        "- `class_intensity_summary.csv`: global raw-intensity summary per remapped class",
        "- `split_class_intensity_summary.csv`: same summary split by train/validation/test",
        "- `distance_bucket_intensity_summary.csv`: class intensity by distance bucket",
        "- `plots/class_intensity_histograms.png`: normalized intensity histograms",
        "- `plots/distance_intensity_medians.png`: median intensity and IQR by distance",
        "- `manifest.json`: provenance and validation details",
        "",
        "## Key Checks",
        "",
        f"- frames_processed: `{frames_processed}`",
        f"- training count validation: `{count_validation}`",
        "",
        "## Initial Findings",
        "",
        f"- Global median intensity: road `{road['median']}`, marking `{marking['median']}`, other `{other['median']}`.",
        f"- Road and marking middle-50% intensity overlap width: `{road_marking_overlap:.1f}` raw-intensity units.",
        f"- Marking and other middle-50% intensity overlap width: `{marking_other_overlap:.1f}` raw-intensity units.",
        "- Use the distance plot to check whether marking remains brighter than road at the same range. If marking and road medians converge by distance, remaining D0 marking-road confusion is likely partly input-limited for LiDAR-only training.",
        "- This analysis is dataset-only. It does not use D0 predictions and is therefore relevant for D1/D2 and for deciding whether Milestone E RGB/color features are justified.",
        "",
        "## Split Stability",
        "",
        "| split | road median | marking median | other median | marking count |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLIT_ORDER:
        split_by_class = {
            str(row["class_name"]): row for row in split_rows if row["split"] == split
        }
        lines.append(
            f"| {split} | {split_by_class['road']['median']} | "
            f"{split_by_class['marking']['median']} | {split_by_class['other']['median']} | "
            f"{int(split_by_class['marking']['count']):,} |"
        )
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = read_dataset_root(args.dataset_root, args.dataset_root_file)
    split_ids = read_split_ids(args.split_dir)
    sequence_plan = build_sequence_plan(split_ids, args.sequence_ids, args.max_sequences)
    ds = DataSet(str(dataset_root))
    reapply_sensor = sensor_reapply_required()

    stats_by_class = {class_id: HistStats() for class_id in CLASS_ORDER}
    stats_by_split_class = {
        (split, class_id): HistStats()
        for split in SPLIT_ORDER
        for class_id in CLASS_ORDER
    }
    stats_by_bucket = {
        (class_id, label): HistStats()
        for class_id in CLASS_ORDER
        for _low, _high, label in DISTANCE_BUCKETS
    }

    frame_rows: list[dict[str, object]] = []
    frames_processed = 0
    for split, seq_id in sequence_plan:
        if args.max_frames is not None and frames_processed >= args.max_frames:
            break
        print(f"sequence_start split={split} seq_id={seq_id}", flush=True)
        seq = ds[seq_id]
        try:
            seq.load_lidar().load_semseg()
        except Exception:
            seq.load_lidar()
            seq.load_semseg()
        frame_count = get_frame_count(seq)
        for frame_idx in range(frame_count):
            if args.max_frames is not None and frames_processed >= args.max_frames:
                break
            frame_rows.append(
                process_frame(
                    seq=seq,
                    seq_id=seq_id,
                    frame_idx=frame_idx,
                    split=split,
                    sensor_reapply=reapply_sensor,
                    stats_by_class=stats_by_class,
                    stats_by_split_class=stats_by_split_class,
                    stats_by_bucket=stats_by_bucket,
                )
            )
            frames_processed += 1
            if frames_processed % 100 == 0:
                print(f"frames_processed {frames_processed}", flush=True)

        seq.lidar._data = None
        seq.lidar._poses = None
        seq.lidar._timestamps = None
        if seq.semseg is not None:
            seq.semseg._data = None
        gc.collect()

    class_rows = []
    for class_id in CLASS_ORDER:
        cls = CLASS_DEFS[class_id]
        class_rows.append(
            stats_by_class[class_id].row(
                {
                    "class_id": class_id,
                    "class_name": cls["name"],
                    "raw_definition": cls["raw_definition"],
                }
            )
        )
    add_shares(class_rows, ())
    write_csv(
        out_dir / "class_intensity_summary.csv",
        class_rows,
        stat_row_fields(("class_id", "class_name", "raw_definition")),
    )

    split_rows = []
    for split in SPLIT_ORDER:
        for class_id in CLASS_ORDER:
            cls = CLASS_DEFS[class_id]
            split_rows.append(
                stats_by_split_class[(split, class_id)].row(
                    {
                        "split": split,
                        "class_id": class_id,
                        "class_name": cls["name"],
                        "raw_definition": cls["raw_definition"],
                    }
                )
            )
    add_shares(split_rows, ("split",))
    write_csv(
        out_dir / "split_class_intensity_summary.csv",
        split_rows,
        stat_row_fields(("split", "class_id", "class_name", "raw_definition")),
    )

    distance_rows = []
    for class_id in CLASS_ORDER:
        cls = CLASS_DEFS[class_id]
        for low, high, label in DISTANCE_BUCKETS:
            distance_rows.append(
                stats_by_bucket[(class_id, label)].row(
                    {
                        "class_id": class_id,
                        "class_name": cls["name"],
                        "bucket": label,
                        "range_min_m": low,
                        "range_max_m": "" if np.isinf(high) else high,
                    }
                )
            )
    add_shares(distance_rows, ("bucket",))
    write_csv(
        out_dir / "distance_bucket_intensity_summary.csv",
        distance_rows,
        stat_row_fields(("class_id", "class_name", "bucket", "range_min_m", "range_max_m")),
    )

    write_csv(
        out_dir / "frame_remapped_class_counts.csv",
        frame_rows,
        [
            "split",
            "seq_id",
            "frame_idx",
            "active_points_forward_sensor",
            "road_count",
            "marking_count",
            "other_count",
            "ignored_count",
            "raw_8_count",
            "raw_9_count",
            "raw_10_count",
        ],
    )

    count_validation = validate_training_counts(
        split_rows,
        args.stats_file,
        skip=bool(args.skip_count_validation or args.max_frames or args.max_sequences or args.sequence_ids),
    )
    plot_class_histograms(stats_by_class, plots_dir / "class_intensity_histograms.png")
    plot_distance_medians(distance_rows, plots_dir / "distance_intensity_medians.png")
    write_readme(out_dir, dataset_root, frames_processed, class_rows, split_rows, count_validation)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "dataset_root": str(dataset_root),
        "label_mode": LABEL_MODE_ROAD_MARKING3,
        "forward_sensor_id": FORWARD_SENSOR_ID,
        "frames_processed": frames_processed,
        "sensor_reapply_per_frame": reapply_sensor,
        "class_order": {
            "1": "road",
            "2": "marking",
            "3": "other",
        },
        "raw_marking_subtypes": {
            "8": "lane line marking",
            str(RAW_STOP_LINE_ID): "stop line marking",
            str(RAW_OTHER_ROAD_MARKING_ID): "other road marking",
        },
        "raw_road_marking_ids": sorted(int(value) for value in RAW_ROAD_MARKING_IDS),
        "raw_road_id": RAW_ROAD_ID,
        "distance_buckets": [
            {"label": label, "min_m": low, "max_m": None if np.isinf(high) else high}
            for low, high, label in DISTANCE_BUCKETS
        ],
        "count_validation": count_validation,
        "outputs": [
            "class_intensity_summary.csv",
            "split_class_intensity_summary.csv",
            "distance_bucket_intensity_summary.csv",
            "frame_remapped_class_counts.csv",
            "plots/class_intensity_histograms.png",
            "plots/distance_intensity_medians.png",
            "README.md",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_dir}", flush=True)
    print("script_status PASS", flush=True)


if __name__ == "__main__":
    main()
