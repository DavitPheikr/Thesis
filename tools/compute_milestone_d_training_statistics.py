"""Compute Milestone D road-marking3 training statistics and class weights.

This is the Milestone D equivalent of the C0/Milestone B statistics flow.
It scans the training split only, using the forward-facing LiDAR and the
road_marking3 remap:

  raw 7       -> road
  raw 8/9/10  -> marking
  raw 1/2/3/4 -> ignore
  everything else active -> other

Default output:
  - logs/milestone_d/road_marking3_training_statistics.json
  - logs/milestone_d/road_marking3_statistics_progress.jsonl
  - logs/milestone_d/reports/road_marking3_training_statistics.md
  - logs/milestone_d/cache/stats_per_seq/{seq_id}.npy

Emits script_status PASS/FAIL as final line.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCHED_DEVKIT = PROJECT_ROOT / "pandaset-devkit/python"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PATCHED_DEVKIT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open3d._ml3d.datasets.utils import DataProcessing
from pandaset import DataSet, geometry as pds_geometry

from thesis_pipeline.adapters.pandaset_ff_lane3 import (
    LABEL_MODE_ROAD_MARKING3,
    RAW_LANE_ID,
    RAW_OTHER_ROAD_MARKING_ID,
    RAW_ROAD_ID,
    RAW_STOP_LINE_ID,
    remap_raw_pandaset_ids,
)
from thesis_pipeline.core.pandaset_compat import get_frame_count


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs/milestone_d"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "configs/splits"
DEFAULT_DATASET_ROOT_FILE = PROJECT_ROOT / "logs/dataset_root.txt"
DEFAULT_PREFLIGHT_PATTERN_FILE = (
    PROJECT_ROOT / "logs/milestone_b_preflight_sensor_pattern.txt"
)
DEFAULT_C0_STATS_FILE = PROJECT_ROOT / "logs/milestone_b_training_statistics.json"
DEFAULT_RAW_MARKING_FRAME_COUNTS = (
    PROJECT_ROOT / "logs/raw_road_marking_analysis/frame_marking_counts.csv"
)
DEFAULT_LOSS_INTERFACE_FILE = PROJECT_ROOT / "logs/milestone_b_loss_interface.json"

STATS_FILENAME = "road_marking3_training_statistics.json"
PROGRESS_FILENAME = "road_marking3_statistics_progress.jsonl"
REPORT_FILENAME = "road_marking3_training_statistics.md"
PER_SEQUENCE_SAMPLE_CAP = 100_000
SKIP_RATE_THRESHOLD = 0.05
FORWARD_SENSOR_ID = 1
GRID_SIZE = 0.04
NUM_POINTS_INTENT = 32768
LARGE_WEIGHT_THRESHOLD = 50.0
RANDOM_SEED = 42
MAX_RECORDED_FRAME_ERRORS = 20

RAW_INTEREST_CLASSES = {
    RAW_ROAD_ID: "raw_7_road",
    RAW_LANE_ID: "raw_8_lane_line_marking",
    RAW_STOP_LINE_ID: "raw_9_stop_line_marking",
    RAW_OTHER_ROAD_MARKING_ID: "raw_10_other_road_marking",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-root-file", type=Path, default=DEFAULT_DATASET_ROOT_FILE)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--preflight-pattern-file", type=Path, default=DEFAULT_PREFLIGHT_PATTERN_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-marking-frame-counts", type=Path, default=DEFAULT_RAW_MARKING_FRAME_COUNTS)
    parser.add_argument("--c0-stats-file", type=Path, default=DEFAULT_C0_STATS_FILE)
    parser.add_argument("--loss-interface-file", type=Path, default=DEFAULT_LOSS_INTERFACE_FILE)
    parser.add_argument("--sequence-ids", nargs="*", default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--skip-rate-threshold", type=float, default=SKIP_RATE_THRESHOLD)
    parser.add_argument("--per-sequence-sample-cap", type=int, default=PER_SEQUENCE_SAMPLE_CAP)
    parser.add_argument("--num-points-intent", type=int, default=NUM_POINTS_INTENT)
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_dataset_root(dataset_root: Path | None, dataset_root_file: Path) -> Path:
    if dataset_root is not None:
        return resolve_project_path(dataset_root)
    root = Path(resolve_project_path(dataset_root_file).read_text().strip()).expanduser()
    if root.is_absolute():
        return root
    return PROJECT_ROOT / root


def sensor_reapply_required(preflight_pattern_file: Path) -> bool:
    if not preflight_pattern_file.exists():
        return False
    return "reapply_per_frame: true" in preflight_pattern_file.read_text()


def is_limited_run(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.sequence_ids,
            args.max_sequences,
            args.max_frames_per_sequence,
        )
    )


def validate_output_safety(args: argparse.Namespace, out_dir: Path) -> None:
    if out_dir.resolve() != DEFAULT_OUTPUT_DIR.resolve():
        return
    if is_limited_run(args):
        raise SystemExit(
            "Refusing to write a limited statistics run to the official "
            "Milestone D output directory. Pass --out-dir to a scratch path, "
            "for example /tmp/milestone_d_stats_check."
        )


def build_sequence_plan(
    train_ids: list[str],
    sequence_ids: list[str] | None,
    max_sequences: int | None,
) -> list[str]:
    requested = {seq_id.zfill(3) for seq_id in sequence_ids} if sequence_ids else None
    plan = [seq_id for seq_id in train_ids if requested is None or seq_id in requested]
    if max_sequences is not None:
        plan = plan[:max_sequences]
    return plan


def is_completed_record(record: dict[str, Any]) -> bool:
    return record.get("error") is None


def load_canonical_progress(progress_path: Path) -> dict[str, dict[str, Any]]:
    canonical: dict[str, dict[str, Any]] = {}
    if not progress_path.exists():
        return canonical

    for line_no, line in enumerate(progress_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Malformed JSON in {progress_path} at line {line_no}: {exc}"
            ) from exc

        seq_id = record.get("seq_id")
        if not seq_id:
            raise RuntimeError(
                f"Missing seq_id in {progress_path} at line {line_no}: {record}"
            )

        existing = canonical.get(seq_id)
        if existing is None:
            canonical[seq_id] = record
            continue
        if record != existing:
            raise RuntimeError(
                "Conflicting statistics records found for seq_id "
                f"{seq_id} in {progress_path}. Resolve manually before resuming."
            )

    return canonical


def sample_intensity_values(
    intensity_chunks: list[np.ndarray],
    sample_cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not intensity_chunks:
        return np.array([], dtype=np.float32)

    values = np.concatenate(intensity_chunks).astype(np.float32, copy=False)
    if values.shape[0] <= sample_cap:
        return values

    idx = rng.choice(values.shape[0], size=sample_cap, replace=False)
    return values[idx]


def empty_numeric_counts() -> dict[int, int]:
    return {0: 0, 1: 0, 2: 0, 3: 0}


def empty_raw_interest_counts() -> dict[str, int]:
    return {name: 0 for name in RAW_INTEREST_CLASSES.values()}


def add_count_dicts(target: dict[Any, int], source: dict[Any, int], keys: list[Any]) -> None:
    for key in keys:
        target[key] += int(source.get(key, source.get(str(key), 0)))


def process_sequence(
    ds: DataSet,
    seq_id: str,
    sensor_reapply: bool,
    per_seq_dir: Path,
    sample_cap: int,
    max_frames_per_sequence: int | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "seq_id": seq_id,
        "label_mode": LABEL_MODE_ROAD_MARKING3,
        "lidar_frames": 0,
        "semseg_frames": 0,
        "frames_attempted": 0,
        "frames_processed": 0,
        "frames_semseg_missing": 0,
        "frames_failed": 0,
        "frame_errors": [],
        "class_counts": empty_numeric_counts(),
        "raw_interest_counts": empty_raw_interest_counts(),
        "intensity_sample_path": None,
        "error": None,
    }

    rng = np.random.default_rng(seed=RANDOM_SEED + (int(seq_id) * 1009))

    try:
        seq = ds[seq_id]
        try:
            seq.load_lidar().load_semseg()
        except Exception:
            seq.load_lidar()
            seq.load_semseg()

        if seq.semseg is None:
            result["error"] = "semseg_unavailable"
            return result

        lidar_n = get_frame_count(seq)
        result["lidar_frames"] = lidar_n

        try:
            semseg_n = len(seq.semseg.data)
        except Exception:
            semseg_n = lidar_n
        result["semseg_frames"] = semseg_n

        if not sensor_reapply:
            seq.lidar.set_sensor(FORWARD_SENSOR_ID)

        frame_limit = lidar_n
        if max_frames_per_sequence is not None:
            frame_limit = min(frame_limit, max_frames_per_sequence)

        intensity_chunks: list[np.ndarray] = []
        for frame_idx in range(frame_limit):
            result["frames_attempted"] += 1
            try:
                if sensor_reapply:
                    seq.lidar.set_sensor(FORWARD_SENSOR_ID)
                pc_df = seq.lidar[frame_idx]

                if frame_idx >= semseg_n:
                    result["frames_semseg_missing"] += 1
                    continue

                semseg_df = seq.semseg[frame_idx]
                raw_labels = semseg_df.loc[pc_df.index, "class"].to_numpy(dtype=np.int32)
                remapped = remap_raw_pandaset_ids(
                    raw_labels,
                    label_mode=LABEL_MODE_ROAD_MARKING3,
                )

                for cls in (0, 1, 2, 3):
                    result["class_counts"][cls] += int((remapped == cls).sum())

                for raw_id, name in RAW_INTEREST_CLASSES.items():
                    result["raw_interest_counts"][name] += int((raw_labels == raw_id).sum())

                intensity_chunks.append(pc_df["i"].to_numpy(dtype=np.float32))
                result["frames_processed"] += 1
            except Exception as exc:
                result["frames_failed"] += 1
                if len(result["frame_errors"]) < MAX_RECORDED_FRAME_ERRORS:
                    result["frame_errors"].append(
                        {
                            "frame_idx": frame_idx,
                            "error": str(exc),
                        }
                    )

        samples = sample_intensity_values(intensity_chunks, sample_cap, rng)
        if samples.size > 0:
            out_path = per_seq_dir / f"{seq_id}.npy"
            np.save(out_path, samples)
            result["intensity_sample_path"] = str(out_path)
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        try:
            seq.lidar._data = None
            seq.lidar._poses = None
            seq.lidar._timestamps = None
            seq.semseg._data = None
        except Exception:
            pass
        gc.collect()

    return result


def aggregate_from_artifacts(
    completed_records: dict[str, dict[str, Any]],
) -> tuple[dict[int, int], dict[str, int], np.ndarray, dict[str, Any]]:
    all_intensity = []
    total_class_counts = empty_numeric_counts()
    total_raw_interest_counts = empty_raw_interest_counts()
    summary = {
        "frames_attempted": 0,
        "frames_processed": 0,
        "frames_semseg_missing": 0,
        "frames_failed": 0,
        "frame_errors": [],
        "missing_intensity_sample_files": [],
    }

    for record in completed_records.values():
        add_count_dicts(total_class_counts, record["class_counts"], [0, 1, 2, 3])
        for key in total_raw_interest_counts:
            total_raw_interest_counts[key] += int(record["raw_interest_counts"].get(key, 0))

        for key in (
            "frames_attempted",
            "frames_processed",
            "frames_semseg_missing",
            "frames_failed",
        ):
            summary[key] += int(record.get(key, 0))
        summary["frame_errors"].extend(record.get("frame_errors", []))

        npy_path = record.get("intensity_sample_path")
        if npy_path and Path(npy_path).exists():
            all_intensity.append(np.load(npy_path))
        elif record.get("frames_processed", 0) > 0:
            summary["missing_intensity_sample_files"].append(npy_path)

    if all_intensity:
        intensity_all = np.concatenate(all_intensity).astype(np.float32, copy=False)
    else:
        intensity_all = np.array([], dtype=np.float32)

    return total_class_counts, total_raw_interest_counts, intensity_all, summary


def normalize_by_min(weights_by_class: dict[str, float]) -> dict[str, float]:
    weight_min = min(weights_by_class.values())
    return {key: value / weight_min for key, value in weights_by_class.items()}


def compute_class_weights(
    road_count: int,
    marking_count: int,
    other_count: int,
    loss_interface_file: Path,
) -> dict[str, Any]:
    active_counts = {
        "road": road_count,
        "marking": marking_count,
        "other": other_count,
    }
    if any(value <= 0 for value in active_counts.values()):
        raise RuntimeError(f"Active class count cannot be zero: {active_counts}")

    active_total = road_count + marking_count + other_count
    k_classes = 3

    raw_direct = normalize_by_min(
        {
            name: active_total / (k_classes * count)
            for name, count in active_counts.items()
        }
    )
    sqrt_direct = normalize_by_min(
        {
            name: math.sqrt(active_total / (k_classes * count))
            for name, count in active_counts.items()
        }
    )

    raw_direct_list = [raw_direct["road"], raw_direct["marking"], raw_direct["other"]]
    sqrt_direct_list = [
        sqrt_direct["road"],
        sqrt_direct["marking"],
        sqrt_direct["other"],
    ]

    raw_marking_weight = raw_direct["marking"]
    candidate_recommended_variant = (
        "sqrt_inverse_frequency"
        if raw_marking_weight > LARGE_WEIGHT_THRESHOLD
        else "raw_inverse_frequency"
    )
    candidate_recommended_list = (
        sqrt_direct_list
        if candidate_recommended_variant == "sqrt_inverse_frequency"
        else raw_direct_list
    )

    runtime_input_by_class = {
        "road": float(road_count),
        "marking": float(marking_count),
        "other": float(other_count),
    }
    runtime_input_as_list = [
        runtime_input_by_class["road"],
        runtime_input_by_class["marking"],
        runtime_input_by_class["other"],
    ]
    runtime_effective_ce_weights = DataProcessing.get_class_weights(
        runtime_input_as_list
    ).astype(np.float32).tolist()

    interface: dict[str, Any] = {}
    if loss_interface_file.exists():
        interface = json.loads(loss_interface_file.read_text())

    return {
        "deviation_note": (
            "For local Open3D-ML, dataset.cfg.class_weights is transformed "
            "internally via DataProcessing.get_class_weights before "
            "CrossEntropyLoss is built. Use the count-like runtime list for "
            "training configs, not the direct candidate weights."
        ),
        "class_order": "[road, marking, other] after ignored label 0 is filtered out",
        "candidate_weight_semantics": (
            "direct final CrossEntropyLoss weights for analysis only; not "
            "consumed directly by the local Open3D runtime"
        ),
        "raw_inverse_frequency": {
            "by_class": raw_direct,
            "as_list": raw_direct_list,
            "marking_weight": raw_marking_weight,
        },
        "sqrt_inverse_frequency": {
            "by_class": sqrt_direct,
            "as_list": sqrt_direct_list,
            "marking_weight": sqrt_direct["marking"],
        },
        "candidate_recommended_variant": candidate_recommended_variant,
        "candidate_recommended_list": candidate_recommended_list,
        "runtime_open3d_native_transform": {
            "input_semantics": (
                "per-class counts or count-like values passed into "
                "DataProcessing.get_class_weights"
            ),
            "runtime_input_source": "measured active-class counts from class_counts",
            "runtime_input_by_class": runtime_input_by_class,
            "runtime_input_as_list": runtime_input_as_list,
            "runtime_effective_ce_weights": runtime_effective_ce_weights,
        },
        "weight_format_source": str(loss_interface_file),
        "weight_scope_discovered": interface.get("weight_scope", "UNKNOWN"),
        "weight_list_length_discovered": interface.get("weight_list_length", "UNKNOWN"),
        "sanity_run_recommended_variant": "open3d_native_from_measured_counts",
        "sanity_run_recommended_list": runtime_input_as_list,
        "large_weight_warning": raw_marking_weight > LARGE_WEIGHT_THRESHOLD,
    }


def compute_intensity_stats(intensity_all: np.ndarray) -> dict[str, float]:
    if intensity_all.size == 0:
        raise RuntimeError("No intensity samples collected")

    p_low = float(np.percentile(intensity_all, 0.5))
    p_high = float(np.percentile(intensity_all, 99.5))
    clipped = np.clip(intensity_all, p_low, p_high)
    clip_mean = float(clipped.mean())
    clip_std = float(clipped.std())
    if clip_std <= 0:
        raise RuntimeError("clip_std is zero after clipping")

    return {
        "clip_low_p0p5": p_low,
        "clip_high_p99p5": p_high,
        "clip_mean": clip_mean,
        "clip_std": clip_std,
    }


def build_grid_subsampling_metadata(
    ds: DataSet,
    seq_id: str,
    sensor_reapply: bool,
    grid_size: float,
    num_points_intent: int,
) -> dict[str, Any]:
    seq = ds[seq_id]
    try:
        seq.load_lidar().load_semseg()
    except Exception:
        seq.load_lidar()
        seq.load_semseg()

    if sensor_reapply:
        seq.lidar.set_sensor(FORWARD_SENSOR_ID)
    else:
        seq.lidar.set_sensor(FORWARD_SENSOR_ID)

    frame_idx = 0
    pc_df = seq.lidar[frame_idx]
    semseg_df = seq.semseg[frame_idx]
    raw_labels = semseg_df.loc[pc_df.index, "class"].to_numpy(dtype=np.int32)
    labels = remap_raw_pandaset_ids(raw_labels, label_mode=LABEL_MODE_ROAD_MARKING3)

    xyz_world = pc_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    pose = seq.lidar.poses[frame_idx]
    points = pds_geometry.lidar_points_to_ego(xyz_world, pose).astype(
        np.float32,
        copy=False,
    )
    features = pc_df["i"].to_numpy(dtype=np.float32)[:, None]

    sub_points, _sub_feat, _sub_label = DataProcessing.grid_subsampling(
        points,
        features=features,
        labels=labels,
        grid_size=grid_size,
    )

    try:
        seq.lidar._data = None
        seq.lidar._poses = None
        seq.lidar._timestamps = None
        seq.semseg._data = None
    except Exception:
        pass
    gc.collect()

    post_grid_point_count = int(sub_points.shape[0])
    raw_point_count = int(points.shape[0])
    return {
        "grid_size": grid_size,
        "source_sequence_id": seq_id,
        "source_frame_idx": frame_idx,
        "raw_point_count": raw_point_count,
        "post_grid_point_count": post_grid_point_count,
        "below_num_points_intent": post_grid_point_count < num_points_intent,
        "duplication_will_occur": post_grid_point_count < num_points_intent,
        "num_points_intent": num_points_intent,
        "note": "Informational only; final D0 config is created in a later stage.",
    }


def load_raw_marking_training_counts(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    totals = {
        "raw_7_road": 0,
        "raw_8_lane_line_marking": 0,
        "raw_9_stop_line_marking": 0,
        "raw_10_other_road_marking": 0,
    }
    rows = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") != "training":
                continue
            rows += 1
            totals["raw_7_road"] += int(row["raw_7_count"])
            totals["raw_8_lane_line_marking"] += int(row["raw_8_count"])
            totals["raw_9_stop_line_marking"] += int(row["raw_9_count"])
            totals["raw_10_other_road_marking"] += int(row["raw_10_count"])

    totals["raw_8_9_10_marking_sum"] = (
        totals["raw_8_lane_line_marking"]
        + totals["raw_9_stop_line_marking"]
        + totals["raw_10_other_road_marking"]
    )
    return {
        "source": str(path),
        "training_rows": rows,
        "counts": totals,
    }


def build_c0_comparison(c0_stats_file: Path, raw_interest_counts: dict[str, int]) -> dict[str, Any] | None:
    if not c0_stats_file.exists():
        return None

    c0_stats = json.loads(c0_stats_file.read_text())
    c0_counts = c0_stats["class_counts"]
    raw9 = raw_interest_counts["raw_9_stop_line_marking"]
    raw10 = raw_interest_counts["raw_10_other_road_marking"]
    return {
        "source": str(c0_stats_file),
        "c0_counts": c0_counts,
        "expected_marking_from_c0_lane_plus_raw9_raw10": int(
            c0_counts["lane"] + raw9 + raw10
        ),
        "expected_other_from_c0_other_minus_raw9_raw10": int(
            c0_counts["other"] - raw9 - raw10
        ),
        "note": (
            "Expected values assume same train split, same forward sensor, and "
            "only remapping raw 9/10 from other into marking."
        ),
    }


def build_sanity_checks(
    class_counts_named: dict[str, int],
    raw_interest_counts: dict[str, int],
    raw_marking_estimate: dict[str, Any] | None,
    c0_comparison: dict[str, Any] | None,
    limited_run: bool,
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "road_equals_raw7": (
            class_counts_named["road"] == raw_interest_counts["raw_7_road"]
        ),
        "marking_equals_raw8_plus_raw9_plus_raw10": (
            class_counts_named["marking"]
            == raw_interest_counts["raw_8_lane_line_marking"]
            + raw_interest_counts["raw_9_stop_line_marking"]
            + raw_interest_counts["raw_10_other_road_marking"]
        ),
    }
    if limited_run:
        checks["external_full_run_checks"] = (
            "skipped because this is a limited run"
        )
        return checks

    if raw_marking_estimate is not None:
        estimate_counts = raw_marking_estimate["counts"]
        checks["matches_raw_road_marking_analysis_training_counts"] = {
            "road": class_counts_named["road"] == estimate_counts["raw_7_road"],
            "marking": (
                class_counts_named["marking"]
                == estimate_counts["raw_8_9_10_marking_sum"]
            ),
        }
    if c0_comparison is not None:
        checks["matches_c0_expected_remap_shift"] = {
            "road": class_counts_named["road"]
            == c0_comparison["c0_counts"]["road"],
            "marking": class_counts_named["marking"]
            == c0_comparison["expected_marking_from_c0_lane_plus_raw9_raw10"],
            "other": class_counts_named["other"]
            == c0_comparison["expected_other_from_c0_other_minus_raw9_raw10"],
        }
    return checks


def write_report(stats: dict[str, Any], report_path: Path) -> None:
    counts = stats["class_counts"]
    weights = stats["class_weights"]["runtime_open3d_native_transform"]
    intensity = stats["intensity"]
    provenance = stats["provenance"]

    lines = [
        "# Milestone D Training Statistics",
        "",
        "## Scope",
        "",
        "- split: training only",
        "- label mode: road_marking3",
        "- positive class: raw 8 + raw 9 + raw 10 road markings",
        "",
        "## Class Counts",
        "",
        "| class | count | active share |",
        "| --- | ---: | ---: |",
    ]

    active_total = counts["road"] + counts["marking"] + counts["other"]
    for name in ("road", "marking", "other"):
        share = counts[name] / active_total if active_total else 0.0
        lines.append(f"| {name} | {counts[name]:,} | {share:.6%} |")
    lines.append(f"| ignore | {counts['ignore']:,} | n/a |")

    lines.extend(
        [
            "",
            "## Recommended Config Weights",
            "",
            "Use this count-like list in the Open3D-ML config:",
            "",
            "```text",
            str(weights["runtime_input_as_list"]),
            "```",
            "",
            "Open3D-ML transforms that list internally into effective CE weights:",
            "",
            "```text",
            str(weights["runtime_effective_ce_weights"]),
            "```",
            "",
            "## Intensity Normalization",
            "",
            "```text",
            f"clip_low_p0p5  {intensity['clip_low_p0p5']:.6f}",
            f"clip_high_p99p5 {intensity['clip_high_p99p5']:.6f}",
            f"clip_mean      {intensity['clip_mean']:.6f}",
            f"clip_std       {intensity['clip_std']:.6f}",
            "```",
            "",
            "## Run Health",
            "",
            "```text",
            f"sequences_completed {provenance['sequences_completed']}",
            f"frames_processed    {provenance['frames_processed']}",
            f"frames_failed       {provenance['frames_failed']}",
            f"skip_rate           {provenance['skip_rate']:.6f}",
            "```",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    print("script_status FAIL")
    sys.exit(1)


def main() -> None:
    args = parse_args()

    out_dir = resolve_project_path(args.out_dir)
    split_dir = resolve_project_path(args.split_dir)
    dataset_root_file = resolve_project_path(args.dataset_root_file)
    preflight_pattern_file = resolve_project_path(args.preflight_pattern_file)
    raw_marking_frame_counts = resolve_project_path(args.raw_marking_frame_counts)
    c0_stats_file = resolve_project_path(args.c0_stats_file)
    loss_interface_file = resolve_project_path(args.loss_interface_file)

    validate_output_safety(args, out_dir)

    stats_file = out_dir / STATS_FILENAME
    progress_file = out_dir / PROGRESS_FILENAME
    per_seq_dir = out_dir / "cache/stats_per_seq"
    if out_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        report_file = out_dir / "reports" / REPORT_FILENAME
    else:
        report_file = out_dir / REPORT_FILENAME

    train_ids = read_lines(split_dir / "train.txt")
    sequence_plan = build_sequence_plan(
        train_ids,
        args.sequence_ids,
        args.max_sequences,
    )
    if not sequence_plan:
        fail("empty training sequence plan")

    dataset_root = read_dataset_root(args.dataset_root, dataset_root_file)
    sensor_reapply = sensor_reapply_required(preflight_pattern_file)

    print(f"label_mode {LABEL_MODE_ROAD_MARKING3}")
    print(f"dataset_root {dataset_root}")
    print(f"sensor_reapply_per_frame {sensor_reapply}")
    print(f"training_sequences_full {len(train_ids)}")
    print(f"sequences_in_plan {len(sequence_plan)}")
    print(f"out_dir {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    per_seq_dir.mkdir(parents=True, exist_ok=True)

    ds = DataSet(str(dataset_root))
    completed = {
        seq_id: record
        for seq_id, record in load_canonical_progress(progress_file).items()
        if is_completed_record(record)
    }
    print(f"already_completed {len(completed)}")

    with progress_file.open("a") as progress_handle:
        for seq_id in sequence_plan:
            if seq_id in completed:
                continue
            print(f"  processing {seq_id}...", end=" ", flush=True)
            record = process_sequence(
                ds=ds,
                seq_id=seq_id,
                sensor_reapply=sensor_reapply,
                per_seq_dir=per_seq_dir,
                sample_cap=args.per_sequence_sample_cap,
                max_frames_per_sequence=args.max_frames_per_sequence,
            )
            print(
                f"processed={record['frames_processed']} "
                f"failed={record['frames_failed']} "
                f"missing={record['frames_semseg_missing']} "
                f"err={record['error']}"
            )
            progress_handle.write(json.dumps(record) + "\n")
            progress_handle.flush()
            if is_completed_record(record):
                completed[seq_id] = record
            del record
            gc.collect()

    plan_completed = {
        seq_id: record for seq_id, record in completed.items() if seq_id in sequence_plan
    }
    (
        class_counts_numeric,
        raw_interest_counts,
        intensity_all,
        aggregate_summary,
    ) = aggregate_from_artifacts(plan_completed)

    frames_processed = aggregate_summary["frames_processed"]
    frames_semseg_missing = aggregate_summary["frames_semseg_missing"]
    frames_failed = aggregate_summary["frames_failed"]
    denominator = frames_processed + frames_semseg_missing
    skip_rate = frames_semseg_missing / denominator if denominator else 0.0

    print()
    print(f"sequences_completed {len(plan_completed)}")
    print(f"sequences_expected {len(sequence_plan)}")
    print(f"frames_processed {frames_processed}")
    print(f"frames_semseg_missing {frames_semseg_missing}")
    print(f"frames_failed {frames_failed}")
    print(f"skip_rate {skip_rate:.6f}")
    print(f"class_counts_numeric {class_counts_numeric}")
    print(f"raw_interest_counts {raw_interest_counts}")

    if len(plan_completed) != len(sequence_plan):
        fail("not all planned sequences completed")
    if frames_failed > 0:
        fail("one or more frames failed during statistics computation")
    if skip_rate > args.skip_rate_threshold:
        fail(
            f"skip_rate {skip_rate:.6f} exceeds threshold "
            f"{args.skip_rate_threshold:.6f}"
        )
    if aggregate_summary["missing_intensity_sample_files"]:
        fail(
            "progress references missing intensity sample files; delete the "
            "partial output directory and rerun"
        )

    marking_count = int(class_counts_numeric[2])
    if marking_count <= 0:
        fail("zero marking-class points in training split")

    try:
        intensity_stats = compute_intensity_stats(intensity_all)
        class_counts_named = {
            "ignore": int(class_counts_numeric[0]),
            "road": int(class_counts_numeric[1]),
            "marking": marking_count,
            "other": int(class_counts_numeric[3]),
        }
        class_weights = compute_class_weights(
            road_count=class_counts_named["road"],
            marking_count=class_counts_named["marking"],
            other_count=class_counts_named["other"],
            loss_interface_file=loss_interface_file,
        )
        grid_metadata = build_grid_subsampling_metadata(
            ds=ds,
            seq_id=sequence_plan[0],
            sensor_reapply=sensor_reapply,
            grid_size=GRID_SIZE,
            num_points_intent=args.num_points_intent,
        )
    except Exception as exc:
        fail(str(exc))

    active_total = (
        class_counts_named["road"]
        + class_counts_named["marking"]
        + class_counts_named["other"]
    )
    raw_marking_estimate = load_raw_marking_training_counts(raw_marking_frame_counts)
    c0_comparison = build_c0_comparison(c0_stats_file, raw_interest_counts)
    sanity_checks = build_sanity_checks(
        class_counts_named=class_counts_named,
        raw_interest_counts=raw_interest_counts,
        raw_marking_estimate=raw_marking_estimate,
        c0_comparison=c0_comparison,
        limited_run=is_limited_run(args),
    )

    stats = {
        "label_mode": LABEL_MODE_ROAD_MARKING3,
        "active_label_names": {
            "0": "ignore",
            "1": "road",
            "2": "marking",
            "3": "other",
        },
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "split_file": str(split_dir / "train.txt"),
            "dataset_root": str(dataset_root),
            "dataset_root_file": str(dataset_root_file),
            "output_dir": str(out_dir),
            "limited_run": is_limited_run(args),
            "sequences_in_full_train_split": len(train_ids),
            "sequences_in_plan": len(sequence_plan),
            "sequences_completed": len(plan_completed),
            "frames_attempted": aggregate_summary["frames_attempted"],
            "frames_processed": frames_processed,
            "frames_semseg_missing": frames_semseg_missing,
            "frames_failed": frames_failed,
            "frame_errors": aggregate_summary["frame_errors"][:MAX_RECORDED_FRAME_ERRORS],
            "max_recorded_frame_errors": MAX_RECORDED_FRAME_ERRORS,
            "skip_rate": skip_rate,
            "skip_rate_threshold": args.skip_rate_threshold,
            "forward_sensor_id": FORWARD_SENSOR_ID,
            "sensor_reapply_per_frame": sensor_reapply,
            "intensity_samples_total": int(intensity_all.shape[0]),
            "per_sequence_sample_cap": args.per_sequence_sample_cap,
            "random_seed": RANDOM_SEED,
            "class_weight_runtime_note": (
                "Local Open3D SemSegLoss transforms dataset.cfg.class_weights "
                "internally; use class_weights.sanity_run_recommended_list in "
                "Milestone D configs."
            ),
        },
        "intensity": intensity_stats,
        "class_counts": class_counts_named,
        "raw_interest_counts": raw_interest_counts,
        "derived": {
            "active_total": active_total,
            "road_share_active": class_counts_named["road"] / active_total,
            "marking_share_active": class_counts_named["marking"] / active_total,
            "other_share_active": class_counts_named["other"] / active_total,
            "raw_8_9_10_marking_sum": (
                raw_interest_counts["raw_8_lane_line_marking"]
                + raw_interest_counts["raw_9_stop_line_marking"]
                + raw_interest_counts["raw_10_other_road_marking"]
            ),
        },
        "class_weights": class_weights,
        "grid_subsampling": grid_metadata,
        "sanity_checks": sanity_checks,
        "external_cross_checks": {
            "raw_road_marking_analysis_training_counts": raw_marking_estimate,
            "c0_lane3_comparison": c0_comparison,
        },
    }

    stats_file.write_text(json.dumps(stats, indent=2) + "\n")
    write_report(stats, report_file)

    print(f"intensity_clip_low {intensity_stats['clip_low_p0p5']:.6f}")
    print(f"intensity_clip_high {intensity_stats['clip_high_p99p5']:.6f}")
    print(f"intensity_clip_mean {intensity_stats['clip_mean']:.6f}")
    print(f"intensity_clip_std {intensity_stats['clip_std']:.6f}")
    print(f"marking_points {marking_count}")
    print(
        "runtime_sanity_list "
        f"{class_weights['runtime_open3d_native_transform']['runtime_input_as_list']}"
    )
    print(
        "runtime_effective_ce_weights "
        f"{class_weights['runtime_open3d_native_transform']['runtime_effective_ce_weights']}"
    )
    print(f"stats_file {stats_file}")
    print(f"report_file {report_file}")
    print("statistics_complete True")
    print("script_status PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
