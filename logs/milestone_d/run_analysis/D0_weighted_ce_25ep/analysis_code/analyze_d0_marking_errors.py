"""Sampled D0 epoch-18 marking error analysis.

This runs a fresh sampled validation inference pass for the D0 best checkpoint.
It is not the exact epoch-18 validation sample from training; the D0 run stores
aggregate metrics, not per-point predictions.

Outputs are written under:

  logs/milestone_d/run_analysis/D0_weighted_ce_25ep/sampled_error_analysis_epoch18/
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from tqdm import tqdm


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "logs/milestone_d/configs/d0_weighted_ce.yml").exists():
            return parent
    raise RuntimeError("Could not find project root from analysis script path")


PROJECT_ROOT = find_project_root()
PATCHED_DEVKIT = PROJECT_ROOT / "pandaset-devkit/python"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PATCHED_DEVKIT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import open3d.ml.torch as ml3d
from open3d._ml3d.datasets.utils import DataProcessing
from open3d._ml3d.torch.dataloaders import TorchDataloader, get_sampler
from open3d._ml3d.torch.modules.losses.semseg_loss import filter_valid_label
from open3d._ml3d.torch.pipelines import SemanticSegmentation
from pandaset import geometry as pds_geometry

from thesis_pipeline.adapters.pandaset_ff_lane3 import (
    LABEL_MODE_ROAD_MARKING3,
    RAW_OTHER_ROAD_MARKING_ID,
    RAW_ROAD_MARKING_IDS,
    RAW_STOP_LINE_ID,
    remap_raw_pandaset_ids,
)
from thesis_pipeline.datasets.pandaset_ff_lane3_dataset import PandaSetFFLane3Dataset


RUN_NAME = "D0_weighted_ce_25ep"
DEFAULT_CONFIG = PROJECT_ROOT / "logs/milestone_d/configs/d0_weighted_ce.yml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / f"logs/milestone_d/runs/{RUN_NAME}/checkpoints/ckpt_epoch_00018.pth"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / f"logs/milestone_d/run_analysis/{RUN_NAME}/sampled_error_analysis_epoch18"
)

CLASS_NAMES = ("road", "marking", "other")
GROUPS = (
    "marking_tp",
    "marking_to_road",
    "marking_to_other",
    "road_tp",
    "road_to_marking",
    "other_tp",
    "other_to_marking",
)
RAW_MARKING_SUBTYPES = {
    8: "lane_line_marking",
    RAW_STOP_LINE_ID: "stop_line_marking",
    RAW_OTHER_ROAD_MARKING_ID: "other_road_marking",
}
DISTANCE_BUCKETS = (
    (0.0, 10.0, "0_10m"),
    (10.0, 20.0, "10_20m"),
    (20.0, 30.0, "20_30m"),
    (30.0, 40.0, "30_40m"),
    (40.0, 60.0, "40_60m"),
    (60.0, np.inf, "60m_plus"),
)
REPO_LOCAL_DATASET_PATHS = {
    "dataset_path": "pandaset/PandaSet",
    "split_dir": "configs/splits",
    "stats_file": "logs/milestone_d/road_marking3_training_statistics.json",
    "dataset_root_file": "logs/dataset_root.txt",
    "preflight_pattern_file": "logs/milestone_b_preflight_sensor_pattern.txt",
    "cache_dir": "logs/milestone_d/run_analysis/D0_weighted_ce_25ep/sampled_error_analysis_epoch18/cache",
    "test_result_folder": "logs/milestone_d/test_results/D0_weighted_ce_analysis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--steps", type=int, default=2160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--max-raw-remap-mismatch-rate",
        type=float,
        default=0.001,
        help="Fail if raw subtype labels remap differently from sampled remapped labels above this rate.",
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(name)


def repo_local_path(value: str | None, fallback_relative: str) -> str:
    fallback = PROJECT_ROOT / fallback_relative
    if value is None:
        return str(fallback)
    path = Path(value)
    if path.is_absolute():
        return str(path) if path.exists() else str(fallback)
    return str((PROJECT_ROOT / path).resolve())


def normalize_dataset_paths(cfg: dict[str, Any]) -> None:
    dataset_cfg = cfg["dataset"]
    for key, fallback_relative in REPO_LOCAL_DATASET_PATHS.items():
        dataset_cfg[key] = repo_local_path(dataset_cfg.get(key), fallback_relative)


def load_cfg(path: Path, steps: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    cfg = yaml.safe_load(path.read_text())
    if cfg["dataset"].get("label_mode") != LABEL_MODE_ROAD_MARKING3:
        raise RuntimeError(f"Expected label_mode road_marking3, got {cfg['dataset'].get('label_mode')}")
    normalize_dataset_paths(cfg)
    cfg["dataset"]["sampler"] = {"name": "SemSegRandomSampler"}
    cfg["dataset"]["steps_per_epoch_valid"] = steps
    # Do not reuse the training cache: it does not contain analysis raw labels.
    cfg["dataset"]["use_cache"] = False
    cfg["pipeline"]["batch_size"] = 1
    cfg["pipeline"]["val_batch_size"] = 1
    cfg["pipeline"]["num_workers"] = 0
    cfg["pipeline"]["pin_memory"] = False
    cfg["model"]["ckpt_path"] = None
    cfg["model"]["is_resume"] = False
    return cfg


class PandaSetFFLane3DatasetWithRaw(PandaSetFFLane3Dataset):
    """Analysis-only dataset variant that returns raw PandaSet labels too."""

    def _load_sample(self, seq_id: str, frame_idx: int) -> dict[str, np.ndarray]:
        seq = self._data[seq_id]
        try:
            seq.load_lidar().load_semseg()
        except Exception:
            seq.load_lidar()
            seq.load_semseg()
        if seq.semseg is None:
            raise RuntimeError(f"Sequence {seq_id} has no semseg available")

        seq.lidar.set_sensor(1)
        if self._sensor_reapply:
            seq.lidar.set_sensor(1)
        pc_df = seq.lidar[frame_idx]
        semseg_df = seq.semseg[frame_idx]
        raw_labels = semseg_df.loc[pc_df.index, "class"].to_numpy(dtype=np.int32)
        labels = remap_raw_pandaset_ids(raw_labels, label_mode=self.label_mode)

        xyz_world = pc_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        pose = seq.lidar.poses[frame_idx]
        xyz_ego = pds_geometry.lidar_points_to_ego(xyz_world, pose).astype(
            np.float32, copy=False
        )
        intensity = pc_df["i"].to_numpy(dtype=np.float32)
        intensity = np.clip(intensity, self.intensity_clip_low, self.intensity_clip_high)
        intensity = (intensity - self.intensity_mean) / self.intensity_std
        feat = intensity[:, None].astype(np.float32, copy=False)

        sample = {
            "point": xyz_ego,
            "feat": feat,
            "label": labels.astype(np.int32, copy=False),
            "raw_label": raw_labels.astype(np.int32, copy=False),
        }
        seq.lidar._data = None
        seq.lidar._poses = None
        seq.lidar._timestamps = None
        seq.semseg._data = None
        gc.collect()
        return sample


def attach_analysis_preprocess_and_transform(model) -> None:  # noqa: ANN001
    """Attach ranges and raw labels while matching RandLA-Net transform logic."""

    def preprocess_with_raw(data, attr):  # noqa: ANN001
        cfg = model.cfg
        points = np.asarray(data["point"][:, 0:3], dtype=np.float32)
        labels = np.asarray(data["label"], dtype=np.int32).reshape((-1,))
        raw_labels = np.asarray(data["raw_label"], dtype=np.int32).reshape((-1,))
        feat = None if data.get("feat") is None else np.asarray(data["feat"], dtype=np.float32)

        if feat is None:
            sub_points, sub_labels = DataProcessing.grid_subsampling(
                points, labels=labels, grid_size=cfg.grid_size
            )
            sub_feat = None
        else:
            sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
                points, features=feat, labels=labels, grid_size=cfg.grid_size
            )
        raw_points, sub_raw_labels = DataProcessing.grid_subsampling(
            points, labels=raw_labels, grid_size=cfg.grid_size
        )
        if sub_points.shape != raw_points.shape or not np.allclose(sub_points, raw_points, atol=1e-5):
            raise RuntimeError(
                "Raw-label and remapped-label grid subsampling produced different point sets; "
                "raw subtype analysis would not be aligned."
            )
        return {
            "point": sub_points,
            "feat": sub_feat,
            "label": sub_labels,
            "raw_label": sub_raw_labels.astype(np.int32, copy=False),
            "search_tree": KDTree(sub_points),
        }

    def transform_with_raw(data, attr, min_possibility_idx=None):  # noqa: ANN001, ARG001
        if torch.utils.data.get_worker_info():
            seedseq = np.random.SeedSequence(
                torch.utils.data.get_worker_info().seed
                + torch.utils.data.get_worker_info().id
            )
            rng = np.random.default_rng(seedseq.spawn(1)[0])
        else:
            rng = model.rng

        cfg = model.cfg
        inputs = {}
        pc = data["point"].copy()
        label = data["label"].copy()
        raw_label = data["raw_label"].copy()
        feat = data["feat"].copy() if data["feat"] is not None else None
        tree = data["search_tree"]
        ego_ranges = np.linalg.norm(pc[:, :3], axis=1).astype(np.float32)

        pc, selected_idxs, _center_point = model.trans_point_sampler(
            pc=pc,
            feat=feat,
            label=label,
            search_tree=tree,
            num_points=model.cfg.num_points,
        )
        label = label[selected_idxs]
        raw_label = raw_label[selected_idxs]
        selected_ranges = ego_ranges[selected_idxs]
        if feat is not None:
            feat = feat[selected_idxs]

        augment_cfg = cfg.get("augment", {}).copy()
        val_augment_cfg = {}
        if "recenter" in augment_cfg:
            val_augment_cfg["recenter"] = augment_cfg.pop("recenter")
        if "normalize" in augment_cfg:
            val_augment_cfg["normalize"] = augment_cfg.pop("normalize")
        model.augmenter.augment(pc, feat, label, val_augment_cfg, seed=rng)

        if attr["split"] in ["training", "train"]:
            raise RuntimeError("D0 sampled error analysis must not run on the training split.")

        if feat is None:
            feat = pc.copy()
        else:
            feat = np.concatenate([pc, feat], axis=1)
        if cfg.in_channels != feat.shape[1]:
            raise RuntimeError(
                "Wrong feature dimension, please update in_channels in config: "
                f"expected {cfg.in_channels}, got {feat.shape[1]}"
            )

        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []
        for i in range(cfg.num_layers):
            neighbour_idx = DataProcessing.knn_search(pc, pc, cfg.num_neighbors)
            sub_points = pc[: pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[: pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            up_i = DataProcessing.knn_search(sub_points, pc, 1)
            input_points.append(pc)
            input_neighbors.append(neighbour_idx.astype(np.int64))
            input_pools.append(pool_i.astype(np.int64))
            input_up_samples.append(up_i.astype(np.int64))
            pc = sub_points

        inputs["coords"] = input_points
        inputs["neighbor_indices"] = input_neighbors
        inputs["sub_idx"] = input_pools
        inputs["interp_idx"] = input_up_samples
        inputs["features"] = feat
        inputs["point_inds"] = selected_idxs
        inputs["labels"] = label.astype(np.int64)
        inputs["raw_labels"] = raw_label.astype(np.int64)
        inputs["ranges"] = selected_ranges.astype(np.float32, copy=False)
        return inputs

    model.preprocess = preprocess_with_raw
    model.transform = transform_with_raw


def load_model(cfg: dict[str, Any], checkpoint: Path, device: torch.device):
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    model = ml3d.models.RandLANet(**cfg["model"])
    attach_analysis_preprocess_and_transform(model)
    model.device = device
    checkpoint_data = torch.load(checkpoint, map_location=device)
    state = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, checkpoint_data


def summarize_values(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def percentile_or_none(values: np.ndarray, percentile: float) -> float | None:
    values = np.asarray(values)
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bucket_for_ranges(ranges: np.ndarray, low: float, high: float) -> np.ndarray:
    if np.isinf(high):
        return ranges >= low
    return (ranges >= low) & (ranges < high)


def init_bucket_stats() -> dict[str, dict[str, Any]]:
    out = {}
    for low, high, label in DISTANCE_BUCKETS:
        out[label] = {
            "range_min_m": low,
            "range_max_m": None if np.isinf(high) else high,
            "active_points": 0,
            "true_marking": 0,
            "marking_tp": 0,
            "marking_to_road": 0,
            "marking_to_other": 0,
            "road_total": 0,
            "road_tp": 0,
            "road_to_marking": 0,
            "other_total": 0,
            "other_tp": 0,
            "other_to_marking": 0,
            "marking_tp_intensity": [],
            "marking_to_road_intensity": [],
            "marking_to_other_intensity": [],
            "road_tp_intensity": [],
            "road_to_marking_intensity": [],
            "other_to_marking_intensity": [],
        }
    return out


def init_raw_distance_stats() -> dict[tuple[int, str], dict[str, Any]]:
    stats = {}
    for raw_id in RAW_MARKING_SUBTYPES:
        for low, high, label in DISTANCE_BUCKETS:
            stats[(raw_id, label)] = {
                "raw_id": raw_id,
                "raw_name": RAW_MARKING_SUBTYPES[raw_id],
                "bucket": label,
                "range_min_m": low,
                "range_max_m": None if np.isinf(high) else high,
                "support": 0,
                "tp": 0,
                "to_road": 0,
                "to_other": 0,
                "tp_intensity": [],
                "missed_intensity": [],
            }
    return stats


def masks_from_labels(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "marking_tp": (y_true == 1) & (y_pred == 1),
        "marking_to_road": (y_true == 1) & (y_pred == 0),
        "marking_to_other": (y_true == 1) & (y_pred == 2),
        "road_tp": (y_true == 0) & (y_pred == 0),
        "road_to_marking": (y_true == 0) & (y_pred == 1),
        "other_tp": (y_true == 2) & (y_pred == 2),
        "other_to_marking": (y_true == 2) & (y_pred == 1),
    }


def bucket_rows_from_stats(bucket_stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for _low, _high, label in DISTANCE_BUCKETS:
        stats = bucket_stats[label]
        true_marking = int(stats["true_marking"])
        marking_tp = int(stats["marking_tp"])
        marking_to_road = int(stats["marking_to_road"])
        marking_to_other = int(stats["marking_to_other"])
        road_total = int(stats["road_total"])
        other_total = int(stats["other_total"])
        predicted_marking = marking_tp + int(stats["road_to_marking"]) + int(stats["other_to_marking"])
        row = {
            "bucket": label,
            "range_min_m": stats["range_min_m"],
            "range_max_m": stats["range_max_m"],
            "active_points": int(stats["active_points"]),
            "true_marking": true_marking,
            "marking_tp": marking_tp,
            "marking_to_road": marking_to_road,
            "marking_to_other": marking_to_other,
            "marking_recall": None if true_marking == 0 else marking_tp / true_marking,
            "marking_precision": None if predicted_marking == 0 else marking_tp / predicted_marking,
            "marking_to_road_rate": None if true_marking == 0 else marking_to_road / true_marking,
            "marking_to_other_rate": None if true_marking == 0 else marking_to_other / true_marking,
            "road_total": road_total,
            "road_tp": int(stats["road_tp"]),
            "road_to_marking": int(stats["road_to_marking"]),
            "road_to_marking_rate": None if road_total == 0 else int(stats["road_to_marking"]) / road_total,
            "other_total": other_total,
            "other_tp": int(stats["other_tp"]),
            "other_to_marking": int(stats["other_to_marking"]),
            "other_to_marking_rate": None if other_total == 0 else int(stats["other_to_marking"]) / other_total,
        }
        for group in (
            "marking_tp",
            "marking_to_road",
            "marking_to_other",
            "road_tp",
            "road_to_marking",
            "other_to_marking",
        ):
            values = (
                np.concatenate(stats[f"{group}_intensity"])
                if stats[f"{group}_intensity"]
                else np.asarray([], dtype=np.float32)
            )
            row[f"{group}_intensity_p25"] = percentile_or_none(values, 25)
            row[f"{group}_intensity_median"] = percentile_or_none(values, 50)
            row[f"{group}_intensity_p75"] = percentile_or_none(values, 75)
        rows.append(row)
    return rows


def raw_subtype_rows(
    raw_stats: dict[int, dict[str, Any]],
    raw_distance_stats: dict[tuple[int, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for raw_id in RAW_MARKING_SUBTYPES:
        stats = raw_stats[raw_id]
        support = int(stats["support"])
        tp = int(stats["tp"])
        to_road = int(stats["to_road"])
        to_other = int(stats["to_other"])
        tp_values = np.concatenate(stats["tp_intensity"]) if stats["tp_intensity"] else np.asarray([])
        missed_values = (
            np.concatenate(stats["missed_intensity"]) if stats["missed_intensity"] else np.asarray([])
        )
        rows.append(
            {
                "raw_id": raw_id,
                "raw_name": RAW_MARKING_SUBTYPES[raw_id],
                "support": support,
                "tp": tp,
                "to_road": to_road,
                "to_other": to_other,
                "recall": None if support == 0 else tp / support,
                "to_road_rate": None if support == 0 else to_road / support,
                "to_other_rate": None if support == 0 else to_other / support,
                "tp_intensity_median": percentile_or_none(tp_values, 50),
                "missed_intensity_median": percentile_or_none(missed_values, 50),
                "tp_intensity_p25": percentile_or_none(tp_values, 25),
                "tp_intensity_p75": percentile_or_none(tp_values, 75),
                "missed_intensity_p25": percentile_or_none(missed_values, 25),
                "missed_intensity_p75": percentile_or_none(missed_values, 75),
            }
        )
    distance_rows = []
    for raw_id in RAW_MARKING_SUBTYPES:
        for _low, _high, label in DISTANCE_BUCKETS:
            stats = raw_distance_stats[(raw_id, label)]
            support = int(stats["support"])
            tp = int(stats["tp"])
            to_road = int(stats["to_road"])
            to_other = int(stats["to_other"])
            tp_values = (
                np.concatenate(stats["tp_intensity"]) if stats["tp_intensity"] else np.asarray([])
            )
            missed_values = (
                np.concatenate(stats["missed_intensity"]) if stats["missed_intensity"] else np.asarray([])
            )
            distance_rows.append(
                {
                    "raw_id": raw_id,
                    "raw_name": RAW_MARKING_SUBTYPES[raw_id],
                    "bucket": label,
                    "range_min_m": stats["range_min_m"],
                    "range_max_m": stats["range_max_m"],
                    "support": support,
                    "tp": tp,
                    "to_road": to_road,
                    "to_other": to_other,
                    "recall": None if support == 0 else tp / support,
                    "to_road_rate": None if support == 0 else to_road / support,
                    "to_other_rate": None if support == 0 else to_other / support,
                    "tp_intensity_median": percentile_or_none(tp_values, 50),
                    "missed_intensity_median": percentile_or_none(missed_values, 50),
                }
            )
    return rows, distance_rows


def plot_histograms(groups: dict[str, list[np.ndarray]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "marking_tp": "#1b9e77",
        "marking_to_road": "#d73027",
        "road_tp": "#5f6368",
        "road_to_marking": "#fdae61",
        "other_to_marking": "#7570b3",
    }
    for group, color in colors.items():
        values = np.concatenate(groups[group]) if groups[group] else np.asarray([])
        if values.size == 0:
            continue
        ax.hist(values, bins=80, range=(0, 114), density=True, alpha=0.42, label=group, color=color)
    ax.set_title("D0 sampled outcome intensity distributions")
    ax.set_xlabel("raw intensity recovered from model input")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "intensity_hist_by_outcome.png", dpi=220)
    plt.close(fig)


def write_readme(
    out_dir: Path,
    summary: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> None:
    top_miss = frame_rows[0] if frame_rows else None
    best_raw = max(raw_rows, key=lambda row: -1 if row["recall"] is None else row["recall"])
    worst_raw = min(raw_rows, key=lambda row: math.inf if row["recall"] is None else row["recall"])
    lines = [
        "# D0 Epoch-18 Sampled Marking Error Analysis",
        "",
        "This is a fresh sampled validation inference pass. It is not the exact original epoch-18 validation sample from training.",
        "",
        "## Provenance",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- script: `{summary['script']}`",
        f"- config: `{summary['config']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- checkpoint_epoch: `{summary['checkpoint_epoch']}`",
        f"- split: `{summary['split']}`",
        f"- steps: `{summary['steps']}`",
        f"- seed: `{summary['seed']}`",
        f"- device: `{summary['device']}`",
        "",
        "## Class Order",
        "",
        "- active class `0`: road",
        "- active class `1`: marking",
        "- active class `2`: other",
        "",
        "## Outputs",
        "",
        "- `summary.json`",
        "- `confusion_matrix.npy`",
        "- `group_intensity_summary.csv`",
        "- `group_range_summary.csv`",
        "- `distance_bucket_summary.csv`",
        "- `frame_error_summary.csv`",
        "- `top_frames_by_marking_to_road.csv`",
        "- `top_frames_by_road_to_marking.csv`",
        "- `top_frames_by_other_to_marking.csv`",
        "- `raw_marking_subtype_summary.csv`",
        "- `distance_raw_marking_subtype_summary.csv`",
        "- `plots/intensity_hist_by_outcome.png`",
        "",
        "## Initial Findings",
        "",
        f"- sampled marking IoU: `{summary['sampled_metrics']['marking_iou']:.6f}`",
        f"- sampled marking precision: `{summary['sampled_metrics']['marking_precision']:.6f}`",
        f"- sampled marking recall: `{summary['sampled_metrics']['marking_recall']:.6f}`",
        f"- predicted/true marking ratio: `{summary['sampled_metrics']['predicted_true_marking_ratio']:.3f}`",
    ]
    if top_miss:
        lines.append(
            f"- worst frame by marking-to-road count: `{top_miss['seq_id']}/{top_miss['frame_idx']}` "
            f"with `{top_miss['marking_to_road']}` marking points predicted as road"
        )
    lines += [
        f"- easiest raw marking subtype by recall: raw `{best_raw['raw_id']}` `{best_raw['raw_name']}` recall `{best_raw['recall']}`",
        f"- hardest raw marking subtype by recall: raw `{worst_raw['raw_id']}` `{worst_raw['raw_name']}` recall `{worst_raw['recall']}`",
        "",
        "Use the plot script in `analysis_code/plot_d0_marking_error_analysis.py` to create the remaining diagnostic figures and `diagnostic_plots.md`.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    device = choose_device(args.device)
    out_dir = args.out_dir.resolve()
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.config, args.steps)
    dataset = PandaSetFFLane3DatasetWithRaw(**cfg["dataset"])
    model, checkpoint = load_model(cfg, args.checkpoint, device)
    split = dataset.get_split(args.split)
    sampler = split.sampler
    model.trans_point_sampler = sampler.get_point_sampler()
    torch_split = TorchDataloader(
        dataset=split,
        preprocess=model.preprocess,
        transform=model.transform,
        sampler=sampler,
        use_cache=False,
        steps_per_epoch=args.steps,
    )

    pipeline = SemanticSegmentation(model=model, dataset=dataset, **cfg["pipeline"])
    batcher = pipeline.get_batcher(device)
    loader = DataLoader(
        torch_split,
        batch_size=1,
        sampler=get_sampler(sampler),
        num_workers=0,
        pin_memory=False,
        collate_fn=batcher.collate_fn,
    )

    intensity_groups: dict[str, list[np.ndarray]] = {name: [] for name in GROUPS}
    range_groups: dict[str, list[np.ndarray]] = {name: [] for name in GROUPS}
    bucket_stats = init_bucket_stats()
    raw_stats = {
        raw_id: {
            "support": 0,
            "tp": 0,
            "to_road": 0,
            "to_other": 0,
            "tp_intensity": [],
            "missed_intensity": [],
        }
        for raw_id in RAW_MARKING_SUBTYPES
    }
    raw_distance_stats = init_raw_distance_stats()
    frame_rows: list[dict[str, Any]] = []
    cm = np.zeros((3, 3), dtype=np.int64)
    raw_remap_mismatch = 0
    raw_remap_checked = 0

    mean = float(dataset.intensity_mean)
    std = float(dataset.intensity_std)

    with torch.no_grad():
        for step, inputs in enumerate(tqdm(loader, desc="sampled_validation"), start=1):
            if hasattr(inputs["data"], "to"):
                inputs["data"].to(device)
            results = model(inputs["data"])
            scores, y_true = filter_valid_label(
                results,
                inputs["data"]["labels"],
                model.cfg.num_classes,
                model.cfg.ignored_label_inds,
                device,
            )
            y_pred = torch.argmax(scores, dim=-1)

            labels_raw_remapped = inputs["data"]["labels"].detach().cpu().numpy().reshape(-1)
            valid_mask = ~np.isin(labels_raw_remapped, np.asarray(model.cfg.ignored_label_inds))
            raw_labels = inputs["data"]["raw_labels"].detach().cpu().numpy().reshape(-1)
            raw_labels_valid = raw_labels[valid_mask]
            features = inputs["data"]["features"].detach().cpu().numpy().reshape(-1, 4)
            ranges = inputs["data"]["ranges"].detach().cpu().numpy().reshape(-1)
            intensity = (features[:, 3] * std + mean).astype(np.float32)

            y_true_np = y_true.detach().cpu().numpy().astype(np.int64).reshape(-1)
            y_pred_np = y_pred.detach().cpu().numpy().astype(np.int64).reshape(-1)
            intensity_valid = intensity[valid_mask]
            ranges_valid = ranges[valid_mask]

            expected_active = remap_raw_pandaset_ids(
                raw_labels_valid.astype(np.int32), label_mode=LABEL_MODE_ROAD_MARKING3
            ) - 1
            mismatch_mask = expected_active != y_true_np
            raw_remap_mismatch += int(mismatch_mask.sum())
            raw_remap_checked += int(y_true_np.size)

            if not (y_true_np.shape == y_pred_np.shape == intensity_valid.shape == ranges_valid.shape == raw_labels_valid.shape):
                raise RuntimeError(
                    "Sampled batch shape mismatch: "
                    f"y_true={y_true_np.shape} y_pred={y_pred_np.shape} "
                    f"intensity={intensity_valid.shape} ranges={ranges_valid.shape} raw={raw_labels_valid.shape}"
                )

            for i in range(3):
                for j in range(3):
                    cm[i, j] += int(((y_true_np == i) & (y_pred_np == j)).sum())

            masks = masks_from_labels(y_true_np, y_pred_np)
            for group in GROUPS:
                intensity_groups[group].append(intensity_valid[masks[group]])
                range_groups[group].append(ranges_valid[masks[group]])

            for low, high, label in DISTANCE_BUCKETS:
                in_bucket = bucket_for_ranges(ranges_valid, low, high)
                stats = bucket_stats[label]
                stats["active_points"] += int(in_bucket.sum())
                stats["true_marking"] += int(((y_true_np == 1) & in_bucket).sum())
                stats["marking_tp"] += int((masks["marking_tp"] & in_bucket).sum())
                stats["marking_to_road"] += int((masks["marking_to_road"] & in_bucket).sum())
                stats["marking_to_other"] += int((masks["marking_to_other"] & in_bucket).sum())
                stats["road_total"] += int(((y_true_np == 0) & in_bucket).sum())
                stats["road_tp"] += int((masks["road_tp"] & in_bucket).sum())
                stats["road_to_marking"] += int((masks["road_to_marking"] & in_bucket).sum())
                stats["other_total"] += int(((y_true_np == 2) & in_bucket).sum())
                stats["other_tp"] += int((masks["other_tp"] & in_bucket).sum())
                stats["other_to_marking"] += int((masks["other_to_marking"] & in_bucket).sum())
                for group in (
                    "marking_tp",
                    "marking_to_road",
                    "marking_to_other",
                    "road_tp",
                    "road_to_marking",
                    "other_to_marking",
                ):
                    stats[f"{group}_intensity"].append(intensity_valid[masks[group] & in_bucket])

            raw_frame_counts: dict[str, int] = {}
            for raw_id in RAW_MARKING_SUBTYPES:
                subtype = raw_labels_valid == raw_id
                subtype_tp = subtype & masks["marking_tp"]
                subtype_to_road = subtype & masks["marking_to_road"]
                subtype_to_other = subtype & masks["marking_to_other"]
                raw_stats[raw_id]["support"] += int(subtype.sum())
                raw_stats[raw_id]["tp"] += int(subtype_tp.sum())
                raw_stats[raw_id]["to_road"] += int(subtype_to_road.sum())
                raw_stats[raw_id]["to_other"] += int(subtype_to_other.sum())
                raw_stats[raw_id]["tp_intensity"].append(intensity_valid[subtype_tp])
                raw_stats[raw_id]["missed_intensity"].append(
                    intensity_valid[subtype_to_road | subtype_to_other]
                )
                raw_frame_counts[f"raw_{raw_id}_true"] = int(subtype.sum())
                raw_frame_counts[f"raw_{raw_id}_tp"] = int(subtype_tp.sum())
                raw_frame_counts[f"raw_{raw_id}_to_road"] = int(subtype_to_road.sum())
                raw_frame_counts[f"raw_{raw_id}_to_other"] = int(subtype_to_other.sum())

                for low, high, label in DISTANCE_BUCKETS:
                    in_bucket = bucket_for_ranges(ranges_valid, low, high)
                    stats = raw_distance_stats[(raw_id, label)]
                    stats["support"] += int((subtype & in_bucket).sum())
                    stats["tp"] += int((subtype_tp & in_bucket).sum())
                    stats["to_road"] += int((subtype_to_road & in_bucket).sum())
                    stats["to_other"] += int((subtype_to_other & in_bucket).sum())
                    stats["tp_intensity"].append(intensity_valid[subtype_tp & in_bucket])
                    stats["missed_intensity"].append(
                        intensity_valid[(subtype_to_road | subtype_to_other) & in_bucket]
                    )

            attr = inputs["attr"]
            seq_id = attr["seq_id"][0] if isinstance(attr["seq_id"], list) else str(attr["seq_id"])
            frame_idx_raw = attr["frame_idx"]
            frame_idx = int(frame_idx_raw[0]) if hasattr(frame_idx_raw, "__len__") else int(frame_idx_raw)
            true_marking = int((y_true_np == 1).sum())
            marking_tp = int(masks["marking_tp"].sum())
            marking_to_road = int(masks["marking_to_road"].sum())
            marking_to_other = int(masks["marking_to_other"].sum())
            road_to_marking = int(masks["road_to_marking"].sum())
            other_to_marking = int(masks["other_to_marking"].sum())
            predicted_marking = marking_tp + road_to_marking + other_to_marking
            marking_union = true_marking + road_to_marking + other_to_marking
            frame_rows.append(
                {
                    "step": step,
                    "seq_id": seq_id,
                    "frame_idx": frame_idx,
                    "active_points": int(y_true_np.size),
                    "true_marking": true_marking,
                    "marking_tp": marking_tp,
                    "marking_to_road": marking_to_road,
                    "marking_to_other": marking_to_other,
                    "road_to_marking": road_to_marking,
                    "other_to_marking": other_to_marking,
                    "predicted_marking": predicted_marking,
                    "marking_precision": None if predicted_marking == 0 else marking_tp / predicted_marking,
                    "marking_recall": None if true_marking == 0 else marking_tp / true_marking,
                    "marking_f1": None if true_marking + predicted_marking == 0 else 2 * marking_tp / (true_marking + predicted_marking),
                    "marking_iou": None if marking_union == 0 else marking_tp / marking_union,
                    "marking_to_road_rate": None if true_marking == 0 else marking_to_road / true_marking,
                    "road_to_marking_rate": None if (y_true_np == 0).sum() == 0 else road_to_marking / int((y_true_np == 0).sum()),
                    "other_to_marking_rate": None if (y_true_np == 2).sum() == 0 else other_to_marking / int((y_true_np == 2).sum()),
                    "marking_tp_intensity_median": percentile_or_none(intensity_valid[masks["marking_tp"]], 50),
                    "marking_to_road_intensity_median": percentile_or_none(intensity_valid[masks["marking_to_road"]], 50),
                    "road_to_marking_intensity_median": percentile_or_none(intensity_valid[masks["road_to_marking"]], 50),
                    "marking_tp_range_median": percentile_or_none(ranges_valid[masks["marking_tp"]], 50),
                    "marking_to_road_range_median": percentile_or_none(ranges_valid[masks["marking_to_road"]], 50),
                    "road_to_marking_range_median": percentile_or_none(ranges_valid[masks["road_to_marking"]], 50),
                    **raw_frame_counts,
                }
            )

    mismatch_rate = 0.0 if raw_remap_checked == 0 else raw_remap_mismatch / raw_remap_checked
    if mismatch_rate > args.max_raw_remap_mismatch_rate:
        raise RuntimeError(
            "Raw subtype labels do not align with remapped sampled labels: "
            f"mismatch_rate={mismatch_rate:.6f}, threshold={args.max_raw_remap_mismatch_rate}"
        )

    group_intensity_rows = []
    group_range_rows = []
    for group in GROUPS:
        intensity_values = (
            np.concatenate(intensity_groups[group]) if intensity_groups[group] else np.asarray([])
        )
        range_values = np.concatenate(range_groups[group]) if range_groups[group] else np.asarray([])
        group_intensity_rows.append({"group": group, **summarize_values(intensity_values)})
        group_range_rows.append({"group": group, **summarize_values(range_values)})

    distance_rows = bucket_rows_from_stats(bucket_stats)
    raw_rows, raw_distance_rows = raw_subtype_rows(raw_stats, raw_distance_stats)
    frame_rows_by_marking_to_road = sorted(
        frame_rows,
        key=lambda row: (int(row["marking_to_road"]), int(row["true_marking"])),
        reverse=True,
    )
    frame_rows_by_road_to_marking = sorted(
        frame_rows,
        key=lambda row: int(row["road_to_marking"]),
        reverse=True,
    )
    frame_rows_by_other_to_marking = sorted(
        frame_rows,
        key=lambda row: int(row["other_to_marking"]),
        reverse=True,
    )

    write_csv(out_dir / "frame_error_summary.csv", frame_rows)
    write_csv(out_dir / "top_frames_by_marking_to_road.csv", frame_rows_by_marking_to_road[:50])
    write_csv(out_dir / "top_frames_by_road_to_marking.csv", frame_rows_by_road_to_marking[:50])
    write_csv(out_dir / "top_frames_by_other_to_marking.csv", frame_rows_by_other_to_marking[:50])
    write_csv(out_dir / "group_intensity_summary.csv", group_intensity_rows)
    write_csv(out_dir / "group_range_summary.csv", group_range_rows)
    write_csv(out_dir / "distance_bucket_summary.csv", distance_rows)
    write_csv(out_dir / "raw_marking_subtype_summary.csv", raw_rows)
    write_csv(out_dir / "distance_raw_marking_subtype_summary.csv", raw_distance_rows)
    np.save(out_dir / "confusion_matrix.npy", cm)
    plot_histograms(intensity_groups, plots_dir)

    true_marking = int(cm[1, :].sum())
    pred_marking = int(cm[:, 1].sum())
    marking_tp = int(cm[1, 1])
    marking_fp = pred_marking - marking_tp
    marking_fn = true_marking - marking_tp
    marking_precision = float("nan") if pred_marking == 0 else marking_tp / pred_marking
    marking_recall = float("nan") if true_marking == 0 else marking_tp / true_marking
    marking_iou = float("nan") if marking_tp + marking_fp + marking_fn == 0 else marking_tp / (marking_tp + marking_fp + marking_fn)
    marking_f1 = float("nan") if true_marking + pred_marking == 0 else 2 * marking_tp / (true_marking + pred_marking)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "split": args.split,
        "steps": args.steps,
        "seed": args.seed,
        "device": str(device),
        "note": "Fresh sampled inference pass; not exact original D0 epoch-18 validation samples.",
        "class_order": {"0": "road", "1": "marking", "2": "other"},
        "raw_remap_alignment": {
            "checked_points": raw_remap_checked,
            "mismatched_points": raw_remap_mismatch,
            "mismatch_rate": mismatch_rate,
            "threshold": args.max_raw_remap_mismatch_rate,
        },
        "sampled_metrics": {
            "marking_iou": marking_iou,
            "marking_precision": marking_precision,
            "marking_recall": marking_recall,
            "marking_f1": marking_f1,
            "true_marking": true_marking,
            "predicted_marking": pred_marking,
            "predicted_true_marking_ratio": float("nan") if true_marking == 0 else pred_marking / true_marking,
        },
        "confusion_matrix": cm.tolist(),
        "group_intensity_summary": group_intensity_rows,
        "group_range_summary": group_range_rows,
        "distance_bucket_summary": distance_rows,
        "raw_marking_subtype_summary": raw_rows,
        "top_frame_by_marking_to_road": frame_rows_by_marking_to_road[0] if frame_rows_by_marking_to_road else None,
        "top_frame_by_road_to_marking": frame_rows_by_road_to_marking[0] if frame_rows_by_road_to_marking else None,
        "top_frame_by_other_to_marking": frame_rows_by_other_to_marking[0] if frame_rows_by_other_to_marking else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_readme(out_dir, summary, frame_rows_by_marking_to_road, raw_rows)
    print(f"wrote {out_dir}")
    print("script_status PASS")


if __name__ == "__main__":
    main()
