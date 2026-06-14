#!/usr/bin/env python
"""Sampled F0 epoch-13 error analysis with RGB-valid stratification.

This runs a fresh sampled validation inference pass for the F0 best checkpoint.
It is not the exact epoch-13 validation sample from training. The purpose is to
answer the key F0 diagnostic questions:

    Did softened class weighting reduce RGB-valid road-to-marking errors?
    Are remaining false positives still concentrated where RGB is valid?

Outputs are written under:

    logs/milestone_f/run_analysis/F0_rgb_soft_weights/sampled_error_analysis_epoch13/
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

import numpy as np
import torch
import yaml
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from tqdm import tqdm


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "logs/milestone_f/configs/f0_rgb_soft_weights.yml").exists():
            return parent
    raise RuntimeError("Could not find project root from analysis script path")


PROJECT_ROOT = find_project_root()
PATCHED_DEVKIT = PROJECT_ROOT / "pandaset-devkit/python"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PATCHED_DEVKIT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import open3d.ml.torch as ml3d  # noqa: E402
from pandaset import geometry as pds_geometry  # noqa: E402
from open3d._ml3d.datasets.utils import DataProcessing  # noqa: E402
from open3d._ml3d.torch.dataloaders import TorchDataloader, get_sampler  # noqa: E402
from open3d._ml3d.torch.modules.losses.semseg_loss import filter_valid_label  # noqa: E402
from open3d._ml3d.torch.pipelines import SemanticSegmentation  # noqa: E402

from thesis_pipeline.adapters.pandaset_ff_lane3 import (  # noqa: E402
    FEATURE_MODE_INTENSITY_RGB_FRONT,
    LABEL_MODE_ROAD_MARKING3,
    RAW_OTHER_ROAD_MARKING_ID,
    RAW_STOP_LINE_ID,
    remap_raw_pandaset_ids,
)
from thesis_pipeline.datasets.pandaset_ff_lane3_dataset import (  # noqa: E402
    PandaSetFFLane3Dataset,
)


RUN_NAME = "F0_rgb_soft_weights"
DEFAULT_CONFIG = PROJECT_ROOT / "logs/milestone_f/configs/f0_rgb_soft_weights.yml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / f"logs/milestone_f/runs/{RUN_NAME}/checkpoints/ckpt_epoch_00013.pth"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / f"logs/milestone_f/run_analysis/{RUN_NAME}/sampled_error_analysis_epoch13"
)

CLASS_NAMES = ("road", "marking", "other")
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
    (60.0, math.inf, "60m_plus"),
)
GROUPS = (
    "marking_tp",
    "marking_to_road",
    "marking_to_other",
    "road_tp",
    "road_to_marking",
    "other_tp",
    "other_to_marking",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--steps", type=int, default=2160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--rgb-valid-threshold", type=float, default=0.5)
    parser.add_argument("--max-raw-remap-mismatch-rate", type=float, default=0.001)
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


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def repo_path(value: str | None, fallback: str) -> str:
    fallback_path = PROJECT_ROOT / fallback
    if value is None:
        return str(fallback_path)
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def load_cfg(path: Path, steps: int) -> dict[str, Any]:
    require_file(path)
    cfg = yaml.safe_load(path.read_text())
    dataset_cfg = cfg["dataset"]
    if dataset_cfg.get("label_mode") != LABEL_MODE_ROAD_MARKING3:
        raise RuntimeError(f"Expected road_marking3, got {dataset_cfg.get('label_mode')}")
    if dataset_cfg.get("feature_mode") != FEATURE_MODE_INTENSITY_RGB_FRONT:
        raise RuntimeError(
            f"Expected intensity_rgb_front, got {dataset_cfg.get('feature_mode')}"
        )

    dataset_cfg["dataset_path"] = repo_path(dataset_cfg.get("dataset_path"), "pandaset/PandaSet")
    dataset_cfg["split_dir"] = repo_path(dataset_cfg.get("split_dir"), "configs/splits")
    dataset_cfg["stats_file"] = repo_path(
        dataset_cfg.get("stats_file"), "logs/milestone_d/road_marking3_training_statistics.json"
    )
    dataset_cfg["dataset_root_file"] = repo_path(
        dataset_cfg.get("dataset_root_file"), "logs/dataset_root.txt"
    )
    dataset_cfg["preflight_pattern_file"] = repo_path(
        dataset_cfg.get("preflight_pattern_file"),
        "logs/milestone_b_preflight_sensor_pattern.txt",
    )
    dataset_cfg["cache_dir"] = repo_path(
        dataset_cfg.get("cache_dir"), "logs/milestone_f/cache/F0_rgb_soft_weights_v1"
    )
    dataset_cfg["test_result_folder"] = repo_path(
        dataset_cfg.get("test_result_folder"), "logs/milestone_f/test_results/F0_rgb_soft_weights"
    )
    dataset_cfg["sampler"] = {"name": "SemSegRandomSampler"}
    dataset_cfg["steps_per_epoch_valid"] = steps
    # Do not use Open3D preprocessing cache here: this analysis adds raw labels
    # and RGB-valid side channels that are not part of the training cache.
    dataset_cfg["use_cache"] = False

    cfg["pipeline"]["batch_size"] = 1
    cfg["pipeline"]["val_batch_size"] = 1
    cfg["pipeline"]["num_workers"] = 0
    cfg["pipeline"]["pin_memory"] = False
    cfg["model"]["ckpt_path"] = None
    cfg["model"]["is_resume"] = False
    return cfg


class F0DatasetWithRaw(PandaSetFFLane3Dataset):
    """Analysis-only F0 dataset that returns raw labels alongside RGB features."""

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
        intensity_col = intensity[:, None].astype(np.float32, copy=False)

        rgb, rgb_valid = self._compute_rgb_features(seq_id, frame_idx, xyz_world)
        feat = np.concatenate([intensity_col, rgb, rgb_valid[:, None]], axis=1).astype(
            np.float32, copy=False
        )

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
    def preprocess_with_raw(data, attr):  # noqa: ANN001, ARG001
        cfg = model.cfg
        points = np.asarray(data["point"][:, 0:3], dtype=np.float32)
        labels = np.asarray(data["label"], dtype=np.int32).reshape((-1,))
        raw_labels = np.asarray(data["raw_label"], dtype=np.int32).reshape((-1,))
        feat = np.asarray(data["feat"], dtype=np.float32)

        sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
            points, features=feat, labels=labels, grid_size=cfg.grid_size
        )
        raw_points, sub_raw_labels = DataProcessing.grid_subsampling(
            points, labels=raw_labels, grid_size=cfg.grid_size
        )
        if sub_points.shape != raw_points.shape or not np.allclose(
            sub_points, raw_points, atol=1e-5
        ):
            raise RuntimeError("Raw-label and feature grid subsampling are not aligned.")

        return {
            "point": sub_points,
            "feat": sub_feat,
            "label": sub_labels,
            "raw_label": sub_raw_labels.astype(np.int32, copy=False),
            "search_tree": KDTree(sub_points),
        }

    def transform_with_raw(data, attr, min_possibility_idx=None):  # noqa: ANN001, ARG001
        rng = model.rng
        cfg = model.cfg
        pc = data["point"].copy()
        label = data["label"].copy()
        raw_label = data["raw_label"].copy()
        feat = data["feat"].copy()
        tree = data["search_tree"]
        ranges = np.linalg.norm(pc[:, :3], axis=1).astype(np.float32)

        pc, selected_idxs, _center_point = model.trans_point_sampler(
            pc=pc,
            feat=feat,
            label=label,
            search_tree=tree,
            num_points=model.cfg.num_points,
        )
        label = label[selected_idxs]
        raw_label = raw_label[selected_idxs]
        ranges = ranges[selected_idxs]
        feat = feat[selected_idxs]

        analysis_intensity_z = feat[:, 0].astype(np.float32, copy=False)
        analysis_rgb = feat[:, 1:4].astype(np.float32, copy=False)
        analysis_rgb_valid = feat[:, 4].astype(np.float32, copy=False)

        augment_cfg = cfg.get("augment", {}).copy()
        val_augment_cfg = {}
        if "recenter" in augment_cfg:
            val_augment_cfg["recenter"] = augment_cfg.pop("recenter")
        if "normalize" in augment_cfg:
            val_augment_cfg["normalize"] = augment_cfg.pop("normalize")
        model.augmenter.augment(pc, feat, label, val_augment_cfg, seed=rng)
        if attr["split"] in ["training", "train"]:
            raise RuntimeError("F0 sampled error analysis must not run on training split.")

        model_features = np.concatenate([pc, feat], axis=1)
        if cfg.in_channels != model_features.shape[1]:
            raise RuntimeError(
                f"Wrong feature dimension: expected {cfg.in_channels}, "
                f"got {model_features.shape[1]}"
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

        return {
            "coords": input_points,
            "neighbor_indices": input_neighbors,
            "sub_idx": input_pools,
            "interp_idx": input_up_samples,
            "features": model_features,
            "point_inds": selected_idxs,
            "labels": label.astype(np.int64),
            "raw_labels": raw_label.astype(np.int64),
            "ranges": ranges,
            "analysis_intensity_z": analysis_intensity_z,
            "analysis_rgb": analysis_rgb,
            "analysis_rgb_valid": analysis_rgb_valid,
        }

    model.preprocess = preprocess_with_raw
    model.transform = transform_with_raw


def load_model(cfg: dict[str, Any], checkpoint: Path, device: torch.device):
    require_file(checkpoint)
    model = ml3d.models.RandLANet(**cfg["model"])
    attach_analysis_preprocess_and_transform(model)
    model.device = device
    checkpoint_data = torch.load(checkpoint, map_location=device)
    state = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, checkpoint_data


def metrics_from_cm(cm: np.ndarray) -> dict[str, float | int]:
    tp = int(cm[1, 1])
    true_marking = int(cm[1, :].sum())
    pred_marking = int(cm[:, 1].sum())
    fp = pred_marking - tp
    fn = true_marking - tp
    precision = float("nan") if pred_marking == 0 else tp / pred_marking
    recall = float("nan") if true_marking == 0 else tp / true_marking
    f1 = float("nan") if true_marking + pred_marking == 0 else 2 * tp / (
        true_marking + pred_marking
    )
    iou = float("nan") if tp + fp + fn == 0 else tp / (tp + fp + fn)
    return {
        "active_points": int(cm.sum()),
        "true_marking": true_marking,
        "predicted_marking": pred_marking,
        "marking_tp": tp,
        "marking_fp": int(fp),
        "marking_fn": int(fn),
        "marking_iou": iou,
        "marking_precision": precision,
        "marking_recall": recall,
        "marking_f1": f1,
        "predicted_true_marking_ratio": (
            float("nan") if true_marking == 0 else pred_marking / true_marking
        ),
        "road_to_marking": int(cm[0, 1]),
        "marking_to_road": int(cm[1, 0]),
        "other_to_marking": int(cm[2, 1]),
    }


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


def summarize(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"count": 0, "mean": None, "p25": None, "median": None, "p75": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def bucket_name(ranges: np.ndarray) -> np.ndarray:
    names = np.full(ranges.shape, "60m_plus", dtype=object)
    for low, high, name in DISTANCE_BUCKETS:
        if math.isinf(high):
            mask = ranges >= low
        else:
            mask = (ranges >= low) & (ranges < high)
        names[mask] = name
    return names


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    device = choose_device(args.device)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.config, args.steps)
    dataset = F0DatasetWithRaw(**cfg["dataset"])
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

    cm_all = np.zeros((3, 3), dtype=np.int64)
    cm_by_rgb = {
        "rgb_valid": np.zeros((3, 3), dtype=np.int64),
        "rgb_invalid": np.zeros((3, 3), dtype=np.int64),
    }
    cm_by_seq: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    cm_by_bucket: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    raw_rows_counter: dict[tuple[int, str], np.ndarray] = defaultdict(
        lambda: np.zeros((3, 3), dtype=np.int64)
    )
    group_values: dict[str, dict[str, list[np.ndarray]]] = {
        group: {
            "intensity": [],
            "rgb_valid": [],
            "red": [],
            "green": [],
            "blue": [],
            "range": [],
        }
        for group in GROUPS
    }
    frame_rows: list[dict[str, Any]] = []
    raw_remap_mismatch = 0
    raw_remap_checked = 0

    mean = float(dataset.intensity_mean)
    std = float(dataset.intensity_std)

    with torch.no_grad():
        for step, inputs in enumerate(tqdm(loader, desc="f0_sampled_validation"), start=1):
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
            ranges = inputs["data"]["ranges"].detach().cpu().numpy().reshape(-1)
            intensity_z = (
                inputs["data"]["analysis_intensity_z"].detach().cpu().numpy().reshape(-1)
            )
            rgb = inputs["data"]["analysis_rgb"].detach().cpu().numpy().reshape(-1, 3)
            rgb_valid = (
                inputs["data"]["analysis_rgb_valid"].detach().cpu().numpy().reshape(-1)
            )

            y_true_np = y_true.detach().cpu().numpy().astype(np.int64).reshape(-1)
            y_pred_np = y_pred.detach().cpu().numpy().astype(np.int64).reshape(-1)
            ranges_valid = ranges[valid_mask]
            intensity_valid = (intensity_z[valid_mask] * std + mean).astype(np.float32)
            rgb_valid_values = rgb_valid[valid_mask]
            rgb_valid_mask = rgb_valid_values > args.rgb_valid_threshold
            rgb_values = rgb[valid_mask]

            expected_active = remap_raw_pandaset_ids(
                raw_labels_valid.astype(np.int32), label_mode=LABEL_MODE_ROAD_MARKING3
            ) - 1
            mismatch = expected_active != y_true_np
            raw_remap_mismatch += int(mismatch.sum())
            raw_remap_checked += int(y_true_np.size)

            if not (
                y_true_np.shape
                == y_pred_np.shape
                == ranges_valid.shape
                == intensity_valid.shape
                == rgb_valid_values.shape
            ):
                raise RuntimeError("Sampled batch shape mismatch.")

            for i in range(3):
                for j in range(3):
                    mask = (y_true_np == i) & (y_pred_np == j)
                    cm_all[i, j] += int(mask.sum())
                    cm_by_rgb["rgb_valid"][i, j] += int((mask & rgb_valid_mask).sum())
                    cm_by_rgb["rgb_invalid"][i, j] += int((mask & ~rgb_valid_mask).sum())

            attr = inputs["attr"]
            seq_id = attr["seq_id"][0] if isinstance(attr["seq_id"], list) else str(attr["seq_id"])
            frame_idx_raw = attr["frame_idx"]
            frame_idx = int(frame_idx_raw[0]) if hasattr(frame_idx_raw, "__len__") else int(frame_idx_raw)
            cm_by_seq[seq_id] += cm_all * 0  # ensure key exists without sharing cm_all
            for i in range(3):
                for j in range(3):
                    cm_by_seq[seq_id][i, j] += int(((y_true_np == i) & (y_pred_np == j)).sum())

            buckets = bucket_name(ranges_valid)
            for name in np.unique(buckets):
                in_bucket = buckets == name
                for i in range(3):
                    for j in range(3):
                        cm_by_bucket[str(name)][i, j] += int(
                            ((y_true_np == i) & (y_pred_np == j) & in_bucket).sum()
                        )

            masks = masks_from_labels(y_true_np, y_pred_np)
            for group, mask in masks.items():
                group_values[group]["intensity"].append(intensity_valid[mask])
                group_values[group]["rgb_valid"].append(rgb_valid_values[mask])
                group_values[group]["red"].append(rgb_values[mask, 0])
                group_values[group]["green"].append(rgb_values[mask, 1])
                group_values[group]["blue"].append(rgb_values[mask, 2])
                group_values[group]["range"].append(ranges_valid[mask])

            for raw_id in RAW_MARKING_SUBTYPES:
                raw_mask = raw_labels_valid == raw_id
                for stratum, stratum_mask in (
                    ("all", np.ones_like(raw_mask, dtype=bool)),
                    ("rgb_valid", rgb_valid_mask),
                    ("rgb_invalid", ~rgb_valid_mask),
                ):
                    combined = raw_mask & stratum_mask
                    cm = raw_rows_counter[(raw_id, stratum)]
                    for i in range(3):
                        for j in range(3):
                            cm[i, j] += int(((y_true_np == i) & (y_pred_np == j) & combined).sum())

            frame_cm = np.zeros((3, 3), dtype=np.int64)
            for i in range(3):
                for j in range(3):
                    frame_cm[i, j] = int(((y_true_np == i) & (y_pred_np == j)).sum())
            frame_metrics = metrics_from_cm(frame_cm)
            frame_rows.append(
                {
                    "step": step,
                    "seq_id": seq_id,
                    "frame_idx": frame_idx,
                    "active_points": int(y_true_np.size),
                    "rgb_valid_ratio": float(rgb_valid_mask.mean()) if y_true_np.size else 0.0,
                    **frame_metrics,
                }
            )

    mismatch_rate = 0.0 if raw_remap_checked == 0 else raw_remap_mismatch / raw_remap_checked
    if mismatch_rate > args.max_raw_remap_mismatch_rate:
        raise RuntimeError(
            f"raw remap mismatch_rate={mismatch_rate:.6f} exceeds "
            f"{args.max_raw_remap_mismatch_rate}"
        )

    summary_rows = []
    for stratum, cm in (("all", cm_all), *cm_by_rgb.items()):
        summary_rows.append({"stratum": stratum, **metrics_from_cm(cm)})
    write_csv(out_dir / "rgb_valid_stratified_metrics.csv", summary_rows)

    seq_rows = []
    for seq_id, cm in sorted(cm_by_seq.items()):
        seq_rows.append({"seq_id": seq_id, **metrics_from_cm(cm)})
    write_csv(out_dir / "per_sequence_metrics.csv", seq_rows)

    bucket_rows = []
    for name, cm in sorted(cm_by_bucket.items()):
        bucket_rows.append({"bucket": name, **metrics_from_cm(cm)})
    write_csv(out_dir / "distance_bucket_metrics.csv", bucket_rows)

    raw_rows = []
    for (raw_id, stratum), cm in sorted(raw_rows_counter.items()):
        raw_rows.append(
            {
                "raw_id": raw_id,
                "raw_name": RAW_MARKING_SUBTYPES[raw_id],
                "stratum": stratum,
                **metrics_from_cm(cm),
            }
        )
    write_csv(out_dir / "raw_subtype_rgb_stratified_metrics.csv", raw_rows)

    group_rows = []
    for group in GROUPS:
        row: dict[str, Any] = {"group": group}
        for field in ("intensity", "rgb_valid", "red", "green", "blue", "range"):
            values = (
                np.concatenate(group_values[group][field])
                if group_values[group][field]
                else np.asarray([])
            )
            for key, value in summarize(values).items():
                row[f"{field}_{key}"] = value
        group_rows.append(row)
    write_csv(out_dir / "group_feature_summary.csv", group_rows)
    write_csv(out_dir / "frame_error_summary.csv", frame_rows)
    write_csv(
        out_dir / "top_frames_by_road_to_marking.csv",
        sorted(frame_rows, key=lambda row: int(row["road_to_marking"]), reverse=True)[:50],
    )
    write_csv(
        out_dir / "top_frames_by_marking_to_road.csv",
        sorted(frame_rows, key=lambda row: int(row["marking_to_road"]), reverse=True)[:50],
    )
    np.save(out_dir / "confusion_matrix.npy", cm_all)

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
        "rgb_valid_threshold": args.rgb_valid_threshold,
        "note": "Fresh sampled inference pass; not exact original F0 epoch-13 validation sample.",
        "class_order": {"0": "road", "1": "marking", "2": "other"},
        "raw_remap_alignment": {
            "checked_points": raw_remap_checked,
            "mismatched_points": raw_remap_mismatch,
            "mismatch_rate": mismatch_rate,
            "threshold": args.max_raw_remap_mismatch_rate,
        },
        "sampled_metrics": metrics_from_cm(cm_all),
        "rgb_valid_metrics": metrics_from_cm(cm_by_rgb["rgb_valid"]),
        "rgb_invalid_metrics": metrics_from_cm(cm_by_rgb["rgb_invalid"]),
        "confusion_matrix": cm_all.tolist(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    readme = [
        "# F0 Epoch-13 Sampled Error Analysis",
        "",
        "Fresh sampled validation inference pass. This is not the exact training-time",
        "epoch-13 validation sample.",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- steps: `{args.steps}`",
        f"- seed: `{args.seed}`",
        f"- device: `{device}`",
        "",
        "## Main Metrics",
        "",
        "| stratum | marking IoU | precision | recall | pred/true | road->marking | marking->road |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        readme.append(
            f"| {row['stratum']} | {row['marking_iou']:.6f} | "
            f"{row['marking_precision']:.6f} | {row['marking_recall']:.6f} | "
            f"{row['predicted_true_marking_ratio']:.3f} | "
            f"{int(row['road_to_marking'])} | {int(row['marking_to_road'])} |"
        )
    readme.extend(
        [
            "",
            "Interpretation rule:",
            "",
            "- If `rgb_valid` still has most road->marking errors, remaining false",
            "  positives are still tied to valid front-camera RGB regions.",
            "- If `rgb_invalid` is disproportionately bad, invalid-RGB fallback behavior",
            "  is a major issue.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(readme))

    print("F0 sampled error analysis complete")
    print(f"out_dir {out_dir}")
    print("script_status PASS")


if __name__ == "__main__":
    main()
