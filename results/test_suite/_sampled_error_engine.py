#!/usr/bin/env python
"""Sampled G0 best-epoch error analysis with RGB-valid stratification.

Adapted from the Milestone F0 sampled error analysis. The inference and
stratification machinery is loss-agnostic (predictions are argmax of model
scores; the loss is never invoked), so the only changes versus F0 are:

  - paths point at Milestone G,
  - the checkpoint epoch is discovered dynamically from G0 ``eval_history.csv``
    (max raw marking IoU), instead of being hard-coded,
  - the output subfolder is named ``sampled_error_analysis_epoch{best}``.

This runs a fresh sampled validation inference pass for the G0 best checkpoint.
It is not the exact best-epoch validation sample from training. The purpose is
to answer the key residual-error questions and keep G comparable to F0:

    Did adding Lovasz change RGB-valid road-to-marking overprediction?
    Are remaining false positives still concentrated where RGB is valid?

Outputs are written under:

    logs/milestone_g/run_analysis/G0_rgb_lovasz/sampled_error_analysis_epoch{best}/
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
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from tqdm import tqdm


def find_project_root() -> Path:
    # Generic: repo root = the dir containing src/thesis_pipeline/ and logs/.
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "thesis_pipeline").is_dir() and (parent / "logs").is_dir():
            return parent
    raise RuntimeError("Could not find project root (dir with src/thesis_pipeline/ and logs/)")


PROJECT_ROOT = find_project_root()
PATCHED_DEVKIT = PROJECT_ROOT / "pandaset-devkit/python"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PATCHED_DEVKIT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import open3d.ml.torch as ml3d  # noqa: E402
from pandaset import geometry as pds_geometry  # noqa: E402
from open3d._ml3d.datasets.utils import DataProcessing  # noqa: E402
from open3d._ml3d.datasets.samplers.semseg_random import SemSegRandomSampler  # noqa: E402
from open3d._ml3d.datasets.samplers.semseg_spatially_regular import (  # noqa: E402
    SemSegSpatiallyRegularSampler,
)
from open3d._ml3d.torch.dataloaders import TorchDataloader, get_sampler  # noqa: E402
from open3d._ml3d.torch.modules.losses.semseg_loss import filter_valid_label  # noqa: E402
from open3d._ml3d.torch.pipelines import SemanticSegmentation  # noqa: E402

from thesis_pipeline.adapters.pandaset_ff_lane3 import (  # noqa: E402
    FEATURE_MODE_INTENSITY,
    FEATURE_MODE_INTENSITY_RGB_FRONT,
    LABEL_MODE_ROAD_MARKING3,
    RAW_OTHER_ROAD_MARKING_ID,
    RAW_STOP_LINE_ID,
    remap_raw_pandaset_ids,
)
from thesis_pipeline.datasets.pandaset_ff_lane3_dataset import (  # noqa: E402
    PandaSetFFLane3Dataset,
)


RUN_NAME = "G0_rgb_lovasz"
DEFAULT_CONFIG = PROJECT_ROOT / "logs/milestone_g/configs/g0_rgb_lovasz.yml"
G0_RUN_DIR = PROJECT_ROOT / f"logs/milestone_g/runs/{RUN_NAME}"
ANALYSIS_DIR = PROJECT_ROOT / f"logs/milestone_g/run_analysis/{RUN_NAME}"

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


def discover_best_epoch(run_dir: Path) -> int:
    """Best epoch = max raw marking IoU (lane_iou), same rule as checkpoint selection."""
    history_path = run_dir / "eval_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(
            f"{history_path} not found; cannot discover G0 best epoch. "
            "Run G0 to completion first."
        )
    df = pd.read_csv(history_path)
    if "lane_iou" not in df.columns:
        raise ValueError(f"{history_path} has no lane_iou column.")
    return int(df.loc[df["lane_iou"].idxmax(), "epoch"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint .pth. Default: discovered best marking-IoU epoch of G0.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir. Default: sampled_error_analysis_epoch{best}.",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--coverage",
        choices=("sampled", "full"),
        default="sampled",
        help=(
            "sampled = --steps random patches across all frames (matches the "
            "validation diagnostics); full = spatially-regular FULL coverage of "
            "every frame (headline final-evaluation protocol, ~8 patches/frame). "
            "full also emits coverage-count, agreement-rate and voted-CM diagnostics."
        ),
    )
    parser.add_argument("--steps", type=int, default=2160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--rgb-valid-threshold", type=float, default=0.5)
    parser.add_argument("--max-raw-remap-mismatch-rate", type=float, default=0.001)
    return parser.parse_args()


def resolve_checkpoint_and_outdir(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
        # Best-effort epoch parse for the default out-dir name.
        try:
            epoch = int(checkpoint.stem.split("_")[-1])
        except ValueError:
            epoch = discover_best_epoch(G0_RUN_DIR)
    else:
        epoch = discover_best_epoch(G0_RUN_DIR)
        checkpoint = G0_RUN_DIR / "checkpoints" / f"ckpt_epoch_{epoch:05d}.pth"
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else ANALYSIS_DIR / f"sampled_error_analysis_epoch{epoch}"
    )
    return checkpoint, out_dir


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
    # Generalized for the test suite: accept BOTH the RGB models (E0/F0/G2/H0,
    # feature_mode=intensity_rgb_front, 8-channel) AND the LiDAR-only baseline D0
    # (feature_mode=intensity, 4-channel). D0 has no RGB/rgb_valid, so the
    # RGB-specific diagnostics are skipped for it (see has_rgb gating below);
    # core metrics, confusion, distance and subtype are computed for both.
    feature_mode = dataset_cfg.get("feature_mode") or FEATURE_MODE_INTENSITY
    if feature_mode not in (FEATURE_MODE_INTENSITY_RGB_FRONT, FEATURE_MODE_INTENSITY):
        raise RuntimeError(
            f"Unsupported feature_mode {feature_mode!r}; expected "
            f"{FEATURE_MODE_INTENSITY_RGB_FRONT!r} or {FEATURE_MODE_INTENSITY!r}"
        )
    dataset_cfg["feature_mode"] = feature_mode  # normalize so the dataset gets a valid string

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
        dataset_cfg.get("cache_dir"), "logs/milestone_g/cache/G0_rgb_lovasz_v1"
    )
    dataset_cfg["test_result_folder"] = repo_path(
        dataset_cfg.get("test_result_folder"), "logs/milestone_g/test_results/G0_rgb_lovasz"
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
    # The loss is irrelevant for inference; drop it so the stock pipeline init
    # never tries to build a combined loss during a forward-only analysis pass.
    cfg["pipeline"].pop("loss", None)
    cfg["model"]["ckpt_path"] = None
    cfg["model"]["is_resume"] = False
    return cfg


class G0DatasetWithRaw(PandaSetFFLane3Dataset):
    """Analysis-only G0 dataset that returns raw labels alongside RGB features."""

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

        if self.feature_mode == FEATURE_MODE_INTENSITY_RGB_FRONT:
            rgb, rgb_valid = self._compute_rgb_features(seq_id, frame_idx, xyz_world)
            feat = np.concatenate([intensity_col, rgb, rgb_valid[:, None]], axis=1).astype(
                np.float32, copy=False
            )
        else:
            # LiDAR-only (D0): intensity is the only feature channel (4-ch with xyz).
            feat = intensity_col

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
        if feat.shape[1] >= 5:  # RGB models: [intensity, r, g, b, rgb_valid]
            analysis_rgb = feat[:, 1:4].astype(np.float32, copy=False)
            analysis_rgb_valid = feat[:, 4].astype(np.float32, copy=False)
        else:  # LiDAR-only (D0): no RGB — dummy zeros keep the loop interface intact;
            # the RGB-specific outputs are not written for non-RGB models (has_rgb gating).
            analysis_rgb = np.zeros((feat.shape[0], 3), dtype=np.float32)
            analysis_rgb_valid = np.zeros(feat.shape[0], dtype=np.float32)

        augment_cfg = cfg.get("augment", {}).copy()
        val_augment_cfg = {}
        if "recenter" in augment_cfg:
            val_augment_cfg["recenter"] = augment_cfg.pop("recenter")
        if "normalize" in augment_cfg:
            val_augment_cfg["normalize"] = augment_cfg.pop("normalize")
        model.augmenter.augment(pc, feat, label, val_augment_cfg, seed=rng)
        if attr["split"] in ["training", "train"]:
            raise RuntimeError("G0 sampled error analysis must not run on training split.")

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


def hist_percentiles(counter: Counter) -> dict[str, int | None]:
    """median / p10 / p90 / max of a value->frequency histogram (coverage counts)."""
    if not counter:
        return {"count": 0, "median": None, "p10": None, "p90": None, "max": None}
    vals = np.array(sorted(counter), dtype=np.int64)
    freqs = np.array([counter[int(v)] for v in vals], dtype=np.int64)
    cum = np.cumsum(freqs)
    total = int(cum[-1])

    def q(p: float) -> int:
        i = int(np.searchsorted(cum, p * total, side="left"))
        return int(vals[min(i, len(vals) - 1)])

    return {"count": total, "median": q(0.5), "p10": q(0.1), "p90": q(0.9), "max": int(vals[-1])}


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
    checkpoint, out_dir = resolve_checkpoint_and_outdir(args)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"using_checkpoint {checkpoint}")
    print(f"out_dir {out_dir}")

    cfg = load_cfg(args.config, args.steps)
    dataset = G0DatasetWithRaw(**cfg["dataset"])
    model, checkpoint_data = load_model(cfg, checkpoint, device)

    split = dataset.get_split(args.split)
    # Sampling protocol (TEST_PLAN.md §1, §3). Open3D's BaseDatasetSplit
    # HARD-CODES SemSegSpatiallyRegularSampler for the 'test' split (ignoring
    # cfg.sampler), so we force the sampler explicitly for BOTH splits:
    #   * "sampled": SemSegRandomSampler -> exactly --steps random patches spread
    #     across ALL frames (index % len(dataset) wraparound). Identical to the
    #     validation diagnostics; used for the G2/H0 sampled-vs-full cross-check.
    #   * "full": SemSegSpatiallyRegularSampler with length = #frames and NO step
    #     cap. gen_test walks every frame in order, fully covering each (~8
    #     patches/frame) before advancing, so EVERY frame's points are evaluated
    #     >= once. This is the headline final-evaluation protocol.
    # (The earlier bug: capping the spatially-regular sampler at --steps made it
    # cover only the FIRST --steps frames in order -> biased subset / IndexError.)
    if args.coverage == "full":
        if not isinstance(split.sampler, SemSegSpatiallyRegularSampler):
            split.sampler = SemSegSpatiallyRegularSampler(split)
        steps_per_epoch = None  # -> dataloader length = len(split); covers all frames
    else:
        if not isinstance(split.sampler, SemSegRandomSampler):
            split.sampler = SemSegRandomSampler(split)
        steps_per_epoch = int(args.steps)
    sampler = split.sampler
    model.trans_point_sampler = sampler.get_point_sampler()
    torch_split = TorchDataloader(
        dataset=split,
        preprocess=model.preprocess,
        transform=model.transform,
        sampler=sampler,
        use_cache=False,
        steps_per_epoch=steps_per_epoch,
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

    # --- full-coverage no-voting diagnostics (only when --coverage full) -------
    # Per frame, accumulate each evaluation-cloud point's predicted-class votes
    # across the overlapping patches that cover it (point_inds index the frame's
    # subsampled cloud). From that we get, per true class: the coverage-count
    # distribution, the prediction-agreement rate, and a majority-voted confusion
    # matrix -- the empirical no-voting check. All of this is purely additive and
    # must never affect the headline (patch-accumulated) metrics.
    cov_enabled = args.coverage == "full"
    cov_warned = {"missing": False}
    cov_state: dict[str, Any] = {"key": None, "idx": [], "pred": [], "true": []}
    cov_hist = {0: Counter(), 1: Counter(), 2: Counter()}
    agree_total = {0: 0, 1: 0, 2: 0}
    agree_yes = {0: 0, 1: 0, 2: 0}
    voted_cm = np.zeros((3, 3), dtype=np.int64)

    def finalize_frame() -> None:
        if not cov_state["idx"]:
            return
        idx = np.concatenate(cov_state["idx"])
        pred = np.concatenate(cov_state["pred"])
        true = np.concatenate(cov_state["true"])
        cov_state["idx"], cov_state["pred"], cov_state["true"] = [], [], []
        if idx.size == 0:
            return
        n = int(idx.max()) + 1
        vote = np.zeros((n, 3), dtype=np.int64)
        np.add.at(vote, (idx, pred), 1)
        true_cls = np.full(n, -1, dtype=np.int64)
        true_cls[idx] = true  # constant per point; duplicate writes are identical
        coverage = vote.sum(axis=1)
        covered = coverage > 0
        agree = covered & (vote.max(axis=1) == coverage)  # all patches agree on class
        voted_pred = vote.argmax(axis=1)
        for c in (0, 1, 2):
            cmask = covered & (true_cls == c)
            if cmask.any():
                vals, cnts = np.unique(coverage[cmask], return_counts=True)
                for v, ct in zip(vals.tolist(), cnts.tolist()):
                    cov_hist[c][int(v)] += int(ct)
                agree_total[c] += int(cmask.sum())
                agree_yes[c] += int((agree & cmask).sum())
        np.add.at(voted_cm, (true_cls[covered], voted_pred[covered]), 1)

    with torch.no_grad():
        for step, inputs in enumerate(tqdm(loader, desc="g0_sampled_validation"), start=1):
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

            if cov_enabled:
                pinds = inputs["data"].get("point_inds")
                if pinds is None:
                    if not cov_warned["missing"]:
                        print("[warn] point_inds absent; full-coverage diagnostics disabled")
                        cov_warned["missing"] = True
                    cov_enabled = False
                else:
                    if hasattr(pinds, "detach"):
                        pinds = pinds.detach().cpu().numpy()
                    pinds = np.asarray(pinds).reshape(-1)[valid_mask].astype(np.int64)
                    frame_key = (seq_id, frame_idx)
                    if frame_key != cov_state["key"]:
                        finalize_frame()  # flush the frame we just finished
                        cov_state["key"] = frame_key
                    cov_state["idx"].append(pinds)
                    cov_state["pred"].append(y_pred_np)
                    cov_state["true"].append(y_true_np)

    if cov_enabled:
        finalize_frame()  # flush the final frame

    mismatch_rate = 0.0 if raw_remap_checked == 0 else raw_remap_mismatch / raw_remap_checked
    if mismatch_rate > args.max_raw_remap_mismatch_rate:
        raise RuntimeError(
            f"raw remap mismatch_rate={mismatch_rate:.6f} exceeds "
            f"{args.max_raw_remap_mismatch_rate}"
        )

    # Whether this model actually has RGB. For the LiDAR-only baseline (D0) the
    # rgb_valid stratification / brightness fingerprint are meaningless, so they
    # are not written; core metrics, confusion, distance and subtype are.
    has_rgb = getattr(dataset, "feature_mode", FEATURE_MODE_INTENSITY) == FEATURE_MODE_INTENSITY_RGB_FRONT

    summary_rows = [{"stratum": "all", **metrics_from_cm(cm_all)}]
    if has_rgb:
        for stratum, cm in cm_by_rgb.items():
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

    # Raw marking-subtype recall. RGB models get all three strata
    # (all / rgb_valid / rgb_invalid); the LiDAR-only baseline gets only "all".
    raw_rows = []
    for (raw_id, stratum), cm in sorted(raw_rows_counter.items()):
        if not has_rgb and stratum != "all":
            continue
        raw_rows.append(
            {
                "raw_id": raw_id,
                "raw_name": RAW_MARKING_SUBTYPES[raw_id],
                "stratum": stratum,
                **metrics_from_cm(cm),
            }
        )
    write_csv(out_dir / "raw_subtype_rgb_stratified_metrics.csv", raw_rows)

    # Per-outcome feature profile. RGB columns only for RGB models; the LiDAR-only
    # baseline keeps intensity + range (the only features it has).
    group_fields = ("intensity", "rgb_valid", "red", "green", "blue", "range") if has_rgb else ("intensity", "range")
    group_rows = []
    for group in GROUPS:
        row: dict[str, Any] = {"group": group}
        for field in group_fields:
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

    # Full-coverage no-voting diagnostics (only present for --coverage full).
    if cov_enabled:
        cov_rows = []
        all_hist: Counter = Counter()
        tot_total = tot_yes = 0
        for c, name in enumerate(CLASS_NAMES):
            pr = hist_percentiles(cov_hist[c])
            all_hist.update(cov_hist[c])
            tot_total += agree_total[c]
            tot_yes += agree_yes[c]
            cov_rows.append({
                "true_class": name,
                "covered_points": pr["count"],
                "coverage_median": pr["median"],
                "coverage_p10": pr["p10"],
                "coverage_p90": pr["p90"],
                "coverage_max": pr["max"],
                "agreement_rate": (agree_yes[c] / agree_total[c]) if agree_total[c] else None,
            })
        pr_all = hist_percentiles(all_hist)
        cov_rows.append({
            "true_class": "all",
            "covered_points": pr_all["count"],
            "coverage_median": pr_all["median"],
            "coverage_p10": pr_all["p10"],
            "coverage_p90": pr_all["p90"],
            "coverage_max": pr_all["max"],
            "agreement_rate": (tot_yes / tot_total) if tot_total else None,
        })
        write_csv(out_dir / "coverage_count_by_class.csv", cov_rows)
        np.save(out_dir / "confusion_matrix_voted.npy", voted_cm)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "config": str(args.config),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_data.get("epoch") if isinstance(checkpoint_data, dict) else None,
        "split": args.split,
        "steps": args.steps,
        "seed": args.seed,
        "device": str(device),
        "rgb_valid_threshold": args.rgb_valid_threshold,
        "note": "Fresh sampled inference pass; not exact original G0 best-epoch validation sample.",
        "class_order": {"0": "road", "1": "marking", "2": "other"},
        "raw_remap_alignment": {
            "checked_points": raw_remap_checked,
            "mismatched_points": raw_remap_mismatch,
            "mismatch_rate": mismatch_rate,
            "threshold": args.max_raw_remap_mismatch_rate,
        },
        "feature_mode": getattr(dataset, "feature_mode", FEATURE_MODE_INTENSITY),
        "has_rgb": has_rgb,
        "sampled_metrics": metrics_from_cm(cm_all),
        "confusion_matrix": cm_all.tolist(),
    }
    if has_rgb:
        summary["rgb_valid_metrics"] = metrics_from_cm(cm_by_rgb["rgb_valid"])
        summary["rgb_invalid_metrics"] = metrics_from_cm(cm_by_rgb["rgb_invalid"])
    summary["coverage"] = args.coverage
    if cov_enabled:
        # Empirical no-voting check: per-point majority-voted metrics alongside the
        # headline patch-accumulated ones (expected to agree closely).
        summary["voted_metrics"] = metrics_from_cm(voted_cm)
        summary["voted_confusion_matrix"] = voted_cm.tolist()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    readme = [
        "# G0 Best-Epoch Sampled Error Analysis",
        "",
        "Fresh sampled validation inference pass. This is not the exact training-time",
        "best-epoch validation sample.",
        "",
        f"- checkpoint: `{checkpoint}`",
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
            "Interpretation rule (compare to F0 sampled epoch-13):",
            "",
            "- If `rgb_valid` pred/true fell below F0's 1.550, Lovasz reduced the",
            "  RGB-valid overprediction fingerprint.",
            "- If `rgb_valid` still holds most road->marking errors, residual false",
            "  positives remain tied to valid front-camera RGB regions.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(readme))

    print("G0 sampled error analysis complete")
    print(f"out_dir {out_dir}")
    print("script_status PASS")


if __name__ == "__main__":
    main()
