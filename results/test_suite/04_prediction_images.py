#!/usr/bin/env python
"""Interactive front-camera overlay for Milestone G predictions.

The viewer runs RandLA-Net inference on sampled forward-LiDAR points from a
selected PandaSet sequence, projects those predicted points into the front
camera, and overlays them on the image.

Default class colors:

    road     -> blue
    marking  -> red
    other    -> green

Important limitations:

* With ``--passes-per-frame 1`` the overlay shows one RandLA-Net sampled patch.
  With larger values, the script accumulates repeated sampled passes over the
  same frame and overlays all covered grid-subsampled points. This is the
  practical dense-view mode for this fixed-size-patch RandLA-Net setup.
* Projection is geometric only. It does not perform image-space occlusion
  reasoning, so points behind foreground objects can still project onto them.
* The model is trained on ego-frame points, but projection must use world-frame
  points. This script keeps both and verifies shapes through the preprocessing
  path.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader


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
from open3d._ml3d.datasets.utils import DataProcessing  # noqa: E402
from open3d._ml3d.datasets.samplers.semseg_random import SemSegRandomSampler  # noqa: E402
from open3d._ml3d.torch.dataloaders import TorchDataloader  # noqa: E402
from open3d._ml3d.torch.modules.losses.semseg_loss import filter_valid_label  # noqa: E402
from open3d._ml3d.torch.pipelines import SemanticSegmentation  # noqa: E402
from pandaset import geometry as pds_geometry  # noqa: E402

from thesis_pipeline.adapters.pandaset_ff_lane3 import (  # noqa: E402
    FEATURE_MODE_INTENSITY,
    FEATURE_MODE_INTENSITY_RGB_FRONT,
    LABEL_MODE_ROAD_MARKING3,
    project_points_to_camera,
    remap_raw_pandaset_ids,
)
from thesis_pipeline.datasets.pandaset_ff_lane3_dataset import (  # noqa: E402
    PandaSetFFLane3Dataset,
)


DEFAULT_CONFIG = PROJECT_ROOT / "logs/milestone_g/configs/g1_schedule_extend.yml"
FALLBACK_CONFIG = PROJECT_ROOT / "logs/milestone_g/configs/g0_rgb_lovasz.yml"
DEFAULT_RUN_DIR = PROJECT_ROOT / "logs/milestone_g/runs/G1_schedule_extend"
FALLBACK_RUN_DIR = PROJECT_ROOT / "logs/milestone_g/runs/G0_rgb_lovasz"
DEFAULT_SAVE_DIR = (
    PROJECT_ROOT
    / "logs/milestone_g/run_analysis/front_camera_predictions"
)

CLASS_NAMES = ("road", "marking", "other")
ACTIVE_CLASS_COLORS_BGR = {
    0: (255, 0, 0),    # road: blue
    1: (0, 0, 255),    # marking: red
    2: (0, 180, 0),    # other: green
}
ACTIVE_CLASS_COLORS_RGB = {
    key: (value[2], value[1], value[0])
    for key, value in ACTIVE_CLASS_COLORS_BGR.items()
}
GT_LABEL_TO_ACTIVE = {
    1: 0,  # road
    2: 1,  # marking
    3: 2,  # other
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else FALLBACK_CONFIG,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR if DEFAULT_RUN_DIR.exists() else FALLBACK_RUN_DIR,
        help="Run directory used to auto-discover best checkpoint from eval_history.csv.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to load. Default: best lane_iou/marking_iou checkpoint in --run-dir.",
    )
    parser.add_argument("--split", choices=("training", "validation", "test"), default="validation")
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Sequence id, e.g. 034 or 54. Default: first sequence in split.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--passes-per-frame",
        type=int,
        default=1,
        help=(
            "Repeated sampled inference passes per lidar frame. Use 1 for fast "
            "patch view; use 4-12 for near whole-frame coverage."
        ),
    )
    parser.add_argument(
        "--min-coverage-warning",
        type=float,
        default=0.90,
        help="Print a warning for frames below this active-grid-point coverage.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Point opacity 0-1. Default 1.0 = fully opaque (no transparency).")
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=("pred", "gt", "error", "plain"),
        default="pred",
        help="Initial display mode. Keys p/g/e switch modes while running. "
             "'plain' = raw front camera, no overlay and no HUD.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help="Directory used when pressing s, and for --save-only.",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Render frames to --save-dir without opening a GUI window.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open cv2.imshow. Useful on headless servers.",
    )
    return parser.parse_args()


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


def normalize_sequence_id(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.isdigit():
        return f"{int(text):03d}"
    return text


def discover_best_checkpoint(run_dir: Path) -> tuple[int, Path]:
    history_path = require_file(run_dir / "eval_history.csv")
    df = pd.read_csv(history_path)
    if "lane_iou" not in df.columns or "epoch" not in df.columns:
        raise ValueError(f"{history_path} must contain epoch and lane_iou columns")
    best_epoch = int(df.loc[df["lane_iou"].idxmax(), "epoch"])
    checkpoint = run_dir / "checkpoints" / f"ckpt_epoch_{best_epoch:05d}.pth"
    return best_epoch, require_file(checkpoint)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(name)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cfg(path: Path) -> dict[str, Any]:
    require_file(path)
    cfg = yaml.safe_load(path.read_text())
    dataset_cfg = cfg["dataset"]
    if dataset_cfg.get("label_mode") != LABEL_MODE_ROAD_MARKING3:
        raise RuntimeError(f"Expected road_marking3, got {dataset_cfg.get('label_mode')}")
    # Generalized: accept the RGB models (8-ch) and the LiDAR-only baseline D0
    # (4-ch, feature_mode=intensity). D0 has no RGB; overlays still render from the
    # predicted/true/error classes (rgb_valid is metadata only, zeroed for D0).
    feature_mode = dataset_cfg.get("feature_mode") or FEATURE_MODE_INTENSITY
    if feature_mode not in (FEATURE_MODE_INTENSITY_RGB_FRONT, FEATURE_MODE_INTENSITY):
        raise RuntimeError(
            f"Unsupported feature_mode {feature_mode!r}; expected "
            f"{FEATURE_MODE_INTENSITY_RGB_FRONT!r} or {FEATURE_MODE_INTENSITY!r}"
        )
    dataset_cfg["feature_mode"] = feature_mode

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
    dataset_cfg["use_cache"] = False
    dataset_cfg["sampler"] = {"name": "SemSegRandomSampler"}

    cfg["pipeline"]["batch_size"] = 1
    cfg["pipeline"]["val_batch_size"] = 1
    cfg["pipeline"]["num_workers"] = 0
    cfg["pipeline"]["pin_memory"] = False
    cfg["pipeline"].pop("loss", None)
    cfg["model"]["ckpt_path"] = None
    cfg["model"]["is_resume"] = False
    return cfg


class ViewerDataset(PandaSetFFLane3Dataset):
    """Analysis dataset that preserves world-frame points for image projection."""

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
        else:  # LiDAR-only (D0): intensity is the only feature channel.
            feat = intensity_col

        sample = {
            "point": xyz_ego,
            "point_world": xyz_world,
            "feat": feat,
            "label": labels.astype(np.int32, copy=False),
        }

        seq.lidar._data = None
        seq.lidar._poses = None
        seq.lidar._timestamps = None
        seq.semseg._data = None
        return sample


def attach_visual_preprocess_and_transform(model) -> None:  # noqa: ANN001
    def preprocess_visual(data, attr):  # noqa: ANN001, ARG001
        cfg = model.cfg
        points = np.asarray(data["point"][:, :3], dtype=np.float32)
        world = np.asarray(data["point_world"][:, :3], dtype=np.float32)
        feat = np.asarray(data["feat"], dtype=np.float32)
        labels = np.asarray(data["label"], dtype=np.int32).reshape((-1,))
        feat_plus_world = np.concatenate([feat, world], axis=1).astype(
            np.float32, copy=False
        )
        sub_points, sub_feat_world, sub_labels = DataProcessing.grid_subsampling(
            points,
            features=feat_plus_world,
            labels=labels,
            grid_size=cfg.grid_size,
        )
        feat_dim = feat.shape[1]
        sub_feat = sub_feat_world[:, :feat_dim].astype(np.float32, copy=False)
        sub_world = sub_feat_world[:, feat_dim:].astype(np.float32, copy=False)
        return {
            "point": sub_points,
            "point_world": sub_world,
            "feat": sub_feat,
            "label": sub_labels,
            "search_tree": KDTree(sub_points),
        }

    def transform_visual(data, attr, min_possibility_idx=None):  # noqa: ANN001, ARG001
        rng = model.rng
        cfg = model.cfg
        pc = data["point"].copy()
        world = data["point_world"].copy()
        label = data["label"].copy()
        feat = data["feat"].copy()
        tree = data["search_tree"]
        full_grid_count = int(pc.shape[0])
        full_active_count = int((label != 0).sum())

        pc, selected_idxs, _center_point = model.trans_point_sampler(
            pc=pc,
            feat=feat,
            label=label,
            search_tree=tree,
            num_points=model.cfg.num_points,
        )
        label = label[selected_idxs]
        feat = feat[selected_idxs]
        world = world[selected_idxs]
        ranges = np.linalg.norm(pc[:, :3], axis=1).astype(np.float32)
        if feat.shape[1] >= 5:  # RGB models have an rgb_valid channel at index 4
            rgb_valid = feat[:, 4].astype(np.float32, copy=False)
        else:  # LiDAR-only (D0): no rgb_valid; zeros (used only for the caption ratio)
            rgb_valid = np.zeros(feat.shape[0], dtype=np.float32)

        augment_cfg = cfg.get("augment", {}).copy()
        val_augment_cfg = {}
        if "recenter" in augment_cfg:
            val_augment_cfg["recenter"] = augment_cfg.pop("recenter")
        if "normalize" in augment_cfg:
            val_augment_cfg["normalize"] = augment_cfg.pop("normalize")
        model.augmenter.augment(pc, feat, label, val_augment_cfg, seed=rng)

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
            "full_grid_count": full_grid_count,
            "full_active_count": full_active_count,
            "world_points": world.astype(np.float32, copy=False),
            "ranges": ranges,
            "rgb_valid": rgb_valid,
        }

    model.preprocess = preprocess_visual
    model.transform = transform_visual


def load_model(cfg: dict[str, Any], checkpoint: Path, device: torch.device):
    model = ml3d.models.RandLANet(**cfg["model"])
    attach_visual_preprocess_and_transform(model)
    model.device = device
    checkpoint_data = torch.load(checkpoint, map_location=device)
    state = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, checkpoint_data


def pick_sequence(dataset: ViewerDataset, split: str, requested: str | None) -> str:
    split_list = dataset.get_split_list(split)
    if not split_list:
        raise RuntimeError(f"Split {split} is empty")
    available = [seq for seq in dataset._split_ids[split]]
    if requested is None:
        return available[0]
    if requested not in available:
        raise ValueError(
            f"Sequence {requested} is not in split {split}. "
            f"Available: {', '.join(available)}"
        )
    return requested


def build_sequence_split(dataset: ViewerDataset, split: str, sequence: str, start: int, stride: int, max_frames: int | None):
    frame_entries = [
        (seq_id, frame_idx)
        for seq_id, frame_idx in dataset.get_split_list(split)
        if seq_id == sequence and frame_idx >= start and (frame_idx - start) % stride == 0
    ]
    if max_frames is not None:
        frame_entries = frame_entries[:max_frames]
    if not frame_entries:
        raise RuntimeError(
            f"No frames selected for split={split} sequence={sequence} "
            f"start={start} stride={stride}"
        )
    split_obj = dataset.get_split(split)
    split_obj.path_list = frame_entries
    return split_obj


def repeat_split_frames(split_obj, passes_per_frame: int):  # noqa: ANN001
    if passes_per_frame < 1:
        raise ValueError("--passes-per-frame must be >= 1")
    original = list(split_obj.path_list)
    repeated = []
    for entry in original:
        repeated.extend([entry] * passes_per_frame)
    split_obj.path_list = repeated
    return original


def tensor_to_numpy(value) -> np.ndarray:  # noqa: ANN001
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def get_attr_value(attr: dict[str, Any], key: str) -> Any:
    value = attr[key]
    if isinstance(value, list):
        return value[0]
    if isinstance(value, tuple):
        return value[0]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return value


def choose_camera_frame(dataset: ViewerDataset, seq_id: str, frame_idx: int) -> tuple[int, float, dict]:
    meta = dataset._get_camera_metadata(seq_id)
    target_t = float(meta["lid_ts"][frame_idx])
    diffs = np.abs(meta["cam_ts"] - target_t)
    cam_idx = int(np.argmin(diffs))
    dt = float(meta["cam_ts"][cam_idx] - target_t)
    return cam_idx, dt, meta


def draw_filled_circle(draw: ImageDraw.ImageDraw, x: int, y: int, radius: int, color: tuple[int, int, int, int]) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def alpha_overlay(base_rgb: np.ndarray, draw_fn) -> np.ndarray:  # noqa: ANN001
    base = Image.fromarray(base_rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw_fn(draw)
    return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"), dtype=np.uint8)


def draw_points(image_rgb: np.ndarray, uv: np.ndarray, labels: np.ndarray, alpha: float, radius: int) -> np.ndarray:
    alpha_byte = int(np.clip(alpha, 0.0, 1.0) * 255)

    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        for class_id in (2, 0, 1):
            mask = labels == class_id
            if not mask.any():
                continue
            color_rgb = ACTIVE_CLASS_COLORS_RGB[class_id]
            color = (*color_rgb, alpha_byte)
            coords = np.rint(uv[mask]).astype(np.int32)
            for u, v in coords:
                draw_filled_circle(draw, int(u), int(v), radius, color)

    return alpha_overlay(image_rgb, draw_fn)


def draw_error_points(image_rgb: np.ndarray, uv: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, alpha: float, radius: int) -> np.ndarray:
    alpha_byte = int(np.clip(alpha, 0.0, 1.0) * 255)
    colors_rgb = {
        "marking_tp": (255, 0, 0),
        "marking_missed": (255, 165, 0),
        "road_to_marking": (255, 0, 255),
        "other_to_marking": (255, 255, 0),
        "correct_other": (180, 180, 180),
    }
    masks = {
        "correct_other": (y_true == y_pred) & (y_true != 1),
        "marking_tp": (y_true == 1) & (y_pred == 1),
        "marking_missed": (y_true == 1) & (y_pred != 1),
        "road_to_marking": (y_true == 0) & (y_pred == 1),
        "other_to_marking": (y_true == 2) & (y_pred == 1),
    }

    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        for key in (
            "correct_other",
            "marking_tp",
            "marking_missed",
            "road_to_marking",
            "other_to_marking",
        ):
            mask = masks[key]
            if not mask.any():
                continue
            draw_radius = max(radius, 3) if key != "correct_other" else max(1, radius - 1)
            coords = np.rint(uv[mask]).astype(np.int32)
            color = (*colors_rgb[key], alpha_byte)
            for u, v in coords:
                draw_filled_circle(draw, int(u), int(v), draw_radius, color)

    return alpha_overlay(image_rgb, draw_fn)


def draw_label_background(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 5
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0, 210),
    )
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)


def add_hud(
    image_rgb: np.ndarray,
    *,
    mode: str,
    seq_id: str,
    lidar_frame: int,
    cam_frame: int,
    dt: float,
    checkpoint: Path,
    best_epoch: int | None,
    n_projected: int,
    n_points: int,
    rgb_valid_ratio: float,
    coverage: float | None = None,
    passes_seen: int | None = None,
) -> np.ndarray:
    img = Image.fromarray(image_rgb, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    coverage_text = ""
    if coverage is not None and passes_seen is not None:
        coverage_text = f"  coverage={coverage:.3f} passes={passes_seen}"
    lines = [
        f"mode={mode}  seq={seq_id} lidar={lidar_frame:02d} cam={cam_frame:02d} dt={dt:+.4f}s",
        f"projected={n_projected}/{n_points}  rgb_valid_selected={rgb_valid_ratio:.3f}{coverage_text}  checkpoint_epoch={best_epoch if best_epoch is not None else 'manual'}",
        f"ckpt={checkpoint.name}  keys: n/space next, b prev, p/g/e mode, s save, q quit",
    ]
    x, y = 16, 16
    for line in lines:
        draw_label_background(draw, (x, y), line, font)
        y += 24

    legend = [
        ("road", ACTIVE_CLASS_COLORS_RGB[0]),
        ("marking", ACTIVE_CLASS_COLORS_RGB[1]),
        ("other", ACTIVE_CLASS_COLORS_RGB[2]),
    ]
    lx, ly = 16, img.height - 22 * len(legend) - 14
    for label, color in legend:
        draw.ellipse((lx - 6, ly - 6, lx + 6, ly + 6), fill=(*color, 255))
        draw_label_background(draw, (lx + 14, ly - 7), label, font)
        ly += 22
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def save_frame(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb, mode="RGB").save(path)


def prepare_display_frame(
    *,
    image_rgb: np.ndarray,
    uv: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mode: str,
    alpha: float,
    radius: int,
) -> np.ndarray:
    if mode == "plain":
        return image_rgb  # raw front camera, no overlay
    if mode == "pred":
        return draw_points(image_rgb, uv, y_pred, alpha, radius)
    if mode == "gt":
        return draw_points(image_rgb, uv, y_true, alpha, radius)
    return draw_error_points(image_rgb, uv, y_true, y_pred, alpha, radius)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    device = choose_device(args.device)
    config = args.config.resolve()
    run_dir = args.run_dir.resolve()
    best_epoch = None
    if args.checkpoint is None:
        best_epoch, checkpoint = discover_best_checkpoint(run_dir)
    else:
        checkpoint = require_file(args.checkpoint.resolve())

    print(f"config {config}")
    print(f"run_dir {run_dir}")
    print(f"checkpoint {checkpoint}")
    print(f"device {device}")

    cfg = load_cfg(config)
    dataset = ViewerDataset(**cfg["dataset"])
    sequence = pick_sequence(dataset, args.split, normalize_sequence_id(args.sequence))
    split = build_sequence_split(
        dataset,
        args.split,
        sequence,
        args.start_frame,
        args.stride,
        args.max_frames,
    )
    unique_frame_entries = repeat_split_frames(split, args.passes_per_frame)

    model, checkpoint_data = load_model(cfg, checkpoint, device)
    if best_epoch is None and isinstance(checkpoint_data, dict):
        best_epoch = checkpoint_data.get("epoch")

    # Open3D forces SemSegSpatiallyRegularSampler on the 'test' split; its point
    # sampler reads self.cloud_id (set only by the cloud generator during full
    # iteration, never in this per-frame viewer) -> AttributeError. Force the
    # stateless random point sampler instead (same as the validation path; with
    # --passes-per-frame it still covers the frame).
    if not isinstance(split.sampler, SemSegRandomSampler):
        split.sampler = SemSegRandomSampler(split)
    sampler = split.sampler
    model.trans_point_sampler = sampler.get_point_sampler()
    torch_split = TorchDataloader(
        dataset=split,
        preprocess=model.preprocess,
        transform=model.transform,
        sampler=sampler,
        use_cache=False,
        steps_per_epoch=len(split.path_list),
    )
    pipeline = SemanticSegmentation(model=model, dataset=dataset, **cfg["pipeline"])
    batcher = pipeline.get_batcher(device)
    loader = DataLoader(
        torch_split,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=batcher.collate_fn,
    )

    mode = args.mode
    frames: list[dict[str, Any]] = []
    print(f"selected_sequence {sequence}")
    print(f"selected_frames {len(unique_frame_entries)}")
    print(f"passes_per_frame {args.passes_per_frame}")
    print(f"total_inference_steps {len(split.path_list)}")

    accumulators: dict[tuple[str, int], dict[str, Any]] = {}
    with torch.no_grad():
        for inputs in loader:
            if hasattr(inputs["data"], "to"):
                inputs["data"].to(device)
            results = model(inputs["data"])
            scores, y_true_t = filter_valid_label(
                results,
                inputs["data"]["labels"],
                model.cfg.num_classes,
                model.cfg.ignored_label_inds,
                device,
            )
            probs = torch.softmax(scores, dim=-1).detach().cpu().numpy().astype(np.float32)
            y_true = y_true_t.detach().cpu().numpy().astype(np.int64).reshape(-1)

            labels_full = tensor_to_numpy(inputs["data"]["labels"]).reshape(-1)
            valid_mask = ~np.isin(labels_full, np.asarray(model.cfg.ignored_label_inds))
            world = tensor_to_numpy(inputs["data"]["world_points"]).reshape(-1, 3)[valid_mask]
            rgb_valid = tensor_to_numpy(inputs["data"]["rgb_valid"]).reshape(-1)[valid_mask]
            point_inds = (
                tensor_to_numpy(inputs["data"]["point_inds"])
                .reshape(-1)[valid_mask]
                .astype(np.int64)
            )
            full_grid_count = int(
                tensor_to_numpy(inputs["data"]["full_grid_count"]).reshape(-1)[0]
            )
            full_active_count = int(
                tensor_to_numpy(inputs["data"]["full_active_count"]).reshape(-1)[0]
            )

            attr = inputs["attr"]
            seq_id = str(get_attr_value(attr, "seq_id"))
            frame_idx = int(get_attr_value(attr, "frame_idx"))
            key = (seq_id, frame_idx)
            if key not in accumulators:
                accumulators[key] = {
                    "seq_id": seq_id,
                    "frame_idx": frame_idx,
                    "prob_sum": np.zeros((full_grid_count, 3), dtype=np.float32),
                    "seen": np.zeros(full_grid_count, dtype=np.int32),
                    "y_true_full": np.full(full_grid_count, -1, dtype=np.int64),
                    "world_full": np.zeros((full_grid_count, 3), dtype=np.float32),
                    "rgb_valid_full": np.zeros(full_grid_count, dtype=np.float32),
                    "full_active_count": full_active_count,
                    "passes_seen": 0,
                }
            acc = accumulators[key]
            if acc["prob_sum"].shape[0] != full_grid_count:
                raise RuntimeError(f"Grid size changed across passes for {seq_id}/{frame_idx}")
            np.add.at(acc["prob_sum"], point_inds, probs)
            np.add.at(acc["seen"], point_inds, 1)
            acc["y_true_full"][point_inds] = y_true
            acc["world_full"][point_inds] = world
            acc["rgb_valid_full"][point_inds] = rgb_valid
            acc["passes_seen"] += 1

    for key in sorted(accumulators):
        acc = accumulators[key]
        covered = acc["seen"] > 0
        active_covered = covered & (acc["y_true_full"] >= 0)
        coverage = float(active_covered.sum()) / max(int(acc["full_active_count"]), 1)
        seq_id = acc["seq_id"]
        frame_idx = acc["frame_idx"]
        cam_idx, dt, meta = choose_camera_frame(dataset, seq_id, frame_idx)
        image_path = Path(meta["cam_dir"]) / f"{cam_idx:02d}.jpg"
        with Image.open(image_path) as img:
            image_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
            image_w, image_h = img.size

        world = acc["world_full"][active_covered]
        prob_sum = acc["prob_sum"][active_covered]
        y_true_acc = acc["y_true_full"][active_covered]
        y_pred_acc = np.argmax(prob_sum, axis=1).astype(np.int64)
        rgb_valid_acc = acc["rgb_valid_full"][active_covered]

        uv, _depth, in_img = project_points_to_camera(
            world.astype(np.float64, copy=False),
            meta["cam_poses"][cam_idx],
            meta["intrinsics"],
            image_w,
            image_h,
        )
        uv = uv[in_img]
        y_true_img = y_true_acc[in_img]
        y_pred_img = y_pred_acc[in_img]

        if not (uv.shape[0] == y_true_img.shape[0] == y_pred_img.shape[0]):
            raise RuntimeError("Projection/prediction shape mismatch")

        if coverage < args.min_coverage_warning:
            print(
                "coverage_warning "
                f"seq={seq_id} frame={frame_idx} coverage={coverage:.3f} "
                f"passes={acc['passes_seen']} threshold={args.min_coverage_warning:.3f}"
            )

        frames.append(
            {
                "seq_id": seq_id,
                "frame_idx": frame_idx,
                "cam_idx": cam_idx,
                "dt": dt,
                "image_rgb": image_rgb,
                "uv": uv,
                "y_true": y_true_img,
                "y_pred": y_pred_img,
                "n_valid_points": int(active_covered.sum()),
                "n_projected": int(uv.shape[0]),
                "rgb_valid_ratio": float(rgb_valid_acc.mean()) if rgb_valid_acc.size else 0.0,
                "coverage": coverage,
                "passes_seen": int(acc["passes_seen"]),
            }
        )

    if not frames:
        raise RuntimeError("No frames were loaded for visualization")

    no_window = args.no_window or args.save_only or not os.environ.get("DISPLAY")
    if no_window and not args.save_only:
        print("window_disabled reason=no_DISPLAY_or_no_window_flag")

    idx = 0
    while 0 <= idx < len(frames):
        item = frames[idx]
        display = prepare_display_frame(
            image_rgb=item["image_rgb"],
            uv=item["uv"],
            y_true=item["y_true"],
            y_pred=item["y_pred"],
            mode=mode,
            alpha=args.alpha,
            radius=args.point_radius,
        )
        if mode != "plain":
            display = add_hud(
                display,
                mode=mode,
                seq_id=item["seq_id"],
                lidar_frame=item["frame_idx"],
                cam_frame=item["cam_idx"],
                dt=item["dt"],
                checkpoint=checkpoint,
                best_epoch=best_epoch,
                n_projected=item["n_projected"],
                n_points=item["n_valid_points"],
                rgb_valid_ratio=item["rgb_valid_ratio"],
                coverage=item.get("coverage"),
                passes_seen=item.get("passes_seen"),
            )

        out_path = (
            args.save_dir
            / run_dir.name
            / item["seq_id"]
            / mode
            / f"{item['seq_id']}_lidar{item['frame_idx']:02d}_frontcam{item['cam_idx']:02d}_{mode}.png"
        )
        if args.save_only:
            save_frame(out_path, display)
            print(f"saved {out_path}")
            idx += 1
            continue

        if no_window:
            save_frame(out_path, display)
            print(f"saved {out_path}")
            idx += 1
            continue

        import cv2

        cv2.imshow(
            "Milestone G front-camera predictions",
            cv2.cvtColor(display, cv2.COLOR_RGB2BGR),
        )
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("n"), ord(" "), 83):
            idx += 1
        elif key in (ord("b"), 81):
            idx = max(0, idx - 1)
        elif key == ord("p"):
            mode = "pred"
        elif key == ord("g"):
            mode = "gt"
        elif key == ord("e"):
            mode = "error"
        elif key == ord("s"):
            save_frame(out_path, display)
            print(f"saved {out_path}")
        else:
            idx += 1

    if not no_window:
        import cv2

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
