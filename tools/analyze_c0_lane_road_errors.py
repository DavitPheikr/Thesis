"""Analyze C0 lane-vs-road errors on sampled validation inputs.

This script loads a C0 checkpoint, runs validation-style sampled inference, and
summarizes where true lane points are predicted as road. It intentionally uses
the same Open3D/RandLA-Net preprocessing path as Milestone C training.

Important: this is a new sampled inference pass. It does not reconstruct the
exact random samples used during the original epoch-18 validation, because the
run artifacts store aggregate metrics, not per-point predictions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCHED_DEVKIT = PROJECT_ROOT / "pandaset-devkit/python"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PATCHED_DEVKIT))

import open3d.ml.torch as ml3d
from open3d._ml3d.torch.dataloaders import TorchDataloader, get_sampler
from open3d._ml3d.torch.modules.losses.semseg_loss import filter_valid_label
from open3d._ml3d.torch.pipelines import SemanticSegmentation

from datasets.pandaset_ff_lane3 import PandaSetFFLane3Dataset
from tools.train_milestone_c import attach_ego_ranges_to_transform


DEFAULT_CONFIG = PROJECT_ROOT / "configs/randlanet_pandaset_ff_lane3.yml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/checkpoints/ckpt_epoch_00018.pth"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "logs/milestone_c/analysis/c0_epoch18_lane_road_errors"
)

GROUPS = (
    "lane_tp",
    "lane_to_road",
    "lane_to_other",
    "road_tp",
    "road_to_lane",
    "other_to_lane",
)

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
    "stats_file": "logs/milestone_b_training_statistics.json",
    "dataset_root_file": "logs/dataset_root.txt",
    "preflight_pattern_file": "logs/milestone_b_preflight_sensor_pattern.txt",
    "cache_dir": "logs/cache",
    "test_result_folder": "logs/test_results",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of sampled frames/steps to analyze. Use 720 for full validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Inference device.",
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
    """Resolve config paths against the current repo instead of one machine."""
    fallback = PROJECT_ROOT / fallback_relative
    if value is None:
        return str(fallback)

    path = Path(value)
    if path.is_absolute():
        return str(path) if path.exists() else str(fallback)

    return str((PROJECT_ROOT / path).resolve())


def normalize_dataset_paths(cfg: dict) -> None:
    dataset_cfg = cfg["dataset"]
    for key, fallback_relative in REPO_LOCAL_DATASET_PATHS.items():
        dataset_cfg[key] = repo_local_path(dataset_cfg.get(key), fallback_relative)


def load_cfg(path: Path, steps: int) -> dict:
    cfg = yaml.safe_load(path.read_text())
    normalize_dataset_paths(cfg)
    cfg["dataset"]["sampler"] = {"name": "SemSegRandomSampler"}
    if "validation" in cfg["dataset"]:
        raise RuntimeError("Unexpected nested validation dataset config")
    cfg["dataset"]["steps_per_epoch_valid"] = steps
    cfg["pipeline"]["val_batch_size"] = 1
    cfg["pipeline"]["batch_size"] = 1
    cfg["pipeline"]["num_workers"] = 0
    cfg["pipeline"]["pin_memory"] = False
    cfg["model"]["ckpt_path"] = None
    cfg["model"]["is_resume"] = False
    return cfg


def load_model(cfg: dict, checkpoint: Path, device: torch.device):
    model = ml3d.models.RandLANet(**cfg["model"])
    attach_ego_ranges_to_transform(model)
    model.device = device
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, ckpt


def summarize_values(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0])
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join("" if row[k] is None else str(row[k]) for k in keys))
    path.write_text("\n".join(lines) + "\n")


def plot_histograms(groups: dict[str, list[np.ndarray]], out_dir: Path) -> None:
    intensity = {
        name: np.concatenate(parts) if parts else np.asarray([], dtype=np.float32)
        for name, parts in groups.items()
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, color in (
        ("lane_tp", "tab:green"),
        ("lane_to_road", "tab:red"),
        ("road_tp", "0.35"),
        ("road_to_lane", "tab:orange"),
    ):
        values = intensity[name]
        if values.size:
            ax.hist(values, bins=80, range=(0, 114), density=True, alpha=0.45, label=name, color=color)
    ax.set_title("Model-input intensity by C0 prediction outcome")
    ax.set_xlabel("clipped raw intensity units recovered from standardized input")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "intensity_hist_by_outcome.png", dpi=180)
    plt.close(fig)


def empty_bucket_stats() -> dict[str, dict]:
    return {
        label: {
            "range_min_m": lo,
            "range_max_m": None if np.isinf(hi) else hi,
            "active_points": 0,
            "true_lane": 0,
            "lane_tp": 0,
            "lane_to_road": 0,
            "lane_to_other": 0,
            "road_total": 0,
            "road_tp": 0,
            "road_to_lane": 0,
            "other_total": 0,
            "other_to_lane": 0,
            "lane_tp_intensity": [],
            "lane_to_road_intensity": [],
            "road_tp_intensity": [],
        }
        for lo, hi, label in DISTANCE_BUCKETS
    }


def bucket_rows_from_stats(bucket_stats: dict[str, dict]) -> list[dict]:
    rows = []
    for _lo, _hi, label in DISTANCE_BUCKETS:
        stats = bucket_stats[label]
        lane_tp = stats["lane_tp"]
        lane_to_road = stats["lane_to_road"]
        true_lane = stats["true_lane"]
        road_total = stats["road_total"]
        road_tp = stats["road_tp"]
        other_total = stats["other_total"]
        lane_tp_intensity = (
            np.concatenate(stats["lane_tp_intensity"])
            if stats["lane_tp_intensity"]
            else np.asarray([], dtype=np.float32)
        )
        lane_to_road_intensity = (
            np.concatenate(stats["lane_to_road_intensity"])
            if stats["lane_to_road_intensity"]
            else np.asarray([], dtype=np.float32)
        )
        road_tp_intensity = (
            np.concatenate(stats["road_tp_intensity"])
            if stats["road_tp_intensity"]
            else np.asarray([], dtype=np.float32)
        )
        rows.append(
            {
                "bucket": label,
                "range_min_m": stats["range_min_m"],
                "range_max_m": stats["range_max_m"],
                "active_points": stats["active_points"],
                "true_lane": true_lane,
                "lane_tp": lane_tp,
                "lane_to_road": lane_to_road,
                "lane_to_other": stats["lane_to_other"],
                "lane_recall": None if true_lane == 0 else lane_tp / true_lane,
                "lane_to_road_rate": None if true_lane == 0 else lane_to_road / true_lane,
                "road_total": road_total,
                "road_tp": road_tp,
                "road_to_lane": stats["road_to_lane"],
                "road_to_lane_rate": None if road_total == 0 else stats["road_to_lane"] / road_total,
                "other_total": other_total,
                "other_to_lane": stats["other_to_lane"],
                "other_to_lane_rate": None if other_total == 0 else stats["other_to_lane"] / other_total,
                "lane_tp_intensity_p25": percentile_or_none(lane_tp_intensity, 25),
                "lane_tp_intensity_median": percentile_or_none(lane_tp_intensity, 50),
                "lane_tp_intensity_p75": percentile_or_none(lane_tp_intensity, 75),
                "lane_to_road_intensity_p25": percentile_or_none(lane_to_road_intensity, 25),
                "lane_to_road_intensity_median": percentile_or_none(lane_to_road_intensity, 50),
                "lane_to_road_intensity_p75": percentile_or_none(lane_to_road_intensity, 75),
                "road_tp_intensity_p25": percentile_or_none(road_tp_intensity, 25),
                "road_tp_intensity_median": percentile_or_none(road_tp_intensity, 50),
                "road_tp_intensity_p75": percentile_or_none(road_tp_intensity, 75),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    device = choose_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.config, args.steps)
    dataset = PandaSetFFLane3Dataset(**cfg["dataset"])
    model, ckpt = load_model(cfg, args.checkpoint, device)

    split = dataset.get_split(args.split)
    sampler = split.sampler
    model.trans_point_sampler = sampler.get_point_sampler()
    torch_split = TorchDataloader(
        dataset=split,
        preprocess=model.preprocess,
        transform=model.transform,
        sampler=sampler,
        use_cache=dataset.cfg.use_cache,
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
    bucket_stats = empty_bucket_stats()
    frame_rows: list[dict] = []
    cm = np.zeros((3, 3), dtype=np.int64)

    mean = float(dataset.intensity_mean)
    std = float(dataset.intensity_std)

    with torch.no_grad():
        for step, inputs in enumerate(loader, start=1):
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

            raw_labels = inputs["data"]["labels"].detach().cpu().numpy().reshape(-1)
            valid_mask = ~np.isin(raw_labels, np.asarray(model.cfg.ignored_label_inds))
            features = inputs["data"]["features"].detach().cpu().numpy().reshape(-1, 4)
            ranges = inputs["data"]["ranges"].detach().cpu().numpy().reshape(-1)
            intensity = (features[:, 3] * std + mean).astype(np.float32)

            y_true_np = y_true.detach().cpu().numpy().astype(np.int64).reshape(-1)
            y_pred_np = y_pred.detach().cpu().numpy().astype(np.int64).reshape(-1)
            intensity = intensity[valid_mask]
            ranges = ranges[valid_mask]

            for i in range(3):
                for j in range(3):
                    cm[i, j] += int(((y_true_np == i) & (y_pred_np == j)).sum())

            masks = {
                "lane_tp": (y_true_np == 1) & (y_pred_np == 1),
                "lane_to_road": (y_true_np == 1) & (y_pred_np == 0),
                "lane_to_other": (y_true_np == 1) & (y_pred_np == 2),
                "road_tp": (y_true_np == 0) & (y_pred_np == 0),
                "road_to_lane": (y_true_np == 0) & (y_pred_np == 1),
                "other_to_lane": (y_true_np == 2) & (y_pred_np == 1),
            }
            for name, mask in masks.items():
                intensity_groups[name].append(intensity[mask])
                range_groups[name].append(ranges[mask])

            for lo, hi, label in DISTANCE_BUCKETS:
                bucket_mask = (ranges >= lo) & (ranges < hi)
                stats = bucket_stats[label]
                stats["active_points"] += int(bucket_mask.sum())
                stats["true_lane"] += int(((y_true_np == 1) & bucket_mask).sum())
                stats["lane_tp"] += int((masks["lane_tp"] & bucket_mask).sum())
                stats["lane_to_road"] += int((masks["lane_to_road"] & bucket_mask).sum())
                stats["lane_to_other"] += int((masks["lane_to_other"] & bucket_mask).sum())
                stats["road_total"] += int(((y_true_np == 0) & bucket_mask).sum())
                stats["road_tp"] += int((masks["road_tp"] & bucket_mask).sum())
                stats["road_to_lane"] += int((masks["road_to_lane"] & bucket_mask).sum())
                stats["other_total"] += int(((y_true_np == 2) & bucket_mask).sum())
                stats["other_to_lane"] += int((masks["other_to_lane"] & bucket_mask).sum())
                stats["lane_tp_intensity"].append(intensity[masks["lane_tp"] & bucket_mask])
                stats["lane_to_road_intensity"].append(
                    intensity[masks["lane_to_road"] & bucket_mask]
                )
                stats["road_tp_intensity"].append(intensity[masks["road_tp"] & bucket_mask])

            attr = inputs["attr"]
            seq_id = attr["seq_id"][0] if isinstance(attr["seq_id"], list) else str(attr["seq_id"])
            frame_idx_raw = attr["frame_idx"]
            frame_idx = int(frame_idx_raw[0]) if hasattr(frame_idx_raw, "__len__") else int(frame_idx_raw)

            lane_total = int((y_true_np == 1).sum())
            lane_tp = int(masks["lane_tp"].sum())
            lane_to_road = int(masks["lane_to_road"].sum())
            lane_to_other = int(masks["lane_to_other"].sum())
            road_to_lane = int(masks["road_to_lane"].sum())
            other_to_lane = int(masks["other_to_lane"].sum())
            predicted_lane = lane_tp + road_to_lane + other_to_lane
            lane_false_positive = road_to_lane + other_to_lane
            lane_union = lane_total + lane_false_positive
            row = {
                "step": step,
                "seq_id": seq_id,
                "frame_idx": frame_idx,
                "active_points": int(y_true_np.size),
                "true_lane": lane_total,
                "lane_tp": lane_tp,
                "lane_to_road": lane_to_road,
                "lane_to_other": lane_to_other,
                "road_to_lane": road_to_lane,
                "other_to_lane": other_to_lane,
                "predicted_lane": predicted_lane,
                "lane_false_positive": lane_false_positive,
                "lane_recall": None if lane_total == 0 else lane_tp / lane_total,
                "lane_precision": None if predicted_lane == 0 else lane_tp / predicted_lane,
                "lane_f1": None
                if lane_total + predicted_lane == 0
                else (2 * lane_tp) / (lane_total + predicted_lane),
                "lane_iou": None if lane_union == 0 else lane_tp / lane_union,
                "lane_to_road_rate": None if lane_total == 0 else lane_to_road / lane_total,
                "lane_tp_intensity_mean": None
                if lane_tp == 0
                else float(np.mean(intensity[masks["lane_tp"]])),
                "lane_tp_intensity_median": percentile_or_none(intensity[masks["lane_tp"]], 50),
                "lane_to_road_intensity_mean": None
                if lane_to_road == 0
                else float(np.mean(intensity[masks["lane_to_road"]])),
                "lane_to_road_intensity_median": percentile_or_none(
                    intensity[masks["lane_to_road"]], 50
                ),
                "road_to_lane_intensity_mean": None
                if road_to_lane == 0
                else float(np.mean(intensity[masks["road_to_lane"]])),
                "road_to_lane_intensity_median": percentile_or_none(
                    intensity[masks["road_to_lane"]], 50
                ),
                "lane_tp_range_mean": None
                if lane_tp == 0
                else float(np.mean(ranges[masks["lane_tp"]])),
                "lane_tp_range_median": percentile_or_none(ranges[masks["lane_tp"]], 50),
                "lane_to_road_range_mean": None
                if lane_to_road == 0
                else float(np.mean(ranges[masks["lane_to_road"]])),
                "lane_to_road_range_median": percentile_or_none(
                    ranges[masks["lane_to_road"]], 50
                ),
            }
            frame_rows.append(row)

    intensity_rows = []
    range_rows = []
    for name in GROUPS:
        intensity_values = (
            np.concatenate(intensity_groups[name])
            if intensity_groups[name]
            else np.asarray([], dtype=np.float32)
        )
        range_values = (
            np.concatenate(range_groups[name])
            if range_groups[name]
            else np.asarray([], dtype=np.float32)
        )
        intensity_rows.append({"group": name, **summarize_values(intensity_values)})
        range_rows.append({"group": name, **summarize_values(range_values)})

    frame_rows_sorted = sorted(
        frame_rows,
        key=lambda row: (row["lane_to_road"], row["true_lane"]),
        reverse=True,
    )

    write_csv(args.out_dir / "frame_error_summary.csv", frame_rows)
    write_csv(args.out_dir / "top_frames_by_lane_to_road.csv", frame_rows_sorted[:50])
    write_csv(args.out_dir / "group_intensity_summary.csv", intensity_rows)
    write_csv(args.out_dir / "group_range_summary.csv", range_rows)
    distance_bucket_rows = bucket_rows_from_stats(bucket_stats)
    write_csv(args.out_dir / "distance_bucket_summary.csv", distance_bucket_rows)
    np.save(args.out_dir / "confusion_matrix.npy", cm)
    plot_histograms(intensity_groups, args.out_dir)

    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "split": args.split,
        "steps": args.steps,
        "seed": args.seed,
        "device": str(device),
        "note": "New sampled inference pass; not the exact original epoch validation samples.",
        "confusion_matrix": cm.tolist(),
        "intensity_summary": intensity_rows,
        "range_summary": range_rows,
        "distance_bucket_summary": distance_bucket_rows,
        "top_frame_by_lane_to_road": frame_rows_sorted[0] if frame_rows_sorted else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# C0 Lane-To-Road Error Analysis",
        "",
        "This is a new sampled inference pass, not the exact original epoch-18 validation sample.",
        "",
        "## Run",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- checkpoint_epoch: `{summary['checkpoint_epoch']}`",
        f"- split: `{args.split}`",
        f"- steps: `{args.steps}`",
        f"- seed: `{args.seed}`",
        f"- device: `{device}`",
        "",
        "## Outputs",
        "",
        "- `frame_error_summary.csv`",
        "- `top_frames_by_lane_to_road.csv`",
        "- `group_intensity_summary.csv`",
        "- `group_range_summary.csv`",
        "- `distance_bucket_summary.csv`",
        "- `confusion_matrix.npy`",
        "- `intensity_hist_by_outcome.png`",
        "- `summary.json`",
    ]
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
