#!/usr/bin/env python
"""Eval-only G1 RGB brightness sensitivity probe.

This script tests whether the frozen G1 epoch-27 model is sensitive to absolute
front-camera RGB brightness. It performs sampled validation inference with the
same G1 model/config/checkpoint, but scales RGB channels inside the analysis
batch before the forward pass. It never trains, never changes the dataset/cache,
and never writes into the G1 run directory.

Outputs:
  logs/milestone_h/run_analysis/g1_rgb_brightness_sensitivity/
    brightness_sensitivity_summary.csv
    brightness_sensitivity_confusions.json
    brightness_sensitivity_recommendation.md
    plots/*.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "logs/milestone_g/configs/g1_schedule_extend.yml").exists():
            return parent
    raise RuntimeError("Could not find project root from analysis script path")


PROJECT_ROOT = find_project_root()
G_ANALYSIS_CODE = PROJECT_ROOT / "logs/milestone_g/run_analysis/analysis_code"
sys.path.insert(0, str(G_ANALYSIS_CODE))

import g0_sampled_error_analysis as base  # noqa: E402


RUN_NAME = "G1_schedule_extend"
G1_RUN_DIR = PROJECT_ROOT / "logs/milestone_g/runs/G1_schedule_extend"
DEFAULT_CONFIG = PROJECT_ROOT / "logs/milestone_g/configs/g1_schedule_extend.yml"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "logs/milestone_h/run_analysis/g1_rgb_brightness_sensitivity"
)
DEFAULT_SCALES = (0.80, 0.85, 1.00, 1.15, 1.20)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--steps", type=int, default=2160)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--rgb-valid-threshold", type=float, default=0.5)
    p.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES))
    p.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def discover_best_epoch() -> int:
    history_path = G1_RUN_DIR / "eval_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing G1 eval history: {history_path}")
    df = pd.read_csv(history_path)
    if "lane_iou" not in df.columns:
        raise RuntimeError(f"{history_path} missing lane_iou")
    return int(df.loc[df["lane_iou"].idxmax(), "epoch"])


def default_checkpoint() -> Path:
    epoch = discover_best_epoch()
    return G1_RUN_DIR / "checkpoints" / f"ckpt_epoch_{epoch:05d}.pth"


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


def load_g1_cfg(path: Path, steps: int) -> dict[str, Any]:
    cfg = base.load_cfg(path, steps)
    cfg["dataset"]["cache_dir"] = str(
        (PROJECT_ROOT / "logs/milestone_g/cache/G1_schedule_extend").resolve()
    )
    cfg["dataset"]["test_result_folder"] = str(
        (PROJECT_ROOT / "logs/milestone_g/test_results/G1_schedule_extend").resolve()
    )
    return cfg


def confusion_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            cm[i, j] = int(((y_true == i) & (y_pred == j)).sum())
    return cm


def class_iou(cm: np.ndarray, idx: int) -> float:
    tp = float(cm[idx, idx])
    fp = float(cm[:, idx].sum() - cm[idx, idx])
    fn = float(cm[idx, :].sum() - cm[idx, idx])
    denom = tp + fp + fn
    return float("nan") if denom == 0 else tp / denom


def metrics_from_cm(cm: np.ndarray) -> dict[str, float | int]:
    base_metrics = base.metrics_from_cm(cm)
    road_iou = class_iou(cm, 0)
    marking_iou = class_iou(cm, 1)
    other_iou = class_iou(cm, 2)
    miou = float(np.nanmean([road_iou, marking_iou, other_iou]))
    return {
        **base_metrics,
        "road_iou": road_iou,
        "other_iou": other_iou,
        "miou": miou,
    }


def apply_rgb_brightness_scale(data: Any, scale: float, threshold: float) -> None:
    """Scale RGB in-place on a copied feature tensor inside the batch data.

    Model-input feature layout is [x, y, z, intensity, r, g, b, rgb_valid].
    """
    features = data["features"]
    if features.shape[-1] != 8:
        raise RuntimeError(
            f"Expected model-input features with last dim 8, got {tuple(features.shape)}"
        )
    scaled = features.clone()
    rgb = scaled[..., 4:7]
    rgb_valid = scaled[..., 7]
    valid = rgb_valid >= threshold
    if valid.any():
        rgb[valid] = torch.clamp(rgb[valid] * float(scale), 0.0, 1.0)
    data["features"] = scaled


def restore_features(data: Any, features: torch.Tensor) -> None:
    data["features"] = features


def percent_change(value: float, baseline: float) -> float:
    if baseline == 0 or math.isnan(baseline):
        return float("nan")
    return 100.0 * (value - baseline) / baseline


def add_delta_columns(rows: list[dict[str, Any]]) -> None:
    baseline = next((r for r in rows if abs(float(r["scale"]) - 1.0) < 1e-9), None)
    if baseline is None:
        raise RuntimeError("Scale 1.00 baseline missing from rows")
    fields = [
        "road_to_marking",
        "marking_precision",
        "marking_recall",
        "marking_iou",
        "marking_f1",
        "predicted_true_marking_ratio",
    ]
    for row in rows:
        for field in fields:
            value = float(row[field])
            base_value = float(baseline[field])
            row[f"{field}_abs_change_from_1.00"] = value - base_value
            row[f"{field}_pct_change_from_1.00"] = percent_change(value, base_value)


def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting
        print(f"plot_skip reason={exc}", flush=True)
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    x = df["scale"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(x, df["road_to_marking"], marker="o", linewidth=2)
    ax.axvline(1.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel("RGB brightness scale")
    ax.set_ylabel("road -> marking false positives")
    ax.set_title("G1 brightness sensitivity: road false positives")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "road_to_marking_fp_vs_scale.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for col, label in [
        ("marking_precision", "precision"),
        ("marking_recall", "recall"),
        ("marking_iou", "IoU"),
    ]:
        ax.plot(x, df[col], marker="o", linewidth=2, label=label)
    ax.axvline(1.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel("RGB brightness scale")
    ax.set_ylabel("metric")
    ax.set_title("G1 brightness sensitivity: marking metrics")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "marking_precision_recall_iou_vs_scale.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(x, df["predicted_true_marking_ratio"], marker="o", linewidth=2)
    ax.axhline(1.0, color="0.5", linestyle=":", linewidth=1)
    ax.axvline(1.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel("RGB brightness scale")
    ax.set_ylabel("predicted / true marking ratio")
    ax.set_title("G1 brightness sensitivity: marking volume")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "pred_true_vs_scale.png", dpi=220)
    plt.close(fig)


def trend_label(values: dict[float, float], lower_key: float, upper_key: float) -> str:
    baseline = values[1.0]
    low = values[lower_key]
    high = values[upper_key]
    low_delta = percent_change(low, baseline)
    high_delta = percent_change(high, baseline)
    if low_delta < -2.0 and high_delta > 2.0:
        return "yes"
    if abs(low_delta) <= 2.0 and abs(high_delta) <= 2.0:
        return "no"
    return "mixed"


def write_recommendation(
    out_dir: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    checkpoint: Path,
    device: torch.device,
) -> None:
    by_scale = {float(r["scale"]): r for r in rows}
    fp_values = {s: float(r["road_to_marking"]) for s, r in by_scale.items()}
    iou_values = {s: float(r["marking_iou"]) for s, r in by_scale.items()}
    precision_values = {s: float(r["marking_precision"]) for s, r in by_scale.items()}
    recall_values = {s: float(r["marking_recall"]) for s, r in by_scale.items()}
    ptr_values = {s: float(r["predicted_true_marking_ratio"]) for s, r in by_scale.items()}

    dark_reduces = trend_label(fp_values, 0.85, 1.15) in {"yes", "mixed"} and (
        fp_values.get(0.85, fp_values[1.0]) < fp_values[1.0]
        or fp_values.get(0.80, fp_values[1.0]) < fp_values[1.0]
    )
    bright_increases = trend_label(fp_values, 0.85, 1.15) in {"yes", "mixed"} and (
        fp_values.get(1.15, fp_values[1.0]) > fp_values[1.0]
        or fp_values.get(1.20, fp_values[1.0]) > fp_values[1.0]
    )

    scales_sorted = sorted(by_scale)
    fp_series = [fp_values[s] for s in scales_sorted]
    monotonic_fp = all(a <= b for a, b in zip(fp_series, fp_series[1:])) or all(
        a >= b for a, b in zip(fp_series, fp_series[1:])
    )

    mild_fp_effect = max(
        abs(float(by_scale[0.85]["road_to_marking_pct_change_from_1.00"])),
        abs(float(by_scale[1.15]["road_to_marking_pct_change_from_1.00"])),
    )
    strong_fp_effect = max(
        abs(float(by_scale[0.80]["road_to_marking_pct_change_from_1.00"])),
        abs(float(by_scale[1.20]["road_to_marking_pct_change_from_1.00"])),
    )

    clear_effect = strong_fp_effect >= 2.0 and (dark_reduces or bright_increases)
    mild_enough = mild_fp_effect >= 2.0
    if clear_effect:
        decision = "GO: H0_rgb_jitter is justified by eval-time brightness sensitivity."
        h0_range = "0.85-1.15" if mild_enough else "0.80-1.20"
    else:
        decision = (
            "NO-GO: brightness scaling barely changes the target errors; keep G1 "
            "or run a larger probe before spending a full training run."
        )
        h0_range = "not recommended"

    def fmt_delta(scale: float, field: str) -> str:
        row = by_scale[scale]
        return (
            f"{row[field]} "
            f"({row[f'{field}_abs_change_from_1.00']:+.6g}, "
            f"{row[f'{field}_pct_change_from_1.00']:+.2f}%)"
        )

    lines = [
        "# G1 RGB Brightness Sensitivity Recommendation",
        "",
        "Eval-only probe. No training, no dataset/cache/config mutation, no writes to the G1 run directory.",
        "",
        "## Provenance",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- script: `logs/milestone_h/run_analysis/analysis_code/g1_rgb_brightness_sensitivity.py`",
        f"- config: `{args.config}`",
        f"- checkpoint: `{checkpoint}`",
        f"- split: `{args.split}`",
        f"- steps: `{args.steps}`",
        f"- seed: `{args.seed}`",
        f"- device: `{device}`",
        f"- rgb_valid threshold: `{args.rgb_valid_threshold}`",
        f"- scales: `{', '.join(f'{s:.2f}' for s in scales_sorted)}`",
        "",
        "## Answers",
        "",
        f"1. **Does darkening RGB reduce road->marking false positives?** {'Yes' if dark_reduces else 'No'}."
        f" 0.85: {fmt_delta(0.85, 'road_to_marking')}; 0.80: {fmt_delta(0.80, 'road_to_marking')}.",
        f"2. **Does brightening RGB increase road->marking false positives?** {'Yes' if bright_increases else 'No'}."
        f" 1.15: {fmt_delta(1.15, 'road_to_marking')}; 1.20: {fmt_delta(1.20, 'road_to_marking')}.",
        f"3. **Is the effect monotonic across scales?** {'Yes' if monotonic_fp else 'No'} for road->marking FP count.",
        f"4. **Is 0.85-1.15 enough?** {'Yes' if mild_enough else 'No'} "
        f"(max mild FP change {mild_fp_effect:.2f}%; max strong FP change {strong_fp_effect:.2f}%).",
        f"5. **Should we train H0_rgb_jitter?** {decision}",
        f"6. **Recommended H0 brightness range:** `{h0_range}`.",
        "",
        "## Scale Table",
        "",
        "| scale | road->marking | precision | recall | IoU | F1 | pred/true | mIoU |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in scales_sorted:
        r = by_scale[s]
        lines.append(
            f"| {s:.2f} | {int(r['road_to_marking'])} | "
            f"{r['marking_precision']:.6f} | {r['marking_recall']:.6f} | "
            f"{r['marking_iou']:.6f} | {r['marking_f1']:.6f} | "
            f"{r['predicted_true_marking_ratio']:.3f} | {r['miou']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Limitation",
            "",
            "This probe tests sensitivity to absolute/global RGB brightness scaling only.",
            "It does not test all possible relative, contextual, hue, saturation, camera-exposure,",
            "or geometry/RGB interaction shortcuts.",
            "",
            "## Thesis-safe interpretation",
            "",
        ]
    )
    if clear_effect:
        lines.extend(
            [
                "The frozen G1 model is measurably sensitive to global RGB brightness:",
                "road->marking false positives move when RGB is darkened/brightened.",
                "This supports running a controlled train-only RGB jitter experiment to test",
                "whether mild brightness robustness can reduce bright-road false positives.",
            ]
        )
    else:
        lines.extend(
            [
                "The frozen G1 model is not strongly affected by global RGB brightness scaling",
                "in this sampled probe. This weakens the case for spending a full run on simple",
                "brightness jitter; G1 should remain the default final model unless a larger",
                "probe or a richer RGB representation experiment is justified.",
            ]
        )
    (out_dir / "brightness_sensitivity_recommendation.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    scales = sorted({round(float(s), 6) for s in args.scales})
    if 1.0 not in scales:
        raise SystemExit("--scales must include 1.00 as baseline")
    required = {0.80, 0.85, 1.00, 1.15, 1.20}
    if required.issubset(set(scales)) is False:
        raise SystemExit("This probe expects scales 0.80 0.85 1.00 1.15 1.20")

    set_seeds(args.seed)
    device = choose_device(args.device)
    checkpoint = args.checkpoint or default_checkpoint()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"config {args.config}", flush=True)
    print(f"checkpoint {checkpoint}", flush=True)
    print(f"out_dir {out_dir}", flush=True)
    print(f"device {device}", flush=True)
    print(f"scales {' '.join(f'{s:.2f}' for s in scales)}", flush=True)

    cfg = load_g1_cfg(args.config, args.steps)
    dataset = base.G0DatasetWithRaw(**cfg["dataset"])
    model, _checkpoint_data = base.load_model(cfg, checkpoint, device)
    split = dataset.get_split(args.split)
    sampler = split.sampler
    model.trans_point_sampler = sampler.get_point_sampler()
    torch_split = base.TorchDataloader(
        dataset=split,
        preprocess=model.preprocess,
        transform=model.transform,
        sampler=sampler,
        use_cache=False,
        steps_per_epoch=args.steps,
    )
    pipeline = base.SemanticSegmentation(model=model, dataset=dataset, **cfg["pipeline"])
    batcher = pipeline.get_batcher(device)
    loader = DataLoader(
        torch_split,
        batch_size=1,
        sampler=base.get_sampler(sampler),
        num_workers=0,
        pin_memory=False,
        collate_fn=batcher.collate_fn,
    )

    cms = {scale: np.zeros((3, 3), dtype=np.int64) for scale in scales}
    raw_remap_mismatch = 0
    raw_remap_checked = 0

    with torch.no_grad():
        for inputs in tqdm(loader, desc="g1_rgb_brightness_probe"):
            if hasattr(inputs["data"], "to"):
                inputs["data"].to(device)

            labels_raw_remapped = inputs["data"]["labels"].detach().cpu().numpy().reshape(-1)
            valid_mask = ~np.isin(labels_raw_remapped, np.asarray(model.cfg.ignored_label_inds))
            raw_labels = inputs["data"]["raw_labels"].detach().cpu().numpy().reshape(-1)
            raw_labels_valid = raw_labels[valid_mask]
            expected_active = base.remap_raw_pandaset_ids(
                raw_labels_valid.astype(np.int32),
                label_mode=base.LABEL_MODE_ROAD_MARKING3,
            ) - 1

            original_features = inputs["data"]["features"]
            for scale in scales:
                restore_features(inputs["data"], original_features)
                if abs(scale - 1.0) < 1e-9:
                    pass
                else:
                    apply_rgb_brightness_scale(
                        inputs["data"],
                        scale=scale,
                        threshold=args.rgb_valid_threshold,
                    )

                results = model(inputs["data"])
                scores, y_true = base.filter_valid_label(
                    results,
                    inputs["data"]["labels"],
                    model.cfg.num_classes,
                    model.cfg.ignored_label_inds,
                    device,
                )
                y_pred = torch.argmax(scores, dim=-1)
                y_true_np = y_true.detach().cpu().numpy().astype(np.int64).reshape(-1)
                y_pred_np = y_pred.detach().cpu().numpy().astype(np.int64).reshape(-1)

                if y_true_np.shape != y_pred_np.shape:
                    raise RuntimeError("Prediction/label shape mismatch")
                cms[scale] += confusion_from_arrays(y_true_np, y_pred_np)

                if abs(scale - 1.0) < 1e-9:
                    mismatch = expected_active != y_true_np
                    raw_remap_mismatch += int(mismatch.sum())
                    raw_remap_checked += int(y_true_np.size)

            restore_features(inputs["data"], original_features)

    mismatch_rate = 0.0 if raw_remap_checked == 0 else raw_remap_mismatch / raw_remap_checked
    if mismatch_rate > 0.001:
        raise RuntimeError(f"raw remap mismatch_rate {mismatch_rate:.6f} exceeds 0.001")

    rows: list[dict[str, Any]] = []
    confusions: dict[str, list[list[int]]] = {}
    for scale in scales:
        cm = cms[scale]
        confusions[f"{scale:.2f}"] = cm.tolist()
        rows.append({"scale": scale, **metrics_from_cm(cm)})
    add_delta_columns(rows)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "brightness_sensitivity_summary.csv", index=False)
    (out_dir / "brightness_sensitivity_confusions.json").write_text(
        json.dumps(
            {
                "class_order": {"0": "road", "1": "marking", "2": "other"},
                "scales": [f"{s:.2f}" for s in scales],
                "confusions": confusions,
                "provenance": {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "script": "logs/milestone_h/run_analysis/analysis_code/g1_rgb_brightness_sensitivity.py",
                    "config": str(args.config),
                    "checkpoint": str(checkpoint),
                    "split": args.split,
                    "steps": args.steps,
                    "seed": args.seed,
                    "device": str(device),
                    "rgb_valid_threshold": args.rgb_valid_threshold,
                    "feature_layout": "[x, y, z, intensity, r, g, b, rgb_valid]",
                    "rgb_columns": [4, 5, 6],
                    "rgb_valid_column": 7,
                    "note": "Eval-only global RGB brightness scaling; no training/cache/config mutation.",
                },
            },
            indent=2,
        )
    )
    if args.make_plots:
        make_plots(df, out_dir)
    write_recommendation(out_dir, rows, args, checkpoint, device)

    print(f"wrote {out_dir / 'brightness_sensitivity_summary.csv'}", flush=True)
    print(f"wrote {out_dir / 'brightness_sensitivity_recommendation.md'}", flush=True)
    print("g1_rgb_brightness_sensitivity_status PASS", flush=True)


if __name__ == "__main__":
    main()
