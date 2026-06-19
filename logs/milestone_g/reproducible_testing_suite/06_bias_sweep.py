#!/usr/bin/env python
"""Marking-logit bias sweep (operating-point diagnostic) — analysis only, no training.

Run AFTER the run finishes, on the selected best checkpoint. Question: at the
default argmax operating point, how is precision/recall balanced, and is there
head-room by shifting the marking decision threshold? We answer WITHOUT retraining:

    pred = argmax( scores + b * e_marking )    for a range of biases b

Inference runs ONCE; for every batch the per-point scores are computed a single
time, then re-argmax'd in numpy for all biases (one confusion matrix per bias,
overall + RGB-valid/invalid + per distance bucket). No re-inference per bias.

Reports, per bias: marking IoU, precision, recall, F1, road->marking FP,
marking->road FN, pred/true ratio (+ RGB-valid/invalid and distance pred/true).

IMPORTANT: this is a DIAGNOSTIC. The official evaluation is always plain argmax
(b = 0). A bias-tuned number must NEVER be reported as the model's result; the
sweep only shows the precision/recall trade and whether the argmax point is
well-calibrated. Per the single-seed limitation, treat IoU gains below the noise
floor (~0.008) as not real.

Outputs: <ANALYSIS_OUT>/bias_sweep/bias_sweep_metrics.csv + bias_sweep_summary.md
"""

from __future__ import annotations

import argparse
import csv as _csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from _suite_paths import (  # noqa: E402
    REPO, RUN_NAME, RUN_DIR, ANALYSIS_OUT, CONFIG_YAML, CACHE_DIR, TEST_RESULTS,
)
import _sampled_error_engine as base  # noqa: E402

MARKING_IDX = 1  # active class order: 0 road, 1 marking, 2 other
DEFAULT_BIASES = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
IOU_GATE = 0.008  # single-seed noise floor: gains below this are not "real"


def patch_base() -> None:
    """Point the shared engine at this run (mirrors 02_sampled_error_analysis)."""
    base.RUN_NAME = RUN_NAME
    base.DEFAULT_CONFIG = CONFIG_YAML
    base.G0_RUN_DIR = RUN_DIR          # engine uses this name generically for checkpoints
    base.ANALYSIS_DIR = ANALYSIS_OUT
    original_load_cfg = base.load_cfg

    def load_cfg_run(path: Path, steps: int):  # noqa: ANN001
        cfg = original_load_cfg(path, steps)
        cfg["dataset"]["cache_dir"] = str(CACHE_DIR.resolve())
        cfg["dataset"]["test_result_folder"] = str(TEST_RESULTS.resolve())
        return cfg

    base.load_cfg = load_cfg_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIG_YAML)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Default: discovered best marking-IoU epoch of the run.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--steps", type=int, default=2160)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--rgb-valid-threshold", type=float, default=0.5)
    p.add_argument("--biases", type=float, nargs="+", default=list(DEFAULT_BIASES))
    return p.parse_args()


def cm_from(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    idx = y_true * 3 + y_pred
    return np.bincount(idx, minlength=9).reshape(3, 3).astype(np.int64)


def plot_bias_sweep(rows: list, best_b: float, out_dir: Path) -> None:
    """Operating-point figure (CPU): metrics + error counts vs marking-logit bias."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import _suite_style as S

    S.apply()
    b = [r["bias"] for r in rows]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.6))

    for key, label in [("marking_iou", "IoU"), ("marking_precision", "precision"),
                       ("marking_recall", "recall"), ("marking_f1", "F1")]:
        axL.plot(b, [r[key] for r in rows], marker="o", label=label,
                 color=S.METRIC[label.lower()])
    axL.axvline(0.0, color=S.REFERENCE, ls="--", lw=1.3, label="official argmax (b=0)")
    if abs(best_b) > 1e-9:
        axL.axvline(best_b, color=S.NEUTRAL, ls=":", lw=1.5, label=f"best IoU (b={best_b:+.2f})")
    axL.set_xlabel("marking-logit bias  b")
    axL.set_ylabel("score (0–1)")
    axL.set_ylim(0, 1.0)
    axL.set_title("marking metrics vs decision bias")
    S.style_axes(axL)
    axL.legend(ncol=2)

    axR.plot(b, [r["road_to_marking_fp"] for r in rows], marker="o",
             label="road→marking FP", color=S.TRANSITION["road_to_marking"])
    axR.plot(b, [r["marking_to_road_fn"] for r in rows], marker="s",
             label="marking→road FN", color=S.TRANSITION["marking_to_road"])
    axR.axvline(0.0, color=S.REFERENCE, ls="--", lw=1.3, label="official argmax (b=0)")
    axR.set_xlabel("marking-logit bias  b")
    axR.set_ylabel("error count (sampled points)")
    axR.set_title("precision/recall trade vs decision bias")
    S.style_axes(axR)
    axR.legend()

    fig.suptitle(f"{RUN_NAME}: marking operating-point sweep (diagnostic only; official = b=0)",
                 y=1.0, fontsize=14, fontweight="bold")
    fig.tight_layout(pad=1.6)
    fig.savefig(out_dir / "bias_sweep_operating_point.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    patch_base()
    args = parse_args()
    base.set_seeds(args.seed)
    device = base.choose_device(args.device)

    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    else:
        best = base.discover_best_epoch(RUN_DIR)
        checkpoint = RUN_DIR / "checkpoints" / f"ckpt_epoch_{best:05d}.pth"
    out_dir = (args.out_dir or (ANALYSIS_OUT / "bias_sweep")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    biases = [round(float(b), 4) for b in args.biases]
    print(f"run {RUN_NAME}  using_checkpoint {checkpoint}")
    print(f"biases {biases}")

    cfg = base.load_cfg(args.config, args.steps)
    dataset = base.G0DatasetWithRaw(**cfg["dataset"])
    model, _ = base.load_model(cfg, checkpoint, device)

    split = dataset.get_split(args.split)
    sampler = split.sampler
    model.trans_point_sampler = sampler.get_point_sampler()
    torch_split = base.TorchDataloader(
        dataset=split, preprocess=model.preprocess, transform=model.transform,
        sampler=sampler, use_cache=False, steps_per_epoch=args.steps,
    )
    pipeline = base.SemanticSegmentation(model=model, dataset=dataset, **cfg["pipeline"])
    batcher = pipeline.get_batcher(device)
    loader = base.DataLoader(
        torch_split, batch_size=1, sampler=base.get_sampler(sampler),
        num_workers=0, pin_memory=False, collate_fn=batcher.collate_fn,
    )

    bucket_names = [name for _, _, name in base.DISTANCE_BUCKETS]
    acc = {
        b: {
            "all": np.zeros((3, 3), np.int64),
            "rgb_valid": np.zeros((3, 3), np.int64),
            "rgb_invalid": np.zeros((3, 3), np.int64),
            "bucket": {n: np.zeros((3, 3), np.int64) for n in bucket_names},
        }
        for b in biases
    }
    raw_remap_mismatch = 0
    raw_remap_checked = 0

    with torch.no_grad():
        for inputs in tqdm(loader, desc="bias_sweep"):
            if hasattr(inputs["data"], "to"):
                inputs["data"].to(device)
            results = model(inputs["data"])
            scores, y_true = base.filter_valid_label(
                results, inputs["data"]["labels"],
                model.cfg.num_classes, model.cfg.ignored_label_inds, device,
            )
            scores_np = scores.detach().cpu().numpy().astype(np.float64).reshape(-1, 3)
            y_true_np = y_true.detach().cpu().numpy().astype(np.int64).reshape(-1)

            labels = inputs["data"]["labels"].detach().cpu().numpy().reshape(-1)
            valid_mask = ~np.isin(labels, np.asarray(model.cfg.ignored_label_inds))
            ranges = inputs["data"]["ranges"].detach().cpu().numpy().reshape(-1)[valid_mask]
            rgb_valid = inputs["data"]["analysis_rgb_valid"].detach().cpu().numpy().reshape(-1)[valid_mask]
            rgb_valid_mask = rgb_valid > args.rgb_valid_threshold

            raw_labels = inputs["data"]["raw_labels"].detach().cpu().numpy().reshape(-1)[valid_mask]
            expected_active = base.remap_raw_pandaset_ids(
                raw_labels.astype(np.int32), label_mode=base.LABEL_MODE_ROAD_MARKING3) - 1
            raw_remap_mismatch += int((expected_active != y_true_np).sum())
            raw_remap_checked += int(y_true_np.size)

            if not (y_true_np.shape == ranges.shape == rgb_valid_mask.shape == scores_np.shape[:1]):
                raise RuntimeError("Bias-sweep batch shape mismatch.")

            buckets = base.bucket_name(ranges)
            bucket_masks = {n: (buckets == n) for n in np.unique(buckets)}

            for b in biases:
                biased = scores_np.copy()
                biased[:, MARKING_IDX] += b
                y_pred = np.argmax(biased, axis=1).astype(np.int64)
                a = acc[b]
                a["all"] += cm_from(y_true_np, y_pred)
                if rgb_valid_mask.any():
                    a["rgb_valid"] += cm_from(y_true_np[rgb_valid_mask], y_pred[rgb_valid_mask])
                inv = ~rgb_valid_mask
                if inv.any():
                    a["rgb_invalid"] += cm_from(y_true_np[inv], y_pred[inv])
                for n, m in bucket_masks.items():
                    a["bucket"][n] += cm_from(y_true_np[m], y_pred[m])

    mismatch_rate = 0.0 if raw_remap_checked == 0 else raw_remap_mismatch / raw_remap_checked
    if mismatch_rate > 0.001:
        raise RuntimeError(f"raw remap mismatch_rate={mismatch_rate:.6f} exceeds 0.001")

    rows = []
    for b in biases:
        a = acc[b]
        m = base.metrics_from_cm(a["all"])
        mv = base.metrics_from_cm(a["rgb_valid"])
        mi = base.metrics_from_cm(a["rgb_invalid"])
        row = {
            "bias": b,
            "marking_iou": m["marking_iou"],
            "marking_precision": m["marking_precision"],
            "marking_recall": m["marking_recall"],
            "marking_f1": m["marking_f1"],
            "pred_true": m["predicted_true_marking_ratio"],
            "road_to_marking_fp": m["road_to_marking"],
            "marking_to_road_fn": m["marking_to_road"],
            "rgb_valid_pred_true": mv["predicted_true_marking_ratio"],
            "rgb_invalid_pred_true": mi["predicted_true_marking_ratio"],
            "rgb_valid_iou": mv["marking_iou"],
            "rgb_invalid_iou": mi["marking_iou"],
        }
        for n in bucket_names:
            row[f"pred_true_{n}"] = base.metrics_from_cm(a["bucket"][n])["predicted_true_marking_ratio"]
        rows.append(row)

    keys = list(rows[0].keys())
    with (out_dir / "bias_sweep_metrics.csv").open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    by_bias = {r["bias"]: r for r in rows}
    base_iou = by_bias.get(0.0, {}).get("marking_iou", float("nan"))
    best = max(rows, key=lambda r: r["marking_iou"])
    best_b, gain = best["bias"], best["marking_iou"] - base_iou

    if abs(best_b) < 1e-9 or gain < IOU_GATE:
        verdict = (f"Argmax (b=0) is at/near the best operating point (best bias {best_b:+.2f}, "
                   f"IoU gain {gain:+.4f} < {IOU_GATE} noise floor). The model is well-calibrated; "
                   "the precision/recall trade is not freely improvable by thresholding.")
    elif best_b > 0:
        verdict = (f"A positive marking bias (b={best_b:+.2f}) raises IoU by {gain:+.4f} -> the "
                   "argmax point is slightly conservative (recall head-room). Diagnostic only.")
    else:
        verdict = (f"A negative marking bias (b={best_b:+.2f}) raises IoU by {gain:+.4f} -> the "
                   "argmax point slightly over-predicts (precision head-room). Diagnostic only.")

    lines = [
        f"# {RUN_NAME} — Marking-Logit Bias Sweep (operating-point diagnostic)",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- split/steps/seed: {args.split} / {args.steps} / {args.seed}",
        f"- raw-remap mismatch rate: {mismatch_rate:.2e}",
        "",
        "**Diagnostic only.** The official evaluation is plain argmax (b = 0); a bias-tuned",
        "number is NOT reported as the model's result. Treat IoU gains below the single-seed",
        f"noise floor (~{IOU_GATE}) as not real.",
        "",
        "| bias | IoU | precision | recall | F1 | pred/true | road->mk FP | mk->road FN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        star = "  <-- b=0" if abs(r["bias"]) < 1e-9 else ("  <-- best IoU" if r["bias"] == best_b else "")
        lines.append(
            f"| {r['bias']:+.2f} | {r['marking_iou']:.6f} | {r['marking_precision']:.4f} | "
            f"{r['marking_recall']:.4f} | {r['marking_f1']:.4f} | {r['pred_true']:.3f} | "
            f"{int(r['road_to_marking_fp'])} | {int(r['marking_to_road_fn'])} |{star}"
        )
    lines += [
        "", f"- IoU at b=0 (official argmax): `{base_iou:.6f}`",
        f"- best IoU at b=`{best_b:+.2f}`: `{best['marking_iou']:.6f}` (gain `{gain:+.4f}`)",
        "", "## Verdict", "", verdict, "",
        "Per-distance pred/true columns (`pred_true_*`) are in the CSV: a positive bias pulling "
        "the far buckets back toward 1.0 indicates under-prediction at range.", "",
    ]
    (out_dir / "bias_sweep_summary.md").write_text("\n".join(lines))

    plot_bias_sweep(rows, best_b, out_dir)

    print(f"out_dir {out_dir}")
    print(f"base_iou {base_iou:.6f} best_bias {best_b:+.2f} best_iou {best['marking_iou']:.6f} gain {gain:+.4f}")
    print("bias_sweep_status PASS")


if __name__ == "__main__":
    main()
