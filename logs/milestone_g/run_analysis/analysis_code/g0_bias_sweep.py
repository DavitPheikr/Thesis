#!/usr/bin/env python
"""Marking-logit bias sweep on the G0 best checkpoint (decision diagnostic).

Question: is G0 already well-calibrated at argmax, or is it too conservative
(leaving marking recall / IoU on the table)? We answer it WITHOUT retraining by
shifting the marking logit before argmax and measuring how marking IoU responds:

    pred = argmax( logits + b * e_marking )      for b in BIASES

Efficiency: inference runs ONCE. For every batch we compute the per-point class
scores a single time, then re-argmax in numpy for all biases and accumulate one
confusion matrix per bias (overall + RGB-valid/invalid + per distance bucket).
No re-inference per bias.

This is a DIAGNOSTIC, not a deliverable: a bias-tuned IoU must never be reported
as the model's result. It only tells us the *direction* of any headroom:

    best bias ~ 0      -> G0 is well-calibrated      -> don't lower lambda
    positive bias best -> G0 is too conservative     -> lambda=0.35 is justified
    negative bias best -> G0 still slightly overpred -> don't lower lambda

Decision gate: act only if the best bias improves marking IoU by >= ~0.008-0.010
over b=0 (all runs are single-seed, so smaller deltas are within noise).

Reuses the G0 sampled-error-analysis machinery (dataset/model/transform) so the
inference path is byte-for-byte the same as g0_sampled_error_analysis.py.

Outputs:
    logs/milestone_g/run_analysis/G0_rgb_lovasz/bias_sweep/bias_sweep_metrics.csv
    logs/milestone_g/run_analysis/G0_rgb_lovasz/bias_sweep/bias_sweep_summary.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Reuse the exact sampled-analysis setup (heavy imports happen here).
import g0_sampled_error_analysis as base


MARKING_IDX = 1  # active class order: 0 road, 1 marking, 2 other
DEFAULT_BIASES = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
IOU_GATE = 0.008  # minimum IoU gain over b=0 to be considered real (single-seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=base.DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Default: discovered best marking-IoU epoch of G0.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--steps", type=int, default=2160)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--rgb-valid-threshold", type=float, default=0.5)
    p.add_argument("--biases", type=float, nargs="+", default=list(DEFAULT_BIASES))
    return p.parse_args()


def cm_from(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """3x3 confusion via bincount; rows=true, cols=pred."""
    idx = y_true * 3 + y_pred
    return np.bincount(idx, minlength=9).reshape(3, 3).astype(np.int64)


def main() -> None:
    args = parse_args()
    base.set_seeds(args.seed)
    device = base.choose_device(args.device)

    # Resolve checkpoint (default = best marking-IoU epoch) and output dir.
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    else:
        best = base.discover_best_epoch(base.G0_RUN_DIR)
        checkpoint = base.G0_RUN_DIR / "checkpoints" / f"ckpt_epoch_{best:05d}.pth"
    out_dir = (args.out_dir or (base.ANALYSIS_DIR / "bias_sweep")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    biases = [round(float(b), 4) for b in args.biases]
    print(f"using_checkpoint {checkpoint}")
    print(f"biases {biases}")

    cfg = base.load_cfg(args.config, args.steps)
    dataset = base.G0DatasetWithRaw(**cfg["dataset"])
    model, checkpoint_data = base.load_model(cfg, checkpoint, device)

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
    # accumulators[bias] = {"all":3x3, "rgb_valid":3x3, "rgb_invalid":3x3,
    #                       "bucket": {name: 3x3}}
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
        for inputs in tqdm(loader, desc="g0_bias_sweep"):
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
            rgb_valid = (
                inputs["data"]["analysis_rgb_valid"].detach().cpu().numpy().reshape(-1)[valid_mask]
            )
            rgb_valid_mask = rgb_valid > args.rgb_valid_threshold

            # raw-remap alignment guard (same as sampled analysis)
            raw_labels = inputs["data"]["raw_labels"].detach().cpu().numpy().reshape(-1)[valid_mask]
            expected_active = base.remap_raw_pandaset_ids(
                raw_labels.astype(np.int32), label_mode=base.LABEL_MODE_ROAD_MARKING3
            ) - 1
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

    # Build per-bias metrics table.
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
            "road_to_marking": m["road_to_marking"],
            "marking_to_road": m["marking_to_road"],
            "rgb_valid_pred_true": mv["predicted_true_marking_ratio"],
            "rgb_invalid_pred_true": mi["predicted_true_marking_ratio"],
            "rgb_valid_iou": mv["marking_iou"],
            "rgb_invalid_iou": mi["marking_iou"],
        }
        for n in bucket_names:
            row[f"pred_true_{n}"] = base.metrics_from_cm(a["bucket"][n])["predicted_true_marking_ratio"]
        rows.append(row)

    import csv as _csv
    keys = list(rows[0].keys())
    with (out_dir / "bias_sweep_metrics.csv").open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # Interpretation.
    by_bias = {r["bias"]: r for r in rows}
    base_iou = by_bias.get(0.0, {}).get("marking_iou", float("nan"))
    best = max(rows, key=lambda r: r["marking_iou"])
    best_b = best["bias"]
    gain = best["marking_iou"] - base_iou

    if abs(best_b) < 1e-9 or gain < IOU_GATE:
        verdict = ("b=0 is best (or gain below the single-seed gate) -> G0 is "
                   "well-calibrated. Do NOT lower lambda. Stop, or run only a "
                   "schedule continuation as polish.")
        route = "CASE B: keep G0; optional schedule-extend run."
    elif best_b > 0:
        verdict = (f"positive bias b={best_b:+.2f} improves IoU by {gain:+.4f} "
                   f"(>= {IOU_GATE}) -> G0 is too conservative. lambda=0.35 is "
                   "justified as the overnight run.")
        route = "CASE A: run G1_rgb_lovasz035 (lovasz_lambda 0.5 -> 0.35)."
    else:
        verdict = (f"negative bias b={best_b:+.2f} improves IoU by {gain:+.4f} "
                   "-> G0 still slightly overpredicts. Do NOT lower lambda. Stop "
                   "(or schedule polish only).")
        route = "CASE C: stop; do not lower lambda."

    lines = [
        "# G0 Marking-Logit Bias Sweep",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- split/steps/seed: {args.split} / {args.steps} / {args.seed}",
        f"- raw-remap mismatch rate: {mismatch_rate:.2e}",
        f"- decision gate: IoU gain over b=0 must be >= {IOU_GATE}",
        "",
        "**This is a diagnostic only.** A bias-tuned IoU must NOT be reported as the",
        "model's result; it only reveals the direction of calibration headroom.",
        "",
        "| bias | IoU | precision | recall | F1 | pred/true | rgb_valid p/t | rgb_invalid p/t |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        star = "  <-- b=0" if abs(r["bias"]) < 1e-9 else ("  <-- best IoU" if r["bias"] == best_b else "")
        lines.append(
            f"| {r['bias']:+.2f} | {r['marking_iou']:.6f} | {r['marking_precision']:.4f} | "
            f"{r['marking_recall']:.4f} | {r['marking_f1']:.4f} | {r['pred_true']:.3f} | "
            f"{r['rgb_valid_pred_true']:.3f} | {r['rgb_invalid_pred_true']:.3f} |{star}"
        )
    lines += [
        "",
        f"- IoU at b=0: `{base_iou:.6f}`",
        f"- best IoU at b=`{best_b:+.2f}`: `{best['marking_iou']:.6f}` (gain `{gain:+.4f}`)",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        f"**Route:** {route}",
        "",
        "Long-range pred/true columns (`pred_true_*`) are in the CSV: check whether a",
        "positive bias pulls the >30 m buckets back toward 1.0 (G0 under-predicts there).",
        "",
    ]
    (out_dir / "bias_sweep_summary.md").write_text("\n".join(lines))

    print(f"out_dir {out_dir}")
    print(f"base_iou {base_iou:.6f} best_bias {best_b:+.2f} best_iou {best['marking_iou']:.6f} gain {gain:+.4f}")
    print(route)
    print("script_status PASS")


if __name__ == "__main__":
    main()
