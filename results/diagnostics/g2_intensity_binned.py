#!/usr/bin/env python
"""G2 held-out test diagnostic: intensity-binned road-marking behaviour.

ADDITIVE DIAGNOSTIC ONLY. This script does not modify, overwrite, or delete any
existing file. It reuses the *exact* full-coverage G2 test evaluation machinery
of ``results/test_suite/_sampled_error_engine.py`` (imported, never edited) with
the same config snapshot, checkpoint, spatially-regular full-coverage sampler,
and seed that produced
``results/per_model/G2_lidar_rgb_lovasz/test/full/seed_42/``. It then tallies the
per-point quantities the engine already computes (un-normalized intensity,
predicted/true active class, rgb_valid, range) into intensity bins and writes four
CSVs + a README into ``results/diagnostics/G2_intensity_binned_test/``.

Counts are PATCH-ACCUMULATED active evaluations (a physical point covered by
several overlapping patches is counted once per patch) -- identical to the
convention behind the existing full-coverage confusion matrix. See README.

Hard validation (abort, write nothing, if any check fails):
  1. sum of bin point_count == evaluated active-point count == confusion total.
  2. summed 3x3 confusion across the run == existing seed-42 full-test
     confusion_matrix.npy (exact).
  3. reconstructed marking IoU / precision / recall == seed-42 full-test values
     (IoU ~= 0.5177, precision ~= 0.6811, recall ~= 0.6833) within tolerance.

Run on the server (A100); see the command printed in the project chat.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- import the existing engine WITHOUT modifying it -------------------------
SUITE_DIR = Path(__file__).resolve().parents[1] / "test_suite"
sys.path.insert(0, str(SUITE_DIR))
import _sampled_error_engine as eng  # noqa: E402  (sets project root, sys.path, imports ml3d)

PROJECT_ROOT = eng.PROJECT_ROOT
OUT_DIR = PROJECT_ROOT / "results/diagnostics/G2_intensity_binned_test"

# Fixed analysis constants (G2 final model, seed-42 full-coverage test).
G2_RUN_DIR = PROJECT_ROOT / "logs/milestone_g/runs/G2_schedule_extend_100"
DEFAULT_CONFIG = G2_RUN_DIR / "config_snapshot.yml"
DEFAULT_CKPT = G2_RUN_DIR / "checkpoints/ckpt_epoch_00068.pth"
EXISTING_CM = (
    PROJECT_ROOT
    / "results/per_model/G2_lidar_rgb_lovasz/test/full/seed_42/confusion_matrix.npy"
)

# Intensity bins: half-open [lo, hi) except the final bin [80, 114] (inclusive).
# Intensity is clipped to [0, 114] during normalization, so values > 114 cannot
# occur and no "114+" bin is emitted (asserted at runtime).
INTENSITY_BINS = [
    (0.0, 10.0, "0-10"),
    (10.0, 20.0, "10-20"),
    (20.0, 30.0, "20-30"),
    (30.0, 40.0, "30-40"),
    (40.0, 60.0, "40-60"),
    (60.0, 80.0, "60-80"),
    (80.0, 114.0, "80-114"),
]
CLIP_HIGH = 114.0

# Distance buckets: reuse the engine's exact bucketing (identical names to the
# existing distance_bucket_metrics.csv).
DIST_BUCKET_ORDER = [name for (_lo, _hi, name) in eng.DISTANCE_BUCKETS]

# seed-42 full-test reference marking metrics (for validation check #3).
REF_IOU, REF_PREC, REF_RECALL = 0.5177, 0.6811, 0.6833
REF_TOL = 5e-3

ROAD, MARKING, OTHER = 0, 1, 2


# ---------------------------------------------------------------------------
# Metric block for an arbitrary boolean mask over all evaluated active points.
# ---------------------------------------------------------------------------
def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else float("nan")


def metrics_block(T, P, R, V, mask) -> dict:
    t = T[mask]
    p = P[mask]
    r = R[mask]
    v = V[mask]
    n = int(t.size)

    true_road = int((t == ROAD).sum())
    true_marking = int((t == MARKING).sum())
    true_other = int((t == OTHER).sum())
    pred_marking = int((p == MARKING).sum())

    marking_tp = int(((t == MARKING) & (p == MARKING)).sum())
    marking_fn = int(((t == MARKING) & (p != MARKING)).sum())
    marking_to_road = int(((t == MARKING) & (p == ROAD)).sum())
    marking_to_other = int(((t == MARKING) & (p == OTHER)).sum())
    road_to_marking_fp = int(((t == ROAD) & (p == MARKING)).sum())
    other_to_marking_fp = int(((t == OTHER) & (p == MARKING)).sum())
    marking_fp_total = int(((p == MARKING) & (t != MARKING)).sum())

    return {
        "point_count": n,
        "true_road_count": true_road,
        "true_marking_count": true_marking,
        "true_other_count": true_other,
        "pred_marking_count": pred_marking,
        "marking_tp": marking_tp,
        "marking_fn": marking_fn,
        "marking_to_road": marking_to_road,
        "marking_to_other": marking_to_other,
        "road_to_marking_fp": road_to_marking_fp,
        "other_to_marking_fp": other_to_marking_fp,
        "marking_fp_total": marking_fp_total,
        "marking_recall": _safe_div(marking_tp, true_marking),
        "marking_precision": _safe_div(marking_tp, pred_marking),
        "marking_iou": _safe_div(marking_tp, marking_tp + marking_fp_total + marking_fn),
        "pred_marking_rate": _safe_div(pred_marking, n),
        "road_to_marking_rate": _safe_div(road_to_marking_fp, true_road),
        "other_to_marking_rate": _safe_div(other_to_marking_fp, true_other),
        "pred_true_ratio_marking": _safe_div(pred_marking, true_marking),
        "rgb_valid_ratio": (float(v.mean()) if n > 0 else float("nan")),
        "mean_range_m": (float(r.mean()) if n > 0 else float("nan")),
        "median_range_m": (float(np.median(r)) if n > 0 else float("nan")),
    }


def cm_from_TP(T, P) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            cm[i, j] = int(((T == i) & (P == j)).sum())
    return cm


def marking_metrics_from_cm(cm: np.ndarray) -> tuple[float, float, float]:
    tp = int(cm[1, 1])
    fn = int(cm[1, 0] + cm[1, 2])
    fp = int(cm[0, 1] + cm[2, 1])
    iou = _safe_div(tp, tp + fp + fn)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    return iou, prec, rec


# ---------------------------------------------------------------------------
def run_inference(args) -> dict:
    """Reuse the engine's exact setup; return concatenated per-point arrays."""
    eng.set_seeds(args.seed)
    device = eng.choose_device(args.device)
    print(f"device={device}  checkpoint={args.checkpoint}")

    cfg = eng.load_cfg(args.config, args.steps)
    dataset = eng.G0DatasetWithRaw(**cfg["dataset"])
    model, _ = eng.load_model(cfg, args.checkpoint, device)

    split = dataset.get_split(args.split)
    # Full-coverage spatially-regular sampler over every frame (headline protocol).
    if not isinstance(split.sampler, eng.SemSegSpatiallyRegularSampler):
        split.sampler = eng.SemSegSpatiallyRegularSampler(split)
    split.sampler.split = "test"  # forces gen_test full-coverage walk
    sampler = split.sampler
    model.trans_point_sampler = sampler.get_point_sampler()
    torch_split = eng.TorchDataloader(
        dataset=split,
        preprocess=model.preprocess,
        transform=model.transform,
        sampler=sampler,
        use_cache=False,
        steps_per_epoch=None,  # uncapped -> covers all frames
    )
    pipeline = eng.SemanticSegmentation(model=model, dataset=dataset, **cfg["pipeline"])
    batcher = pipeline.get_batcher(device)
    loader = DataLoader(
        torch_split,
        batch_size=1,
        sampler=eng.get_sampler(sampler),
        num_workers=0,
        pin_memory=False,
        collate_fn=batcher.collate_fn,
    )

    mean = float(dataset.intensity_mean)
    std = float(dataset.intensity_std)

    I_parts, R_parts, T_parts, P_parts, V_parts = [], [], [], [], []
    raw_mismatch = 0
    raw_checked = 0

    with torch.no_grad():
        for inputs in tqdm(loader, desc="g2_full_coverage_test"):
            if hasattr(inputs["data"], "to"):
                inputs["data"].to(device)
            results = model(inputs["data"])
            scores, y_true = eng.filter_valid_label(
                results,
                inputs["data"]["labels"],
                model.cfg.num_classes,
                model.cfg.ignored_label_inds,
                device,
            )
            y_pred = torch.argmax(scores, dim=-1)

            labels_remapped = inputs["data"]["labels"].detach().cpu().numpy().reshape(-1)
            valid_mask = ~np.isin(labels_remapped, np.asarray(model.cfg.ignored_label_inds))
            raw_labels = inputs["data"]["raw_labels"].detach().cpu().numpy().reshape(-1)
            ranges = inputs["data"]["ranges"].detach().cpu().numpy().reshape(-1)
            intensity_z = (
                inputs["data"]["analysis_intensity_z"].detach().cpu().numpy().reshape(-1)
            )
            rgb_valid = (
                inputs["data"]["analysis_rgb_valid"].detach().cpu().numpy().reshape(-1)
            )

            y_true_np = y_true.detach().cpu().numpy().astype(np.int64).reshape(-1)
            y_pred_np = y_pred.detach().cpu().numpy().astype(np.int64).reshape(-1)
            ranges_valid = ranges[valid_mask]
            intensity_valid = (intensity_z[valid_mask] * std + mean).astype(np.float32)
            rgb_valid_valid = rgb_valid[valid_mask]
            rgb_valid_mask = rgb_valid_valid > args.rgb_valid_threshold

            # Integrity: raw remap must reproduce the active labels (engine check).
            expected_active = (
                eng.remap_raw_pandaset_ids(
                    raw_labels[valid_mask].astype(np.int32),
                    label_mode=eng.LABEL_MODE_ROAD_MARKING3,
                )
                - 1
            )
            raw_mismatch += int((expected_active != y_true_np).sum())
            raw_checked += int(y_true_np.size)

            if not (
                y_true_np.shape
                == y_pred_np.shape
                == ranges_valid.shape
                == intensity_valid.shape
                == rgb_valid_mask.shape
            ):
                raise RuntimeError("per-point array shape mismatch in batch")

            I_parts.append(intensity_valid)
            R_parts.append(ranges_valid.astype(np.float32))
            T_parts.append(y_true_np.astype(np.int8))
            P_parts.append(y_pred_np.astype(np.int8))
            V_parts.append(rgb_valid_mask.astype(bool))

    out = {
        "I": np.concatenate(I_parts),
        "R": np.concatenate(R_parts),
        "T": np.concatenate(T_parts),
        "P": np.concatenate(P_parts),
        "V": np.concatenate(V_parts),
        "raw_mismatch_rate": _safe_div(raw_mismatch, raw_checked),
        "intensity_mean": mean,
        "intensity_std": std,
    }
    return out


# ---------------------------------------------------------------------------
def assign_intensity_bins(I: np.ndarray):
    label = np.empty(I.shape, dtype=object)
    assigned = np.zeros(I.shape, dtype=bool)
    for k, (lo, hi, name) in enumerate(INTENSITY_BINS):
        last = k == len(INTENSITY_BINS) - 1
        m = (I >= lo) & (I <= hi) if last else (I >= lo) & (I < hi)
        label[m] = name
        assigned |= m
    return label, assigned


def percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {k: float("nan") for k in
                ("count", "mean", "p05", "p10", "p25", "median", "p75", "p90", "p95", "min", "max")}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def build_tables(data: dict):
    I, R, T, P, V = data["I"], data["R"], data["T"], data["P"], data["V"]
    N = int(I.size)

    ibin, assigned = assign_intensity_bins(I)
    if not assigned.all():
        bad = I[~assigned]
        raise SystemExit(
            f"VALIDATION FAIL: {bad.size} active points fell outside the intensity "
            f"bins (min={bad.min()}, max={bad.max()}). Intensity was expected to be "
            f"clipped to [0, {CLIP_HIGH}]. Aborting; no CSVs written."
        )
    dist = eng.bucket_name(R)

    base_cols = [
        "intensity_bin", "intensity_min", "intensity_max",
        "point_count", "true_road_count", "true_marking_count", "true_other_count",
        "pred_marking_count", "marking_tp", "marking_fn", "marking_to_road",
        "marking_to_other", "road_to_marking_fp", "other_to_marking_fp",
        "marking_fp_total", "marking_recall", "marking_precision", "marking_iou",
        "pred_marking_rate", "road_to_marking_rate", "other_to_marking_rate",
        "pred_true_ratio_marking", "rgb_valid_ratio", "mean_range_m", "median_range_m",
    ]

    # 1) intensity_bin_metrics
    rows = []
    for lo, hi, name in INTENSITY_BINS:
        mask = ibin == name
        row = {"intensity_bin": name, "intensity_min": lo, "intensity_max": hi}
        row.update(metrics_block(T, P, R, V, mask))
        rows.append(row)
    main_df = pd.DataFrame(rows, columns=base_cols)

    # 2) intensity_bin x rgb group
    rows = []
    for lo, hi, name in INTENSITY_BINS:
        for grp, gmask in (("rgb_valid", V), ("rgb_invalid", ~V)):
            mask = (ibin == name) & gmask
            row = {"intensity_bin": name, "intensity_min": lo, "intensity_max": hi,
                   "rgb_group": grp}
            row.update(metrics_block(T, P, R, V, mask))
            rows.append(row)
    rgb_df = pd.DataFrame(rows, columns=["rgb_group"] + base_cols)

    # 3) intensity_bin x distance bucket
    rows = []
    for lo, hi, name in INTENSITY_BINS:
        for b in DIST_BUCKET_ORDER:
            mask = (ibin == name) & (dist == b)
            row = {"intensity_bin": name, "intensity_min": lo, "intensity_max": hi,
                   "distance_bucket": b}
            row.update(metrics_block(T, P, R, V, mask))
            rows.append(row)
    dist_df = pd.DataFrame(rows, columns=["distance_bucket"] + base_cols)

    # 4) overlap summary
    overlap_df = build_overlap(I, T)

    return main_df, rgb_df, dist_df, overlap_df, N


def build_overlap(I: np.ndarray, T: np.ndarray) -> pd.DataFrame:
    road_I = I[T == ROAD]
    mark_I = I[T == MARKING]
    other_I = I[T == OTHER]
    stats = {"road": percentiles(road_I), "marking": percentiles(mark_I),
             "other": percentiles(other_I)}

    cols = ["row_type", "class_name", "count", "mean", "p05", "p10", "p25",
            "median", "p75", "p90", "p95", "min", "max",
            "metric", "value", "count_value", "pct_value", "basis_count"]
    rows = []
    for cls in ("road", "marking", "other"):
        s = stats[cls]
        rows.append({"row_type": "class_stat", "class_name": cls, **s,
                     "metric": "", "value": float("nan"),
                     "count_value": float("nan"), "pct_value": float("nan"),
                     "basis_count": float("nan")})

    road_p10, road_p25, road_p75, road_p90 = (
        stats["road"]["p10"], stats["road"]["p25"], stats["road"]["p75"], stats["road"]["p90"])
    mark_p25, mark_med, mark_p75 = (
        stats["marking"]["p25"], stats["marking"]["median"], stats["marking"]["p75"])

    def thr(name, val):
        rows.append({"row_type": "threshold", "class_name": "", "metric": name,
                     "value": float(val), "count_value": float("nan"),
                     "pct_value": float("nan"), "basis_count": float("nan"),
                     **{k: float("nan") for k in
                        ("count", "mean", "p05", "p10", "p25", "median", "p75", "p90", "p95", "min", "max")}})

    for nm, val in (("road_p10", road_p10), ("road_p25", road_p25), ("road_p75", road_p75),
                    ("road_p90", road_p90), ("marking_p25", mark_p25),
                    ("marking_median", mark_med), ("marking_p75", mark_p75)):
        thr(nm, val)

    n_mark = int(mark_I.size)
    n_road = int(road_I.size)

    def overlap(name, count, basis, subject):
        rows.append({"row_type": "overlap", "class_name": subject, "metric": name,
                     "value": float("nan"),
                     "count_value": int(count),
                     "pct_value": _safe_div(100.0 * count, basis),
                     "basis_count": int(basis),
                     **{k: float("nan") for k in
                        ("count", "mean", "p05", "p10", "p25", "median", "p75", "p90", "p95", "min", "max")}})

    # marking points with road-like intensity
    overlap("marking_intensity_le_road_p75", int((mark_I <= road_p75).sum()), n_mark, "marking")
    overlap("marking_intensity_in_road_p25_p75",
            int(((mark_I >= road_p25) & (mark_I <= road_p75)).sum()), n_mark, "marking")
    overlap("marking_intensity_in_road_p10_p90",
            int(((mark_I >= road_p10) & (mark_I <= road_p90)).sum()), n_mark, "marking")
    # road points with marking-like intensity
    overlap("road_intensity_ge_marking_p25", int((road_I >= mark_p25).sum()), n_road, "road")
    overlap("road_intensity_ge_marking_median", int((road_I >= mark_med).sum()), n_road, "road")
    overlap("road_intensity_ge_marking_p75", int((road_I >= mark_p75).sum()), n_road, "road")

    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
def validate(main_df: pd.DataFrame, data: dict, N: int) -> dict:
    T, P = data["T"], data["P"]
    cm = cm_from_TP(T, P)

    report = {}
    # check 1: bin point_count sum == active count == cm total
    bin_sum = int(main_df["point_count"].sum())
    report["check1_bin_sum"] = bin_sum
    report["check1_active_N"] = int(N)
    report["check1_cm_total"] = int(cm.sum())
    c1 = bin_sum == N == int(cm.sum())

    # check 2: confusion matches existing seed-42 full-test cm exactly
    if not EXISTING_CM.exists():
        raise SystemExit(f"VALIDATION FAIL: reference confusion not found: {EXISTING_CM}")
    ref_cm = np.load(EXISTING_CM).astype(np.int64)
    c2 = bool(np.array_equal(cm, ref_cm))
    report["check2_cm_match"] = c2
    report["this_cm"] = cm.tolist()
    report["ref_cm"] = ref_cm.tolist()

    # check 3: reconstructed marking metrics within tolerance of seed-42 values
    iou, prec, rec = marking_metrics_from_cm(cm)
    report["recon_iou"] = iou
    report["recon_precision"] = prec
    report["recon_recall"] = rec
    c3 = (abs(iou - REF_IOU) <= REF_TOL and abs(prec - REF_PREC) <= REF_TOL
          and abs(rec - REF_RECALL) <= REF_TOL)
    report["raw_mismatch_rate"] = data["raw_mismatch_rate"]

    report["check1_pass"] = bool(c1)
    report["check2_pass"] = bool(c2)
    report["check3_pass"] = bool(c3)
    report["all_pass"] = bool(c1 and c2 and c3)
    return report


def write_readme(path: Path, args, data: dict, report: dict, N: int) -> None:
    txt = f"""# G2 intensity-binned test diagnostic

Additive diagnostic for thesis Chapter 6.6. Created by
`results/diagnostics/g2_intensity_binned.py`. No existing file is modified.

## Provenance
- Model / run: G2 (`logs/milestone_g/runs/G2_schedule_extend_100`), final reported model.
- Checkpoint: `{args.checkpoint}` (epoch 68).
- Config: `{args.config}` (the run's frozen `config_snapshot.yml`).
- Split: `{args.split}` (held-out test).
- Coverage: **full** (spatially-regular, every frame covered ~8x).
- Seed: {args.seed} (matches the existing seed-42 full-coverage stratified analysis;
  NOT the 3-seed headline mean).
- Inference: **rerun** (no per-point data was saved by the original run; this is a
  fresh forward pass with the identical setup).

## Per-point sources (from the reused engine, `_sampled_error_engine.py`)
- true active class (road=0, marking=1, other=2): `filter_valid_label(...)` output `y_true`.
- predicted class: `argmax(model scores)`.
- raw/un-normalized intensity: `analysis_intensity_z * std + mean`
  (mean={data['intensity_mean']:.6f}, std={data['intensity_std']:.6f}); clipped to [0, 114].
- rgb_valid: `analysis_rgb_valid > {args.rgb_valid_threshold}`.
- range (m): per-point `ranges`.
- raw->active remap mismatch rate: {report['raw_mismatch_rate']:.2e} (integrity check).

## IMPORTANT: counts are patch-accumulated, not unique physical points
Full coverage evaluates each point under several overlapping patches, and counts
are accumulated **per patch** (per-patch accumulation, not per-point voting). A
physical point covered N times contributes N to the counts. This is exactly the
convention behind the existing full-coverage confusion matrix
(`results/per_model/G2_lidar_rgb_lovasz/test/full/seed_42/confusion_matrix.npy`),
which is why the summed confusion here matches it exactly. Recall / precision /
overlap percentages are therefore over patch-accumulated active evaluations, on
the same basis as the reported headline metrics.

## Intensity bins
{', '.join(f'[{lo},{hi}{"]" if i==len(INTENSITY_BINS)-1 else ")"}={nm}' for i,(lo,hi,nm) in enumerate(INTENSITY_BINS))}.
Half-open `[min,max)` except the final bin `[80,114]` (closed). Intensity is clipped
to 114, so no `114+` bin exists.

## Distance buckets
{', '.join(DIST_BUCKET_ORDER)} (identical to the existing `distance_bucket_metrics.csv`).

## Metric definitions
marking_tp = label==marking & pred==marking; marking_fn = label==marking & pred!=marking;
marking_to_road / marking_to_other = label==marking & pred==road / other;
road_to_marking_fp = label==road & pred==marking; other_to_marking_fp = label==other & pred==marking;
marking_fp_total = pred==marking & label!=marking;
marking_recall = tp/true_marking_count; marking_precision = tp/pred_marking_count;
marking_iou = tp/(tp+marking_fp_total+marking_fn) (within-bin IoU);
pred_marking_rate = pred_marking_count/point_count;
road_to_marking_rate = road_to_marking_fp/true_road_count;
other_to_marking_rate = other_to_marking_fp/true_other_count;
pred_true_ratio_marking = pred_marking_count/true_marking_count.
Undefined divisions are NaN (not zero).

## Validation checks (all required to pass before any CSV is written)
1. bin point_count sum ({report['check1_bin_sum']}) == active N ({report['check1_active_N']})
   == confusion total ({report['check1_cm_total']}): {report['check1_pass']}.
2. summed 3x3 confusion == existing seed-42 full-test confusion_matrix.npy (exact): {report['check2_pass']}.
3. reconstructed marking IoU={report['recon_iou']:.4f} (ref ~0.5177),
   precision={report['recon_precision']:.4f} (ref ~0.6811),
   recall={report['recon_recall']:.4f} (ref ~0.6833), tol {REF_TOL}: {report['check3_pass']}.

## Files
- intensity_bin_metrics.csv          (main: metrics per intensity bin)
- intensity_bin_by_rgb_valid_metrics.csv (intensity bin x rgb_valid/rgb_invalid)
- intensity_bin_by_distance_metrics.csv  (intensity bin x distance bucket)
- intensity_overlap_summary.csv      (class intensity stats + multi-threshold overlap)

## Limitations
- Associational/group-level diagnostic; no causal claim about intensity.
- `marking_iou` per bin is a within-bin IoU (points grouped by their own intensity),
  not a global IoU.
- Counts are patch-accumulated (see above), single training seed (42), single
  evaluation pass; matches the existing seed-42 full-coverage analysis.
"""
    path.write_text(txt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--split", default="test", choices=("test", "validation"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=2160)  # unused under full coverage
    ap.add_argument("--device", default="cuda", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--rgb-valid-threshold", type=float, default=0.5)
    args = ap.parse_args()

    eng.require_file(args.config)
    eng.require_file(args.checkpoint)

    print("=== G2 intensity-binned diagnostic (full-coverage test, seed 42) ===")
    data = run_inference(args)
    print(f"collected {data['I'].size:,} patch-accumulated active evaluations")

    main_df, rgb_df, dist_df, overlap_df, N = build_tables(data)
    report = validate(main_df, data, N)

    print("--- validation ---")
    for k in ("check1_pass", "check2_pass", "check3_pass"):
        print(f"  {k}: {report[k]}")
    print(f"  reconstructed marking IoU/prec/recall: "
          f"{report['recon_iou']:.4f} / {report['recon_precision']:.4f} / {report['recon_recall']:.4f}")

    if not report["all_pass"]:
        print("\nVALIDATION FAILED — writing NOTHING. Details:")
        if not report["check2_pass"]:
            print(f"  this_cm={report['this_cm']}")
            print(f"  ref_cm ={report['ref_cm']}")
        raise SystemExit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_df.to_csv(OUT_DIR / "intensity_bin_metrics.csv", index=False)
    rgb_df.to_csv(OUT_DIR / "intensity_bin_by_rgb_valid_metrics.csv", index=False)
    dist_df.to_csv(OUT_DIR / "intensity_bin_by_distance_metrics.csv", index=False)
    overlap_df.to_csv(OUT_DIR / "intensity_overlap_summary.csv", index=False)
    write_readme(OUT_DIR / "README.md", args, data, report, N)
    print(f"\nAll checks passed. Wrote 5 files to {OUT_DIR}")


if __name__ == "__main__":
    main()
