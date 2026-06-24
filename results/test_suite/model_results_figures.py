#!/usr/bin/env python
"""Standalone final-model results figures for the thesis (default G2).

For ONE model (the chosen final model), from its full-coverage test pass, makes a
self-contained set of results figures + a summary table -> results/<model>_final/.
CPU / file-only (reads the CSVs + confusion_matrix.npy already on disk).

Figures (each answers one thesis question):
  fig_per_class_iou       - road/marking/other IoU (which classes are hard)
  fig_marking_metrics     - marking IoU/precision/recall/F1
  fig_confusion           - row-normalized 3x3 confusion (error structure)
  fig_by_distance         - marking IoU + precision + recall vs range bucket
  fig_by_sequence         - marking IoU per test sequence (065 night highlighted)
  fig_by_subtype          - recall per marking subtype (lane / stop / other)
  fig_by_rgb_validity     - marking IoU + recall in camera-visible vs not (RGB models)
  fig_frame_iou_hist      - distribution of per-frame marking IoU (consistency)

Usage:
  python results/test_suite/model_results_figures.py            # G2
  python results/test_suite/model_results_figures.py --model D0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import _suite_style as S
    S.apply()
    HAVE_STYLE = True
except Exception:
    HAVE_STYLE = False


def repo_root() -> Path:
    for p in [HERE, *HERE.parents]:
        if (p / "src" / "thesis_pipeline").is_dir() and (p / "logs").is_dir():
            return p
    raise RuntimeError("repo root not found")


REPO = repo_root()
FOLDERS = {"D0": "D0_lidar", "E0": "E0_lidar_rgb", "F0": "F0_lidar_rgb_calibrated",
           "G2": "G2_lidar_rgb_lovasz", "H0": "H0_lidar_rgb_lovasz_jitter"}
MARK = getattr(S, "MARKING", "#d62728") if HAVE_STYLE else "#d62728"
ROAD = getattr(S, "ROAD", "#7f8c9a") if HAVE_STYLE else "#7f8c9a"
OTHER = getattr(S, "OTHER", "#b0b0b0") if HAVE_STYLE else "#b0b0b0"
ACC = getattr(S, "CANDIDATE", "#2e7d32") if HAVE_STYLE else "#2e7d32"


def style(ax):
    if HAVE_STYLE and hasattr(S, "style_axes"):
        S.style_axes(ax)


def legend(ax, **kw):
    (S.legend_clear(ax, **kw) if HAVE_STYLE and hasattr(S, "legend_clear") else ax.legend())


def save(fig, out, name):
    fig.tight_layout(); fig.savefig(out / name, dpi=140); plt.close(fig)


def metrics_from_cm(cm):
    cm = np.asarray(cm, float); o = {}
    for ci, n in ((0, "road"), (1, "marking"), (2, "other")):
        tp = cm[ci, ci]; fp = cm[:, ci].sum() - tp; fn = cm[ci, :].sum() - tp
        o[f"{n}_iou"] = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G2", choices=list(FOLDERS))
    args = ap.parse_args()
    folder = FOLDERS[args.model]
    d = REPO / "results" / "per_model" / folder / "test" / "full" / "seed_42"
    if not (d / "confusion_matrix.npy").exists():
        raise SystemExit(f"missing: {d}")
    out = REPO / "results" / f"{args.model}_final"
    out.mkdir(parents=True, exist_ok=True)
    name = (S.display_name(args.model) if HAVE_STYLE and hasattr(S, "display_name") else args.model)

    cm = np.load(d / "confusion_matrix.npy")
    cls = metrics_from_cm(cm)

    # 1) per-class IoU
    fig, ax = plt.subplots(figsize=(6.4, 5))
    vals = [cls["road_iou"], cls["marking_iou"], cls["other_iou"]]
    ax.bar(["road", "marking", "other"], vals, color=[ROAD, MARK, OTHER])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center")
    ax.set_ylim(0, 1); ax.set_ylabel("IoU (test)"); ax.set_title(f"Per-class IoU — {name}")
    style(ax); save(fig, out, "fig_per_class_iou.png")

    # 2) marking metrics
    sm = pd.read_csv(d / "rgb_valid_stratified_metrics.csv") if (d / "rgb_valid_stratified_metrics.csv").exists() else None
    row = (sm[sm["stratum"] == "all"].iloc[0] if sm is not None else None)
    if row is not None:
        ks = [("marking_iou", "IoU"), ("marking_precision", "precision"),
              ("marking_recall", "recall"), ("marking_f1", "F1")]
        fig, ax = plt.subplots(figsize=(6.8, 5))
        vv = [float(row[k]) for k, _ in ks]
        ax.bar([l for _, l in ks], vv, color=ACC)
        for i, v in enumerate(vv):
            ax.text(i, v + 0.012, f"{v:.3f}", ha="center")
        ax.set_ylim(0, 1); ax.set_ylabel("score (test)"); ax.set_title(f"Marking metrics — {name}")
        style(ax); save(fig, out, "fig_marking_metrics.png")

    # 3) confusion matrix (row-normalized)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(["road", "marking", "other"])
    ax.set_yticks(range(3)); ax.set_yticklabels(["road", "marking", "other"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Confusion (row-normalized) — {name}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); save(fig, out, "fig_confusion.png")

    # 4) by distance: IoU + precision + recall
    db = pd.read_csv(d / "distance_bucket_metrics.csv")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = range(len(db))
    ax.plot(x, db["marking_iou"], marker="o", label="IoU", color=MARK)
    ax.plot(x, db["marking_precision"], marker="s", label="precision", color=ACC)
    ax.plot(x, db["marking_recall"], marker="^", label="recall", color="#1f77b4")
    ax.set_xticks(list(x)); ax.set_xticklabels(db["bucket"]); ax.set_ylim(0, 1)
    ax.set_ylabel("score (test)"); ax.set_xlabel("distance bucket")
    ax.set_title(f"Marking performance by distance — {name}")
    legend(ax, loc="lower left"); style(ax); save(fig, out, "fig_by_distance.png")

    # 5) by sequence (065 night highlighted)
    ps = pd.read_csv(d / "per_sequence_metrics.csv", dtype={"seq_id": str})
    ps["sn"] = ps["seq_id"].astype(int); ps = ps.sort_values("sn")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    xs = range(len(ps))
    bars = ax.bar(xs, ps["marking_iou"], color=ACC)
    for i, s in enumerate(ps["seq_id"]):
        if int(s) == 65:
            bars[i].set_color("#8a6d00"); ax.text(i, ps["marking_iou"].iloc[i] + 0.01, "night", ha="center", fontsize=8)
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"{int(s):03d}" for s in ps["seq_id"]])
    ax.set_ylabel("marking IoU (test)"); ax.set_xlabel("test sequence")
    ax.set_title(f"Marking IoU per sequence — {name}  (065 = night)")
    style(ax); save(fig, out, "fig_by_sequence.png")

    # 6) by subtype: recall (subtype stratum = recall; precision trivially 1.0)
    st = pd.read_csv(d / "raw_subtype_rgb_stratified_metrics.csv")
    allr = st[st["stratum"] == "all"]
    fig, ax = plt.subplots(figsize=(7.2, 5))
    labels = allr["raw_name"].str.replace("_marking", "").str.replace("_", " ")
    ax.bar(range(len(allr)), allr["marking_recall"], color=MARK)
    for i, v in enumerate(allr["marking_recall"]):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center")
    ax.set_xticks(range(len(allr))); ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1); ax.set_ylabel("recall (test)")
    ax.set_title(f"Recall by marking subtype — {name}")
    style(ax); save(fig, out, "fig_by_subtype.png")

    # 7) by rgb-validity (RGB models only)
    if sm is not None and {"rgb_valid", "rgb_invalid"}.issubset(set(sm["stratum"])):
        sub = sm.set_index("stratum")
        cats = ["rgb_valid", "rgb_invalid"]
        x = np.arange(len(cats)); w = 0.38
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x - w / 2, [sub.loc[c, "marking_iou"] for c in cats], w, label="IoU", color=MARK)
        ax.bar(x + w / 2, [sub.loc[c, "marking_recall"] for c in cats], w, label="recall", color=ACC)
        ax.set_xticks(x); ax.set_xticklabels(["camera-visible\n(rgb valid)", "not visible\n(rgb invalid)"])
        ax.set_ylim(0, 1); ax.set_ylabel("score (test)")
        ax.set_title(f"Performance where the camera sees vs not — {name}")
        legend(ax, loc="upper right"); style(ax); save(fig, out, "fig_by_rgb_validity.png")

    # 8) per-frame marking IoU distribution (aggregate patches -> frames)
    fe = pd.read_csv(d / "frame_error_summary.csv", dtype={"seq_id": str})
    g = fe.groupby(["seq_id", "frame_idx"]).agg(tp=("marking_tp", "sum"), fp=("marking_fp", "sum"),
                                                fn=("marking_fn", "sum"), tm=("true_marking", "sum")).reset_index()
    g = g[g["tm"] >= 2000]
    g["iou"] = g["tp"] / (g["tp"] + g["fp"] + g["fn"]).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(g["iou"].dropna(), bins=20, color=ACC, edgecolor="white")
    ax.axvline(g["iou"].median(), color=MARK, ls="--", label=f"median {g['iou'].median():.3f}")
    ax.set_xlabel("per-frame marking IoU (frames with marking)"); ax.set_ylabel("frames")
    ax.set_title(f"Per-frame marking IoU distribution — {name}  (n={len(g)})")
    legend(ax, loc="upper left"); style(ax); save(fig, out, "fig_frame_iou_hist.png")

    print(f"[model_results_figures] {args.model} -> {out.relative_to(REPO)}")
    for f in sorted(out.glob("*.png")):
        print("   ", f.name)
    print("model_results_figures_status PASS")


if __name__ == "__main__":
    main()
