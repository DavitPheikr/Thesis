#!/usr/bin/env python
"""Standalone final-model results figures for the thesis (default G2).

For ONE model (the chosen final model), from its full-coverage test pass, makes a
self-contained set of results figures + a summary table -> results/<model>_final/.
CPU / file-only (reads the CSVs + confusion_matrix.npy already on disk).

Colour rule (via _suite_style): blue=road, red=marking, green=other for class
figures; a separate palette (red/orange/purple/gold) for the marking IoU/precision/
recall/F1 family so it never collides with class colours.

Figures (each answers one thesis question):
  fig_per_class_iou    - road/marking/other IoU
  fig_marking_metrics  - marking IoU/precision/recall/F1
  fig_confusion        - 3x3 confusion with ACTUAL COUNTS (+ row %)
  fig_by_distance      - marking IoU/precision/recall vs range (rescaled; share/bucket)
  fig_by_sequence      - marking IoU per sequence (065 night marked)
  fig_by_subtype       - recall per marking subtype (lane / stop / other)
  fig_by_rgb_validity  - marking IoU + recall where camera sees vs not (RGB models)
  fig_frame_iou_hist   - per-frame marking IoU distribution (consistency)

Usage:  python results/test_suite/model_results_figures.py [--model G2]
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
ROAD = getattr(S, "ROAD", "#1f77b4") if HAVE_STYLE else "#1f77b4"
MARK = getattr(S, "MARKING", "#d62728") if HAVE_STYLE else "#d62728"
OTHER = getattr(S, "OTHER", "#2ca02c") if HAVE_STYLE else "#2ca02c"
MET = getattr(S, "METRIC", None) if HAVE_STYLE else None
if not MET:
    MET = {"iou": MARK, "precision": "#ff7f0e", "recall": "#9467bd", "f1": "#e6ab02"}


def style(ax):
    if HAVE_STYLE and hasattr(S, "style_axes"):
        S.style_axes(ax)


def legend(ax, **kw):
    (S.legend_clear(ax, **kw) if HAVE_STYLE and hasattr(S, "legend_clear") else ax.legend())


def save(fig, out, name):
    fig.tight_layout(); fig.savefig(out / name, dpi=140); plt.close(fig)


def fmt(n) -> str:
    n = int(round(float(n)))
    if abs(n) >= 1_000_000:
        return f"{n/1e6:.2f}M"
    if abs(n) >= 10_000:
        return f"{n/1e3:.0f}k"
    return f"{n:,}"


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
    CLS = ["road", "marking", "other"]

    cm = np.load(d / "confusion_matrix.npy")
    cls = metrics_from_cm(cm)

    # 1) per-class IoU  (blue / red / green)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    vals = [cls["road_iou"], cls["marking_iou"], cls["other_iou"]]
    ax.bar(CLS, vals, color=[ROAD, MARK, OTHER])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center")
    ax.set_ylim(0, 1); ax.set_ylabel("IoU (test)"); ax.set_title(f"Per-class IoU — {name}")
    style(ax); save(fig, out, "fig_per_class_iou.png")

    # 2) marking metrics  (metric palette)
    sm = pd.read_csv(d / "rgb_valid_stratified_metrics.csv") if (d / "rgb_valid_stratified_metrics.csv").exists() else None
    row = (sm[sm["stratum"] == "all"].iloc[0] if sm is not None else None)
    ks = [("marking_iou", "IoU", MET["iou"]), ("marking_precision", "precision", MET["precision"]),
          ("marking_recall", "recall", MET["recall"]), ("marking_f1", "F1", MET["f1"])]
    if row is not None:
        fig, ax = plt.subplots(figsize=(6.8, 5))
        vv = [float(row[k]) for k, _l, _c in ks]
        ax.bar([l for _k, l, _c in ks], vv, color=[c for _k, _l, c in ks])
        for i, v in enumerate(vv):
            ax.text(i, v + 0.012, f"{v:.3f}", ha="center")
        ax.set_ylim(0, 1); ax.set_ylabel("score (test)"); ax.set_title(f"Marking metrics — {name}")
        style(ax); save(fig, out, "fig_marking_metrics.png")

    # 3) confusion matrix with ACTUAL COUNTS (+ row %); colour = row-normalized
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLS)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (counts; cell %% of true row) — {name}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{fmt(cm[i, j])}\n{cmn[i, j]*100:.1f}%", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    save(fig, out, "fig_confusion.png")

    # 4) by distance: IoU + precision + recall, rescaled, with marking share/bucket
    db = pd.read_csv(d / "distance_bucket_metrics.csv")
    share = (db["true_marking"] / db["true_marking"].sum() * 100).round(0).astype(int)
    xt = [f"{b}\n({s}% of mks)" for b, s in zip(db["bucket"], share)]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = range(len(db))
    ax.plot(x, db["marking_iou"], marker="o", label="IoU", color=MET["iou"], lw=2)
    ax.plot(x, db["marking_precision"], marker="s", label="precision", color=MET["precision"])
    ax.plot(x, db["marking_recall"], marker="^", label="recall", color=MET["recall"])
    lo = float(min(db[["marking_iou", "marking_precision", "marking_recall"]].min()))
    hi = float(max(db[["marking_iou", "marking_precision", "marking_recall"]].max()))
    ax.set_ylim(max(0, lo - 0.08), min(1, hi + 0.08))
    ax.set_xticks(list(x)); ax.set_xticklabels(xt, fontsize=9)
    ax.set_ylabel("score (test)"); ax.set_xlabel("distance bucket")
    ax.set_title(f"Marking performance by distance — {name}")
    legend(ax, loc="lower left"); style(ax); save(fig, out, "fig_by_distance.png")

    # 5) by sequence (marking red; 065 night marked)
    ps = pd.read_csv(d / "per_sequence_metrics.csv", dtype={"seq_id": str})
    ps["sn"] = ps["seq_id"].astype(int); ps = ps.sort_values("sn")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    xs = range(len(ps))
    bars = ax.bar(xs, ps["marking_iou"], color=MARK)
    for i, s in enumerate(ps["seq_id"]):
        if int(s) == 65:
            bars[i].set_edgecolor("black"); bars[i].set_linewidth(2)
            ax.text(i, ps["marking_iou"].iloc[i] + 0.012, "night", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"{int(s):03d}" for s in ps["seq_id"]])
    ax.set_ylim(0, float(ps["marking_iou"].max()) + 0.10)
    ax.set_ylabel("marking IoU (test)"); ax.set_xlabel("test sequence")
    ax.set_title(f"Marking IoU per sequence — {name}  (065 = night)")
    style(ax); save(fig, out, "fig_by_sequence.png")

    # 6) by subtype: recall (subtype stratum -> recall; precision trivially 1)
    st = pd.read_csv(d / "raw_subtype_rgb_stratified_metrics.csv")
    allr = st[st["stratum"] == "all"]
    fig, ax = plt.subplots(figsize=(7.2, 5))
    labels = allr["raw_name"].str.replace("_marking", "").str.replace("_", " ")
    ax.bar(range(len(allr)), allr["marking_recall"], color=MET["recall"])  # recall = purple
    for i, v in enumerate(allr["marking_recall"]):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center")
    ax.set_xticks(range(len(allr))); ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1); ax.set_ylabel("recall (test)")
    ax.set_title(f"Recall by marking subtype — {name}")
    style(ax); save(fig, out, "fig_by_subtype.png")

    # 7) by rgb-validity (RGB models only): IoU + recall
    if sm is not None and {"rgb_valid", "rgb_invalid"}.issubset(set(sm["stratum"])):
        sub = sm.set_index("stratum"); cats = ["rgb_valid", "rgb_invalid"]
        x = np.arange(len(cats)); w = 0.38
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x - w / 2, [sub.loc[c, "marking_iou"] for c in cats], w, label="IoU", color=MET["iou"])
        ax.bar(x + w / 2, [sub.loc[c, "marking_recall"] for c in cats], w, label="recall", color=MET["recall"])
        ax.set_xticks(x); ax.set_xticklabels(["camera-visible\n(rgb valid)", "not visible\n(rgb invalid)"])
        ax.set_ylim(0, 1); ax.set_ylabel("score (test)")
        ax.set_title(f"Performance where the camera sees vs not — {name}")
        legend(ax, loc="upper right"); style(ax); save(fig, out, "fig_by_rgb_validity.png")

    # 8) per-frame marking IoU distribution
    fe = pd.read_csv(d / "frame_error_summary.csv", dtype={"seq_id": str})
    g = fe.groupby(["seq_id", "frame_idx"]).agg(tp=("marking_tp", "sum"), fp=("marking_fp", "sum"),
                                                fn=("marking_fn", "sum"), tm=("true_marking", "sum")).reset_index()
    g = g[g["tm"] >= 2000]
    g["iou"] = g["tp"] / (g["tp"] + g["fp"] + g["fn"]).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(g["iou"].dropna(), bins=20, color=MARK, edgecolor="white")
    ax.axvline(g["iou"].median(), color="black", ls="--", lw=1.6, label=f"median {g['iou'].median():.3f}")
    ax.set_xlabel("per-frame marking IoU (frames with marking)"); ax.set_ylabel("number of frames")
    ax.set_title(f"Per-frame marking IoU distribution — {name}  (n={len(g)} frames)")
    legend(ax, loc="upper left"); style(ax); save(fig, out, "fig_frame_iou_hist.png")

    print(f"[model_results_figures] {args.model} -> {out.relative_to(REPO)}")
    for f in sorted(out.glob("*.png")):
        print("   ", f.name)
    print("model_results_figures_status PASS")


if __name__ == "__main__":
    main()
