#!/usr/bin/env python
"""Focused D0 (LiDAR) vs G2 (LiDAR + RGB + Lovász) comparison for the thesis.

The thesis spine is D0 -> G2: geometry-only vs the best geometry+appearance model.
The D0->G2 gain is almost entirely PRECISION / CALIBRATION (D0 over-predicts
markings; G2 is calibrated), NOT recall. These figures/tables make that explicit.

CPU / file-only (no GPU, no pandaset). Reads the seed_42 full-coverage test pass
of each model from results/per_model/<folder>/test/full/seed_42/ and writes to
results/comparisons/d0_vs_g2/.

Usage:  python results/test_suite/compare_d0_g2.py
"""
from __future__ import annotations

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
    import _suite_style as S  # noqa: E402
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
PER_MODEL = REPO / "results" / "per_model"
OUT = REPO / "results" / "comparisons" / "d0_vs_g2"

D0_FOLDER = "D0_lidar"
G2_FOLDER = "G2_lidar_rgb_lovasz"


def _disp(code: str, fallback: str) -> str:
    """Thesis display name (matches the other plots); never the internal code."""
    if HAVE_STYLE and hasattr(S, "display_name"):
        try:
            return S.display_name(code)
        except Exception:
            return fallback
    return fallback


D0_NAME = _disp("D0", "LiDAR")
G2_NAME = _disp("G2", "LiDAR + RGB + Lovász")
D0_COLOR = getattr(S, "NEUTRAL", "#7f8c9a") if HAVE_STYLE else "#7f8c9a"
G2_COLOR = getattr(S, "CANDIDATE", "#2e7d32") if HAVE_STYLE else "#2e7d32"
CLASSES = ["road", "marking", "other"]


def primary(folder: str) -> Path:
    d = PER_MODEL / folder / "test" / "full" / "seed_42"
    if not (d / "confusion_matrix.npy").exists():
        raise SystemExit(f"missing full-coverage test pass: {d}/confusion_matrix.npy")
    return d


def metrics_from_cm(cm: np.ndarray) -> dict:
    cm = np.asarray(cm, dtype=np.float64)
    out: dict = {}
    for ci, name in ((0, "road"), (1, "marking"), (2, "other")):
        tp = cm[ci, ci]; fp = cm[:, ci].sum() - tp; fn = cm[ci, :].sum() - tp
        d = tp + fp + fn
        out[f"{name}_iou"] = float(tp / d) if d else 0.0
    tp, fp, fn = cm[1, 1], cm[0, 1] + cm[2, 1], cm[1, 0] + cm[1, 2]
    out["marking_iou"] = float(tp / (tp + fp + fn)) if (tp + fp + fn) else 0.0
    out["marking_precision"] = float(tp / (tp + fp)) if (tp + fp) else 0.0
    out["marking_recall"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
    p, r = out["marking_precision"], out["marking_recall"]
    out["marking_f1"] = float(2 * p * r / (p + r)) if (p + r) else 0.0
    out["miou"] = (out["road_iou"] + out["marking_iou"] + out["other_iou"]) / 3.0
    out["pred_true"] = float(cm[:, 1].sum() / cm[1, :].sum()) if cm[1, :].sum() else 0.0
    out["road_to_marking"] = float(cm[0, 1])
    out["other_to_marking"] = float(cm[2, 1])
    out["marking_to_road"] = float(cm[1, 0])
    out["marking_to_other"] = float(cm[1, 2])
    return out


def style(ax):
    if HAVE_STYLE and hasattr(S, "style_axes"):
        S.style_axes(ax)


def legend(ax, **kw):
    if HAVE_STYLE and hasattr(S, "legend_clear"):
        S.legend_clear(ax, **kw)
    else:
        ax.legend()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def fig_metrics(d0, g2):
    keys = [("marking_iou", "IoU"), ("marking_precision", "precision"),
            ("marking_recall", "recall"), ("marking_f1", "F1")]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.bar(x - w / 2, [d0[k] for k, _ in keys], w, label=D0_NAME, color=D0_COLOR)
    ax.bar(x + w / 2, [g2[k] for k, _ in keys], w, label=G2_NAME, color=G2_COLOR)
    for i, (k, _) in enumerate(keys):
        ax.text(i - w / 2, d0[k] + 0.012, f"{d0[k]:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, g2[k] + 0.012, f"{g2[k]:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in keys])
    ax.set_ylabel("score (test)"); ax.set_ylim(0, 1.0)
    ax.set_title("Marking metrics on the held-out test set")
    legend(ax, loc="upper right")
    style(ax)
    save(fig, "fig_metrics_bars.png")


def fig_overprediction(d0, g2):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    vals = [d0["pred_true"], g2["pred_true"]]
    ax.bar([0, 1], vals, 0.5, color=[D0_COLOR, G2_COLOR])
    ax.axhline(1.0, color="0.3", ls=":", lw=1.4, label="calibrated (1.0)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels([D0_NAME, G2_NAME], rotation=8)
    ax.set_ylabel("predicted / true marking ratio")
    ax.set_title("Marking over-prediction (>1 = paints too much marking)")
    legend(ax, loc="upper right")
    style(ax)
    save(fig, "fig_overprediction.png")


def fig_confusion(cm, name, fname):
    cm = np.asarray(cm, dtype=np.float64)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Row-normalized confusion — {name}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, fname)


def fig_error_breakdown(d0, g2):
    # absolute per-frame-set counts (false marking vs missed marking)
    groups = [("road_to_marking", "road→marking\n(false +)"),
              ("other_to_marking", "other→marking\n(false +)"),
              ("marking_to_road", "marking→road\n(missed)"),
              ("marking_to_other", "marking→other\n(missed)")]
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.bar(x - w / 2, [d0[k] / 1e6 for k, _ in groups], w, label=D0_NAME, color=D0_COLOR)
    ax.bar(x + w / 2, [g2[k] / 1e6 for k, _ in groups], w, label=G2_NAME, color=G2_COLOR)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in groups], fontsize=9)
    ax.set_ylabel("points (millions, full-coverage test)")
    ax.set_title("Error structure: false markings vs missed markings")
    legend(ax, loc="upper right")
    style(ax)
    save(fig, "fig_error_breakdown.png")


def per_sequence(d0_dir, g2_dir):
    a = pd.read_csv(d0_dir / "per_sequence_metrics.csv", dtype={"seq_id": str})
    b = pd.read_csv(g2_dir / "per_sequence_metrics.csv", dtype={"seq_id": str})
    cols = ["seq_id", "marking_iou", "marking_precision", "marking_recall", "true_marking"]
    m = a[cols].merge(b[cols], on="seq_id", suffixes=("_D0", "_G2"))
    m["seq_num"] = m["seq_id"].astype(int)
    m = m.sort_values("seq_num")
    m["seq"] = m["seq_id"].apply(lambda s: f"{int(s):03d}")
    m["d_iou_G2_minus_D0"] = (m["marking_iou_G2"] - m["marking_iou_D0"]).round(4)
    m.to_csv(OUT / "per_sequence_d0_vs_g2.csv", index=False)

    x = np.arange(len(m)); w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.bar(x - w / 2, m["marking_iou_D0"], w, label=D0_NAME, color=D0_COLOR)
    ax.bar(x + w / 2, m["marking_iou_G2"], w, label=G2_NAME, color=G2_COLOR)
    # highlight night 065
    for i, s in enumerate(m["seq_id"]):
        if s == "65":
            ax.axvspan(i - 0.5, i + 0.5, color="#fff3cd", zorder=0)
            ax.text(i, 0.02, "night", ha="center", fontsize=8, color="#8a6d00")
    ax.set_xticks(x); ax.set_xticklabels(m["seq"], rotation=0)
    ax.set_ylabel("marking IoU (test)"); ax.set_xlabel("test sequence")
    ax.set_title("Marking IoU per sequence  (065 = night)")
    ax.set_ylim(0, max(0.8, m[["marking_iou_D0", "marking_iou_G2"]].values.max() + 0.08))
    legend(ax, loc="upper left")
    style(ax)
    save(fig, "fig_per_sequence.png")
    return m


def distance(d0_dir, g2_dir):
    a = pd.read_csv(d0_dir / "distance_bucket_metrics.csv")
    b = pd.read_csv(g2_dir / "distance_bucket_metrics.csv")
    m = a[["bucket", "marking_iou"]].merge(b[["bucket", "marking_iou"]], on="bucket", suffixes=("_D0", "_G2"))
    m["d_iou_G2_minus_D0"] = (m["marking_iou_G2"] - m["marking_iou_D0"]).round(4)
    m.to_csv(OUT / "distance_d0_vs_g2.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    xr = range(len(m))
    ax.plot(xr, m["marking_iou_D0"], marker="o", label=D0_NAME, color=D0_COLOR)
    ax.plot(xr, m["marking_iou_G2"], marker="o", label=G2_NAME, color=G2_COLOR)
    ax.set_xticks(list(xr)); ax.set_xticklabels(m["bucket"])
    ax.set_ylabel("marking IoU (test)"); ax.set_xlabel("distance bucket")
    ax.set_title("Marking IoU by distance")
    legend(ax, loc="upper right")
    style(ax)
    save(fig, "fig_distance.png")
    return m


def write_table(d0, g2):
    rows = []
    for k in ["marking_iou", "marking_precision", "marking_recall", "marking_f1",
              "miou", "pred_true", "road_iou", "other_iou",
              "road_to_marking", "other_to_marking", "marking_to_road", "marking_to_other"]:
        rows.append({"metric": k, "D0": round(d0[k], 6), "G2": round(g2[k], 6),
                     "G2_minus_D0": round(g2[k] - d0[k], 6)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics_d0_vs_g2.csv", index=False)
    return df


def write_readme(tbl, seqdf, distdf):
    night = seqdf[seqdf["seq_id"] == "65"]
    lines = [
        "# D0 (LiDAR) vs G2 (LiDAR + RGB + Lovász) — focused comparison", "",
        "Full-coverage held-out **test** set (seed 42). D0→G2 is the thesis spine",
        "(geometry-only vs best geometry+appearance). NOTE: not a clean RGB ablation",
        "(G2 also adds Lovász loss, more features, more epochs) — system-level comparison.",
        "", "## Headline",
        "| metric | D0 | G2 | G2 − D0 |", "| --- | ---: | ---: | ---: |",
    ]
    show = {"marking_iou": "marking IoU", "marking_precision": "precision",
            "marking_recall": "recall", "marking_f1": "F1", "pred_true": "pred/true",
            "road_iou": "road IoU", "other_iou": "other IoU"}
    t = tbl.set_index("metric")
    for k, lab in show.items():
        lines.append(f"| {lab} | {t.loc[k,'D0']:.4f} | {t.loc[k,'G2']:.4f} | {t.loc[k,'G2_minus_D0']:+.4f} |")
    lines += [
        "", "**Read:** the gain is precision/calibration — G2 lifts precision "
        f"({t.loc['marking_precision','D0']:.3f}→{t.loc['marking_precision','G2']:.3f}) and "
        f"fixes over-prediction (pred/true {t.loc['pred_true','D0']:.2f}→{t.loc['pred_true','G2']:.2f}), "
        f"at a small recall cost ({t.loc['marking_recall','D0']:.3f}→{t.loc['marking_recall','G2']:.3f}).",
    ]
    if not night.empty:
        n = night.iloc[0]
        lines += ["", "## Night sequence 065",
                  f"- D0 IoU {n['marking_iou_D0']:.3f} → G2 IoU {n['marking_iou_G2']:.3f} "
                  f"(Δ {n['d_iou_G2_minus_D0']:+.3f}). RGB { 'still helps' if n['d_iou_G2_minus_D0']>0 else 'does not help' } at night here."]
    lines += ["", "## Files", "",
              "- `metrics_d0_vs_g2.csv` — full headline table.",
              "- `per_sequence_d0_vs_g2.csv` / `fig_per_sequence.png` — per scene (065 = night).",
              "- `distance_d0_vs_g2.csv` / `fig_distance.png` — by range.",
              "- `fig_metrics_bars.png`, `fig_overprediction.png`, `fig_error_breakdown.png`.",
              "- `fig_confusion_D0.png`, `fig_confusion_G2.png` — row-normalized confusion.", ""]
    (OUT / "README.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d0_dir, g2_dir = primary(D0_FOLDER), primary(G2_FOLDER)
    cm_d0 = np.load(d0_dir / "confusion_matrix.npy")
    cm_g2 = np.load(g2_dir / "confusion_matrix.npy")
    d0, g2 = metrics_from_cm(cm_d0), metrics_from_cm(cm_g2)

    tbl = write_table(d0, g2)
    fig_metrics(d0, g2)
    fig_overprediction(d0, g2)
    fig_confusion(cm_d0, D0_NAME, "fig_confusion_D0.png")
    fig_confusion(cm_g2, G2_NAME, "fig_confusion_G2.png")
    fig_error_breakdown(d0, g2)
    seqdf = per_sequence(d0_dir, g2_dir)
    distdf = distance(d0_dir, g2_dir)
    write_readme(tbl, seqdf, distdf)

    print(f"[compare_d0_g2] wrote -> {OUT.relative_to(REPO)}")
    for f in sorted(OUT.iterdir()):
        print("   ", f.name)
    print("compare_d0_g2_status PASS")


if __name__ == "__main__":
    main()
