#!/usr/bin/env python
"""Aggregate the per-model TEST outputs into the thesis results package.

Reads results/per_model/<folder>/test/seed_*/ (produced by run_test_all.py),
re-derives every metric from each seed's confusion_matrix.npy (single source of
truth), and writes:

  results/test_master_table.csv                  # headline: all models, val + test
  results/comparisons/rgb_effect__D0_vs_E0/      # clean "add RGB" effect
  results/comparisons/system__D0_vs_G2_vs_H0/    # LiDAR baseline vs full RGB system
  results/comparisons/shortcut__G2_vs_H0/        # brightness shortcut (rgb_valid gap + fingerprint)
  results/comparisons/plots/                     # cross-model figures (thesis-styled)
  results/README.md                              # provenance + the master table

CPU / file-only: no model inference, no GPU, no pandaset import -> runs anywhere.
For G2/H0 (multi-seed) metrics are mean +/- std over the available eval seeds.

Usage:  python results/test_suite/build_comparison.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _suite_style as S  # noqa: E402

S.apply()


def repo_root() -> Path:
    for p in [HERE, *HERE.parents]:
        if (p / "src" / "thesis_pipeline").is_dir() and (p / "logs").is_dir():
            return p
    raise RuntimeError("repo root not found")


REPO = repo_root()
PER_MODEL = REPO / "results" / "per_model"
COMP = REPO / "results" / "comparisons"
PLOTS = COMP / "plots"

# code, display label, output-folder, run-dir (rel), selected epoch, has_rgb
MODELS = [
    ("D0", "LiDAR",                         "D0_lidar",                   "logs/milestone_d/runs/D0_weighted_ce_25ep",    18, False),
    ("E0", "LiDAR + RGB",                   "E0_lidar_rgb",               "logs/milestone_e/runs/E0_rgb_front_v1",        14, True),
    ("F0", "LiDAR + RGB (calibrated)",      "F0_lidar_rgb_calibrated",    "logs/milestone_f/runs/F0_rgb_soft_weights",    13, True),
    ("G2", "LiDAR + RGB + Lovász",          "G2_lidar_rgb_lovasz",        "logs/milestone_g/runs/G2_schedule_extend_100", 68, True),
    ("H0", "LiDAR + RGB + Lovász + Jitter", "H0_lidar_rgb_lovasz_jitter", "logs/milestone_h/runs/H0_rgb_jitter",          37, True),
]
ORDER = [m[0] for m in MODELS]
MARK_KEYS = ["marking_iou", "marking_precision", "marking_recall", "marking_f1", "miou", "pred_true"]


def metrics_from_cm(cm: np.ndarray) -> dict:
    """All metrics re-derived from a 3x3 confusion matrix (rows=true, cols=pred,
    order road/marking/other). Single source of truth."""
    cm = np.asarray(cm, dtype=np.float64)
    out: dict = {}
    for ci, name in ((0, "road"), (1, "marking"), (2, "other")):
        tp = cm[ci, ci]
        fp = cm[:, ci].sum() - tp
        fn = cm[ci, :].sum() - tp
        denom = tp + fp + fn
        out[f"{name}_iou"] = float(tp / denom) if denom else 0.0
    tp, fp, fn = cm[1, 1], cm[0, 1] + cm[2, 1], cm[1, 0] + cm[1, 2]
    out["marking_iou"] = float(tp / (tp + fp + fn)) if (tp + fp + fn) else 0.0
    out["marking_precision"] = float(tp / (tp + fp)) if (tp + fp) else 0.0
    out["marking_recall"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
    p, r = out["marking_precision"], out["marking_recall"]
    out["marking_f1"] = float(2 * p * r / (p + r)) if (p + r) else 0.0
    out["miou"] = (out["road_iou"] + out["marking_iou"] + out["other_iou"]) / 3.0
    out["pred_true"] = float(cm[:, 1].sum() / cm[1, :].sum()) if cm[1, :].sum() else 0.0
    out["road_to_marking"] = int(cm[0, 1])
    out["marking_to_road"] = int(cm[1, 0])
    return out


def seed_dirs(folder: str) -> list[Path]:
    base = PER_MODEL / folder / "test"
    if not base.is_dir():
        return []
    return sorted(d for d in base.glob("seed_*") if (d / "confusion_matrix.npy").exists())


def aggregate_test(folder: str) -> dict | None:
    """Mean (+/- std) of every metric over the available eval seeds, from the npy."""
    dirs = seed_dirs(folder)
    if not dirs:
        return None
    per = [metrics_from_cm(np.load(d / "confusion_matrix.npy")) for d in dirs]
    agg: dict = {"n_seeds": len(dirs), "seed_dirs": [str(d.relative_to(REPO)) for d in dirs]}
    keys = [k for k in per[0] if isinstance(per[0][k], float)]
    for k in keys:
        vals = np.array([m[k] for m in per], dtype=np.float64)
        agg[k] = float(vals.mean())
        agg[k + "_std"] = float(vals.std(ddof=0))
    for k in ("road_to_marking", "marking_to_road"):
        agg[k] = int(np.mean([m[k] for m in per]))
    return agg


def val_metrics(run_dir_rel: str, epoch: int) -> dict:
    """Canonical validation numbers from the run's per-epoch eval_history at the
    selected epoch (best marking-IoU epoch chosen during development)."""
    p = REPO / run_dir_rel / "eval_history.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    row = df.loc[df["epoch"] == epoch]
    if row.empty:
        return {}
    row = row.iloc[0]

    def g(*names):
        for n in names:
            if n in row:
                return float(row[n])
        return float("nan")

    return {
        "marking_iou": g("lane_iou", "marking_iou"),
        "marking_precision": g("lane_precision", "marking_precision"),
        "marking_recall": g("lane_recall", "marking_recall"),
        "marking_f1": g("lane_f1", "marking_f1"),
        "miou": g("miou"),
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def primary_dir(folder: str) -> Path | None:
    """seed_42 if present, else the first available seed dir."""
    dirs = seed_dirs(folder)
    if not dirs:
        return None
    for d in dirs:
        if d.name == "seed_42":
            return d
    return dirs[0]


def fmt(v, std=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.4f}" + (f" ± {std:.4f}" if std else "")


# --------------------------------------------------------------------------- #

def build_master(data: dict) -> Path:
    rows = []
    for code, label, folder, rd, ep, has_rgb in MODELS:
        t = data[code]["test"]
        v = data[code]["val"]
        if t is None:
            continue
        rows.append({
            "code": code, "model": label, "selected_epoch": ep, "n_test_seeds": t["n_seeds"],
            "val_marking_iou": round(v.get("marking_iou", float("nan")), 6),
            "test_marking_iou": round(t["marking_iou"], 6),
            "test_marking_iou_std": round(t["marking_iou_std"], 6),
            "val_minus_test_iou": round(v.get("marking_iou", float("nan")) - t["marking_iou"], 6),
            "test_precision": round(t["marking_precision"], 6),
            "test_recall": round(t["marking_recall"], 6),
            "test_f1": round(t["marking_f1"], 6),
            "test_miou": round(t["miou"], 6),
            "test_road_iou": round(t["road_iou"], 6),
            "test_other_iou": round(t["other_iou"], 6),
            "test_pred_true": round(t["pred_true"], 4),
            "test_road_to_marking": t["road_to_marking"],
        })
    out = REPO / "results" / "test_master_table.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def write_comparison(name: str, codes: list[str], data: dict) -> None:
    d = COMP / name
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for code in codes:
        t = data[code]["test"]
        if t is None:
            continue
        rows.append({"code": code, "model": dict((m[0], m[1]) for m in MODELS)[code],
                     **{k: round(t[k], 6) for k in MARK_KEYS},
                     "road_iou": round(t["road_iou"], 6), "other_iou": round(t["other_iou"], 6),
                     "road_to_marking": t["road_to_marking"]})
    pd.DataFrame(rows).to_csv(d / "test_metrics.csv", index=False)
    # deltas vs first listed (the baseline)
    if len(rows) >= 2:
        base = rows[0]
        deltas = []
        for r in rows[1:]:
            deltas.append({"vs_baseline": base["code"], "model": r["model"],
                           **{f"d_{k}": round(r[k] - base[k], 6) for k in MARK_KEYS if isinstance(r[k], float)}})
        pd.DataFrame(deltas).to_csv(d / "deltas_vs_baseline.csv", index=False)


def write_shortcut(data: dict) -> None:
    d = COMP / "shortcut__G2_vs_H0"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for code in ("G2", "H0"):
        pj = load_json(primary_dir(dict((m[0], m[2]) for m in MODELS)[code]) / "summary.json") \
            if primary_dir(dict((m[0], m[2]) for m in MODELS)[code]) else {}
        rv = pj.get("rgb_valid_metrics", {})
        ri = pj.get("rgb_invalid_metrics", {})
        rows.append({
            "model": code,
            "rgb_valid_precision": rv.get("marking_precision"),
            "rgb_valid_pred_true": rv.get("predicted_true_marking_ratio"),
            "rgb_invalid_precision": ri.get("marking_precision"),
            "rgb_invalid_pred_true": ri.get("predicted_true_marking_ratio"),
            "overprediction_gap": (rv.get("predicted_true_marking_ratio") - ri.get("predicted_true_marking_ratio"))
            if (rv and ri) else None,
            "road_to_marking_total": data[code]["test"]["road_to_marking"] if data[code]["test"] else None,
        })
    pd.DataFrame(rows).to_csv(d / "rgb_valid_gap.csv", index=False)
    # brightness fingerprint of the road->marking FP, from group_feature_summary
    fp_rows = []
    for code in ("G2", "H0"):
        pd_ = primary_dir(dict((m[0], m[2]) for m in MODELS)[code])
        gf = pd_ / "group_feature_summary.csv" if pd_ else None
        if gf and gf.exists():
            g = pd.read_csv(gf).set_index("group")
            for grp in ("road_tp", "road_to_marking", "marking_tp"):
                if grp in g.index:
                    fp_rows.append({"model": code, "group": grp,
                                    "red_mean": g.loc[grp].get("red_mean", None),
                                    "green_mean": g.loc[grp].get("green_mean", None),
                                    "blue_mean": g.loc[grp].get("blue_mean", None),
                                    "intensity_mean": g.loc[grp].get("intensity_mean", None)})
    if fp_rows:
        df = pd.DataFrame(fp_rows)
        df["brightness_mean"] = df[["red_mean", "green_mean", "blue_mean"]].mean(axis=1)
        df.to_csv(d / "brightness_fingerprint.csv", index=False)


# ------------------------------- plots ------------------------------------- #

def _present(data):
    return [(c, lab, f) for c, lab, f, *_ in MODELS if data[c]["test"] is not None]


def plot_progression(data):
    import matplotlib.pyplot as plt
    pres = _present(data)
    labels = [S.display_name(c) for c, _l, _f in pres]
    vals = [data[c]["test"]["marking_iou"] for c, _l, _f in pres]
    errs = [data[c]["test"]["marking_iou_std"] for c, _l, _f in pres]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    bars = ax.bar(range(len(vals)), vals, yerr=errs, capsize=4,
                  color=[S.CANDIDATE if c in ("G2", "H0") else S.NEUTRAL for c, _l, _f in pres])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("marking IoU (test)")
    ax.set_title("Marking IoU on the held-out test set")
    S.headroom(ax, 0.14)
    S.style_axes(ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "progression_marking_iou_test.png")
    plt.close(fig)


def plot_val_vs_test(data):
    import matplotlib.pyplot as plt
    pres = _present(data)
    labels = [S.display_name(c) for c, _l, _f in pres]
    val = [data[c]["val"].get("marking_iou", np.nan) for c, _l, _f in pres]
    test = [data[c]["test"]["marking_iou"] for c, _l, _f in pres]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.bar(x - w / 2, val, w, label="validation (development)", color=S.NEUTRAL)
    ax.bar(x + w / 2, test, w, label="test (held-out)", color=S.CANDIDATE)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("marking IoU")
    ax.set_title("Validation vs test marking IoU (generalization gap)")
    S.headroom(ax, 0.16)
    S.legend_clear(ax, loc="upper left")
    S.style_axes(ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "val_vs_test_marking_iou.png")
    plt.close(fig)


def plot_pr_f1(data):
    import matplotlib.pyplot as plt
    pres = _present(data)
    labels = [S.display_name(c) for c, _l, _f in pres]
    x = np.arange(len(labels)); w = 0.2
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for i, (key, lab, col) in enumerate([
        ("marking_iou", "IoU", S.METRIC["iou"]), ("marking_precision", "precision", S.METRIC["precision"]),
        ("marking_recall", "recall", S.METRIC["recall"]), ("marking_f1", "F1", S.METRIC["f1"]),
    ]):
        ax.bar(x + (i - 1.5) * w, [data[c]["test"][key] for c, _l, _f in pres], w, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("score (0–1)"); ax.set_ylim(0, 1.0)
    ax.set_title("Marking metrics on the test set")
    S.legend_clear(ax, loc="upper left", ncol=4)
    S.style_axes(ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "marking_metrics_test.png")
    plt.close(fig)


def plot_per_class(data):
    import matplotlib.pyplot as plt
    pres = _present(data)
    labels = [S.display_name(c) for c, _l, _f in pres]
    x = np.arange(len(labels)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for i, (key, lab, col) in enumerate([("road_iou", "road", S.ROAD), ("marking_iou", "marking", S.MARKING), ("other_iou", "other", S.OTHER)]):
        ax.bar(x + (i - 1) * w, [data[c]["test"][key] for c, _l, _f in pres], w, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("IoU (test)"); ax.set_ylim(0, 1.0)
    ax.set_title("Per-class IoU on the test set")
    S.legend_clear(ax, loc="lower left", ncol=3)
    S.style_axes(ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "per_class_iou_test.png")
    plt.close(fig)


def plot_shortcut(data):
    import matplotlib.pyplot as plt
    gf = COMP / "shortcut__G2_vs_H0" / "rgb_valid_gap.csv"
    if not gf.exists():
        return
    df = pd.read_csv(gf)
    if df.empty:
        return
    x = np.arange(len(df)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.bar(x - w / 2, df["rgb_valid_pred_true"], w, label="rgb_valid", color=S.RGB["rgb_valid"])
    ax.bar(x + w / 2, df["rgb_invalid_pred_true"], w, label="rgb_invalid", color=S.RGB["rgb_invalid"])
    ax.axhline(1.0, color=S.CALIBRATED, ls=":", lw=1.3, label="calibrated (1.0)")
    ax.set_xticks(x); ax.set_xticklabels([S.display_name(c) for c in df["model"]], rotation=10, ha="right")
    ax.set_ylabel("predicted / true marking ratio (test)")
    ax.set_title("Brightness shortcut: over-prediction by RGB validity")
    S.headroom(ax, 0.16)
    S.legend_clear(ax, loc="upper right")
    S.style_axes(ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "shortcut_overprediction_g2_vs_h0.png")
    plt.close(fig)


def plot_distance(data):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.6))
    drew = False
    for code in ("D0", "G2", "H0"):
        folder = dict((m[0], m[2]) for m in MODELS)[code]
        pdir = primary_dir(folder)
        f = pdir / "distance_bucket_metrics.csv" if pdir else None
        if f and f.exists():
            df = pd.read_csv(f)
            ax.plot(range(len(df)), df["marking_iou"], marker="o", label=S.display_name(code),
                    color=S.run_colors("G2", ["D0", "H0"]).get(code, None))
            ax.set_xticks(range(len(df))); ax.set_xticklabels(df["bucket"], rotation=0)
            drew = True
    if not drew:
        plt.close(fig); return
    ax.set_ylabel("marking IoU (test)"); ax.set_xlabel("distance bucket")
    ax.set_title("Marking IoU by distance (test)")
    S.legend_clear(ax, loc="upper right")
    S.style_axes(ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "distance_marking_iou_test.png")
    plt.close(fig)


def write_readme(master: Path) -> None:
    df = pd.read_csv(master)
    lines = [
        "# Test-set results", "",
        "Final held-out **test** evaluation. Generated by `results/test_suite/build_comparison.py`",
        "from the per-model passes in `results/per_model/<model>/test/seed_*/` (run by",
        "`run_test_all.py`). Every number is re-derived from each seed's `confusion_matrix.npy`.",
        "",
        "- **Sampled** evaluation (random patches, 2160 steps) — same protocol as the validation",
        "  diagnostics; not exhaustive per-point.",
        "- **Single training seed (42)**; G2/H0 test numbers are mean ± std over 3 *evaluation*",
        "  seeds (sampling variance only, NOT training variance). ~0.008 marking-IoU noise floor:",
        "  treat smaller differences as ties.",
        "- Selection (epoch, model) was done on validation; test is reported once, not tuned on.",
        "- Labels = `road_marking3` (marking = raw 8+9+10); `lane_*` == `marking_*`.",
        "", "## Master table", "",
        "| model | sel.ep | val IoU | test IoU | val−test | test P | test R | test F1 | test mIoU | pred/true |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model']} | {int(r['selected_epoch'])} | {r['val_marking_iou']:.4f} | "
            f"{r['test_marking_iou']:.4f} | {r['val_minus_test_iou']:+.4f} | {r['test_precision']:.4f} | "
            f"{r['test_recall']:.4f} | {r['test_f1']:.4f} | {r['test_miou']:.4f} | {r['test_pred_true']:.3f} |")
    lines += ["", "## Comparisons", "",
              "- `comparisons/rgb_effect__D0_vs_E0/` — clean 'add RGB' effect (LiDAR → +RGB).",
              "- `comparisons/system__D0_vs_G2_vs_H0/` — LiDAR baseline vs the full RGB systems.",
              "- `comparisons/shortcut__G2_vs_H0/` — brightness shortcut (rgb_valid over-prediction gap + FP fingerprint).",
              "- `comparisons/plots/` — thesis-styled figures.",
              "", "> Note: validation = per-epoch development metric (720-step); test = fresh 2160-step",
              "> sampled pass. Both sampled; the val−test column is the generalization gap.", ""]
    (REPO / "results" / "README.md").write_text("\n".join(lines))


def main() -> None:
    if not PER_MODEL.is_dir():
        raise SystemExit(f"No per-model outputs at {PER_MODEL}. Run run_test_all.py first.")
    PLOTS.mkdir(parents=True, exist_ok=True)

    data: dict = {}
    missing = []
    for code, _label, folder, rd, ep, _has in MODELS:
        t = aggregate_test(folder)
        if t is None:
            missing.append(code)
        data[code] = {"test": t, "val": val_metrics(rd, ep)}
    present = [c for c in ORDER if data[c]["test"] is not None]
    if missing:
        print(f"[warn] no test outputs yet for: {missing} (skipping those rows)")
    if not present:
        raise SystemExit("No model has test outputs yet — nothing to aggregate.")
    print(f"[build_comparison] aggregating: {present}")

    master = build_master(data)
    write_comparison("rgb_effect__D0_vs_E0", [c for c in ("D0", "E0") if data[c]["test"]], data)
    write_comparison("system__D0_vs_G2_vs_H0", [c for c in ("D0", "G2", "H0") if data[c]["test"]], data)
    if data["G2"]["test"] and data["H0"]["test"]:
        write_shortcut(data)

    plot_progression(data)
    plot_val_vs_test(data)
    plot_pr_f1(data)
    plot_per_class(data)
    plot_distance(data)
    if data["G2"]["test"] and data["H0"]["test"]:
        plot_shortcut(data)

    write_readme(master)
    print(f"wrote {master.relative_to(REPO)}")
    print(f"wrote {(REPO / 'results' / 'README.md').relative_to(REPO)}")
    print(f"wrote plots -> {PLOTS.relative_to(REPO)}")
    print("build_comparison_status PASS")


if __name__ == "__main__":
    main()
