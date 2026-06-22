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


def seed_dirs(folder: str, split: str = "test", cov: str = "full") -> list[Path]:
    base = PER_MODEL / folder / split / cov
    if not base.is_dir():
        return []
    return sorted(d for d in base.glob("seed_*") if (d / "confusion_matrix.npy").exists())


def aggregate(folder: str, split: str = "test", cov: str = "full") -> dict | None:
    """Mean (+/- std) of every metric over the available seeds, from the npy.
    Also carries the per-point voted marking IoU from summary.json when present."""
    dirs = seed_dirs(folder, split, cov)
    if not dirs:
        return None
    per = [metrics_from_cm(np.load(d / "confusion_matrix.npy")) for d in dirs]
    agg: dict = {"n_seeds": len(dirs), "coverage": cov,
                 "seed_dirs": [str(d.relative_to(REPO)) for d in dirs]}
    keys = [k for k in per[0] if isinstance(per[0][k], float)]
    for k in keys:
        vals = np.array([m[k] for m in per], dtype=np.float64)
        agg[k] = float(vals.mean())
        agg[k + "_std"] = float(vals.std(ddof=0))
    for k in ("road_to_marking", "marking_to_road"):
        agg[k] = int(np.mean([m[k] for m in per]))
    voted = [load_json(d / "summary.json").get("voted_metrics", {}).get("marking_iou") for d in dirs]
    voted = [x for x in voted if x is not None]
    if voted:
        agg["voted_marking_iou"] = float(np.mean(voted))
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


def primary_dir(folder: str, split: str = "test", cov: str = "full") -> Path | None:
    """seed_42 if present, else the first available seed dir (test/full by default)."""
    dirs = seed_dirs(folder, split, cov)
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
        fv = data[code]["full_val"]
        sp = data[code]["sampled"]
        if t is None:
            continue
        sel_val = v.get("marking_iou", float("nan"))
        full_val_iou = fv["marking_iou"] if fv else float("nan")
        gap_ref = full_val_iou if fv else sel_val   # matched gap when full-val exists
        rows.append({
            "code": code, "model": label, "selected_epoch": ep, "n_test_seeds": t["n_seeds"],
            "selection_val_iou": round(sel_val, 6),
            "full_val_iou": round(full_val_iou, 6),
            "full_test_iou": round(t["marking_iou"], 6),
            "full_test_iou_std": round(t["marking_iou_std"], 6),
            "voted_test_iou": round(t.get("voted_marking_iou", float("nan")), 6),
            "sampled_test_iou": round(sp["marking_iou"], 6) if sp else float("nan"),
            "val_minus_test_iou": round(gap_ref - t["marking_iou"], 6),
            "gap_ref": "full_val" if fv else "selection_val",
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


def write_crosscheck(data: dict) -> None:
    """Sampled-vs-full agreement (the gate for whether full-coverage validation
    is needed). Small, systematic gap = sampled is mildly optimistic vs full."""
    rows = []
    for code in ORDER:
        full = data[code]["test"]
        samp = data[code]["sampled"]
        if not (full and samp):
            continue
        rows.append({
            "code": code,
            "full_iou": round(full["marking_iou"], 6),
            "full_iou_std": round(full["marking_iou_std"], 6),
            "sampled_iou": round(samp["marking_iou"], 6),
            "sampled_iou_std": round(samp["marking_iou_std"], 6),
            "gap_sampled_minus_full": round(samp["marking_iou"] - full["marking_iou"], 6),
        })
    if rows:
        COMP.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(COMP / "crosscheck_sampled_vs_full.csv", index=False)


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
    # Prefer full-coverage validation where it exists (matched protocol); else the
    # sampled selection validation. Label by whichever is shown.
    def vref(c):
        fv = data[c]["full_val"]
        return fv["marking_iou"] if fv else data[c]["val"].get("marking_iou", np.nan)
    any_full_val = any(data[c]["full_val"] for c, _l, _f in pres)
    val = [vref(c) for c, _l, _f in pres]
    test = [data[c]["test"]["marking_iou"] for c, _l, _f in pres]
    vlabel = "validation (full-coverage)" if any_full_val else "validation (sampled, selection)"
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.bar(x - w / 2, val, w, label=vlabel, color=S.NEUTRAL)
    ax.bar(x + w / 2, test, w, label="test (full-coverage, held-out)", color=S.CANDIDATE)
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
    any_full_val = df["full_val_iou"].notna().any()
    lines = [
        "# Test-set results", "",
        "Final held-out **test** evaluation. Generated by `results/test_suite/build_comparison.py`",
        "from the per-model passes in `results/per_model/<model>/test/full/seed_*/` (run by",
        "`run_test_all.py`). Every number is re-derived from each seed's `confusion_matrix.npy`.",
        "",
        "- **Full spatial coverage** (spatially-regular sampler, every frame covered) — the held-out",
        "  test number. Near-deterministic across evaluation seeds.",
        "- **Selection validation** (`selection_val_iou`) = the *sampled* training-time metric used to",
        "  choose epoch/model. **Full-coverage validation** (`full_val_iou`) = optional, protocol-",
        "  matched reference (only present if run); `val−test` uses it when available, else selection-val.",
        "- **Cross-check** (`comparisons/crosscheck_sampled_vs_full.csv`): sampled vs full coverage on",
        "  G2/H0 — sampled runs mildly optimistic; this licenses comparing test to the sampled selection-val.",
        "- **Voted IoU** (`voted_test_iou`) = per-point majority-voted confusion matrix; ≈ the",
        "  patch-accumulated headline (the no-voting check). Metrics are patch-accumulated, not logit voting.",
        "- **Single training seed (42)**; G2/H0 test = mean ± std over 3 *evaluation* seeds (sampling",
        "  variance only). ~0.008 marking-IoU noise floor: treat smaller differences as ties.",
        "- Selection was done on validation; test is reported once, not tuned on.",
        "- Labels = `road_marking3` (marking = raw 8+9+10); `lane_*` == `marking_*`.",
        "", "## Master table", "",
        "| model | sel.ep | sel-val IoU | "
        + ("full-val IoU | " if any_full_val else "")
        + "test IoU (full) | val−test | test P | test R | test F1 | test mIoU | pred/true | voted IoU |",
        "| --- | ---: | ---: | " + ("---: | " if any_full_val else "")
        + "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in df.iterrows():
        fv = f" {r['full_val_iou']:.4f} |" if any_full_val and pd.notna(r['full_val_iou']) else (" — |" if any_full_val else "")
        lines.append(
            f"| {r['model']} | {int(r['selected_epoch'])} | {r['selection_val_iou']:.4f} |"
            + fv
            + f" {r['full_test_iou']:.4f} ± {r['full_test_iou_std']:.4f} | {r['val_minus_test_iou']:+.4f} | "
            f"{r['test_precision']:.4f} | {r['test_recall']:.4f} | {r['test_f1']:.4f} | "
            f"{r['test_miou']:.4f} | {r['test_pred_true']:.3f} | {r['voted_test_iou']:.4f} |")
    lines += ["", "## Comparisons", "",
              "- `comparisons/rgb_effect__D0_vs_E0/` — clean 'add RGB' effect (LiDAR → +RGB).",
              "- `comparisons/system__D0_vs_G2_vs_H0/` — LiDAR baseline vs the full RGB systems.",
              "- `comparisons/shortcut__G2_vs_H0/` — brightness shortcut (rgb_valid over-prediction gap + FP fingerprint).",
              "- `comparisons/crosscheck_sampled_vs_full.csv` — sampled-vs-full agreement (G2/H0).",
              "- `comparisons/plots/` — thesis-styled figures.",
              "", "> `val−test` is the generalization gap, using full-coverage validation where present",
              "> (matched protocol), otherwise the sampled selection validation (see the cross-check).", ""]
    (REPO / "results" / "README.md").write_text("\n".join(lines))


def main() -> None:
    if not PER_MODEL.is_dir():
        raise SystemExit(f"No per-model outputs at {PER_MODEL}. Run run_test_all.py first.")
    PLOTS.mkdir(parents=True, exist_ok=True)

    data: dict = {}
    missing = []
    for code, _label, folder, rd, ep, _has in MODELS:
        t = aggregate(folder, "test", "full")
        if t is None:
            missing.append(code)
        data[code] = {
            "test": t,                                        # headline: full-coverage test
            "sampled": aggregate(folder, "test", "sampled"),  # cross-check
            "full_val": aggregate(folder, "validation", "full"),  # conditional matched val
            "val": val_metrics(rd, ep),                       # sampled selection-val (eval_history)
        }
    present = [c for c in ORDER if data[c]["test"] is not None]
    if missing:
        print(f"[warn] no test outputs yet for: {missing} (skipping those rows)")
    if not present:
        raise SystemExit("No model has test outputs yet — nothing to aggregate.")
    print(f"[build_comparison] aggregating: {present}")

    master = build_master(data)
    write_comparison("rgb_effect__D0_vs_E0", [c for c in ("D0", "E0") if data[c]["test"]], data)
    write_comparison("system__D0_vs_G2_vs_H0", [c for c in ("D0", "G2", "H0") if data[c]["test"]], data)
    write_crosscheck(data)
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
