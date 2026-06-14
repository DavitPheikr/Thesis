#!/usr/bin/env python
"""Milestone G0 loss-component analysis: weighted CE vs Lovasz vs total.

This is the genuinely new, G-specific stage. It does not run inference. It reads
``loss_components.csv`` and ``eval_history.csv`` for G0, joins them on epoch,
reads ``lovasz_lambda`` from the G0 config, and answers:

  - scale:      is the Lovasz contribution tiny, comparable, or dominant?
  - stability:  is the Lovasz term stable epoch-to-epoch or noisy?
  - alignment:  does Lovasz track marking IoU better than CE did? (In F0 the CE
                kept falling after the marking-IoU peak -- the mismatch G targets.)

It fails loudly if ``loss_components.csv`` is missing or inconsistent, and uses a
tolerance check for total == CE + lambda * Lovasz.

Fair-comparison rule encoded here: the ONLY loss-vs-loss comparison against F0 is
F0 ``val_loss`` (pure weighted CE) against G0 ``val_ce_loss`` (CE component).
G0 ``val_loss`` is the TOTAL and is never compared to F0.

Outputs:
    logs/milestone_g/run_analysis/G0_rgb_lovasz/loss_component_summary.csv
    logs/milestone_g/run_analysis/G0_rgb_lovasz/loss_alignment.csv
    logs/milestone_g/run_analysis/G0_rgb_lovasz/loss_component_report.md
    logs/milestone_g/run_analysis/G0_rgb_lovasz/plots/  (component plots)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUN_NAME = "G0_rgb_lovasz"
G0_RUN_DIR = PROJECT_ROOT / f"logs/milestone_g/runs/{RUN_NAME}"
F0_RUN_DIR = PROJECT_ROOT / "logs/milestone_f/runs/F0_rgb_soft_weights"
G0_CONFIG = PROJECT_ROOT / "logs/milestone_g/configs/g0_rgb_lovasz.yml"
ANALYSIS_DIR = PROJECT_ROOT / f"logs/milestone_g/run_analysis/{RUN_NAME}"
PLOTS_DIR = ANALYSIS_DIR / "plots"

CE_COLOR = "#4c78a8"
LOVASZ_COLOR = "#e45756"
TOTAL_COLOR = "#54a24b"
IOU_COLOR = "#d95f02"
F0_COLOR = "#9d4edd"

COMPOSE_ATOL = 1e-4
COMPOSE_RTOL = 1e-3


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= COMPOSE_ATOL + COMPOSE_RTOL * abs(float(b))


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.4)
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read_lovasz_lambda() -> float:
    cfg = yaml.safe_load(require_file(G0_CONFIG).read_text())
    loss_cfg = (cfg.get("pipeline", {}) or {}).get("loss", {}) or {}
    if loss_cfg.get("name") != "weighted_ce_lovasz":
        raise RuntimeError(
            f"G0 config loss name is {loss_cfg.get('name')!r}, expected weighted_ce_lovasz."
        )
    return float(loss_cfg.get("lovasz_lambda", 0.5))


def load_components_and_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    comp_path = G0_RUN_DIR / "loss_components.csv"
    if not comp_path.exists():
        raise RuntimeError(
            f"loss_components.csv not found in {G0_RUN_DIR}. The combined-loss path "
            "did not log components -- hard failure, not a skip."
        )
    comp = pd.read_csv(comp_path)
    if comp.empty:
        raise RuntimeError(f"{comp_path} is empty.")
    required = {
        "epoch",
        "train_ce_loss",
        "train_lovasz_loss",
        "train_total_loss",
        "val_ce_loss",
        "val_lovasz_loss",
        "val_total_loss",
    }
    missing = sorted(required - set(comp.columns))
    if missing:
        raise RuntimeError(f"{comp_path} missing columns: {missing}")
    history = pd.read_csv(require_file(G0_RUN_DIR / "eval_history.csv"))
    if len(comp) != len(history):
        raise RuntimeError(
            f"loss_components rows ({len(comp)}) != eval_history rows ({len(history)})."
        )
    comp = comp.sort_values("epoch").reset_index(drop=True)
    history = history.sort_values("epoch").reset_index(drop=True)
    if not np.array_equal(comp["epoch"].to_numpy(), history["epoch"].to_numpy()):
        raise RuntimeError("loss_components epochs do not match eval_history epochs.")
    return comp, history


def validate_composition(comp: pd.DataFrame, history: pd.DataFrame, lam: float) -> dict:
    max_compose_val = 0.0
    max_compose_train = 0.0
    max_total_val = 0.0
    for (_, c), (_, h) in zip(comp.iterrows(), history.iterrows()):
        train_compose = c["train_ce_loss"] + lam * c["train_lovasz_loss"]
        val_compose = c["val_ce_loss"] + lam * c["val_lovasz_loss"]
        if not close(c["train_total_loss"], train_compose):
            raise RuntimeError(
                f"epoch {int(c['epoch'])}: train_total != ce + {lam}*lovasz "
                f"({c['train_total_loss']} vs {train_compose})"
            )
        if not close(c["val_total_loss"], val_compose):
            raise RuntimeError(
                f"epoch {int(c['epoch'])}: val_total != ce + {lam}*lovasz "
                f"({c['val_total_loss']} vs {val_compose})"
            )
        if not close(c["val_total_loss"], h["val_loss"]):
            raise RuntimeError(
                f"epoch {int(c['epoch'])}: val_total_loss != eval_history val_loss "
                f"({c['val_total_loss']} vs {h['val_loss']})"
            )
        max_compose_train = max(max_compose_train, abs(c["train_total_loss"] - train_compose))
        max_compose_val = max(max_compose_val, abs(c["val_total_loss"] - val_compose))
        max_total_val = max(max_total_val, abs(c["val_total_loss"] - h["val_loss"]))
    return {
        "max_compose_train_residual": float(max_compose_train),
        "max_compose_val_residual": float(max_compose_val),
        "max_total_val_residual": float(max_total_val),
    }


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def cv(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    mean = np.mean(values)
    if mean == 0:
        return float("nan")
    return float(np.std(values) / abs(mean))


def build_summary(comp: pd.DataFrame, history: pd.DataFrame, lam: float) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "epoch": comp["epoch"].to_numpy(),
            "train_ce": comp["train_ce_loss"].to_numpy(),
            "train_lovasz_raw": comp["train_lovasz_loss"].to_numpy(),
            "train_lovasz_scaled": (lam * comp["train_lovasz_loss"]).to_numpy(),
            "train_total": comp["train_total_loss"].to_numpy(),
            "val_ce": comp["val_ce_loss"].to_numpy(),
            "val_lovasz_raw": comp["val_lovasz_loss"].to_numpy(),
            "val_lovasz_scaled": (lam * comp["val_lovasz_loss"]).to_numpy(),
            "val_total": comp["val_total_loss"].to_numpy(),
            "marking_iou": history["lane_iou"].to_numpy(),
        }
    )
    df["val_lovasz_share"] = df["val_lovasz_scaled"] / df["val_total"]
    df["train_lovasz_share"] = df["train_lovasz_scaled"] / df["train_total"]
    return df


def classify_scale(share: float) -> str:
    if share < 0.10:
        return "tiny"
    if share < 0.40:
        return "comparable"
    return "dominant"


def classify_stability(coef_var: float) -> str:
    if np.isnan(coef_var):
        return "undefined"
    if coef_var < 0.15:
        return "stable"
    if coef_var < 0.40:
        return "moderate"
    return "noisy"


def plot_components(df: pd.DataFrame, split: str, out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(df["epoch"], df[f"{split}_total"], marker="o", linewidth=2.4, color=TOTAL_COLOR, label="total")
    ax.plot(df["epoch"], df[f"{split}_ce"], marker="o", linewidth=2.2, color=CE_COLOR, label="weighted CE")
    ax.plot(df["epoch"], df[f"{split}_lovasz_scaled"], marker="s", linewidth=2.0, color=LOVASZ_COLOR, label="lambda * Lovasz (scaled)")
    ax.plot(df["epoch"], df[f"{split}_lovasz_raw"], marker="^", linewidth=1.6, linestyle="--", color=LOVASZ_COLOR, alpha=0.7, label="Lovasz (raw)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"G0 {split} loss components: CE vs Lovasz vs total")
    style_axes(ax)
    ax.legend()
    savefig(fig, PLOTS_DIR / out_name)


def plot_loss_vs_iou(df: pd.DataFrame, loss_col: str, loss_label: str, out_name: str) -> None:
    fig, ax1 = plt.subplots(figsize=(10.5, 5.8))
    color = {"val_ce": CE_COLOR, "val_lovasz_raw": LOVASZ_COLOR, "val_total": TOTAL_COLOR}.get(loss_col, CE_COLOR)
    ax1.plot(df["epoch"], df[loss_col], marker="o", linewidth=2.2, color=color, label=loss_label)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel(loss_label, color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    style_axes(ax1)
    best_epoch = int(df.loc[df["marking_iou"].idxmax(), "epoch"])
    ax1.axvline(best_epoch, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label=f"best IoU ep{best_epoch}")
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["marking_iou"], marker="s", linewidth=2.2, color=IOU_COLOR, label="marking IoU")
    ax2.set_ylabel("marking IoU", color=IOU_COLOR)
    ax2.tick_params(axis="y", labelcolor=IOU_COLOR)
    corr = safe_corr(df[loss_col].to_numpy(), df["marking_iou"].to_numpy())
    ax1.set_title(f"G0 {loss_label} vs marking IoU (corr={corr:+.3f})")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    savefig(fig, PLOTS_DIR / out_name)


def plot_f0_vs_g0_val_ce(df: pd.DataFrame) -> None:
    f0_hist = pd.read_csv(require_file(F0_RUN_DIR / "eval_history.csv"))
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    # F0 val_loss IS pure weighted CE; G0 val_ce is the CE component. Comparable.
    ax.plot(f0_hist["epoch"], f0_hist["val_loss"], marker="o", linewidth=2.2, color=F0_COLOR, label="F0 val CE (= val_loss)")
    ax.plot(df["epoch"], df["val_ce"], marker="s", linewidth=2.2, color=CE_COLOR, label="G0 val CE (component)")
    f0_best = int(f0_hist.loc[f0_hist["lane_iou"].idxmax(), "epoch"])
    g0_best = int(df.loc[df["marking_iou"].idxmax(), "epoch"])
    ax.axvline(f0_best, color=F0_COLOR, linestyle=":", linewidth=1.6, alpha=0.8, label=f"F0 best ep{f0_best}")
    ax.axvline(g0_best, color=CE_COLOR, linestyle="--", linewidth=1.6, alpha=0.8, label=f"G0 best ep{g0_best}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation weighted CE")
    ax.set_title("F0 vs G0 validation CE (fair: pure CE only)")
    style_axes(ax)
    ax.legend()
    savefig(fig, PLOTS_DIR / "f0_vs_g0_val_ce.png")


def write_report(
    df: pd.DataFrame,
    lam: float,
    compose: dict,
    alignment: dict,
    out_path: Path,
) -> None:
    best_epoch = int(df.loc[df["marking_iou"].idxmax(), "epoch"])
    best_row = df.loc[df["epoch"] == best_epoch].iloc[0]
    text = f"""# G0 Loss-Component Analysis

Loss = `weighted_CE + {lam} * Lovasz-Softmax`. This report characterizes the
Lovasz term: scale, stability, and whether it aligns with marking IoU better
than CE did.

## Composition Provenance (tolerance checks passed)

- max |val_total - (val_ce + {lam}*val_lovasz)|: `{compose['max_compose_val_residual']:.2e}`
- max |train_total - (train_ce + {lam}*train_lovasz)|: `{compose['max_compose_train_residual']:.2e}`
- max |val_total - eval_history val_loss|: `{compose['max_total_val_residual']:.2e}`

## Scale: is Lovasz tiny, comparable, or dominant?

At the best marking-IoU epoch ({best_epoch}):

- val CE: `{best_row['val_ce']:.6f}`
- val Lovasz (raw): `{best_row['val_lovasz_raw']:.6f}`
- val Lovasz (scaled, x{lam}): `{best_row['val_lovasz_scaled']:.6f}`
- val total: `{best_row['val_total']:.6f}`
- Lovasz share of total: `{best_row['val_lovasz_share']:.3f}`  -> **{alignment['scale_verdict']}**

Mean Lovasz share across training: `{alignment['mean_val_lovasz_share']:.3f}`.

## Stability: is Lovasz noisy across epochs?

- val Lovasz coefficient of variation: `{alignment['val_lovasz_cv']:.3f}`  -> **{alignment['stability_verdict']}**
- val CE coefficient of variation: `{alignment['val_ce_cv']:.3f}`

## Alignment: does Lovasz track marking IoU better than CE?

Lower loss should accompany higher IoU, so a stronger NEGATIVE correlation with
IoU means better alignment.

- corr(val CE, marking IoU): `{alignment['corr_ce_iou']:+.3f}`
- corr(val Lovasz, marking IoU): `{alignment['corr_lovasz_iou']:+.3f}`
- epoch of min val CE: `{alignment['argmin_val_ce_epoch']}`
- epoch of min val Lovasz: `{alignment['argmin_val_lovasz_epoch']}`
- epoch of max marking IoU: `{alignment['argmax_iou_epoch']}`

Verdict: **{alignment['alignment_verdict']}**

In F0, validation CE kept improving after the marking-IoU peak (the CE/IoU
mismatch). The key question for G is whether the Lovasz minimum coincides with
the IoU peak more tightly than the CE minimum does. Here the Lovasz minimum is
{abs(alignment['argmin_val_lovasz_epoch'] - alignment['argmax_iou_epoch'])} epoch(s)
from the IoU peak, versus {abs(alignment['argmin_val_ce_epoch'] - alignment['argmax_iou_epoch'])}
epoch(s) for CE.

## Outputs

- `loss_component_summary.csv` (per-epoch CE/Lovasz/total/share/IoU)
- `loss_alignment.csv` (the scalar diagnostics above)
- `plots/train_loss_components.png`, `plots/val_loss_components.png`
- `plots/val_ce_vs_marking_iou.png`, `plots/val_lovasz_vs_marking_iou.png`, `plots/val_total_vs_marking_iou.png`
- `plots/f0_vs_g0_val_ce.png`
"""
    out_path.write_text(text)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    lam = read_lovasz_lambda()
    comp, history = load_components_and_history()
    compose = validate_composition(comp, history, lam)
    df = build_summary(comp, history, lam)
    df.to_csv(ANALYSIS_DIR / "loss_component_summary.csv", index=False)

    best_epoch = int(df.loc[df["marking_iou"].idxmax(), "epoch"])
    best_share = float(df.loc[df["epoch"] == best_epoch, "val_lovasz_share"].iloc[0])
    val_lovasz_cv = cv(df["val_lovasz_raw"].to_numpy())
    val_ce_cv = cv(df["val_ce"].to_numpy())
    corr_ce_iou = safe_corr(df["val_ce"].to_numpy(), df["marking_iou"].to_numpy())
    corr_lovasz_iou = safe_corr(df["val_lovasz_raw"].to_numpy(), df["marking_iou"].to_numpy())
    argmin_ce = int(df.loc[df["val_ce"].idxmin(), "epoch"])
    argmin_lovasz = int(df.loc[df["val_lovasz_raw"].idxmin(), "epoch"])
    argmax_iou = int(df.loc[df["marking_iou"].idxmax(), "epoch"])

    lovasz_closer = abs(argmin_lovasz - argmax_iou) <= abs(argmin_ce - argmax_iou)
    corr_better = (
        not np.isnan(corr_lovasz_iou)
        and not np.isnan(corr_ce_iou)
        and corr_lovasz_iou <= corr_ce_iou
    )
    if lovasz_closer and corr_better:
        alignment_verdict = "Lovasz aligns with marking IoU better than CE"
    elif lovasz_closer or corr_better:
        alignment_verdict = "Lovasz aligns with marking IoU comparably to CE (mixed signal)"
    else:
        alignment_verdict = "Lovasz does NOT align with marking IoU better than CE"

    alignment = {
        "best_epoch": best_epoch,
        "best_val_lovasz_share": best_share,
        "mean_val_lovasz_share": float(df["val_lovasz_share"].mean()),
        "scale_verdict": classify_scale(best_share),
        "val_lovasz_cv": val_lovasz_cv,
        "val_ce_cv": val_ce_cv,
        "stability_verdict": classify_stability(val_lovasz_cv),
        "corr_ce_iou": corr_ce_iou,
        "corr_lovasz_iou": corr_lovasz_iou,
        "argmin_val_ce_epoch": argmin_ce,
        "argmin_val_lovasz_epoch": argmin_lovasz,
        "argmax_iou_epoch": argmax_iou,
        "alignment_verdict": alignment_verdict,
        **compose,
    }
    pd.DataFrame([alignment]).to_csv(ANALYSIS_DIR / "loss_alignment.csv", index=False)
    (ANALYSIS_DIR / "loss_alignment.json").write_text(json.dumps(alignment, indent=2))

    plot_components(df, "train", "train_loss_components.png")
    plot_components(df, "val", "val_loss_components.png")
    plot_loss_vs_iou(df, "val_ce", "validation CE", "val_ce_vs_marking_iou.png")
    plot_loss_vs_iou(df, "val_lovasz_raw", "validation Lovasz", "val_lovasz_vs_marking_iou.png")
    plot_loss_vs_iou(df, "val_total", "validation total", "val_total_vs_marking_iou.png")
    plot_f0_vs_g0_val_ce(df)

    write_report(df, lam, compose, alignment, ANALYSIS_DIR / "loss_component_report.md")

    print("G0 loss-component analysis complete")
    print(f"out_dir {ANALYSIS_DIR}")
    print(f"scale_verdict {alignment['scale_verdict']} (share {best_share:.3f})")
    print(f"stability_verdict {alignment['stability_verdict']} (cv {val_lovasz_cv:.3f})")
    print(f"alignment_verdict {alignment['alignment_verdict']}")
    print("script_status PASS")


if __name__ == "__main__":
    main()
