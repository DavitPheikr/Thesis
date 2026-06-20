#!/usr/bin/env python
"""Thesis-ready per-sequence and per-subtype plots (CPU only, no inference).

Reads the CSVs that 02_sampled_error_analysis.py already wrote
(`per_sequence_metrics.csv`, `raw_subtype_rgb_stratified_metrics.csv`) and turns
them into figures. Generic: works for any run via `_suite_paths.py`.

Outputs (under <ANALYSIS_OUT>/plots/):
  per_sequence_marking_performance.png  — marking IoU per sequence (sorted) + pred/true
  raw_subtype_marking_recall.png        — recall per marking subtype (all / rgb_valid / rgb_invalid)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _suite_style as S  # noqa: E402
from _suite_paths import ANALYSIS_OUT, CANDIDATE_LABEL  # noqa: E402

S.apply()
PLOTS_DIR = ANALYSIS_OUT / "plots"


def find_sampled_dir() -> Path:
    matches = sorted(ANALYSIS_OUT.glob("sampled_error_analysis_epoch*"))
    if not matches:
        raise FileNotFoundError(
            f"No sampled_error_analysis_epoch* under {ANALYSIS_OUT}; run 02 first.")
    return matches[-1]


def plot_per_sequence(sampled: Path) -> None:
    df = pd.read_csv(sampled / "per_sequence_metrics.csv", dtype={"seq_id": str})
    df = df.sort_values("marking_iou").reset_index(drop=True)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(df)), 5.5))
    ax.bar(x, df["marking_iou"], color=S.MARKING, alpha=0.85, label="marking IoU")
    ax.axhline(df["marking_iou"].mean(), color=S.MARKING, ls="--", lw=1.2,
               label=f"mean IoU {df['marking_iou'].mean():.3f}")
    ax.set_ylabel("marking IoU", color=S.MARKING)
    ax.tick_params(axis="y", labelcolor=S.MARKING)
    ax.set_ylim(0, max(0.8, df["marking_iou"].max() + 0.05))
    ax.set_xticks(x)
    ax.set_xticklabels(df["seq_id"], rotation=0)
    ax.set_xlabel("sequence")
    S.style_axes(ax)

    ax2 = ax.twinx()
    ax2.plot(x, df["predicted_true_marking_ratio"], color=S.PRED_TRUE, marker="o",
             lw=1.6, label="predicted / true ratio")
    ax2.axhline(1.0, color=S.CALIBRATED, ls=":", lw=1.2, label="calibrated (1.0)")
    ax2.set_ylabel("predicted / true marking ratio", color=S.PRED_TRUE)
    ax2.tick_params(axis="y", labelcolor=S.PRED_TRUE)
    ax2.spines["top"].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    ax.set_title(f"{CANDIDATE_LABEL}: per-sequence marking IoU and calibration (best checkpoint)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "per_sequence_marking_performance.png")
    plt.close(fig)


def plot_per_subtype(sampled: Path) -> None:
    df = pd.read_csv(sampled / "raw_subtype_rgb_stratified_metrics.csv")
    df = df[df["stratum"].isin(["all", "rgb_valid", "rgb_invalid"])]
    subtypes = df[df["stratum"] == "all"].sort_values("raw_id")
    names = [f"{r.raw_name}\n(raw {int(r.raw_id)}, n={int(r.true_marking):,})"
             for r in subtypes.itertuples()]
    order = list(subtypes["raw_id"])
    x = np.arange(len(order))
    w = 0.26

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(order)), 5.5))
    for i, stratum in enumerate(["all", "rgb_valid", "rgb_invalid"]):
        s = df[df["stratum"] == stratum].set_index("raw_id").reindex(order)
        ax.bar(x + (i - 1) * w, s["marking_recall"], w, label=stratum, color=S.RGB[stratum])
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("marking recall")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{CANDIDATE_LABEL}: marking recall by raw subtype and RGB validity (best checkpoint)")
    ax.legend(title="stratum")
    S.style_axes(ax)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "raw_subtype_marking_recall.png")
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sampled = find_sampled_dir()
    plot_per_sequence(sampled)
    plot_per_subtype(sampled)
    print(f"wrote {PLOTS_DIR}/per_sequence_marking_performance.png")
    print(f"wrote {PLOTS_DIR}/raw_subtype_marking_recall.png")
    print("stratified_plots_status PASS")


if __name__ == "__main__":
    main()
