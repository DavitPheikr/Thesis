#!/usr/bin/env python
"""RGB luminance / color shortcut fingerprint analysis (analysis-only).

Question: are the residual road->marking false positives associated mainly with
RGB *brightness/luminance* rather than a specific marking *color*?

This script does NOT run inference and does NOT touch training. It reuses the
existing sampled-error outputs:

    sampled_error_analysis_epoch*/group_feature_summary.csv      (per-outcome RGB/intensity summaries)
    sampled_error_analysis_epoch*/rgb_valid_stratified_metrics.csv (RGB-valid vs invalid error counts)

Scope/fidelity note (read before trusting saturation):
- brightness, luminance, warmth, per-channel means, intensity, rgb_valid ratio,
  and counts are computed from the per-channel ``*_mean`` columns. Because these
  metrics are LINEAR in R/G/B, the group ``*_mean``-derived values are EXACT
  group means.
- ``max_rgb``/``min_rgb``/``saturation_proxy`` are NON-LINEAR, so they are only
  AGGREGATE approximations here (computed from the per-channel medians, i.e.
  max/min of the central channel values), not true per-point statistics. They
  are flagged ``_approx`` in outputs.
- A true per-point density scatter is not possible from these aggregated outputs;
  the brightness-vs-intensity scatter here is group-level (one point per group).

Wording discipline: results are phrased as evidence that predictions are
ASSOCIATED with a luminance-based RGB shortcut. This script does not and cannot
prove the model internally computes brightness.

Runs on F0 now and G0 after it finishes; supports an F0-vs-G0 comparison.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]

OUTCOME_GROUPS = (
    "road_tp",
    "road_to_marking",
    "marking_tp",
    "marking_to_road",
    "other_to_marking",
)
GROUP_LABELS = {
    "road_tp": "road TP",
    "road_to_marking": "road->marking (FP)",
    "marking_tp": "marking TP",
    "marking_to_road": "marking->road (FN)",
    "other_to_marking": "other->marking (FP)",
}
GROUP_COLORS = {
    "road_tp": "#666666",
    "road_to_marking": "#d62728",
    "marking_tp": "#2ca02c",
    "marking_to_road": "#ff7f0e",
    "other_to_marking": "#9467bd",
}

PRESETS = {
    "f0": {
        "run_dir": REPO_ROOT / "logs/milestone_f/run_analysis/F0_rgb_soft_weights",
        "label": "F0",
    },
    "g0": {
        "run_dir": REPO_ROOT / "logs/milestone_g/run_analysis/G0_rgb_lovasz",
        "label": "G0",
    },
}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def find_sampled_dir(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("sampled_error_analysis_epoch*"))
    if not matches:
        raise FileNotFoundError(
            f"No sampled_error_analysis_epoch* dir under {run_dir}; run the "
            "sampled error analysis first."
        )
    return matches[-1]


def load_group_summary(sampled_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(require_file(sampled_dir / "group_feature_summary.csv"))
    df = df.set_index("group")
    missing = [g for g in OUTCOME_GROUPS if g not in df.index]
    if missing:
        raise RuntimeError(f"group_feature_summary.csv missing groups: {missing}")
    needed = [
        "red_mean", "green_mean", "blue_mean",
        "red_median", "green_median", "blue_median",
        "intensity_mean", "intensity_median",
        "rgb_valid_mean", "red_count",
    ]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"group_feature_summary.csv missing columns: {missing_cols}")
    return df


def load_rgb_strat(sampled_dir: Path) -> pd.DataFrame | None:
    path = sampled_dir / "rgb_valid_stratified_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path).set_index("stratum")


LUMINANCE_WEIGHTS = (0.299, 0.587, 0.114)


def compute_fingerprint(group_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in OUTCOME_GROUPS:
        g = group_df.loc[group]
        r_mean, g_mean, b_mean = float(g["red_mean"]), float(g["green_mean"]), float(g["blue_mean"])
        r_med, g_med, b_med = float(g["red_median"]), float(g["green_median"]), float(g["blue_median"])
        meds = (r_med, g_med, b_med)
        rows.append(
            {
                "group": group,
                "count": int(g["red_count"]),
                "rgb_valid_ratio": float(g["rgb_valid_mean"]),
                # exact group means (linear in channels)
                "R_mean": r_mean,
                "G_mean": g_mean,
                "B_mean": b_mean,
                "brightness_mean": (r_mean + g_mean + b_mean) / 3.0,
                "luminance_mean": (
                    LUMINANCE_WEIGHTS[0] * r_mean
                    + LUMINANCE_WEIGHTS[1] * g_mean
                    + LUMINANCE_WEIGHTS[2] * b_mean
                ),
                "warmth_mean": r_mean - b_mean,
                # medians (channel-wise, exact per channel; combinations are aggregate)
                "R_median": r_med,
                "G_median": g_med,
                "B_median": b_med,
                "brightness_median_approx": (r_med + g_med + b_med) / 3.0,
                "warmth_median": r_med - b_med,
                "max_rgb_median_approx": max(meds),
                "min_rgb_median_approx": min(meds),
                "saturation_proxy_median_approx": max(meds) - min(meds),
                # LiDAR intensity
                "intensity_mean": float(g["intensity_mean"]),
                "intensity_median": float(g["intensity_median"]),
            }
        )
    return pd.DataFrame(rows).set_index("group").reindex(OUTCOME_GROUPS)


def answer_questions(fp: pd.DataFrame, strat: pd.DataFrame | None) -> dict[str, Any]:
    def b(group: str) -> float:
        return float(fp.loc[group, "brightness_mean"])

    def lum(group: str) -> float:
        return float(fp.loc[group, "luminance_mean"])

    def inten(group: str) -> float:
        return float(fp.loc[group, "intensity_mean"])

    def bmed(group: str) -> float:
        return float(fp.loc[group, "brightness_median_approx"])

    def sat(group: str) -> float:
        return float(fp.loc[group, "saturation_proxy_median_approx"])

    fp_b = b("road_to_marking")
    road_b = b("road_tp")
    mark_b = b("marking_tp")
    fp_inten = inten("road_to_marking")
    road_inten = inten("road_tp")
    mark_inten = inten("marking_tp")

    # Intensity position is well-defined because marking intensity > road intensity
    # (retroreflective paint). 0 = road-like, 1 = marking-like.
    def norm(value: float, lo: float, hi: float) -> float:
        return float((value - lo) / (hi - lo)) if hi != lo else float("nan")

    fp_inten_pos = norm(fp_inten, road_inten, mark_inten)

    answers = {
        "q1_fp_brighter_than_road_tp": {
            "value": fp_b > road_b,
            "fp_brightness_mean": fp_b,
            "road_tp_brightness_mean": road_b,
            "delta_mean": fp_b - road_b,
            "fp_brightness_median": bmed("road_to_marking"),
            "road_tp_brightness_median": bmed("road_tp"),
            "delta_median": bmed("road_to_marking") - bmed("road_tp"),
        },
        "q2_fp_brightness_vs_true_marking": {
            "fp_brightness_mean": fp_b,
            "marking_tp_brightness_mean": mark_b,
            "delta_fp_minus_marking_mean": fp_b - mark_b,
            "fp_brightness_median": bmed("road_to_marking"),
            "marking_tp_brightness_median": bmed("marking_tp"),
            "delta_fp_minus_marking_median": bmed("road_to_marking") - bmed("marking_tp"),
            "fp_at_least_as_bright_as_marking": fp_b >= mark_b and bmed("road_to_marking") >= bmed("marking_tp"),
        },
        "q3_marking_colored_or_bright_gray": {
            # 'colored vs gray' is best read from saturation (channel spread), not
            # warmth alone; warmth is reported as secondary context.
            "fp_saturation_proxy_approx": sat("road_to_marking"),
            "marking_saturation_proxy_approx": sat("marking_tp"),
            "fp_warmth": float(fp.loc["road_to_marking", "warmth_mean"]),
            "marking_warmth": float(fp.loc["marking_tp", "warmth_mean"]),
            "fp_less_saturated_than_marking": sat("road_to_marking") < sat("marking_tp"),
        },
        "q4_missed_markings_darker_than_detected": {
            "value": b("marking_to_road") < mark_b,
            "missed_brightness_mean": b("marking_to_road"),
            "detected_brightness_mean": mark_b,
            "delta_mean": b("marking_to_road") - mark_b,
        },
        "q5_brightness_intensity_disagreement_in_fp": {
            # FP is brighter than road (an elevated, marking-associated RGB cue),
            # yet its LiDAR intensity stays close to road (road-like).
            "fp_brighter_than_road": fp_b > road_b,
            "fp_intensity_position_road0_marking1": fp_inten_pos,
            "fp_intensity_mean": fp_inten,
            "road_tp_intensity_mean": road_inten,
            "marking_tp_intensity_mean": mark_inten,
            "disagreement": (
                (fp_b > road_b)
                and (not np.isnan(fp_inten_pos))
                and (fp_inten_pos < 0.5)
            ),
        },
    }
    if strat is not None and {"all", "rgb_valid", "rgb_invalid"}.issubset(strat.index):
        valid_fp = int(strat.loc["rgb_valid", "road_to_marking"])
        invalid_fp = int(strat.loc["rgb_invalid", "road_to_marking"])
        total_fp = valid_fp + invalid_fp
        answers["q6_concentrated_in_rgb_valid"] = {
            "road_to_marking_rgb_valid": valid_fp,
            "road_to_marking_rgb_invalid": invalid_fp,
            "rgb_valid_share_of_fp": float(valid_fp / total_fp) if total_fp else float("nan"),
            "pred_true_rgb_valid": float(strat.loc["rgb_valid", "predicted_true_marking_ratio"]),
            "pred_true_rgb_invalid": float(strat.loc["rgb_invalid", "predicted_true_marking_ratio"]),
        }
    else:
        answers["q6_concentrated_in_rgb_valid"] = {"note": "rgb_valid_stratified_metrics.csv not available"}
    return answers


# ----------------------------- plots -------------------------------------- #

def _bar_positions(n_groups: int, n_series: int, width: float):
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * width
    return np.arange(n_groups), offsets


def plot_brightness_luminance(fp: pd.DataFrame, label: str, out_path: Path) -> None:
    groups = list(fp.index)
    x, offsets = _bar_positions(len(groups), 2, 0.38)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x + offsets[0], fp["brightness_mean"], 0.38, label="brightness = mean(R,G,B)", color="#4c78a8")
    ax.bar(x + offsets[1], fp["luminance_mean"], 0.38, label="luminance 0.299R+0.587G+0.114B", color="#f58518")
    ax.set_xticks(x, [GROUP_LABELS[g] for g in groups], rotation=20, ha="right")
    ax.set_ylabel("RGB value (0-1, exact group mean)")
    ax.set_title(f"{label}: brightness / luminance by outcome group")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    for i, g in enumerate(groups):
        ax.text(i + offsets[0], float(fp.loc[g, "brightness_mean"]) + 0.008, f"{float(fp.loc[g, 'brightness_mean']):.3f}", ha="center", fontsize=8)
    fig.tight_layout(pad=1.4)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def plot_warmth_saturation(fp: pd.DataFrame, label: str, out_path: Path) -> None:
    groups = list(fp.index)
    x, offsets = _bar_positions(len(groups), 2, 0.38)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x + offsets[0], fp["warmth_mean"], 0.38, label="warmth = R - B (exact mean)", color="#e45756")
    ax.bar(x + offsets[1], fp["saturation_proxy_median_approx"], 0.38, label="saturation proxy = max-min (approx, medians)", color="#72b7b2")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x, [GROUP_LABELS[g] for g in groups], rotation=20, ha="right")
    ax.set_ylabel("RGB units (0-1)")
    ax.set_title(f"{label}: warmth and saturation proxy by outcome group")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    fig.tight_layout(pad=1.4)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def plot_brightness_vs_intensity(fp: pd.DataFrame, label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    counts = fp["count"].to_numpy(dtype=float)
    sizes = 120 + 1400 * (counts / counts.max())
    for g in fp.index:
        ax.scatter(
            float(fp.loc[g, "brightness_mean"]),
            float(fp.loc[g, "intensity_mean"]),
            s=float(sizes[list(fp.index).index(g)]),
            color=GROUP_COLORS[g],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
            label=f"{GROUP_LABELS[g]} (n={int(fp.loc[g, 'count']):,})",
        )
        ax.annotate(
            GROUP_LABELS[g],
            (float(fp.loc[g, "brightness_mean"]), float(fp.loc[g, "intensity_mean"])),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
        )
    # reference lines at road_tp and marking_tp brightness
    ax.axvline(float(fp.loc["road_tp", "brightness_mean"]), color="#666666", linestyle=":", alpha=0.7)
    ax.axvline(float(fp.loc["marking_tp", "brightness_mean"]), color="#2ca02c", linestyle=":", alpha=0.7)
    ax.set_xlabel("RGB brightness = mean(R,G,B) [group mean]")
    ax.set_ylabel("LiDAR intensity [group mean]")
    ax.set_title(f"{label}: brightness vs LiDAR intensity by outcome group\n(group-level means; marker size ~ point count)")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout(pad=1.4)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


# --------------------------- conclusions ---------------------------------- #

def write_conclusions(label: str, fp: pd.DataFrame, answers: dict, out_path: Path) -> None:
    q1 = answers["q1_fp_brighter_than_road_tp"]
    q2 = answers["q2_fp_brightness_vs_true_marking"]
    q3 = answers["q3_marking_colored_or_bright_gray"]
    q4 = answers["q4_missed_markings_darker_than_detected"]
    q5 = answers["q5_brightness_intensity_disagreement_in_fp"]
    q6 = answers["q6_concentrated_in_rgb_valid"]

    def yn(v: bool) -> str:
        return "YES" if v else "NO"

    lines = [
        f"# {label} RGB Shortcut Fingerprint",
        "",
        "**Framing.** These results are evidence that predictions are *associated*",
        "with a luminance-based RGB shortcut. They do not prove the model internally",
        "computes or uses brightness; they characterize the RGB/intensity profile of",
        "the points the model labels marking vs road.",
        "",
        "Linear metrics (brightness, luminance, warmth, per-channel means) are exact",
        "group means. `saturation_proxy` is an aggregate approximation from per-channel",
        "medians, not a per-point statistic.",
        "",
        "## Fingerprint table (key columns)",
        "",
        "brightness shown as mean / median(approx); markings are mean-skewed by a dark",
        "tail, so the median is the more robust 'typical' value.",
        "",
        "| group | n | brightness mean/med | luminance | warmth(R-B) | sat_proxy~ | intensity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in fp.index:
        lines.append(
            f"| {GROUP_LABELS[g]} | {int(fp.loc[g, 'count']):,} | "
            f"{float(fp.loc[g, 'brightness_mean']):.3f} / {float(fp.loc[g, 'brightness_median_approx']):.3f} | "
            f"{float(fp.loc[g, 'luminance_mean']):.3f} | "
            f"{float(fp.loc[g, 'warmth_mean']):+.3f} | {float(fp.loc[g, 'saturation_proxy_median_approx']):.3f} | "
            f"{float(fp.loc[g, 'intensity_mean']):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"**Q1. Are road->marking FPs brighter than road TPs?** {yn(q1['value'])} "
            f"(mean {q1['fp_brightness_mean']:.3f} vs {q1['road_tp_brightness_mean']:.3f}, "
            f"delta {q1['delta_mean']:+.3f}; median {q1['fp_brightness_median']:.3f} vs "
            f"{q1['road_tp_brightness_median']:.3f}, delta {q1['delta_median']:+.3f}).",
            "",
            f"**Q2. Are FPs similar to true markings in brightness?** FP is in fact at least as "
            f"bright: {yn(q2['fp_at_least_as_bright_as_marking'])} "
            f"(mean {q2['fp_brightness_mean']:.3f} vs marking {q2['marking_tp_brightness_mean']:.3f}, "
            f"delta {q2['delta_fp_minus_marking_mean']:+.3f}; median {q2['fp_brightness_median']:.3f} "
            f"vs {q2['marking_tp_brightness_median']:.3f}, delta {q2['delta_fp_minus_marking_median']:+.3f}).",
            "",
            f"**Q3. Marking-colored, or bright gray/white road?** FP is less color-saturated than "
            f"true markings: {yn(q3['fp_less_saturated_than_marking'])} "
            f"(saturation proxy FP {q3['fp_saturation_proxy_approx']:.3f} vs marking "
            f"{q3['marking_saturation_proxy_approx']:.3f}; warmth FP {q3['fp_warmth']:+.3f} vs marking "
            f"{q3['marking_warmth']:+.3f}). High brightness with lower saturation is consistent with "
            "bright near-neutral road (glare/concrete) rather than marking-specific color.",
            "",
            f"**Q4. Are missed markings darker than detected markings?** {yn(q4['value'])} "
            f"(missed {q4['missed_brightness_mean']:.3f} vs detected {q4['detected_brightness_mean']:.3f}, "
            f"delta {q4['delta_mean']:+.3f}).",
            "",
            f"**Q5. Does RGB brightness disagree with LiDAR intensity in FPs?** "
            f"{yn(q5['disagreement'])}. FPs are brighter than road TPs (an elevated, "
            "marking-associated RGB cue), yet their LiDAR intensity stays road-like: on a "
            f"road(0)->marking(1) intensity scale the FPs sit at only "
            f"{q5['fp_intensity_position_road0_marking1']:.2f} "
            f"(FP intensity {q5['fp_intensity_mean']:.1f} vs road {q5['road_tp_intensity_mean']:.1f}, "
            f"marking {q5['marking_tp_intensity_mean']:.1f}).",
            "",
        ]
    )
    if "road_to_marking_rgb_valid" in q6:
        lines.append(
            f"**Q6. Concentrated in rgb_valid points?** road->marking FPs: "
            f"{q6['road_to_marking_rgb_valid']:,} rgb_valid vs "
            f"{q6['road_to_marking_rgb_invalid']:,} rgb_invalid "
            f"({q6['rgb_valid_share_of_fp']:.1%} of FPs are rgb_valid); pred/true "
            f"{q6['pred_true_rgb_valid']:.3f} valid vs {q6['pred_true_rgb_invalid']:.3f} invalid."
        )
    else:
        lines.append(f"**Q6. Concentrated in rgb_valid points?** {q6.get('note')}")

    lines.extend(
        [
            "",
            "## Overall reading",
            "",
            "If FPs are (a) brighter than road TPs, (b) as bright as or brighter than true",
            "markings, (c) less warm/saturated than true markings, and (d) road-like in LiDAR",
            "intensity, that is consistent with predictions being *associated with a",
            "luminance-based RGB shortcut*: bright near-neutral road surfaces (glare, light",
            "concrete, overexposed pixels) carry the same high-RGB signature as white paint.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def write_compare(
    base_label: str,
    base_fp: pd.DataFrame,
    base_ans: dict,
    cmp_label: str,
    cmp_fp: pd.DataFrame,
    cmp_ans: dict,
    out_dir: Path,
) -> None:
    def fp_count(ans: dict) -> int | None:
        q6 = ans.get("q6_concentrated_in_rgb_valid", {})
        if "road_to_marking_rgb_valid" in q6:
            return int(q6["road_to_marking_rgb_valid"] + q6["road_to_marking_rgb_invalid"])
        return None

    base_count = fp_count(base_ans) or int(base_fp.loc["road_to_marking", "count"])
    cmp_count = fp_count(cmp_ans) or int(cmp_fp.loc["road_to_marking", "count"])

    cols = ["brightness_mean", "luminance_mean", "warmth_mean", "saturation_proxy_median_approx", "intensity_mean"]
    rows = []
    for g in OUTCOME_GROUPS:
        row = {"group": g}
        for c in cols:
            row[f"{base_label}_{c}"] = float(base_fp.loc[g, c])
            row[f"{cmp_label}_{c}"] = float(cmp_fp.loc[g, c])
            row[f"delta_{c}"] = float(cmp_fp.loc[g, c] - base_fp.loc[g, c])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "rgb_shortcut_compare.csv", index=False)

    # comparison plot: road->marking FP fingerprint, base vs compare
    fp_metrics = ["brightness_mean", "luminance_mean", "warmth_mean", "saturation_proxy_median_approx"]
    labels = ["brightness", "luminance", "warmth", "sat_proxy~"]
    base_vals = [float(base_fp.loc["road_to_marking", m]) for m in fp_metrics]
    cmp_vals = [float(cmp_fp.loc["road_to_marking", m]) for m in fp_metrics]
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    ax1.bar(x - 0.2, base_vals, 0.4, label=base_label, color="#54a24b")
    ax1.bar(x + 0.2, cmp_vals, 0.4, label=cmp_label, color="#b279a2")
    ax1.set_xticks(x, labels)
    ax1.set_title("road->marking FP fingerprint (does the shortcut profile persist?)")
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend()
    ax2.bar([base_label, cmp_label], [base_count, cmp_count], color=["#54a24b", "#b279a2"])
    ax2.set_title("road->marking FP count (symptom)")
    ax2.set_ylabel("sampled false-positive points")
    for i, v in enumerate([base_count, cmp_count]):
        ax2.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"{base_label} vs {cmp_label}: RGB shortcut symptom vs mechanism", y=0.99, fontsize=14)
    fig.tight_layout(pad=1.4)
    fig.savefig(out_dir / "rgb_shortcut_compare.png", dpi=240, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)

    fp_reduced = cmp_count < base_count
    base_fp_bright = float(base_fp.loc["road_to_marking", "brightness_mean"])
    cmp_fp_bright = float(cmp_fp.loc["road_to_marking", "brightness_mean"])
    fingerprint_same = abs(cmp_fp_bright - base_fp_bright) < 0.02  # ~2% RGB units

    lines = [
        f"# {base_label} vs {cmp_label}: RGB Shortcut Symptom vs Mechanism",
        "",
        "Same framing: association, not proof of internal brightness use.",
        "",
        "## Did the symptom shrink?",
        "",
        f"- road->marking FP count: {base_label} {base_count:,} -> {cmp_label} {cmp_count:,} "
        f"({'reduced' if fp_reduced else 'not reduced'}).",
        "",
        "## Do the remaining FPs have the same brightness fingerprint?",
        "",
        f"- road->marking FP brightness: {base_label} {base_fp_bright:.3f} -> {cmp_label} {cmp_fp_bright:.3f} "
        f"(delta {cmp_fp_bright - base_fp_bright:+.3f}).",
        f"- fingerprint approximately unchanged: {'YES' if fingerprint_same else 'NO'}.",
        "",
        "## Interpretation",
        "",
    ]
    if fp_reduced and fingerprint_same:
        lines.append(
            f"{cmp_label} reduced the *symptom* (fewer road->marking FPs) but the remaining "
            "FPs keep the same bright near-neutral profile -> evidence the luminance-based "
            "RGB-shortcut *association* is largely unchanged; the loss change suppressed the "
            "rate of firing, not the cue the firing is associated with."
        )
    elif fp_reduced and not fingerprint_same:
        lines.append(
            f"{cmp_label} reduced FPs AND shifted their brightness profile -> the change "
            "affected which points get mislabeled, not just how many."
        )
    else:
        lines.append(
            f"{cmp_label} did not reduce road->marking FPs; the shortcut symptom persists."
        )
    lines.append("")
    (out_dir / "rgb_shortcut_compare_conclusions.md").write_text("\n".join(lines))


# ------------------------------ driver ------------------------------------ #

def analyze_one(run_dir: Path, label: str, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    sampled_dir = find_sampled_dir(run_dir)
    group_df = load_group_summary(sampled_dir)
    strat = load_rgb_strat(sampled_dir)
    fp = compute_fingerprint(group_df)
    answers = answer_questions(fp, strat)

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fp.to_csv(out_dir / "rgb_shortcut_fingerprint.csv")
    (out_dir / "rgb_shortcut_questions.json").write_text(json.dumps(answers, indent=2, default=float))
    plot_brightness_luminance(fp, label, plots_dir / "brightness_luminance_by_group.png")
    plot_warmth_saturation(fp, label, plots_dir / "warmth_saturation_by_group.png")
    plot_brightness_vs_intensity(fp, label, plots_dir / "brightness_vs_intensity_scatter.png")
    write_conclusions(label, fp, answers, out_dir / "rgb_shortcut_conclusions.md")

    print(f"[{label}] sampled_dir {sampled_dir}")
    print(f"[{label}] wrote {out_dir}")
    return fp, answers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("f0", "g0", "f0_vs_g0"), default=None,
                   help="Convenience presets for repo default run dirs.")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="run_analysis/<RUN> dir; sampled_error_analysis_epoch* is auto-found.")
    p.add_argument("--label", default="run")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--compare-run-dir", type=Path, default=None)
    p.add_argument("--compare-label", default="G0")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_run_dir = args.run_dir
    base_label = args.label
    cmp_run_dir = args.compare_run_dir
    cmp_label = args.compare_label

    if args.preset == "f0":
        base_run_dir, base_label = PRESETS["f0"]["run_dir"], "F0"
    elif args.preset == "g0":
        base_run_dir, base_label = PRESETS["g0"]["run_dir"], "G0"
    elif args.preset == "f0_vs_g0":
        base_run_dir, base_label = PRESETS["f0"]["run_dir"], "F0"
        cmp_run_dir, cmp_label = PRESETS["g0"]["run_dir"], "G0"

    if base_run_dir is None:
        raise SystemExit("Provide --preset or --run-dir.")

    out_dir = args.out_dir or (base_run_dir / "rgb_shortcut_analysis")
    base_fp, base_ans = analyze_one(Path(base_run_dir), base_label, Path(out_dir))

    if cmp_run_dir is not None:
        cmp_out = Path(cmp_run_dir) / "rgb_shortcut_analysis"
        cmp_fp, cmp_ans = analyze_one(Path(cmp_run_dir), cmp_label, cmp_out)
        compare_out = Path(args.out_dir) if args.out_dir else cmp_out
        compare_out.mkdir(parents=True, exist_ok=True)
        write_compare(base_label, base_fp, base_ans, cmp_label, cmp_fp, cmp_ans, compare_out)
        print(f"[compare] wrote {compare_out / 'rgb_shortcut_compare_conclusions.md'}")

    print("script_status PASS")


if __name__ == "__main__":
    main()
