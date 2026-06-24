"""Shared plotting style for the reproducible testing suite.

ONE source of truth for colours and typography so every figure in the suite is
mutually consistent and thesis-ready. The rule this enforces: **the same entity
is always the same colour**, in every plot it appears in.

Import in any plotting script:

    import _suite_style as S
    S.apply()                      # uniform fonts / sizes / grid (call once)
    ax.plot(..., color=S.MARKING)  # semantic colours
    S.style_axes(ax)               # hide top/right spines, soft y-grid
    S.legend(ax)                   # consistent, readable legend box

Colour namespaces (each entity has exactly one colour everywhere):
  * classes     : ROAD / MARKING / OTHER
  * runs         : CANDIDATE (the run being analysed) / BASELINE (its comparison)
  * marking-metric family (shown together) : METRIC["iou"|"precision"|"recall"|"f1"]
  * calibration  : PRED_TRUE (predicted/true marking ratio) + CALIBRATED ref line
  * RGB validity : RGB["all"|"rgb_valid"|"rgb_invalid"]
  * error flows  : TRANSITION["road_tp"|"road_to_marking"|...]  (used by 05)
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------- palette ----------------------------------- #
# Class identity (kept stable across the whole thesis). RULE: blue=road,
# red=marking, green=other -- used for ANY figure that colours by class.
ROAD = "#1f77b4"      # blue
MARKING = "#d62728"   # red
OTHER = "#2ca02c"     # green
CLASS = {"road": ROAD, "marking": MARKING, "other": OTHER}

# Run identity (distinct from class colours; never co-occur with class colours).
CANDIDATE = "#1b9e77"  # teal  — the run under analysis
BASELINE = "#8c6bb1"   # purple — the (primary) run it is compared against
# When several baselines are compared at once (e.g. G2 vs F0 vs D0), each baseline
# gets a distinct, stable colour by its order in BASELINE_LABELS. The candidate is
# always teal and drawn thicker so it stands out.
BASELINE_CYCLE = ["#8c6bb1", "#9c755f", "#bab0ac", "#7f7f7f"]  # purple, brown, taupe, grey


def run_colors(candidate: str, baselines) -> dict:
    """Map each run label to its colour: candidate = teal, baselines by order."""
    m = {candidate: CANDIDATE}
    for i, b in enumerate(baselines):
        m[b] = BASELINE_CYCLE[i % len(BASELINE_CYCLE)]
    return m


# --------------------- thesis-facing run display names --------------------- #
# Plots show these descriptive names instead of internal run codes (D0/F0/G2/H0),
# because the thesis should not reference runs by alphabet codes. The names build
# up incrementally so the reader sees what each run adds:
#   LiDAR  ->  + RGB  ->  + Lovász loss  ->  + jitter augmentation.
# Edit here only — every figure reads through display_name(). CSV/JSON/file names
# keep the short codes for stability; only rendered figure text changes.
DISPLAY_NAMES = {
    "D0": "LiDAR",
    "F0": "LiDAR + RGB",
    "G0": "LiDAR + RGB + Lovász",
    "G1": "LiDAR + RGB + Lovász",
    "G2": "LiDAR + RGB + Lovász",
    "H0": "LiDAR + RGB + Lovász + Jitter",
}


def display_name(label: str) -> str:
    """Thesis-facing name for a run code (falls back to the code if unmapped)."""
    return DISPLAY_NAMES.get(label, label)


# Marking-metric family. IoU ties to the marking colour; the other three are
# mutually distinct and do NOT reuse the candidate/baseline hues.
METRIC = {
    "iou": MARKING,       # red (IoU is the headline marking metric)
    "precision": "#ff7f0e",  # orange  (decoupled from class blue/green)
    "recall": "#9467bd",     # purple
    "f1": "#e6ab02",         # gold
}

# Calibration: predicted/true marking ratio has ONE colour everywhere it is a
# single series; the perfect-calibration reference line is always black.
PRED_TRUE = "#7570b3"  # indigo
CALIBRATED = "#000000"

# RGB-validity strata (one fixed colour each, reused in every stratified figure).
RGB = {"all": "#666666", "rgb_valid": "#1f77b4", "rgb_invalid": "#ff7f0e"}

# Error-flow groups for the RGB-shortcut fingerprint (05). road->marking is the
# headline false positive and owns the one red in the suite.
TRANSITION = {
    "road_tp": "#666666",
    "road_to_marking": "#d62728",   # red — headline FP
    "marking_tp": "#2ca02c",
    "marking_to_road": "#ff7f0e",
    "other_to_marking": "#9467bd",
}

# Structural / neutral.
REFERENCE = "#000000"   # best-epoch / zero / guide lines
TRAIN = "#666666"       # train series in loss plots (val uses CANDIDATE)
GRID = "#d0d0d0"
NEUTRAL = "#999999"


def apply() -> None:
    """Set uniform, thesis-grade typography and figure defaults (call once)."""
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelsize": 13,
        "axes.labelpad": 6,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.framealpha": 0.92,
        "legend.facecolor": "white",
        "legend.edgecolor": "#bbbbbb",
        "legend.borderpad": 0.6,
        "axes.axisbelow": True,           # grid/markers behind data lines
        "lines.linewidth": 2.1,
        "lines.markersize": 5,
    })


def style_axes(ax, grid_axis: str = "y") -> None:
    """Hide the top/right spines and add a soft grid (consistent everywhere)."""
    ax.grid(True, axis=grid_axis, color=GRID, alpha=0.5, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def legend(ax, loc: str = "best", ncol: int = 1, **kw):
    """Readable legend with a white box that sits above the data (axisbelow)."""
    leg = ax.legend(loc=loc, ncol=ncol, **kw)
    if leg is not None:
        leg.set_zorder(6)
        leg.get_frame().set_linewidth(0.8)
    return leg


def headroom(ax, frac: float = 0.16, bottom: float | None = None) -> None:
    """Add vertical head-room so a 'best'/upper legend never sits on the data."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo if bottom is None else bottom, hi + (hi - lo) * frac)


def legend_clear(ax, loc: str = "upper left", frac: float = 0.18, ncol: int = 1,
                 handles=None, labels=None, **kw):
    """Draw a legend that does not sit on the data: first add top head-room,
    then place the legend in an upper corner. Use this instead of bare
    ax.legend() whenever lines/bars would otherwise cross the legend box."""
    headroom(ax, frac=frac)
    if handles is not None:
        leg = ax.legend(handles, labels, loc=loc, ncol=ncol, **kw)
    else:
        leg = ax.legend(loc=loc, ncol=ncol, **kw)
    if leg is not None:
        leg.set_zorder(6)
        leg.get_frame().set_linewidth(0.8)
    return leg


def legend_outside(ax, **kw):
    """Place the legend just outside the axes (right side) so it can never
    overlap the data. Use for dense bar/multi-series panels."""
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    borderaxespad=0.0, **kw)
    if leg is not None:
        leg.set_zorder(6)
        leg.get_frame().set_linewidth(0.8)
    return leg
