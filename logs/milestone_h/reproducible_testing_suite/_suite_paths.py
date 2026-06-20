"""Central paths + per-run config for the reproducible testing suite.

This is the ONE file to edit when you copy the suite to a new run (G2, H0, ...).
Everything else discovers the repo root robustly and reads the knobs below, so
re-targeting the suite is a single-file change.

`repo_root()` walks upward until it finds the repo (the dir containing both
`src/thesis_pipeline/` and `logs/`), so the scripts work no matter where the
suite directory is placed.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "src" / "thesis_pipeline").is_dir() and (parent / "logs").is_dir():
            return parent
    raise RuntimeError(
        "Repo root not found (looked for a parent dir containing src/thesis_pipeline/ and logs/)."
    )


REPO = repo_root()

# ============================ PER-RUN CONFIG ============================
# Edit these for a new run (e.g. G2, H0). Names follow the existing repo layout:
#   logs/<MILESTONE>/runs/<RUN_NAME>/            (checkpoints, eval_history.csv, ...)
#   logs/<MILESTONE>/configs/<CONFIG_FILE>       (the training config yaml)
#   logs/<MILESTONE>/run_analysis/<RUN_NAME>/    (where this suite writes outputs)
#   logs/<MILESTONE>/cache/<CACHE_NAME>          (the feature cache used for inference)
MILESTONE = "milestone_h"


def _autodetect_run_name(milestone: str) -> str:
    """Return the single run dir under logs/<milestone>/runs/ that has an
    eval_history.csv. Returns "" if zero or more than one are found (then set
    RUN_NAME manually below). Lets the suite 'just work' on the server where the
    H0 run exists, without hard-coding its exact directory name."""
    runs_dir = REPO / f"logs/{milestone}/runs"
    if not runs_dir.is_dir():
        return ""
    cands = sorted(p.parent.name for p in runs_dir.glob("*/eval_history.csv"))
    return cands[0] if len(cands) == 1 else ""


# H0 run dir is auto-detected (the single 100-epoch jitter run). If autodetect
# returns "" (run not present locally yet, or >1 run), the fallback name is used —
# edit it to the real dir name from `ls logs/milestone_h/runs/`.
RUN_NAME = _autodetect_run_name(MILESTONE) or "H0_rgb_jitter"
# H0 reuses the G1/G2 Lovasz config's model+dataset sections (PandaSetFFLane3 /
# RandLANet / 8-ch / 3-class / weighted_ce_lovasz). The H0 config additionally has a
# pipeline.rgb_jitter block that is TRAIN-ONLY and is NOT read by the inference
# engine (the suite evaluates clean, un-jittered validation — verified). So H0 is
# compared to G2 on identical eval conditions.
CONFIG_FILE = "h0_rgb_jitter.yml"
# Cache name is vestigial for inference: the engine sets use_cache=False and
# recomputes features fresh, so this path does not need to exist.
CACHE_NAME = "G0_rgb_lovasz_v1"

# Best-checkpoint rule: max of this per-epoch metric column in eval_history.csv
# ("lane_iou" is the marking-IoU alias used throughout the repo).
BEST_EPOCH_METRIC = "lane_iou"

# ---- Comparison / lineage (steps 01, 03) ----
# CANDIDATE = the run being analysed (this run, = RUN_NAME). BASELINE_LABELS = the
# run(s) it is compared against in the comparison plots, drawn in list order
# (primary first). Each must be a "best_final" row in LINEAGE below.
# H0: candidate = H0, primary baseline = G2 (best model so far), secondary = F0.
CANDIDATE_LABEL = "H0"
BASELINE_LABELS = ["G2", "F0"]   # H0 vs G2 (best, the model to beat) and vs F0 (pre-Lovasz)
BASELINE_LABEL = BASELINE_LABELS[0]  # primary baseline (back-compat; used for headline deltas)
# Ordered runs shown in the summary table. Each:
#   (label, run-dir relative to repo, has_lovasz_components, role, official_epoch)
#   role "official"   -> a single official epoch (uses official_epoch)
#   role "best_final" -> best + final rows (official_epoch ignored)
# The CANDIDATE and every BASELINE_LABELS entry must be present with role
# "best_final"; for a Lovasz run set has_lovasz=True. (G0/G1 omitted on purpose:
# not the comparison of interest and their raw run dirs are not always present.)
LINEAGE = [
    ("D0", "logs/milestone_d/runs/D0_weighted_ce_25ep",      False, "best_final", None),
    ("F0", "logs/milestone_f/runs/F0_rgb_soft_weights",      False, "best_final", None),
    ("G2", "logs/milestone_g/runs/G2_schedule_extend_100",   True,  "best_final", None),
    ("H0", f"logs/milestone_h/runs/{RUN_NAME}",              True,  "best_final", None),
]
# =======================================================================

RUN_DIR = REPO / f"logs/{MILESTONE}/runs/{RUN_NAME}"
CONFIG_YAML = REPO / f"logs/{MILESTONE}/configs/{CONFIG_FILE}"
ANALYSIS_OUT = REPO / f"logs/{MILESTONE}/run_analysis/{RUN_NAME}"
CACHE_DIR = REPO / f"logs/{MILESTONE}/cache/{CACHE_NAME}"
TEST_RESULTS = REPO / f"logs/{MILESTONE}/test_results/{RUN_NAME}"
LINEAGE_DIRS = {label: REPO / rel for label, rel, _h, _r, _e in LINEAGE}
BASELINE_DIRS = {label: LINEAGE_DIRS[label] for label in BASELINE_LABELS}
BASELINE_DIR = BASELINE_DIRS[BASELINE_LABEL]
