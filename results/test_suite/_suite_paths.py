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
MILESTONE = "milestone_g"
RUN_NAME = "G2_schedule_extend_100"
# G2 reuses the G1 config for inference: model + dataset sections are identical
# (verified PandaSetFFLane3 / RandLANet / 8-ch / 3-class / weighted_ce_lovasz).
# Only model+dataset matter for inference; cache/test paths are overridden below.
CONFIG_FILE = "g1_schedule_extend.yml"
# Shared PandaSet feature cache (identical features across G0/G1/G2).
CACHE_NAME = "G0_rgb_lovasz_v1"

# Best-checkpoint rule: max of this per-epoch metric column in eval_history.csv
# ("lane_iou" is the marking-IoU alias used throughout the repo).
BEST_EPOCH_METRIC = "lane_iou"

# ---- Comparison / lineage (steps 01, 03) ----
# CANDIDATE = the run being analysed (this run, = RUN_NAME). BASELINE_LABELS = the
# run(s) it is compared against in the comparison plots, drawn in list order
# (primary first). Each must be a "best_final" row in LINEAGE below.
# Example for G2:  CANDIDATE_LABEL="G2", BASELINE_LABELS=["F0", "D0"].
CANDIDATE_LABEL = "G2"
BASELINE_LABELS = ["F0", "D0"]   # G2 vs F0 (pre-Lovasz) and vs D0 (no-RGB baseline)
BASELINE_LABEL = BASELINE_LABELS[0]  # primary baseline (back-compat; used for headline deltas)
# Ordered runs shown in the summary table. Each:
#   (label, run-dir relative to repo, has_lovasz_components, role, official_epoch)
#   role "official"   -> a single official epoch (uses official_epoch)
#   role "best_final" -> best + final rows (official_epoch ignored)
# The CANDIDATE and every BASELINE_LABELS entry must be present with role
# "best_final"; for a Lovasz run set has_lovasz=True.
LINEAGE = [
    ("D0", "logs/milestone_d/runs/D0_weighted_ce_25ep",      False, "best_final", None),
    ("F0", "logs/milestone_f/runs/F0_rgb_soft_weights",      False, "best_final", None),
    ("G0", "logs/milestone_g/runs/G0_rgb_lovasz",            True,  "best_final", None),
    ("G1", "logs/milestone_g/runs/G1_schedule_extend",       True,  "best_final", None),
    ("G2", "logs/milestone_g/runs/G2_schedule_extend_100",   True,  "best_final", None),
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
