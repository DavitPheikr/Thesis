#!/usr/bin/env python
"""Create the core run-level analysis package for G1_schedule_extend.

G1 is a continuation copy of G0, resumed from epoch 25 and trained to epoch 35
with the original G0 settings unchanged. This script uses saved training
artifacts only. It does not run inference.

Class order in confusion matrices:
    0 = road, 1 = marking, 2 = other

For road_marking3, eval_history lane_* columns are marking_* metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


from _suite_paths import (  # noqa: E402
    REPO as PROJECT_ROOT, RUN_DIR as CAND_RUN_DIR, ANALYSIS_OUT as OUT_DIR,
    MILESTONE, RUN_NAME, LINEAGE, LINEAGE_DIRS, CANDIDATE_LABEL, BASELINE_LABEL,
    BASELINE_LABELS, BASELINE_DIR,
)

# The current run's run dir (candidate). The comparison plots compare the
# candidate against BASELINE_LABEL; the full lineage drives the summary table.
G1_RUN_DIR = CAND_RUN_DIR  # kept as an alias used below

CLASS_NAMES = ("road", "marking", "other")
D0_OFFICIAL_EPOCH = 18
LOVASZ_LAMBDA = 0.5
COMPOSE_ATOL = 1e-4
COMPOSE_RTOL = 1e-3


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= COMPOSE_ATOL + COMPOSE_RTOL * abs(float(b))


def load_history(run_dir: Path) -> pd.DataFrame:
    path = require_file(run_dir / "eval_history.csv")
    df = pd.read_csv(path)
    required = {
        "epoch",
        "train_loss",
        "val_loss",
        "miou",
        "road_iou",
        "lane_iou",
        "other_iou",
        "lane_precision",
        "lane_recall",
        "lane_f1",
        "lr",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")
    return df


def load_components(run_dir: Path) -> pd.DataFrame:
    path = require_file(run_dir / "loss_components.csv")
    df = pd.read_csv(path)
    required = {
        "epoch",
        "train_ce_loss",
        "train_lovasz_loss",
        "train_total_loss",
        "val_ce_loss",
        "val_lovasz_loss",
        "val_total_loss",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")
    return df


def validate_loss_components(run_dir: Path, history: pd.DataFrame) -> dict[str, Any]:
    comp = load_components(run_dir).sort_values("epoch").reset_index(drop=True)
    hist = history.sort_values("epoch").reset_index(drop=True)
    if len(comp) != len(hist):
        raise RuntimeError(f"{run_dir} loss_components rows {len(comp)} != history rows {len(hist)}")
    if not np.array_equal(comp["epoch"].to_numpy(), hist["epoch"].to_numpy()):
        raise RuntimeError(f"{run_dir} loss component epochs do not match eval_history")

    max_resid = {
        "train_total_vs_history": 0.0,
        "val_total_vs_history": 0.0,
        "train_total_vs_components": 0.0,
        "val_total_vs_components": 0.0,
    }
    for _, row in comp.iterrows():
        epoch = int(row["epoch"])
        hrow = hist[hist["epoch"] == epoch].iloc[0]
        train_total = float(row["train_ce_loss"]) + LOVASZ_LAMBDA * float(row["train_lovasz_loss"])
        val_total = float(row["val_ce_loss"]) + LOVASZ_LAMBDA * float(row["val_lovasz_loss"])
        checks = [
            ("train_total_vs_history", float(row["train_total_loss"]), float(hrow["train_loss"])),
            ("val_total_vs_history", float(row["val_total_loss"]), float(hrow["val_loss"])),
            ("train_total_vs_components", float(row["train_total_loss"]), train_total),
            ("val_total_vs_components", float(row["val_total_loss"]), val_total),
        ]
        for key, actual, expected in checks:
            max_resid[key] = max(max_resid[key], abs(actual - expected))
            if not close(actual, expected):
                raise RuntimeError(
                    f"{run_dir.name} epoch {epoch}: {key} failed; "
                    f"{actual:.8f} != {expected:.8f}"
                )
    return {"rows": int(len(comp)), "lovasz_lambda": LOVASZ_LAMBDA, **max_resid}


def validate_artifacts(run_dir: Path, run_name: str, require_components: bool) -> dict[str, Any]:
    hist = load_history(run_dir)
    expected = len(hist)
    counts = {
        "run": run_name,
        "eval_history_rows": expected,
        "eval_json_count": len(list(run_dir.glob("eval_epoch_*.json"))),
        "confusion_count": len(list(run_dir.glob("confusion_epoch_*.npy"))),
        "checkpoint_count": len(list((run_dir / "checkpoints").glob("ckpt_epoch_*.pth"))),
        "loss_components_rows": -1,
    }
    # Non-fatal: older baseline runs (D0/F0) may have slightly different artifact
    # bookkeeping. We only need eval_history + the best/final confusion npys, which
    # are checked where they are actually loaded. Record mismatches as a warning.
    mismatches = [k for k in ("eval_json_count", "confusion_count", "checkpoint_count")
                  if int(counts[k]) != expected]
    if mismatches:
        counts["artifact_mismatches"] = ";".join(f"{k}={counts[k]}" for k in mismatches)
        print(f"[warn] {run_name}: artifact counts != eval_history rows ({expected}): "
              f"{counts['artifact_mismatches']}")
    else:
        counts["artifact_mismatches"] = ""
    if require_components:
        comp = load_components(run_dir)
        counts["loss_components_rows"] = len(comp)
        if len(comp) != expected:
            raise RuntimeError(f"{run_name}: loss_components rows={len(comp)} but eval_history rows={expected}")
    return counts


def validate_g1_config() -> dict[str, Any]:
    cfg = yaml.safe_load(require_file(G1_RUN_DIR / "config_snapshot.yml").read_text())
    dataset = cfg.get("dataset", {})
    model = cfg.get("model", {})
    pipeline = cfg.get("pipeline", {})
    scheduler = pipeline.get("scheduler", {}) or {}
    loss = pipeline.get("loss", {}) or {}
    checks = {
        "feature_mode": dataset.get("feature_mode"),
        "camera_name": dataset.get("camera_name"),
        "in_channels": model.get("in_channels"),
        "dim_features": model.get("dim_features"),
        "num_layers": model.get("num_layers"),
        "num_points": model.get("num_points"),
        "num_neighbors": model.get("num_neighbors"),
        "class_weights": [float(v) for v in dataset.get("class_weights", [])],
        "pin_memory": pipeline.get("pin_memory"),
        "num_workers": pipeline.get("num_workers"),
        "watch_metric": scheduler.get("watch_metric"),
        "scheduler_patience": scheduler.get("patience"),
        "loss_name": loss.get("name"),
        "lovasz_lambda": loss.get("lovasz_lambda"),
        "lovasz_classes": loss.get("lovasz_classes"),
    }
    expected = {
        "feature_mode": "intensity_rgb_front",
        "camera_name": "front_camera",
        "in_channels": 8,
        "dim_features": 16,
        "num_layers": 3,
        "num_points": 32768,
        "num_neighbors": 24,
        "class_weights": [119562394.0, 14344000.0, 173473484.0],
        "pin_memory": True,
        "num_workers": 0,
        "watch_metric": "marking_iou",
        "scheduler_patience": 6,
        "loss_name": "weighted_ce_lovasz",
        "lovasz_lambda": 0.5,
        "lovasz_classes": "present",
    }
    mismatches = {}
    for key, expected_value in expected.items():
        actual = checks[key]
        if key == "class_weights":
            if len(actual) != len(expected_value) or any(abs(a - b) > 1e-6 for a, b in zip(actual, expected_value)):
                mismatches[key] = (actual, expected_value)
        elif key == "lovasz_lambda":
            if actual is None or abs(float(actual) - float(expected_value)) > 1e-9:
                mismatches[key] = (actual, expected_value)
        elif actual != expected_value:
            mismatches[key] = (actual, expected_value)
    # Non-fatal in the reusable suite: a new run (e.g. G2 at 100 epochs) may
    # legitimately differ. Mismatches are recorded and surfaced in analysis.md.
    checks["config_mismatches"] = {k: list(v) for k, v in mismatches.items()}
    if mismatches:
        print(f"[01] WARNING: candidate config differs from the G1 reference: {mismatches}")
    return checks


def load_confusion(run_dir: Path, epoch: int) -> np.ndarray:
    cm = np.load(require_file(run_dir / f"confusion_epoch_{epoch:03d}.npy"))
    if cm.shape != (3, 3):
        raise RuntimeError(f"{run_dir.name} epoch {epoch} confusion shape {cm.shape}; expected (3, 3)")
    return cm.astype(np.int64, copy=False)


def best_row(df: pd.DataFrame) -> pd.Series:
    return df.loc[df["lane_iou"].idxmax()]


def row_metrics(run: str, label: str, row: pd.Series, comp: pd.DataFrame | None) -> dict[str, Any]:
    epoch = int(row["epoch"])
    if comp is not None:
        crow = comp[comp["epoch"] == epoch].iloc[0]
        train_ce = float(crow["train_ce_loss"])
        val_ce = float(crow["val_ce_loss"])
        train_lovasz = float(crow["train_lovasz_loss"])
        val_lovasz = float(crow["val_lovasz_loss"])
    else:
        train_ce = float(row["train_loss"])
        val_ce = float(row["val_loss"])
        train_lovasz = 0.0
        val_lovasz = 0.0
    return {
        "run": run,
        "label": label,
        "epoch": epoch,
        "train_loss": float(row["train_loss"]),
        "val_loss": float(row["val_loss"]),
        "train_ce": train_ce,
        "val_ce": val_ce,
        "train_lovasz": train_lovasz,
        "val_lovasz": val_lovasz,
        "miou": float(row["miou"]),
        "road_iou": float(row["road_iou"]),
        "marking_iou": float(row["lane_iou"]),
        "other_iou": float(row["other_iou"]),
        "marking_precision": float(row["lane_precision"]),
        "marking_recall": float(row["lane_recall"]),
        "marking_f1": float(row["lane_f1"]),
        "lr": float(row["lr"]),
    }


def pred_true_rows(run: str, label: str, epoch: int, cm: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for idx, cls in enumerate(CLASS_NAMES):
        true_count = int(cm[idx, :].sum())
        pred_count = int(cm[:, idx].sum())
        rows.append(
            {
                "run": run,
                "label": label,
                "epoch": epoch,
                "class": cls,
                "true_count": true_count,
                "pred_count": pred_count,
                "pred_true_ratio": float("nan") if true_count == 0 else pred_count / true_count,
            }
        )
    return rows


def confusion_rows(run: str, label: str, epoch: int, cm: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    row_sums = cm.sum(axis=1)
    for i, true_cls in enumerate(CLASS_NAMES):
        for j, pred_cls in enumerate(CLASS_NAMES):
            rows.append(
                {
                    "run": run,
                    "label": label,
                    "epoch": epoch,
                    "true_class": true_cls,
                    "pred_class": pred_cls,
                    "error_type": f"{true_cls}->{pred_cls}",
                    "count": int(cm[i, j]),
                    "true_class_total": int(row_sums[i]),
                    "row_percent": 0.0 if row_sums[i] == 0 else 100.0 * float(cm[i, j]) / float(row_sums[i]),
                }
            )
    return rows


def all_epoch_pred_true(run: str, run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.glob("confusion_epoch_*.npy")):
        epoch = int(path.stem.split("_")[-1])
        cm = np.load(path)
        if cm.shape != (3, 3):
            raise RuntimeError(f"{path} shape {cm.shape}; expected (3, 3)")
        rows.extend(pred_true_rows(run, "epoch", epoch, cm))
    return pd.DataFrame(rows)


def lr_events(run: str, hist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    hist = hist.sort_values("epoch").reset_index(drop=True)
    for i in range(1, len(hist)):
        before = float(hist.loc[i - 1, "lr"])
        after = float(hist.loc[i, "lr"])
        if abs(before - after) > 1e-12:
            rows.append(
                {
                    "run": run,
                    "epoch": int(hist.loc[i, "epoch"]),
                    "lr_before": before,
                    "lr_after": after,
                    "factor": after / before if before else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def loss_component_summary(hist: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    merged = hist[["epoch", "lane_iou", "lane_precision", "lane_recall", "lane_f1", "miou", "lr"]].merge(comp, on="epoch")
    merged["train_lovasz_scaled"] = LOVASZ_LAMBDA * merged["train_lovasz_loss"]
    merged["val_lovasz_scaled"] = LOVASZ_LAMBDA * merged["val_lovasz_loss"]
    merged["train_lovasz_total_share"] = merged["train_lovasz_scaled"] / merged["train_total_loss"]
    merged["val_lovasz_total_share"] = merged["val_lovasz_scaled"] / merged["val_total_loss"]
    return merged


def loss_alignment(summary: pd.DataFrame) -> pd.DataFrame:
    target = summary["lane_iou"]
    rows = []
    for col, label in [
        ("val_ce_loss", "validation CE"),
        ("val_lovasz_loss", "validation Lovasz raw"),
        ("val_lovasz_scaled", "validation Lovasz scaled"),
        ("val_total_loss", "validation total"),
        ("train_total_loss", "training total"),
    ]:
        rows.append(
            {
                "signal": label,
                "column": col,
                "pearson_corr_with_marking_iou": float(summary[col].corr(target)),
                "argmin_epoch": int(summary.loc[summary[col].idxmin(), "epoch"]),
                "argmax_marking_iou_epoch": int(summary.loc[summary["lane_iou"].idxmax(), "epoch"]),
            }
        )
    return pd.DataFrame(rows)


def write_visual_notes(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        "\n".join(
            [
                f"# {CANDIDATE_LABEL} Visual Inspection Notes",
                "",
                "Use this file to record manual observations from front-camera overlay PNGs.",
                "",
                "Overlay location:",
                f"`logs/{MILESTONE}/run_analysis/front_camera_predictions/{RUN_NAME}/`",
                "",
                "Suggested first sequences: `054`, `123`, `124`, `037`, `034`.",
                "",
                "## Checklist",
                "",
                "- Are road-marking predictions visually aligned with paint?",
                "- Are false positives mostly bright road, lane boundaries, shadows, or projection artifacts?",
                "- Are misses mostly faint/occluded/far markings?",
                f"- Does behaviour change after the LR drop compared with {BASELINE_LABEL}?",
                "",
                "## Sequence Notes",
                "",
                "### 054",
                "",
                "### 123",
                "",
                "### 124",
                "",
                "### 037",
                "",
                "### 034",
                "",
            ]
        )
    )


def write_analysis_md(
    summary: pd.DataFrame,
    best_vs_final: pd.DataFrame,
    pred_true: pd.DataFrame,
    events: pd.DataFrame,
    config_checks: dict[str, Any],
    loss_check: dict[str, Any],
) -> None:
    def s(run: str, label: str) -> pd.Series:
        return summary[(summary["run"] == run) & (summary["label"] == label)].iloc[0]

    def ratio(run: str, epoch: float) -> float:
        m = pred_true[(pred_true["run"] == run) & (pred_true["epoch"] == int(epoch))
                      & (pred_true["class"] == "marking")]["pred_true_ratio"]
        return float(m.iloc[0]) if len(m) else float("nan")

    base_b = s(BASELINE_LABEL, "best")
    cand_b = s(CANDIDATE_LABEL, "best")
    cand_f = s(CANDIDATE_LABEL, "final")
    cand_ratio = ratio(CANDIDATE_LABEL, cand_b["epoch"])
    base_ratio = ratio(BASELINE_LABEL, base_b["epoch"])
    residual = loss_check.get("val_total_vs_components")

    lines = [
        f"# {CANDIDATE_LABEL} Run Analysis",
        "",
        f"Candidate run **{CANDIDATE_LABEL}** compared against baseline **{BASELINE_LABEL}**. "
        "All runs are single-seed; see `single_seed_limitation.md`.",
        "",
        "## Validation",
        "",
        f"- feature mode: `{config_checks.get('feature_mode')}`",
        f"- loss: `{config_checks.get('loss_name')}` with lambda `{config_checks.get('lovasz_lambda')}`",
        f"- scheduler patience: `{config_checks.get('scheduler_patience')}`",
        (f"- loss composition residual, val: `{residual:.2e}`" if residual is not None
         else "- loss composition residual, val: n/a"),
        f"- config differences vs G1 reference: `{config_checks.get('config_mismatches', {})}`",
        "",
        "## Headline",
        "",
        "| run | epoch | marking IoU | F1 | precision | recall | mIoU | pred/true marking | LR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {BASELINE_LABEL} best | {int(base_b['epoch'])} | {base_b['marking_iou']:.6f} | {base_b['marking_f1']:.6f} | {base_b['marking_precision']:.6f} | {base_b['marking_recall']:.6f} | {base_b['miou']:.6f} | {base_ratio:.3f} | {base_b['lr']:.6f} |",
        f"| {CANDIDATE_LABEL} best | {int(cand_b['epoch'])} | {cand_b['marking_iou']:.6f} | {cand_b['marking_f1']:.6f} | {cand_b['marking_precision']:.6f} | {cand_b['marking_recall']:.6f} | {cand_b['miou']:.6f} | {cand_ratio:.3f} | {cand_b['lr']:.6f} |",
        f"| {CANDIDATE_LABEL} final | {int(cand_f['epoch'])} | {cand_f['marking_iou']:.6f} | {cand_f['marking_f1']:.6f} | {cand_f['marking_precision']:.6f} | {cand_f['marking_recall']:.6f} | {cand_f['miou']:.6f} | | {cand_f['lr']:.6f} |",
        "",
        "## LR Events",
        "",
    ]
    if events.empty:
        lines.append("No LR changes were detected.")
    else:
        lines.extend(["| epoch | LR before | LR after | factor |", "| ---: | ---: | ---: | ---: |"])
        for _, row in events.iterrows():
            lines.append(f"| {int(row['epoch'])} | {row['lr_before']:.6f} | {row['lr_after']:.6f} | {row['factor']:.3f} |")
    lines.extend(
        [
            "",
            "## Best vs Final Drift",
            "",
            "| run | metric | best epoch | final epoch | best | final | final-best |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in best_vs_final.iterrows():
        lines.append(
            f"| {row['run']} | {row['metric']} | {int(row['best_epoch'])} | "
            f"{int(row['final_epoch'])} | {row['best']:.6f} | {row['final']:.6f} | "
            f"{row['final_minus_best']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Output Map",
            "",
            "- Core plots: `plots/`",
            "- Sampled best-checkpoint analysis: `sampled_error_analysis_epochXX/`",
            "- Visual inspection notes: `visual_inspection_notes.md`",
            "",
        ]
    )
    (OUT_DIR / "analysis.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dirs = LINEAGE_DIRS
    cand, base = CANDIDATE_LABEL, BASELINE_LABEL

    artifact_counts = pd.DataFrame(
        [validate_artifacts(dirs[lab], lab, has_lov) for lab, _r, has_lov, _role, _ep in LINEAGE]
    )
    artifact_counts.to_csv(OUT_DIR / "artifact_counts.csv", index=False)

    config_checks = validate_g1_config()
    histories = {lab: load_history(dirs[lab]) for lab, *_ in LINEAGE}
    components = {lab: load_components(dirs[lab]) for lab, _r, has_lov, *_ in LINEAGE if has_lov}
    for lab, _r, has_lov, *_ in LINEAGE:
        if has_lov:
            validate_loss_components(dirs[lab], histories[lab])
    cand_loss_check = validate_loss_components(dirs[cand], histories[cand]) if cand in components else {}

    summary_rows = []
    for lab, _r, has_lov, role, official_ep in LINEAGE:
        comp = components.get(lab)
        hist = histories[lab]
        if role == "official":
            summary_rows.append(
                row_metrics(lab, "official", hist[hist["epoch"] == official_ep].iloc[0], comp))
        else:
            summary_rows.append(row_metrics(lab, "best", best_row(hist), comp))
            summary_rows.append(row_metrics(lab, "final", hist.iloc[-1], comp))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    cand_best_epoch = int(summary[(summary["run"] == cand) & (summary["label"] == "best")]["epoch"].iloc[0])
    if not (CAND_RUN_DIR / "checkpoints" / f"ckpt_epoch_{cand_best_epoch:05d}.pth").exists():
        raise FileNotFoundError(f"{cand} best checkpoint missing for epoch {cand_best_epoch}")

    pair = [*BASELINE_LABELS, cand]  # all baselines then candidate (all best_final runs)

    drift_rows = []
    for run in pair:
        best = summary[(summary["run"] == run) & (summary["label"] == "best")].iloc[0]
        final = summary[(summary["run"] == run) & (summary["label"] == "final")].iloc[0]
        for metric in ("marking_iou", "marking_f1", "marking_precision", "marking_recall", "miou"):
            drift_rows.append({
                "run": run, "metric": metric,
                "best_epoch": int(best["epoch"]), "final_epoch": int(final["epoch"]),
                "best": float(best[metric]), "final": float(final[metric]),
                "final_minus_best": float(final[metric] - best[metric]),
            })
    best_vs_final = pd.DataFrame(drift_rows)
    best_vs_final.to_csv(OUT_DIR / "best_vs_final.csv", index=False)

    epoch_rows = []
    for run in pair:
        hist = histories[run].copy()
        comp = components.get(run)
        merged = hist.merge(comp, on="epoch", how="left") if comp is not None else hist
        merged.insert(0, "run", run)
        epoch_rows.append(merged)
    pd.concat(epoch_rows, ignore_index=True).to_csv(OUT_DIR / "comparison_epoch_metrics.csv", index=False)

    pred_true = pd.concat([all_epoch_pred_true(run, dirs[run]) for run in pair], ignore_index=True)
    pred_true.to_csv(OUT_DIR / "pred_true_ratio_by_epoch.csv", index=False)

    events = pd.concat([lr_events(run, histories[run]) for run in pair], ignore_index=True)
    events.to_csv(OUT_DIR / "lr_events.csv", index=False)

    confusion = []
    for run in pair:
        for label in ("best", "final"):
            epoch = int(summary[(summary["run"] == run) & (summary["label"] == label)]["epoch"].iloc[0])
            confusion.extend(confusion_rows(run, label, epoch, load_confusion(dirs[run], epoch)))
    pd.DataFrame(confusion).to_csv(OUT_DIR / "confusion_breakdown.csv", index=False)

    if cand in components:
        loss_summary = loss_component_summary(histories[cand], components[cand])
        loss_summary.to_csv(OUT_DIR / "loss_component_summary.csv", index=False)
        loss_alignment(loss_summary).to_csv(OUT_DIR / "loss_alignment.csv", index=False)

    write_visual_notes(OUT_DIR / "visual_inspection_notes.md")
    write_analysis_md(summary, best_vs_final, pred_true,
                      events[events["run"] == cand], config_checks, cand_loss_check)

    print(f"candidate_best_epoch {cand_best_epoch}")
    print(f"wrote {OUT_DIR}")
    print("script_status PASS")


if __name__ == "__main__":
    main()
