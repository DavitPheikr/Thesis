#!/usr/bin/env python
"""Zero-GPU ReduceLROnPlateau replay for the G0 marking-IoU trajectory.

Replays G0's observed per-epoch marking IoU through the *real* PyTorch
ReduceLROnPlateau (the same class and config the runner uses) to answer:

  - With the actual patience=6, at which epoch would/did the LR first drop?
  - Would a more aggressive patience (5, 4, 3) drop BEFORE the IoU peak (ep18),
    which would risk lowering the peak?

Faithfulness: the runner feeds the scheduler a `smoothing_window`-epoch running
MEAN of the raw marking IoU (train_milestone_d.py `_scheduler_metric_from_row`),
then calls `scheduler.step(smoothed)`. We replicate exactly that, using a dummy
optimizer so PyTorch's internal num_bad_epochs / cooldown logic is authoritative.

IMPORTANT CAVEAT (interpretation): this replays the *observed* G0 curve. If LR
had actually dropped earlier, training would have diverged and the curve would
differ. So patience=6 gives the true drop epoch for the real run; patience<6 is a
"what the scheduler would have done on this same curve" estimate, not a forecast
of a different run.

Reads:  logs/milestone_g/run_analysis/G0_rgb_lovasz/loss_component_summary.csv
Writes: logs/milestone_g/run_analysis/G0_rgb_lovasz/scheduler_replay.csv
        logs/milestone_g/run_analysis/G0_rgb_lovasz/scheduler_replay.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "logs/milestone_g/run_analysis/G0_rgb_lovasz"

# Must match logs/milestone_g/configs/g0_rgb_lovasz.yml pipeline.scheduler
BASE_LR = 0.0014
FACTOR = 0.5
THRESHOLD = 0.01
THRESHOLD_MODE = "abs"
COOLDOWN = 1
MIN_LR = 1e-6
SMOOTHING_WINDOW = 3
PATIENCES = (6, 5, 4, 3)
IOU_PEAK_EPOCH = 18  # observed argmax(marking IoU)


def load_marking_iou() -> pd.DataFrame:
    path = OUT_DIR / "loss_component_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path).sort_values("epoch").reset_index(drop=True)
    if "marking_iou" not in df.columns:
        raise ValueError(f"{path} has no marking_iou column")
    return df[["epoch", "marking_iou"]]


def smoothed_series(raw: list[float], window: int) -> list[float]:
    """Running mean of the last `window` values, matching the runner."""
    out = []
    for t in range(len(raw)):
        w = raw[max(0, t - window + 1) : t + 1]
        out.append(float(np.mean(w)))
    return out


def replay(raw_iou: list[float], smoothed: list[float], patience: int) -> dict:
    dummy = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([dummy], lr=BASE_LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=FACTOR, patience=patience,
        threshold=THRESHOLD, threshold_mode=THRESHOLD_MODE,
        cooldown=COOLDOWN, min_lr=MIN_LR,
    )
    lrs = []
    drop_epochs = []
    prev_lr = BASE_LR
    for ep_idx, sm in enumerate(smoothed, start=1):
        sched.step(sm)
        lr = opt.param_groups[0]["lr"]
        lrs.append(lr)
        if lr < prev_lr - 1e-12:
            drop_epochs.append(ep_idx)
        prev_lr = lr
    first_drop = drop_epochs[0] if drop_epochs else None
    return {
        "patience": patience,
        "first_drop_epoch": first_drop,
        "all_drop_epochs": drop_epochs,
        "final_lr": lrs[-1],
        "lrs": lrs,
        "drop_before_peak": (first_drop is not None and first_drop < IOU_PEAK_EPOCH),
    }


def main() -> None:
    df = load_marking_iou()
    raw = df["marking_iou"].tolist()
    epochs = df["epoch"].tolist()
    smoothed = smoothed_series(raw, SMOOTHING_WINDOW)

    results = [replay(raw, smoothed, p) for p in PATIENCES]

    # Per-epoch CSV (lr trajectory per patience)
    table = {"epoch": epochs, "marking_iou": raw, "smoothed_iou": smoothed}
    for r in results:
        table[f"lr_patience{r['patience']}"] = r["lrs"]
    pd.DataFrame(table).to_csv(OUT_DIR / "scheduler_replay.csv", index=False)

    lines = [
        "# G0 ReduceLROnPlateau Replay (zero-GPU)",
        "",
        "Replays the observed G0 marking-IoU curve through the real PyTorch",
        f"ReduceLROnPlateau (mode=max, factor={FACTOR}, threshold={THRESHOLD} "
        f"{THRESHOLD_MODE}, cooldown={COOLDOWN}, min_lr={MIN_LR}, "
        f"smoothing_window={SMOOTHING_WINDOW}, base_lr={BASE_LR}).",
        f"Observed marking-IoU peak epoch = {IOU_PEAK_EPOCH}.",
        "",
        "| patience | first LR drop epoch | drop before peak? | all drop epochs | final lr |",
        "| ---: | ---: | :---: | --- | ---: |",
    ]
    for r in results:
        fd = r["first_drop_epoch"]
        fd_s = str(fd) if fd is not None else "never (within 25 ep)"
        before = "YES" if r["drop_before_peak"] else "no"
        drops = ", ".join(map(str, r["all_drop_epochs"])) or "none"
        lines.append(
            f"| {r['patience']} | {fd_s} | {before} | {drops} | {r['final_lr']:.6g} |"
        )

    p6 = next(r for r in results if r["patience"] == 6)
    lines += [
        "",
        "## Reading",
        "",
        f"- **patience=6 (the run's actual config):** first LR drop at epoch "
        f"`{p6['first_drop_epoch']}`"
        + (
            " — i.e. within the 25-epoch budget the scheduler "
            + ("DID" if p6["first_drop_epoch"] else "did NOT")
            + " enter a lower-LR phase."
            if True
            else ""
        ),
        "",
        "Interpretation:",
        "- If patience=6 drops at/after the peak but only near epoch ~24-25, a plain",
        "  **epochs-only extension** (e.g. 25->35, patience unchanged) gives the model",
        "  the lower-LR refinement phase it never really used — the clean single-variable",
        "  schedule test.",
        "- Any patience whose 'drop before peak?' = YES is risky: it would cut LR before",
        "  epoch 18 and could lower the peak. Prefer such patiences only if you explicitly",
        "  want to test earlier annealing.",
        "",
        "Caveat: patience<6 rows are 'what the scheduler would do on the OBSERVED curve';",
        "a real run with earlier LR drops would follow a different curve.",
        "",
    ]
    (OUT_DIR / "scheduler_replay.md").write_text("\n".join(lines))

    print("scheduler replay complete")
    for r in results:
        print(f"patience={r['patience']} first_drop={r['first_drop_epoch']} "
              f"drops={r['all_drop_epochs']} before_peak={r['drop_before_peak']}")
    print(f"out_dir {OUT_DIR}")
    print("script_status PASS")


if __name__ == "__main__":
    main()
