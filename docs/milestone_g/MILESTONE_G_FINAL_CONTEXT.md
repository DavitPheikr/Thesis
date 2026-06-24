# Milestone G Final Context

Status: completed — **three runs G0 → G1 → G2** (a single resume chain). **G2 (100 epochs) is the current best model in the project: epoch 68, marking IoU 0.550444** (fully re-derived from raw artifacts). G2 is now local and re-derivable (raw run committed). The **§G2 section below is the primary, detailed record** (the user's focus); G0/G1 are retained as the lineage that led to it.

> **G2 supersedes G1 as the canonical Milestone G model.** G2 = G1 continued (same config, Lovász loss, RGB pipeline — **no feature change**) to 100 epochs. It is the best run on marking IoU and the most stable (tiny best→final drift). **Key carry-over to Milestone H:** even at 100 epochs the **RGB luminance "brightness shortcut" persists essentially unchanged** (§G2.6) — longer training did *not* fix it — which is exactly why H0 introduces train-only RGB brightness/contrast jitter to attack it.

This file is the canonical Milestone G factual context for thesis writing. It is not thesis prose. It was created during the Milestone G factual-context audit on 2026-06-18.

> **Provenance status (read first): MIXED — G2 is fully re-derivable; G0/G1 are derived-only.**
> - **G2 (current best): FULLY locally re-derivable** — the complete raw run is committed (`logs/milestone_g/runs/G2_schedule_extend_100/`: 100-row `eval_history.csv`, 100 eval JSON + confusion `.npy`, both config snapshots, full metadata, `ckpt_epoch_00068.pth`). Every G2 headline number was re-derived from raw artifacts in this audit (best IoU 0.550444 @ ep68, matches `confusion_epoch_068.npy` to 1e-12). Detail in §G2.
> - **G0/G1: still derived-only** — their raw run directories are **not committed** (only `.gitkeep`). Their numbers come from committed derived analysis (`run_analysis/G0_rgb_lovasz/`, `run_analysis/G1_schedule_extend/`) plus a re-derivable sampled `confusion_matrix.npy` each (G0 ep18 0.522980, G1 ep27 0.534884), and the source configs/loss code. To bring G0/G1 to G2's tier, push their raw run dirs.
>
> **What this means for each number:**
> - **Sampled-pass metrics ARE locally re-derivable.** Verified in this audit: loading `sampled_error_analysis_epoch18/confusion_matrix.npy` (G0) and `sampled_error_analysis_epoch27/confusion_matrix.npy` (G1) and recomputing marking IoU reproduces `summary.json` exactly — G0 sampled IoU `0.522980`, G1 sampled IoU `0.534884` (1e-6).
> - **Run-level training-time metrics are derived-only.** The headline selection numbers (G0 best ep18 `0.523048`, G1 best ep27 `0.542659`, the per-epoch loss/LR trajectories, drift, loss-alignment correlations, training-time confusion) come from server-side `eval_history.csv` + per-epoch `.npy` that are **not committed**; locally they exist only as committed derived CSV/MD tables. They are corroborated by the re-derivable sampled pass (training-time G1 0.542659 vs sampled 0.534884; same conclusion) but **cannot be re-derived from raw eval history in this repo**.
>
> This is exactly the tier Milestone F was at **before** its raw run was restored. To bring G to full D/E/F parity, push the raw `runs/G0_rgb_lovasz/` and `runs/G1_schedule_extend/` directories (eval_history, per-epoch JSON/npy, config snapshots, metadata, selected checkpoints `ckpt_epoch_00018.pth` and `ckpt_epoch_00027.pth`).

Primary reliability rule: raw run artifacts first, then derived analysis, then sampled diagnostics, then code/configs, then git history, then notes, then plans.

---

# G2 — 100-Epoch Schedule Extension (CURRENT BEST MODEL)

*Added 2026-06-19 after the raw G2 run was committed. This is the primary, end-to-end record of Milestone G's best model. Every number here was re-derived from raw artifacts in this audit.*

## G2.1 What G2 is

G2 (`G2_schedule_extend_100`) continues the G-series **resume chain** to 100 epochs: G0 (1–25) → G1 (resumed to 35) → **G2 (resumed from epoch 35 to 100)**. The single G2 run directory carries the full 1–100 history. It is a **pure single-variable change vs G1: training budget only (35 → 100 epochs).** Loss (`weighted_CE + 0.5·Lovász`, λ=0.5), RGB pipeline (`intensity_rgb_front`, `in_channels 8`), `dim_features 16`, class weights (effective road 2.445 / marking 15.0 / other 1.711), uniform `SemSegRandomSampler`, AdamW lr 0.0014, and `ReduceLROnPlateau` are all **identical to G1**. **There is NO RGB jitter and NO feature change in G2** — RGB jitter is a separate Milestone-H (H0) idea (§G2.6).

Run metadata (raw): seed 42; commit `882295d`, clean tree; start `2026-06-14T15:38:16` → end `2026-06-19T00:59:39`; `milestone_g_run_complete … run_name=G2_schedule_extend_100 epochs=100 wall_clock=37756.193`; resume snapshot `config_snapshot_resume.yml` `max_epoch: 99` (0-indexed → 100 epochs); `resume_events.jsonl` records the resume from `ckpt_epoch_00035.pth` with `target_epochs: 100`, `resume_latest: true`. Source: `logs/milestone_g/runs/G2_schedule_extend_100/`.

## G2.2 Evidence tier — FULLY locally re-derivable

The complete raw G2 run is committed: `eval_history.csv` (100 epochs), `eval_epoch_001..100.json`, `confusion_epoch_001..100.npy`, `config_snapshot.yml` + `config_snapshot_resume.yml`, `cli_args.json`, `seed/start/end/git_commit/stdout/training_log`, `loss_components.csv`, `resume_events.jsonl`, and the selected `checkpoints/ckpt_epoch_00068.pth`. Verified: **G2 best marking IoU `0.550443757144` at epoch 68 = the max over all 100 raw `eval_history.csv` rows, and recomputing IoU from raw `confusion_epoch_068.npy` reproduces it to 1e-12.** `eval_epoch_068.json` agrees. So G2 is at full D/E/F evidence parity.

## G2.3 Results — best (ep68) vs final (ep100), validation

> **Metric-type correction:** `g2_conclusions.md` labels these as "held-out test set" — that is **incorrect**. The summary.csv values equal the per-epoch `eval_history.csv` rows exactly, i.e. they are **validation** metrics (720 random val steps/epoch), not test. There is no test-set evaluation. Treat all G2 numbers as **validation**.

| metric | G2 best ep68 | G2 final ep100 | Δ final−best |
| --- | ---: | ---: | ---: |
| marking IoU (`lane_iou`) | **0.5504437571438177** | 0.5465588721620231 | **−0.003884884981794512** |
| marking precision | 0.6356172360248448 | 0.6302132673125452 | −0.005403968712 |
| marking recall | 0.8042191605884231 | 0.804592853077486 | +0.000373692489 |
| marking F1 | 0.710046726438925 | 0.7068064229562204 | −0.003240303483 |
| mIoU | 0.827824829372536 | 0.826876286528266 | −0.000948542844 |
| val_loss (total = CE+0.5·Lovász) | 0.2074216640735459 | 0.2037799292140537 | — |
| lr | 4.375e-05 | 2.734375e-06 | — |

Source: `eval_history.csv` rows 68 & 100; `best_vs_final.csv`. **Selected checkpoint = epoch 68** (max raw marking IoU). The best→final drift is **−0.0039** — the most stable run in the series (G0 −0.0445, G1 −0.0097, G2 −0.0039): 100 epochs with full LR annealing produced an almost flat tail.

## G2.4 Comparison vs F0/D0 (same label, validation) and vs G1 (with noise caveat)

Source: `summary.csv`, `g2_conclusions.md`. Same `road_marking3` label; `lane_*` = `marking_*`; all validation. **Do not compare to C0** (strict-lane labels).

| run / ckpt | marking IoU | precision | recall | F1 | mIoU | pred/true |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 best ep18 | 0.440294 | 0.514788 | 0.752636 | 0.611394 | 0.780829 | 1.462 |
| F0 best ep13 | 0.482770 | 0.549249 | 0.799543 | 0.651173 | 0.798682 | 1.456 |
| **G2 best ep68** | **0.550444** | **0.635617** | **0.804219** | **0.710047** | **0.827825** | **1.265** |

- vs **F0**: marking IoU **+0.067674**, precision **+0.086368**, recall +0.004676, pred/true 1.456 → 1.265 (better calibrated). These are well above the single-seed noise floor — a real improvement.
- vs **D0**: marking IoU **+0.110150**.
- vs **G1** best (0.542659): **+0.0078**, which is **at/just below the ~0.008 single-seed noise floor** (`logs/milestone_g/notes/single_seed_limitation.md`). So the G2-over-G1 *peak* gain is **marginal and within noise** — G2's defensible contribution over G1 is **stability** (flat tail, −0.0039 drift) and confirming the result holds at 100 epochs, **not** a clearly-significant IoU jump. State G2 as "ties/edges G1 at the peak, far more stable," not "beats G1."

## G2.5 Training behavior — LR schedule and loss/IoU alignment

- **LR annealed heavily** (`lr_events.csv`): 9 ReduceLROnPlateau drops at epochs 27, 36, 45, 53, 61, 69, 77, 85, 93 — 0.0014 down to 2.73e-6 by ep93. The best epoch 68 sits deep in the low-LR regime (lr 4.375e-05). The long annealing tail is why the model stops drifting.
- **Loss/metric alignment** (`loss_alignment.csv`): corr(val Lovász, marking IoU) = **−0.9845** (argmin epoch 80), corr(val CE, IoU) = **−0.8331** (argmin epoch 31), corr(val total, IoU) = −0.9144 (argmin 80); argmax IoU = 68. Lovász remains far better aligned with marking IoU than CE (CE's minimum is at epoch 31, nowhere near the IoU peak) — reconfirming the G hypothesis at 100 epochs. (Note: Lovász's own argmin at 80 lags the IoU peak at 68 slightly, so marking IoU — not loss — is still the selection metric.) **G2 `val_loss` is total loss; never compare it to F/E/D pure-CE `val_loss`.**

## G2.6 The RGB brightness "shortcut" at G2 — DETAILED (and why H0 changes it)

This is the most important qualitative finding to carry forward. **Even after 100 epochs, G2's residual road→marking false positives are still associated with a luminance-based RGB shortcut — essentially unchanged from F0/G1.** Longer training did not remove it, because it is a *feature* problem, not an epochs problem. Source: `sampled_error_analysis_epoch68/` (fresh sampled pass, ckpt 68, seed 42, 2160 steps, 70,609,605 points; raw-remap mismatch 0.0), `rgb_shortcut_analysis/rgb_shortcut_fingerprint.csv`, `group_feature_summary.csv`.

**Per-outcome RGB/intensity fingerprint** (RGB in [0,1]; brightness = mean(R,G,B); warmth = R−B; intensity = raw LiDAR 0–114):

| group | n | brightness mean / median | warmth | saturation~ | LiDAR intensity (mean/med) |
| --- | ---: | ---: | ---: | ---: | ---: |
| road TP | 27,521,255 | 0.394 / 0.392 | −0.018 | 0.034 | 27.9 / 29 |
| **road→marking (FP)** | 532,367 | **0.407 / 0.478** | +0.031 | 0.038 | **31.9 / 30** |
| marking TP | 916,369 | 0.379 / 0.390 | +0.023 | 0.064 | 38.7 / 34 |
| marking→road (FN, missed) | 229,626 | 0.328 / 0.313 | −0.013 | 0.052 | 25.2 / 28 |

The fingerprint answers (all shortcut-consistent, `rgb_shortcut_conclusions.md`):
1. **road→marking FPs are brighter than road TPs** (median 0.478 vs 0.392, +0.085).
2. **FPs are at least as bright as true markings** (median 0.478 vs 0.390, +0.087).
3. **FPs are less colour-saturated than true markings** (0.038 vs 0.064) → they are **bright, near-neutral road** (glare / light concrete / overexposed pixels), not marking-coloured paint.
4. **Missed markings (FN) are darker** than detected markings (0.328 vs 0.379).
5. **RGB brightness disagrees with LiDAR intensity** in the FPs: their intensity (31.9) is road-like — only ~0.37 of the way from road (27.9) to marking (38.7) — while their RGB brightness is elevated above road and even above true markings.
6. **Concentrated in rgb_valid points:** 80.3% of road→marking FPs (427,293 / 532,367) are rgb_valid; the rgb_valid stratum overpredicts more (pred/true **1.289**) than rgb_invalid (**1.205**), with near-equal IoU (0.5402 vs 0.5395). I verified these strata against `rgb_valid_stratified_metrics.csv` / `summary.json`.

**Interpretation (carefully worded, correlational):** this characterizes the RGB profile of the points the model labels marking; it does **not** prove the network internally computes brightness (no channel ablation). But the pattern is the same one seen in E0/F0/G1: *bright near-neutral road carries the same high-RGB signature as white paint, while its LiDAR intensity stays road-like, and the model over-predicts marking there.*

**Why this matters for H0 (the change we made):** because the shortcut is a feature reliance, not an under-training symptom, the Milestone-H response is a **feature-level intervention, not more epochs**. The G1 brightness-sensitivity probe (`logs/milestone_h/run_analysis/g1_rgb_brightness_sensitivity/`) already showed marking predictions are *monotonically* sensitive to RGB brightness (darkening RGB cuts road→marking FPs and raises precision; brightening worsens them). **H0 = the G1/G2 setup + train-only RGB brightness/contrast jitter** (brightness 0.85–1.15, contrast 0.90–1.10, applied only to rgb_valid training points, validation/test unchanged), designed to make the model less reliant on absolute brightness and thus reduce exactly these bright-road→marking false positives. G2 is the evidence that this intervention is warranted: 100 epochs of the unchanged setup left the shortcut intact. See `docs/milestone_h/MILESTONE_H_FINAL_CONTEXT.md`.

## G2.7 Other sampled diagnostics (ep68)

- **Overall sampled** (`summary.json`): marking IoU 0.540046, precision 0.626677, recall 0.796193, pred/true 1.2705 (vs training-time validation 0.5504 / 1.265 — denser readout, ~same story).
- **Distance** (`distance_bucket_metrics.csv`): marking IoU 0–10m 0.6484 → 60m+ 0.3485; largest support 10–20m (647,702 true marking, IoU 0.5417); 60m+ weak but low support.
- **Raw subtype recall** (`raw_subtype_rgb_stratified_metrics.csv`, all stratum): raw 8 lane-line 0.8111, raw 9 stop-line 0.9321, raw 10 other-paint 0.7788 — all merged types learned.
- Per-sequence and top-error-frame tables: `per_sequence_metrics.csv`, `top_frames_by_*`; visual notes `visual_inspection_notes.md`.

## G2.8 What G2 proves / does not prove

**Proves:** extending the Lovász/RGB setup to 100 epochs with full LR annealing yields the project's best and most stable model (marking IoU 0.5504 @ ep68, drift −0.0039), clearly beating F0/D0; Lovász stays aligned with marking IoU at 100 epochs.
**Does not prove:** a significant gain *over G1* (the +0.0078 peak is within the ~0.008 single-seed noise floor — G2's real edge is stability); anything on the **test set** (all validation); and it does **not** fix the RGB brightness shortcut (§G2.6, the open problem H0 targets). Single-seed only.

---

## 1. Canonical-file decision

No prior `docs/milestone_g/` directory existed. This file was created to match the A–F convention. Two pre-existing log-side documents were found and used as **verified secondary evidence** (not the canonical record):

- `logs/milestone_g/notes/milestone_g_context.md` — a thorough 21-section end-to-end note covering **both G0 and G1** (headline, configs, loss composition, G0 result + alignment, the G1 decision, G1 behavior, official result, stability, confusion, sampled analysis, RGB-valid/distance/sequence/subtype, failure modes, thesis narrative, source map). Every numeric claim I spot-checked against the derived CSVs and the re-derivable sampled npy matched. **Note:** this file is currently *modified in the working tree* (uncommitted); I treated the on-disk version as the note and independently verified the numbers from the analysis artifacts.
- `logs/milestone_g/notes/milestone_g_deep_verdict.md` — a strict, well-sourced **G0-era verdict** that itself flags the server-side provenance and the total-loss-vs-CE distinction.

## 2. Milestone purpose

Milestone G addresses the **central training-behavior problem exposed by F0**: the weighted cross-entropy objective was **misaligned with marking IoU**. In F0, validation CE kept *improving* after the best marking-IoU epoch while marking IoU *degraded* — so the loss could not be used to select the checkpoint, and training past the peak drifted toward higher-recall / lower-precision (`docs/milestone_f/MILESTONE_F_FINAL_CONTEXT.md`, F0 best ep13 IoU 0.482770 → final ep25 0.449982 while val CE fell 0.118228 → 0.106725).

G's hypothesis: **add a direct IoU surrogate (Lovász-Softmax) to the loss** so the objective tracks the metric. G0 tests the loss change; G1 tests whether G0 simply stopped before its (unchanged) scheduler delivered an LR-refinement phase.

## 3. How G builds on F and earlier milestones

Same `road_marking3` task, frozen Milestone B split, active index order `0=road, 1=marking, 2=other`, and `lane_* == marking_*` aliasing as D/E/F. Same E0/F0 RGB pipeline and the F0 model (`in_channels 8`, `dim_features 16`) and F0 class weights (effective road 2.445 / marking 15.00 / other 1.711). G changes **only the loss** (G0) and then **only the epoch budget** (G1). The comparison anchor is F0 (best ep13). **Do not compare G to C0** without stating the label change (C0 = strict lane raw 8; D/E/F/G = road marking raw 8+9+10).

## 4. What changed — and whether it is an ablation

| Step | Single variable changed | Held constant | Pure ablation? |
| --- | --- | --- | --- |
| F0 → **G0** | **loss**: `weighted_CE` → `weighted_CE + 0.5·Lovász-Softmax` | RGB pipeline, features (`in_channels 8`), model (`dim_features 16`), class weights, sampler, AdamW lr 0.0014, ReduceLROnPlateau, batch 2, num_points 32768, split, seed | **Yes** — clean single-factor (config comments explicitly mark weights/`dim_features` "unchanged from F0") |
| G0 → **G1** | **training budget only**: continue/resume to 35 epochs | loss, λ=0.5, weights, model, RGB, sampler, optimizer, scheduler patience, LR base, batch, seed, cache | **Yes** — clean single-factor (continuation length) |

This is **cleaner than F0**, which changed *two* variables at once (marking weight + `dim_features`). G0 isolates the loss effect; G1 isolates the schedule effect. Source: `logs/milestone_g/configs/g0_rgb_lovasz.yml`, `g1_schedule_extend.yml`.

## 5. Dataset, label, features (same as D/E/F)

- `PandaSetFFLane3`, `label_mode: road_marking3`. **marking = raw 8 (lane line) + raw 9 (stop line) + raw 10 (other road paint)**; road = raw 7; other = remaining; raw 0 ignored.
- Frozen B split (58 train / 9 val / 9 test); 4640 train steps, 720 val steps/epoch.
- Features (8 ch): `intensity_rgb_front` = xyz + standardized intensity + r + g + b + rgb_valid (intensity clip `[0,114]`, mean 22.471752, std 14.902380; RGB /255).
- **`lane_*` columns in all G metric files mean `marking_*`** (config `metric_alias_note`).

## 6. Loss function (the core of G)

- Config block (`g0_rgb_lovasz.yml`): `pipeline.loss.name: weighted_ce_lovasz`, `lovasz_lambda: 0.5` (λ=0.5 from epoch 1, **no warmup**), `lovasz_classes: present` (only classes present in the batch contribute).
- Implementation: `src/thesis_pipeline/losses/combined_semseg_loss.py` — `CombinedSemSegLoss` wraps the stock Open3D `SemSegLoss` so the **CE term is byte-for-byte identical** to the previous weighted-CE, then adds `lovasz_lambda * lovasz_softmax(softmax(logits), labels, classes="present")`; `total = ce + 0.5·lov`. Lovász kernel: `src/thesis_pipeline/losses/lovasz.py` (`lovasz_softmax`, `lovasz_grad`). Factory `build_loss` keys on name `weighted_ce_lovasz`. Unit test: `tools/test_combined_loss.py`. Runner: `tools/train_milestone_g.py`.
- **Composition verified** in the analysis: max validation compose residual `8.71e-10` (G1) / `8.7e-10` (G0) — the saved `train_loss`/`val_loss` are the **total** (CE + 0.5·Lovász); the CE component is `val_ce`. **Consequence (critical for comparison): G0/G1 `val_loss` is NOT comparable to F0 `val_loss` (pure CE).** The fair CE comparison uses `val_ce`. Source: `run_analysis/G0_rgb_lovasz/loss_alignment.csv`, `run_analysis/G1_schedule_extend/{analysis.md,loss_component_summary.csv}`.

## 7. Training setup (both runs)

RandLANet (`num_layers 3`, `num_neighbors 24`, `sub_sampling_ratio [4,4,4]`, `dim_output [16,64,128]`, `num_points 32768`, `dim_features 16`, `in_channels 8`, `grid_size 0.04`); `SemSegRandomSampler`; AdamW lr `0.0014`, wd `0.0001`; `ReduceLROnPlateau` mode max, watch `marking_iou` (smoothed window 3), factor 0.5, patience 6, threshold 0.01 abs, cooldown 1, min_lr 1e-6; batch 2 / val batch 2; workers 0; pin_memory true; device cuda; seed 42 (sampled-analysis seed 42). G0 `max_epoch: 24` (0-indexed → 25 epochs); **G1 keeps `max_epoch: 24` in its config but the continuation to 35 epochs was CLI-driven (`--resume-latest`, target 35)** — the analysis confirms epochs 1–35 exist and best/final = 27/35. **Hardware/wall-clock/exact run times are not available locally** (no run metadata committed).

## 8. Runs, checkpoints, selection

| Run | Run name | Epochs | Selected ckpt | Selection rule |
| --- | --- | ---: | --- | --- |
| G0 | `G0_rgb_lovasz` | 25 (best 18, final 25) | epoch 18 | max raw marking IoU |
| **G1** (canonical best) | `G1_schedule_extend` | 35 (best 27, final 35) | **epoch 27** | max raw marking IoU |

The **canonical Milestone G model is G1 epoch 27** (`ckpt_epoch_00027.pth`, referenced by the sampled diagnostic — not committed locally). Checkpoint selection is by marking IoU, **not** validation loss (the G analysis shows why — §11).

## 9. Evaluation procedure

Per-epoch validation (no weight update), 720 random val steps (**validation-sampled, not test, not exhaustive**), active-class confusion after ignore filtering, order road/marking/other. The deeper diagnostics use a **separate fresh sampled inference pass** per selected checkpoint (seed 42, 2160 steps, 70,609,605 active points, raw-remap mismatch rate 0.0) — not the original training-time validation sample.

## 10. Metrics

> Tier: §10.1–§10.3 are **training-time validation** (derived from server-side `eval_history.csv` — committed only as `summary.csv`/`best_vs_final.csv`). §10.4 is the **sampled** pass, **re-derivable** from `confusion_matrix.npy`. All values are **validation**, never test.

### 10.1 The full chain (best-epoch marking-class metrics, validation)

Source: `logs/milestone_g/run_analysis/G1_schedule_extend/summary.csv` (consolidates D0/F0/G0/G1; D0/F0 rows match the committed D0/F0 eval JSON exactly).

| run | best ep | marking IoU | marking F1 | precision | recall | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | 18 | 0.4402936634062269 | 0.6113942935289365 | 0.5147877273726487 | 0.7526361971421429 | 0.7808293932909053 |
| E0 | 14 | 0.4384393663916791 | 0.6096042372526316 | 0.4702983574044707 | 0.8661703539710927 | 0.7806584578414802 |
| F0 | 13 | 0.482769554261302 | 0.6511727366857246 | 0.5492490994544084 | 0.7995432172280974 | 0.7986818335955664 |
| G0 | 18 | 0.523048107081327 | 0.6868438424885519 | 0.6299487751362556 | 0.7550364501762566 | 0.8174073701842198 |
| **G1** | **27** | **0.5426594966220241** | **0.7035376216336666** | **0.635132739661364** | **0.7884556423696963** | **0.8232388764440427** |

### 10.2 Best vs final epoch (drift)

Source: `run_analysis/G1_schedule_extend/best_vs_final.csv`.

| metric | G0 best ep18 | G0 final ep25 | Δ | G1 best ep27 | G1 final ep35 | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| marking IoU | 0.523048 | 0.478593 | **−0.044455** | 0.542659 | 0.532956 | **−0.009703** |
| precision | 0.629949 | 0.550910 | −0.079039 | 0.635133 | 0.619992 | −0.015141 |
| recall | 0.755036 | 0.784758 | +0.029722 | 0.788456 | 0.791513 | +0.003057 |
| F1 | 0.686844 | 0.647363 | −0.039481 | 0.703538 | 0.695331 | −0.008206 |
| mIoU | 0.817407 | 0.795974 | −0.021433 | 0.823239 | 0.820118 | −0.003121 |

G0 raised the peak but **still drifted** (−0.0445) because it never got an LR drop in 25 epochs (LR stayed 0.0014). G1's single LR drop at epoch 27 (0.0014→0.0007, `lr_events.csv`) produced a **new, higher, and far more stable** peak (drift −0.0097, ~4.6× smaller). Note: G0's *final* IoU (0.4786) ≈ F0's *best* (0.4828) — i.e. even unstabilized G0 is roughly uniformly better than F0.

### 10.3 Loss/metric alignment (why selection uses IoU, not loss)

Source: `run_analysis/G0_rgb_lovasz/loss_alignment.csv`, `run_analysis/G1_schedule_extend/loss_alignment.csv`.

| signal | corr w/ marking IoU (G0) | argmin epoch (G0) | corr (G1) | argmin (G1) |
| --- | ---: | ---: | ---: | ---: |
| validation CE | −0.9104 | 24 | −0.8888 | 31 |
| validation Lovász | **−0.9751** | **18** | **−0.9814** | **27** |
| validation total | (aligned) | 18 | −0.9344 | 28 |

argmax marking IoU = epoch 18 (G0) / 27 (G1). **Lovász tracks marking IoU much better than CE**, and its minimum lands on the IoU peak; CE's minimum still lags to later epochs (24 / 31). This is the core scientific result: the F0 CE/IoU mismatch is resolved at the loss level, so total loss is now a *usable* (if 1-epoch-lagging) checkpoint proxy — but the official selection is still raw marking IoU.

### 10.4 Sampled best-checkpoint metrics (re-derived from raw `confusion_matrix.npy`)

| run / ckpt | marking IoU | precision | recall | pred/true | road→marking | marking→road | source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G0 ep18 (sampled) | 0.522980 | 0.632140 | 0.751772 | 1.189250 | 489,746 | 275,549 | `G0_rgb_lovasz/sampled_error_analysis_epoch18/` |
| **G1 ep27 (sampled)** | **0.534884** | 0.631334 | 0.777838 | 1.232055 | 507,459 | 250,859 | `G1_schedule_extend/sampled_error_analysis_epoch27/` |

Re-derived directly from each `confusion_matrix.npy` (exact match to `summary.json`). Sampled IoU is ~0.008 below training-time validation IoU (denser readout) but supports the same conclusion.

## 11. Confusion, calibration, precision/recall trade

- **pred/true marking ratio** (1.0 = calibrated): D0 1.46, E0 1.84, F0 1.456 → **G0 best 1.199** (training-time) / 1.189 (sampled), **G1 best 1.241** (training-time) / 1.232 (sampled), G1 final 1.277. G dramatically improves calibration vs E0/F0 and is the least-overpredicting series so far.
- **Training-time confusion** (`confusion_breakdown.csv`): G0 best road→marking 168,907, marking→road 92,659, TP 296,006; G1 best road→marking 177,649, marking→road 83,982, TP 318,883. G1 vs G0: recovers TPs (+22,877) and cuts missed markings (marking→road −8,677), at the cost of slightly more road→marking FPs (+8,742) — net precision still up.
- **Precision/recall trade:** the F0→G0 step is **precision-led** (precision +0.081 vs F0, recall −0.045 back to ~D0). The G0→G1 step recovers recall (+0.033) while nudging precision up (+0.005). So G's gains are calibration/precision-driven, then stabilized.
- **Validation loss vs IoU:** see §10.3 — Lovász/total loss now align with IoU; CE alone still does not.

## 12. Diagnostic / qualitative analysis (sampled G1 ep27)

- **RGB-valid stratification** (`rgb_valid_stratified_metrics.csv`): all IoU 0.534884 (pred/true 1.232); **rgb_valid** (81.7% of points) IoU 0.535114, precision 0.621480, recall 0.793840, pred/true **1.277**; **rgb_invalid** IoU 0.533968, precision 0.674026, recall 0.719865, pred/true **1.068**. Both strata are strong (≈ equal IoU) — the good aggregate is not just because RGB-valid dominates. But RGB-valid still overpredicts more (1.277 vs 1.068): the RGB-brightness false-positive tendency is **reduced, not eliminated**.
- **Distance** (`distance_bucket_metrics.csv`): marking IoU 0–10m 0.6297 → 60m+ 0.3109; most support is 10–20m (647,702 true marking pts, IoU 0.5367); 60m+ weak but low support (10,290).
- **Per-sequence** (`per_sequence_metrics.csv`): best 106 (IoU 0.7075), worst 034 (0.3542, pred/true 1.956); 123/124/034 remain the overprediction-heavy sequences; 116 is the opposite (precision 0.742, recall 0.567, pred/true 0.765).
- **Raw subtype recall**: raw 8 lane-line 0.7857, raw 9 stop-line 0.8838, raw 10 other-paint 0.7682 — all merged types learned; remap choice justified.
- **RGB-shortcut fingerprint** (`G0_rgb_lovasz/rgb_shortcut_analysis/`, applied across runs): the bright-near-neutral-road → marking false-positive mechanism from E0/F0 **persists** in G but at smaller magnitude (correlational, no channel ablation).
- **Front-camera prediction overlays** exist (`run_analysis/front_camera_predictions/G1_schedule_extend/`) plus `visual_inspection_notes.md`; full manual qualitative confirmation is the stated next step.

## 13. Post-G0 decision diagnostics (deviations / ruled-out paths)

Before G1, two committed checks (`run_analysis/analysis_code/g0_bias_sweep.py`, `scheduler_replay.py`): a **marking-logit bias sweep** found best bias b=−0.20 giving only +0.0016 IoU (well-calibrated; ruled out a `lovasz_lambda` change as the next experiment), and a **scheduler replay** motivated the schedule-extension hypothesis. So G1 (extend schedule), not a λ tweak, was the justified clean follow-up. Source: git commits `3c3cb8b`, `ecff373`; `run_analysis/G0_rgb_lovasz/bias_sweep/`, `scheduler_replay.md`.

## 14. What G proves / does not prove

**Proves (with the provenance tier in mind):**
- Adding Lovász-Softmax (single-variable change) to the F0 setup improves the headline marking metric (G0 +0.0403 IoU over F0) and **aligns the objective with marking IoU** (Lovász corr −0.975/−0.981 vs CE −0.91/−0.89).
- Extending the schedule (single-variable change) so the unchanged scheduler anneals LR both **raises the peak** (G1 +0.0196 over G0) and **stabilizes the tail** (drift −0.0097 vs −0.0445).
- G1 epoch 27 is the best model in the series (marking IoU 0.5427, mIoU 0.8232) with the best calibration of any run.

**Does not prove:**
- Not test-set, not exhaustive — all metrics are **validation** (per-epoch 720-step eval; diagnostics a 2160-step sampled pass).
- Lovász did **not** eliminate overprediction or the RGB luminance shortcut (both reduced, still present).
- Does not prove long-range markings are solved (60m+ IoU 0.31).
- Visual quality is **not** fully confirmed (manual overlay inspection pending).
- The G0/G1 `val_loss` is total loss — **not** comparable to F0/E0/D0 `val_loss` (pure CE); use `val_ce`.

## 15. Limitations and caveats

- **Provenance (main caveat):** raw G run artifacts are server-side; run-level training-time metrics are derived-only/corroborated (sampled pass re-derivable). One tier below D/E/F. See top.
- Per-stratum sampled splits reconcile to the verified sampled aggregate but per-point arrays were not saved (consistency-checked, not independently re-derivable).
- **No test-set evaluation** (same as D/E/F).
- G1 config file still says `max_epoch: 24`; the 35-epoch continuation was CLI-driven — a documentation-vs-run subtlety, resolved by the analysis (epochs 1–35 present, best/final 27/35).
- RGB-shortcut conclusion is correlational.

## 16. Contradictions / unresolved points

| Point | Resolution |
| --- | --- |
| Raw G run dirs absent locally | Derived analysis + sampled npy committed and re-derive exactly; run-level numbers derived-only/corroborated. Push raw runs for full parity. |
| G1 `max_epoch: 24` vs 35 epochs run | Config is a G0 copy; continuation was CLI `--resume-latest` to 35. Analysis confirms 35 epochs, best 27 / final 35. No metric impact. |
| G `val_loss` vs F `val_loss` | Different quantities (total vs pure CE). Compare `val_ce` only. Documented. |
| `milestone_g_context.md` modified in working tree | Pre-existing uncommitted edit (not from this audit). Numbers independently verified against analysis CSVs + sampled npy. |

## 17. Relationship to later milestones

**G2 ep68 is the current best checkpoint** (the "longer ~100-epoch budget" idea, now executed; supersedes G1 ep27 — see §G2). The other two supervisor ideas (z-crop, marking-aware sampling) were audited in Milestone H and **rejected** (height crop can't reach road-level errors; sampler leverage too low). The remaining open problem from G2 is the **RGB brightness shortcut** (§G2.6), which 100 epochs did not fix — so **Milestone H0 = the G2 setup + train-only RGB brightness/contrast jitter**, a feature-level intervention targeting exactly the bright-road→marking false positives G2 still exhibits. **H0 has since been run** — trained 100 epochs and finalized 2026-06-21 (`docs/milestone_h/MILESTONE_H_FINAL_CONTEXT.md`), and **test-evaluated** in `results/RESULTS_ANALYSIS.md` (full-coverage test marking IoU 0.510, statistically tied with G2). *(This line originally read "designed/configured but not yet run"; updated after the H0 run completed.)* Do not attribute H results to G.

## 18. Source map

| Claim area | Primary sources |
| --- | --- |
| End-to-end narrative (verified secondary) | `logs/milestone_g/notes/milestone_g_context.md`; G0 verdict `logs/milestone_g/notes/milestone_g_deep_verdict.md` |
| Configs (the two single-variable changes) | `logs/milestone_g/configs/g0_rgb_lovasz.yml`, `g1_schedule_extend.yml` |
| Loss implementation | `src/thesis_pipeline/losses/combined_semseg_loss.py`, `lovasz.py`, `__init__.py`; runner `tools/train_milestone_g.py`; test `tools/test_combined_loss.py` |
| Run-level metrics, best/final, chain | `run_analysis/G1_schedule_extend/summary.csv`, `best_vs_final.csv`, `g0_vs_g1_epoch_metrics.csv` |
| Loss/metric alignment | `run_analysis/G0_rgb_lovasz/loss_alignment.csv`, `run_analysis/G1_schedule_extend/loss_alignment.csv`, `loss_component_summary.csv` |
| Confusion / calibration | `run_analysis/G1_schedule_extend/confusion_breakdown.csv`, `pred_true_ratio_by_epoch.csv`, `lr_events.csv` |
| Sampled diagnostics (raw npy + CSVs) | `run_analysis/G0_rgb_lovasz/sampled_error_analysis_epoch18/`, `run_analysis/G1_schedule_extend/sampled_error_analysis_epoch27/` |
| RGB-shortcut, bias sweep, scheduler replay | `run_analysis/G0_rgb_lovasz/rgb_shortcut_analysis/`, `bias_sweep/`, `scheduler_replay.md` |
| Conclusions | `run_analysis/G0_rgb_lovasz/g0_conclusions.md`, `run_analysis/G1_schedule_extend/g1_conclusions.md` |
| F0 baseline that motivated G | `docs/milestone_f/MILESTONE_F_FINAL_CONTEXT.md` |

## 19. Safe-for-thesis-use summary

Safe to use as factual Milestone G context if these caveats are preserved:

- **Provenance:** G is **partially locally supported** — sampled metrics re-derivable (G0 0.522980, G1 0.534884), run-level training-time metrics derived-only/corroborated (raw runs server-side). One tier below D/E/F. Quote the training-time headline (G1 ep27 0.542659) as "derived and corroborated," not "raw-re-derived," unless/until the raw run is restored.
- G0 and G1 are each **clean single-variable changes** (G0 = loss only; G1 = epochs only) — cleaner attribution than F0's two-variable change.
- `road_marking3`; `lane_* = marking_*`; all metrics **validation** (per-epoch or 2160-step sampled), not test.
- Canonical model = **G1 epoch 27** (max marking IoU); G0 final and G1 final are the final-epoch checkpoints (lower IoU).
- **G/`val_loss` is total (CE + 0.5·Lovász)** — never compare it to F/E/D `val_loss`; use `val_ce`.
- Do not compare G to C0 without stating the strict-lane → road-marking label change.
- G improved peak + calibration + stability but did **not** eliminate overprediction, the RGB shortcut, or long-range weakness; visual confirmation pending.
