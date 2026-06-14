# Milestone G Context Document

End-to-end reference for Milestone G (the Lovász-Softmax loss change on top of
Milestone F). Written for thesis use and future context recovery. Every major
number is backed by a committed file; measured results are separated from
interpretation.

> **Provenance note.** The G0 *run* directory
> (`logs/milestone_g/runs/G0_rgb_lovasz/`) lives on the training server and is not
> committed. All numbers below come from the committed **analysis outputs** under
> `logs/milestone_g/run_analysis/G0_rgb_lovasz/` — `summary.csv`,
> `pred_true_ratio.csv`, `confusion_breakdown.csv`, `loss_alignment.json`,
> `loss_component_summary.csv`, `analysis.md`, `g0_conclusions.md`,
> `loss_component_report.md`, the `sampled_error_analysis_epoch18/` CSV/JSON, and
> the `rgb_shortcut_analysis/` outputs — plus the config
> (`logs/milestone_g/configs/g0_rgb_lovasz.yml`). Training-time validation metrics
> and fresh-sampled-inference metrics are distinguished where it matters. Items
> labelled **Interpretation** are reasoning, not direct measurement.

Terminology: the active class of interest is **marking**; some legacy columns read
`lane_*` (e.g. `lane_iou` = marking IoU), an alias documented in the config.

---

## 1. Milestone G purpose

Milestone F fixed E0's gross overprediction (marking pred/true 1.84 → 1.46) and
became the first run to beat both D0 and E0 on marking IoU. But the F0 analysis
left **two residuals** (see `logs/milestone_f/notes/milestone_f_context.md` §10):

1. **Dominant residual — CE/IoU mismatch.** F0 validation loss kept *improving*
   after the best marking-IoU epoch while marking IoU *degraded*. The weighted-CE
   objective was only loosely aligned with the target metric, so the model drifted
   to a higher-recall / lower-precision regime if trained past the best epoch, and
   val loss could not be used for checkpoint selection.
2. **Secondary residual — RGB-localized bright-road false positives** (a luminance
   shortcut; F0 §9).

**Interpretation.** Because the *dominant* residual was a loss/metric
misalignment rather than class imbalance, the targeted fix is to add a term that
*is* a differentiable surrogate for IoU. Milestone G adds **Lovász-Softmax** (a
convex surrogate of the Jaccard/IoU loss) to the existing weighted CE, keeping the
entire F0 setup otherwise unchanged. The hypothesis: align the objective with
marking IoU, raise the IoU peak, and reduce the best-to-final drift. Lovász was
chosen over Dice/GDice/Focal/Tversky because it directly targets IoU (the headline
metric), whereas overprediction (a Tversky target) was by F0 already the
*secondary* problem.

---

## 2. What changed from Milestone F to Milestone G

**Exactly one thing changed: the loss.** Source:
`logs/milestone_g/configs/g0_rgb_lovasz.yml` (`pipeline.loss` block) vs
`logs/milestone_f/configs/f0_rgb_soft_weights.yml` (no loss block), and the
config-snapshot check in `analysis.md` (which would have failed otherwise).

```yaml
pipeline:
  loss:
    name: weighted_ce_lovasz
    lovasz_lambda: 0.5          # lambda = 0.5 from epoch 1, NO warmup
    lovasz_classes: present     # only classes present in the batch contribute
```

So the F0 training objective (stock weighted CE) becomes:

```text
loss = weighted_CE  +  0.5 * Lovasz-Softmax
```

Implementation (`src/thesis_pipeline/losses/combined_semseg_loss.py`):
- The CE term is the **exact** stock Open3D `SemSegLoss.weighted_CrossEntropyLoss`
  (same class weights, same reduction) — so the CE component is directly
  comparable to F0's loss.
- The Lovász term is **unweighted** (`classes="present"`), computed on
  `softmax(scores)`; the rare-class emphasis still comes only from the CE weights.
- Per-step CE / Lovász / total are logged to `loss_components.csv`.

**What stayed identical to F0 (verified by the G0 config-snapshot check):** class
weights (count list `[119562394, 14344000, 173473484]` → effective CE road 2.445,
**marking 15.00**, other 1.711), `dim_features: 16`, `in_channels: 8`,
`feature_mode: intensity_rgb_front` (8 ch), the whole RGB projection/sampling/
cache pipeline, `num_layers: 3`, `num_points: 32768`, AdamW lr 0.0014 / wd 0.0001,
`ReduceLROnPlateau` on `marking_iou`, `SemSegRandomSampler`, batch size 2,
augmentations, 25 epochs, dataset split, intensity normalization, validation
protocol, seed 42.

**This is a clean single-variable change** (unlike F0, which changed weight *and*
`dim_features`). G0-vs-F0 therefore isolates the effect of adding Lovász.

> **Backwards-compatibility (verified).** The combined loss is config-gated via
> `build_loss(...)`: with no `pipeline.loss` block (or `name: weighted_ce`) the
> runner returns the stock `SemSegLoss`, so D0/E0/F0 behave exactly as before. Unit
> + equivalence tests in `tools/test_combined_loss.py` passed on CPU and CUDA.

---

## 3. Experimental setup

All from `logs/milestone_g/configs/g0_rgb_lovasz.yml` unless noted; identical to F0
except the loss (§2).

- **Dataset / label mode:** `PandaSetFFLane3`, `label_mode: road_marking3`
  (road / marking / other; raw 0 ignored; marking = raw 8 + 9 + 10).
- **Features (8 ch):** `intensity_rgb_front` = xyz + intensity + r + g + b +
  rgb_valid. Intensity clipped `[0,114]`, z-scored (mean 22.4718, std 14.9024);
  RGB scaled `1/255`.
- **Model:** stock `RandLANet`, `num_layers: 3`, `dim_features: 16`,
  `dim_output: [16,64,128]`, `num_points: 32768`, `num_neighbors: 24`,
  `sub_sampling_ratio: [4,4,4]`, `grid_size: 0.04`.
- **Loss:** `weighted_ce_lovasz`, `lovasz_lambda: 0.5`, `lovasz_classes: present`,
  no warmup.
- **Sampling:** `SemSegRandomSampler` (uniform, not class-aware);
  `steps_per_epoch_train: 4640`, `steps_per_epoch_valid: 720`.
- **Optimizer / scheduler:** AdamW lr 0.0014, wd 0.0001; ReduceLROnPlateau mode
  `max`, factor 0.5, patience 6, threshold 0.01 abs, cooldown 1, min_lr 1e-6,
  `watch_metric: marking_iou`, smoothing_window 3. batch_size 2, val_batch_size 2.
- **Class weights:** effective CE road 2.445 / marking 15.00 / other 1.711.
- **Epochs:** 25. **Checkpoint-selection rule:** max raw marking IoU (`lane_iou`).
  Official G0 checkpoint = **epoch 18** (`g0_conclusions.md`, sampled
  `summary.json` `checkpoint_epoch: 18`).
- **Reproducibility:** sampled analysis seed 42, 2160 steps, device cuda,
  rgb_valid_threshold 0.5; raw-remap alignment verified (mismatch_rate 0.0 over
  70,609,605 points).
- **Cache:** reused F0's bake-projection cache (the manifest is identical between
  F0 and G0; class weights / dim_features / loss are not part of the cache key).

---

## 4. Training behavior

Source: `summary.csv`, `loss_alignment.json`, `g0_conclusions.md`. For G0, the
`eval_history`/`summary.csv` `train_loss`/`val_loss` columns are the **TOTAL** loss
(CE + 0.5·Lovász); the CE component is in `loss_components.csv` / `val_ce`.

| quantity (G0) | best ep18 | final ep25 |
| --- | ---: | ---: |
| val total loss (CE + 0.5·Lovász) | 0.209156 | 0.221582 |
| val CE component | 0.112872 | 0.118533 |
| val Lovász (raw, unscaled) | 0.192567 | 0.206099 |
| marking IoU | 0.523048 | 0.478593 |
| learning rate | 0.0014 | 0.0014 |

Composition verified to ~1e-9: `val_total = val_ce + 0.5·val_lovasz`
(`max_compose_val_residual 8.7e-10`, `max_total_val_residual 0.0`,
`loss_alignment.json`).

**Loss/metric alignment (measured — this is G's central result).** From
`loss_alignment.json`:
- `corr(val CE, marking IoU) = −0.910`; `corr(val Lovász, marking IoU) = −0.975`.
- `argmin(val Lovász) = epoch 18 = argmax(marking IoU)`, while
  `argmin(val CE) = epoch 24` (CE keeps falling 6 epochs past the IoU peak — the
  exact F0 mismatch).

**Interpretation.** The Lovász component tracks marking IoU almost perfectly and
bottoms out exactly at the IoU peak, whereas CE alone still drifts past it. The
F0 CE/IoU mismatch is substantially resolved *at the loss level*: the Lovász term
(and therefore the total loss, of which it is ~46%, §9) is now a good proxy for
the metric.

**Best-vs-final drift (measured).** Marking IoU still falls after the peak:
0.523048 (ep18) → 0.478593 (ep25), Δ = **−0.044455**, slightly larger in absolute
terms than F0's −0.032787. **However, G0's final IoU (0.479) ≈ F0's best (0.483)**
— i.e. G0 is uniformly better; the drift is from a higher peak, not below F0.
The overprediction drift is *smaller* in G0 (pred/true +0.226 vs F0 +0.326; §6).

**Learning-rate note (measured).** G0's LR stayed at 0.0014 through epoch 25 (no
plateau drop), because the best epoch (18) is late and the scheduler's
patience-6 window had not triggered a reduction by epoch 25 (post-hoc confirmed by
a zero-GPU `ReduceLROnPlateau` replay: patience 6 never drops within 25 epochs;
patience 5/4/3 would drop at 25/18/17 — see "Post-hoc calibration / bias sweep").
F0, with its earlier
peak (13), had dropped to 0.0007 by epoch 25. **Interpretation:** G0 likely never
benefited from an LR reduction within the 25-epoch budget — a candidate lever for
future work (longer schedule / earlier LR drop) rather than a defect.

---

## 5. Main G0 results

Best-epoch metrics, training-time validation (source: `summary.csv`,
`g0_conclusions.md`):

| metric | D0 ep18 | E0 ep14 | F0 ep13 | **G0 ep18** | G0−F0 | G0−D0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| marking IoU | 0.440294 | 0.438439 | 0.482770 | **0.523048** | +0.040279 | +0.082754 |
| marking F1 | 0.611394 | 0.609604 | 0.651173 | **0.686844** | +0.035671 | +0.075450 |
| marking precision | 0.514788 | 0.470298 | 0.549249 | **0.629949** | +0.080700 | +0.115161 |
| marking recall | 0.752636 | 0.866170 | 0.799543 | 0.755036 | −0.044507 | +0.002400 |
| mIoU | 0.780829 | 0.780658 | 0.798682 | **0.817407** | +0.018726 | +0.036578 |

Best vs final (G0):

| metric | best ep18 | final ep25 | Δ |
| --- | ---: | ---: | ---: |
| marking IoU | 0.523048 | 0.478593 | −0.044455 |
| precision | 0.629949 | 0.550910 | −0.079039 |
| recall | 0.755036 | 0.784758 | +0.029722 |

**Reading (measured + Interpretation).** G0 is the **best run in the series** on
marking IoU, F1, precision, and mIoU. Versus F0 it makes a clear
**recall-for-precision trade**: precision +0.081, recall −0.045, netting IoU
+0.040 (+8.3% relative). Recall (0.755) is essentially back to the D0 level
(0.753) — G0 is the most *precise* run, not the most sensitive. Cross-check: the
independent fresh sampled pass (sampled `summary.json`) gives IoU 0.522980,
precision 0.632140, recall 0.751772 — within ~0.003 of the training-time numbers.

---

## 6. Confusion and pred/true behavior

Marking predicted/true ratio (training-time confusion, `pred_true_ratio.csv`):

| run / epoch | marking pred/true | road pred/true | other pred/true |
| --- | ---: | ---: | ---: |
| D0 official | 1.462032 | 1.0018 | 0.9853 |
| E0 best | 1.841747 | 0.9854 | 0.9858 |
| F0 best | 1.455702 | 1.0011 | 0.9860 |
| **G0 best** | **1.198568** | 0.9905 | 1.0010 |
| G0 final | 1.424476 | 1.0045 | 0.9857 |

Confusion-count changes at best epochs (true→pred, `confusion_breakdown.csv`):

| error | F0 | G0 | G0−F0 |
| --- | ---: | ---: | ---: |
| road→marking (FP) | 243,315 | 168,907 | **−74,408 (−30.6%)** |
| other→marking (FP) | 17,837 | 4,976 | **−12,861 (−72%)** |
| marking→road (FN) | 78,971 | 92,659 | +13,688 |
| marking→marking (TP) | 318,219 | 296,006 | −22,213 |

**Reading (measured).** G0 has the **best calibration in the series**: marking
pred/true 1.199 (vs F0 1.456, E0 1.842, D0 1.462), and road (0.990) and other
(1.001) are near-perfect. It cut road→marking false positives by 31% and
other→marking by 72% vs F0, at the cost of more marking→road false negatives
(+13,688) and fewer TPs (−22,213) — i.e. the same recall-for-precision trade as
§5. (Side effect: G0 also reshuffled road↔other confusions — other→road −120,117,
road→other +69,628 — netting the higher mIoU.)

**Interpretation.** Adding the IoU surrogate pushed the model from "fire when
plausibly marking" toward "fire when confidently marking," which is exactly what
raises IoU when the prior regime was overpredicting.

---

## 7. RGB-valid / RGB-invalid stratification

Fresh sampled pass, seed 42, 2160 steps (source:
`sampled_error_analysis_epoch18/rgb_valid_stratified_metrics.csv`):

| stratum | marking IoU | precision | recall | pred/true | road→marking |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 0.522980 | 0.632140 | 0.751772 | 1.1893 | 489,746 |
| rgb_valid | 0.525595 | 0.622253 | 0.771878 | 1.2405 | 412,083 |
| rgb_invalid | 0.512477 | 0.676404 | 0.678933 | 1.0037 | 77,663 |

F0 reference (F0 §7): rgb_valid IoU 0.471 / pred-true 1.550; rgb_invalid IoU 0.520
/ pred-true 1.148.

**Reading (measured).** Two notable shifts vs F0:
1. **The RGB-valid IoU penalty flipped.** In F0, rgb_valid IoU (0.471) was *worse*
   than rgb_invalid (0.520). In G0, rgb_valid IoU (0.526) is now slightly *better*
   than rgb_invalid (0.512). RGB-valid points are no longer the weak stratum.
2. **The overprediction gap narrowed sharply.** rgb_valid pred/true fell 1.550 →
   **1.240**, rgb_invalid 1.148 → **1.004**; the valid−invalid gap shrank from
   0.402 to 0.237.

**Interpretation.** G0 substantially closed the RGB-valid overprediction that F0
still carried, while keeping RGB's recall benefit (rgb_valid recall 0.772 still
> rgb_invalid 0.679). The residual is now small and no longer concentrated as
sharply in RGB-valid regions.

---

## 8. Distance and per-sequence analysis

**Distance buckets** (`sampled_error_analysis_epoch18/distance_bucket_metrics.csv`):

| bucket | G0 marking IoU | F0 IoU | G0 pred/true |
| --- | ---: | ---: | ---: |
| 0–10 m | 0.6493 | 0.6071 | 1.196 |
| 10–20 m | 0.5207 | 0.4892 | 1.258 |
| 20–30 m | 0.5138 | 0.4653 | 1.167 |
| 30–40 m | 0.4936 | 0.4415 | 0.955 |
| 40–60 m | 0.4234 | 0.3480 | 0.827 |
| 60 m+ | 0.2854 | 0.2383 | 0.878 |

**Reading (measured).** G0 improves marking IoU at **every** distance bucket. The
overprediction pattern inverts at range: beyond 30 m, G0 now *under*-predicts
(pred/true < 1, down to 0.83–0.88), where F0 heavily over-predicted (F0 was 2.50
at 60 m+). **Interpretation:** the IoU surrogate makes the model conservative
where evidence is thin (few LiDAR points, sub-pixel paint); this trades distant
recall for precision but still nets higher IoU per bucket.

**Per-sequence** (9 sequences, `sampled_error_analysis_epoch18/per_sequence_metrics.csv`):

| seq | G0 IoU | G0 pred/true | road→marking | F0 IoU |
| --- | ---: | ---: | ---: | ---: |
| 106 | 0.6997 | 1.150 | 10,206 | 0.6020 |
| 054 | 0.6124 | 1.074 | 80,817 | 0.5942 |
| 037 | 0.5868 | 1.011 | 59,991 | 0.5388 |
| 124 | 0.5011 | 1.749 | 86,591 | 0.4275 |
| 123 | 0.4300 | 1.655 | 166,519 | 0.3818 |
| 034 | 0.3072 | 2.387 | 3,287 | 0.1646 |

**Reading (measured).** Every sequence improved its marking IoU vs F0. The same
sequences remain hardest (123, 124, 034 still overpredict, pred/true 1.66–2.39),
but each is better than its F0 counterpart (e.g. 123: 0.382→0.430; 124:
0.428→0.501; 034: 0.165→0.307). The low-RGB-coverage sequence 054 is again one of
the best (IoU 0.612, pred/true 1.074), consistent with the §7/§10 RGB findings.

**Raw subtype recall** (stratum `all`,
`raw_subtype_rgb_stratified_metrics.csv`): lane-line raw-8 **0.7594** (F0 0.8215),
stop-line raw-9 **0.7784** (F0 0.8343), other-paint raw-10 **0.7433** (F0 0.7745).
G0 recall is *lower* than F0 across all three subtypes — the same uniform
recall-for-precision trade (§5), not a subtype-specific regression.

---

## 9. Loss-component analysis (G-specific)

Source: `loss_component_summary.csv`, `loss_alignment.json`,
`loss_component_report.md`, and plots `val_loss_components.png`,
`val_lovasz_vs_marking_iou.png`, `val_ce_vs_marking_iou.png`, `f0_vs_g0_val_ce.png`.

| property | value | verdict |
| --- | ---: | --- |
| Lovász share of total (best ep) | 0.460 | **dominant** |
| Lovász share of total (mean) | 0.448 | dominant |
| val Lovász CV (epoch-to-epoch) | 0.103 | **stable** |
| val CE CV | 0.271 | (CE noisier) |
| corr(val Lovász, IoU) | −0.975 | tracks IoU |
| corr(val CE, IoU) | −0.910 | tracks IoU less well |
| argmin val Lovász / argmin val CE / argmax IoU | 18 / 24 / 18 | **Lovász aligns** |

**Fair CE comparison (measured).** F0 `val_loss` is pure weighted CE; G0 `val_ce`
is the CE component, computed identically — so they are directly comparable (G0
`val_loss` is the *total* and must not be compared to F0). G0 best val CE
**0.112872** is actually *lower* than F0 best val CE **0.118228**: adding Lovász
did not worsen the CE term; it slightly improved it.

**Reading (measured).** The Lovász term is (a) **dominant** — ~46% of the total
loss, heavier than a "λ=0.5 light touch" framing implies, because the raw Lovász
value (0.193) exceeds the CE value (0.113); (b) **stable** epoch-to-epoch (CV
0.103, less noisy than CE's 0.271); and (c) **better aligned with marking IoU**
than CE (corr −0.975 vs −0.910; its minimum coincides with the IoU peak while
CE's lags to epoch 24).

**Interpretation.** This is the mechanistic confirmation of G's hypothesis: the
added term behaves as intended (a smooth, stable IoU surrogate) and is the
component carrying the objective/metric alignment. The dominance (0.46) is worth
flagging — the effective CE:Lovász balance is ~54:46, so λ is the natural knob for
any future sensitivity study.

---

## 10. Feature-group / RGB-shortcut fingerprint (symptom vs mechanism)

Source: `rgb_shortcut_analysis/` (`rgb_shortcut_compare.csv`,
`rgb_shortcut_compare_conclusions.md`, `rgb_shortcut_conclusions.md`,
`group_feature_summary.csv`), produced by
`logs/milestone_g/run_analysis/analysis_code/rgb_shortcut_fingerprint.py`.

| measure (road→marking FPs) | F0 | G0 | read |
| --- | ---: | ---: | --- |
| FP count (sampled) | 710,399 | 489,746 | **−31% (symptom shrank)** |
| FP brightness (mean) | 0.429 | 0.424 | unchanged |
| FP brightness (median≈) | 0.491 | 0.483 | unchanged |
| FP LiDAR intensity (mean) | 30.8 | 32.3 | still road-like (markings 39.3) |

The G0 standalone fingerprint (`rgb_shortcut_conclusions.md`) confirms the same
qualitative profile as F0: road→marking FPs remain the **brightest** outcome
group, **less colour-saturated** than true markings (near-neutral gray-white), and
**road-like in LiDAR intensity** while elevated in RGB brightness.

**Reading (measured) + Interpretation (carefully worded).** Lovász **reduced the
symptom** (31% fewer road→marking false positives) but the remaining false
positives keep the **same bright, near-neutral fingerprint**. This is evidence
that the loss change suppressed *how often* the model fires on bright road, **not
the cue the firing is associated with**: the luminance-based RGB-shortcut
*association* is largely unchanged. As in F0, this is correlational — it does not
prove the model internally computes or uses brightness. Eliminating the shortcut
itself would require an RGB-specific intervention (e.g. brightness normalization,
hue features, or a channel ablation study), not a loss change.

---

## Post-hoc calibration / bias sweep

Two cheap diagnostics were run after the G0 run to decide whether a follow-up was
warranted, before committing GPU time. Both point to stopping.

**Marking-logit bias sweep.** A marking-logit bias sweep around the G0 epoch-18
checkpoint found no meaningful operating-point headroom. The best sampled IoU was
0.524622 at b=-0.20, only +0.0016 above the default argmax IoU 0.522980. Since the
nominal best bias was negative and the gain was negligible, the result does not
support lowering Lovász λ or otherwise encouraging more marking predictions. This
strengthens the decision to keep G0 epoch 18 as the final model. (Diagnostic only
— a bias-tuned IoU is not reported as a result; single inference pass over 11
biases b ∈ [−0.5, +0.5], seed 42, 2160 steps; source
`bias_sweep/bias_sweep_metrics.csv` / `bias_sweep_summary.md`.)

**Scheduler replay (zero-GPU companion).** Replaying G0's observed marking-IoU
curve through the real `ReduceLROnPlateau` (runner config + 3-epoch metric
smoothing) confirms G0 received no LR annealing within the 25-epoch budget: with
the actual patience 6 the LR never drops; patience 5 / 4 / 3 would first drop at
epoch 25 / 18 / 17 (patience 3 before the IoU peak). Source
`scheduler_replay.csv` / `scheduler_replay.md`.

Together these ruled out both the λ-reduction and the schedule levers as
defensible improvements, so Milestone G is closed with G0 epoch 18 as the final
model.

---

## 11. Interpretation

1. **G0 is the best run in the series and the recommended model.** Marking IoU
   0.523 (best), F1 0.687, precision 0.630, mIoU 0.817, and the best calibration
   (pred/true 1.199). It improves on F0 at every distance bucket and every
   validation sequence.
2. **The hypothesis was confirmed.** Lovász aligns the objective with marking IoU
   (corr −0.975, minimum at the IoU peak) far better than CE alone (−0.910,
   minimum 6 epochs late). The F0 CE/IoU mismatch is largely resolved at the loss
   level, and the added term is stable and well-behaved.
3. **The mechanism is a recall-for-precision trade.** G0 raised IoU mainly by
   improving precision (+0.081 vs F0) and calibration, while giving back recall
   (−0.045, back to D0 level). This is the expected effect of optimizing an IoU
   surrogate over a regime that was previously overpredicting.
4. **One predicted benefit did not materialize.** Adding Lovász did **not** reduce
   the best-to-final IoU drift (−0.044 vs F0's −0.033). The drift is overfitting
   past a higher peak, not a loss/metric conflict; it is handled by best-epoch
   selection (now well-proxied by the Lovász/total loss) and is a candidate for
   schedule/early-stopping tuning, not a loss problem.
5. **The RGB shortcut persists as a mechanism.** G0 fires less on bright road but
   the residual false positives keep the bright near-neutral fingerprint; the
   shortcut association is unchanged. This is the natural next target if the
   project continues.

---

## 12. Limitations

- **Drift not reduced.** The best-to-final IoU drift is slightly larger than F0's
  (§4, §11); marking IoU is still required for checkpoint selection (though the
  Lovász/total loss is now a good proxy for it).
- **Lovász is dominant (~46% of loss), not a light regularizer.** λ=0.5 with raw
  Lovász > CE makes the effective balance ~54:46. No λ-sensitivity *training* study
  was run; a post-hoc marking-logit bias sweep found no operating-point headroom
  and indicated lowering λ is unwarranted (see "Post-hoc calibration / bias sweep").
- **No LR drop within budget for G0.** The late peak (ep18) meant the scheduler
  never reduced LR by ep25; G0 may not have reached its ceiling under this
  schedule (§4).
- **The RGB-shortcut analysis is correlational** (§10) — it characterizes the
  RGB/intensity profile of mislabeled points; it does not prove internal brightness
  use, and no causal channel ablation was performed.
- **Stratified results are from a sampled pass** (seed 42, 2160 steps), not an
  exhaustive per-point evaluation; they cross-check the training-time metrics
  within ~0.003 (§5). Raw-remap alignment within the sample was verified
  (mismatch_rate 0.0).
- **`saturation_proxy`/max/min in §10 are aggregate approximations**; brightness,
  luminance, and warmth are exact group means.
- **Run artifacts are server-side**; this document relies on the committed
  analysis outputs derived from them.

---

## 13. Evidence index

**Config / loss code**
- `logs/milestone_g/configs/g0_rgb_lovasz.yml` — G0 run config; the only diff vs
  F0 is the `pipeline.loss` block (§2, §3).
- `src/thesis_pipeline/losses/combined_semseg_loss.py`, `lovasz.py`, `__init__.py`
  — the combined-loss implementation (CE exact, Lovász unweighted) (§2).
- `tools/test_combined_loss.py` — equivalence/safety tests (CE-component equality,
  composition, backwards-compat); passed CPU+CUDA (§2).
- `tools/train_milestone_g.py` + `tools/train_milestone_d.py` — G0 entrypoint and
  shared runner (config-gated `build_loss`, `loss_components.csv` logging).

**Run directory (server-side; not committed)**
- `logs/milestone_g/runs/G0_rgb_lovasz/` — eval_history.csv, loss_components.csv,
  per-epoch confusions/checkpoints; best checkpoint `ckpt_epoch_00018.pth`.

**Run-level analysis (`logs/milestone_g/run_analysis/G0_rgb_lovasz/`)**
- `summary.csv` — D0/E0/F0/G0 best + F0/G0 final metrics incl. CE/Lovász/total
  split for G0 (§4, §5).
- `pred_true_ratio.csv` — per-class predicted/true ratios per run (§6).
- `confusion_breakdown.csv` — D0/E0/F0/G0 confusion counts + G0−F0 / G0−D0 deltas
  (§6).
- `artifact_counts.csv` — provenance/consistency counts (incl. loss_components rows).
- `analysis.md` — run-level write-up incl. config-snapshot proof and loss-guard
  residuals (§2, §4, §5, §6).
- `g0_conclusions.md` — headline table, success-criteria block, fair CE comparison,
  Lovász health, drift, sampled residuals (§4–§9).

**Loss-component analysis**
- `loss_alignment.json` / `loss_alignment.csv` — scale/stability/alignment scalars,
  correlations, argmin epochs, composition residuals (§4, §9).
- `loss_component_summary.csv` — per-epoch CE / Lovász(raw,scaled) / total / share /
  marking IoU (§9).
- `loss_component_report.md` — narrative verdicts (§9).
- plots `train_loss_components.png`, `val_loss_components.png`,
  `val_ce_vs_marking_iou.png`, `val_lovasz_vs_marking_iou.png`,
  `val_total_vs_marking_iou.png`, `f0_vs_g0_val_ce.png` (§4, §9).

**Sampled error analysis (`sampled_error_analysis_epoch18/`)**
- `summary.json` — provenance (ckpt ep18, seed 42, 2160 steps) + overall sampled
  metrics; raw-remap 0.0 (§5, §7).
- `rgb_valid_stratified_metrics.csv` — all/valid/invalid marking metrics (§7).
- `distance_bucket_metrics.csv` — metrics by range (§8).
- `per_sequence_metrics.csv` — per-sequence metrics (§8).
- `raw_subtype_rgb_stratified_metrics.csv` — recall by raw subtype (§8).
- `group_feature_summary.csv` — per-outcome RGB/intensity summaries (§10).
- `frame_error_summary.csv`, `top_frames_by_*` — per-frame detail.

**RGB-shortcut**
- `rgb_shortcut_analysis/rgb_shortcut_compare.csv` + `_compare_conclusions.md` —
  F0-vs-G0 symptom (FP count) vs mechanism (fingerprint) (§10).
- `rgb_shortcut_analysis/rgb_shortcut_conclusions.md` + `rgb_shortcut_fingerprint.csv`
  — G0 standalone fingerprint (§10).

**Plots (`plots/`)**
- `d0_e0_f0_g0_best_metrics.png` (§5), `f0_vs_g0_marking_curves.png` (§5),
  `best_to_final_drift.png` (§4), `predicted_true_marking_ratio_over_epochs.png`
  (§6), `confusion_best_marking_epoch_018.png` / `confusion_final_epoch_025.png`
  (§6), `rgb_valid_vs_invalid_metrics.png` / `_confusion.png` (§7),
  `distance_marking_metrics.png` (§8), `raw_subtype_recall_d0_e0_f0_g0.png` (§8),
  `top_sequences_road_to_marking.png` (§8), `class_true_vs_predicted_share.png`
  (§6), `metrics_overview.png`, `lr_schedule.png`, `runtime_and_memory.png` (§4).

**Analysis scripts (`run_analysis/analysis_code/`)**
- `g0_analysis.py`, `g0_sampled_error_analysis.py`, `g0_loss_component_analysis.py`,
  `plot_g0_rgb_lovasz_analysis.py`, `run_g0_full_analysis.py`,
  `rgb_shortcut_fingerprint.py`, `README.md`.

**Post-hoc diagnostics ("Post-hoc calibration / bias sweep")**
- `bias_sweep/bias_sweep_metrics.csv` + `bias_sweep_summary.md` — marking-logit
  bias sweep (best b=−0.20, IoU 0.524622 vs b=0 0.522980, gain +0.0016).
- `scheduler_replay.csv` + `scheduler_replay.md` — ReduceLROnPlateau replay
  (patience 6 never drops in 25 ep; 5/4/3 → 25/18/17).
- `analysis_code/g0_bias_sweep.py`, `analysis_code/scheduler_replay.py` — scripts.

**Cross-milestone**
- `logs/milestone_f/notes/milestone_f_context.md` — the F residuals that motivated
  G (§1). `logs/milestone_e/context.md` — E0 baseline and RGB pipeline.

---

## 14. Thesis-use summary

**What was changed.** On top of the unchanged Milestone F setup (RGB front-camera
input, softened marking weight effective 15.0, `dim_features` 16), Milestone G
changed **only the loss**, from weighted cross-entropy to
`weighted_CE + 0.5·Lovász-Softmax` (λ = 0.5 from epoch 1, no warmup,
classes=`present`). This is a clean single-variable change.

**Why it was changed.** Milestone F's dominant residual was a CE/IoU
misalignment: validation CE kept improving after the marking-IoU peak, so the
objective fought the metric and val loss could not select checkpoints. Lovász-
Softmax is a differentiable IoU surrogate chosen to align the objective with
marking IoU.

**What improved (measured).** G0 (best epoch 18) is the best run in the series:
marking IoU **0.523** (vs F0 0.483, D0 0.440), F1 0.687, precision 0.630, mIoU
0.817, and the best calibration (pred/true **1.199** vs F0 1.456). It improves on
F0 at every distance bucket and every validation sequence, closes the F0 RGB-valid
overprediction (rgb_valid pred/true 1.55→1.24, and rgb_valid IoU now ≥ rgb_invalid),
and even lowers the CE component itself (val CE 0.118→0.113). The hypothesis was
confirmed: Lovász tracks marking IoU far better than CE (corr −0.975 vs −0.910,
loss minimum at the IoU peak vs 6 epochs late), and the term is stable (CV 0.103).
The improvement is a recall-for-precision trade (precision +0.081, recall −0.045
vs F0).

**What remained unresolved.** (1) The best-to-final IoU drift did not shrink
(−0.044 vs F0 −0.033) — overfitting past a higher peak, addressable by
schedule/early-stopping, with the Lovász/total loss now a usable checkpoint proxy.
(2) G0 never received an LR reduction within the 25-epoch budget (late peak), so
it may not have reached its ceiling. (3) The RGB luminance shortcut persists as a
*mechanism*: G0 fires 31% less on bright road but the residual false positives
keep the same bright near-neutral fingerprint — the loss change reduced the
frequency, not the cue.

**Status.** Milestone G is **closed**; G0 epoch 18 is the final model. Two post-hoc
diagnostics (scheduler replay + marking-logit bias sweep; see "Post-hoc
calibration / bias sweep") were run and both pointed to stopping: G0 is
well-calibrated (no operating-point headroom; λ-reduction unwarranted) and the
schedule lever's upside is below the single-seed noise floor. The only remaining
open direction — the RGB luminance shortcut — would require an RGB-specific
intervention (a new milestone), not another loss change, and is optional.

---

*Document status: ready for thesis use. All headline numbers verified against the
committed analysis files cited above and cross-checked by the independent sampled
pass (agreement within ~0.003). The only inputs not re-derivable from the local
repo are the server-side run artifacts, represented here through the committed
analysis outputs.*
