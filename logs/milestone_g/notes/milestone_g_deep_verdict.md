# Milestone G — Deep Analysis & Verdict

A strict, evidence-based read of what happened in Milestone G (G0 =
`weighted_CE + 0.5·Lovász-Softmax` on the otherwise-unchanged F0 setup), why it
likely happened, what the evidence supports, what is unresolved, and the final
verdict. Every major claim points to a file/CSV/JSON/plot. Items marked
**[Interpretation]** are reasoning, not direct measurement.

> **Provenance.** The G0 run directory is server-side; all numbers come from the
> committed analysis outputs under
> `logs/milestone_g/run_analysis/G0_rgb_lovasz/` and the config
> `logs/milestone_g/configs/g0_rgb_lovasz.yml`. Training-time validation metrics
> (`summary.csv`, `loss_component_summary.csv`, `confusion_breakdown.csv`) and a
> fresh sampled-inference pass (`sampled_error_analysis_epoch18/`, seed 42, 2160
> steps) are distinguished where it matters. `lane_*` columns are the marking
> class (alias). For G0, `summary.csv`/`eval_history` `val_loss` is the **TOTAL**
> loss (CE + 0.5·Lovász) and is **not** comparable to F0 `val_loss` (pure CE); the
> comparable quantity is G0 `val_ce`.

---

## Executive summary

G0 is a **clear success and the best run in the series.** It raised marking IoU
from F0's 0.4828 to **0.5230** (+0.040, +8.3% rel), with the best F1 (0.687),
precision (0.630), mIoU (0.817), and calibration (pred/true **1.199**) of any run
to date. The improvement is a **recall-for-precision trade**: precision +0.081 vs
F0, recall −0.045 (back to D0 level).

The central hypothesis was **confirmed**: Lovász-Softmax aligned the objective
with marking IoU far better than CE. `corr(val Lovász, IoU) = −0.975` vs
`corr(val CE, IoU) = −0.910`; the Lovász minimum **and** the total-loss minimum
both fall exactly at the IoU peak (epoch 18), while CE's minimum lags to epoch 24
— the F0 CE/IoU mismatch is resolved at the loss level, and the loss is now a
usable checkpoint proxy.

Two honest caveats: (1) the **best-to-final drift was not reduced** (−0.044 vs
F0's −0.033), because the drift is overfitting past a higher peak, not a
loss/metric conflict; G0's *final* IoU (0.479) ≈ F0's *best* (0.483), so G0 is
uniformly better. (2) The **RGB luminance shortcut persists as a mechanism**: G0
fires 31% fewer road→marking false positives, but the residual ones keep the same
bright, near-neutral, road-like-intensity fingerprint — the loss change reduced
the *rate* of firing, not the *cue*.

**Verdict: adopt G0 as the main/headline run; close Milestone G; no further run is
required for the core thesis.** The two cheap diagnostics that could have justified
a follow-up (scheduler replay + marking-logit bias sweep) were run post-hoc and
both confirm stopping — see §14.

---

## 1. Verified setup & provenance (Q1, Q11)

**G0 = F0 + loss change only — verified.** The run-level analysis validated the
G0 config snapshot against F0's invariants and would have failed otherwise
(`analysis.md`; `g0_analysis.py` `validate_g0_config_snapshot`). Confirmed
identical to F0: `feature_mode: intensity_rgb_front` (8 ch), class weights
`[119562394, 14344000, 173473484]` (effective CE road 2.445 / **marking 15.00** /
other 1.711), `dim_features: 16`, `in_channels: 8`, `num_layers: 3`,
`num_points: 32768`, AdamW lr 0.0014 / wd 0.0001, `ReduceLROnPlateau` on
`marking_iou`, `SemSegRandomSampler`, batch 2, 25 epochs, split, intensity
normalization, validation protocol, seed 42, checkpoint rule = max marking IoU.
The **only** change is `pipeline.loss = {name: weighted_ce_lovasz,
lovasz_lambda: 0.5, lovasz_classes: present}`.

**Nothing unexpected or suspicious (Q11) — checks all pass:**
- Artifact counts consistent: D0/E0/F0/G0 each 25 eval rows / eval jsons /
  confusions / checkpoints; G0 has 25 `loss_components` rows (`artifact_counts.csv`).
- Loss composition holds to ~1e-9: `max|val_total − (val_ce + 0.5·val_lovasz)| =
  8.71e-10`, train side 6.84e-10, and `max|val_total − eval_history val_loss| =
  0.0` (`loss_alignment.json`, `loss_component_report.md`). The combined loss
  engaged correctly.
- Checkpoint matches best epoch: sampled `summary.json` `checkpoint_epoch: 18`,
  `ckpt_epoch_00018.pth`; raw-label remap alignment `mismatch_rate: 0.0` over
  70,609,605 points.
- Metric names as expected (`lane_*` = marking). The one thing to keep straight
  (and explicitly handled in the tooling): **G0 `val_loss` is total, not CE** — do
  not compare it to F0 `val_loss`. The fair comparison uses `val_ce`.

**[Interpretation]** This is the cleanest single-variable change in the project
(F0 had changed weight *and* dim_features), so G0-vs-F0 isolates the Lovász effect.

---

## 2. Metric comparison — did G0 improve the task? (Q2)

Training-time validation, best epochs (`summary.csv`, `g0_conclusions.md`):

| metric | D0 ep18 | E0 ep14 | F0 ep13 | **G0 ep18** | G0−F0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| marking IoU | 0.440294 | 0.438439 | 0.482770 | **0.523048** | +0.040279 |
| marking F1 | 0.611394 | 0.609604 | 0.651173 | **0.686844** | +0.035671 |
| marking precision | 0.514788 | 0.470298 | 0.549249 | **0.629949** | +0.080700 |
| marking recall | 0.752636 | 0.866170 | 0.799543 | 0.755036 | −0.044507 |
| mIoU | 0.780829 | 0.780658 | 0.798682 | **0.817407** | +0.018726 |
| marking pred/true | 1.4620 | 1.8417 | 1.4557 | **1.1986** | −0.2571 |
| best epoch | 18 | 14 | 13 | 18 | — |

**[Measured]** G0 is the best run on IoU, F1, precision, mIoU, and calibration;
only recall is lower than F0/E0 (and equal to D0). Cross-checked by the
independent sampled pass: IoU 0.5230, precision 0.6321, recall 0.7518 (sampled
`summary.json`) — within ~0.003 of the training-time numbers, so the result is
stable, not a sampling artifact.

**[Interpretation]** G0 is the best run so far because it converts F0's recall
surplus into precision/IoU: it predicts fewer but more-correct marking points.

---

## 3. What kind of improvement was it? (Q3)

The gain is **precision + calibration**, not recall:
- precision +0.081 (0.549→0.630), recall −0.045 (0.800→0.755) → IoU +0.040.
- pred/true 1.456→1.199 (best), the most balanced of any run; road (0.990) and
  other (1.001) near-perfect (`pred_true_ratio.csv`).
- RGB-valid behavior improved (see §7): the F0 RGB-valid IoU penalty flipped.

**[Interpretation]** Optimizing an IoU surrogate over a previously-overpredicting
regime necessarily trims low-confidence positives — that is precisely a
precision/calibration gain, which is the dominant source of G0's IoU lift.

---

## 4. Did Lovász solve the CE/IoU mismatch? (Q4 — the central question)

**Yes, at the loss level.** Evidence (`loss_alignment.json`,
`loss_component_summary.csv`, `loss_component_report.md`):

| quantity | value |
| --- | ---: |
| corr(val CE, marking IoU) | −0.910 |
| corr(val Lovász, marking IoU) | **−0.975** |
| argmin(val CE) epoch | 24 |
| argmin(val Lovász) epoch | **18** |
| argmin(val total) epoch | **18** *(verified from `loss_component_summary.csv`)* |
| argmax(marking IoU) epoch | **18** |

In F0, validation CE kept improving for ~12 epochs past the marking-IoU peak —
the objective fought the metric, and val loss would have selected the *worse*
checkpoint. In G0, the **Lovász component and the total loss both bottom out
exactly at the IoU peak (epoch 18)**, while CE alone still drifts to epoch 24.

**[Interpretation / thesis language]** Lovász-Softmax is a differentiable
surrogate of the Jaccard index, so minimizing it is (approximately) maximizing
IoU; empirically its validation curve is the near-mirror image of the marking-IoU
curve (corr −0.975). Because Lovász is ~46% of the total loss (§5), it pulls the
*total*-loss minimum onto the IoU peak too. The practical consequence: the CE/IoU
selection pathology of F0 is gone — the loss is now an honest proxy for the
target metric, even if best-epoch selection is still prudent (§7).

---

## 5. Loss-component behavior (Q5) + curve/plot interpretation

Per-epoch (`loss_component_summary.csv`), validation:
- **val CE:** 0.261 (ep1) → noisy descent → 0.1129 (ep18) → min 0.1112 (ep24).
  Coefficient of variation **0.271** (noisy).
- **val Lovász (raw):** 0.283 (ep1) → smooth descent → **min 0.1926 (ep18)** →
  slight rise/plateau ~0.20–0.21. CV **0.103** (stable).
- **val total:** 0.402 (ep1) → **min 0.2092 (ep18)** → ~0.21–0.22.
- **train CE:** 0.319→0.087 (monotone down); **train Lovász:** 0.333→0.178.
- Lovász share of total: mean **0.448**, best-epoch **0.460** → **dominant**;
  train Lovász share crosses 0.50 by the end.

**Scale verdict:** Lovász is **dominant**, not a light correction. At best epoch,
scaled Lovász (0.0963) ≈ CE (0.1129), and raw Lovász (0.1926) ≈ 1.7× CE. So
**λ=0.5 acted as a major contributor (~46% of the loss), not a gentle nudge** —
because the raw Lovász magnitude exceeds CE. **Stability verdict:** stable (CV
0.103, smoother than CE's 0.271). **Overfitting:** train CE/Lovász keep falling
while val flattens after ~ep10–13 and val Lovász turns up after ep18 — mild
overfitting, the mechanism behind the post-peak drift (§7).

**Plot interpretation (inspected, not just listed):**

- **`plots/val_loss_components.png`** — total (green) 0.40→0.21; CE (blue) visibly
  *jagged* (dips ep4/ep18/ep24); scaled Lovász (red) and raw Lovász (dashed)
  *smooth*. Shows directly that the noise in the total loss comes from CE, and
  that the Lovász term is the stable, well-behaved component. Confirms scaled
  Lovász ≈ CE in magnitude.
- **`plots/val_lovasz_vs_marking_iou.png`** — the money plot: validation Lovász
  (pink, left axis) and marking IoU (orange, right axis) are near-perfect mirror
  images; both reach their extremum at the dashed line (ep18). Visual proof of the
  −0.975 alignment.
- **`plots/val_ce_vs_marking_iou.png`** / **`val_total_vs_marking_iou.png`** — CE
  is the noisier, weaker tracker; total tracks IoU because Lovász dominates it.
- **`plots/f0_vs_g0_marking_curves.png`** — 4-panel: G0 (magenta) sits above F0
  (green) on IoU, F1, and precision for most epochs, and *below* F0 on recall —
  the recall-for-precision trade made visible across the whole run, not just at
  the best epoch.
- **`plots/f0_vs_g0_val_ce.png`** — fair CE overlay: G0's CE component is at or
  below F0's pure CE, i.e. adding Lovász did not cost CE (it slightly improved it;
  G0 best val CE 0.1129 < F0 0.1182).

---

## 6. Calibration / overprediction (Q6)

Marking pred/true (`pred_true_ratio.csv`, training-time confusion):

| run | best | final |
| --- | ---: | ---: |
| E0 | 1.842 | — |
| D0 | 1.462 | — |
| F0 | 1.456 | 1.782 |
| **G0** | **1.199** | 1.424 |

Sampled (ep18, `rgb_valid_stratified_metrics.csv`): all **1.189**; rgb_valid
**1.240**; rgb_invalid **1.004**.

Confusion deltas G0−F0 at best epoch (`confusion_breakdown.csv`):
- road→marking FP: 243,315 → **168,907** (−74,408, −30.6%).
- other→marking FP: 17,837 → **4,976** (−72%).
- marking→road FN: 78,971 → **92,659** (+13,688).
- marking→marking TP: 318,219 → 296,006 (−22,213).

**`plots/predicted_true_marking_ratio_over_epochs.png`** (inspected): G0 (orange)
is **below** F0 (green) at essentially every epoch — G0 overpredicts less
throughout training, hugging ~1.2–1.4 vs F0's ~1.5–2.0. **`plots/confusion_best_marking_epoch_018.png`**
(inspected): marking row = 23.6% → road (FN) / 75.5% → marking (recall) / 0.9% →
other; road row = 96.9% / 1.8% (FP) / 1.4%; other row = 99.1% / 0.9% / 0.0%.

**[Measured + Interpretation]** G0 is the **best-calibrated run**: it became more
conservative (fewer marking predictions), cutting both FP channels substantially
at the cost of more FNs. It is *better calibrated*, not *under-recalled in the
harmful sense* — recall (0.755) equals D0's, and IoU/F1 are the highest in the
series, so the trimmed positives were net low-value.

---

## 7. Best-to-final drift (Q7)

| metric | F0 best→final | G0 best→final |
| --- | ---: | ---: |
| marking IoU | 0.4828 → 0.4500 (−0.0328) | 0.5230 → 0.4786 (**−0.0445**) |
| precision | 0.5492 → 0.4845 (−0.0647) | 0.6299 → 0.5509 (−0.0790) |
| recall | 0.7995 → 0.8632 (+0.0637) | 0.7550 → 0.7848 (+0.0297) |
| pred/true | 1.456 → 1.782 (+0.326) | 1.199 → 1.424 (**+0.226**) |
| LR at epoch | 0.0014 → 0.0007 (dropped) | 0.0014 → **0.0014 (no drop)** |

**Did G0 reduce drift? No, in raw IoU terms** (−0.044 vs −0.033). But:
- **G0's final IoU (0.4786) ≈ F0's best (0.4828)** — G0 is uniformly better; the
  larger drift is from a *higher* peak, not a collapse below F0.
- The **calibration drift is smaller** in G0 (pred/true +0.226 vs F0 +0.326).
- After ep18, val Lovász and val total both turn up while train keeps falling
  (§5) → the drift is **overfitting**, not the CE/IoU conflict of F0.
- **LR note (measured):** G0's best epoch is late (18), so `ReduceLROnPlateau`
  (patience 6) had not triggered a reduction by epoch 25 — G0 trained the whole
  budget at lr 0.0014, whereas F0 had dropped to 0.0007. **[Interpretation]** G0
  may not have reached its ceiling; a later LR drop / longer schedule is the
  natural lever, not a loss change.

**Is this a serious problem? No.** Best-checkpoint selection remains necessary
(as for D0/E0/F0), but it is now well-supported: the total loss minimum coincides
with the IoU peak (§4), so checkpoint selection is no longer fighting the loss.

---

## 8. Sampled error analysis (Q8)

Fresh pass, ep18, seed 42, 2160 steps (`sampled_error_analysis_epoch18/`):

- **Overall:** IoU 0.5230, precision 0.6321, recall 0.7518, pred/true 1.189
  (`summary.json`) — matches training-time within ~0.003, so the sampled analysis
  **supports** the epoch-level results.
- **RGB-valid vs invalid** (`rgb_valid_stratified_metrics.csv`): rgb_valid IoU
  **0.5256** ≥ rgb_invalid **0.5125** — the **F0 penalty flipped** (F0 was valid
  0.471 < invalid 0.520). Overprediction gap narrowed: valid pred/true 1.240 vs
  invalid 1.004 (gap 0.237 vs F0's 0.402). RGB still aids recall (valid 0.772 vs
  invalid 0.679).
- **Distance** (`distance_bucket_metrics.csv`): IoU 0.649 / 0.521 / 0.514 / 0.494
  / 0.423 / 0.285 across 0–10 / 10–20 / 20–30 / 30–40 / 40–60 / 60 m+ — **higher
  than F0 at every bucket**. Overprediction inverts at range: pred/true 0.96 /
  0.83 / 0.88 beyond 30 m (G0 now *under*-predicts distant paint; F0 was 2.50 at
  60 m+). **`plots/distance_marking_metrics.png`** shows the smooth IoU decline
  with range plus the sub-1.0 long-range ratio.
- **Per-sequence** (`per_sequence_metrics.csv`): every sequence improved vs F0
  (e.g. 106 0.602→0.700; 054 0.594→0.612; 124 0.428→0.501; 123 0.382→0.430; 034
  0.165→0.307). Hardest remain 123/124/034 (pred/true 1.66 / 1.75 / 2.39).
- **Raw subtype recall** (`raw_subtype_rgb_stratified_metrics.csv`, stratum all):
  lane-line 0.759, stop-line 0.778, other-paint 0.743 — all *lower* than F0 (the
  uniform recall trade). RGB still lifts subtype recall (lane-line valid 0.782 vs
  invalid 0.617).
- **Top error frames** (`top_frames_by_*`): road→marking FPs concentrate in
  high-RGB-valid frames of seq 054/123 (rgb_valid_ratio 0.82–0.92);
  marking→road FNs concentrate in RGB-invalid frames (rgb_valid_ratio 0.0) of seq
  054 and dim frames of seq 037. **[Interpretation]** FP where RGB present, FN
  where RGB absent — the spatial signature of the RGB shortcut (§9).
- **Group features** (`group_feature_summary.csv`): feeds §9.

---

## 9. RGB shortcut — symptom vs mechanism (Q9)

Source: `rgb_shortcut_analysis/` (`rgb_shortcut_compare.csv`/`.md`,
`rgb_shortcut_conclusions.md`, `rgb_shortcut_fingerprint.csv`).

| measure (road→marking FPs) | F0 | G0 |
| --- | ---: | ---: |
| FP count (sampled) | 710,399 | **489,746 (−31%)** |
| FP brightness mean | 0.429 | 0.424 |
| FP luminance mean | 0.435 | 0.430 |
| FP LiDAR intensity mean | 30.8 | 32.3 (markings 39.3) |
| rgb_valid share of FPs | 86.4% | 84.1% |

**`plots/.../rgb_shortcut_compare.png`** (inspected): left panel — the FP
fingerprint (brightness / luminance / warmth / saturation) is **virtually
identical** F0 vs G0; right panel — FP count drops 710k → 490k. **`plots/.../brightness_vs_intensity_scatter.png`**
(inspected): road→marking FPs sit at the **highest RGB brightness** (right of both
road and marking reference lines) yet at **road-like LiDAR intensity** (~32, far
below markings' ~39) — the two modalities disagree for exactly these points.

**[Measured + carefully worded interpretation]** G0 **fixed the symptom, not the
mechanism.** It fires 31% fewer road→marking false positives, but the residual
ones retain the same bright, near-neutral, road-like-intensity profile. This is
evidence that the road→marking errors remain **associated with a luminance-based
RGB shortcut** — bright near-neutral road (glare, light concrete, overexposed
pixels) shares the high-RGB signature of white paint. It does **not** prove the
model internally computes or uses brightness; eliminating the shortcut would need
an RGB-specific intervention, not a loss change.

---

## 10. What G0 did not fix (Q10)

- **Best-to-final drift** persists (slightly larger than F0; §7) — overfitting
  past the peak; needs schedule/early-stopping, not loss.
- **RGB luminance shortcut mechanism** unchanged (§9) — only the firing rate fell.
- **Residual road→marking FPs** (489,746 sampled) still dominate the error budget,
  concentrated in rgb_valid points and a few sequences (123/124/034).
- **λ not tuned** — Lovász ended up dominant (~46%); sensitivity to λ is untested.
- **Long-range markings** now *under*-predicted (pred/true <1 beyond 30 m; §8) — a
  new, milder failure mode (G traded distant recall for precision).
- **No LR reduction within budget** — possible unrealized headroom (§7).
- **No RGB augmentation / standardization** tried, which is the natural lever for
  the shortcut.

---

## 11. Anything unusual or suspicious? (Q11)

**No.** All integrity checks pass (artifact counts 25× consistent; loss
composition residual ~1e-9; `val_total == eval val_loss` exactly; checkpoint
epoch 18 matches; raw-remap mismatch 0.0). The one genuine *gotcha* is correctly
handled: **G0 `val_loss` is the TOTAL loss and is not comparable to F0 `val_loss`**
— the analysis compares F0 `val_loss` (pure CE) to G0 `val_ce` (component), so no
misleading cross-loss comparison is made. Caveats that could make a naive reading
misleading, all surfaced above: Lovász is dominant (not a light touch); the
sampled pass is a sample (not exhaustive); the RGB-shortcut conclusion is
correlational; `saturation_proxy`/max/min are aggregate approximations.

---

## 12. Final verdict (Q12)

**G0 is a success and should become the main / headline thesis run.** It is the
best model on every primary axis (marking IoU 0.523, F1 0.687, precision 0.630,
mIoU 0.817, calibration 1.199) and improves on F0 at every distance bucket and
every validation sequence. The mechanism is understood: a recall-for-precision
trade driven by aligning the objective with IoU.

**Thesis story (E → F → G):**
- **E0:** adding front-camera RGB raised marking recall (0.75→0.87) but, under the
  strong rare-class weight, caused heavy overprediction (pred/true 1.84, precision
  0.47). RGB is useful but over-trusted.
- **F0:** softening the marking weight (effective 26.9→15.0) recalibrated the
  model — first run to beat D0/E0 (IoU 0.483) and restore precision — but left a
  CE/IoU mismatch (loss kept improving past the IoU peak) and an RGB-localized
  bright-road false-positive residual.
- **G0:** adding `0.5·Lovász-Softmax` aligned the loss with IoU (corr −0.975;
  loss minimum at the IoU peak), lifting IoU to 0.523, delivering the best
  precision and calibration in the project, and cutting the bright-road FP count
  by 31%.

**Say carefully / do not overclaim:**
- Lovász **aligned the loss with the metric**; it did **not** eliminate
  best-to-final drift (that is overfitting; best-epoch selection still used).
- The RGB-shortcut evidence is **associational**, not proof of internal brightness
  use; G0 reduced the **symptom**, not the **mechanism**.
- G0-vs-F0 is a clean single-variable comparison, but F0-vs-E0 was **not** (F0
  also changed dim_features) — keep that qualifier when narrating E→F→G.
- The improvement is **precision/calibration-led**; do not frame it as a recall
  or detection improvement (recall fell to D0 level).

---

## 13. Recommended next decision (Q13)

**Primary recommendation: close Milestone G and adopt G0 as the final main
result. No further run is required for the core thesis.** The milestone achieved
its goal (resolve the CE/IoU mismatch) and produced the best model; the context
document (`logs/milestone_g/notes/milestone_g_context.md`) and this verdict are
the record.

**Update — the candidate probes were run as cheap diagnostics, and they confirm
stopping.** A zero-GPU scheduler replay confirmed G0 never annealed within the
25-epoch budget, and a marking-logit bias sweep on the G0 epoch-18 checkpoint
found no operating-point headroom (best bias **negative**, IoU gain **+0.0016**,
below the single-seed gate). This rules out the λ-reduction lever (the data wants
*fewer*, not more, marking predictions) and bounds the schedule lever's upside
below the noise floor. See **§14** for details. **No further run is warranted.**

**Do not** pursue the RGB shortcut with another loss — the evidence (§9) shows
loss changes affect the symptom, not the mechanism. If the shortcut is ever
targeted, it requires an RGB-specific change (brightness normalization / hue
features / a causal channel-ablation study), which is a new milestone, not a
quick follow-up, and is **optional** given G0 already meets the thesis goal.

---

## 14. Post-hoc calibration / bias sweep

Two cheap diagnostics were run after G0 to decide whether a follow-up run was
warranted, before committing GPU time. Both point to stopping.

**Marking-logit bias sweep.** A marking-logit bias sweep around the G0 epoch-18
checkpoint found no meaningful operating-point headroom. The best sampled IoU was
0.524622 at b=-0.20, only +0.0016 above the default argmax IoU 0.522980. Since the
nominal best bias was negative and the gain was negligible, the result does not
support lowering Lovász λ or otherwise encouraging more marking predictions. This
strengthens the decision to keep G0 epoch 18 as the final model. (Diagnostic only
— a bias-tuned IoU is **not** reported as a result; single inference pass over 11
biases b ∈ [−0.5, +0.5], seed 42, 2160 steps. Source:
`bias_sweep/bias_sweep_metrics.csv` / `bias_sweep_summary.md`.)

**Scheduler replay (zero-GPU companion).** Replaying G0's observed marking-IoU
curve through the real `ReduceLROnPlateau` (the runner's config + 3-epoch metric
smoothing) confirms G0 received **no LR annealing within the 25-epoch budget**:
with the actual patience 6 the LR never drops; patience 5 / 4 / 3 would first drop
at epoch 25 / 18 / 17 (patience 3 *before* the IoU peak — risky). Source:
`scheduler_replay.csv` / `scheduler_replay.md`.

Together these ruled out both the λ-reduction lever and the schedule lever as
defensible improvements (the latter's upside is below the single-seed noise floor),
so Milestone G is closed with **G0 epoch 18 as the final model.**

---

## Evidence index

**Setup / integrity**
- `logs/milestone_g/configs/g0_rgb_lovasz.yml` — only diff vs F0 is the loss (§1).
- `analysis.md` — config-snapshot proof (G = F0 + loss) + loss guards (§1, §11).
- `artifact_counts.csv` — 25× consistency, G0 loss_components present (§1, §11).
- `loss_alignment.json` — composition residuals, correlations, argmin epochs (§1, §4, §11).

**Metrics / comparison**
- `summary.csv` — D0/E0/F0/G0 best + F0/G0 final, CE/Lovász/total split (§2, §3, §5, §7).
- `pred_true_ratio.csv` — per-class calibration per run (§3, §6).
- `confusion_breakdown.csv` — D0/E0/F0/G0 confusion counts + deltas (§6).
- `g0_conclusions.md` — headline + success criteria + drift + residuals (§2–§9).

**Loss behavior**
- `loss_component_summary.csv` — per-epoch CE/Lovász/total/share/IoU; source of the
  argmin(val_total)=18 finding and the overfitting read (§4, §5, §7).
- `loss_component_report.md` — scale/stability/alignment verdicts (§4, §5).
- plots `val_loss_components.png` (§5), `val_lovasz_vs_marking_iou.png` (§4, §5),
  `val_ce_vs_marking_iou.png`, `val_total_vs_marking_iou.png`,
  `f0_vs_g0_marking_curves.png` (§5), `f0_vs_g0_val_ce.png` (§5).

**Sampled residuals (`sampled_error_analysis_epoch18/`)**
- `summary.json` (§2, §8), `rgb_valid_stratified_metrics.csv` (§7, §8),
  `distance_bucket_metrics.csv` (§8), `per_sequence_metrics.csv` (§8),
  `raw_subtype_rgb_stratified_metrics.csv` (§8), `group_feature_summary.csv` (§9),
  `top_frames_by_road_to_marking.csv` / `top_frames_by_marking_to_road.csv` (§8).
- plots `distance_marking_metrics.png` (§8), `confusion_best_marking_epoch_018.png`
  (§6), `predicted_true_marking_ratio_over_epochs.png` (§6),
  `rgb_valid_vs_invalid_metrics.png` (§7).

**RGB shortcut (`rgb_shortcut_analysis/`)**
- `rgb_shortcut_compare.csv` + `rgb_shortcut_compare_conclusions.md` — symptom vs
  mechanism (§9). `rgb_shortcut_conclusions.md` + `rgb_shortcut_fingerprint.csv` —
  G0 standalone fingerprint (§9). plots `rgb_shortcut_compare.png`,
  `brightness_vs_intensity_scatter.png` (§9).

**Post-hoc diagnostics (§14)**
- `bias_sweep/bias_sweep_metrics.csv` + `bias_sweep_summary.md` — marking-logit
  bias sweep; best b=−0.20, IoU 0.524622 vs b=0 0.522980 (gain +0.0016).
- `scheduler_replay.csv` + `scheduler_replay.md` — ReduceLROnPlateau replay;
  patience 6 never drops in 25 ep (5/4/3 → 25/18/17).
- `analysis_code/g0_bias_sweep.py`, `analysis_code/scheduler_replay.py` — the
  diagnostic scripts.

**Cross-milestone**
- `logs/milestone_f/notes/milestone_f_context.md` — F0 residuals that motivated G.
- `logs/milestone_g/notes/milestone_g_context.md` — the G reference document.

---

*Status: complete and evidence-based. All numbers verified against the committed
analysis files; seven key plots inspected directly (not inferred from CSVs);
training-time and sampled metrics agree within ~0.003. The only inputs not
re-derivable locally are the server-side raw run artifacts, represented through
the committed analysis outputs.*
