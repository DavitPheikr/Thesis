# Milestone B Project Context

## Training-Ready Pipeline: From Verified Build to Mechanically Trainable Baseline

---

## 0. Milestone-Boundary Correction — Read Before Anything Else

The revised thesis master document defines "Milestone A" as including a tiny real training sanity
run, prediction decoding, and at least one qualitative output. **None of those things happened.**

The actual Day 1–4 work (Milestone A as implemented) stopped at:

- a verified environment and import chain
- readable PandaSet data on one sequence and one frame
- a working single-frame adapter returning `point / feat / label`
- a parsing config scaffold
- a locally constructing RandLA-Net instance
- a documented stop boundary forbidding real training

**The tiny real training sanity run is a Milestone B deliverable, not a Milestone A assumption.
Milestone B inherits the Day 1–4 foundation only.** Any document or agent that assumes a prior
sanity run, prior decoded predictions, or prior qualitative outputs is working from stale
aspirational text, not operational truth.

---

## 1. Purpose of Milestone B

Milestone B establishes that the project has a **mechanically complete training pipeline**: one
that can accept real PandaSet data across a frozen split, apply thesis-correct preprocessing, feed
samples through a proper Open3D-ML dataset interface, and execute at least a few training
iterations without crashing, with finite, non-exploding loss.

Milestone B does **exactly** five things:

1. verifies multi-frame and multi-sequence data-loading safety beyond the one verified sample
2. discovers the semseg-enabled sequence pool and freezes the train / val / test split
3. measures real intensity statistics and class-count statistics on the training split
4. builds a proper Open3D-ML-compatible dataset class replacing the single-frame adapter
5. proves the assembled pipeline can mechanically execute training steps with finite, stable loss

Milestone B is **not** trying to produce a trained model, interpret any performance number, or
make any claim about lane segmentation quality. It is trying to remove every remaining
mechanical obstacle between the current build state and the first real training run.

---

## 2. What the Three Certainty Categories Mean

All facts in Milestone B documents are assigned to one of three categories. Coding agents must
respect these distinctions:

### Category A — Fixed Project Truths

These are invariants from the master specification. They must not change during Milestone B and
cannot be overridden by any local discovery.

### Category B — Sample-Verified Local Truths

These were verified locally on **sequence `001`, frame `0`, sensor `1`**. They are trusted for
that sample. They cannot be assumed to hold across all sequences, all frames, or all sensors
without multi-sequence verification.

### Category C — Still-Unverified for Multi-Sequence Use

These have never been tested in a multi-sequence or multi-frame context. They must be
explicitly verified during Milestone B before any large-scale scan is trusted.

---

## 3. Current Verified State Entering Milestone B

### 3.1 Category A — Fixed Project Truths

These are fixed and must not change:

| Property                | Value                                                                                                            |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Task                    | Point-wise semantic segmentation on PandaSet LiDAR                                                               |
| Active classes          | `road` (1), `lane` (2), `other` (3); `ignore` (0) excluded from loss                                             |
| Positive class          | Lane Line Marking **only** (raw PandaSet ID `8`)                                                                 |
| Sensor                  | Forward-facing LiDAR only, sensor ID `1`                                                                         |
| Coordinate frame        | Ego-local                                                                                                        |
| Feature concept         | Ego-local `xyz` (3 channels) + one processed intensity channel (1 channel)                                       |
| `in_channels`           | `4` (fixed: xyz concatenated with one feature channel)                                                           |
| `num_classes`           | `3`                                                                                                              |
| `ignored_label_inds`    | `[0]`                                                                                                            |
| Model                   | RandLA-Net via Open3D-ML                                                                                         |
| Training                | From scratch — no pretrained weights                                                                             |
| Adapter contract        | Returns `point` / `feat` / `label`                                                                               |
| Remap                   | `{1,2,3,4}→0`, `7→1`, `8→2`, all else→`3`                                                                        |
| Intensity pipeline spec | Clip to training-split P0.5/P99.5; standardize with training-split clip-mean and clip-std; output as N×1 float32 |
| Split concept           | Sequence-level; val=9, test=9, train=remainder; both holdouts must contain ≥1 lane-bearing sequence              |

### 3.2 Category B — Sample-Verified Local Truths

Verified on **sequence `001`, frame `0`, sensor `1`** only. Do not generalize without Milestone B
multi-sequence verification.

| Property                                                                   | Verified Value                                                                                                |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Project root                                                               | `/home/pheikara/University/Y3S2/Thesis/Pipeline`                                                              |
| Virtual environment                                                        | `panda` at `…/Pipeline/panda`                                                                                 |
| Python                                                                     | `3.12.3`                                                                                                      |
| PyTorch                                                                    | `2.2.2+cu121`                                                                                                 |
| Open3D                                                                     | `0.19.0`                                                                                                      |
| NumPy                                                                      | `1.26.4`                                                                                                      |
| pandas                                                                     | `3.0.2`                                                                                                       |
| `open3d.ml.torch`, `RandLANet`, `pandaset`, `geometry.lidar_points_to_ego` | Import verified                                                                                               |
| `torch.cuda.is_available()`                                                | `True`                                                                                                        |
| Dataset root                                                               | `logs/dataset_root.txt` (canonical path file)                                                                 |
| Total local sequence directories                                           | `103`                                                                                                         |
| Devkit                                                                     | Patched local install with `.pkl` frame fallback; reinstall: `pip install --no-deps ./pandaset-devkit/python` |
| Sequence `001` frame count                                                 | 80 LiDAR frames, 80 semseg frames                                                                             |
| LiDAR columns                                                              | `x, y, z, i, t, d`                                                                                            |
| Sensor ID column                                                           | `d`                                                                                                           |
| Forward sensor ID                                                          | `1`                                                                                                           |
| Forward-only point count on verified sample                                | `62,285`                                                                                                      |
| Intensity column                                                           | `i`, range `[0, 255]` on verified sample                                                                      |
| Pose access path                                                           | `seq.lidar.poses[frame_idx]`                                                                                  |
| Ego conversion                                                             | `geometry.lidar_points_to_ego` (verified to exist; behavior across all sequences not verified)                |
| Verified sample adapter output                                             | `point (62285,3)`, `feat (62285,1)`, `label (62285,)`, unique labels `{0,1,2,3}`                              |
| Lane fraction on verified sample                                           | `0.0025688368`                                                                                                |
| Verified raw IDs                                                           | `4→Reflection, 7→Road, 8→Lane Line Marking, 9→Stop Line, 10→Other Road Marking`                               |
| `src/thesis_pipeline/core/pandaset_compat.py`                              | Exists; `get_frame_count()` handles container-shape variability                                               |
| Config scaffold                                                            | `configs/randlanet_pandaset_ff_lane3.yml` parses; RandLA-Net constructs locally                               |
| `real_training_allowed`                                                    | `false` in config (deliberate stop boundary)                                                                  |

### 3.3 Category C — Still-Unverified for Multi-Sequence Use

These must be treated as unknown until Milestone B verification completes. Do not assume these
hold for sequences other than `001` or frames other than `0`:

- `set_sensor(1)` persistence across repeated consecutive frame accesses (tested on one frame)
- Semseg annotation presence across all 103 sequences (only one sequence confirmed)
- Semseg frame count equals LiDAR frame count within each sequence (not checked)
- Devkit `.pkl` fallback behavior across all 103 sequences
- Lane-class point presence per sequence (measured on one frame of one sequence)
- Open3D-ML dataset class interface (base class, required methods, return key names)
- Open3D-ML pipeline interface (constructor, training API, iteration control)
- `SemSegLoss` class_weights format (list length, ordering, dtype, ignore-slot policy)
- Whether `SemSegSpatiallyRegularSampler` is available in Open3D `0.19.0`
- Whether xyz+feat concatenation happens in the pipeline, dataset class, or model
- GPU memory behavior under actual training workload
- Post-grid-subsampling point count under `grid_size: 0.04`

---

## 4. What is Already Fixed — Must Not Change During Milestone B

### 4.1 Remap (verified locally, already implemented)

```
{1, 2, 3, 4} → 0 (ignore)
7             → 1 (road)
8             → 2 (lane)
all else      → 3 (other)
```

Single canonical implementation: `src/thesis_pipeline/adapters/pandaset_ff_lane3.py::remap_raw_pandaset_ids()`.
No new script may reimplement this logic. All Milestone B scripts that need remapping must
import from this location.

### 4.2 PandaSet compatibility helper (verified locally, already implemented)

`src/thesis_pipeline/core/pandaset_compat.py::get_frame_count(seq)` — handles `seq.lidar.data`
container-shape variability. All Milestone B scripts that count frames must use this helper. No
script may call `len(seq.lidar.data)` directly.

### 4.3 Intended model scaffold values (design intent; API acceptance locally unverified)

| Parameter            | Intended value                  | Status                                                                              |
| -------------------- | ------------------------------- | ----------------------------------------------------------------------------------- |
| `num_layers`         | `3`                             | Design intent; verify API accepts it                                                |
| `sub_sampling_ratio` | `[4, 4, 4]`                     | Design intent                                                                       |
| `grid_size`          | `0.04`                          | Design intent                                                                       |
| `num_points`         | `16384`                         | Design intent; may require temporary reduction for GPU OOM during sanity check only |
| `num_neighbors`      | `16`                            | Design intent                                                                       |
| sampler              | `SemSegSpatiallyRegularSampler` | Design intent; verify availability in Open3D `0.19.0`                               |

`num_points` may be temporarily reduced for the sanity check if GPU OOM forces it. Any such
reduction must be documented as a sanity-check-only concession. The intended value of `16384`
must be restored before full training. This is the **only** scaffold value that may be temporarily
changed during Milestone B.

---

## 5. What Milestone B Must Produce and Prove

Milestone B is trying to prove **mechanical pipeline completeness** across seven claims:

1. **Multi-sequence data loading is reliable** — `set_sensor(1)` persists; semseg alignment holds across multiple frames; the devkit works on all semseg-enabled sequences.
2. **The local dataset has enough semseg-enabled sequences** for the planned split.
3. **Real measured statistics replace provisional placeholders** — intensity clipping bounds and normalization statistics come from the training split; class counts are measured.
4. **A proper dataset class serves the full pipeline** — not just one-frame adapter calls.
5. **Loss weights are in the correct format** for the locally installed `SemSegLoss`.
6. **The full pipeline chain is mechanically sound** — from raw PandaSet data to a finite, non-exploding training loss.
7. **Explicit stop conditions are set** before any full training run begins.

---

## 6. Canonical Source of Truth for Measured Artifacts

To prevent drift between files, one explicit rule governs authority:

| Artifact                               | Canonical location                                                                                       |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Train sequence IDs                     | `configs/splits/train.txt` (one ID per line)                                                             |
| Val sequence IDs                       | `configs/splits/val.txt` (one ID per line)                                                               |
| Test sequence IDs                      | `configs/splits/test.txt` (one ID per line)                                                              |
| All measured statistics + loss weights | `logs/milestone_b_training_statistics.json` — **single authoritative source**                            |
| Open3D-ML loss interface               | `logs/milestone_b_loss_interface.json`                                                                   |
| Open3D-ML dataset class interface      | `logs/milestone_b_dataset_interface.json`                                                                |
| Open3D-ML pipeline interface           | `logs/milestone_b_pipeline_interface.json`                                                               |
| Per-sequence intensity samples         | `logs/stats_per_seq/{seq_id}.npy`                                                                        |
| YAML config                            | `configs/randlanet_pandaset_ff_lane3.yml` — mirrors the statistics JSON; if they disagree, the JSON wins |
| Dataset class                          | Reads from split files and statistics JSON; must contain **no hardcoded statistics values**              |
| Stop conditions                        | `logs/milestone_b_stop_conditions.txt`                                                                   |

**Drift rule:** If the YAML config disagrees with `logs/milestone_b_training_statistics.json` on any
numerical value (intensity bounds, class weights), the JSON wins. The `check_dataset_class.py`
script enforces this programmatically.

---

## 7. Phase-by-Phase Meaning of the Work

### Phase 1: Pre-flight Verification, Sequence Pool Discovery, and Split Freeze

Establishing multi-sequence data-loading safety and data-pool reality.

Answers: Does `set_sensor(1)` persist? How many sequences have semseg? Which have lane points?
What is the exact split?

### Phase 2: Open3D-ML Interface Discovery

Establishing what the local installed Open3D-ML stack actually expects.

Answers: What format does `SemSegLoss` expect for class weights? Where does xyz+feat
concatenation happen? What does the dataset class API look like? How is training invoked?

This phase must happen **before** Phase 3 and Phase 4 code is written. Phases 3 and 4 are
gated on the three interface JSON artifacts produced here.

### Phase 3: Training-Split Measurement and Loss Weights

Establishing data-measurement reality.

Answers: What are the actual intensity clipping bounds and normalization statistics on the training
split? What are the actual class counts? What loss weight candidates are safe for the sanity run?

### Phase 4: Dataset Class and Config Assembly

Establishing pipeline-interface reality.

Delivers: A proper Open3D-ML dataset class built from the discovered interface. An updated
YAML config with measured values. Verified dataset output shapes and YAML/JSON consistency.

### Phase 5: Pipeline Sanity Check and Stop Conditions

Establishing end-to-end mechanical soundness.

Delivers: A fully resolved sanity training script (no placeholder markers). A passing sanity run
with finite, non-exploding loss. Explicit stop conditions before full training.

---

## 8. Non-Goals for Milestone B

The following are explicitly outside Milestone B scope:

- full baseline training to convergence
- any performance claim or metric evaluation
- any ablation experiment
- any qualitative visualization of model predictions
- final hyperparameter tuning
- lane-aware patch oversampling implementation (only the need-assessment is done here)
- focal loss
- test-set evaluation
- thesis write-up
- any claim that the model "works" or "learns"

Lane-aware patch oversampling: Milestone B computes `E = num_points × p_lane / 64` as a
global approximation and records whether oversampling appears needed. This estimate uses
global active-class frequency and is **not** a direct measurement of per-patch training support
after grid subsampling; it is a risk-triage heuristic only. Implementation is deferred to the next
milestone if indicated.

---

## 9. Important Brittleness and Risk Areas

### 9.1 Open3D-ML internal API assumptions

The single largest risk. Multiple interface properties are completely unknown: `SemSegLoss`
weight format, dataset class base API, pipeline training API, concatenation location. Mitigation:
**read local source files before writing any of Phases 4–5 code**. Produce machine-readable JSON
artifacts from that reading. Never guess these interfaces.

### 9.2 Statistics resume corruption

If the statistics script crashes mid-run and resumes using an in-memory accumulator pattern, it
silently computes final statistics from only the re-processed remainder — wrong statistics that
propagate to every training run with no downstream detection. Mitigation: per-sequence
intensity samples must be saved to `logs/stats_per_seq/{seq_id}.npy`; per-sequence class counts
must be written to the JSONL; final aggregation must re-read ALL completed per-sequence
artifacts, not the current run's in-memory state.

### 9.3 Compound silent contamination in multi-sequence scans

If `set_sensor(1)` does not persist AND per-frame exception handlers silently skip misaligned
frames, a long-running statistics scan produces plausible-looking but wrong output. The
pre-flight check exists to catch this before any scan runs.

### 9.4 Loss weight format

`SemSegLoss` may expect weights in a format different from what any planning document or AI
assumes. The wrong format produces silent semantic corruption (wrong class gets wrong weight),
not a crash. Weight format must be discovered from local source before any weight computation.

### 9.5 Explosive loss divergence during sanity check

A sanity run where loss explodes from 1.2 to 850 satisfies "loss is finite and changed" but is not
a mechanical success. The sanity success criterion requires final loss ≤ 10× initial loss. If this
threshold is exceeded, the sqrt-inverse-frequency dampened weight variant must be tried before
the milestone is considered passed.

### 9.6 Post-grid point count and upsampling

`grid_size: 0.04` reduces the 62K forward-only cloud. If the post-grid count falls below
`num_points: 16384`, the sampler repeats points. This must be measured and documented
before declaring the configuration validated.

### 9.7 Semseg frame count versus LiDAR frame count

Within a sequence, semseg may have fewer annotated frames than LiDAR. If the statistics loop
uses LiDAR frame count as its bound, later frames are silently skipped with no error. The
statistics script must detect mismatches and track skip rate. Skip rate above 5% indicates a
systematic problem; above 5%, stop and investigate rather than continuing.

### 9.8 GPU memory

If `num_points: 16384` exceeds GPU memory during the sanity check, temporarily reduce it,
document as sanity-check-only, and restore `16384` before full training. Do not permanently
change this value due to a sanity-check OOM.

---

## 10. Decisions Milestone B Must Make Explicitly

### 10.1 Class-weight transform

Two candidates must be computed from measured class counts, both using the format
discovered from local `SemSegLoss` source:

- Raw inverse-frequency normalized: `w_c = (active_total / (K × count_c)) / w_min`
- Sqrt-inverse-frequency normalized: `w_c = sqrt(active_total / (K × count_c)) / w_min`

Both must be saved in `logs/milestone_b_training_statistics.json`. If the raw lane weight
exceeds 50, the sqrt-inverse-frequency variant is the recommended sanity-check weight. The
stop conditions must record which variant was used. Neither variant is declared "final" — the
final weighting policy is deferred to the next milestone.

### 10.2 Split assignment

Must be deterministic and reproducible. Val=9, test=9, train=remainder from the semseg-enabled
pool. Val and test must each contain at least one lane-bearing sequence, confirmed by explicit
Python assertion. The split report must record the seed, manifest file hash, and the lane
allocation decision.

### 10.3 Whether lane-aware oversampling is needed

Milestone B records the estimate `E = num_points × p_lane / 64` and notes whether the result
is below 2.0 (indicating likely starvation). If indicated, oversampling must be implemented
before or explicitly rejected with evidence in the next milestone.

### 10.4 Concatenation location for xyz+feat

Must be discovered from the local Open3D-ML pipeline source and recorded in
`logs/milestone_b_dataset_interface.json` under `concat_location`. The dataset class and
sanity script must both be consistent with the discovered answer.

---

## 11. Readiness Gates Before Moving Beyond Milestone B

All 20 gates must be satisfied before work moves beyond Milestone B:

1. `set_sensor(1)` persistence confirmed across ≥5 consecutive frames
2. Semseg alignment confirmed across those frames
3. Semseg-enabled sequence pool discovered and recorded in manifest
4. Lane presence checked per sequence with all-frames coverage
5. Pool size known and ≥70 (minimum for planned split)
6. Split frozen; no overlaps between splits
7. Val and test each contain ≥1 lane-bearing sequence (Python assertion passed)
8. Split provenance recorded (seed, manifest hash, lane allocation)
9. `logs/milestone_b_loss_interface.json` populated (SemSegLoss format fully discovered)
10. `logs/milestone_b_dataset_interface.json` populated including `concat_location`
11. `logs/milestone_b_pipeline_interface.json` populated
12. Intensity statistics (P0.5, P99.5, clip mean, clip std) measured on training split
13. Class counts measured on training split; lane count is positive
14. Frame skip rate during statistics scan is below 5%
15. Both raw and sqrt-inverse-frequency weight candidates computed and saved in statistics JSON
16. Dataset class implemented with no hardcoded statistics; imports remap from `src/thesis_pipeline/adapters/`
17. YAML config updated with measured values consistent with statistics JSON (programmatically verified)
18. Backward compatibility: `build_one_sample()` lane fraction ≈ `0.0026` (within 10%)
19. Post-grid point count measured and documented
20. Sanity training script has no remaining placeholder markers
21. Sanity run executed: loss is finite; loss changed; final loss ≤ 10× initial loss
22. Stop conditions written explicitly
23. No full training has occurred
24. No performance claim has been made

### Partial Degradation Policy

| Failure type                                    | Action                                                                                                              |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Zero lane-class points in training split        | **Blocking** — stop immediately                                                                                     |
| `SemSegLoss` weight format cannot be determined | **Blocking** — stop                                                                                                 |
| `set_sensor(1)` does not persist                | **Blocking** — stop; must find working re-application pattern before any scan                                       |
| Pipeline crashes on every sanity attempt        | **Blocking** — stop and diagnose                                                                                    |
| <5% of frames fail to load during statistics    | **Flagged continuation** — document count and pattern; proceed                                                      |
| Post-grid count < `num_points`                  | **Flagged continuation** — document duplication rate; proceed                                                       |
| GPU OOM at `num_points: 16384` during sanity    | **Flagged continuation** — reduce to safe value, document as sanity-only concession                                 |
| Skip rate >5% during statistics                 | **Blocking** — investigate before proceeding                                                                        |
| Specific sequences consistently fail to load    | **Explicit exclusion** — document and exclude; pool must remain ≥70                                                 |
| Sanity loss diverges (>10× ratio)               | Not a pass — retry with sqrt-inverse-frequency weights; if still diverging, record `INSTABILITY` in stop conditions |

---

## 12. Code Architecture Rules

The two-layer architecture from Days 1–4 must be respected throughout Milestone B:

**`src/thesis_pipeline/...`** — single source of truth for all reusable logic:

- `src/thesis_pipeline/adapters/pandaset_ff_lane3.py` — remap, single-frame loading, intensity preprocessing helpers
- `src/thesis_pipeline/core/pandaset_compat.py` — `get_frame_count()` and compatibility helpers
- `src/thesis_pipeline/datasets/` — new location for the core dataset class logic (not the guide-facing wrapper)

**`tools/...`** — thin runnable wrappers only; import from `src/thesis_pipeline/...`

**`datasets/pandaset_ff_lane3.py`** — guide-facing thin wrapper; imports core logic from `src/thesis_pipeline/...`; preserves `build_one_sample()` for backward compatibility

**`configs/...`** — YAML config files; mirrors statistics JSON; must not be the source of truth for any measured value

**`logs/...`** — all generated reports, JSON artifacts, and per-sequence data files

Rules:

- No script may reimplement `remap_raw_pandaset_ids` — import from `src/thesis_pipeline/adapters/`
- No script may call `len(seq.lidar.data)` directly — use `get_frame_count(seq)` from `pandaset_compat`
- No script may hardcode statistics values — read from `logs/milestone_b_training_statistics.json`
- All scripts run from the project root
- All major scripts must emit `script_status PASS` or `script_status FAIL` as their final line and exit with code 0 or 1 accordingly

---

## 13. What Remains Intentionally Unfrozen After Milestone B

- whether the baseline training converges
- whether the lane class is learnable under the current configuration
- lane-aware patch oversampling implementation (only the need-assessment is done)
- final class-weight transform (both candidates recorded; final policy selected in next milestone)
- full training hyperparameters (epoch budget, batch size, learning rate schedule)
- any performance metric, ablation result, or thesis-level conclusion
- whether `grid_size: 0.04` sufficiently preserves lane points after grid subsampling (partial evidence from post-grid count; lane-point survival rate not directly measured)

---

## 14. How This Document Relates to Other Project Documents

- **Thesis master document** (`thesis_project_master_document_pandaset_randlanet.pdf`): authoritative on task semantics, label policy, baseline scope, non-goals. If this document conflicts with it on meaning, the master wins.
- **Revised implementation pipeline** (`revised_implementation_pipeline_document.pdf`): authoritative on long-range project structure and evaluation intent.
- **Milestone A implementation record** (`MILESTONE_A_DAY1_TO_DAY4_IMPLEMENTATION.md`): authoritative on what was actually built in Days 1–4. Supersedes any planning text for verified-state claims.
- **This document**: companion execution context for Milestone B. More cautious than the master on local implementation details by design.

---

## 15. How This Document Must Be Used by Coding Agents

1. **Preserve all Category A project truths.** They are invariants.

2. **Treat Category B truths as single-sample facts.** Do not extend them to multi-sequence contexts without the Milestone B verification steps confirming they hold.

3. **Treat Category C items as completely unknown.** Do not write code that depends on guessed interface formats, guessed pool sizes, or guessed concatenation behavior.

4. **Read local Open3D-ML source before writing dataset class or pipeline scripts.** Produce machine-readable JSON artifacts before writing the code that depends on them.

5. **Measure before freezing.** No statistics value, split ID, or loss weight should be hardcoded from prior planning text.

6. **Do not escalate scope.** No performance evaluation, no qualitative visualizations, no ablations, no full training.

7. **Treat generated code as scaffold.** Every script that interacts with Open3D-ML internals must be verified by actual local inspection, not assumed from documentation or model output.

8. **Document all decisions to files.** Every measured value, every interface discovery, every split decision must be recorded in the canonical artifact for that category.

9. **Handle failure modes explicitly.** Do not swallow exceptions silently in multi-sequence loops. Track per-frame failure counts. Emit final `script_status PASS/FAIL` with non-zero exit on blocking failure.

10. **Run all scripts from the project root.** Scripts use relative paths and will produce incorrect results from another directory. Use `python tools/script_name.py` not `cd tools && python script_name.py`.

11. **Use `set -o pipefail` in all piped shell commands.** `python script.py 2>&1 | tee logfile.txt` masks Python failures without it.
