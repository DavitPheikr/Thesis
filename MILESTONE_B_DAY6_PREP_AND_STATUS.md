# Milestone B Day 6 Preparation and Status

## 1. Purpose of this document

This document records the actual Day 6 status of Milestone B as verified from the repository, the Day 6-specific issues that were found, the minimal fixes applied, the handling of stale artifacts from the prior interrupted attempt, and the exact fresh execution procedure for the next Day 6 sanity run.

This is a preparation document only. The sanity run itself has not been launched in this preparation turn.

## 2. Current confirmed state entering Day 6

The following are confirmed from on-disk evidence:

- Days 1-5 are complete and passed.
- `logs/milestone_b_dataset_class_report.txt` ends with `script_status PASS`.
- `tools/sanity_train_check.py` exists and imports cleanly.
- `logs/milestone_b_sanity_train_report.txt` existed as a zero-byte stale artifact before cleanup.
- `logs/milestone_b_sanity_config_snapshot.yml` did not exist before prep.
- `logs/milestone_b_stop_conditions.txt` did not exist before prep.
- Milestone B is not complete because Day 6 runtime completion and stop conditions are still missing.

Relevant evidence:

- `logs/milestone_b_preflight.txt`
- `logs/milestone_b_sequence_audit.txt`
- `logs/milestone_b_sequence_manifest.json`
- `logs/milestone_b_split_report.txt`
- `logs/milestone_b_training_statistics.json`
- `logs/milestone_b_dataset_class_report.txt`
- `logs/milestone_b_sanity_train_report.txt` (stale zero-byte artifact before cleanup)

## 3. What Day 6 is supposed to do according to the final Milestone B docs

Day 6 is the final Milestone B step. It must:

- assemble the full Open3D-ML pipeline
- run a small sanity training check
- confirm that training executes for at least 5 iterations
- confirm losses are finite
- confirm loss changes between the first and last captured iteration
- confirm final loss does not exceed 10x the initial loss, or explicitly record instability if that cannot be achieved
- write stop conditions before any later full-training milestone work

The Day 6 deliverables required by the final implementation guide are:

- `tools/sanity_train_check.py`
- `logs/milestone_b_sanity_train_report.txt`
- `logs/milestone_b_sanity_config_snapshot.yml`
- `logs/milestone_b_stop_conditions.txt`

The Day 6 pass criteria require:

- no placeholder markers left in the sanity script
- a sanity run with at least 5 iterations
- finite losses
- loss change across iterations
- acceptable final-to-initial loss ratio or explicit instability recording
- stop conditions written with the required fields

## 4. Which Day 6 artifacts currently exist and their status

### Present and valid after preparation

- `tools/sanity_train_check.py`
- `logs/milestone_b_day6_smoke_test.txt`

### Archived as stale prior-attempt artifacts

- `logs/milestone_b_sanity_train_report.stale_empty_before_fresh_run.txt`
  - this was previously `logs/milestone_b_sanity_train_report.txt`
  - it was zero bytes and therefore not a valid Day 6 run report
- `logs/RandLANet_PandaSetFFLane3_torch/log_train_2026-04-15_02-05-07.stale_empty.txt`
  - this was previously `logs/RandLANet_PandaSetFFLane3_torch/log_train_2026-04-15_02-05-07.txt`
  - it was zero bytes and therefore not a meaningful training log

### Still intentionally absent before the fresh run

- `logs/milestone_b_sanity_train_report.txt`
- `logs/milestone_b_sanity_config_snapshot.yml`
- `logs/milestone_b_stop_conditions.txt`

These are intentionally absent at the end of prep because the fresh detached sanity run has not yet been launched.

## 5. Current state of `tools/sanity_train_check.py`

The Day 6 sanity script is now in a prep-complete state for the next run.

Confirmed properties:

- no `[FILL_FROM_DISCOVERY]` markers remain
- it uses the correct primary weight access path:
  - `stats["class_weights"]["sanity_run_recommended_list"]`
- it uses the correct fallback weight access path:
  - `stats["class_weights"]["sqrt_inverse_frequency"]["as_list"]`
- it imports the local dataset wrapper:
  - `from datasets.pandaset_ff_lane3 import PandaSetFFLane3Dataset`
- it uses the local Open3D pipeline class:
  - `from open3d._ml3d.torch.pipelines import SemanticSegmentation`

The script also now:

- forces non-resume behavior for the sanity run
- writes its report incrementally so a mid-run crash does not leave an empty outcome artifact again
- distinguishes execution failure from completed-but-unstable training behavior

## 6. How the script matches or differs from the final guide

### Matches the guide

- uses the discovered Open3D pipeline class and runtime semantics
- performs a primary attempt and a fallback attempt
- checks for finite and non-static loss behavior
- is scoped to a small sanity run, not full training
- remains Day-6-only and does not alter earlier milestone phases

### Local runtime-aligned deviations from the guide

- The local Open3D runtime transforms `dataset.cfg.class_weights` internally.
  - Therefore the primary sanity-run weights must come from the repo’s approved runtime recommendation:
    - `stats["class_weights"]["sanity_run_recommended_list"]`
  - The guide’s more generic variant-indexing template is not correct for this repo.

- The script now treats a completed instability outcome as a valid Day 6 report outcome rather than a raw execution failure.
  - This better matches the guide’s instruction that instability should be explicitly recorded and carried forward if needed.

- The script now forces non-resume behavior explicitly.
  - This is a safety refinement based on the local Open3D pipeline, which otherwise auto-resumes from the latest checkpoint if one later appears under `logs/`.

## 7. Current local runtime semantics relevant to Day 6

These were verified from the local Open3D runtime and current repo artifacts:

### Weight semantics

- `SemSegLoss` consumes `dataset.cfg.class_weights`
- local Open3D converts that list through `DataProcessing.get_class_weights(...)`
- therefore the primary runtime-safe list is:
  - `stats["class_weights"]["sanity_run_recommended_list"]`
- the candidate direct CE vectors remain analysis artifacts, not the primary runtime input

Relevant artifacts:

- `logs/milestone_b_training_statistics.json`
- `logs/milestone_b_loss_interface.json`

### Pipeline training entry

- the local training method is `run_train()`
- iteration limiting is governed by:
  - `pipeline.max_epoch`
  - `dataset.steps_per_epoch_train`
  - `dataset.steps_per_epoch_valid`

Current config values:

- `max_epoch: 2`
- `steps_per_epoch_train: 5`
- `steps_per_epoch_valid: 2`

This means the configured sanity run should be able to exceed the 5-step minimum if the loader and model path behave normally.

### `real_training_allowed`

- this is a project sentinel in the YAML
- it is not checked by the Open3D runtime
- the sanity script removes it from the runtime pipeline config before pipeline construction

### Auto-resume risk

- local Open3D `SemanticSegmentation.run_train()` calls:
  - `is_resume = model.cfg.get("is_resume", True)`
  - `self.load_ckpt(model.cfg.ckpt_path, is_resume=is_resume)`
- therefore resume must be forced off explicitly for a clean sanity run

### Loss capture logic

- local `SemanticSegmentation.run_train()` resets `self.losses` each epoch
- `save_logs()` is called once per epoch after training/validation
- the `_SanityCapturer.save_logs()` override is a valid way to accumulate per-step losses across epochs

## 8. Any Day 6 issues found and how they were fixed

### Issue 1: stale zero-byte sanity report

Problem:

- `logs/milestone_b_sanity_train_report.txt` existed but was empty
- it could be mistaken for a real Day 6 artifact even though it contained no outcome

Fix:

- archived it to:
  - `logs/milestone_b_sanity_train_report.stale_empty_before_fresh_run.txt`

### Issue 2: stale zero-byte Open3D train log

Problem:

- `logs/RandLANet_PandaSetFFLane3_torch/log_train_2026-04-15_02-05-07.txt` existed but was empty
- it indicates an interrupted earlier attempt, not a valid run log

Fix:

- archived it to:
  - `logs/RandLANet_PandaSetFFLane3_torch/log_train_2026-04-15_02-05-07.stale_empty.txt`

### Issue 3: sanity script could silently auto-resume in future reruns

Problem:

- the local Open3D pipeline defaults `is_resume` to `True`
- this could silently change a future sanity run if checkpoint files later appear

Fix:

- the sanity script now forces:
  - `ckpt_path = None`
  - `is_resume = False`

### Issue 4: crash-prone empty-report behavior

Problem:

- the prior Day 6 report was empty
- the earlier script wrote the report only after a run attempt completed or failed cleanly

Fix:

- the script now writes its report incrementally from startup onward
- each major state transition is flushed to disk:
  - script start
  - attempt 1 start
  - attempt 1 result
  - attempt 2 start if needed
  - attempt 2 result
  - final outcome

### Issue 5: instability semantics

Problem:

- the earlier script treated “both variants ran but remained unstable” as a raw `FAIL`
- that is harsher than the guide’s intended handling

Fix:

- the script now distinguishes:
  - execution failure: `script_status FAIL`
  - completed instability outcome: `sanity_result INSTABILITY` with `script_status PASS`

## 9. How stale / failed prior Day 6 attempts were handled

The prior Day 6 artifacts were handled conservatively:

- no stale file was deleted outright
- both stale zero-byte files were archived in place
- no new fake Day 6 report was created
- no stop conditions were written prematurely
- no sanity snapshot was created prematurely

This preserves provenance while preventing confusion during the next fresh run.

## 10. Exact fresh sanity-run procedure to use next

### A. Confirm the smoke test already passed

The lightweight prep smoke test has already been run and written to:

- `logs/milestone_b_day6_smoke_test.txt`

Expected content:

- `smoke_ok True`

### B. Save the exact config snapshot before launch

Run:

```bash
cp configs/randlanet_pandaset_ff_lane3.yml logs/milestone_b_sanity_config_snapshot.yml
```

This is intentionally done before the detached run so the exact config is preserved even if the run dies mid-execution.

### C. Launch the sanity run detached from VS Code/session

Run:

```bash
setsid bash -lc 'cd /home/pheikara/University/Y3S2/Thesis/Pipeline && exec ./panda/bin/python -u tools/sanity_train_check.py > logs/milestone_b_sanity_train_report.txt 2>&1 < /dev/null' >/dev/null 2>&1 & echo $! > logs/milestone_b_sanity_train.pid
```

Why this exact form:

- `setsid` detaches from the VS Code process tree
- `python -u` keeps report writes unbuffered
- all output is redirected to the report file
- no terminal attachment remains
- the PID is captured for follow-up status checks

### D. Immediate post-launch checks

Run:

```bash
cat logs/milestone_b_sanity_train.pid
ps -p "$(cat logs/milestone_b_sanity_train.pid)" -o pid=,etime=,cmd=
```

### E. After the process exits

Run:

```bash
tail -n 40 logs/milestone_b_sanity_train_report.txt
```

Then verify one of these outcomes:

- healthy completion:
  - `sanity_train_check_ok True`
  - `script_status PASS`
- explicit instability:
  - `sanity_result INSTABILITY`
  - `script_status PASS`
- execution failure:
  - `script_status FAIL`

### F. After a real Day 6 outcome exists

Only after the detached run finishes:

- inspect the report
- write `logs/milestone_b_stop_conditions.txt`
- then determine whether Milestone B can be declared complete

## 11. Exact pass criteria for the sanity run

Healthy Day 6 pass:

- pipeline instantiates successfully
- at least 5 step-level losses are captured
- all loss values are finite
- loss changes between first and last captured iteration
- final loss is not more than 10x the initial loss
- report ends with:
  - `sanity_train_check_ok True`
  - `script_status PASS`

Completed instability outcome:

- both configured weight variants ran to a recorded outcome
- instability was explicitly recorded
- report ends with:
  - `sanity_result INSTABILITY`
  - `script_status PASS`

Execution failure:

- import/build/runtime failure occurs before a valid Day 6 outcome is produced
- report ends with:
  - `script_status FAIL`

## 12. What remains after the sanity run if it passes

If the detached sanity run produces a valid Day 6 outcome, the remaining work is:

- write `logs/milestone_b_stop_conditions.txt`
- confirm all required Day 6 fields are populated from actual measured artifacts
- confirm whether the outcome is:
  - healthy pass, or
  - explicit instability
- then determine Milestone B completion status

## 13. Current Day 6 readiness verdict

Day 6 is now **prepared for a fresh detached sanity run**.

Readiness basis:

- the sanity script matches the current repo’s weight semantics
- the script now forces non-resume behavior
- the script now writes a crash-robust incremental report
- stale empty prior Day 6 artifacts were archived
- the Day 6 smoke test passed
- the detached execution command is defined

Day 6 is **not yet executed** in this preparation turn.

## 14. Recommended next step

When approved, run the fresh Day 6 execution sequence in this order:

1. copy the config snapshot
2. launch the detached sanity run with `setsid`
3. verify the PID and detached process status
4. wait for completion
5. inspect the final report
6. write stop conditions from the actual outcome
