# First 4 Days Project Context

## Purpose of this stage

The purpose of the first 4 days is to establish that the project is **technically buildable and semantically correct at the earliest implementation level**, without yet claiming that real training, meaningful performance, or final experimental behavior has been achieved. This stage exists to confirm that the project can be grounded on the local machine in a way that matches the revised global master project document’s meaning and constraints.

At this stage, the project is doing four things only:

1. fixing the **semantic meaning** of the project in code,
2. checking that PandaSet can actually be accessed and interpreted correctly on the local system,
3. proving that a custom `point / feat / label` data path can be built correctly for real frames,
4. proving that the model and configuration can be constructed consistently enough to support later training work.

This stage is **not** trying to produce good metrics, validate the thesis scientifically, or optimize the model. It is trying to remove ambiguity and prevent false assumptions from hardening into code.

The first 4 days are therefore a **trust-building stage**. The outcome is not “the model works well.” The outcome is: “the baseline has a technically credible starting point, and the project meaning has not been corrupted during implementation.”

For Copilot, coding agents, and implementation tools, this means:

- treat this stage as **proof-of-correctness for project meaning and data handling**
- do **not** treat it as the start of normal model-development speed
- prefer explicit verification over cleverness, inference, or convenience
- keep **semantic project truth** separate from **local implementation truth** at all times

---

## What is already fixed

The following are already fixed at the level of **project meaning** and must be treated as stable during the first 4 days.

### 1. The task is point-wise semantic segmentation

The project is a **point-wise semantic segmentation** task on PandaSet LiDAR. That means each point receives a semantic class label. It is not a lane-graph task, not a drivable-area estimation task, and not a lane-topology reconstruction task.

### 2. The active semantic classes are road, lane, and other

The project uses three active semantic classes:

- `road`
- `lane`
- `other`

This class structure is project truth, not a provisional implementation guess. It defines the baseline semantic problem.

### 3. The positive class is intentionally narrow

The positive class `lane` means **Lane Line Marking only**, subject to local raw-ID confirmation before coding the remap. The project is deliberately **not** defining lane as “all painted markings” or “all road paint.” That narrow meaning must be preserved in code, comments, and tool use.

### 4. The project is LiDAR-only

The project uses **LiDAR only**. No camera input, no fusion logic, and no map-based support belong in the first 4 days.

### 5. The intended baseline sensor choice is forward-facing LiDAR only

The intended baseline uses **forward-facing LiDAR only**, not a merged multi-sensor cloud. This is fixed at the project-meaning level. What remains unverified is only whether the local implementation of front-only filtering behaves exactly as expected.

### 6. The intended baseline coordinate frame is ego-local

The intended baseline uses **ego-local coordinates**, meaning coordinates expressed relative to the vehicle or sensor frame rather than in the global world frame. This is fixed as project intent. What is not yet fixed is the exact local implementation path that will perform that transformation correctly on the machine being used.

### 7. The intended minimal feature concept is xyz plus one processed intensity channel

The intended baseline feature concept is:

- `xyz` geometry
- one processed intensity-like return-strength channel

This is fixed at the level of project intent. What is not yet fixed is the exact local preprocessing implementation and the exact measured statistics needed for that feature.

### 8. The model family and framework intent are fixed

The intended baseline model family is **RandLA-Net**, and the intended framework is **Open3D-ML**. This means that the project is supposed to be implemented around that architecture and framework family unless a later verified local failure makes that impossible. It does **not** mean that the local environment has already proven this combination works.

### 9. The baseline philosophy is fixed

The project is supposed to begin with:
- a small, technically serious baseline,
- from-scratch training later,
- a custom dataset adapter,
- careful local verification before any real training,
- no premature claims about results.

### 10. Days 1–4 stop before real training

This is fixed and must not be violated.

During the first 4 days, the work stops at:
- environment setup,
- PandaSet verification,
- adapter creation and testing,
- config scaffold creation,
- model construction,
- one model-build or sample-contract check,
- explicit stop conditions before real training.

A tiny real training sanity run belongs **after** Day 4, in the build-up to Milestone A, not inside the first 4 days.

For agents, these fixed items are **non-negotiable semantic constraints**. They are not optional defaults and must not be silently broadened, merged, reinterpreted, or simplified.

---

## What is still provisional

The following are **not yet fixed facts**. They must remain provisional until tested or measured locally.

### 1. Raw semantic ID mapping  
**[PROVISIONAL UNTIL LOCALLY VERIFIED]**

The project assumes that the local PandaSet semantic ID space matches the expected meaning used in the global master document. That assumption must be confirmed from the local dataset copy before the remap is trusted.

### 2. Intensity preprocessing statistics  
**[SET AFTER LOCAL VERIFICATION: compute clipping percentiles and global mean/std on the training split after front-only filtering]**

The concept of using one processed intensity channel is fixed. The actual numbers are not.

### 3. Measured class counts  
**[SET AFTER LOCAL VERIFICATION: count active class frequencies on the training split after remap]**

The classes are fixed, but their measured frequency on the local dataset is not.

### 4. Loss weights  
**[SET AFTER LOCAL VERIFICATION: define and document the transform from measured class counts to loss weights]**

Class counts and loss weights are not the same thing. Any document, tool, or Copilot suggestion that collapses them into one concept must be corrected.

### 5. Sparse-class-safe configuration values  
**[PROVISIONAL UNTIL LOCALLY VERIFIED]**

The following are intended scaffold values, not yet frozen local truths:

- `num_layers = 3`
- `sub_sampling_ratio = [4, 4, 4]`
- `grid_size = 0.04`
- `num_points = 16384`

These are planning values until the local machine and the local PandaSet installation confirm they are operationally safe.

### 6. Exact sequence split membership  
**[SET AFTER LOCAL VERIFICATION: freeze exact 58/9/9 IDs after auditing the semseg-enabled sequence pool for lane presence and split diversity]**

The split concept is part of the project plan. The exact local sequence lists are not yet frozen.

### 7. Exact environment choices  
**[PROVISIONAL UNTIL LOCALLY VERIFIED]**

These include:
- OS assumptions
- Python version choice on the actual system
- Torch/Open3D compatibility
- CUDA availability
- GPU use
- package install success
- filesystem layout
- local path conventions

### 8. Exact API behavior  
**[PROVISIONAL UNTIL LOCALLY VERIFIED]**

Anything that depends on the installed versions of:
- PandaSet devkit
- Open3D / Open3D-ML
- Torch
- local package combinations

must be treated as locally unverified until tested.

For agents, everything in this section is **implementation truth pending verification**, not project truth. It may influence file structure, code shape, imports, commands, or local tooling choices, but it must not change semantic meaning.

---

## What must be verified locally before coding assumptions are frozen

The first 4 days must verify the following locally before any implementation assumption is allowed to become “truth in code.”

### 1. Local dataset reality

It must be verified that:
- PandaSet exists locally where the code expects it,
- the local folder structure is usable by the installed devkit,
- semantic segmentation files are present,
- the local raw semantic IDs match the intended label policy.

Without this, label-remap code is not trustworthy.

### 2. Front-only filtering reality

It must be verified that the local devkit and local LiDAR dataframe behavior actually produce the expected front-only cloud and that this filtering does not silently break semantic alignment.

### 3. Semseg alignment reality

It must be verified on real filtered frames that semantic labels still align to the surviving point indices exactly. This is one of the most important truths of the first 4 days. If this is wrong, nearly everything downstream is invalid.

### 4. Adapter contract reality

It must be verified that a real frame can be converted into:
- `point`
- `feat`
- `label`

with the expected shapes and intended meanings.

The adapter must be tested on actual data, not just written.

### 5. Feature-path reality

It must be verified that:
- ego-local coordinates can actually be produced on the local system,
- intensity can actually be read,
- the processed intensity path is structurally valid,
- the data returned by the adapter matches the intended baseline meaning.

### 6. Model-construction reality

It must be verified that RandLA-Net can be constructed locally against the current sample contract and config scaffold without immediate shape or interface failure.

### 7. Config-scaffold reality

It must be verified that the initial config scaffold is at least structurally coherent enough for model construction and later training entry, even though some values remain provisional.

### 8. Stop-condition reality

Before moving beyond Day 4, it must be clear whether the project is actually ready to proceed, or whether one of the local truths failed and must be corrected first.

For agents, these are not optional checks. They are the **minimum verification boundary** before a local implementation detail becomes trusted enough to reuse elsewhere.

---

## What parent-document details must be treated as unverified until tested on the local system

The parent documents are project truth for meaning and planning. They are **not proof** that local execution details already work.

The following must be treated as **unverified until locally tested**, even if they appear in earlier AI-generated documents.

### 1. Environment stack details

Do not assume these are already proven locally:
- Linux assumption
- Python version compatibility
- CUDA availability
- GPU driver compatibility
- Torch install success
- Open3D install success
- Open3D-ML Torch path import success
- exact package versions working together

These are planning targets, not local facts.

### 2. Package and API compatibility

Do not assume:
- imports will succeed unchanged,
- module paths are identical across versions,
- helper functions exist exactly as written,
- the local PandaSet devkit API matches generated examples,
- Open3D config or dataset internals behave exactly as expected.

### 3. Filesystem and path assumptions

Do not assume:
- the local PandaSet path matches generated examples,
- file names are exactly where earlier documents expected,
- relative paths in scaffold code are already correct,
- local write permissions already work for logs, cache, or outputs.

### 4. Config values that depend on data measurement

Do not assume these are final:
- class counts
- class weights
- intensity clipping values
- intensity mean and std
- safe `num_points`
- exact split IDs
- sparse-class-safe configuration values

### 5. Code skeleton correctness

Do not assume generated code will run unchanged just because it is plausible or well structured. Generated code in the parent chat is a **scaffold**, not a guarantee.

### 6. GPU usage

Do not assume the project is ready for GPU just because the intended device is GPU. That remains a system-dependent implementation truth until the local environment proves it.

For agents, this section is the main guard against **false certainty inherited from planning**. If a detail from a parent document touches the local system, it stays untrusted until tested.

---

## What the coding work in Days 1–4 is trying to prove

The coding work in Days 1–4 is trying to prove **technical viability at the earliest trustworthy level**.

More precisely, it is trying to prove:

### 1. The local machine can host the project at all

This is the environment question. It is not glamorous, but it is fundamental.

### 2. The local PandaSet installation can be read correctly

This is the data-access question.

### 3. The project’s semantic meaning survives contact with real local data

This is the label-policy question. It is the bridge between project truth and implementation truth.

### 4. Front-only and semseg-aligned samples can be constructed correctly

This is the most important data-integrity question of the first 4 days.

### 5. The intended baseline contract can be expressed in code

This is the adapter question:
- can a real sample become `point / feat / label` without corrupting meaning?

### 6. The model/config stack can be constructed against that contract

This is the earliest structural model question.

Days 1–4 are therefore trying to prove:

> the baseline is **buildable**, **semantically faithful**, and **structurally coherent enough** to justify later Milestone A work.

They are **not** trying to prove that the model already learns well.

For Copilot and coding agents, the correct mindset is:
- prove data meaning first
- prove contract integrity second
- prove model construction third
- stop before real training

---

## Day-by-day meaning of the work

### Day 1

Day 1 is about establishing **local system reality** and **local dataset reality**.

Meaning of the work:
- determine whether the intended software stack can even exist locally,
- determine whether PandaSet is actually present and readable,
- determine whether semantic segmentation files and local dataset structure are real and accessible,
- determine whether the raw semantic ID assumptions are true on the actual local dataset.

What Day 1 is proving:
- the project is not based on a false environment assumption,
- the label policy is not built on an imagined local ID space.

What Day 1 is not doing:
- writing training code,
- freezing config values,
- trusting generated remap code before local verification.

### Day 2

Day 2 is about **data integrity under the project’s actual sensor choice**.

Meaning of the work:
- test front-only filtering,
- verify that semantic labels still align after filtering,
- inspect real point counts and local data behavior,
- begin measuring whether the sparse-class planning assumptions are plausible.

What Day 2 is proving:
- the intended baseline sensor policy is actually implementable,
- filtered point clouds still carry valid labels,
- the project’s semantic problem survives the forward-only restriction.

What Day 2 is not doing:
- claiming sparse-class-safe settings are final,
- claiming the lane class is already well supported,
- moving into training.

### Day 3

Day 3 is about **turning project meaning into the actual adapter contract**.

Meaning of the work:
- write the custom dataset adapter,
- test it on real frames,
- verify `point / feat / label` structure,
- verify ego-local conversion path as an implementation reality rather than only a conceptual plan,
- verify the processed intensity path structurally.

What Day 3 is proving:
- the thesis semantics can be represented cleanly in code,
- the adapter is not only written but grounded in real local frames.

What Day 3 is not doing:
- optimizing preprocessing,
- finalizing class weights,
- proving that the model will learn.

### Day 4

Day 4 is about **structural model readiness without crossing into real training**.

Meaning of the work:
- create the minimal project structure,
- create the minimal adapter module file,
- create the minimal config scaffold,
- create the model-build or contract-check script,
- verify that the current adapter and config are structurally compatible,
- define the exact stop conditions before real training.

What Day 4 is proving:
- the project has a credible local implementation starting point,
- the baseline can be constructed consistently enough to justify Milestone A work next.

What Day 4 is not doing:
- a tiny sanity training run,
- decoding predictions from a real training run,
- saving training-based qualitative outputs,
- making any performance claim.

---

## Non-goals for Days 1–4

The following are explicitly outside scope for the first 4 days:

- real baseline training
- final hyperparameter selection
- ablation execution
- final train/val/test split freezing
- final class-weight freezing
- focal loss decisions
- performance interpretation
- metric claims
- thesis-result writing
- any claim that the baseline “works well”
- any claim that intensity is already validated as useful
- any claim that the sparse-class-safe scaffold is already final
- any claim that GPU execution is already locally proven unless it has actually been tested on the local machine

Days 1–4 are about **local grounding**, not results.

For agents, anything that looks like:
- a training loop,
- a performance claim,
- an ablation branch,
- a results table,
- an optimization pass,

belongs outside this document’s scope.

---

## Readiness gates before moving beyond Day 4

The work must not move beyond Day 4 unless all of the following are true:

### Gate 1: local environment truth is good enough
The intended core stack imports successfully on the local system, or the local environment deviations are clearly understood and documented.

### Gate 2: local PandaSet truth is confirmed
The dataset is locally readable and semantic segmentation access is real.

### Gate 3: raw IDs are locally verified
The raw semantic ID assumptions used by the label policy are confirmed against the local dataset.

### Gate 4: front-only filtering is real
The forward-only baseline can be implemented on the local data.

### Gate 5: semseg alignment is preserved
Filtered points and semantic labels still align exactly.

### Gate 6: adapter output is real
The adapter returns correct `point / feat / label` on real frames.

### Gate 7: config scaffold exists
A minimal config scaffold exists and is consistent with the intended baseline meaning.

### Gate 8: model-build or sample-contract check succeeds
A real sample and the model can coexist structurally without immediate failure.

### Gate 9: no false certainty remains hidden
Any unresolved local uncertainty is explicitly marked as provisional rather than silently treated as solved.

If any gate fails, the project remains in the first-4-days corrective phase and must not advance.

For coding agents, these gates are **hard boundaries**, not suggestions.

---

## How this document must be used with the full global master project document

This document is a **companion** to the revised global master project document.

The full global master project document remains the authoritative source for:
- project meaning
- scope
- label semantics
- baseline intent
- what counts as the project
- what does not count as the project

This first-4-days companion must be used to translate that master truth into **early-phase constraints** and **implementation caution**.

If this document appears to say something different from the global master project document about what the project means, the global master project document wins.

If this document appears more cautious than the global master project document about local implementation details, that is intentional and correct.

---

## How this document must be used by Copilot, coding agents, and implementation tools

Copilot, coding agents, and implementation tools must use this document as an **early-phase constraints document**, not as a shell script and not as a license to assume local correctness.

They must treat the following as mandatory behavior:

### 1. Preserve project meaning
Do not change:
- class semantics
- task definition
- baseline intent
- forward-only baseline policy
- ego-local baseline intent
- custom adapter requirement

### 2. Preserve provisional status
If something is tagged:
- **[PROVISIONAL UNTIL LOCALLY VERIFIED]**
- **[SET AFTER LOCAL VERIFICATION: ...]**

then do not silently hard-code it as final truth without a local check.

### 3. Prefer verification over invention
If local reality disagrees with an earlier AI-generated assumption, local reality wins.

### 4. Do not escalate scope
Do not drift into:
- real training
- ablations
- performance analysis
- later-stage optimization

during the first 4 days.

### 5. Treat generated code as scaffold
Generated code is allowed as a starting structure, but must be treated as:
- inspectable,
- testable,
- revisable,
- not pre-validated.

### 6. Preserve the Day 4 stop boundary
No tool should behave as though the project has already moved into Milestone A if the Day 4 readiness gates are not satisfied.

### 7. Keep project truth and implementation truth separate
- **Project truth** controls meaning, semantics, and baseline intent.
- **Implementation truth** must be earned locally through verification.
- Never let a local convenience rewrite a fixed project truth.

---

## Content rules

- Use the revised global master project document as the source of truth for project meaning.
- Keep the explanation limited to the first 4 days.
- Do not include real training beyond model-build readiness unless the global master document clearly places it inside the first 4 days.
- Explain technical terms briefly the first time they appear.
- Where something remains unfixed, preserve the exact provisional-tagging logic from the global document.
- Explicitly call out system-dependent assumptions that must not be trusted until tested locally.
- Do not output shell-command checklists.
- Do not output a coding workflow.
- Do not output full code unless a tiny illustrative fragment is absolutely necessary.
- This is a context-and-constraints document, not the execution guide.
