# Single-Seed Limitation (reusable methodology note)

A reusable caveat for the Milestone reports (D/E/F/G/G2/H). State this wherever a
run-to-run comparison is made.

## What it is

Every training run in this project is a **single-seed run (seed 42)**: the data
shuffling, weight initialisation, and augmentation randomness are fixed by one
seed, so each model is the result of **one** optimisation trajectory. Comparisons
between runs (e.g. G1 vs G2, or H0 vs G1) are therefore **controlled single-seed
comparisons** — every other factor is held identical and only one variable changes,
but each side is observed **once**.

## Why it matters

With one run per side you cannot separate a *real* improvement from **random
run-to-run variation**. Small differences in marking IoU — on the order of the
estimated **noise floor (~0.008 IoU)** — can be produced by seed luck alone and
**must not be over-claimed** as evidence that one method beats another.

## How conclusions should be drawn (the rule)

- **Do not** claim superiority from a marking-IoU gap that is within ~0.008 of the
  baseline. A sub-noise-floor IoU change is **inconclusive**, not a win.
- **Do** base stronger conclusions on a **consistent, multi-signal** picture — the
  same direction across several metrics that are not all driven by the same noise:
  - marking **IoU**, **precision**, **recall**, **F1**
  - **road→marking false-positive count** (the dominant error; robust, less
    seed-sensitive than a fractional IoU delta)
  - **pred/true marking ratio** (calibration)
  - **stratified** results (rgb_valid, distance, per-sequence, raw subtype) moving
    coherently
  - **qualitative** prediction overlays agreeing with the numbers
- A result is **defensible** when these move **together and beyond the noise floor**;
  it is **weak** when only a tiny IoU number moves.

## Strengthening the evidence (when feasible)

The clean fix for variance is **multiple seeds**: re-run the headline comparison
with 2–3 seeds and report **mean ± standard deviation**, claiming a win only when
the means differ by **more than the spread**. If compute/time forbids this, the
single-seed numbers stand **only** under the multi-signal rule above, and the
limitation must be stated explicitly in the write-up.

## Thesis-safe wording

> All runs are controlled single-seed experiments (seed 42): only one factor differs
> between compared runs, but each is observed once, so marking-IoU differences within
> the estimated ~0.008 noise floor are treated as inconclusive. Conclusions are drawn
> from consistent movement across IoU, precision, recall, F1, the road→marking
> false-positive count, the predicted/true marking ratio, and the stratified/qualitative
> analyses, rather than from a single metric near the noise floor.
