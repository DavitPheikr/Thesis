# Milestone H Dataset Analysis

Dataset-level analyses for H live here.

Completed analyses:

```text
z_crop_audit/
marking_aware_sampling_audit/
```

## z_crop_audit

Purpose:

- evaluate candidate z-crop ranges
- count kept/removed points by class
- verify marking retention
- quantify reduction of irrelevant `other` points
- estimate whether a crop would touch G1's actual errors

Decision:

```text
do not crop
```

Source:

```text
z_crop_audit/z_crop_recommendation.md
```

## marking_aware_sampling_audit

Purpose:

- test whether changing patch centers would meaningfully alter sampled patch
  composition
- quantify marking exposure under uniform, marking-centered, hard-negative, and
  hybrid candidate policies
- check whether a sampler change is justified before changing training

Decision:

```text
do not change sampler for H0
```

Source:

```text
marking_aware_sampling_audit/sampling_recommendation.md
```

Current implication:

```text
Milestone H should target RGB brightness robustness rather than z-cropping or
marking-aware sampling.
```
