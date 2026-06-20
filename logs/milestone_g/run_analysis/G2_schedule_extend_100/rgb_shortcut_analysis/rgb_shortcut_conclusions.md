# G2_schedule_extend_100 RGB Shortcut Fingerprint

**Framing.** These results are evidence that predictions are *associated*
with a luminance-based RGB shortcut. They do not prove the model internally
computes or uses brightness; they characterize the RGB/intensity profile of
the points the model labels marking vs road.

Linear metrics (brightness, luminance, warmth, per-channel means) are exact
group means. `saturation_proxy` is an aggregate approximation from per-channel
medians, not a per-point statistic.

## Fingerprint table (key columns)

brightness shown as mean / median(approx); markings are mean-skewed by a dark
tail, so the median is the more robust 'typical' value.

| group | n | brightness mean/med | luminance | warmth(R-B) | sat_proxy~ | intensity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| road TP | 27,521,255 | 0.394 / 0.392 | 0.392 | -0.018 | 0.034 | 27.9 |
| road->marking (FP) | 532,367 | 0.407 / 0.478 | 0.413 | +0.031 | 0.038 | 31.9 |
| marking TP | 916,369 | 0.379 / 0.390 | 0.383 | +0.023 | 0.064 | 38.7 |
| marking->road (FN) | 229,626 | 0.328 / 0.313 | 0.326 | -0.013 | 0.052 | 25.2 |
| other->marking (FP) | 13,531 | 0.291 / 0.293 | 0.290 | -0.004 | 0.029 | 24.3 |

## Answers

**Q1. Are road->marking FPs brighter than road TPs?** YES (mean 0.407 vs 0.394, delta +0.014; median 0.478 vs 0.392, delta +0.085).

**Q2. Are FPs similar to true markings in brightness?** FP is in fact at least as bright: YES (mean 0.407 vs marking 0.379, delta +0.028; median 0.478 vs 0.390, delta +0.087).

**Q3. Marking-colored, or bright gray/white road?** FP is less color-saturated than true markings: YES (saturation proxy FP 0.038 vs marking 0.064; warmth FP +0.031 vs marking +0.023). High brightness with lower saturation is consistent with bright near-neutral road (glare/concrete) rather than marking-specific color.

**Q4. Are missed markings darker than detected markings?** YES (missed 0.328 vs detected 0.379, delta -0.052).

**Q5. Does RGB brightness disagree with LiDAR intensity in FPs?** YES. FPs are brighter than road TPs (an elevated, marking-associated RGB cue), yet their LiDAR intensity stays road-like: on a road(0)->marking(1) intensity scale the FPs sit at only 0.37 (FP intensity 31.9 vs road 27.9, marking 38.7).

**Q6. Concentrated in rgb_valid points?** road->marking FPs: 427,293 rgb_valid vs 105,074 rgb_invalid (80.3% of FPs are rgb_valid); pred/true 1.289 valid vs 1.205 invalid.

## Overall reading

If FPs are (a) brighter than road TPs, (b) as bright as or brighter than true
markings, (c) less warm/saturated than true markings, and (d) road-like in LiDAR
intensity, that is consistent with predictions being *associated with a
luminance-based RGB shortcut*: bright near-neutral road surfaces (glare, light
concrete, overexposed pixels) carry the same high-RGB signature as white paint.
