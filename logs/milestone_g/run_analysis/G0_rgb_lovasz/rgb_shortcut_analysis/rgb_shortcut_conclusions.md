# G0 RGB Shortcut Fingerprint

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
| road TP | 27,471,463 | 0.393 / 0.392 | 0.392 | -0.018 | 0.034 | 27.9 |
| road->marking (FP) | 489,746 | 0.424 / 0.483 | 0.430 | +0.037 | 0.044 | 32.3 |
| marking TP | 865,243 | 0.388 / 0.396 | 0.392 | +0.024 | 0.061 | 39.3 |
| marking->road (FN) | 275,549 | 0.308 / 0.304 | 0.307 | -0.012 | 0.055 | 25.9 |
| other->marking (FP) | 13,764 | 0.281 / 0.297 | 0.282 | +0.016 | 0.016 | 30.1 |

## Answers

**Q1. Are road->marking FPs brighter than road TPs?** YES (mean 0.424 vs 0.393, delta +0.031; median 0.483 vs 0.392, delta +0.091).

**Q2. Are FPs similar to true markings in brightness?** FP is in fact at least as bright: YES (mean 0.424 vs marking 0.388, delta +0.036; median 0.483 vs 0.396, delta +0.087).

**Q3. Marking-colored, or bright gray/white road?** FP is less color-saturated than true markings: YES (saturation proxy FP 0.044 vs marking 0.061; warmth FP +0.037 vs marking +0.024). High brightness with lower saturation is consistent with bright near-neutral road (glare/concrete) rather than marking-specific color.

**Q4. Are missed markings darker than detected markings?** YES (missed 0.308 vs detected 0.388, delta -0.080).

**Q5. Does RGB brightness disagree with LiDAR intensity in FPs?** YES. FPs are brighter than road TPs (an elevated, marking-associated RGB cue), yet their LiDAR intensity stays road-like: on a road(0)->marking(1) intensity scale the FPs sit at only 0.39 (FP intensity 32.3 vs road 27.9, marking 39.3).

**Q6. Concentrated in rgb_valid points?** road->marking FPs: 412,083 rgb_valid vs 77,663 rgb_invalid (84.1% of FPs are rgb_valid); pred/true 1.240 valid vs 1.004 invalid.

## Overall reading

If FPs are (a) brighter than road TPs, (b) as bright as or brighter than true
markings, (c) less warm/saturated than true markings, and (d) road-like in LiDAR
intensity, that is consistent with predictions being *associated with a
luminance-based RGB shortcut*: bright near-neutral road surfaces (glare, light
concrete, overexposed pixels) carry the same high-RGB signature as white paint.
