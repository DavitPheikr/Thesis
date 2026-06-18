# F0 RGB Shortcut Fingerprint

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
| road TP | 27,446,449 | 0.393 / 0.392 | 0.391 | -0.018 | 0.034 | 27.9 |
| road->marking (FP) | 710,399 | 0.429 / 0.491 | 0.435 | +0.028 | 0.037 | 30.8 |
| marking TP | 919,646 | 0.384 / 0.392 | 0.388 | +0.023 | 0.063 | 38.9 |
| marking->road (FN) | 228,774 | 0.307 / 0.307 | 0.305 | -0.014 | 0.055 | 24.0 |
| other->marking (FP) | 53,654 | 0.309 / 0.292 | 0.308 | +0.003 | 0.033 | 23.8 |

## Answers

**Q1. Are road->marking FPs brighter than road TPs?** YES (mean 0.429 vs 0.393, delta +0.036; median 0.491 vs 0.392, delta +0.099).

**Q2. Are FPs similar to true markings in brightness?** FP is in fact at least as bright: YES (mean 0.429 vs marking 0.384, delta +0.045; median 0.491 vs 0.392, delta +0.099).

**Q3. Marking-colored, or bright gray/white road?** FP is less color-saturated than true markings: YES (saturation proxy FP 0.037 vs marking 0.063; warmth FP +0.028 vs marking +0.023). High brightness with lower saturation is consistent with bright near-neutral road (glare/concrete) rather than marking-specific color.

**Q4. Are missed markings darker than detected markings?** YES (missed 0.307 vs detected 0.384, delta -0.077).

**Q5. Does RGB brightness disagree with LiDAR intensity in FPs?** YES. FPs are brighter than road TPs (an elevated, marking-associated RGB cue), yet their LiDAR intensity stays road-like: on a road(0)->marking(1) intensity scale the FPs sit at only 0.26 (FP intensity 30.8 vs road 27.9, marking 38.9).

**Q6. Concentrated in rgb_valid points?** road->marking FPs: 613,824 rgb_valid vs 96,575 rgb_invalid (86.4% of FPs are rgb_valid); pred/true 1.550 valid vs 1.148 invalid.

## Overall reading

If FPs are (a) brighter than road TPs, (b) as bright as or brighter than true
markings, (c) less warm/saturated than true markings, and (d) road-like in LiDAR
intensity, that is consistent with predictions being *associated with a
luminance-based RGB shortcut*: bright near-neutral road surfaces (glare, light
concrete, overexposed pixels) carry the same high-RGB signature as white paint.
