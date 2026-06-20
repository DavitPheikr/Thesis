# LiDAR + RGB + Lovász + Jitter RGB Shortcut Fingerprint

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
| road TP | 27,631,369 | 0.393 / 0.392 | 0.392 | -0.018 | 0.034 | 27.9 |
| road->marking (FP) | 459,380 | 0.409 / 0.484 | 0.416 | +0.040 | 0.046 | 32.3 |
| marking TP | 878,338 | 0.381 / 0.393 | 0.385 | +0.025 | 0.065 | 39.2 |
| marking->road (FN) | 268,250 | 0.328 / 0.312 | 0.327 | -0.015 | 0.053 | 25.4 |
| other->marking (FP) | 14,072 | 0.298 / 0.291 | 0.297 | -0.004 | 0.027 | 22.4 |

## Answers

**Q1. Are road->marking FPs brighter than road TPs?** YES (mean 0.409 vs 0.393, delta +0.016; median 0.484 vs 0.392, delta +0.092).

**Q2. Are FPs similar to true markings in brightness?** FP is in fact at least as bright: YES (mean 0.409 vs marking 0.381, delta +0.028; median 0.484 vs 0.393, delta +0.091).

**Q3. Marking-colored, or bright gray/white road?** FP is less color-saturated than true markings: YES (saturation proxy FP 0.046 vs marking 0.065; warmth FP +0.040 vs marking +0.025). High brightness with lower saturation is consistent with bright near-neutral road (glare/concrete) rather than marking-specific color.

**Q4. Are missed markings darker than detected markings?** YES (missed 0.328 vs detected 0.381, delta -0.052).

**Q5. Does RGB brightness disagree with LiDAR intensity in FPs?** YES. FPs are brighter than road TPs (an elevated, marking-associated RGB cue), yet their LiDAR intensity stays road-like: on a road(0)->marking(1) intensity scale the FPs sit at only 0.39 (FP intensity 32.3 vs road 27.9, marking 39.2).

**Q6. Concentrated in rgb_valid points?** road->marking FPs: 363,876 rgb_valid vs 95,504 rgb_invalid (79.2% of FPs are rgb_valid); pred/true 1.183 valid vs 1.145 invalid.

## Overall reading

If FPs are (a) brighter than road TPs, (b) as bright as or brighter than true
markings, (c) less warm/saturated than true markings, and (d) road-like in LiDAR
intensity, that is consistent with predictions being *associated with a
luminance-based RGB shortcut*: bright near-neutral road surfaces (glare, light
concrete, overexposed pixels) carry the same high-RGB signature as white paint.
