# D0 Sampled Diagnostic Plots

These plots summarize the D0 epoch-18 sampled validation analysis. They are designed to explain marking misses and false-positive markings.

## `distance_marking_outcomes.png`

Purpose: shows what happens to true marking points in each distance bucket.

How to read it: green is correctly predicted marking, red is marking predicted as road, purple is marking predicted as other. The blue line is true marking support.

Finding: largest marking support is `10_20m` with `647,702` true marking points. Highest marking-to-road rate is `30_40m` at `26.3%`.

## `distance_marking_intensity_medians.png`

Purpose: compares intensity medians for correct markings, missed markings, correct road, and road false positives by distance.

How to read it: if red marking-to-road tracks gray road-TP more closely than green marking-TP, the miss is intensity-consistent with road at that range.

## `frame_marking_to_road_scatter.png`

Purpose: identifies whether missed markings are concentrated in a small set of frames.

Finding: top sampled frame by marking-to-road count is `037/52` with `1,065` marking points predicted as road.

## `frame_road_to_marking_scatter.png` and `frame_other_to_marking_scatter.png`

Purpose: identifies false-positive marking concentration from road and other.

How to read it: high points indicate frames where D0 is over-predicting marking. These frames are good viewer targets.

## `confusion_matrix_row_normalized.png`

Purpose: shows the sampled confusion matrix as row percentages and counts.

Finding: sampled marking recall is `75.6%`, marking-to-road is `24.0%`, road-to-marking is `2.6%`, and other-to-marking is `0.2%`.

## `raw_marking_subtype_recall.png`

Purpose: checks whether D0 learned all merged raw marking classes or mostly one subtype.

Finding: hardest subtype in this sampled pass is raw `8` `lane_line_marking` with recall `74.6%`.

## Provenance

- checkpoint: `/home/coder/project/logs/milestone_d/runs/D0_weighted_ce_25ep/checkpoints/ckpt_epoch_00018.pth`
- steps: `2160`
- seed: `42`
- device: `cuda`
- note: this is a fresh sampled inference pass, not the exact training validation sample.
