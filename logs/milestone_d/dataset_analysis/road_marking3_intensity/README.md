# Road Marking3 Dataset Intensity Analysis

This analysis describes the dataset under the Milestone D remap before using any D0 model predictions.

## Scope

- generated_at: `2026-06-10T11:25:56`
- script: `analysis_code/analyze_road_marking3_intensity.py`
- dataset_root: `/home/pheikara/University/Y3S2/Thesis/Pipeline/pandaset/PandaSet`
- sensor: forward-facing LiDAR only (`sensor_id=1`)
- splits: training, validation, test
- distance buckets: `0-10`, `10-20`, `20-30`, `30-40`, `40-60`, `60+` meters

## Class Definition

- `road`: raw `7 Road`
- `marking`: raw `8 Lane Line Marking` + raw `9 Stop Line Marking` + raw `10 Other Road Marking`
- `other`: all other non-ignored raw classes

## Outputs

- `class_intensity_summary.csv`: global raw-intensity summary per remapped class
- `split_class_intensity_summary.csv`: same summary split by train/validation/test
- `distance_bucket_intensity_summary.csv`: class intensity by distance bucket
- `plots/class_intensity_histograms.png`: normalized intensity histograms
- `plots/distance_intensity_medians.png`: median intensity and IQR by distance
- `manifest.json`: provenance and validation details

## Key Checks

- frames_processed: `6080`
- training count validation: `{'skipped': False, 'computed': {'road': 119562394, 'marking': 5129328, 'other': 173473484}, 'expected': {'road': 119562394, 'marking': 5129328, 'other': 173473484}}`

## Initial Findings

- Global median intensity: road `28.0`, marking `33.0`, other `19.0`.
- Road and marking middle-50% intensity overlap width: `3.0` raw-intensity units.
- Marking and other middle-50% intensity overlap width: `0.0` raw-intensity units.
- Use the distance plot to check whether marking remains brighter than road at the same range. If marking and road medians converge by distance, remaining D0 marking-road confusion is likely partly input-limited for LiDAR-only training.
- This analysis is dataset-only. It does not use D0 predictions and is therefore relevant for D1/D2 and for deciding whether Milestone E RGB/color features are justified.

## Split Stability

| split | road median | marking median | other median | marking count |
| --- | ---: | ---: | ---: | ---: |
| training | 28.0 | 33.0 | 19.0 | 5,129,328 |
| validation | 29.0 | 32.0 | 17.0 | 747,235 |
| test | 28.0 | 32.0 | 20.0 | 725,754 |

