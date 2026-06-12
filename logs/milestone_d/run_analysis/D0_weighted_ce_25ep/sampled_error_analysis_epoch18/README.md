# D0 Epoch-18 Sampled Marking Error Analysis

This is a fresh sampled validation inference pass. It is not the exact original epoch-18 validation sample from training.

## Provenance

- generated_at: `2026-06-10T12:23:27`
- script: `logs/milestone_d/run_analysis/D0_weighted_ce_25ep/analysis_code/analyze_d0_marking_errors.py`
- config: `/home/coder/project/logs/milestone_d/configs/d0_weighted_ce.yml`
- checkpoint: `/home/coder/project/logs/milestone_d/runs/D0_weighted_ce_25ep/checkpoints/ckpt_epoch_00018.pth`
- checkpoint_epoch: `18`
- split: `validation`
- steps: `2160`
- seed: `42`
- device: `cuda`

## Class Order

- active class `0`: road
- active class `1`: marking
- active class `2`: other

## Outputs

- `summary.json`
- `confusion_matrix.npy`
- `group_intensity_summary.csv`
- `group_range_summary.csv`
- `distance_bucket_summary.csv`
- `frame_error_summary.csv`
- `top_frames_by_marking_to_road.csv`
- `top_frames_by_road_to_marking.csv`
- `top_frames_by_other_to_marking.csv`
- `raw_marking_subtype_summary.csv`
- `distance_raw_marking_subtype_summary.csv`
- `plots/intensity_hist_by_outcome.png`

## Initial Findings

- sampled marking IoU: `0.441153`
- sampled marking precision: `0.514379`
- sampled marking recall: `0.756034`
- predicted/true marking ratio: `1.470`
- worst frame by marking-to-road count: `037/52` with `1065` marking points predicted as road
- easiest raw marking subtype by recall: raw `9` `stop_line_marking` recall `0.922971114167813`
- hardest raw marking subtype by recall: raw `8` `lane_line_marking` recall `0.7459396751740139`

Use the plot script in `analysis_code/plot_d0_marking_error_analysis.py` to create the remaining diagnostic figures and `diagnostic_plots.md`.
