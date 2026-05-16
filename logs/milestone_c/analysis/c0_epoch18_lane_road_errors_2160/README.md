# C0 Lane-To-Road Error Analysis

This is a new sampled inference pass, not the exact original epoch-18 validation sample.

## Run

- checkpoint: `/home/coder/project/logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/checkpoints/ckpt_epoch_00018.pth`
- checkpoint_epoch: `18`
- split: `validation`
- steps: `2160`
- seed: `42`
- device: `cuda`

## Outputs

- `frame_error_summary.csv`
- `top_frames_by_lane_to_road.csv`
- `group_intensity_summary.csv`
- `group_range_summary.csv`
- `distance_bucket_summary.csv`
- `confusion_matrix.npy`
- `intensity_hist_by_outcome.png`
- `summary.json`
