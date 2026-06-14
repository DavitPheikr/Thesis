# F0 Conclusions

## Main Conclusion

F0 is the first clear improvement over the D0 LiDAR-only baseline and the E0
RGB-front run. The official F0 checkpoint is epoch `13`, selected
by maximum raw marking IoU.

| metric | D0 epoch 18 | E0 epoch 14 | F0 epoch 13 |
| --- | ---: | ---: | ---: |
| marking IoU | 0.440294 | 0.438439 | 0.482770 |
| F1 | 0.611394 | 0.609604 | 0.651173 |
| precision | 0.514788 | 0.470298 | 0.549249 |
| recall | 0.752636 | 0.866170 | 0.799543 |
| mIoU | 0.780829 | 0.780658 | 0.798682 |

Interpretation: E0 showed that RGB increased recall but overpredicted marking.
F0 softened the marking weight and widened the first feature embedding, improving
precision while keeping recall above D0.

## Best Epoch vs Final Epoch

F0 still should not use the final epoch as the official checkpoint.

| metric | epoch 13 best | epoch 25 final |
| --- | ---: | ---: |
| marking IoU | 0.482770 | 0.449982 |
| precision | 0.549249 | 0.484524 |
| recall | 0.799543 | 0.863239 |
| val loss | 0.118228 | 0.106725 |

Validation loss continues improving after the best marking IoU, so loss alone
would again select a later, more recall-heavy but less balanced checkpoint.

## Overprediction

F0 reduced E0's overprediction pattern. Training-time marking predicted/true
ratio at the best checkpoint:

```text
E0: 1.842
F0: 1.456
```

The sampled epoch-13 pass gives:

```text
all:         pred/true 1.463, IoU 0.480235
rgb_valid:   pred/true 1.550, IoU 0.471318
rgb_invalid: pred/true 1.148, IoU 0.519858
```

## Remaining Questions

Use the sampled CSVs and plots to inspect whether remaining errors are still
concentrated in RGB-valid points, long range, specific validation sequences,
or raw marking subtypes. The key files are:

- `sampled_error_analysis_epoch13/rgb_valid_stratified_metrics.csv`
- `sampled_error_analysis_epoch13/per_sequence_metrics.csv`
- `sampled_error_analysis_epoch13/distance_bucket_metrics.csv`
- `sampled_error_analysis_epoch13/raw_subtype_rgb_stratified_metrics.csv`
- `plots/README.md`

## Provenance

- sampled analysis checkpoint: `/home/coder/project/logs/milestone_f/runs/F0_rgb_soft_weights/checkpoints/ckpt_epoch_00013.pth`
- sampled steps: `2160`
- sampled seed: `42`
- sampled split: `validation`
