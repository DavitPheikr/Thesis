# Milestone C Run Summary: C0_baseline_full_30ep_random_bs1

## Headline Curves

| metric | first epoch | final epoch | best epoch | best value |
| --- | ---: | ---: | ---: | ---: |
| train_loss | 0.433675 | 0.113191 | 30 | 0.113191 |
| val_loss | 0.461187 | 0.271130 | 20 | 0.243199 |
| mIoU | 0.613929 | 0.683506 | 18 | 0.703805 |
| lane_iou | 0.187062 | 0.264658 | 18 | 0.310355 |
| lane_f1 | 0.315167 | 0.418545 | 18 | 0.473696 |

## Final Epoch Lane Metrics

```text
epoch          30
lane_iou       0.264658
lane_precision 0.321885
lane_recall    0.598172
lane_f1        0.418545
```

## Final Epoch Class Shares

| class | true percent | predicted percent | support | predicted count |
| --- | ---: | ---: | ---: | ---: |
| road | 38.358% | 41.804% | 4,514,429 | 4,919,925 |
| lane | 0.800% | 1.487% | 94,197 | 175,050 |
| other | 60.841% | 56.709% | 7,160,431 | 6,674,082 |

## Generated Figures

- `plots/metrics_overview.png`
- `plots/loss_curves.png`
- `plots/per_class_iou.png`
- `plots/lane_precision_recall_f1.png`
- `plots/lane_recall_by_distance.png`
- `plots/runtime_and_memory.png`
- `plots/class_true_vs_predicted_share.png`
- `plots/confusion_epoch_030.png`
- `plots/confusion_best_lane_f1_epoch_018.png`

## Reading Guide

- Prefer the best validation epoch for analysis if lane F1/IoU peaks before the final epoch.
- Watch lane precision and predicted-lane percent together; high recall with very low precision means lane over-prediction.
- Use distance-bucket lane recall to see whether the model loses lanes at range.
- Use row-normalized confusion matrices to diagnose whether lane is confused with road or other.
