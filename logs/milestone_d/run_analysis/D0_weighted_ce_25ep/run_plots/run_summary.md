# D0 Weighted CE Run Summary

This report is generated from the official D0 run artifacts. It does not run new inference.

## Provenance

- generated_at: `2026-06-10T11:04:42`
- run_dir: `logs/milestone_d/runs/D0_weighted_ce_25ep`
- script: `logs/milestone_d/run_analysis/D0_weighted_ce_25ep/analysis_code/plot_d0_run.py`
- label_mode: `road_marking3`
- class index order after Open3D ignore filtering: `0 road`, `1 marking`, `2 other`
- metric alias: CSV columns named `lane_*` mean `marking_*` for Milestone D

## Headline Metrics

| metric | epoch 1 | epoch 18 best marking IoU | epoch 25 final | best epoch | best value |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_loss | 0.397247 | 0.132025 | 0.116028 | 24 | 0.115669 |
| val_loss | 0.305221 | 0.184213 | 0.167437 | 24 | 0.157847 |
| mIoU | 0.650682 | 0.780829 | 0.771793 | 18 | 0.780829 |
| marking IoU (`lane_iou`) | 0.201979 | 0.440294 | 0.410337 | 18 | 0.440294 |
| marking F1 (`lane_f1`) | 0.336077 | 0.611394 | 0.581900 | 18 | 0.611394 |
| marking precision | 0.230028 | 0.514788 | 0.451201 | 11 | 0.540384 |
| marking recall | 0.623554 | 0.752636 | 0.819195 | 22 | 0.852056 |

## Epoch Selection

- Best checkpoint by raw marking IoU is epoch `18`.
- Epoch `25` has higher recall but lower precision and lower marking IoU than epoch `18`.
- Best checkpoint selection is post-run by raw `lane_iou`/marking IoU, not validation loss and not the smoothed scheduler metric.

## Marking Prediction Share

| epoch | true marking share | predicted marking share | predicted/true ratio | marking->road | road->marking | other->marking |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 18 | 1.666% | 2.435% | 1.46 | 95,638 | 249,846 | 28,267 |
| 25 | 1.569% | 2.848% | 1.82 | 65,298 | 346,403 | 21,488 |

## Initial Reading

- D0 learned the marking class substantially in early epochs: marking IoU rises from epoch 1 to epoch 18.
- After epoch 18, recall stays high but precision drops enough that marking IoU and F1 decline.
- The final checkpoint is therefore not the best checkpoint for the positive class.
- The LR scheduler reduced LR after the plateau, but the best raw marking IoU had already occurred before the reduction produced a clear improvement.
- The main run-level symptom to investigate in sampled error analysis is marking overprediction: predicted marking share grows relative to true marking share.

## Generated Figures

- `run_plots/metrics_overview.png`
- `run_plots/loss_curves.png`
- `run_plots/per_class_iou.png`
- `run_plots/marking_precision_recall_f1.png`
- `run_plots/marking_recall_by_distance.png`
- `run_plots/class_true_vs_predicted_share.png`
- `run_plots/runtime_and_memory.png`
- `run_plots/confusion_best_marking_epoch_018.png`
- `run_plots/confusion_epoch_025.png`

