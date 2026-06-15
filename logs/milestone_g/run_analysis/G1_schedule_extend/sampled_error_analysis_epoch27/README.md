# G1 Best-Checkpoint Sampled Error Analysis

Fresh sampled validation inference pass. This is not the exact training-time validation sample.

- run: `G1_schedule_extend`
- checkpoint: `/home/coder/project/logs/milestone_g/runs/G1_schedule_extend/checkpoints/ckpt_epoch_00027.pth`
- checkpoint epoch from checkpoint payload: `27`
- split: `validation`
- steps: `2160`
- seed: `42`
- device: `cuda`
- command args: `--checkpoint /home/coder/project/logs/milestone_g/runs/G1_schedule_extend/checkpoints/ckpt_epoch_00027.pth --out-dir /home/coder/project/logs/milestone_g/run_analysis/G1_schedule_extend/sampled_error_analysis_epoch27 --config /home/coder/project/logs/milestone_g/configs/g1_schedule_extend.yml --device cuda --steps 2160 --seed 42 --rgb-valid-threshold 0.5`

## RGB-Validity Metrics

| stratum | active share | marking IoU | precision | recall | F1 | pred/true | road->marking | marking->road |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 1.000 | 0.534884 | 0.631334 | 0.777838 | 0.696970 | 1.232 | 507459 | 250859 |
| rgb_valid | 0.817 | 0.535114 | 0.621480 | 0.793840 | 0.697165 | 1.277 | 424004 | 182172 |
| rgb_invalid | 0.183 | 0.533968 | 0.674026 | 0.719865 | 0.696192 | 1.068 | 83455 | 68687 |

Use this file with `distance_bucket_metrics.csv`, `per_sequence_metrics.csv`,
`raw_subtype_rgb_stratified_metrics.csv`, and the top-frame CSVs to understand
whether G1's aggregate result is driven by RGB-valid points, a distance band,
or a small number of sequences/frames.
