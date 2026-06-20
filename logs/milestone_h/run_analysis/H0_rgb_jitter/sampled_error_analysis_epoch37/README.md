# G1 Best-Checkpoint Sampled Error Analysis

Fresh sampled validation inference pass. This is not the exact training-time validation sample.

- run: `H0_rgb_jitter`
- checkpoint: `/home/coder/project/logs/milestone_h/runs/H0_rgb_jitter/checkpoints/ckpt_epoch_00037.pth`
- checkpoint epoch from checkpoint payload: `37`
- split: `validation`
- steps: `2160`
- seed: `42`
- device: `cuda`
- command args: `--checkpoint /home/coder/project/logs/milestone_h/runs/H0_rgb_jitter/checkpoints/ckpt_epoch_00037.pth --out-dir /home/coder/project/logs/milestone_h/run_analysis/H0_rgb_jitter/sampled_error_analysis_epoch37 --config /home/coder/project/logs/milestone_h/configs/h0_rgb_jitter.yml --device cuda --steps 2160 --seed 42`

## RGB-Validity Metrics

| stratum | active share | marking IoU | precision | recall | F1 | pred/true | road->marking | marking->road |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 1.000 | 0.540719 | 0.649759 | 0.763150 | 0.701904 | 1.175 | 459380 | 268250 |
| rgb_valid | 0.817 | 0.540583 | 0.647591 | 0.765889 | 0.701790 | 1.183 | 363876 | 207903 |
| rgb_invalid | 0.183 | 0.541221 | 0.657872 | 0.753226 | 0.702327 | 1.145 | 95504 | 60347 |

Use this file with `distance_bucket_metrics.csv`, `per_sequence_metrics.csv`,
`raw_subtype_rgb_stratified_metrics.csv`, and the top-frame CSVs to understand
whether G1's aggregate result is driven by RGB-valid points, a distance band,
or a small number of sequences/frames.
