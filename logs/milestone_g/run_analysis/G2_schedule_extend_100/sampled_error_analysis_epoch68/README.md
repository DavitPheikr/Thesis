# G1 Best-Checkpoint Sampled Error Analysis

Fresh sampled validation inference pass. This is not the exact training-time validation sample.

- run: `G2_schedule_extend_100`
- checkpoint: `/home/coder/project/logs/milestone_g/runs/G2_schedule_extend_100/checkpoints/ckpt_epoch_00068.pth`
- checkpoint epoch from checkpoint payload: `68`
- split: `validation`
- steps: `2160`
- seed: `42`
- device: `cuda`
- command args: `--checkpoint /home/coder/project/logs/milestone_g/runs/G2_schedule_extend_100/checkpoints/ckpt_epoch_00068.pth --out-dir /home/coder/project/logs/milestone_g/run_analysis/G2_schedule_extend_100/sampled_error_analysis_epoch68 --config /home/coder/project/logs/milestone_g/configs/g1_schedule_extend.yml --device cuda --steps 2160 --seed 42`

## RGB-Validity Metrics

| stratum | active share | marking IoU | precision | recall | F1 | pred/true | road->marking | marking->road |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 1.000 | 0.540046 | 0.626677 | 0.796193 | 0.701337 | 1.271 | 532367 | 229626 |
| rgb_valid | 0.817 | 0.540187 | 0.622907 | 0.802675 | 0.701456 | 1.289 | 427293 | 174771 |
| rgb_invalid | 0.183 | 0.539515 | 0.641284 | 0.772710 | 0.700889 | 1.205 | 105074 | 54855 |

Use this file with `distance_bucket_metrics.csv`, `per_sequence_metrics.csv`,
`raw_subtype_rgb_stratified_metrics.csv`, and the top-frame CSVs to understand
whether G1's aggregate result is driven by RGB-valid points, a distance band,
or a small number of sequences/frames.
