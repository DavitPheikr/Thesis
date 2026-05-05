# Milestone C Notes

Generated: 2026-05-05

Milestone C is the real RandLA-Net experiment stage. It starts from the Milestone B dataset/class-weight foundation and moves toward C0-C4 training runs.

## Current C0 State

C0 is now defined as:

```text
RandLA-Net + xyz_ego + standardized intensity + SemSegRandomSampler + class-weighted CE
```

Important current decisions:

- Use `SemSegRandomSampler` for C0.
- Defer `SemSegSpatiallyRegularSampler` because it eagerly preprocesses the full split before epoch 1.
- Keep Open3D-native measured-count class weights; server verification confirmed they are active in the CE loss.
- Use the server dataset path `/home/coder/project/pandaset/PandaSet` for cloud runs.

## Key Reports

- `logs/milestone_c/reports/c0_sampler_and_server_readiness.md`
  - server readiness, dataset checks, sampler benchmark, smoke results, and C0 sampler decision.
- `logs/milestone_c/reports/class_weight_decision.md`
  - class-weight policy and server loss verification.
- `logs/milestone_c/reports/validation_metrics_prep.md`
  - validation metric plumbing, class indexing, and artifact design.
- `logs/milestone_c/reports/c0_prep_summary.md`
  - historical C0 prep summary plus current server update.
- `docs/milestone_c_option_a_execution_plan.md`
  - full C0-C4 experiment plan.

## Server-Specific Reminder

Server activation:

```bash
cd /home/coder/project
source envStart.sh
```

The server-specific configs created under `logs/milestone_c/configs/` may be generated and untracked. Before a long run, inspect the config snapshot carefully and ensure it is not still a smoke config with `steps_per_epoch_train: 1` and `steps_per_epoch_valid: 1`.
