# Road Marking3 Dataset Intensity Analysis Code

Run from the project root:

```bash
./panda/bin/python logs/milestone_d/dataset_analysis/road_marking3_intensity/analysis_code/analyze_road_marking3_intensity.py
```

This script analyzes the PandaSet forward-facing LiDAR under the Milestone D
`road_marking3` remap:

- `road`: raw `7`
- `marking`: raw `8 + 9 + 10`
- `other`: all other non-ignored raw classes

Validation rules built into the script:

- uses the frozen train/validation/test split files under `configs/splits`
- uses the same raw-to-remapped label function as the training dataset
- validates full training counts against `logs/milestone_d/road_marking3_training_statistics.json`
- preserves the C0/D distance buckets: `0-10`, `10-20`, `20-30`, `30-40`, `40-60`, `60+`
- uses raw PandaSet intensity values, not standardized model inputs
- writes provenance to `manifest.json` and `README.md`

Use `--max-frames` only for local debugging. Full thesis outputs should be
generated without frame limits.
