# Milestone B Implementation Pipeline Guide

## Execution Guide: Training-Ready Pipeline

---

## Preamble — Three Certainty Categories

Before reading any step in this guide, internalize the three certainty categories defined in the
Milestone B Project Context document. They appear throughout this guide:

- **[FIXED]** — project truth; cannot change; safe to code against now
- **[SAMPLE-VERIFIED]** — verified on sequence `001`, frame `0` only; must not be assumed across sequences
- **[UNVERIFIED]** — must be discovered locally before any code depending on it is finalized

The tiny real training sanity run is a **Milestone B deliverable**. No prior sanity run exists.
Milestone B inherits only the Day 1–4 build foundation (environment, single-frame adapter,
config scaffold, constructing RandLA-Net). Nothing beyond that is assumed.

---

## Canonical Source of Truth for Measured Artifacts

| Artifact                                         | Canonical location                           | Notes                                     |
| ------------------------------------------------ | -------------------------------------------- | ----------------------------------------- |
| Train split IDs                                  | `configs/splits/train.txt`                   | One ID per line                           |
| Val split IDs                                    | `configs/splits/val.txt`                     | One ID per line                           |
| Test split IDs                                   | `configs/splits/test.txt`                    | One ID per line                           |
| All measured statistics + loss weight candidates | `logs/milestone_b_training_statistics.json`  | **Single authoritative source**           |
| SemSegLoss interface                             | `logs/milestone_b_loss_interface.json`       | Written by Phase 2, Day 3                 |
| Dataset class interface                          | `logs/milestone_b_dataset_interface.json`    | Written by Phase 2, Day 3                 |
| Pipeline interface                               | `logs/milestone_b_pipeline_interface.json`   | Written by Phase 2, Day 3                 |
| Per-sequence intensity samples                   | `logs/stats_per_seq/{seq_id}.npy`            | Written per-sequence by statistics script |
| Per-sequence statistics progress                 | `logs/milestone_b_statistics_progress.jsonl` | Resume checkpoint                         |
| Per-sequence audit progress                      | `logs/milestone_b_audit_progress.jsonl`      | Resume checkpoint                         |
| YAML config                                      | `configs/randlanet_pandaset_ff_lane3.yml`    | Mirrors JSON; JSON wins on conflict       |
| Stop conditions                                  | `logs/milestone_b_stop_conditions.txt`       | Must be written before full training      |

**Drift rule:** If the YAML config value disagrees with the statistics JSON, the JSON wins. The
`check_dataset_class.py` script enforces this programmatically.

---

## Script Execution Rules

All scripts run from the **project root**: `/home/pheikara/University/Y3S2/Thesis/Pipeline`.
Relative paths (`logs/...`, `configs/...`, `src/...`) are correct only from that directory.

Use `set -o pipefail` for all piped shell commands:

```bash
set -o pipefail
python tools/script.py 2>&1 | tee logs/report.txt
```

All major scripts must emit one of these as their final printed line:

```
script_status PASS
script_status FAIL
```

and must exit with code `0` or `1` accordingly. This allows agent orchestration to detect failure
without parsing logs.

---

## Day Estimate and Rationale

**Estimated duration: 6 days**

Day count is driven by the dependency chain: each phase depends on artifacts from the previous
one. Some phases (statistics computation, interface discovery) cannot be parallelized. The
estimate assumes a competent implementer; a coding agent may need Day 7 as contingency for
sanity-check debugging or weight-format issues.

| Day   | Phase                                   | What gates it                                   |
| ----- | --------------------------------------- | ----------------------------------------------- |
| Day 1 | Phase 1a: Pre-flight + Sequence Audit   | Nothing: must be first                          |
| Day 2 | Phase 1b: Split Freeze                  | Requires audit manifest                         |
| Day 3 | Phase 2: Open3D-ML Interface Discovery  | Split must exist (training sequences known)     |
| Day 4 | Phase 3: Statistics + Loss Weights      | Interface JSON must exist (weight format known) |
| Day 5 | Phase 4: Dataset Class + Config Update  | Statistics JSON must exist                      |
| Day 6 | Phase 5: Sanity Check + Stop Conditions | Dataset class must be complete                  |

---

# Day 1 — Phase 1a: Pre-flight Safety Check and Sequence Audit

## Objective

Confirm that `set_sensor(1)` persists and semseg alignment holds across multiple frames, then
audit all 103 local sequences to discover which have usable semseg and which contain lane points.

## Why Day 1

All later work depends on multi-sequence data loading being trustworthy. The single-frame
verification from Days 1–4 does not prove persistence. This day establishes that the data
infrastructure is safe to use at scale.

---

### Step 1.1 — Pre-flight multi-frame check

**Why this step exists:**
`set_sensor(1)` was verified on exactly one frame (sequence `001`, frame `0`). If it does not
persist across repeated frame accesses, every large-scale scan produces silently contaminated
output: wrong statistics, wrong class counts, wrong everything. This check must pass before
any scan runs.

**Create** `tools/preflight_multi_frame_check.py`:

```python
"""
Pre-flight check: verify set_sensor(1) persistence and semseg alignment
across 5 consecutive frames. Must pass before any multi-sequence scan.

Uses get_frame_count from pandaset_compat — never hardcodes frame counts.
Emits script_status PASS/FAIL as final line.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from thesis_pipeline.core.pandaset_compat import get_frame_count
from pandaset import DataSet

def main():
    root = Path('logs/dataset_root.txt').read_text().strip()
    seq_id = Path('logs/day2_chosen_sequence.txt').read_text().strip()

    ds = DataSet(root)
    seq = ds[seq_id]
    try:
        seq.load_lidar().load_semseg()
    except Exception:
        seq.load_lidar()
        seq.load_semseg()

    seq.lidar.set_sensor(1)
    n_frames = get_frame_count(seq)
    frames_to_check = list(range(min(5, n_frames)))

    all_ok = True
    for fi in frames_to_check:
        pc_df = seq.lidar[fi]
        semseg_df = seq.semseg[fi]

        if 'd' in pc_df.columns:
            unique_sensors = sorted(pc_df['d'].unique().tolist())
            sensor_ok = (unique_sensors == [1])
        else:
            sensor_ok = len(pc_df) > 0
            unique_sensors = 'col_absent'

        aligned = semseg_df.index.isin(pc_df.index)
        alignment_ok = (aligned.sum() == len(pc_df))

        print(f'frame {fi}: n={len(pc_df)} sensors={unique_sensors} '
              f'sensor_ok={sensor_ok} alignment_ok={alignment_ok}')

        if not sensor_ok or not alignment_ok:
            all_ok = False

    print()
    if all_ok:
        print('preflight_multi_frame_ok True')
        print('script_status PASS')
        sys.exit(0)
    else:
        print('preflight_multi_frame_ok False')
        print('FATAL: set_sensor(1) persistence or semseg alignment failed.')
        print('  If set_sensor(1) does not persist: add seq.lidar.set_sensor(1)')
        print('  before each frame access in all later scripts and re-test.')
        print('  Record the working access pattern before proceeding.')
        print('script_status FAIL')
        sys.exit(1)

if __name__ == '__main__':
    main()
```

**Run:**

```bash
set -o pipefail
python tools/preflight_multi_frame_check.py 2>&1 | tee logs/milestone_b_preflight.txt
```

**Pass condition:** All 5 frames show `sensor_ok=True` and `alignment_ok=True`. Final line is
`script_status PASS`.

**If FAIL:** If `set_sensor(1)` does not persist, add `seq.lidar.set_sensor(1)` before **every**
individual frame access `seq.lidar[fi]` in all later scripts. Re-run this check and confirm it passes
with the re-application pattern. Record the working pattern in
`logs/milestone_b_preflight_sensor_pattern.txt` (e.g., `reapply_per_frame: true`). **Do not
proceed to Step 1.2 until this check passes.**

---

### Step 1.2 — Full sequence pool audit

**Why this step exists:**
103 local sequence directories exist but only a subset have usable semseg. Lane presence must
be checked across all frames per sequence (not sampled frames — sampling produces false
negatives for sparse lane content). The audit uses a JSONL progress file so crashes mid-run
are recoverable.

**Create** `tools/audit_semseg_sequences.py`:

```python
"""
Audit all local PandaSet sequences for semseg availability and lane presence.
Checks ALL forward-only frames per sequence (not sampled subset).
Writes per-sequence results to JSONL incrementally for crash recovery.
Consolidated manifest JSON written at completion.

Distinguishes:
  - frames_failed: exception-caused skips
  - frames_semseg_missing: structural semseg absence for that frame index
  - sensor_first_frame_ok: did set_sensor(1) produce forward-only data on frame 0

Emits script_status PASS/FAIL as final line.
"""
import sys
import gc
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from thesis_pipeline.core.pandaset_compat import get_frame_count
from thesis_pipeline.adapters.pandaset_ff_lane3 import remap_raw_pandaset_ids
from pandaset import DataSet
import numpy as np

LANE_RAW_ID = 8
PROGRESS_FILE = Path('logs/milestone_b_audit_progress.jsonl')
MANIFEST_FILE = Path('logs/milestone_b_sequence_manifest.json')


def load_completed(progress_path):
    """Return set of seq_ids already successfully completed."""
    completed = set()
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                if rec.get('error') is None:
                    completed.add(rec['seq_id'])
            except json.JSONDecodeError:
                pass
    return completed


def audit_one_sequence(ds, seq_id, sensor_reapply_per_frame=False):
    result = {
        'seq_id': seq_id,
        'semseg_available': False,
        'lane_present': False,
        'lidar_frame_count': 0,
        'semseg_frame_count': 0,
        'frames_scanned': 0,
        'frames_with_lane': 0,
        'total_lane_points': 0,
        'frames_failed': 0,
        'frames_semseg_missing': 0,
        'sensor_first_frame_ok': False,
        'error': None,
    }
    try:
        seq = ds[seq_id]
        try:
            seq.load_lidar().load_semseg()
        except Exception:
            seq.load_lidar()
            seq.load_semseg()

        lidar_n = get_frame_count(seq)
        result['lidar_frame_count'] = lidar_n

        try:
            semseg_n = len(seq.semseg.data)
        except Exception:
            semseg_n = lidar_n
        result['semseg_frame_count'] = semseg_n

        if semseg_n == 0:
            result['error'] = 'semseg_empty'
            return result

        result['semseg_available'] = True

        if not sensor_reapply_per_frame:
            seq.lidar.set_sensor(1)

        for fi in range(lidar_n):
            try:
                if sensor_reapply_per_frame:
                    seq.lidar.set_sensor(1)
                pc_df = seq.lidar[fi]

                # Check sensor filter on frame 0
                if fi == 0 and 'd' in pc_df.columns:
                    result['sensor_first_frame_ok'] = (
                        sorted(pc_df['d'].unique().tolist()) == [1]
                    )
                elif fi == 0:
                    result['sensor_first_frame_ok'] = len(pc_df) > 0

                # Check semseg availability for this frame
                if fi >= semseg_n:
                    result['frames_semseg_missing'] += 1
                    continue

                semseg_df = seq.semseg[fi]
                raw_labels = semseg_df.loc[pc_df.index, 'class'].to_numpy(dtype='int32')
                lane_count = int((raw_labels == LANE_RAW_ID).sum())

                result['frames_scanned'] += 1
                result['total_lane_points'] += lane_count
                if lane_count > 0:
                    result['frames_with_lane'] += 1

            except Exception as e:
                result['frames_failed'] += 1
                continue

        result['lane_present'] = result['total_lane_points'] > 0

    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    root = Path('logs/dataset_root.txt').read_text().strip()
    ds = DataSet(root)

    # Detect sensor access pattern from preflight log
    sensor_reapply = False
    preflight_pattern = Path('logs/milestone_b_preflight_sensor_pattern.txt')
    if preflight_pattern.exists():
        content = preflight_pattern.read_text()
        if 'reapply_per_frame: true' in content:
            sensor_reapply = True
    print(f'sensor_reapply_per_frame {sensor_reapply}')

    all_seq_ids = sorted([p.name for p in Path(root).iterdir() if p.is_dir()])
    print(f'total_sequence_dirs {len(all_seq_ids)}')

    completed = load_completed(PROGRESS_FILE)
    print(f'already_completed {len(completed)}')

    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

    results = []
    # Load already-completed records
    if PROGRESS_FILE.exists():
        for line in PROGRESS_FILE.read_text().splitlines():
            try:
                rec = json.loads(line)
                if rec.get('error') is None:
                    results.append(rec)
            except json.JSONDecodeError:
                pass

    with PROGRESS_FILE.open('a') as progress_f:
        for seq_id in all_seq_ids:
            if seq_id in completed:
                continue
            print(f'  auditing {seq_id}...', end=' ', flush=True)
            r = audit_one_sequence(ds, seq_id, sensor_reapply)
            print(f'semseg={r["semseg_available"]} lane={r["lane_present"]} '
                  f'failed={r["frames_failed"]} missing={r["frames_semseg_missing"]} '
                  f'err={r["error"]}')
            progress_f.write(json.dumps(r) + '\n')
            progress_f.flush()
            if r['error'] is None:
                results.append(r)
            del r
            gc.collect()

    # Build manifest
    semseg_pool = [r for r in results if r['semseg_available']]
    lane_pool = [r for r in semseg_pool if r['lane_present']]

    manifest = {
        'total_dirs': len(all_seq_ids),
        'semseg_enabled_count': len(semseg_pool),
        'lane_bearing_count': len(lane_pool),
        'semseg_enabled_ids': sorted(r['seq_id'] for r in semseg_pool),
        'lane_bearing_ids': sorted(r['seq_id'] for r in lane_pool),
        'non_lane_semseg_ids': sorted(
            r['seq_id'] for r in semseg_pool if not r['lane_present']
        ),
        'per_sequence_details': {r['seq_id']: r for r in semseg_pool},
        'global_frames_failed': sum(r['frames_failed'] for r in semseg_pool),
        'global_frames_semseg_missing': sum(r['frames_semseg_missing'] for r in semseg_pool),
    }

    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))
    print(f'\naudit_complete True')
    print(f'semseg_enabled_count {manifest["semseg_enabled_count"]}')
    print(f'lane_bearing_count {manifest["lane_bearing_count"]}')
    print(f'pool_size_adequate {manifest["semseg_enabled_count"] >= 70}')
    print(f'global_frames_failed {manifest["global_frames_failed"]}')

    ok = manifest['semseg_enabled_count'] >= 70 and manifest['lane_bearing_count'] >= 10
    print(f'script_status {"PASS" if ok else "FAIL"}')
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
```

**Run (this will be slow — allow 30–90 min for all 103 sequences):**

```bash
mkdir -p logs/stats_per_seq
set -o pipefail
python tools/audit_semseg_sequences.py 2>&1 | tee logs/milestone_b_sequence_audit.txt
```

**Pass condition:**

- `semseg_enabled_count ≥ 70`
- `lane_bearing_count ≥ 10`
- `script_status PASS`
- `logs/milestone_b_sequence_manifest.json` written

**If semseg pool < 70:** The planned 58/9/9 split is not possible. Stop and diagnose. Do not
proceed.

**If global_frames_failed is large (>5% of total scanned frames):** Review the failure pattern
before proceeding to the split.

## Day 1 Deliverables

- `tools/preflight_multi_frame_check.py`
- `tools/audit_semseg_sequences.py`
- `logs/milestone_b_preflight.txt` (script_status PASS)
- `logs/milestone_b_audit_progress.jsonl`
- `logs/milestone_b_sequence_manifest.json`
- `logs/milestone_b_sequence_audit.txt` (script_status PASS)
- `logs/milestone_b_preflight_sensor_pattern.txt` (if sensor re-application was needed)

## Day 1 Pass Criteria

- Pre-flight passes on ≥5 consecutive frames
- Semseg-enabled pool size ≥70
- Lane-bearing sequence count ≥10
- Sensor access pattern for all later scripts is determined and consistent with the pre-flight result

---

# Day 2 — Phase 1b: Split Freeze

## Objective

Freeze the deterministic train / val / test split from the audited semseg-enabled pool. Confirm
lane presence in both holdouts by assertion.

## Why Day 2

Statistics computation (Day 4) must run on training-split sequences only. The split must be
frozen before statistics are computed. This day is short but gated: nothing from the audit can be
bypassed.

---

### Step 2.1 — Freeze split

**Create** `tools/freeze_split.py`:

```python
"""
Freeze the train/val/test split from the audited semseg-enabled sequence pool.

Rules:
  - val = 9 sequences, test = 9 sequences, train = remainder
  - both val and test must contain >= 1 lane-bearing sequence (asserted)
  - split is deterministic given the seed and manifest
  - provenance is fully recorded in the split report

Emits script_status PASS/FAIL as final line.
"""
import sys
import json
import random
import hashlib
from pathlib import Path

SEED = 42
VAL_SIZE = 9
TEST_SIZE = 9
MIN_LANE_PER_HOLDOUT = 1

MANIFEST_FILE = Path('logs/milestone_b_sequence_manifest.json')
SPLIT_DIR = Path('configs/splits')
REPORT_FILE = Path('logs/milestone_b_split_report.txt')


def main():
    manifest = json.loads(MANIFEST_FILE.read_text())
    manifest_hash = hashlib.md5(MANIFEST_FILE.read_bytes()).hexdigest()

    pool = sorted(manifest['semseg_enabled_ids'])
    lane_ids = set(manifest['lane_bearing_ids'])
    pool_size = len(pool)
    lane_seqs = [s for s in pool if s in lane_ids]

    assert pool_size >= VAL_SIZE + TEST_SIZE + 1, (
        f"Pool size {pool_size} too small for {VAL_SIZE}+{TEST_SIZE}+1 split")
    assert len(lane_seqs) >= 2, (
        f"Need >= 2 lane-bearing sequences for holdout guarantee; got {len(lane_seqs)}")

    rng = random.Random(SEED)

    # Guarantee at least 1 lane-bearing sequence per holdout
    lane_shuffled = lane_seqs.copy()
    rng.shuffle(lane_shuffled)
    lane_for_val = [lane_shuffled[0]]
    lane_for_test = [lane_shuffled[1]]
    reserved = set(lane_for_val + lane_for_test)

    # Fill remaining holdout slots from non-reserved pool
    remaining = [s for s in pool if s not in reserved]
    rng.shuffle(remaining)
    val_fill = remaining[:VAL_SIZE - 1]
    test_fill = remaining[VAL_SIZE - 1: VAL_SIZE - 1 + TEST_SIZE - 1]
    train = remaining[VAL_SIZE - 1 + TEST_SIZE - 1:]

    val = sorted(lane_for_val + val_fill)
    test = sorted(lane_for_test + test_fill)

    # Assertions
    assert len(val) == VAL_SIZE, f"val has {len(val)} != {VAL_SIZE}"
    assert len(test) == TEST_SIZE, f"test has {len(test)} != {TEST_SIZE}"
    assert not set(val) & set(test), "val/test overlap"
    assert not set(val) & set(train), "val/train overlap"
    assert not set(test) & set(train), "test/train overlap"
    assert set(val) | set(test) | set(train) == set(pool), "split does not cover pool"

    val_lane = [s for s in val if s in lane_ids]
    test_lane = [s for s in test if s in lane_ids]
    assert len(val_lane) >= MIN_LANE_PER_HOLDOUT, (
        f"val has {len(val_lane)} lane-bearing sequences, need >= {MIN_LANE_PER_HOLDOUT}")
    assert len(test_lane) >= MIN_LANE_PER_HOLDOUT, (
        f"test has {len(test_lane)} lane-bearing sequences, need >= {MIN_LANE_PER_HOLDOUT}")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / 'train.txt').write_text('\n'.join(sorted(train)))
    (SPLIT_DIR / 'val.txt').write_text('\n'.join(val))
    (SPLIT_DIR / 'test.txt').write_text('\n'.join(test))

    report_lines = [
        f'seed {SEED}',
        f'manifest_file {MANIFEST_FILE}',
        f'manifest_hash_md5 {manifest_hash}',
        f'pool_size {pool_size}',
        f'train_count {len(train)}',
        f'val_count {len(val)}',
        f'test_count {len(test)}',
        f'train_ids {sorted(train)}',
        f'val_ids {val}',
        f'test_ids {test}',
        f'val_lane_ids {val_lane}',
        f'test_lane_ids {test_lane}',
        f'lane_per_holdout_val {len(val_lane)}',
        f'lane_per_holdout_test {len(test_lane)}',
        f'lane_guaranteed_in_val True',
        f'lane_guaranteed_in_test True',
        f'split_status FROZEN',
    ]
    REPORT_FILE.write_text('\n'.join(report_lines))

    print('\n'.join(report_lines))
    print(f'script_status PASS')
    sys.exit(0)

if __name__ == '__main__':
    main()
```

**Run:**

```bash
set -o pipefail
python tools/freeze_split.py 2>&1 | tee logs/milestone_b_split_report.txt
```

**Pass condition:**

- `configs/splits/train.txt`, `val.txt`, `test.txt` all written
- `split_status FROZEN`
- Both holdouts have lane-bearing sequences (assertion passed)
- `script_status PASS`

**If assertion fails:** Pool does not have enough lane-bearing sequences. Review the audit
manifest. Consider excluding consistently-failing sequences only if the pool remains ≥70 after
exclusion and is documented.

## Day 2 Deliverables

- `tools/freeze_split.py`
- `configs/splits/train.txt`
- `configs/splits/val.txt`
- `configs/splits/test.txt`
- `logs/milestone_b_split_report.txt` (split_status FROZEN)

## Day 2 Pass Criteria

- Split is frozen with no overlaps
- Val and test each contain ≥1 lane-bearing sequence (assertion logged)
- Split provenance fully recorded (seed, manifest hash, lane allocation)
- `script_status PASS`

---

# Day 3 — Phase 2: Open3D-ML Interface Discovery

## Objective

Read the local Open3D-ML source code for `SemSegLoss`, the base dataset class, and the
`SemanticSegmentation` pipeline. Produce three machine-readable JSON interface artifacts
that all later code will depend on.

## Why Day 3

Phases 4 and 5 produce code that calls into Open3D-ML internals. Writing that code before
reading the local source will result in guessed interfaces that may silently corrupt loss weights,
break dataset class integration, or fail to invoke training correctly. The JSON artifacts produced
here act as the formal dependency contract between discovery and implementation.

This is primarily a **reading and recording** day. No training code is written today.

---

### Step 3.1 — Locate Open3D-ML source files

```bash
python -c "
import open3d.ml.torch as ml3d
import inspect, pathlib

# Dataset base class
try:
    from ml3d.datasets.base_dataset import BaseDataset
    print('base_dataset:', inspect.getfile(BaseDataset))
except Exception as e:
    print('base_dataset_err:', e)

# SemSegLoss
try:
    from ml3d.torch.modules.losses.semseg_loss import SemSegLoss
    print('semseg_loss:', inspect.getfile(SemSegLoss))
except Exception as e:
    print('semseg_loss_err:', e)

# SemanticSegmentation pipeline
try:
    from ml3d.torch.pipelines.semantic_segmentation import SemanticSegmentation
    print('pipeline:', inspect.getfile(SemanticSegmentation))
except Exception as e:
    print('pipeline_err:', e)

# SemSegSpatiallyRegularSampler
try:
    from ml3d.datasets.samplers.semseg_spatially_regular import SemSegSpatiallyRegularSampler
    print('sampler_available True')
    print('sampler:', inspect.getfile(SemSegSpatiallyRegularSampler))
except Exception as e:
    print('sampler_available False')
    print('sampler_err:', e)

# Stock PandaSet dataset class for reference
try:
    from ml3d.datasets.pandaset import PandaSet as StockPandaSet
    print('stock_pandaset:', inspect.getfile(StockPandaSet))
except Exception as e:
    print('stock_pandaset_err:', e)
" 2>&1 | tee logs/milestone_b_source_locations.txt
```

---

### Step 3.2 — Read source and produce interface JSON artifacts

Read each source file found in Step 3.1. For each file, answer the specific questions below and
write the results to the corresponding JSON artifact. **Do not guess. Read the actual code.**

**Questions to answer from `SemSegLoss` source:**

1. What is the exact class name of the loss? (e.g., `SemSegLoss`)
2. What is the full path to the source file?
3. How are `class_weights` consumed? Is the weight list expected to be:
   - length `num_classes` (covering only active classes: road/lane/other)?
   - length `num_classes + 1` (including the ignored class at index 0)?
   - some other length?
4. What Python type does the loss expect: Python list, numpy array, or torch.Tensor?
5. What dtype: float32, float64?
6. Is there an `ignore_label` or equivalent that maps to the `ignored_label_inds` config field?
7. Write one sentence summarizing how the weight-to-class mapping works.

**Write** `logs/milestone_b_loss_interface.json` with this structure:

```json
{
  "loss_class_name": "<discovered>",
  "loss_source_path": "<absolute path>",
  "weight_scope": "<active-classes-only | all-including-ignore | other: describe>",
  "weight_list_length": "<num_classes=3 | num_classes+1=4 | other: describe>",
  "weight_container_type": "<list | numpy_array | torch_tensor>",
  "weight_dtype": "<float32 | float64>",
  "class_order": "<[road, lane, other] or [ignore, road, lane, other] or other: describe>",
  "ignore_slot_policy": "<prepend_zero | omit | other: describe>",
  "short_note": "<one sentence summary of weight-to-class mapping>"
}
```

**Questions to answer from the base dataset class and stock PandaSet class:**

1. What is the name of the base class to inherit from?
2. What abstract methods must be implemented? (`get_data`, `__len__`, `get_split`, etc.)
3. What does `get_data(idx)` return? What are the exact key names in the return dict?
4. Does the pipeline concatenate `point` and `feat` into a single 4-channel input automatically,
   or must the dataset class or some preprocessing step do this? (**Critical: examine the
   pipeline's data-loading and preprocessing calls explicitly.**)
5. What config fields does the dataset class expect (e.g., `dataset_path`, `class_weights`, etc.)?
6. How does the pipeline split the dataset into train/val/test? (By calling a `get_split` method?
   By passing a split name? By registry?)
7. Is a dataset registry needed? If so, how is a custom class registered?

**Write** `logs/milestone_b_dataset_interface.json` with this structure:

```json
{
  "base_class": "<class name and import path>",
  "required_methods": ["<method1>", "<method2>"],
  "get_data_return_keys": {
    "points_key": "<discovered key name or 'point'>",
    "features_key": "<discovered key name or 'feat'>",
    "labels_key": "<discovered key name or 'label'>"
  },
  "concat_location": "<pipeline | dataset_class | model | preprocessing_step: describe>",
  "split_object_api": "<describe how the pipeline obtains train/val/test splits>",
  "required_cfg_fields": ["<field1>", "<field2>"],
  "registry_required": "<yes | no>",
  "registry_note": "<if yes: how to register>",
  "short_note": "<one paragraph summary of what the custom class must implement>"
}
```

**Questions to answer from the `SemanticSegmentation` pipeline source:**

1. What are the required arguments to the pipeline constructor?
2. What is the method or function to call to start training?
3. How can training be limited to a small number of iterations (for sanity check)? Options:
   - `max_epoch` config field?
   - Programmatic stop after N steps?
   - Other mechanism?
4. How is the loss path invoked? Is it part of the pipeline or separate?
5. What config fields does the pipeline require?

**Write** `logs/milestone_b_pipeline_interface.json` with this structure:

```json
{
  "pipeline_class": "<class name and import path>",
  "constructor_required_args": ["<arg1>", "<arg2>"],
  "training_method": "<method name to invoke training>",
  "iteration_limit_mechanism": "<max_epoch | programmatic_stop | other: describe>",
  "required_config_fields": ["<field1>", "<field2>"],
  "loss_invocation": "<describe how loss is called>",
  "short_note": "<one paragraph summary of how to assemble and run the pipeline>"
}
```

**Also write** `logs/milestone_b_open3d_interface_notes.txt` with full free-text notes from the
source reading, including any unexpected behaviors, version-specific differences from
documentation, and the full answers to all questions above in prose form.

---

### Step 3.3 — Verify interface artifacts are complete

```python
# Run inline or as a quick check script
import json
from pathlib import Path

for fname in [
    'logs/milestone_b_loss_interface.json',
    'logs/milestone_b_dataset_interface.json',
    'logs/milestone_b_pipeline_interface.json',
]:
    data = json.loads(Path(fname).read_text())
    # Check no field is still "<discovered>" placeholder
    for k, v in data.items():
        assert '<' not in str(v) or 'describe' in str(v).lower() or str(v).startswith('<'), \
            f"Unfilled field in {fname}: {k} = {v}"
    print(f'{fname}: OK')
print('All interface artifacts populated.')
```

**Pass condition:** All three JSON files exist with no remaining `<discovered>` placeholder values.

## Day 3 Deliverables

- `logs/milestone_b_source_locations.txt`
- `logs/milestone_b_loss_interface.json` (fully populated)
- `logs/milestone_b_dataset_interface.json` (fully populated, including `concat_location`)
- `logs/milestone_b_pipeline_interface.json` (fully populated)
- `logs/milestone_b_open3d_interface_notes.txt`

## Day 3 Pass Criteria

- All three interface JSON artifacts exist with no placeholder values
- `concat_location` is explicitly resolved in the dataset interface JSON
- Key names for `get_data` return dict are explicitly resolved
- SemSegLoss weight format (scope, length, type) is explicitly resolved
- Sampler availability in Open3D `0.19.0` is recorded

---

# Day 4 — Phase 3: Training Statistics and Loss Weights

## Objective

Compute intensity preprocessing statistics and class-count statistics on the training split only.
Derive both loss weight candidates. Record everything in the canonical statistics JSON.

## Why Day 4

The dataset class must not hardcode any statistics. Statistics come from the training split only,
never from val or test sequences. The loss weight candidates depend on both the measured class
counts (Day 4) and the discovered weight format (Day 3). This day cannot start until the split
(Day 2) and interface discovery (Day 3) are complete.

---

### Step 4.1 — Compute training statistics

**Design constraints enforced by this implementation:**

- Processes one sequence at a time; calls `gc.collect()` between sequences
- Per-sequence intensity samples saved to `logs/stats_per_seq/{seq_id}.npy` (reservoir: ~100K pts/seq)
- Per-sequence class counts written to JSONL progress file
- Final aggregation reads ALL completed per-sequence artifacts (not in-memory accumulators)
- Resume-safe: crashed runs skip already-completed sequences on restart
- Tracks semseg/LiDAR frame count mismatches; stops if skip rate > 5%

**Create** `tools/compute_training_statistics.py`:

```python
"""
Compute intensity statistics and class counts on the training split only.
Resume-safe: persists per-sequence artifacts; aggregates from ALL completed ones.

Output: logs/milestone_b_training_statistics.json (partial; weights added by next script)
        logs/stats_per_seq/{seq_id}.npy (per-sequence intensity reservoir samples)
        logs/milestone_b_statistics_progress.jsonl (resume checkpoint)

Emits script_status PASS/FAIL as final line.
"""
import sys
import gc
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from thesis_pipeline.core.pandaset_compat import get_frame_count
from thesis_pipeline.adapters.pandaset_ff_lane3 import remap_raw_pandaset_ids
from pandaset import DataSet, geometry as pds_geometry

TRAIN_SPLIT_FILE = Path('configs/splits/train.txt')
PROGRESS_FILE = Path('logs/milestone_b_statistics_progress.jsonl')
STATS_FILE = Path('logs/milestone_b_training_statistics.json')
PER_SEQ_DIR = Path('logs/stats_per_seq')
RESERVOIR_SIZE = 100_000
SKIP_RATE_THRESHOLD = 0.05

PREFLIGHT_PATTERN = Path('logs/milestone_b_preflight_sensor_pattern.txt')


def get_sensor_reapply():
    if PREFLIGHT_PATTERN.exists():
        return 'reapply_per_frame: true' in PREFLIGHT_PATTERN.read_text()
    return False


def reservoir_sample(existing, new_values, rng):
    """Add new_values into existing reservoir (list) using reservoir sampling."""
    for v in new_values:
        if len(existing) < RESERVOIR_SIZE:
            existing.append(v)
        else:
            j = rng.integers(0, len(existing) + 1)
            if j < RESERVOIR_SIZE:
                existing[j] = v


def load_completed():
    completed = {}
    if PROGRESS_FILE.exists():
        for line in PROGRESS_FILE.read_text().splitlines():
            try:
                rec = json.loads(line)
                if rec.get('error') is None:
                    completed[rec['seq_id']] = rec
            except json.JSONDecodeError:
                pass
    return completed


def process_sequence(ds, seq_id, sensor_reapply, rng):
    result = {
        'seq_id': seq_id,
        'lidar_frames': 0,
        'semseg_frames': 0,
        'frames_processed': 0,
        'frames_semseg_missing': 0,
        'frames_failed': 0,
        'class_counts': {0: 0, 1: 0, 2: 0, 3: 0},
        'intensity_sample_path': None,
        'error': None,
    }
    try:
        seq = ds[seq_id]
        try:
            seq.load_lidar().load_semseg()
        except Exception:
            seq.load_lidar()
            seq.load_semseg()

        lidar_n = get_frame_count(seq)
        result['lidar_frames'] = lidar_n
        try:
            semseg_n = len(seq.semseg.data)
        except Exception:
            semseg_n = lidar_n
        result['semseg_frames'] = semseg_n

        if not sensor_reapply:
            seq.lidar.set_sensor(1)

        intensity_buf = []
        for fi in range(lidar_n):
            try:
                if sensor_reapply:
                    seq.lidar.set_sensor(1)
                pc_df = seq.lidar[fi]

                if fi >= semseg_n:
                    result['frames_semseg_missing'] += 1
                    continue

                semseg_df = seq.semseg[fi]
                raw_labels = semseg_df.loc[pc_df.index, 'class'].to_numpy(dtype='int32')
                remapped = remap_raw_pandaset_ids(raw_labels)

                for cls in [0, 1, 2, 3]:
                    result['class_counts'][cls] += int((remapped == cls).sum())

                intensities = pc_df['i'].to_numpy(dtype='float32')
                reservoir_sample(intensity_buf, intensities, rng)
                result['frames_processed'] += 1

            except Exception as e:
                result['frames_failed'] += 1
                continue

        # Save per-sequence intensity sample
        if intensity_buf:
            arr = np.array(intensity_buf, dtype=np.float32)
            out_path = PER_SEQ_DIR / f'{seq_id}.npy'
            np.save(out_path, arr)
            result['intensity_sample_path'] = str(out_path)

    except Exception as e:
        result['error'] = str(e)

    return result


def aggregate_from_artifacts(completed_records):
    """Aggregate statistics by re-reading all per-sequence artifacts. Never uses in-memory."""
    all_intensity = []
    total_class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_frames_processed = 0
    total_frames_semseg_missing = 0

    for rec in completed_records.values():
        for cls in [0, 1, 2, 3]:
            total_class_counts[cls] += rec['class_counts'].get(str(cls),
                                       rec['class_counts'].get(cls, 0))
        total_frames_processed += rec['frames_processed']
        total_frames_semseg_missing += rec['frames_semseg_missing']

        npy_path = rec.get('intensity_sample_path')
        if npy_path and Path(npy_path).exists():
            arr = np.load(npy_path)
            all_intensity.append(arr)

    intensity_all = np.concatenate(all_intensity) if all_intensity else np.array([], dtype=np.float32)

    skip_rate = (total_frames_semseg_missing /
                 (total_frames_processed + total_frames_semseg_missing + 1e-9))

    return total_class_counts, intensity_all, total_frames_processed, total_frames_semseg_missing, skip_rate


def main():
    sensor_reapply = get_sensor_reapply()
    print(f'sensor_reapply_per_frame {sensor_reapply}')

    train_ids = [
        l.strip() for l in TRAIN_SPLIT_FILE.read_text().splitlines() if l.strip()
    ]
    print(f'training_sequences {len(train_ids)}')

    root = Path('logs/dataset_root.txt').read_text().strip()
    ds = DataSet(root)
    rng = np.random.default_rng(seed=42)

    PER_SEQ_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_completed()
    print(f'already_completed {len(completed)}')

    with PROGRESS_FILE.open('a') as pf:
        for seq_id in train_ids:
            if seq_id in completed:
                continue
            print(f'  processing {seq_id}...', end=' ', flush=True)
            r = process_sequence(ds, seq_id, sensor_reapply, rng)
            skip = r['frames_semseg_missing'] / max(1, r['frames_processed'] + r['frames_semseg_missing'])
            print(f'processed={r["frames_processed"]} failed={r["frames_failed"]} '
                  f'missing={r["frames_semseg_missing"]} err={r["error"]}')
            pf.write(json.dumps(r) + '\n')
            pf.flush()
            if r['error'] is None:
                completed[seq_id] = r
            del r
            gc.collect()

    # Aggregate from all completed artifacts (not in-memory)
    class_counts, intensity_all, n_proc, n_missing, skip_rate = aggregate_from_artifacts(completed)

    print(f'\nsequences_completed {len(completed)}')
    print(f'sequences_expected {len(train_ids)}')
    print(f'frames_processed {n_proc}')
    print(f'frames_semseg_missing {n_missing}')
    print(f'skip_rate {skip_rate:.4f}')
    print(f'class_counts {class_counts}')

    if skip_rate > SKIP_RATE_THRESHOLD:
        print(f'FAIL: skip_rate {skip_rate:.4f} exceeds threshold {SKIP_RATE_THRESHOLD}')
        print(f'  Investigate semseg frame count mismatches before proceeding.')
        print('script_status FAIL')
        sys.exit(1)

    assert class_counts.get(2, class_counts.get('2', 0)) > 0, \
        "FAIL: zero lane-class points in training split"

    if len(intensity_all) == 0:
        print('FAIL: no intensity samples collected')
        print('script_status FAIL')
        sys.exit(1)

    p_low = float(np.percentile(intensity_all, 0.5))
    p_high = float(np.percentile(intensity_all, 99.5))
    clipped = np.clip(intensity_all, p_low, p_high)
    clip_mean = float(clipped.mean())
    clip_std = float(clipped.std())

    assert clip_std > 0, "FAIL: clip_std is zero — intensity has no variance after clipping"

    # Partial statistics JSON (weights added by next script)
    stats = {
        'provenance': {
            'split_file': str(TRAIN_SPLIT_FILE),
            'sequences_in_split': len(train_ids),
            'sequences_completed': len(completed),
            'frames_processed': n_proc,
            'frames_semseg_missing': n_missing,
            'skip_rate': skip_rate,
            'intensity_samples_total': len(intensity_all),
            'reservoir_size_per_seq': RESERVOIR_SIZE,
        },
        'intensity': {
            'clip_low_p0p5': p_low,
            'clip_high_p99p5': p_high,
            'clip_mean': clip_mean,
            'clip_std': clip_std,
        },
        'class_counts': {
            'ignore': int(class_counts.get(0, class_counts.get('0', 0))),
            'road': int(class_counts.get(1, class_counts.get('1', 0))),
            'lane': int(class_counts.get(2, class_counts.get('2', 0))),
            'other': int(class_counts.get(3, class_counts.get('3', 0))),
        },
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2))

    print(f'\nintensity_clip_low {p_low:.4f}')
    print(f'intensity_clip_high {p_high:.4f}')
    print(f'intensity_clip_mean {clip_mean:.4f}')
    print(f'intensity_clip_std {clip_std:.4f}')
    print(f'lane_points {stats["class_counts"]["lane"]}')
    print(f'statistics_complete True')
    print('script_status PASS')
    sys.exit(0)

if __name__ == '__main__':
    main()
```

**Run (this will be slow — allow 60–180 min for all training sequences):**

```bash
set -o pipefail
python tools/compute_training_statistics.py 2>&1 | tee logs/milestone_b_statistics_log.txt
```

**Pass condition:**

- `statistics_complete True`
- `lane_points > 0`
- `skip_rate < 0.05`
- `logs/milestone_b_training_statistics.json` written
- `script_status PASS`

**If skip_rate > 0.05:** Investigate semseg frame count mismatches. Do not proceed until resolved.

---

### Step 4.2 — Compute loss weight candidates

**Create** `tools/compute_loss_weights.py`:

```python
"""
Derive two class-weight candidates from measured class counts.
Uses the weight format discovered in logs/milestone_b_loss_interface.json.

Computes:
  1. Raw inverse-frequency normalized weights
  2. Sqrt-inverse-frequency normalized weights

Both are saved to logs/milestone_b_training_statistics.json.
Recommends the sqrt variant if raw lane weight exceeds 50.

Emits script_status PASS/FAIL as final line.
"""
import sys
import json
import math
from pathlib import Path

STATS_FILE = Path('logs/milestone_b_training_statistics.json')
INTERFACE_FILE = Path('logs/milestone_b_loss_interface.json')
LARGE_WEIGHT_THRESHOLD = 50.0


def build_weight_list(weights_by_class, interface):
    """
    Build the weight list in the format the local SemSegLoss expects.
    Reads weight_scope and weight_list_length from the discovered interface.

    interface['weight_scope']:
      'active-classes-only' -> list of length 3: [w_road, w_lane, w_other]
      'all-including-ignore' -> list of length 4: [0.0, w_road, w_lane, w_other]
      (adjust logic below if local source reveals other format)
    """
    w_road = weights_by_class['road']
    w_lane = weights_by_class['lane']
    w_other = weights_by_class['other']

    scope = interface.get('weight_scope', '')
    if 'active' in scope.lower() or '3' in str(interface.get('weight_list_length', '')):
        # Length 3: active classes only
        return [w_road, w_lane, w_other]
    elif 'ignore' in scope.lower() or '4' in str(interface.get('weight_list_length', '')):
        # Length 4: include ignore slot as 0.0
        return [0.0, w_road, w_lane, w_other]
    else:
        # Unknown format — produce both and flag for manual review
        print(f'WARNING: weight_scope "{scope}" is ambiguous. '
              f'Producing length-3 list. Verify against local source manually.')
        return [w_road, w_lane, w_other]


def main():
    stats = json.loads(STATS_FILE.read_text())
    interface = json.loads(INTERFACE_FILE.read_text())

    counts = stats['class_counts']
    c_road = counts['road']
    c_lane = counts['lane']
    c_other = counts['other']

    assert c_lane > 0, "FAIL: lane class has zero points — cannot derive weights"

    active_total = c_road + c_lane + c_other
    K = 3  # number of active classes

    # Raw inverse-frequency
    raw = {
        'road':  active_total / (K * c_road),
        'lane':  active_total / (K * c_lane),
        'other': active_total / (K * c_other),
    }
    w_min_raw = min(raw.values())
    raw_norm = {k: v / w_min_raw for k, v in raw.items()}

    # Sqrt-inverse-frequency (dampened)
    sqrt_raw = {
        'road':  math.sqrt(active_total / (K * c_road)),
        'lane':  math.sqrt(active_total / (K * c_lane)),
        'other': math.sqrt(active_total / (K * c_other)),
    }
    w_min_sqrt = min(sqrt_raw.values())
    sqrt_norm = {k: v / w_min_sqrt for k, v in sqrt_raw.items()}

    raw_lane_weight = raw_norm['lane']
    sqrt_lane_weight = sqrt_norm['lane']

    recommend_sqrt = raw_lane_weight > LARGE_WEIGHT_THRESHOLD

    raw_list = build_weight_list(raw_norm, interface)
    sqrt_list = build_weight_list(sqrt_norm, interface)

    sanity_recommended_variant = 'sqrt_inverse_frequency' if recommend_sqrt else 'raw_inverse_frequency'
    sanity_recommended_list = sqrt_list if recommend_sqrt else raw_list

    # Update statistics JSON
    stats['class_weights'] = {
        'raw_inverse_frequency': {
            'by_class': raw_norm,
            'as_list': raw_list,
            'lane_weight': raw_lane_weight,
        },
        'sqrt_inverse_frequency': {
            'by_class': sqrt_norm,
            'as_list': sqrt_list,
            'lane_weight': sqrt_lane_weight,
        },
        'weight_format_source': str(INTERFACE_FILE),
        'weight_scope_discovered': interface.get('weight_scope', 'UNKNOWN'),
        'weight_list_length_discovered': interface.get('weight_list_length', 'UNKNOWN'),
        'sanity_run_recommended_variant': sanity_recommended_variant,
        'sanity_run_recommended_list': sanity_recommended_list,
        'large_weight_warning': recommend_sqrt,
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2))

    print(f'raw_lane_weight {raw_lane_weight:.2f}')
    print(f'sqrt_lane_weight {sqrt_lane_weight:.2f}')
    print(f'large_weight_warning {recommend_sqrt}')
    print(f'recommended_sanity_variant {sanity_recommended_variant}')
    print(f'recommended_list {sanity_recommended_list}')
    if recommend_sqrt:
        print(f'NOTE: raw lane weight {raw_lane_weight:.1f} exceeds threshold {LARGE_WEIGHT_THRESHOLD}.')
        print(f'  Use sqrt-inverse-frequency weights for the sanity run.')
    print('script_status PASS')
    sys.exit(0)

if __name__ == '__main__':
    main()
```

**Run:**

```bash
set -o pipefail
python tools/compute_loss_weights.py 2>&1 | tee logs/milestone_b_weights_log.txt
```

---

### Step 4.3 — Post-grid point count measurement

```python
# Run inline or add to compute_training_statistics.py
# Measure post-grid point count on one representative frame to check num_points validity.
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve() / 'src'))

import open3d.ml.torch as ml3d
import numpy as np
from thesis_pipeline.adapters.pandaset_ff_lane3 import build_one_sample

sample = build_one_sample()  # verified reference sample
points = sample['point']  # (N, 3)

# Simulate grid subsampling at grid_size=0.04
from open3d.ml.torch.ops import voxelize
import torch

pts_t = torch.tensor(points, dtype=torch.float32)
# Use Open3D ML grid subsampling — check if it's available and how to call it
# The exact API depends on local source inspection. Record the result.
print(f'raw_point_count {len(points)}')
# After subsampling, record:
# print(f'post_grid_point_count <value>')
# print(f'below_num_points {post_grid_count < 16384}')
# print(f'duplication_will_occur {post_grid_count < 16384}')
```

The exact grid subsampling API call must be adapted based on what the local source inspection
in Day 3 reveals. Record the post-grid point count in `logs/milestone_b_statistics_log.txt` or
append to the statistics JSON under `'grid_subsampling'`.

**Key question to answer and record:**

- Post-grid count: `<N_post>`
- Will `num_points: 16384` cause point duplication? `yes/no`

## Day 4 Deliverables

- `tools/compute_training_statistics.py`
- `tools/compute_loss_weights.py`
- `logs/milestone_b_statistics_progress.jsonl`
- `logs/milestone_b_statistics_log.txt` (script_status PASS)
- `logs/milestone_b_weights_log.txt` (script_status PASS)
- `logs/stats_per_seq/{seq_id}.npy` (per training sequence)
- `logs/milestone_b_training_statistics.json` (with intensity stats, class counts, both weight candidates)
- Post-grid point count recorded

## Day 4 Pass Criteria

- Statistics computed over all training-split sequences
- Lane class has positive point count
- Skip rate < 5%
- Both raw and sqrt-inverse-frequency weight candidates computed and saved
- `recommended_sanity_variant` is set
- Post-grid point count recorded
- `script_status PASS` on both scripts

---

# Day 5 — Phase 4: Dataset Class and Config Update

## Objective

Implement a proper Open3D-ML-compatible dataset class. Update the YAML config with measured
values. Verify the dataset class and confirm YAML/JSON consistency.

## Why Day 5

The dataset class implementation depends on the discovered interface (Day 3) and measured
statistics (Day 4). It cannot be written before both are complete. This day bridges the
measurement phase to the pipeline assembly phase.

---

### Step 5.1 — Implement the dataset class

**Architecture:** Core logic in `src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py`.
Thin guide-facing wrapper in `datasets/pandaset_ff_lane3.py` (preserves `build_one_sample()` for
backward compatibility).

**Implementation requirements** (all derived from the discovered interface JSON):

1. Inherit from the discovered base class (`milestone_b_dataset_interface.json::base_class`)
2. Implement all required methods (`required_methods`)
3. `get_data(idx)` returns a dict with key names from `get_data_return_keys`
4. Handle xyz+feat concatenation **at the location specified in `concat_location`**:
   - If `concat_location == "dataset_class"`: concatenate before returning
   - If `concat_location == "pipeline"`: return point and feat separately
   - If `concat_location == "other"`: follow the discovered behavior
5. Apply intensity preprocessing using stats from `logs/milestone_b_training_statistics.json` — **no hardcoded values**
6. Apply the remap by importing `remap_raw_pandaset_ids` from `src/thesis_pipeline/adapters/pandaset_ff_lane3`
7. Apply ego-local conversion using `geometry.lidar_points_to_ego`
8. Support all three splits using the split files in `configs/splits/`
9. Maintain a deterministic frame index table: for each sample index, record `(split, seq_id, frame_idx)` without loading all frames upfront
10. Respect the sensor access pattern from `logs/milestone_b_preflight_sensor_pattern.txt`

**Create** `src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py`:

```python
"""
Core dataset class for PandaSet forward-facing LiDAR lane-line marking segmentation.
Implements the Open3D-ML dataset contract based on locally discovered interface.

IMPORTANT: This file's class structure is a concrete template with the fixed parts
filled in. The [FILL_FROM_DISCOVERY] markers below represent the ONLY parts that
require local source inspection to complete. Each marker specifies exactly what
information it needs and where to find it.
"""
import json
import numpy as np
from pathlib import Path
from pandaset import DataSet, geometry as pds_geometry

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from thesis_pipeline.adapters.pandaset_ff_lane3 import remap_raw_pandaset_ids
from thesis_pipeline.core.pandaset_compat import get_frame_count

# [FILL_FROM_DISCOVERY: Import the correct base class]
# Read logs/milestone_b_dataset_interface.json :: base_class
# Example: from open3d.ml.torch.datasets import BaseDataset
# Replace the next line with the discovered import:
# from <discovered_module> import <DiscoveredBaseClass>

STATS_FILE = Path('logs/milestone_b_training_statistics.json')
SPLIT_DIR = Path('configs/splits')


def _load_stats():
    return json.loads(STATS_FILE.read_text())


def _get_sensor_reapply():
    p = Path('logs/milestone_b_preflight_sensor_pattern.txt')
    return p.exists() and 'reapply_per_frame: true' in p.read_text()


class PandaSetFFLane3Dataset:  # [FILL_FROM_DISCOVERY: replace with (DiscoveredBaseClass)]
    """
    Open3D-ML compatible dataset class for PandaSet forward-facing LiDAR
    three-class lane-line marking segmentation.
    """

    def __init__(self, dataset_path, cfg=None, **kwargs):
        # [FILL_FROM_DISCOVERY: call super().__init__ with the args the base class expects]
        # Read logs/milestone_b_dataset_interface.json :: required_cfg_fields
        # super().__init__(dataset_path=dataset_path, cfg=cfg, **kwargs)

        self.dataset_path = dataset_path
        self.cfg = cfg or {}
        self._stats = _load_stats()
        self._sensor_reapply = _get_sensor_reapply()
        self._ds = DataSet(dataset_path)

        self.intensity_clip_low = self._stats['intensity']['clip_low_p0p5']
        self.intensity_clip_high = self._stats['intensity']['clip_high_p99p5']
        self.intensity_mean = self._stats['intensity']['clip_mean']
        self.intensity_std = self._stats['intensity']['clip_std']

        self._split_ids = {
            'train': self._load_split('train'),
            'val':   self._load_split('val'),
            'test':  self._load_split('test'),
        }
        # Build deterministic frame index table (no data loaded)
        self._index_table = self._build_index_table()

    def _load_split(self, split_name):
        return [
            l.strip()
            for l in (SPLIT_DIR / f'{split_name}.txt').read_text().splitlines()
            if l.strip()
        ]

    def _build_index_table(self):
        """
        Map each global sample index to (split_name, seq_id, frame_idx).
        Uses get_frame_count to avoid loading all frame data.
        """
        import gc
        table = []
        for split_name, seq_ids in self._split_ids.items():
            for seq_id in seq_ids:
                seq = self._ds[seq_id]
                try:
                    seq.load_lidar()
                except Exception:
                    pass
                n = get_frame_count(seq)
                for fi in range(n):
                    table.append((split_name, seq_id, fi))
                del seq
                gc.collect()
        return table

    def __len__(self):
        return len(self._index_table)

    def get_data(self, idx):
        """
        Returns one sample dict.
        Key names are per logs/milestone_b_dataset_interface.json :: get_data_return_keys.
        """
        split_name, seq_id, frame_idx = self._index_table[idx]
        seq = self._ds[seq_id]
        try:
            seq.load_lidar().load_semseg()
        except Exception:
            seq.load_lidar()
            seq.load_semseg()

        if not self._sensor_reapply:
            seq.lidar.set_sensor(1)
        else:
            seq.lidar.set_sensor(1)

        if self._sensor_reapply:
            seq.lidar.set_sensor(1)
        pc_df = seq.lidar[frame_idx]
        semseg_df = seq.semseg[frame_idx]

        raw_labels = semseg_df.loc[pc_df.index, 'class'].to_numpy(dtype='int32')
        label = remap_raw_pandaset_ids(raw_labels)

        xyz_world = pc_df[['x', 'y', 'z']].to_numpy(dtype='float32')
        pose = seq.lidar.poses[frame_idx]
        xyz_ego = pds_geometry.lidar_points_to_ego(xyz_world, pose).astype(np.float32)

        intensity = pc_df['i'].to_numpy(dtype='float32')
        intensity = np.clip(intensity, self.intensity_clip_low, self.intensity_clip_high)
        intensity = (intensity - self.intensity_mean) / self.intensity_std
        feat = intensity[:, None]  # (N, 1)

        # [FILL_FROM_DISCOVERY: handle concat_location]
        # Read logs/milestone_b_dataset_interface.json :: concat_location
        # If concat_location == "dataset_class":
        #   data = np.concatenate([xyz_ego, feat], axis=1)  # (N, 4)
        #   return {points_key: data, labels_key: label}
        # If concat_location == "pipeline":
        #   return {points_key: xyz_ego, features_key: feat, labels_key: label}
        # Use the exact key names from get_data_return_keys in the interface JSON.

        # Placeholder — replace with discovered behavior:
        points_key = 'point'    # [FILL_FROM_DISCOVERY: use discovered key name]
        features_key = 'feat'   # [FILL_FROM_DISCOVERY: use discovered key name]
        labels_key = 'label'    # [FILL_FROM_DISCOVERY: use discovered key name]

        return {
            points_key: xyz_ego,
            features_key: feat,
            labels_key: label,
        }

    def get_split(self, split):
        # [FILL_FROM_DISCOVERY: implement how the pipeline requests splits]
        # Read logs/milestone_b_dataset_interface.json :: split_object_api
        # This method may need to return a split object of a specific type,
        # or may not be needed if the pipeline uses a different mechanism.
        pass

    # [FILL_FROM_DISCOVERY: add any other required abstract methods]
    # Read logs/milestone_b_dataset_interface.json :: required_methods
    # Implement each one found there that is not already above.
```

**Important:** After writing this class, go through every `[FILL_FROM_DISCOVERY]` marker and
replace it with the discovered value from `logs/milestone_b_dataset_interface.json`. The class
must have no remaining placeholder markers before it is used in any script. This is a hard
requirement — a class with placeholders is not a deliverable.

**Then update** `datasets/pandaset_ff_lane3.py` to import the new class and preserve backward
compatibility:

```python
"""
Guide-facing thin wrapper. Imports core logic from src/thesis_pipeline/datasets/.
Preserves build_one_sample() for backward compatibility.
"""
from src.thesis_pipeline.datasets.pandaset_ff_lane3_dataset import PandaSetFFLane3Dataset

# Backward-compatible single-frame interface (from Days 1-4)
from src.thesis_pipeline.adapters.pandaset_ff_lane3 import build_one_sample  # noqa: F401
```

---

### Step 5.2 — Update YAML config with measured values

Update `configs/randlanet_pandaset_ff_lane3.yml` to include:

```yaml
# --- Measured values from logs/milestone_b_training_statistics.json ---
# Replace <VALUE> with actual values from the statistics JSON.
# Do not copy values from memory or planning documents — read from the JSON.

dataset:
  dataset_path: /home/pheikara/University/Y3S2/Thesis/Pipeline/pandaset/PandaSet
  class_weights: <sanity_run_recommended_list from statistics JSON>
  intensity_clip_low: <clip_low_p0p5>
  intensity_clip_high: <clip_high_p99p5>
  intensity_mean: <clip_mean>
  intensity_std: <clip_std>

model:
  name: RandLANet
  num_classes: 3
  in_channels: 4
  num_layers: 3
  num_neighbors: 16
  sub_sampling_ratio: [4, 4, 4]
  grid_size: 0.04
  num_points: 16384

pipeline:
  name: SemanticSegmentation
  # [FILL_FROM_DISCOVERY: add fields required by the pipeline per milestone_b_pipeline_interface.json]
  max_epoch: 2 # sanity check only; increase for full training
  batch_size: 2 # sanity check value; verify GPU memory allows this
  val_batch_size: 1
  test_batch_size: 1
  ignored_label_inds: [0]
  device: cuda
  real_training_allowed: false # Must be set to true explicitly before full training
```

---

### Step 5.3 — Verify dataset class

**Create** `tools/check_dataset_class.py`:

```python
"""
Verify the dataset class satisfies the Open3D-ML interface.
Must have no [FILL_FROM_DISCOVERY] markers remaining before running.

Checks:
  - All three splits instantiate
  - get_data() produces correct shapes, dtypes, label range for first 3 samples
  - YAML class_weights matches statistics JSON (programmatic drift check)
  - build_one_sample() lane fraction ≈ 0.0026 (backward compat)

Emits script_status PASS/FAIL as final line.
"""
import sys
import json
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# [FILL_FROM_DISCOVERY: import the dataset class after all markers are resolved]
# from datasets.pandaset_ff_lane3 import PandaSetFFLane3Dataset
from datasets.pandaset_ff_lane3 import build_one_sample

STATS_FILE = Path('logs/milestone_b_training_statistics.json')
CFG_FILE = Path('configs/randlanet_pandaset_ff_lane3.yml')
REPORT_FILE = Path('logs/milestone_b_dataset_class_report.txt')

EXPECTED_LANE_FRACTION = 0.0026
LANE_FRACTION_TOLERANCE = 0.10  # within 10% of reference

def main():
    lines = []
    all_ok = True

    # 1. YAML/JSON weight consistency
    stats = json.loads(STATS_FILE.read_text())
    cfg = yaml.safe_load(CFG_FILE.read_text())
    json_weights = stats['class_weights']['sanity_run_recommended_list']
    yaml_weights = cfg.get('dataset', {}).get('class_weights', [])
    if list(yaml_weights) == list(json_weights):
        lines.append('yaml_json_weights_consistent True')
    else:
        lines.append(f'FAIL yaml_json_weights_consistent False')
        lines.append(f'  yaml: {yaml_weights}')
        lines.append(f'  json: {json_weights}')
        all_ok = False

    # 2. Backward compatibility check
    sample = build_one_sample()
    label = sample['label']
    lane_fraction = float((label == 2).sum()) / len(label)
    diff = abs(lane_fraction - EXPECTED_LANE_FRACTION) / EXPECTED_LANE_FRACTION
    compat_ok = diff <= LANE_FRACTION_TOLERANCE
    lines.append(f'backward_compat_lane_fraction {lane_fraction:.6f}')
    lines.append(f'backward_compat_ok {compat_ok}')
    if not compat_ok:
        lines.append(f'FAIL: lane fraction {lane_fraction:.6f} differs from reference '
                     f'{EXPECTED_LANE_FRACTION} by {diff*100:.1f}% (threshold {LANE_FRACTION_TOLERANCE*100:.0f}%)')
        all_ok = False

    # 3. Dataset class instantiation and sample checks
    # [FILL_FROM_DISCOVERY: after implementing the dataset class with all markers resolved]
    # dataset_path = Path('logs/dataset_root.txt').read_text().strip()
    # for split_name in ['train', 'val', 'test']:
    #     dataset = PandaSetFFLane3Dataset(dataset_path=dataset_path)
    #     # get split object per discovered split_object_api
    #     split_obj = ...
    #     for i in range(min(3, len(split_obj))):
    #         data = split_obj.get_data(i)
    #         # Check shapes, dtypes, label range
    #         assert data['point'].shape[1] in [3, 4], f"Unexpected point shape: {data['point'].shape}"
    #         assert data['label'].dtype in [np.int32, np.int64], "label wrong dtype"
    #         assert set(np.unique(data['label'])).issubset({0,1,2,3}), "label out of range"
    #     lines.append(f'split_{split_name}_ok True')

    REPORT_FILE.write_text('\n'.join(lines))
    for line in lines:
        print(line)

    status = 'PASS' if all_ok else 'FAIL'
    print(f'script_status {status}')
    sys.exit(0 if all_ok else 1)

if __name__ == '__main__':
    main()
```

**Before running:** Resolve all `[FILL_FROM_DISCOVERY]` markers in both the dataset class and
this verification script. A script with placeholder markers is not a deliverable.

**Run:**

```bash
set -o pipefail
python tools/check_dataset_class.py 2>&1 | tee logs/milestone_b_dataset_class_report.txt
```

**Pass condition:**

- `yaml_json_weights_consistent True`
- `backward_compat_ok True`
- All three split instantiations OK
- `script_status PASS`

## Day 5 Deliverables

- `src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py` (all `[FILL_FROM_DISCOVERY]` resolved)
- `datasets/pandaset_ff_lane3.py` (updated thin wrapper; `build_one_sample()` preserved)
- `configs/randlanet_pandaset_ff_lane3.yml` (measured values filled in)
- `tools/check_dataset_class.py`
- `logs/milestone_b_dataset_class_report.txt` (script_status PASS)

## Day 5 Pass Criteria

- Dataset class has no remaining placeholder markers
- YAML/JSON weight consistency confirmed programmatically
- Backward compatibility confirmed (lane fraction within 10% of `0.0026`)
- All three split objects instantiate and return correct sample shapes
- `script_status PASS`

---

# Day 6 — Phase 5: Pipeline Sanity Check and Stop Conditions

## Objective

Assemble and run the full pipeline for a few training iterations. Confirm finite, non-exploding
loss. Write stop conditions before any full training begins.

## Why Day 6

This is the milestone's final and defining step. All prior work (split, statistics, interface
discovery, dataset class, config) feeds into this test. A passing sanity run with non-diverging
loss is the proof of mechanical completeness.

---

### Step 6.1 — Implement the sanity training script

**Before writing this script:** Read `logs/milestone_b_pipeline_interface.json` fully.
Every `[FILL_FROM_DISCOVERY]` marker in this script depends on the discovered pipeline API.
All markers must be resolved before the script is considered a deliverable.

**Create** `tools/sanity_train_check.py`:

```python
"""
Sanity training check: assemble the full Open3D-ML pipeline and run a few iterations.

Pass conditions (all must hold):
  1. Pipeline instantiates without error
  2. At least 5 iterations execute
  3. All loss values are finite (not NaN, not Inf)
  4. Loss changes between first and last iteration
  5. Final loss does not exceed 10x the initial loss

If condition 5 fails: retry with the sqrt-inverse-frequency weights.
If still failing after retry: record INSTABILITY in stop conditions and proceed.

Emits script_status PASS/FAIL as final line.
"""
import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import open3d.ml.torch as ml3d

STATS_FILE = Path('logs/milestone_b_training_statistics.json')
CFG_FILE = Path('configs/randlanet_pandaset_ff_lane3.yml')
REPORT_FILE = Path('logs/milestone_b_sanity_train_report.txt')
DIVERGENCE_THRESHOLD = 10.0

# [FILL_FROM_DISCOVERY: import the pipeline class]
# Read logs/milestone_b_pipeline_interface.json :: pipeline_class
# Example: from open3d.ml.torch.pipelines import SemanticSegmentation

# [FILL_FROM_DISCOVERY: import the dataset class]
# from datasets.pandaset_ff_lane3 import PandaSetFFLane3Dataset


def run_sanity(weight_variant):
    """
    Attempt to run the sanity check with the given weight variant.
    Returns (loss_values, error_message_or_None).
    """
    stats = json.loads(STATS_FILE.read_text())
    weights = stats['class_weights'][weight_variant]['as_list']
    dataset_path = Path('logs/dataset_root.txt').read_text().strip()

    import yaml
    cfg = yaml.safe_load(CFG_FILE.read_text())
    cfg['dataset']['class_weights'] = weights

    # [FILL_FROM_DISCOVERY: instantiate dataset]
    # dataset = PandaSetFFLane3Dataset(dataset_path=dataset_path, cfg=cfg.get('dataset', {}))

    # [FILL_FROM_DISCOVERY: instantiate model]
    model = ml3d.models.RandLANet(**cfg['model'])

    # [FILL_FROM_DISCOVERY: instantiate pipeline]
    # Read logs/milestone_b_pipeline_interface.json :: constructor_required_args
    # Example: pipeline = SemanticSegmentation(model=model, dataset=dataset, **cfg['pipeline'])

    # [FILL_FROM_DISCOVERY: run training for a small number of iterations]
    # Read logs/milestone_b_pipeline_interface.json :: training_method, iteration_limit_mechanism
    # The sanity run must:
    #   - Execute at least 5 iterations
    #   - Capture loss values per iteration
    #   - Stop after max_epoch=1 or after 5 steps, whichever the API supports
    # Example: pipeline.run_train()  # with max_epoch=1 in config
    # Or: pipeline.<training_method>()

    loss_values = []  # [FILL_FROM_DISCOVERY: populate with actual loss values from training]
    return loss_values, None


def check_losses(loss_values):
    if len(loss_values) < 5:
        return False, f'Only {len(loss_values)} iterations completed (need >= 5)'
    if any(not math.isfinite(v) for v in loss_values):
        return False, f'Non-finite loss found: {loss_values}'
    if loss_values[-1] == loss_values[0]:
        return False, f'Loss did not change: first={loss_values[0]}, last={loss_values[-1]}'
    ratio = loss_values[-1] / (loss_values[0] + 1e-9)
    if ratio > DIVERGENCE_THRESHOLD:
        return False, f'Loss diverged: ratio={ratio:.2f} (threshold={DIVERGENCE_THRESHOLD})'
    return True, None


def main():
    lines = []
    stats = json.loads(STATS_FILE.read_text())
    recommended_variant = stats['class_weights'].get(
        'sanity_run_recommended_variant', 'raw_inverse_frequency')

    print(f'recommended_weight_variant {recommended_variant}')

    # First attempt: recommended variant
    loss_values, err = run_sanity(recommended_variant)

    if err:
        lines.append(f'attempt_1_error {err}')
        print(f'attempt_1_error {err}')
        print('script_status FAIL')
        sys.exit(1)

    ok, msg = check_losses(loss_values)
    lines.append(f'attempt_1_variant {recommended_variant}')
    lines.append(f'attempt_1_loss_values {loss_values}')
    lines.append(f'attempt_1_ok {ok}')

    if not ok:
        print(f'attempt_1_failed: {msg}')
        # Try alternative variant
        alt_variant = ('sqrt_inverse_frequency' if 'raw' in recommended_variant
                       else 'raw_inverse_frequency')
        print(f'retrying_with_variant {alt_variant}')
        loss_values_2, err2 = run_sanity(alt_variant)

        if err2:
            lines.append(f'attempt_2_error {err2}')
            print(f'attempt_2_error {err2}')
            print('script_status FAIL')
            REPORT_FILE.write_text('\n'.join(lines))
            sys.exit(1)

        ok2, msg2 = check_losses(loss_values_2)
        lines.append(f'attempt_2_variant {alt_variant}')
        lines.append(f'attempt_2_loss_values {loss_values_2}')
        lines.append(f'attempt_2_ok {ok2}')

        if not ok2:
            lines.append(f'sanity_result INSTABILITY')
            lines.append(f'instability_note: Both weight variants produced diverging loss.')
            lines.append(f'  Record this as INSTABILITY in stop conditions.')
            lines.append(f'  Do not block Milestone B completion, but flag for next milestone.')
            lines.append('script_status FAIL')
            REPORT_FILE.write_text('\n'.join(lines))
            print('\n'.join(lines[-6:]))
            sys.exit(1)

        lines.append(f'sanity_weight_variant_used {alt_variant}')
        used_variant = alt_variant
        used_losses = loss_values_2
    else:
        lines.append(f'sanity_weight_variant_used {recommended_variant}')
        used_variant = recommended_variant
        used_losses = loss_values

    ratio = used_losses[-1] / (used_losses[0] + 1e-9)
    lines.append(f'loss_first {used_losses[0]:.6f}')
    lines.append(f'loss_last {used_losses[-1]:.6f}')
    lines.append(f'loss_ratio_final_to_initial {ratio:.4f}')
    lines.append(f'sanity_train_check_ok True')
    lines.append('script_status PASS')

    REPORT_FILE.write_text('\n'.join(lines))
    for line in lines:
        print(line)
    sys.exit(0)

if __name__ == '__main__':
    main()
```

**Before running:** Every `[FILL_FROM_DISCOVERY]` marker must be resolved using the pipeline
interface JSON. The script must run without any placeholder remaining.

---

### Step 6.2 — Resolve all `[FILL_FROM_DISCOVERY]` markers

Using `logs/milestone_b_pipeline_interface.json`:

1. Replace the pipeline and dataset import placeholders with actual discovered imports
2. Replace the dataset instantiation placeholder with the correct constructor call
3. Replace the pipeline instantiation placeholder with the correct constructor
4. Replace the training invocation placeholder with the discovered training method
5. Populate `loss_values` by hooking into the training loop loss outputs

**After resolving:** The script must pass a dry import check with no placeholder text:

```bash
python -c "
import ast, sys
src = open('tools/sanity_train_check.py').read()
assert '[FILL_FROM_DISCOVERY' not in src, 'Placeholder markers remain — not a deliverable'
ast.parse(src)
print('no_placeholders_remain True')
"
```

---

### Step 6.3 — Run the sanity check

```bash
set -o pipefail
python tools/sanity_train_check.py 2>&1 | tee logs/milestone_b_sanity_train_report.txt
```

Then save the exact config used:

```bash
cp configs/randlanet_pandaset_ff_lane3.yml logs/milestone_b_sanity_config_snapshot.yml
```

**Pass condition (all must hold):**

- Pipeline instantiates
- ≥5 iterations execute
- All loss values are finite
- Loss changes between first and last iteration
- Final loss ≤ 10× initial loss
- `sanity_train_check_ok True`
- `script_status PASS`

**If GPU OOM at `num_points: 16384`:**
Temporarily reduce `num_points` in the config (e.g., to `4096` or `8192`). Document this as
a **sanity-check-only concession** in the stop conditions. The intended value of `16384` must
be restored before full training. This is the only permitted temporary change to scaffold values.

**If loss diverges (ratio > 10):**
The script automatically retries with the alternative weight variant. If both fail, the script
records `INSTABILITY`. This does not necessarily block Milestone B completion, but it must be
recorded explicitly and investigated at the start of the next milestone.

---

### Step 6.4 — Write stop conditions

**Create** `logs/milestone_b_stop_conditions.txt` with these required fields
(fill in measured values):

```
milestone_b_stop_conditions
===========================

pipeline_sanity_check_passed: [yes / no / INSTABILITY]
semseg_pool_discovered: yes
split_frozen: yes
intensity_statistics_measured: yes
class_weights_measured: yes
dataset_class_tested: yes
config_updated_with_measured_values: yes

no_full_training_has_occurred: yes
no_performance_claim_has_been_made: yes
real_training_allowed_in_config: false

sanity_weight_variant_used: [raw_inverse_frequency / sqrt_inverse_frequency]
loss_ratio_final_to_initial: [measured value]
sanity_num_points_used: [value used during sanity run]
num_points_baseline_intent: 16384
num_points_concession_note: [empty if no concession; else describe the OOM and reduction]

lane_fraction_measured: [global p_lane value from class counts]
expected_deep_anchor_count: [num_points * p_lane / 64 — global approximation only]
oversampling_indicated: [yes if E < 2.0 / no]
oversampling_note: [if yes: must implement before or explicitly reject with evidence in next milestone]

post_grid_point_count: [measured value]
grid_duplication_will_occur: [yes / no]

concat_location_discovered: [pipeline / dataset_class / model]
weight_scope_discovered: [active-classes-only / all-including-ignore]
sampler_available: [yes / no]

frame_skip_rate_statistics_scan: [measured skip rate]
sequences_in_training_split: [count]
semseg_pool_size: [count]

if_instability_recorded:
  [describe both weight variants tried and resulting loss ratios]
  [note: investigate at start of next milestone]

ready_for_full_training_milestone: yes
```

## Day 6 Deliverables

- `tools/sanity_train_check.py` (all `[FILL_FROM_DISCOVERY]` markers resolved)
- `logs/milestone_b_sanity_train_report.txt` (script_status PASS or INSTABILITY noted)
- `logs/milestone_b_sanity_config_snapshot.yml`
- `logs/milestone_b_stop_conditions.txt` (all required fields populated)

## Day 6 Pass Criteria

- Sanity training script has no remaining placeholder markers (verified by import check)
- Sanity run executed with ≥5 iterations
- Loss is finite and changed across iterations
- Final loss ≤ 10× initial loss (or INSTABILITY recorded if not achievable)
- Stop conditions written with all required fields
- No full training has occurred
- `script_status PASS` (or explicitly noted INSTABILITY)

---

# Complete Artifact List

## Scripts

| Script                                 | Day   | Status                         |
| -------------------------------------- | ----- | ------------------------------ |
| `tools/preflight_multi_frame_check.py` | Day 1 | New                            |
| `tools/audit_semseg_sequences.py`      | Day 1 | New                            |
| `tools/freeze_split.py`                | Day 2 | New                            |
| `tools/compute_training_statistics.py` | Day 4 | New                            |
| `tools/compute_loss_weights.py`        | Day 4 | New                            |
| `tools/check_dataset_class.py`         | Day 5 | New                            |
| `tools/sanity_train_check.py`          | Day 6 | New; all placeholders resolved |

## Source Modules

| Module                                                      | Day   | Status                                                 |
| ----------------------------------------------------------- | ----- | ------------------------------------------------------ |
| `src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py` | Day 5 | New; all placeholders resolved                         |
| `datasets/pandaset_ff_lane3.py`                             | Day 5 | Updated (thin wrapper; `build_one_sample()` preserved) |

## Split Files

| File                       | Day   |
| -------------------------- | ----- |
| `configs/splits/train.txt` | Day 2 |
| `configs/splits/val.txt`   | Day 2 |
| `configs/splits/test.txt`  | Day 2 |

## Updated Config

| File                                      | Day                            |
| ----------------------------------------- | ------------------------------ |
| `configs/randlanet_pandaset_ff_lane3.yml` | Day 5 (measured values filled) |

## Log and Data Artifacts

| File                                            | Day               |
| ----------------------------------------------- | ----------------- |
| `logs/milestone_b_preflight.txt`                | Day 1             |
| `logs/milestone_b_preflight_sensor_pattern.txt` | Day 1 (if needed) |
| `logs/milestone_b_audit_progress.jsonl`         | Day 1             |
| `logs/milestone_b_sequence_manifest.json`       | Day 1             |
| `logs/milestone_b_sequence_audit.txt`           | Day 1             |
| `logs/milestone_b_split_report.txt`             | Day 2             |
| `logs/milestone_b_source_locations.txt`         | Day 3             |
| `logs/milestone_b_loss_interface.json`          | Day 3             |
| `logs/milestone_b_dataset_interface.json`       | Day 3             |
| `logs/milestone_b_pipeline_interface.json`      | Day 3             |
| `logs/milestone_b_open3d_interface_notes.txt`   | Day 3             |
| `logs/stats_per_seq/{seq_id}.npy`               | Day 4             |
| `logs/milestone_b_statistics_progress.jsonl`    | Day 4             |
| `logs/milestone_b_statistics_log.txt`           | Day 4             |
| `logs/milestone_b_weights_log.txt`              | Day 4             |
| `logs/milestone_b_training_statistics.json`     | Day 4             |
| `logs/milestone_b_dataset_class_report.txt`     | Day 5             |
| `logs/milestone_b_sanity_train_report.txt`      | Day 6             |
| `logs/milestone_b_sanity_config_snapshot.yml`   | Day 6             |
| `logs/milestone_b_stop_conditions.txt`          | Day 6             |

---

# Conditions That Must Be True Before Work Moves Beyond Milestone B

1. `set_sensor(1)` persistence confirmed across ≥5 consecutive frames — `milestone_b_preflight.txt` shows `script_status PASS`
2. Semseg alignment confirmed across those frames
3. Semseg-enabled pool discovered — `milestone_b_sequence_manifest.json` written
4. Lane presence checked per sequence with all-frames coverage
5. Pool size ≥70 confirmed
6. Split frozen with no overlaps — `milestone_b_split_report.txt` shows `split_status FROZEN`
7. Val and test each contain ≥1 lane-bearing sequence — Python assertion passed and logged
8. Split provenance fully recorded (seed, manifest hash, lane allocation)
9. `logs/milestone_b_loss_interface.json` populated with no placeholder values
10. `logs/milestone_b_dataset_interface.json` populated including `concat_location`
11. `logs/milestone_b_pipeline_interface.json` populated
12. Intensity statistics (P0.5, P99.5, clip mean, clip std) measured on training split
13. Class counts measured on training split; lane count is positive
14. Frame skip rate during statistics scan < 5%
15. Both raw and sqrt-inverse-frequency weight candidates computed — `milestone_b_training_statistics.json` contains both
16. `sanity_run_recommended_variant` set in statistics JSON
17. Dataset class implemented — `src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py` has no placeholder markers
18. `datasets/pandaset_ff_lane3.py` updated; `build_one_sample()` backward compatibility verified
19. YAML config updated with measured values consistent with statistics JSON — programmatically confirmed by `check_dataset_class.py`
20. Post-grid point count measured and recorded
21. Sanity training script has no remaining placeholder markers — verified by dry import check
22. Sanity run executed: loss is finite; loss changed; final loss ≤ 10× initial loss (or INSTABILITY explicitly recorded)
23. Stop conditions written with all required fields
24. No full training has occurred
25. No performance claim has been made
26. `real_training_allowed` remains `false` in config unless explicitly changed for the next milestone

---

# What the Next Milestone After Milestone B Must Address

Milestone B deliberately leaves these unresolved:

- full baseline training to convergence
- lane-aware patch oversampling (if indicated by `oversampling_indicated: yes` in stop conditions)
- final class-weight transform (both candidates recorded; policy not finalized)
- training hyperparameter selection (epoch budget, learning rate, batch size)
- loss convergence monitoring
- initial metric evaluation on the validation split
- qualitative inspection of predictions
- any comparison between intensity and xyz-only baselines
- any claim about whether the model learns the lane class
- investigation of INSTABILITY if recorded in stop conditions
- restoration of `num_points: 16384` if reduced during sanity check
