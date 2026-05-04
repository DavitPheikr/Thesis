# Implementation Pipeline

## First 4 Days Execution Guide

## Assumptions carried from the global documents

The following assumptions are carried from the revised global documents as **project-level planning truth** for the first 4 days:

- the project task is **point-wise semantic segmentation** on PandaSet LiDAR
- the active semantic classes are **road**, **lane**, and **other**
- the positive class is intended to mean **Lane Line Marking only**
- the intended baseline sensor choice is **forward-facing LiDAR only**
- the intended baseline coordinate frame is **ego-local**
- the intended minimal feature concept is **ego-local `xyz` plus one processed intensity channel**
- the intended model family is **RandLA-Net**
- the intended framework family is **Open3D-ML**
- the first 4 days stop before real training
- the first 4 days are supposed to end with a technically credible build path, not with performance results

These assumptions define **what the project is trying to build**. They do **not** prove that the local machine, local packages, local PandaSet installation, or local APIs already support that build path.

## Which of those assumptions must be treated as unverified on the local system

The following inherited assumptions must be treated as **unverified until tested locally**:

- any exact install line beyond the currently working local environment
- any exact package reinstall path if the local environment needs repair
- any assumption that a future reinstall will behave exactly like the current working environment
- any exact helper-function availability inside the local PandaSet devkit unless that helper has been tested locally in the relevant script
- any exact Open3D model import path unless that import has been tested locally in the relevant script
- any exact constructor keyword set unless the local runtime accepts it
- any exact dataset-root assumption unless it comes from the canonical local root file
- any exact sequence, frame, or sensor assumption until the local precheck writes them down and confirms they are usable
- any exact dataframe column-name assumption until the local frame inspection step confirms it
- any exact measured count, statistic, or safe configuration value that depends on local data measurement

The first 4 days are therefore a sequence of **local verification steps** that either confirm or reject those inherited assumptions one by one.

# Day 1

## Goal

Confirm the already-working local environment, record the exact current runtime state, and verify that the core project libraries import correctly inside the intended project virtual environment.

## Milestone by end of Day 1

By the end of Day 1, the local system must be able to import:
- Python from the intended project virtual environment
- PyTorch
- Open3D
- `open3d.ml.torch`
- PandaSet devkit objects

If that does not work, the project does not move to Day 2.

### Step 1

Action:
Activate the existing project virtual environment from the confirmed project root.

Why this step exists:
The environment is already working locally. Day 1 must verify and reuse that exact environment rather than silently creating a different one.

Command or file:
```bash id="s975a1"
cd /home/pheikara/University/Y3S2/Thesis/Pipeline
source panda/bin/activate
which python
python --version
```

Expected pass result:
- the working directory is the project root
- `which python` points inside `/home/pheikara/University/Y3S2/Thesis/Pipeline/panda`
- `python --version` prints `Python 3.12.3`

If it fails, that means:
- the current shell is not using the intended project environment
- or the `panda` virtual environment is missing or broken

Next fix to try:
Re-activate `panda` from the project root. If activation still fails, stop and repair the local environment before continuing.

### Step 2

Action:
Create or refresh the environment report file with the current local runtime and path facts.

Why this step exists:
The parent documents no longer need hypothetical environment discovery for this machine. They need a current local record that future agent runs can trust.

Command or file:
Create or overwrite `logs/day1_environment_report.txt` by running:
```bash id="dbyxur"
mkdir -p logs tools
python - <<'PY' > logs/day1_environment_report.txt
import platform, sys
from pathlib import Path
print('project_root', Path.cwd())
print('python_executable', sys.executable)
print('python_version', sys.version)
print('platform', platform.platform())
print('machine', platform.machine())
print('system', platform.system())
PY
cat logs/day1_environment_report.txt
```

Expected pass result:
- the file `logs/day1_environment_report.txt` exists
- it contains project root, Python executable, Python version, OS/platform, machine architecture, and system name

If it fails, that means:
The Python environment is not functioning cleanly even before import checks.

Next fix to try:
Re-activate `panda` and rerun the step. If it still fails, stop and repair the environment before continuing.

### Step 3

Action:
Record the canonical local PandaSet root path in one place and use that file as the only dataset-root source for later scripts.

Why this step exists:
The guide must not require repeated manual path replacement. All later scripts must read the same canonical root.

Command or file:
Create or overwrite `logs/dataset_root.txt` with exactly this content:
```text
/home/pheikara/University/Y3S2/Thesis/Pipeline/pandaset/PandaSet
```

Then verify it with:
```bash id="9nvmmr"
python - <<'PY'
from pathlib import Path
root = Path('logs/dataset_root.txt').read_text().strip()
print('dataset_root', root)
print('exists', Path(root).exists())
PY
```

Expected pass result:
- `logs/dataset_root.txt` exists
- the printed dataset root matches the confirmed local dataset root
- `exists True`

If it fails, that means:
The canonical dataset-root source is missing or points to the wrong place.

Next fix to try:
Rewrite `logs/dataset_root.txt` with the correct absolute path before continuing.

### Step 4

Action:
Create and run an explicit version-verification script against the currently active environment.

Why this step exists:
The environment is already working locally, but the exact package/runtime state still needs to be recorded once in a stable script that agents can rerun later.

Command or file:
Create `tools/check_versions.py` with this exact content:

```python id="cul37z"
import sys
import torch
import open3d as o3d
import numpy as np
import pandas as pd
import open3d.ml.torch as ml3d

print('python', sys.version)
print('torch', torch.__version__)
print('open3d', o3d.__version__)
print('numpy', np.__version__)
print('pandas', pd.__version__)
print('cuda_available', torch.cuda.is_available())
print('ml_torch_ok', True)  # reaching this line means import succeeded
```

Then run:
```bash id="7cy9vk"
python tools/check_versions.py | tee -a logs/day1_environment_report.txt
```

Expected pass result:
- the script runs without crashing
- Python prints `3.12.3`
- Torch prints `2.2.2+cu121`
- Open3D prints `0.19.0`
- NumPy prints `1.26.4`
- pandas prints `3.0.2`
- `ml_torch_ok True`

If it fails, that means:
At least one currently assumed local package fact is no longer true in the active environment.

Next fix to try:
Check the first failing import or version mismatch and repair only that package or environment layer before retrying.

### Step 5

Action:
Create and run a focused import-verification script for Open3D Torch backend and PandaSet devkit availability.

Why this step exists:
The most important Day 1 execution truth is not just package presence. It is that the current environment still exposes the exact backends and devkit objects the project depends on.

Command or file:
Create `tools/check_imports.py` with this exact content:

```python id="mmew0j"
import sys
import inspect
import torch
import open3d as o3d
import open3d.ml.torch as ml3d
from open3d.ml.torch.models import RandLANet
from pandaset import DataSet, geometry
import pandaset

print('python', sys.version)
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('open3d', o3d.__version__)
print('ml_torch_ok', True)  # reaching this line means import succeeded
print('randlanet_ok', RandLANet is not None)
print('pandaset_ok', DataSet is not None)
print('ego_helper_exists', hasattr(geometry, 'lidar_points_to_ego'))
print('pandaset_module_path', inspect.getfile(pandaset))
```

Then run:
```bash id="ri7a2u"
python tools/check_imports.py | tee -a logs/day1_environment_report.txt
```

Expected pass result:
- the script runs without crashing
- `ml_torch_ok True`
- `randlanet_ok True`
- `pandaset_ok True`
- the module path printed for `pandaset` points into the current project environment

If it fails, that means:
At least one inherited environment assumption from the parent documents is false in the active environment.

Next fix to try:
Read the first failing import and repair that exact package layer before retrying. Do not continue with partial import success.

### Step 6

Action:
Create and run a patched-devkit provenance check.

Why this step exists:
The current local setup depends on a patched PandaSet devkit that supports `.pkl` frame fallback. A reinstall from an unpatched source can silently break dataset reading later.

Command or file:
Create `tools/check_pandaset_devkit_origin.py` with this exact content:

```python id="nzh1vk"
import inspect
import pandaset
from pathlib import Path

module_path = Path(inspect.getfile(pandaset)).resolve()
print('pandaset_module_path', module_path)
print('contains_project_root', '/home/pheikara/University/Y3S2/Thesis/Pipeline' in str(module_path))
print('local_patched_source_exists', Path('./pandaset-devkit/python').resolve().exists())
print('preserve_patch_reinstall_command', 'pip install --no-deps ./pandaset-devkit/python')
```

Then run:
```bash id="rith1v"
python tools/check_pandaset_devkit_origin.py | tee -a logs/day1_environment_report.txt
```

Expected pass result:
- the script runs
- the pandaset module path prints successfully
- the local patched source path exists
- the reinstall command is recorded

If it fails, that means:
The project may no longer be using the intended patched devkit setup or the local patched source path is missing.

Next fix to try:
If the current install is broken or ambiguous, reinstall the devkit from the local patched source using:
`pip install --no-deps ./pandaset-devkit/python`
and rerun the import checks.

## Day 1 deliverables

- `logs/day1_environment_report.txt`
- `logs/dataset_root.txt`
- `tools/check_versions.py`
- `tools/check_imports.py`
- `tools/check_pandaset_devkit_origin.py`
- one complete import-verification result tied to the active `panda` environment

## Day 1 stop condition

Stop Day 1 only when:
- the active environment is confirmed to be `panda`
- the core stack imports successfully
- the canonical dataset root file exists and points to the real dataset root
- the patched devkit setup is confirmed well enough that Day 2 can trust the current install

Do not proceed to Day 2 with an unresolved `open3d.ml.torch` import failure or an unresolved PandaSet devkit origin problem.

## What evidence to record before Day 2

Record in `logs/day1_environment_report.txt`:
- the exact local Python version
- the exact installed Torch version
- the exact installed Open3D version
- the exact installed NumPy version
- the exact installed pandas version
- whether CUDA is visible to Torch
- whether `open3d.ml.torch` imported
- whether PandaSet imported
- whether `geometry.lidar_points_to_ego` exists locally
- the current pandaset module path
- the local patched reinstall command

# Day 2

## Goal

Verify the local PandaSet installation, confirm semantic segmentation access, confirm raw label-ID meaning, confirm forward-only filtering behavior, and confirm that semantic labels still align after filtering.

## Milestone by end of Day 2

By the end of Day 2, the project must know whether:
- PandaSet is locally readable
- semantic segmentation is locally accessible
- the raw semantic IDs match the intended project meaning
- the chosen local sequence and frame are usable for early checks
- the sensor selection used for the forward-only baseline is locally usable
- semantic labels still align after filtering
- intensity is actually present and readable on local frames

### Step 1

Action:
Create and run a local sequence and sensor precheck before any later Day 2 step assumes a particular sequence, frame, or sensor selection.

Why this step exists:
Later Day 2 checks must not quietly depend on unverified assumptions such as sequence `'001'`, frame `10`, or the operational meaning of sensor selector `1`.

Command or file:
Create `tools/precheck_sequence_and_sensor.py` with this exact content:

```python id="m08x50"
from pathlib import Path
from pandaset import DataSet

root = Path('logs/dataset_root.txt').read_text().strip()
ds = DataSet(root)

sequence_ids = sorted([p.name for p in Path(root).iterdir() if p.is_dir() and p.name.isdigit()])
print('sequence_count', len(sequence_ids))
print('sequence_head', sequence_ids[:10])

chosen_seq = '001' if '001' in sequence_ids else sequence_ids[0]
seq = ds[chosen_seq]

try:
    seq.load_lidar().load_semseg()
except Exception:
    seq.load_lidar()
    seq.load_semseg()

# Frame keys may be integers or strings ('00', '01', ...) depending on the devkit version.
# Normalise to int where possible so downstream scripts can use int(chosen_frame) safely.
raw_frame_keys = sorted(seq.lidar.data.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
first_key = raw_frame_keys[0]
chosen_frame = int(first_key) if str(first_key).isdigit() else first_key

pc_all = seq.lidar[chosen_frame]
all_cols = list(pc_all.columns)
print('chosen_seq', chosen_seq)
print('chosen_frame', chosen_frame)
print('all_points', len(pc_all))
print('all_columns', all_cols)

sensor_ok = False
sensor_column_present = 'd' in all_cols
if sensor_column_present:
    print('all_unique_sensor_ids', sorted(pc_all['d'].unique().tolist()))

seq.lidar.set_sensor(1)
pc_front = seq.lidar[chosen_frame]
print('front_points', len(pc_front))
if sensor_column_present and 'd' in pc_front.columns:
    front_ids = sorted(pc_front['d'].unique().tolist())
    print('front_unique_sensor_ids', front_ids)
    sensor_ok = len(pc_front) > 0 and front_ids == [1]
else:
    print('front_unique_sensor_ids', 'COLUMN_NOT_AVAILABLE')
    sensor_ok = len(pc_front) > 0

Path('logs/day2_chosen_sequence.txt').write_text(f'{chosen_seq}\n')
Path('logs/day2_chosen_frame.txt').write_text(f'{chosen_frame}\n')
Path('logs/day2_forward_sensor.txt').write_text('1\n')
print('forward_sensor_candidate', 1)
print('forward_sensor_usable', sensor_ok)
```

Then run:
```bash id="hlx925"
python tools/precheck_sequence_and_sensor.py | tee logs/day2_dataset_report.txt
```

Expected pass result:
- the script reports a positive `sequence_count`
- it chooses one usable local sequence and frame
- it writes:
  - `logs/day2_chosen_sequence.txt`
  - `logs/day2_chosen_frame.txt`
  - `logs/day2_forward_sensor.txt`
- `forward_sensor_usable` is `True`

If it fails, that means:
A later Day 2 assumption about the chosen sequence, chosen frame, or forward-sensor selection is not safe locally.

Next fix to try:
Inspect the printed sequence count, chosen sequence, frame keys, and sensor-ID behavior, then adjust the chosen sequence or frame before continuing.

### Step 2

Action:
Create and run a sequence/frame access check using the sequence and frame selected by the precheck.

Why this step exists:
This step verifies that the chosen local sequence and frame are actually usable as later Day 2 and Day 3 references.

Command or file:
Create `tools/check_pandaset_frame_access.py` with this exact content:

```python id="t4w3nc"
from pathlib import Path
from pandaset import DataSet

root = Path('logs/dataset_root.txt').read_text().strip()
seq_id = Path('logs/day2_chosen_sequence.txt').read_text().strip()
frame_idx = int(Path('logs/day2_chosen_frame.txt').read_text().strip())

ds = DataSet(root)
seq = ds[seq_id]
seq.load_lidar()
frame = seq.lidar[frame_idx]
print('chosen_seq', seq_id)
print('chosen_frame', frame_idx)
print('frame_type', type(frame))
print('frame_rows', len(frame))
print('frame_columns', list(frame.columns))
```

Then run:
```bash id="h2cdb8"
python tools/check_pandaset_frame_access.py | tee -a logs/day2_dataset_report.txt
```

Expected pass result:
- the chosen sequence loads
- the chosen frame object exists
- row count is positive
- LiDAR columns are printed

If it fails, that means:
- the chosen sequence or frame from the precheck is not actually stable in script form
- or the dataset root in `logs/dataset_root.txt` is wrong

Next fix to try:
Rerun the precheck, then repeat this step using the new chosen sequence and frame.

### Step 3

Action:
Create and run a semantic-segmentation access check using the chosen local sequence.

Why this step exists:
The thesis depends on point-wise semantic segmentation. If semseg access fails locally, the project meaning cannot be implemented.

Command or file:
Create `tools/check_pandaset_semseg_access.py` with this exact content:

```python id="ecjmp7"
from pathlib import Path
from pandaset import DataSet

root = Path('logs/dataset_root.txt').read_text().strip()
seq_id = Path('logs/day2_chosen_sequence.txt').read_text().strip()

ds = DataSet(root)
seq = ds[seq_id]
seq.load_semseg()
print('chosen_seq', seq_id)
print('semseg_loaded', seq.semseg is not None)
print('classes_type', type(seq.semseg.classes))
print('classes_sample', list(seq.semseg.classes.items())[:12])
```

Then run:
```bash id="zzazya"
python tools/check_pandaset_semseg_access.py | tee -a logs/day2_dataset_report.txt
```

Expected pass result:
- `semseg_loaded` is true
- semantic classes can be inspected
- the class mapping prints successfully

If it fails, that means:
The local PandaSet installation does not expose semantic segmentation the way the project expects.

Next fix to try:
Check that the local dataset includes semantic annotations and that the current devkit install is still the patched working one.

### Step 4

Action:
Verify the raw label-ID meanings that the project treats as critical.

Why this step exists:
The project meaning depends on the raw ID space being what the master document assumes. This must be confirmed through the local devkit, not trusted from planning text.

Command or file:
Create `tools/check_raw_label_ids.py` with this exact content:

```python id="5dutg2"
from pathlib import Path
from pandaset import DataSet

root = Path('logs/dataset_root.txt').read_text().strip()
seq_id = Path('logs/day2_chosen_sequence.txt').read_text().strip()

ds = DataSet(root)
seq = ds[seq_id]
seq.load_semseg()

classes = seq.semseg.classes
print('classes_type', type(classes))

# The devkit may store class keys as integers or as strings depending on version.
# Try both to avoid a silent NOT_FOUND when keys are string-typed.
for k in [1, 2, 3, 4, 7, 8, 9, 10]:
    val = classes.get(k, classes.get(str(k), 'NOT_FOUND'))
    print(f'id_{k}', val)
```

Then run:
```bash id="q6cdsn"
python tools/check_raw_label_ids.py | tee -a logs/day2_dataset_report.txt
```

Expected pass result:
The local mapping confirms the project-critical meanings:
- `id_4 Reflection`
- `id_7 Road`
- `id_8 Lane Line Marking`
- `id_9 Stop Line Marking`

If it fails, that means:
The parent-document raw-ID assumption is wrong for the local dataset copy.

Next fix to try:
Replace the remap plan with the locally verified ID meanings before any adapter code is written.

### Step 5

Action:
Verify forward-only filtering on the chosen local frame.

Why this step exists:
The project fixes forward-facing LiDAR as baseline meaning, but the actual local filtering behavior must be validated.

Command or file:
Create `tools/check_forward_only_filter.py` with this exact content:

```python id="ppjqzi"
from pathlib import Path
from pandaset import DataSet

root = Path('logs/dataset_root.txt').read_text().strip()
seq_id = Path('logs/day2_chosen_sequence.txt').read_text().strip()
frame_idx = int(Path('logs/day2_chosen_frame.txt').read_text().strip())
forward_sensor = int(Path('logs/day2_forward_sensor.txt').read_text().strip())

ds = DataSet(root)
seq = ds[seq_id]
seq.load_lidar()
pc_all = seq.lidar[frame_idx]
print('chosen_seq', seq_id)
print('chosen_frame', frame_idx)
print('all_points', len(pc_all))
print('all_columns', list(pc_all.columns))

if 'd' in pc_all.columns:
    print('all_unique_sensor_ids', sorted(pc_all['d'].unique().tolist()))

seq.lidar.set_sensor(forward_sensor)
pc_front = seq.lidar[frame_idx]
print('front_sensor', forward_sensor)
print('front_points', len(pc_front))
if 'd' in pc_front.columns:
    print('front_unique_sensor_ids', sorted(pc_front['d'].unique().tolist()))
else:
    print('front_unique_sensor_ids', 'COLUMN_NOT_AVAILABLE')
```

Then run:
```bash id="eu80ur"
python tools/check_forward_only_filter.py | tee -a logs/day2_dataset_report.txt
```

Expected pass result:
- front-only point count is smaller than or equal to all-points count
- the surviving points report only sensor ID `1`, if the sensor-ID column exists
- if the sensor-ID column is not available in the filtered frame, the filtered frame is still non-empty and usable

If it fails, that means:
The local devkit behavior is not matching the inherited assumption about forward-only filtering.

Next fix to try:
Inspect the returned dataframe structure and the actual meaning of the local sensor-ID column before writing adapter code.

### Step 6

Action:
Verify semantic alignment after forward-only filtering.

Why this step exists:
This is one of the most important inherited technical truths of the whole project. If alignment fails, the project is not semantically valid.

Command or file:
Create `tools/check_semseg_alignment.py` with this exact content:

```python id="fvfkbj"
from pathlib import Path
from pandaset import DataSet

root = Path('logs/dataset_root.txt').read_text().strip()
seq_id = Path('logs/day2_chosen_sequence.txt').read_text().strip()
frame_idx = int(Path('logs/day2_chosen_frame.txt').read_text().strip())
forward_sensor = int(Path('logs/day2_forward_sensor.txt').read_text().strip())

ds = DataSet(root)
seq = ds[seq_id]

try:
    seq.load_lidar().load_semseg()
except Exception:
    seq.load_lidar()
    seq.load_semseg()

seq.lidar.set_sensor(forward_sensor)
pc_df = seq.lidar[frame_idx]
semseg_df = seq.semseg[frame_idx]

raw_labels = semseg_df.loc[pc_df.index, 'class']
print('point_count', len(pc_df))
print('aligned_label_count', len(raw_labels))
print('same_count', len(pc_df) == len(raw_labels))
print('label_head', raw_labels.head().tolist())
```

Then run:
```bash id="9rcf1d"
python tools/check_semseg_alignment.py | tee -a logs/day2_dataset_report.txt
```

Expected pass result:
- point count equals aligned label count exactly
- label access by surviving point index works without error

If it fails, that means:
Either the parent assumption about index-based alignment is wrong locally, or something in the local dataframe behavior differs from the expected one.

Next fix to try:
Inspect the dataframe indices directly and confirm that no reindexing, resetting, or hidden filtering behavior is occurring.

### Step 7

Action:
Confirm that intensity is readable on the chosen local frame.

Why this step exists:
The intended baseline feature concept includes one processed intensity channel. That feature path cannot be trusted until local frames actually expose it.

Command or file:
Create `tools/check_intensity_field.py` with this exact content:

```python id="y5nbg6"
from pathlib import Path
from pandaset import DataSet

root = Path('logs/dataset_root.txt').read_text().strip()
seq_id = Path('logs/day2_chosen_sequence.txt').read_text().strip()
frame_idx = int(Path('logs/day2_chosen_frame.txt').read_text().strip())
forward_sensor = int(Path('logs/day2_forward_sensor.txt').read_text().strip())

ds = DataSet(root)
seq = ds[seq_id]
seq.load_lidar()
seq.lidar.set_sensor(forward_sensor)
pc_df = seq.lidar[frame_idx]

print('columns', list(pc_df.columns))
intensity_column = 'i' if 'i' in pc_df.columns else None
print('intensity_column', intensity_column)
if intensity_column is not None:
    print('intensity_head', pc_df[intensity_column].head().tolist())
    print('intensity_min', float(pc_df[intensity_column].min()))
    print('intensity_max', float(pc_df[intensity_column].max()))
else:
    print('intensity_head', 'COLUMN_NOT_FOUND')
```

Then run:
```bash id="jktgo2"
python tools/check_intensity_field.py | tee -a logs/day2_dataset_report.txt
```

Expected pass result:
- the script prints the local dataframe columns
- the intensity column used by the local frame is identified explicitly
- if the local column is `i`, intensity values print successfully

If it fails, that means:
The inherited assumption about the local LiDAR dataframe schema is wrong or incomplete.

Next fix to try:
Use the printed dataframe columns to determine the correct local intensity field before Day 3.

## Day 2 deliverables

- `logs/day2_dataset_report.txt`
- `logs/day2_chosen_sequence.txt`
- `logs/day2_chosen_frame.txt`
- `logs/day2_forward_sensor.txt`
- `tools/precheck_sequence_and_sensor.py`
- `tools/check_pandaset_frame_access.py`
- `tools/check_pandaset_semseg_access.py`
- `tools/check_raw_label_ids.py`
- `tools/check_forward_only_filter.py`
- `tools/check_semseg_alignment.py`
- `tools/check_intensity_field.py`

## Day 2 stop condition

Stop Day 2 only when:
- the dataset is confirmed locally readable,
- semantic segmentation access is real,
- raw IDs are locally verified,
- a locally usable sequence and frame are chosen and recorded,
- the forward-only sensor selection is locally usable,
- semseg alignment is confirmed,
- intensity is confirmed present and its local column is known.

Do not proceed to Day 3 if any of those remain unresolved.

## What evidence to record before Day 3

Record in `logs/day2_dataset_report.txt`:
- the local sequence count
- the chosen sequence ID
- the chosen frame index
- the forward-sensor selection used for the baseline
- the local raw-ID mapping subset for IDs `1,2,3,4,7,8,9,10`
- one front-only point count
- one all-points count
- one semseg-alignment equality check
- the actual LiDAR dataframe columns
- the local intensity column name and min/max if available

# Day 3

## Goal

Create one minimal adapter-inspection script that turns real PandaSet frames into the intended baseline contract:
- `point`
- `feat`
- `label`

Day 3 proves that the project meaning can be expressed correctly in code on the local system.

## Milestone by end of Day 3

By the end of Day 3, one minimal script must:
- load the chosen real frame from the canonical dataset root,
- apply forward-only filtering,
- preserve point indices for semseg alignment,
- produce ego-local `xyz`,
- produce one processed `feat` channel from intensity,
- remap raw labels into the intended target space,
- print shape and unique-label diagnostics,
- report lane fraction on one real frame.

### Step 1

Action:
Create one minimal script file for adapter inspection only.

Why this step exists:
The first adapter implementation should be small enough to debug directly against real local data.

Command or file:
Create:
`tools/inspect_pandaset_ff_lane3_sample.py`

Expected pass result:
The file exists and is dedicated only to local frame inspection and contract testing.

If it fails, that means:
The project structure is not being created cleanly on the local filesystem.

Next fix to try:
Create the directory and file manually, then confirm write permissions.

### Step 2

Action:
Build the script around one explicit, reusable local input path convention.

Why this step exists:
The script must not rely on repeated manual path edits or on hidden choices from earlier steps.

Command or file:
Inside `tools/inspect_pandaset_ff_lane3_sample.py`, make the script read:
- dataset root from `logs/dataset_root.txt`
- chosen sequence from `logs/day2_chosen_sequence.txt`
- chosen frame from `logs/day2_chosen_frame.txt`
- forward sensor from `logs/day2_forward_sensor.txt`

Expected pass result:
The script has one canonical local source for dataset path and chosen sample selection.

If it fails, that means:
The script is still depending on manual path replacement or hidden hardcoded choices.

Next fix to try:
Replace hardcoded dataset root, sequence, frame, and sensor values with reads from the log files.

### Step 3

Action:
Make the script structure explicit and minimal.

Why this step exists:
Coding agents should not need to reconstruct the intended script structure from memory.

Command or file:
`tools/inspect_pandaset_ff_lane3_sample.py` must contain these logical blocks in this order:
1. imports
2. local-path readers for dataset root, chosen sequence, chosen frame, chosen sensor
3. local frame load
4. forward-only filtering
5. index-preserving semseg alignment
6. ego-local conversion
7. intensity-to-`feat` preprocessing
8. raw-to-target remap
9. shape checks
10. unique-label checks
11. lane-fraction calculation
12. final printed summary

Expected pass result:
The script has a clear beginning-to-end structure matching the intended adapter contract.

If it fails, that means:
The local script is still too ad hoc to serve as a stable inspection tool.

Next fix to try:
Reorganize the script into the required block order before debugging individual lines.

### Step 4

Action:
Implement local frame loading and forward-only filtering.

Why this step exists:
The adapter must reflect the fixed project meaning of forward-only LiDAR only.

Command or file:
Inside the script, use the same local loading and filtering behavior already verified on Day 2.

If the local devkit allows chainable loading, it may be used.
If it does not, use separate `load_lidar()` and `load_semseg()` calls.

Expected pass result:
The script prints the chosen sequence, chosen frame, and a positive filtered point count.

If it fails, that means:
The verified Day 2 data-access pattern was not transferred correctly into the inspection script.

Next fix to try:
Reuse the exact successful Day 2 access pattern before changing anything else.

### Step 5

Action:
Implement index-preserving semantic alignment.

Why this step exists:
This is the single most fragile semantic line in the first 4 days.

Command or file:
Inside the script, implement an index-preserving alignment step whose invariant is:
- semantic labels are selected by the surviving LiDAR dataframe index
- no index reset or reorder happens before alignment

One locally valid form may be:
```python id="jlwmgq"
raw_labels = semseg_df.loc[pc_df.index, 'class'].to_numpy(np.int32)
```

Expected pass result:
- aligned label count equals filtered point count
- no indexing error occurs

If it fails, that means:
The script broke the dataframe index assumption, or the local dataframe schema differs from the parent-generated expectation.

Next fix to try:
Print `pc_df.index[:5]`, `semseg_df.index[:5]`, and inspect the actual column names before changing the alignment logic.

### Step 6

Action:
Implement ego-local `xyz` conversion.

Why this step exists:
The intended baseline coordinate frame is ego-local, but that path is still a locally verified implementation assumption.

Command or file:
Inside the script, convert:
- world-frame `xyz` extracted from the current point dataframe
- using the current frame pose
- into ego-local coordinates

If the local devkit exposes `geometry.lidar_points_to_ego` and it behaves correctly, it may be used.
If not, implement the equivalent inverse-pose transform directly.

Expected pass result:
The script prints a `point_shape` whose last dimension is `3`.

If it fails, that means:
The local geometry helper, pose format, or dataframe schema differs from the parent-generated expectation.

Next fix to try:
Check whether `geometry.lidar_points_to_ego` exists locally. If not, stop using the helper and implement the inverse-pose transform directly from the local pose object.

### Step 7

Action:
Implement intensity-to-`feat` preprocessing.

Why this step exists:
The intended baseline feature concept includes one processed intensity channel, but Day 3 only needs a structurally valid local preprocessing path, not final statistics.

Command or file:
Inside the script:
- read raw intensity from the local intensity column verified on Day 2
- cast to `float32`
- produce `feat` with shape `[N, 1]`
- use a temporary structural preprocessing path only

Use this Day 3 rule:
- frame-local min-max scaling is acceptable **only for adapter-contract inspection**
- do **not** freeze this as thesis truth

Expected pass result:
The script prints a `feat_shape` whose last dimension is `1`.

If it fails, that means:
The local intensity column is different from what the script expects, or `feat` is not shaped correctly.

Next fix to try:
Print the raw intensity dtype and shape, then force `feat = intensity[:, None].astype(np.float32)` before adding any further preprocessing.

### Step 8

Action:
Implement raw-to-target remap.

Why this step exists:
This is where project semantic meaning becomes executable.

Command or file:
Inside the script, write the **locally verified** remap into:
- `0 = ignore`
- `1 = road`
- `2 = lane`
- `3 = other`

Expected pass result:
The script prints unique label values that are a subset of `{0,1,2,3}`.

If it fails, that means:
The remap logic does not match the locally verified raw IDs.

Next fix to try:
Compare the remap table line-by-line against the Day 2 locally verified class mapping before changing anything else.

### Step 9

Action:
Add shape checks, unique-label checks, and lane-fraction measurement.

Why this step exists:
The adapter must prove its structural correctness explicitly, not by assumption.

Command or file:
At the end of the script, print:
- `point_shape`
- `feat_shape`
- `label_shape`
- `label_unique`
- `lane_fraction`

Expected pass result:
- `point_shape` ends with `3`
- `feat_shape` ends with `1`
- `label_shape[0] == point_shape[0]`
- unique labels are a subset of `{0,1,2,3}`
- `lane_fraction` prints as one numeric value

If it fails, that means:
The sample contract is broken or the chosen frame is not suitable for lane-fraction inspection.

Next fix to try:
Correct one tensor path at a time in this order: `point`, then `feat`, then `label`, then `lane_fraction`.

### Step 10

Action:
Annotate the script with the lines most likely to fail locally.

Why this step exists:
The parent-generated implementation is scaffold, not guaranteed truth. Coding tools need to know where fragility is highest.

Command or file:
Add explicit comments marking these likely failure points:
- raw-label alignment line
- geometry helper usage
- pose access path
- intensity column assumption
- raw-ID remap assumptions

Expected pass result:
The script clearly marks local-fragility lines for future debugging.

If it fails, that means:
The script is still being treated as if it were plug and play.

Next fix to try:
Add one-line comments directly above each fragile line.

## Day 3 deliverables

- `tools/inspect_pandaset_ff_lane3_sample.py`
- one locally tested real-frame adapter scaffold
- printed `point / feat / label` shapes
- printed unique label values
- one measured lane fraction
- inline comments marking locally fragile assumptions

## Day 3 stop condition

Stop Day 3 only when the script can run end to end on at least one real frame and print a valid sample contract:
- `point`
- `feat`
- `label`

Do not proceed to Day 4 if the adapter still depends on untested assumptions about shape, label alignment, local APIs, or local field naming.

## What evidence to record before Day 4

Record in `logs/day3_adapter_report.txt`:
- one successful script run
- point shape
- feat shape
- label shape
- unique labels
- one lane-fraction value
- the local intensity field used
- any local deviations from parent-document assumptions

# Day 4

## Goal

Create the minimal local project structure, instantiate RandLA-Net against the local adapter contract, run one model-build or contract check, and define the exact stop conditions before any real training.

## Milestone by end of Day 4

By the end of Day 4, the project must be able to say:

- the project structure exists
- a minimal config placeholder exists
- the model can be instantiated against the current local contract
- the local data contract and model assumptions are structurally compatible enough to justify later Milestone A work
- the exact no-training stop boundary has been reached cleanly

### Step 1

Action:
Create the minimal project structure needed for model-construction testing.

Why this step exists:
Day 4 needs stable file locations for the adapter and minimal config placeholder, but it must still remain pre-training.

Command or file:
Create these required paths if they do not already exist:
- `datasets/`
- `configs/`
- `logs/`
- `tools/`

Also create or promote one adapter module file at:
- `datasets/pandaset_ff_lane3.py`

And create the package init file so Python treats `datasets/` as an importable package:
- `datasets/__init__.py`

```bash
mkdir -p datasets configs logs tools
touch datasets/__init__.py
```

Expected pass result:
- all four directories exist
- `datasets/pandaset_ff_lane3.py` exists
- the project tree is readable by Python from the project root

If it fails, that means:
The local filesystem layout or import assumptions are not stable enough yet.

Next fix to try:
Confirm the current working directory and Python import root before continuing.

### Step 2

Action:
Create a minimal config placeholder.

Why this step exists:
The model-construction stage needs a config object, but Day 4 does not need a full final training configuration.

Command or file:
Create:
`configs/randlanet_pandaset_ff_lane3.yml`

The file must contain, at minimum:
- one `model` section
- one `dataset` section
- one `pipeline` section
- `num_classes: 3`
- a local `in_channels` value consistent with the current adapter contract
- a placeholder `device` field that reflects the current local choice without pretending it is universally fixed

Provisional or locally unresolved values must be expressed in comments, not as invalid YAML syntax.

Expected pass result:
- the YAML file exists
- it parses without syntax error
- it is minimal but structurally usable by a build-check script

If it fails, that means:
The config placeholder is malformed or is trying to encode unresolved planning notes as invalid YAML.

Next fix to try:
Move provisional notes into YAML comments or into surrounding documentation, then validate the YAML syntax before testing model construction.

### Step 3

Action:
Create a dedicated Day 4 model-build check script.

Why this step exists:
Day 4 needs one explicit script whose job is to prove structural compatibility between the local adapter output, the config placeholder, and RandLA-Net construction.

Command or file:
Create:
`tools/check_randlanet_build.py`

That script must do all of the following:
- load the YAML placeholder
- import the adapter module from `datasets/pandaset_ff_lane3.py`
- call one adapter entry point that returns one real sample dict
- print `point_shape`, `feat_shape`, `label_shape`, and `label_unique`
- instantiate RandLA-Net using the local Open3D Torch API that is actually available
- print explicit success or failure lines for:
  - `sample_contract_ok`
  - `config_parse_ok`
  - `randlanet_build_ok`

Important invariant:
- there must be **one adapter entry point in `datasets/pandaset_ff_lane3.py`** that `tools/check_randlanet_build.py` can call to obtain one real sample
- the exact function name is **not fixed** by project truth, but the existence of one callable sample-building entry point is required

Expected pass result:
The script exists and is clearly dedicated to model-build and contract checking, not training.

If it fails, that means:
The project does not yet have a single explicit Day 4 model-validation entry point.

Next fix to try:
Create the script first, even if the internal implementation still needs refinement.

### Step 4

Action:
Give the build-check script a minimal, runnable skeleton.

Why this step exists:
Days 1 and 2 already provide concrete runnable scripts. Day 4 needs the same level of structural starting point so agents do not guess the API.

Command or file:
Use this syntactically valid skeleton as the starting content for `tools/check_randlanet_build.py`:

```python id="955zzr"
import sys
from pathlib import Path
import importlib
import yaml

# Add the project root to sys.path so datasets.pandaset_ff_lane3 is importable
# when this script is run as `python tools/check_randlanet_build.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_cfg(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    cfg_path = Path('configs/randlanet_pandaset_ff_lane3.yml')
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    cfg = load_cfg(str(cfg_path))
    print('config_parse_ok', True)

    module = importlib.import_module('datasets.pandaset_ff_lane3')

    # Set this to the actual adapter entry point name that returns one real sample dict.
    adapter_entry_name = 'build_one_sample'
    if not hasattr(module, adapter_entry_name):
        raise AttributeError(
            f"datasets.pandaset_ff_lane3 is missing the required sample entry point: {adapter_entry_name}"
        )

    sample = getattr(module, adapter_entry_name)()
    point = sample['point']
    feat = sample['feat']
    label = sample['label']

    print('point_shape', getattr(point, 'shape', None))
    print('feat_shape', getattr(feat, 'shape', None))
    print('label_shape', getattr(label, 'shape', None))
    print('label_unique', sorted(set(label.tolist())))
    print('sample_contract_ok', True)

    import open3d.ml.torch as ml3d

    # Local API variation is possible here. Adjust only if the local Open3D API requires it.
    model = ml3d.models.RandLANet(**cfg['model'])
    print('randlanet_build_ok', model is not None)


if __name__ == '__main__':
    main()
```

Expected pass result:
- the script is syntactically valid
- it is explicit about the one required local adapter entry point
- it is explicit about which line may vary with the local Open3D API

If it fails, that means:
The Day 4 build-check script is still too underspecified for agent use.

Next fix to try:
Keep the same script structure and adjust only the local adapter entry name or the locally required Open3D model-construction line.

### Step 5

Action:
Promote the Day 3 adapter logic into the minimal adapter module path used by Day 4.

Why this step exists:
The build-check script must call one real adapter entry point from the module form, not from the Day 3 standalone inspection script.

Command or file:
Inside `datasets/pandaset_ff_lane3.py`, create one entry point that returns a dict with:
- `point`
- `feat`
- `label`

The exact entry-point name is a local implementation choice, but it must be the same name referenced by `tools/check_randlanet_build.py`.

Mandatory comment requirement inside the feature-preprocessing block:
```python id="n1czn8"
# [PROVISIONAL: replace with training-split statistics before training]
```

Expected pass result:
- `datasets/pandaset_ff_lane3.py` exists
- it exposes one callable sample-building entry point
- that entry point returns a real sample dict from local PandaSet data
- the provisional intensity-preprocessing comment is present

If it fails, that means:
The Day 3 adapter logic was not successfully promoted into a reusable module form.

Next fix to try:
Move the Day 3 working logic into the module first, then add the entry point and provisional comment.

### Step 6

Action:
Run the model-build or contract-check script.

Why this step exists:
Day 4 is the first model-level structural compatibility check. It validates parent assumptions about model construction, config compatibility, and basic library API behavior.

Command or file:
```bash id="1v9d1e"
python tools/check_randlanet_build.py | tee logs/day4_model_build_report.txt
```

Expected pass result:
The script prints all of the following as explicit outputs:
- `config_parse_ok True`
- `point_shape ...`
- `feat_shape ...`
- `label_shape ...`
- `label_unique ...`
- `sample_contract_ok True`
- `randlanet_build_ok True`

If it fails, that means:
At least one inherited model, config, or API assumption from the parent documents is wrong locally.

Next fix to try:
Check these in order:
1. the adapter entry point name used by the build-check script
2. `in_channels`
3. `num_classes`
4. model import path
5. config field names
6. whether the local Open3D build exposes the expected RandLA-Net constructor

### Step 7

Action:
Run immediate sanity checks on the structural assumptions.

Why this step exists:
The goal is to validate parent assumptions about data-contract expectations and library behavior before Milestone A.

Command or file:
Use the output of `tools/check_randlanet_build.py` and confirm:
- `point_shape` ends with `3`
- `feat_shape` ends with `1`
- `label_shape[0] == point_shape[0]`
- `label_unique` is a subset of `{0,1,2,3}`
- `num_classes` in the config is `3`
- the adapter and config still reflect the intended project meaning

Expected pass result:
All structural checks are consistent with the project meaning and the current local adapter contract.

If it fails, that means:
The project meaning and local implementation have diverged.

Next fix to try:
Correct the first divergence before any later step is considered.

### Step 8

Action:
Write the exact stop conditions before real training.

Why this step exists:
Day 4 must end with an explicit no-training boundary, not with ambiguity.

Command or file:
Create:
`logs/day4_stop_conditions.txt`

That file must state clearly that work does **not** move into real training unless:
- the environment is stable enough
- PandaSet access is verified
- raw IDs are verified
- forward-only filtering is verified
- semseg alignment is verified
- the adapter is working
- the config placeholder is working
- the model-build or contract check passes

Expected pass result:
- `logs/day4_stop_conditions.txt` exists
- it explicitly excludes real training from the first 4 days
- it matches the parent documents’ boundary between Day 4 and Milestone A

If it fails, that means:
The project is still mixing Day 4 with later Milestone A work.

Next fix to try:
Rewrite the stop note so it explicitly excludes real training from the first 4 days.

## Day 4 deliverables

- `datasets/__init__.py`
- `datasets/pandaset_ff_lane3.py`
- `configs/randlanet_pandaset_ff_lane3.yml`
- `tools/check_randlanet_build.py`
- one successful adapter-module sample build
- one successful model-construction result
- `logs/day4_stop_conditions.txt`
- `logs/day4_model_build_report.txt`

## Day 4 stop condition

Stop Day 4 only when:
- the adapter still works in module form,
- the config placeholder parses,
- RandLA-Net can be constructed locally,
- the project has explicit stop conditions before real training.

Do not start real training on Day 4.

## What evidence to record before Day 5

Record in `logs/day4_model_build_report.txt`:
- final adapter sample shapes
- final adapter unique labels
- successful model-construction output
- successful config-parse output
- the exact local deviations from any parent-document model or config assumptions
- confirmation that `logs/day4_stop_conditions.txt` exists

## Conditions that must be true before work moves beyond Day 4

All of the following must be true before the project is allowed to move beyond Day 4:

1. the local environment is stable enough to import the intended stack
2. PandaSet is locally readable
3. semantic segmentation access is real
4. raw semantic IDs are verified locally
5. forward-only filtering has been confirmed locally
6. semantic alignment after filtering has been confirmed locally
7. one local adapter produces correct `point / feat / label`
8. one minimal config placeholder exists and parses
9. RandLA-Net can be instantiated locally against the current config
10. one model-build or contract check passes
11. explicit stop conditions before real training are written down

If any one of these is false, the project remains in the first-4-days stage and must not continue into Milestone A.
