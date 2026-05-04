# Pipeline Project Setup Context (Current State)

## 1. Workspace layout

Root:

`/home/pheikara/University/Y3S2/Thesis/Pipeline`

Main folders/files in use:

- `panda/`  
  Python virtual environment (active env for this project)
- `pandaset/`  
  Extracted dataset location
- `pandaset/PandaSet/`  
  Actual PandaSet dataset root used by code
- `pandaset-devkit/`  
  Local PandaSet devkit source repository
- `pandaset.zip`  
  Original zip archive copy (already extracted; not needed at runtime)
- `requirements_working_panda.txt`  
  Working dependency reference file

## 2. Dataset path and format

Dataset root used by `DataSet(...)`:

`/home/pheikara/University/Y3S2/Thesis/Pipeline/pandaset/PandaSet`

Important format detail:

- This dataset copy stores per-frame files as **`.pkl`** (not `.pkl.gz`).
- Raw files are valid and readable with `pandas.read_pickle`.

## 3. Virtual environment and core runtime versions

Active environment:

- `VIRTUAL_ENV=/home/pheikara/University/Y3S2/Thesis/Pipeline/panda`

Validated runtime stack:

- Python `3.12.3`
- NumPy `1.26.4`
- PyTorch `2.2.2+cu121`
- Open3D `0.19.0`
- Open3D-ML torch backend import: `open3d.ml.torch` (works)
- `pandaset` installed from local devkit source (version `0.3.dev0`)

## 4. Devkit install status

Current install method used:

```bash
cd /home/pheikara/University/Y3S2/Thesis/Pipeline
source panda/bin/activate
pip uninstall -y pandaset
pip install --no-deps ./pandaset-devkit/python
```

This installs `pandaset` into the `panda` env from the local `pandaset-devkit/python` source tree.

## 5. Problem that existed

Before patching:

- Devkit loader searched only for `*.pkl.gz`.
- Local dataset has only `*.pkl`.
- Result: `seq.load_lidar()` loaded 0 frames.

## 6. Local code changes made (devkit compatibility patch)

Modified files:

1. `pandaset-devkit/python/pandaset/utils.py`
2. `pandaset-devkit/python/pandaset/sensors.py`
3. `pandaset-devkit/python/pandaset/annotations.py`

Change behavior:

- Added `data_files(directory, extension)` helper.
- Loader logic now:
  - **Prefer requested extension first** (e.g., `.pkl.gz`)
  - **Fallback to alternate pickle extension only if none found** (e.g., `.pkl`)
  - Does **not** load both patterns simultaneously.

Compatibility effect:

- Official `.pkl.gz` datasets continue to work.
- Local `.pkl` dataset variant now works.

## 7. Verified working outcomes after patch

Validation results observed:

- Python / NumPy / Torch / Open3D / Open3D-ML imports: **working**
- PandaSet devkit import (`from pandaset import DataSet`): **working**
- Sequence discovery: **working** (`103` sequences found)
- Sequence `001` load results:
  - LiDAR frames loaded: `80`
  - Cuboids frames loaded: `80`
  - Semseg frames loaded: `80`
  - Frame `00` data row counts were non-empty and consistent

## 8. What is now expected to work

- High-level devkit usage on this dataset copy:
  - `seq.load_lidar()`
  - `seq.load_cuboids()`
  - `seq.load_semseg()`
- Raw direct pickle reads
- Open3D-ML with torch backend imports in the same env

## 9. Important maintenance note

This fallback behavior is a **local devkit source change**.  
If `pandaset` is reinstalled from an unpatched source later, this compatibility can be lost.

If needed, reinstall from local patched source:

```bash
source /home/pheikara/University/Y3S2/Thesis/Pipeline/panda/bin/activate
pip install --no-deps /home/pheikara/University/Y3S2/Thesis/Pipeline/pandaset-devkit/python
```

## 10. Minimal sanity check command

```bash
cd /home/pheikara/University/Y3S2/Thesis/Pipeline
source panda/bin/activate
python - <<'PY'
from pandaset import DataSet
ds = DataSet('/home/pheikara/University/Y3S2/Thesis/Pipeline/pandaset/PandaSet')
seq = ds['001']
seq.load_lidar().load_cuboids().load_semseg()
print('lidar:', len(seq.lidar.data), 'cuboids:', len(seq.cuboids.data), 'semseg:', len(seq.semseg.data))
print('frame0 rows:', len(seq.lidar[0]), len(seq.cuboids[0]), len(seq.semseg[0]))
PY
```

