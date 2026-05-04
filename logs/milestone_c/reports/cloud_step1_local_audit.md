# Cloud Step 1 Local Audit

Generated: 2026-05-04

Scope: local readiness check for the first DigitalOcean GPU Droplet iteration. No training, dependency installation, config edits, GitHub setup, or dataset transfer was performed.

## 1. Git State

### `git status`

Command:

```bash
git status --short
```

Result:

```text
fatal: not a git repository (or any of the parent directories): .git
```

Finding: this project directory currently has an empty `.git/` directory, but it is not a valid Git repository.

### Recent Commit History

Command:

```bash
git log --oneline | head -5
```

Result:

```text
fatal: not a git repository (or any of the parent directories): .git
```

Finding: there are no local commit reference points available yet. `tools/train_milestone_c.py` will write `git_available false` to `git_commit.txt` until a real Git repo exists.

## 2. `.gitignore` Check

Result: no `.gitignore` file exists in the project root.

Required exclusions that are currently not enforced by `.gitignore`:

- `pandaset/`
- `panda/`
- `pandaset.zip`
- `__pycache__/`
- `*.pyc`
- `logs/RandLANet_PandaSetFFLane3_torch/`
- likely future run-heavy paths such as `logs/milestone_c/runs/`, `logs/milestone_c/checkpoints/`, and `logs/milestone_c/feature_cache/`

Important non-exclusion:

- `pandaset-devkit/` should remain included when Git is initialized later because it contains the patched PandaSet devkit code.

Recommendation for later GitHub step: create `.gitignore` before the first commit. Do not initialize/commit in this step because GitHub work is explicitly deferred.

## 3. Repository Size

Command:

```bash
du -sh . --exclude='./pandaset' --exclude='./panda'
```

Result:

```text
43G .
```

Reason: `pandaset.zip` is present in the project root and is 43G.

Large local paths:

| Path | Size |
|---|---:|
| `pandaset/` | 81G |
| `panda/` | 7.9G |
| `pandaset.zip` | 43G |
| `logs/` | 42M |
| `pandaset-devkit/` | 30M |
| `pandaset-devkit.worktrees/` | 7.9M |
| `pandaset_001_00_forward_d1_xyz_intensity.pcd` | 3.7M |

Command:

```bash
du -sh .git/
```

Result:

```text
0 .git/
```

Finding: there is no usable Git history yet.

Cloud transfer implication: any rsync command must exclude `pandaset/`, `panda/`, and `pandaset.zip`. Otherwise transfer will be huge and expensive.

## 4. `pandaset-devkit/`

Command:

```bash
du -sh pandaset-devkit/
```

Result:

```text
30M pandaset-devkit/
```

Patch verification:

- File checked: `pandaset-devkit/python/pandaset/utils.py`
- Function checked: `data_files(directory, extension)`
- Patch present: yes

Relevant behavior:

- Primary extension is searched first.
- If `extension == 'pkl.gz'` and no files are found, it falls back to `*.pkl`.
- If `extension == 'pkl'` and no files are found, it falls back to `*.pkl.gz`.

Tracking status:

- Cannot verify Git tracking because this directory is not currently a valid Git repo.
- The directory itself is intact and should be included in the future repository.

## 5. Requirements Files

Root requirements files:

| File | Lines | Modified |
|---|---:|---|
| `requirements_working_panda.txt` | 176 | 2026-04-14 04:27 |

Finding: `requirements_working_panda.txt` exists and is the only root `requirements*.txt` file.

Important pinned versions:

- `numpy==1.26.4`
- `open3d==0.19.0`
- `torch==2.2.2+cu121`
- `torchvision==0.17.2+cu121`
- `torchaudio==2.2.2+cu121`

## 6. Readiness Verdict For Step 1

Proceed with **DigitalOcean droplet access verification and rsync-based code transfer only**.

Do not rely on `git clone` yet because GitHub setup is deferred and this folder is not currently a valid Git repository.

Before any future GitHub-based workflow:

- Initialize or repair Git.
- Add `.gitignore`.
- Ensure `pandaset.zip`, `pandaset/`, and `panda/` are not committed.
- Commit the source, docs, configs, patched `pandaset-devkit/`, and small milestone reports needed for reproducibility.

## 7. Issue 1 Resolution: `.git/`

Rechecked `.git/`:

```text
total 4
dr-xr-xr-x  2 pheikara pheikara   40 May  4 15:37 .
drwxrwxr-x 17 pheikara pheikara 4096 May  4 15:37 ..
```

Additional investigation found that `.git/` is not a normal directory. It is a read-only `tmpfs` mount:

```text
TARGET                                              SOURCE FSTYPE OPTIONS
/home/pheikara/University/Y3S2/Thesis/Pipeline/.git tmpfs  tmpfs  ro,nosuid,nodev,relatime,mode=555,uid=1000,gid=1000,inode64
```

Attempted action:

```bash
rm -rf .git
```

Result:

```text
rm: cannot remove '.git': Device or resource busy
```

Attempted action:

```bash
git init
```

Result:

```text
fatal: cannot copy '/usr/share/git-core/templates/description' to '/home/pheikara/University/Y3S2/Thesis/Pipeline/.git/description': Read-only file system
```

Conclusion: Git initialization is **blocked** by the read-only `.git` mount. This is not a normal corrupt Git repository and cannot be repaired by `git init` until the mount is removed.

Pending user action:

```bash
sudo umount /home/pheikara/University/Y3S2/Thesis/Pipeline/.git
rm -rf /home/pheikara/University/Y3S2/Thesis/Pipeline/.git
cd /home/pheikara/University/Y3S2/Thesis/Pipeline
git init
```

After that, run:

```bash
git status --short
git add -A
git commit -m "Initial commit: Milestone C entry state with validation metric plumbing"
git log --oneline -5
```

Initial commit hash: **not created yet** because Git initialization is blocked by the mounted `.git/` placeholder.

## 8. Issue 2 Resolution: `.gitignore`

Created `.gitignore` at project root.

Entries included:

- `pandaset/`
- `pandaset.zip`
- `panda/`
- `.venv/`, `venv/`
- `__pycache__/`, `pycache/`, `*.pyc`, `*.pyo`, `*.pyd`
- `.vscode/`, `.idea/`, `.claude/`, `.codex/`, `.agents/`
- OS/editor files such as `.DS_Store`, `Thumbs.db`, `*.swp`, `*~`
- Open3D runtime artifacts: `train_log/`, `logs/cache/`, active Open3D checkpoint/log paths
- Milestone C generated runtime artifacts: `logs/milestone_c/runs/`, `logs/milestone_c/checkpoints/`, `logs/milestone_c/feature_cache/`
- temporary files: `*.tmp`
- scratch/generated artifacts: `pandaset-devkit.worktrees/`, `*.pcd`

Important inclusion decision:

- `pandaset-devkit/` is **not** ignored. It must be tracked later because it contains the patched `.pkl` fallback code.
- `logs/raw_intensity_analysis/`, `logs/milestone_b_archive/`, and current `logs/milestone_c/reports/` / `logs/milestone_c/eval/` are **not** ignored.

`git status` after `.gitignore` creation:

```text
fatal: not a git repository (or any of the parent directories): .git
```

Reason: Git status still cannot run until the read-only `.git` mount is removed and `git init` succeeds.

Size check after excluding the local dataset, venv, and zip:

```bash
du -sh . --exclude='./pandaset' --exclude='./panda' --exclude='./pandaset.zip'
```

Result:

```text
84M .
```

This is a reasonable source/artifact size for the first commit once Git is repaired.

## 9. Issue 3: `pandaset.zip` vs `pandaset/`

Command:

```bash
ls -la pandaset.zip pandaset/ 2>/dev/null
```

Result:

```text
-rw-rw-r-- 1 pheikara pheikara 45206769975 Dec 18  2023 pandaset.zip

pandaset/:
total 12
drwxrwxr-x   3 pheikara pheikara 4096 Apr 13 22:07 .
drwxrwxr-x  17 pheikara pheikara 4096 May  4 15:37 ..
drwxrwxr-x 105 pheikara pheikara 4096 Apr 13 22:12 PandaSet
```

Size comparison:

| Path | Size | Status |
|---|---:|---|
| `pandaset.zip` | 43G | backup/archive; ignored |
| `pandaset/` | 81G | extracted working dataset; ignored |

Training dataset status:

- `pandaset/PandaSet/` exists.
- This is the extracted working dataset location used by the training pipeline.
- The zip is not used directly by the pipeline and is fine to keep locally as a backup.

No extraction is needed locally.

## 10. Final Git Repair Result

Final status after manual `.git` cleanup and Codex follow-up:

- Git repository is initialized.
- Current branch: `main`.
- Initial commit was created.
- A follow-up commit corrected `.gitignore` anchoring and ensured the patched PandaSet devkit Python package is tracked as normal files.

Current Git log:

```text
4bf2c74 Track patched PandaSet devkit package files
81eec91 Initial commit: Milestone C entry state with validation metric plumbing
```

Important verification:

- No staged submodule/gitlink entries remain (`git ls-files --stage | awk '$1 == "160000" {print}'` returned no rows).
- `pandaset-devkit/python/pandaset/utils.py` is tracked.
- The patched `.pkl` / `.pkl.gz` fallback code remains present in `pandaset-devkit/python/pandaset/utils.py`.
- Root dataset/env/archive paths remain ignored:
  - `/pandaset/`
  - `/pandaset.zip`
  - `/panda/`

Why a second commit was needed:

- The first `git add` saw `pandaset-devkit/` as an embedded Git repository and staged it as a gitlink.
- That would have been wrong for DigitalOcean/GitHub because the patched devkit source would not be included directly.
- The nested `pandaset-devkit/.git` metadata was removed, the devkit package was staged as normal files, and `.gitignore` was corrected from `pandaset/` to `/pandaset/` so it ignores only the root dataset and not `pandaset-devkit/python/pandaset/`.

Current user-action item:

- Create a remote GitHub repository.
- Add it as `origin`.
- Push branch `main`.
