# DigitalOcean Setup Step 1: Droplet Only

Generated: 2026-05-04

Scope: create a GPU Droplet, verify GPU access, and copy/clone the project code. Do **not** install project dependencies, create a venv, transfer PandaSet, or run training in this step.

Sources checked:

- DigitalOcean GPU Droplets overview: https://docs.digitalocean.com/products/gpu-droplets/
- DigitalOcean recommended GPU setup: https://docs.digitalocean.com/products/droplets/getting-started/recommended-gpu-setup/
- DigitalOcean GPU Droplet creation guide: https://docs.digitalocean.com/products/droplets/how-to/gpu/create/
- DigitalOcean Droplet pricing: https://docs.digitalocean.com/products/droplets/details/pricing/
- DigitalOcean GPU Droplet features/plans: https://docs.digitalocean.com/products/droplets/details/features/

Important current-doc note: DigitalOcean docs now say the NVIDIA **AI/ML-ready** image uses Ubuntu 22.04 with CUDA Toolkit 13.1 and NVIDIA driver stack 590. CUDA 12.9 / NVIDIA driver 575 is listed for DigitalOcean's **inference-optimized** image, not the current AI/ML-ready image. For this thesis setup, use the AI/ML-ready image unless we later discover Open3D/Torch compatibility problems.

## 2.1 Droplet Creation Procedure

1. Log in to the DigitalOcean web dashboard.

2. In the top-right, click **Create**.

3. Choose **GPU Droplets**.

4. Choose the region:

   - Select **AMS3 Amsterdam**.
   - Reason: you are in Bremen, Germany, so Amsterdam should give low latency.

5. Choose the image:

   - Go to the **Marketplace** tab.
   - Select **AI/ML Ready v1.0** or the current NVIDIA AI/ML Ready GPU image shown by DigitalOcean.
   - API image slug for single-GPU NVIDIA plans: `gpu-h100x1-base`.
   - DigitalOcean uses this slug for all single-GPU NVIDIA AI/ML-ready GPU Droplets, even non-H100 plans.

6. Confirm expected base software:

   - Current DigitalOcean docs for NVIDIA AI/ML-ready GPU Droplets: Ubuntu 22.04, CUDA Toolkit 13.1, NVIDIA driver stack 590, `nvidia-container-toolkit` preinstalled.
   - If the dashboard explicitly shows CUDA 12.9 / driver 575, that is likely the inference-optimized image. Avoid it for now unless we deliberately choose it later.

7. Choose the GPU plan:

   - Recommended for this first iteration: **NVIDIA RTX 6000 Ada**.
   - DigitalOcean size slug: `gpu-6000adax1-48gb`.
   - VRAM: 48 GB.
   - Price: `$1.57/hour`.
   - Why: 48 GB VRAM should be enough for RandLA-Net at `num_points: 16384`, and it is cheaper than H100 for smoke/stability tests.
   - H100 is currently listed at `$3.39/hour`; consider it later only if the real baseline benefits from it.

8. Authentication:

   - Use SSH key authentication.
   - Before creating the Droplet, verify your SSH key is uploaded:
   - DigitalOcean dashboard → **Settings** → **Security** → **SSH Keys**.
   - Confirm your laptop's public key is listed.
   - If missing, upload the contents of `~/.ssh/id_ed25519.pub` or the public key you normally use.

9. Hostname:

   - Use `thesis-c0-smoke`.

10. Optional features:

   - Skip backups.
   - Skip metrics/monitoring for this first short iteration.
   - Skip user data/cloud-init.
   - Keep networking/default project settings minimal.

11. Create the Droplet.

12. After creation:

   - The public IPv4 address appears on the Droplet page in the DigitalOcean dashboard.
   - Copy that IP address.
   - Default SSH user for DigitalOcean Droplets is `root`.

13. SSH into the Droplet from your laptop:

```bash
ssh root@<ip>
```

Replace `<ip>` with the Droplet's public IPv4 address.

## 2.2 GPU Access Verification

Run these commands immediately after SSH.

### Check GPU And Driver

```bash
nvidia-smi
```

Expected:

- Shows one NVIDIA GPU.
- GPU should be RTX 6000 Ada if you selected that plan.
- Driver should match the installed image. Current AI/ML-ready docs indicate driver stack 590; older or inference-optimized images may show driver 575.

If `nvidia-smi` is missing or shows no GPU, the image or plan is wrong. Destroy and recreate the Droplet.

### Check CUDA Toolkit

```bash
nvcc --version
```

Expected:

- Current AI/ML-ready image: CUDA Toolkit 13.1 according to DigitalOcean's current docs.
- If it reports CUDA 12.9, verify whether you accidentally selected the inference-optimized image.

If `nvcc` is missing, stop and report back before continuing.

### Check System Python

```bash
python3 --version
```

Expected:

- Ubuntu 22.04 default is usually Python 3.10.
- This is only a system check. Do not create the project venv yet.

### Check Disk Layout

```bash
df -h
```

Expected:

- Boot disk should have enough free space for repo, venv, and later dataset work.
- RTX 6000 Ada GPU Droplets are documented with a 500 GB boot disk.
- Confirm at least 100 GB free before continuing.

If disk space is unexpectedly small, stop and report back.

## 2.3 Cost Monitoring Discipline

DigitalOcean bills Droplets from creation until destruction.

Important:

- Powered-off Droplets still bill because the resources remain reserved.
- To stop billing, destroy the Droplet entirely from the dashboard.

Every cloud session should end like this:

1. Copy any useful artifacts off the Droplet.
2. Destroy the Droplet in the DigitalOcean dashboard.
3. Refresh the dashboard and verify no GPU Droplets are running.

Estimated cost for this step:

- 1 to 2 hours at `$1.57/hour`.
- Approximate total: `$1.57` to `$3.14`.

## 2.4 Clone Or Copy Repo

GitHub setup is not done yet, so there are two paths.

### Option A: GitHub Clone Later

Recommended before real C0 training, but deferred to the next iterative step:

1. Create a GitHub repository.
2. Add a correct `.gitignore`.
3. Commit the project source and small reproducibility artifacts.
4. Clone normally on the Droplet.

Do not do this in the current step because GitHub work is explicitly deferred.

### Option B: Rsync From Laptop For This Step

Use this now if you want to verify Droplet access before GitHub setup.

From your laptop, in a local terminal, run:

```bash
rsync -avh --progress \
  --exclude '.git/' \
  --exclude 'pandaset/' \
  --exclude 'panda/' \
  --exclude 'pandaset.zip' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'logs/milestone_c/runs/' \
  --exclude 'logs/milestone_c/checkpoints/' \
  --exclude 'logs/milestone_c/feature_cache/' \
  --exclude 'logs/RandLANet_PandaSetFFLane3_torch/' \
  "/home/pheikara/University/Y3S2/Thesis/Pipeline/" \
  root@<ip>:~/thesis/
```

Replace `<ip>` with the Droplet public IPv4 address.

Why these excludes matter:

- `pandaset/` is the large local dataset and is not transferred in this step.
- `panda/` is the local Python environment and should not be copied.
- `pandaset.zip` is 43 GB and must not be transferred.
- `__pycache__/` and `*.pyc` are generated Python artifacts.
- Milestone C runs/checkpoints/feature caches are not needed for first droplet verification.
- `logs/RandLANet_PandaSetFFLane3_torch/` is old Open3D run output and not needed on the new machine.

This keeps `pandaset-devkit/` included, which is important because it contains the patched `.pkl` fallback support.

## 2.5 Verify Repo Is On The Droplet

After rsync finishes, SSH into the Droplet if not already connected:

```bash
ssh root@<ip>
```

Then run:

```bash
cd ~/thesis
ls -la
```

Expected:

- You should see project folders like `configs/`, `docs/`, `datasets/`, `src/`, `tools/`, `logs/`, and `pandaset-devkit/`.

Check the training entrypoint exists:

```bash
test -f tools/train_milestone_c.py && echo "train_milestone_c.py exists"
```

Expected:

```text
train_milestone_c.py exists
```

Check the patched PandaSet devkit fallback:

```bash
sed -n '1,50p' pandaset-devkit/python/pandaset/utils.py
```

Expected:

- `data_files(...)` should mention fallback between `.pkl.gz` and `.pkl`.
- It should contain logic for:
  - `if extension == 'pkl.gz': return sorted(glob.glob(... '*.pkl'))`
  - `if extension == 'pkl': return sorted(glob.glob(... '*.pkl.gz'))`

Do **not** yet:

- Create a venv.
- Install requirements.
- Install `pandaset-devkit`.
- Transfer the PandaSet dataset.
- Run training.

Those are for the next iterative step.

## 2.6 Session Shutdown

Before ending the session:

1. Note the Droplet IP and what you verified in a local note.
2. If you created any useful files on the Droplet, copy them back first.
3. In the DigitalOcean dashboard, destroy the Droplet.
4. Refresh the dashboard and confirm no GPU Droplets are still running.

Do not merely power off the Droplet; powered-off Droplets still bill.

