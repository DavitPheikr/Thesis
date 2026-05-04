#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-panda312}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RECREATE_ENV=0

usage() {
    cat <<EOF
Usage: ./setup.sh [--recreate]

Creates/uses a Conda env, installs the project GPU stack, and runs setup checks.

Environment variables:
  ENV_NAME=${ENV_NAME}
  PYTHON_VERSION=${PYTHON_VERSION}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

Options:
  --recreate  Remove and recreate the Conda env before installing.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --recreate) RECREATE_ENV=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $arg" >&2; usage; exit 2 ;;
    esac
done

cd "$(dirname "$0")"

LOG_DIR="logs/milestone_c/reports"
CACHE_DIR=".cache"
MPL_DIR="$CACHE_DIR/matplotlib"
mkdir -p "$LOG_DIR" "$MPL_DIR"

LOG_FILE="$LOG_DIR/server_setup_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1

step() {
    echo
    echo "==> $1"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: required file missing: $1" >&2
        exit 1
    fi
}

require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "ERROR: required directory missing: $1" >&2
        exit 1
    fi
}

step "System checks"
echo "time_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo_root: $(pwd)"
echo "log_file: $LOG_FILE"
echo "cuda_visible_devices: $CUDA_VISIBLE_DEVICES"

require_file "requirements_working_panda.txt"
require_file "tools/train_milestone_c.py"
require_file "src/thesis_pipeline/eval/milestone_c_metrics.py"
require_dir "pandaset-devkit/python/pandaset"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. Use a GPU image with Conda/Miniconda installed." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. This does not look like a GPU-ready server." >&2
    exit 1
fi

conda --version
nvidia-smi
df -h .

step "Create or activate Conda env"
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [[ "$RECREATE_ENV" -eq 1 ]]; then
        conda env remove -n "$ENV_NAME" -y
        conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
    else
        echo "Using existing env: $ENV_NAME"
    fi
else
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(pwd)/$MPL_DIR}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(pwd)/pandaset-devkit/python:$(pwd)/src:${PYTHONPATH:-}"

python --version
python -m pip --version

step "Install PyTorch CUDA stack"
python -m pip install \
    "torch==2.2.2+cu121" \
    "torchvision==0.17.2+cu121" \
    "torchaudio==2.2.2+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

step "Install project requirements"
python -m pip install --no-build-isolation -r requirements_working_panda.txt

step "Verify installed environment"
python - <<'PY'
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import pandaset.utils as pandaset_utils
import torch

utils_path = Path(pandaset_utils.__file__).resolve()
utils_text = utils_path.read_text()

print("python", sys.version)
print("numpy", np.__version__)
print("pandas", pd.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("open3d", o3d.__version__)
print("pandaset_utils", utils_path)

assert np.__version__ == "1.26.4"
assert torch.__version__ == "2.2.2+cu121"
assert torch.cuda.is_available()
assert o3d.__version__ == "0.19.0"
assert "pandaset-devkit/python/pandaset/utils.py" in str(utils_path)
assert "fallback between .pkl.gz and .pkl" in utils_text
PY

step "Run project checks"
python tools/check_versions.py
python tools/check_imports.py
python -m py_compile tools/train_milestone_c.py src/thesis_pipeline/eval/milestone_c_metrics.py
python -c "from src.thesis_pipeline.eval.milestone_c_metrics import _self_test; _self_test()"
python tools/train_milestone_c.py --help >/dev/null

step "Done"
echo "setup_ok: true"
echo "env_name: $ENV_NAME"
echo "log_file: $LOG_FILE"
echo "next: transfer PandaSet, set dataset path, run dataset checks, then GPU smoke test."
