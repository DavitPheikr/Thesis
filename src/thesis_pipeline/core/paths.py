from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGS_DIR = PROJECT_ROOT / "logs"
TOOLS_DIR = PROJECT_ROOT / "tools"
DATASETS_DIR = PROJECT_ROOT / "datasets"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SRC_DIR = PROJECT_ROOT / "src"

PATCHED_DEVKIT_REINSTALL_COMMAND = "pip install --no-deps ./pandaset-devkit/python"


def read_dataset_root() -> str:
    return (LOGS_DIR / "dataset_root.txt").read_text().strip()


def read_day2_chosen_sequence() -> str:
    return (LOGS_DIR / "day2_chosen_sequence.txt").read_text().strip()


def read_day2_chosen_frame():
    value = (LOGS_DIR / "day2_chosen_frame.txt").read_text().strip()
    return int(value) if value.isdigit() else value


def read_day2_forward_sensor() -> int:
    return int((LOGS_DIR / "day2_forward_sensor.txt").read_text().strip())
