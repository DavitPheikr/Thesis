import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_pipeline.checks.day2 import precheck_sequence_and_sensor_main


if __name__ == "__main__":
    precheck_sequence_and_sensor_main()
