import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_pipeline.checks.day2 import check_intensity_field_main


if __name__ == "__main__":
    check_intensity_field_main()
