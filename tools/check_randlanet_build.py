import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib
import yaml
import numpy as np


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg_path = Path("configs/randlanet_pandaset_ff_lane3.yml")
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    cfg = load_cfg(str(cfg_path))
    print("config_parse_ok", True)

    module = importlib.import_module("datasets.pandaset_ff_lane3")
    adapter_entry_name = "build_one_sample"
    if not hasattr(module, adapter_entry_name):
        raise AttributeError(
            f"datasets.pandaset_ff_lane3 is missing the required sample entry point: {adapter_entry_name}"
        )

    sample = getattr(module, adapter_entry_name)()
    point = sample["point"]
    feat = sample["feat"]
    label = sample["label"]

    print("point_shape", getattr(point, "shape", None))
    print("feat_shape", getattr(feat, "shape", None))
    print("label_shape", getattr(label, "shape", None))
    print("label_unique", np.unique(label).tolist())

    if point.shape[1] != 3:
        raise RuntimeError("point_shape does not end with 3")
    if feat.shape[1] != 1:
        raise RuntimeError("feat_shape does not end with 1")
    if label.shape[0] != point.shape[0]:
        raise RuntimeError("label length does not match point length")
    if not set(np.unique(label).tolist()).issubset({0, 1, 2, 3}):
        raise RuntimeError("label_unique contains values outside {0,1,2,3}")
    print("sample_contract_ok", True)

    import open3d.ml.torch as ml3d

    model = ml3d.models.RandLANet(**cfg["model"])
    print("randlanet_build_ok", model is not None)


if __name__ == "__main__":
    main()
