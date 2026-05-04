import inspect
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.ml.torch as ml3d
import pandas as pd
import pandaset
import torch
from open3d.ml.torch.models import RandLANet
from pandaset import DataSet, geometry

from thesis_pipeline.core.paths import PATCHED_DEVKIT_REINSTALL_COMMAND


def check_versions_main() -> None:
    print("python", sys.version)
    print("torch", torch.__version__)
    print("open3d", o3d.__version__)
    print("numpy", np.__version__)
    print("pandas", pd.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("ml_torch_ok", True)  # reaching this line means import succeeded


def check_imports_main() -> None:
    print("python", sys.version)
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("open3d", o3d.__version__)
    print("ml_torch_ok", True)  # reaching this line means import succeeded
    print("randlanet_ok", RandLANet is not None)
    print("pandaset_ok", DataSet is not None)
    print("ego_helper_exists", hasattr(geometry, "lidar_points_to_ego"))
    print("pandaset_module_path", inspect.getfile(pandaset))


def check_pandaset_devkit_origin_main() -> None:
    module_path = Path(inspect.getfile(pandaset)).resolve()
    print("pandaset_module_path", module_path)
    print("contains_project_root", "/home/pheikara/University/Y3S2/Thesis/Pipeline" in str(module_path))
    print("local_patched_source_exists", Path("./pandaset-devkit/python").resolve().exists())
    print("preserve_patch_reinstall_command", PATCHED_DEVKIT_REINSTALL_COMMAND)
