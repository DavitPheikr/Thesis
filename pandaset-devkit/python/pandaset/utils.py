#!/usr/bin/env python3
import glob
import os
from typing import List


def subdirectories(directory: str) -> List[str]:
    """List all subdirectories of a directory.

    Args:
        directory: Relative or absolute path

    Returns:
        List of path strings for every subdirectory in `directory`.
    """
    return [d.path for d in os.scandir(directory) if d.is_dir()]


def data_files(directory: str, extension: str) -> List[str]:
    """List data files and fallback between .pkl.gz and .pkl when needed.

    The primary extension is always preferred. The alternate pickle extension is
    used only if no files are found for the primary extension.
    """
    files = sorted(glob.glob(f'{directory}/*.{extension}'))
    if files:
        return files

    if extension == 'pkl.gz':
        return sorted(glob.glob(f'{directory}/*.pkl'))
    if extension == 'pkl':
        return sorted(glob.glob(f'{directory}/*.pkl.gz'))

    return files


if __name__ == '__main__':
    pass
