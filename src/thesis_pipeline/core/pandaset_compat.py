from __future__ import annotations


def _sort_key(value):
    value_str = str(value)
    return (0, int(value_str)) if value_str.isdigit() else (1, value_str)


def _sensor_data(sensor_or_seq):
    if hasattr(sensor_or_seq, "lidar"):
        return sensor_or_seq.lidar.data
    return sensor_or_seq.data


def get_frame_count(seq) -> int:
    """Return the number of LiDAR frames for a PandaSet sequence.

    The local PandaSet devkit/runtime can expose ``seq.lidar.data`` as either a
    dict-like object keyed by frame index or a list-like container. Milestone B
    scripts must use this helper instead of calling ``len(seq.lidar.data)``
    directly so the counting logic stays compatible across both shapes.
    """
    data = _sensor_data(seq)
    if hasattr(data, "keys"):
        return len(list(data.keys()))
    return len(data)


def sorted_frame_keys(sensor) -> list:
    """Return stable frame keys for PandaSet devkit variants.

    Some local devkit/runtime combinations expose ``sensor.data`` as a dict-like
    object keyed by frame index, while others expose it as a list. The guide
    assumed the dict-like form, so this helper keeps downstream scripts safe in
    both cases.
    """
    data = _sensor_data(sensor)
    if hasattr(data, "keys"):
        raw_keys = list(data.keys())
    else:
        raw_keys = list(range(len(data)))
    return sorted(raw_keys, key=_sort_key)
