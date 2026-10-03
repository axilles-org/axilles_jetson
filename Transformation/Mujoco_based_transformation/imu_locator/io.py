"""Loaders for mocap joint trajectories and raw IMU data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .estimate import ImuData
from .kinematics import TRANSLATIONS
from .signal_utils import fill_nans, make_uniform

G = 9.80665


def _read_table(path):
    """OpenSim .mot/.sto (header until 'endheader') or plain CSV/TSV."""
    lines = Path(path).read_text().splitlines()
    header = {}
    start = 0
    for i, ln in enumerate(lines[:200]):
        if ln.strip().lower() == "endheader":
            start = i + 1
            for h in lines[:i]:
                if "=" in h:
                    k, v = h.split("=", 1)
                    header[k.strip().lower()] = v.strip().lower()
            break
    sep = "\t" if "\t" in lines[start] else ","
    cols = [c.strip() for c in lines[start].split(sep)]
    data = np.genfromtxt(lines[start + 1:], delimiter=sep, dtype=float)
    if data.ndim == 1:
        data = data[None]
    return header, cols, data


def load_coordinates(path, time_col="time", in_degrees=None, coord_map=None,
                     auto_knee_sign=True, verbose=True):
    """Joint coordinates -> (t, {model_joint: series in rad / m}).

    coord_map (dict or JSON path): {"model_joint": {"col": "...", "scale": 1, "offset": 0}}
    for datasets whose column names / signs differ from the OpenSim ones.
    Angles are in degrees if the .mot header says inDegrees=yes, if in_degrees=True,
    or (auto) if any rotational column exceeds 2*pi in magnitude.
    """
    header, cols, data = _read_table(path)
    tcol = next((i for i, c in enumerate(cols) if c.lower() == time_col.lower()), 0)
    t = data[:, tcol]
    raw = {c: fill_nans(data[:, i]) for i, c in enumerate(cols) if i != tcol}

    if isinstance(coord_map, (str, Path)):
        coord_map = json.loads(Path(coord_map).read_text())
    if coord_map:
        mapped = {}
        for joint, spec in coord_map.items():
            spec = {"col": spec} if isinstance(spec, str) else spec
            mapped[joint] = raw[spec["col"]] * spec.get("scale", 1.0) + spec.get("offset", 0.0)
        raw = mapped

    if in_degrees is None:
        if "indegrees" in header:
            in_degrees = header["indegrees"].startswith("y")
        else:
            rot = [np.nanmax(np.abs(v)) for k, v in raw.items() if k not in TRANSLATIONS]
            in_degrees = bool(rot) and max(rot) > 2 * np.pi
    coords = {}
    for k, v in raw.items():
        if k in TRANSLATIONS:
            coords[k] = v
        else:
            coords[k] = np.unwrap(np.radians(v) if in_degrees else v)

    if auto_knee_sign:
        for side in "rl":
            k = f"knee_angle_{side}"
            if k in coords and np.median(coords[k]) < -np.radians(3):
                coords[k] = -coords[k]
                if verbose:
                    print(f"[io] {k}: flexion was negative (gait2392 convention) -> sign flipped")
    t, arr = make_uniform(t, np.column_stack(list(coords.values())))
    coords = dict(zip(coords.keys(), arr.T))
    if verbose:
        print(f"[io] {len(t)} mocap frames @ {1 / np.median(np.diff(t)):.1f} Hz, "
              f"angles in {'deg' if in_degrees else 'rad'}")
    return t, coords


def load_imu_csv(path, time_col="time", acc_cols=("acc_x", "acc_y", "acc_z"),
                 gyr_cols=("gyr_x", "gyr_y", "gyr_z"), acc_unit="auto", gyr_unit="auto",
                 time_unit=1.0, verbose=True) -> ImuData:
    """Raw IMU CSV. acc_unit: 'm/s2' | 'g' | 'auto';  gyr_unit: 'rad/s' | 'deg/s' | 'auto'.
    time_unit multiplies the time column (e.g. 1e-3 for milliseconds)."""
    _, cols, data = _read_table(path)
    ix = {c.lower(): i for i, c in enumerate(cols)}
    get = lambda names: np.column_stack([fill_nans(data[:, ix[n.lower()]]) for n in names])
    t = data[:, ix[time_col.lower()]] * time_unit
    acc, gyr = get(acc_cols), get(gyr_cols)
    if acc_unit == "auto":
        acc_unit = "g" if np.median(np.linalg.norm(acc, axis=1)) < 3 else "m/s2"
    if gyr_unit == "auto":
        gyr_unit = "deg/s" if np.percentile(np.abs(gyr), 99) > 20 else "rad/s"
    if acc_unit == "g":
        acc = acc * G
    if gyr_unit == "deg/s":
        gyr = np.radians(gyr)
    t, both = make_uniform(t, np.hstack([acc, gyr]))
    if verbose:
        print(f"[io] IMU {len(t)} samples @ {1 / np.median(np.diff(t)):.1f} Hz "
              f"(acc {acc_unit}, gyro {gyr_unit})")
    return ImuData(t=t, acc=both[:, :3], gyr=both[:, 3:])
