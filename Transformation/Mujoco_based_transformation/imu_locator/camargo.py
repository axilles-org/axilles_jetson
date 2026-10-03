"""Loader for the Camargo et al. 2021 (Georgia Tech) lower-limb dataset layout:

  <subject>/<date>/<mode>/{ik,imu,markers,conditions,...}/<trial>.mat   (MATLAB tables)
  <subject>/osimxml/<subject>.osim

IK in degrees (OpenSim names), IMU accel in g and gyro in rad/s (right side:
foot, shank, thigh, trunk), markers in mm (OpenSim ground frame, Y-up), 200 Hz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .estimate import ImuData
from .kinematics import TRANSLATIONS
from .signal_utils import fill_nans

G = 9.80665
IMU_NAMES = ("foot", "shank", "thigh", "trunk")


def _table(path):
    from matio import load_from_mat   # pip install mat-io (reads MATLAB table objects)
    return load_from_mat(str(path))["data"]


def trial_files(date_dir, mode):
    return sorted(p.stem for p in (Path(date_dir) / mode / "imu").glob("*.mat"))


def load_ik(date_dir, mode, trial):
    d = _table(Path(date_dir) / mode / "ik" / f"{trial}.mat")
    t = d["Header"].to_numpy(float)
    coords = {}
    for c in d.columns[1:]:
        v = fill_nans(d[c].to_numpy(float))
        coords[c] = v if c in TRANSLATIONS else np.unwrap(np.radians(v))
    return t, coords


def load_imu(date_dir, mode, trial):
    d = _table(Path(date_dir) / mode / "imu" / f"{trial}.mat")
    t = d["Header"].to_numpy(float)
    out = {}
    for s in IMU_NAMES:
        acc = d[[f"{s}_Accel_{c}" for c in "XYZ"]].to_numpy(float) * G
        gyr = d[[f"{s}_Gyro_{c}" for c in "XYZ"]].to_numpy(float)
        out[s] = ImuData(t=t, acc=fill_nans(acc), gyr=fill_nans(gyr))
    return out


def load_markers(date_dir, mode, trial):
    d = _table(Path(date_dir) / mode / "markers" / f"{trial}.mat")
    t = d["Header"].to_numpy(float)
    names = sorted({c.rsplit("_", 1)[0] for c in d.columns[1:]})
    return t, {n: d[[f"{n}_{a}" for a in "xyz"]].to_numpy(float) / 1000.0 for n in names}


def crop(t, arrays, t0, t1):
    m = (t >= t0) & (t <= t1)
    return t[m], {k: v[m] for k, v in arrays.items()}
