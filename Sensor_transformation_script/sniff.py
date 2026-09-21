"""
sniff.py
========
Work out which columns of an exo log are what, without you having to write a
mapping by hand.

Handles the usual header styles:
    foot_ax / foot_gx            (prefixed)
    ax_foot / gx_foot            (suffixed)
    FootAccelX / FootGyroX       (camel)
    foot_accel_x / foot_acc_x    (spelled out)

Import `sniff_columns` or run this file directly on a log to see the mapping:

    python sniff.py mylog.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENT_WORDS = {
    "foot": ("foot", "ft"),
    "shank": ("shank", "tibia", "shin", "lowerleg", "lower_leg", "calf"),
    "thigh": ("thigh", "femur", "upperleg", "upper_leg"),
    "trunk": ("trunk", "torso", "pelvis", "back"),
}
ACCEL_WORDS = ("accel", "acc", "a")
QUAT_SUFFIXES = (("qi", "qj", "qk", "qr"),      # BNO / Bosch i,j,k,real
                 ("qx", "qy", "qz", "qw"),
                 ("quat_x", "quat_y", "quat_z", "quat_w"),
                 ("q1", "q2", "q3", "q0"))
GYRO_WORDS = ("gyro", "gyr", "g")
AXES = ("x", "y", "z")

TIME_NAMES = ("time_s", "time", "timestamp", "timestamp_s", "t", "secs",
              "seconds", "elapsed", "elapsed_s")
TIME_MS_NAMES = ("time_ms", "timestamp_ms", "millis", "ms", "elapsed_ms")
TIME_US_NAMES = ("time_us", "timestamp_us", "micros", "us")

ENCODER_NAMES = ("ankle_encoder_deg", "ankle_encoder", "encoder_deg",
                 "encoder", "ankle_angle_deg", "ankle_deg", "joint_angle_deg")
HEEL_NAMES = ("heel_fsr_raw", "heel_fsr", "fsr_heel", "heel", "fsr0", "heelfsr")
TOE_NAMES = ("toe_fsr_raw", "toe_fsr", "fsr_toe", "toe", "fsr1", "toefsr",
             "forefoot_fsr", "fore_fsr")


def _norm(s):
    """lowercase, split camelCase, collapse separators."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(s))
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _pick(cols, names):
    norm = {_norm(c): c for c in cols}
    for n in names:                       # exact normalised match
        if n in norm:
            return norm[n]
    for n in names:                       # substring fallback
        if len(n) < 4:                    # 't' matches 'foot_ax'; don't
            continue
        for k, orig in norm.items():
            if n in k:
                return orig
    for n in names:                       # short names: whole-token match only
        for k, orig in norm.items():
            if n in k.split("_"):
                return orig
    return None


def _find_triad(cols, segment, kind):
    """
    Find the x/y/z columns for one segment and one sensor kind.
    Returns a list of three original column names, or None.
    """
    seg_words = SEGMENT_WORDS[segment]
    kind_words = ACCEL_WORDS if kind == "accel" else GYRO_WORDS
    other_words = GYRO_WORDS if kind == "accel" else ACCEL_WORDS
    norm = {_norm(c): c for c in cols}

    out = []
    for ax in AXES:
        hit = None
        for k, orig in norm.items():
            toks = k.split("_")
            if not any(w in toks or w in k for w in seg_words):
                continue
            # the axis letter must appear as its own token or glued to the kind
            kind_hit = None
            for kw in kind_words:
                if kw in toks or f"{kw}{ax}" in toks or f"{ax}{kw}" in toks:
                    kind_hit = kw
                    break
            if kind_hit is None:
                continue
            # reject if the opposite kind is a better match (e.g. 'g' vs 'gyro')
            if any(ow in toks and len(ow) > len(kind_hit) for ow in other_words):
                continue
            if not (ax in toks or f"{kind_hit}{ax}" in toks
                    or f"{ax}{kind_hit}" in toks or k.endswith(ax)):
                continue
            hit = orig
            break
        if hit is None:
            return None
        out.append(hit)
    return out if len(set(out)) == 3 else None


def sniff_time(df):
    """Return (column_name, scale_to_seconds) or (None, None)."""
    for names, scale in ((TIME_NAMES, 1.0), (TIME_MS_NAMES, 1e-3),
                         (TIME_US_NAMES, 1e-6)):
        c = _pick(df.columns, names)
        if c is not None:
            return c, scale
    # last resort: a monotonically increasing numeric column
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy()
        if np.all(np.isfinite(v)) and np.all(np.diff(v) > 0):
            span = v[-1] - v[0]
            if 1 < span < 1e4:
                return c, 1.0
            if 1e3 < span < 1e7:
                return c, 1e-3
    return None, None


def _find_quat(cols, segment):
    """
    Find a segment's orientation quaternion, returned in scipy order
    [x, y, z, w]. Handles i/j/k/r (scalar LAST in our output regardless of
    how the sensor names it).
    """
    seg_words = SEGMENT_WORDS[segment]
    norm = {_norm(c): c for c in cols}
    for suf in QUAT_SUFFIXES:
        hit = []
        for s_ in suf:
            found = None
            for k, orig in norm.items():
                toks = k.split("_")
                if not any(w in toks or w in k for w in seg_words):
                    continue
                if s_ in toks or k.endswith(s_):
                    found = orig
                    break
            if found is None:
                break
            hit.append(found)
        if len(hit) == 4 and len(set(hit)) == 4:
            return hit          # already ordered x,y,z,w by QUAT_SUFFIXES
    return None


def sniff_columns(df, segments=("foot", "shank"), verbose=True):
    """Build a mapping dict for `align_io.load_exo`."""
    m = {}
    tcol, tscale = sniff_time(df)
    m["time"], m["time_scale"] = tcol, tscale

    for seg in segments:
        a = _find_triad(df.columns, seg, "accel")
        g = _find_triad(df.columns, seg, "gyro")
        if a and g:
            m[seg] = dict(accel=a, gyro=g, prefix="")
            q = _find_quat(df.columns, seg)
            if q:
                m[seg]["quat"] = q

    m["encoder"] = _pick(df.columns, ENCODER_NAMES)
    m["fsr_heel"] = _pick(df.columns, HEEL_NAMES)
    m["fsr_toe"] = _pick(df.columns, TOE_NAMES)

    if verbose:
        report(df, m, segments)
    return m


def report(df, m, segments=("foot", "shank")):
    print("  column mapping")
    t = m.get("time")
    print(f"    time        : {t}"
          + (f"  (x{m['time_scale']:g} -> s)" if t and m.get("time_scale") != 1 else "")
          + ("   <-- NOT FOUND, pass --fs" if t is None else ""))
    for seg in segments:
        if seg in m:
            print(f"    {seg:<11} : accel {m[seg]['accel']}")
            print(f"    {'':<11}   gyro  {m[seg]['gyro']}")
            if m[seg].get("quat"):
                print(f"    {'':<11}   quat  {m[seg]['quat']}  (x,y,z,w)")
        else:
            print(f"    {seg:<11} : NOT FOUND")
    for k in ("encoder", "fsr_heel", "fsr_toe"):
        print(f"    {k:<11} : {m.get(k)}")
    used = {c for seg in segments if seg in m
            for c in m[seg]["accel"] + m[seg]["gyro"] + m[seg].get("quat", [])}
    used |= {m.get(k) for k in ("time", "encoder", "fsr_heel", "fsr_toe")}
    unused = [c for c in df.columns if c not in used]
    if unused:
        print(f"    unmapped    : {unused}")


def load_any(path, nrows=None):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, nrows=nrows)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("usage: python sniff.py <logfile>")
    df = load_any(sys.argv[1], nrows=5000)
    print(f"{sys.argv[1]}: {len(df)} rows (preview), {len(df.columns)} columns\n")
    print("  all columns:")
    for c in df.columns:
        print(f"    {c!r:<28} {str(df[c].dtype):<10} "
              f"first={df[c].iloc[0]!r}")
    print()
    m = sniff_columns(df)
    t = m.get("time")
    if t:
        v = df[t].to_numpy(float) * m["time_scale"]
        d = np.diff(v)
        print(f"\n  implied rate: {1/np.median(d):.1f} Hz "
              f"(dt median {np.median(d)*1000:.2f} ms, "
              f"jitter sd {np.std(d)*1000:.2f} ms)")