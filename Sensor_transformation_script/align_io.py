"""
align_io.py
===========
Loaders that put both datasets into the same shape:

    Recording(fs, accel{segment}, gyro{segment}, events, meta)

with accelerometers in m/s^2, gyros in rad/s, gyro bias removed, and heel
strikes as sample indices.

Source side  : aicognition/mrsd-exo-ankle Parquet layout.
Target side  : your exo logs. Column names default to what your plot shows;
               override with `columns=` if they differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import align_core as ac


@dataclass
class Recording:
    fs: float
    accel: dict                      # segment -> (N,3) m/s^2
    gyro: dict                       # segment -> (N,3) rad/s
    quat: dict = field(default_factory=dict)     # segment -> (N,4) x,y,z,w
    events: dict = field(default_factory=dict)   # name -> sample indices
    meta: dict = field(default_factory=dict)

    def segments(self):
        return sorted(set(self.accel) & set(self.gyro))


# ---------------------------------------------------------------------------
# source: Hugging Face Parquet
# ---------------------------------------------------------------------------

GATECH_SEGMENTS = ("foot", "shank", "thigh", "trunk")


def load_gatech(root, subject, trial, segments=("foot", "shank"),
                accel_units=None, debias=True):
    """
    Load one trial from the mrsd-exo-ankle layout.

    accel_units : None auto-detects from the specific-force magnitude. The
                  dataset card says m/s^2 but the traces look like g, so the
                  default is to measure rather than trust the label.
    """
    root = Path(root)
    base = root / "subjects" / subject
    imu = pd.read_parquet(base / f"{trial}__imu.parquet")
    fs = 1.0 / float(np.median(np.diff(imu["time_s"].to_numpy())))

    units, mag = (accel_units, None) if accel_units else ac.detect_accel_units(
        imu[[f"trunk_Accel_{x}" for x in "XYZ"]].to_numpy()
        if "trunk_Accel_X" in imu else
        imu[[f"{segments[0]}_Accel_{x}" for x in "XYZ"]].to_numpy())

    accel, gyro = {}, {}
    for seg in segments:
        a = imu[[f"{seg}_Accel_{x}" for x in "XYZ"]].to_numpy(float)
        g = imu[[f"{seg}_Gyro_{x}" for x in "XYZ"]].to_numpy(float)
        accel[seg] = ac.to_si(a, units)
        gyro[seg] = g

    events = {}
    gc_path = base / f"{trial}__gcRight.parquet"
    if gc_path.exists():
        gc = pd.read_parquet(gc_path)
        hs_pct = gc["HeelStrike"].to_numpy(float)
        idx = np.flatnonzero(np.diff(hs_pct) < -50) + 1
        # gcRight shares the IMU rate (200 Hz), but guard anyway
        if len(gc) != len(imu):
            idx = np.round(idx * len(imu) / len(gc)).astype(int)
        events["heel_strike"] = idx

    if debias:
        for seg in segments:
            still = ac.detect_still(accel[seg], gyro[seg], fs)
            gyro[seg] = gyro[seg] - ac.estimate_gyro_bias(gyro[seg], still)

    meta = {"subject": subject, "trial": trial,
            "accel_units_detected": units, "accel_median_mag": mag}
    meta_path = root / "metadata.parquet"
    if meta_path.exists():
        m = pd.read_parquet(meta_path)
        row = m[(m.subject == subject) & (m.trial == trial)]
        if len(row):
            meta.update(row.iloc[0].to_dict())

    return Recording(fs=fs, accel=accel, gyro=gyro, events=events, meta=meta)


def fill_gaps(x, fs, max_gap_s=0.05):
    """
    Interpolate short NaN runs, report anything longer.

    Dropped samples are normal in a logger under load; a long gap is a
    different problem and you should see it rather than have it silently
    interpolated across.
    """
    x = np.asarray(x, float).copy()
    bad = ~np.isfinite(x).all(axis=1)
    if not bad.any():
        return x, 0.0, 0
    n_max = max(1, int(max_gap_s * fs))
    idx = np.arange(len(x))
    long_gaps = 0
    for a, b in runs_bool(bad):
        if b - a > n_max:
            long_gaps += 1
    good = ~bad
    if good.sum() < 10:
        raise ValueError("almost every sample is NaN; check the log")
    for j in range(x.shape[1]):
        x[bad, j] = np.interp(idx[bad], idx[good], x[good, j])
    return x, float(bad.mean()), long_gaps


def runs_bool(mask):
    m = np.asarray(mask, bool).astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def list_gatech_trials(root, speed_range=None):
    """All (subject, trial) pairs, optionally filtered by mean speed."""
    m = pd.read_parquet(Path(root) / "metadata.parquet")
    if speed_range is not None and "speed_mean_mps" in m:
        lo, hi = speed_range
        m = m[(m.speed_mean_mps >= lo) & (m.speed_mean_mps <= hi)]
    return list(zip(m.subject, m.trial))


# ---------------------------------------------------------------------------
# target: your exo logs
# ---------------------------------------------------------------------------

EXO_COLUMNS = {
    "time": "time_s",
    "foot": dict(accel=["ax", "ay", "az"], gyro=["gx", "gy", "gz"],
                 prefix="foot_"),
    "shank": dict(accel=["ax", "ay", "az"], gyro=["gx", "gy", "gz"],
                  prefix="shank_"),
    "encoder": "ankle_encoder_deg",
    "fsr_heel": "heel_fsr_raw",
    "fsr_toe": "toe_fsr_raw",
}


def load_exo(path, columns=None, accel_units=None, fs=None, debias=True,
             verbose=True):
    """
    Load one of your data_collection_*.csv (or .parquet) logs.

    Column names are sniffed automatically (see sniff.py). Pass `columns` to
    override the mapping, or `fs` if there is no usable time column.
    """
    import sniff

    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    cols = sniff.sniff_columns(df, verbose=verbose)
    if columns:
        cols.update(columns)

    tcol, tscale = cols.get("time"), cols.get("time_scale", 1.0)
    if fs is None:
        if tcol is None:
            raise ValueError(
                "no usable time column found. Columns present:\n  "
                + "\n  ".join(map(repr, df.columns))
                + "\n\nPass --fs <rate> (e.g. --fs 200), or name the column "
                  "explicitly with columns={'time': 'yourcol'}.")
        tv = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float) * tscale
        dt = np.diff(tv)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) == 0:
            raise ValueError(f"time column {tcol!r} is not usable; pass --fs")
        fs = 1.0 / float(np.median(dt))

    # decide units ONCE for the whole file, from the segment that spends the
    # most time stationary (that is where |a| is most reliably 1 g)
    raw, raw_q, nan_report = {}, {}, {}
    for seg in ("foot", "shank"):
        if seg not in cols:
            continue
        spec = cols[seg]
        p = spec.get("prefix", "")
        a = df[[p + c for c in spec["accel"]]].to_numpy(float)
        g = df[[p + c for c in spec["gyro"]]].to_numpy(float)
        fs_guess = fs if fs else 200.0
        a, fa, la = fill_gaps(a, fs_guess)
        g, fg, lg = fill_gaps(g, fs_guess)
        nan_report[seg] = dict(accel_frac=fa, gyro_frac=fg,
                               long_gaps=la + lg)
        raw[seg] = (a, g)
        if spec.get("quat"):
            q = df[[p + c for c in spec["quat"]]].to_numpy(float)
            q, fq, lq = fill_gaps(q, fs_guess)
            n = np.linalg.norm(q, axis=1, keepdims=True)
            raw_q[seg] = q / np.maximum(n, 1e-9)
            nan_report[seg]["quat_frac"] = fq
    if verbose:
        for seg, r in nan_report.items():
            if r["accel_frac"] or r["gyro_frac"]:
                print(f"    {seg:<11} : NaN filled -- accel "
                      f"{100*r['accel_frac']:.2f}%, gyro "
                      f"{100*r['gyro_frac']:.2f}%"
                      + (f", {r['long_gaps']} gap(s) >50 ms  <-- INVESTIGATE"
                         if r["long_gaps"] else ""))
    if not raw:
        raise ValueError(
            "no IMU columns recognised. Columns present:\n  "
            + "\n  ".join(map(repr, df.columns))
            + "\n\nSupply them with columns={'foot': {'accel': [...], "
              "'gyro': [...], 'prefix': ''}, ...}")

    units = accel_units
    if units is None:
        guesses = {s: ac.detect_accel_units(a) for s, (a, _) in raw.items()}
        for s in ("foot", "shank"):
            if s in guesses and guesses[s][0] != "unknown":
                units = guesses[s][0]
                break
        if units is None:
            mags = {s: round(v[1], 2) for s, v in guesses.items()}
            raise ValueError(
                f"could not determine accelerometer units. Median |a| per "
                f"segment: {mags} (expected ~1.0 for g or ~9.81 for m/s^2). "
                f"Pass accel_units='g' or 'm/s2'. A value far from either "
                f"usually means the axes are mislabelled or the log is "
                f"saturating.")
        if verbose:
            print(f"    accel units : {units} "
                  f"(median |a| per segment: "
                  f"{ {s: round(v[1], 2) for s, v in guesses.items()} })")

    accel, gyro = {}, {}
    for seg, (a, g) in raw.items():
        accel[seg] = ac.to_si(a, units)
        gyro[seg] = g

    events = {}
    if cols.get("fsr_heel") and cols.get("fsr_toe"):
        hraw = df[cols["fsr_heel"]].to_numpy(float)
        traw = df[cols["fsr_toe"]].to_numpy(float)
        nan_frac = (~np.isfinite(hraw)).mean() + (~np.isfinite(traw)).mean()
        events.update(fsr_events(hraw, traw, fs))
        if verbose:
            rail = (hraw >= 0.995 * np.nanmax(hraw)).mean()
            print(f"    FSR         : {len(events['heel_strike'])} heel strikes,"
                  f" {len(events['toe_off'])} toe offs"
                  + (f", {100*nan_frac:.1f}% NaN" if nan_frac else "")
                  + (f", heel railed {100*rail:.0f}% of samples  <-- "
                     f"clipping" if rail > 0.2 else ""))

    if debias:
        for seg in accel:
            still = ac.detect_still(accel[seg], gyro[seg], fs)
            gyro[seg] = gyro[seg] - ac.estimate_gyro_bias(gyro[seg], still)

    meta = {"file": path.name, "n_rows": len(df), "columns": cols,
            "nan": nan_report,
            "effective_odr_hz": effective_odr(accel[next(iter(accel))], fs)}
    if cols.get("encoder"):
        meta["encoder_deg"] = df[cols["encoder"]].to_numpy(float)
    return Recording(fs=fs, accel=accel, gyro=gyro, quat=raw_q,
                     events=events, meta=meta)


def effective_odr(accel, fs):
    """
    Fraction of samples that exactly repeat the previous one tells you the
    sensor's real output rate. Your current logs are stair-stepped, so run
    this on every file until the firmware fix lands.
    """
    rep = np.all(np.diff(np.asarray(accel, float), axis=0) == 0, axis=1)
    return float(fs * (1.0 - rep.mean()))


def fsr_events(heel, toe, fs, on_frac=0.25, off_frac=0.12):
    """Schmitt-trigger heel strike / toe off from raw FSR counts."""
    def gate(sig):
        s = np.asarray(sig, float)
        if not np.isfinite(s).any():
            return np.zeros(len(s), bool)
        s = np.nan_to_num(s, nan=np.nanmedian(s))
        scale = np.nanpercentile(s, 95)
        if scale <= 0:
            return np.zeros(len(s), bool)
        hi, lo = on_frac * scale, off_frac * scale
        st = np.zeros(len(s), bool)
        cur = s[0] > hi
        for i, v in enumerate(s):
            if not cur and v > hi:
                cur = True
            elif cur and v < lo:
                cur = False
            st[i] = cur
        return st

    h, t = gate(heel), gate(toe)
    return {"heel_strike": np.flatnonzero(np.diff(h.astype(int)) == 1) + 1,
            "toe_off": np.flatnonzero(np.diff(t.astype(int)) == -1) + 1,
            "heel_contact": h, "toe_contact": t}