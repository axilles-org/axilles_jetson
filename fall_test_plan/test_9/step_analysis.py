#!/usr/bin/env python3
"""
step_analysis.py: steps walked, walking speed and motor power from dashboard runs.

    python3 fall_test_plan/test_9/step_analysis.py dashboard/data/runs/20261104_200639_f6931d
    python3 fall_test_plan/test_9/step_analysis.py dashboard/data/runs             # every run inside
    python3 fall_test_plan/test_9/step_analysis.py 'dashboard/data/runs/20261104_*' # quoted glob
    python3 fall_test_plan/test_9/step_analysis.py dashboard/data/runs/<run>/part-00003.parquet
    python3 fall_test_plan/test_9/step_analysis.py <path> --transmission-ratio 3.8 --phase-resistance 0.1

Analysis only. It reads the Parquet files the dashboard RunLogger wrote and
never opens the CAN bus or the I2C bus, so it cannot move the motor.

Paths are resolved relative to the git workspace (exo_ws), not the shell's
current directory, so the commands above work from anywhere. Absolute paths
also work. Each path can be:
    - one .parquet file           -> analysed on its own (one part is ~5 s)
    - a run folder of part-*.parquet files -> one run
    - a folder of run folders      -> every run inside, one row each
    - a glob (quote it so the shell does not expand it against your cwd)

What it reports per run
    Steps      Heel strikes are the 0->1 edges of the logged `stance` flag (or,
               for the older TBE runs without `stance`, the FSR thresholds the ML
               SensorAdapter uses). A heel strike closer than --min-stride-s to
               the previous one is ignored, same idea as the TBE double heel-strike
               fix. Only the exo leg is instrumented, so the logs show strides of
               that leg; steps = 2 x strides is an estimate that assumes the other
               leg stepped in between. Intervals longer than --max-stride-s are
               counted as pauses, not strides.
    Speed      Not logged. Estimated per stride from the foot IMU with zero-velocity
               updates (ZUPT): the foot is still during mid-stance, so accel is
               rotated to the world frame with the integrated gyro, integrated to
               velocity between two still periods, de-drifted so both ends are
               zero, and integrated again to stride length. On a treadmill the
               belt is an inertial frame and the foot is still on it during
               stance, so stride length / stride time estimates the belt speed.
               Strides with a frozen IMU or an implausible length are dropped.
    Power      From the CAN status data logged in the run (`motor_iq_a`):
               mean |Iq| and RMS Iq, and the motor torque Kt*|Iq| (sign taken
               from the command because the reported Iq sign is unreliable).
               Mechanical power = torque x motor speed, where motor speed =
               --transmission-ratio x ankle angular velocity from the encoder.
               Motor speed itself is not in these logs, and the motor-to-ankle
               ratio is not recorded in the repo, so the default ratio of 1.0 is
               an assumption: mechanical power scales directly with it.
               Copper loss 1.5*R*Iq^2 is added only if --phase-resistance is given.
               Bus voltage and supply current are not logged, so battery power
               cannot be measured from these files.

Writes summary.csv, strides.csv and report.json into
fall_test_plan/test_9/results/<date_time>/ (or --out), unless --no-save.
"""

import argparse
import glob
import json
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent

G = 9.80665
KT_NM_PER_A = 0.7044          # AK80-9 torque constant (ML_model/src/exo/deploy/motor.py)

# Stance from FSRs, only for runs that did not log `stance` (TBE runs). Same
# values as ML_model/configs/default.yaml deploy.sensor_adapter.
HEEL_FSR_THRESHOLD = 22446.0
TOE_FSR_THRESHOLD = 8737.2
FSR_DEBOUNCE_S = 0.10

# ZUPT still-foot detection.
STILL_GYRO_RAD_S = 0.8        # |gyro| below this ...
STILL_ACCEL_DEV = 1.5         # ... and | |a| - g | below this (m/s^2) ...
STILL_MIN_S = 0.03            # ... for at least this long
FROZEN_IMU_S = 0.15           # identical foot accel+gyro this long = IMU frozen
STRIDE_LEN_RANGE_M = (0.2, 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Finding and loading runs
# ─────────────────────────────────────────────────────────────────────────────

def workspace_root() -> Path:
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=TEST_DIR,
                             capture_output=True, text=True, check=True)
        return Path(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return TEST_DIR.parents[1]


def resolve_inputs(args_paths, root: Path):
    """Return a list of (name, [parquet files]) recordings, in order."""
    recordings = []
    for raw in args_paths:
        p = Path(raw)
        pattern = str(p if p.is_absolute() else root / p)
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not matches:
            print(f"warning: nothing matches {raw}", file=sys.stderr)
        for m in map(Path, matches):
            if m.is_file():
                if m.suffix == ".parquet":
                    recordings.append((f"{m.parent.name}/{m.name}", [m]))
                else:
                    print(f"warning: skipping non-parquet file {m}", file=sys.stderr)
            elif m.is_dir():
                parts = sorted(m.glob("*.parquet"))
                if parts:
                    recordings.append((m.name, parts))
                    continue
                subdirs = sorted(d for d in m.iterdir() if d.is_dir())
                if not subdirs:          # an empty run folder (run never logged)
                    recordings.append((m.name, []))
                for d in subdirs:
                    recordings.append((d.name, sorted(d.glob("*.parquet"))))
            else:
                print(f"warning: {m} does not exist", file=sys.stderr)
    return recordings


def run_meta(root: Path, run_id: str) -> dict:
    """Controller/model settings the dashboard stored for this run, if any."""
    db = root / "dashboard" / "data" / "runs.db"
    if not db.exists():
        return {}
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            row = con.execute("select status, meta_json from runs where run_id = ?",
                              (run_id,)).fetchone()
    except sqlite3.Error:
        return {}
    if not row:
        return {}
    meta = json.loads(row[1] or "{}")
    return {
        "status": row[0],
        "controller": meta.get("controller", "TBE" if "peak_torque" in meta else ""),
        "model": Path(meta["run_dir"]).name if meta.get("run_dir") else "",
        "assist_delay_ms": meta.get("assistance_delay_ms"),
        "torque_cap_nm": meta.get("torque_cap_nm", meta.get("peak_torque")),
        "subject_mass_kg": meta.get("subject_mass_kg"),
    }


def load(parts) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    tcol = "t_s" if "t_s" in df else "t"
    df = df.sort_values(tcol).reset_index(drop=True)
    df["time_s"] = df[tcol] - df[tcol].iloc[0]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Steps
# ─────────────────────────────────────────────────────────────────────────────

def stance_from_fsr(t, heel, toe):
    """Debounced stance flag from FSR thresholds (for runs without `stance`)."""
    raw = (heel > HEEL_FSR_THRESHOLD) | (toe >= TOE_FSR_THRESHOLD)
    out = np.zeros(len(raw), dtype=int)
    state, since = 0, t[0] if len(t) else 0.0
    for i in range(len(raw)):
        if raw[i] != state:
            if t[i] - since >= FSR_DEBOUNCE_S:
                state, since = int(raw[i]), t[i]
        else:
            since = t[i]
        out[i] = state
    return out


def heel_strikes(t, stance, min_stride_s):
    rises = np.flatnonzero(np.diff(stance.astype(int)) == 1) + 1
    kept = []
    for i in rises:
        if not kept or t[i] - t[kept[-1]] >= min_stride_s:
            kept.append(i)
    return np.array(kept, dtype=int), len(rises) - len(kept)


# ─────────────────────────────────────────────────────────────────────────────
# Speed (ZUPT)
# ─────────────────────────────────────────────────────────────────────────────

def _runs(mask):
    """(start, end_exclusive) of each True run."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _rot_to_z(v):
    """Rotation matrix taking unit vector v onto +z."""
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(v, z))
    axis = np.cross(v, z)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def _expm_so3(w, dt):
    theta = np.linalg.norm(w) * dt
    if theta < 1e-12:
        return np.eye(3)
    k = w / np.linalg.norm(w)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def frozen_mask(x, t):
    """True where every channel of x is unchanged for longer than FROZEN_IMU_S."""
    same = np.concatenate([[False], np.all(np.diff(x, axis=0) == 0, axis=1)])
    out = np.zeros(len(x), dtype=bool)
    for s, e in _runs(same):
        if t[min(e, len(t) - 1)] - t[max(s - 1, 0)] > FROZEN_IMU_S:
            out[max(s - 1, 0):e] = True
    return out


def zupt_strides(t, acc, gyr, stance, hs, min_stride_s, max_stride_s):
    """Per stride between consecutive heel strikes: length (m) and speed (m/s), or NaN."""
    n = len(t)
    a_norm = np.linalg.norm(acc, axis=1)
    on_ground = stance == 1
    g_meas = np.median(a_norm[on_ground]) if on_ground.any() else np.median(a_norm)
    if not np.isfinite(g_meas) or g_meas <= 0:
        return [], np.nan
    acc = acc * (G / g_meas)                 # per-run accel scale correction
    a_norm = a_norm * (G / g_meas)
    w_norm = pd.Series(np.linalg.norm(gyr, axis=1)).rolling(3, center=True, min_periods=1).mean().values
    still = (w_norm < STILL_GYRO_RAD_S) & (np.abs(a_norm - G) < STILL_ACCEL_DEV) & on_ground
    frozen = frozen_mask(np.hstack([acc, gyr]), t)

    # One ZUPT anchor per heel strike: the longest still run before the next heel strike.
    anchors = []
    for k, h in enumerate(hs):
        end = hs[k + 1] if k + 1 < len(hs) else n
        best = None
        for s, e in _runs(still[h:end]):
            s, e = s + h, e + h
            if t[e - 1] - t[s] >= STILL_MIN_S and (best is None or e - s > best[1] - best[0]):
                best = (s, e)
        anchors.append(best)

    results = []
    for k in range(len(hs) - 1):
        period = t[hs[k + 1]] - t[hs[k]]
        row = {"length_m": np.nan, "speed_m_s": np.nan, "reason": ""}
        a0, a1 = anchors[k], anchors[k + 1]
        if not (min_stride_s <= period <= max_stride_s):
            row["reason"] = "pause"
        elif a0 is None or a1 is None:
            row["reason"] = "no still foot"
        elif frozen[a0[0]:a1[1]].any():
            row["reason"] = "IMU frozen"
        else:
            i0, i1 = a0[1] - 1, a1[0]        # integrate from end of still to start of next still
            R = _rot_to_z(acc[a0[0]:a0[1]].mean(axis=0))
            v = np.zeros(3)
            vel = [v.copy()]
            for i in range(i0, i1):
                dt = t[i + 1] - t[i]
                aw = R @ acc[i] - np.array([0.0, 0.0, G])
                v = v + aw * dt
                vel.append(v.copy())
                R = R @ _expm_so3(gyr[i], dt)
            vel = np.array(vel)
            tt = t[i0:i1 + 1]
            span = tt[-1] - tt[0]
            if span <= 0:
                row["reason"] = "no still foot"
            else:
                vel -= np.outer((tt - tt[0]) / span, vel[-1])     # end velocity must be zero
                pos = np.concatenate([[np.zeros(3)],
                                      np.cumsum(0.5 * (vel[1:] + vel[:-1]) * np.diff(tt)[:, None], axis=0)])
                length = float(np.linalg.norm(pos[-1, :2]))
                anchor_dt = t[a1[0]] - t[a0[0]]
                if STRIDE_LEN_RANGE_M[0] <= length <= STRIDE_LEN_RANGE_M[1] and anchor_dt > 0:
                    row["length_m"] = length
                    row["speed_m_s"] = length / anchor_dt
                else:
                    row["reason"] = f"implausible length {length:.2f} m"
        results.append(row)
    return results, g_meas


# ─────────────────────────────────────────────────────────────────────────────
# Power
# ─────────────────────────────────────────────────────────────────────────────

def power_stats(df, ratio, phase_r, mask=None):
    """Motor current, torque and power over the rows in mask (all rows if None)."""
    t = df["time_s"].values
    if "ankle_angle_deg" in df:
        ang = pd.Series(df["ankle_angle_deg"].values).rolling(5, center=True, min_periods=1).mean().values
        w_ankle = np.radians(np.gradient(ang, t))
        tq_cmd = df["sent_torque_nm"].values
    elif "ankle_velocity" in df:
        w_ankle = np.radians(df["ankle_velocity"].values)       # TBE logs deg/s
        tq_cmd = df["torque_cmd"].values
    else:
        return {}
    has_iq = "motor_iq_a" in df
    if has_iq:
        iq = np.abs(df["motor_iq_a"].values)
        tq = np.sign(tq_cmd) * KT_NM_PER_A * iq
    else:
        iq, tq = None, tq_cmd
    p_mech = tq * ratio * w_ankle
    m = np.ones(len(df), dtype=bool) if mask is None else mask
    m &= np.isfinite(p_mech)
    if not m.any():
        return {}
    out = {
        "power_source": "CAN Iq" if has_iq else "commanded torque (no CAN data)",
        "mean_abs_cmd_nm": float(np.mean(np.abs(tq_cmd[m]))),
        "mean_abs_torque_nm": float(np.mean(np.abs(tq[m]))),
        "mech_power_net_w": float(np.mean(p_mech[m])),
        "mech_power_abs_w": float(np.mean(np.abs(p_mech[m]))),
        "mech_power_pos_w": float(np.mean(np.clip(p_mech[m], 0, None))),
        "mech_power_peak_w": float(np.max(np.abs(p_mech[m]))),
    }
    if has_iq:
        out["mean_abs_iq_a"] = float(np.mean(iq[m]))
        out["rms_iq_a"] = float(np.sqrt(np.mean(iq[m] ** 2)))
        out["peak_iq_a"] = float(np.max(iq[m]))
        if phase_r is not None:
            p_cu = 1.5 * phase_r * iq[m] ** 2
            out["copper_loss_w"] = float(np.mean(p_cu))
            out["elec_power_est_w"] = float(np.mean(p_mech[m] + p_cu))
    dur = float(np.sum(np.diff(t, append=t[-1])[m]))
    out["energy_mech_abs_j"] = out["mech_power_abs_w"] * dur
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-run analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse(name, parts, root, args):
    row = {"run": name}
    row.update(run_meta(root, name.split("/")[0]))
    if not parts:
        row["note"] = "no data files"
        return row, []
    df = load(parts)
    t = df["time_s"].values
    row["samples"] = len(df)
    row["duration_s"] = float(t[-1]) if len(t) else 0.0
    row["rate_hz"] = float(1.0 / np.median(np.diff(t))) if len(t) > 1 else np.nan
    if len(df) < 50:
        row["note"] = "too short"
        return row, []

    if "stance" in df:
        stance = df["stance"].values.astype(int)
        row["stance_source"] = "logged stance"
    elif {"heel_fsr", "toe_fsr"} <= set(df.columns):
        stance = stance_from_fsr(t, df["heel_fsr"].values, df["toe_fsr"].values)
        row["stance_source"] = "FSR thresholds"
    else:
        row["note"] = "no stance or FSR data"
        return row, []

    hs, rejected = heel_strikes(t, stance, args.min_stride_s)
    periods = np.diff(t[hs]) if len(hs) > 1 else np.array([])
    walking = (periods >= args.min_stride_s) & (periods <= args.max_stride_s)
    row["heel_strikes"] = int(len(hs))
    row["rejected_double_hs"] = int(rejected)
    row["strides"] = int(walking.sum())
    row["steps_est"] = 2 * row["strides"]
    row["pauses"] = int((~walking).sum())
    row["walking_time_s"] = float(periods[walking].sum())
    if walking.any():
        st = np.median(periods[walking])
        row["stride_time_s"] = float(st)
        row["cadence_steps_min"] = float(120.0 / st)

    strides = []
    imu_cols = [f"foot_{a}" for a in ("ax", "ay", "az", "gx", "gy", "gz")]
    if set(imu_cols) <= set(df.columns) and len(hs) > 1:
        imu = df[imu_cols].values.astype(float)
        row["imu_frozen_frac"] = float(frozen_mask(imu, t).mean())
        res, g_meas = zupt_strides(t, imu[:, :3], imu[:, 3:], stance, hs,
                                   args.min_stride_s, args.max_stride_s)
        row["foot_g_measured"] = float(g_meas)
        for k, r in enumerate(res):
            strides.append({"run": name, "stride": k, "t_start_s": float(t[hs[k]]),
                            "period_s": float(periods[k]), **r})
        lengths = np.array([r["length_m"] for r in res], dtype=float)
        speeds = np.array([r["speed_m_s"] for r in res], dtype=float)
        ok = np.isfinite(speeds)
        row["speed_strides"] = int(ok.sum())
        if ok.any():
            row["speed_median_m_s"] = float(np.median(speeds[ok]))
            row["speed_mean_m_s"] = float(np.mean(speeds[ok]))
            row["stride_len_median_m"] = float(np.median(lengths[ok]))
            # distance: measured strides, plus median length for walking strides without an estimate
            row["distance_est_m"] = float(lengths[ok].sum()
                                          + (row["strides"] - ok.sum()) * np.median(lengths[ok]))
    else:
        for k, p in enumerate(periods):
            strides.append({"run": name, "stride": k, "t_start_s": float(t[hs[k]]),
                            "period_s": float(p), "length_m": np.nan, "speed_m_s": np.nan,
                            "reason": "pause" if not walking[k] else "no foot IMU"})

    for key, val in power_stats(df, args.transmission_ratio, args.phase_resistance).items():
        row[key] = val
    if "assist_on" in df:
        on = df["assist_on"].values.astype(bool)
        row["assist_on_frac"] = float(on.mean())
        p_on = power_stats(df, args.transmission_ratio, args.phase_resistance, on.copy())
        for key in ("mean_abs_iq_a", "mech_power_abs_w", "mech_power_net_w", "elec_power_est_w"):
            if key in p_on:
                row[f"{key}_assist_on"] = p_on[key]
    return row, strides


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def print_report(rows, args):
    cols = [("run", "run", 0), ("model", "model", 0), ("duration_s", "dur s", 0),
            ("strides", "strides", 0), ("steps_est", "steps~", 0), ("cadence_steps_min", "cad/min", 0),
            ("speed_median_m_s", "speed m/s", 2), ("speed_strides", "n spd", 0),
            ("distance_est_m", "dist m", 0), ("mean_abs_iq_a", "|Iq| A", 2),
            ("mech_power_abs_w", "|P| W", 1), ("mech_power_net_w", "P net W", 1)]
    if args.phase_resistance is not None:
        cols.append(("elec_power_est_w", "Pelec W", 1))
    table = [[h for _, h, _ in cols]]
    for r in rows:
        table.append([fmt(r.get(k), nd) if k != "model" else (r.get(k) or "")[-28:] for k, _, nd in cols])
    widths = [max(len(x[i]) for x in table) for i in range(len(cols))]
    for i, line in enumerate(table):
        print("  ".join(c.rjust(w) if j else c.ljust(w) for j, (c, w) in enumerate(zip(line, widths))))
        if i == 0:
            print("  ".join("-" * w for w in widths))
    notes = [(r["run"], r["note"]) for r in rows if r.get("note")]
    for run, note in notes:
        print(f"  {run}: {note}")

    valid = [r for r in rows if "strides" in r]
    if len(valid) > 1:
        strides = sum(r["strides"] for r in valid)
        walk = sum(r.get("walking_time_s", 0.0) for r in valid)
        dist = sum(r.get("distance_est_m", 0.0) for r in valid)
        print(f"\nTotal: {strides} strides (~{2 * strides} steps), {walk:.0f} s walking, "
              f"~{dist:.0f} m estimated distance over {len(valid)} runs")
    print(f"\nSteps~ = 2 x strides of the exo leg. Speed from foot-IMU ZUPT, per stride.")
    print(f"Power uses transmission ratio {args.transmission_ratio:g} (motor speed / ankle speed); "
          f"P scales with it." + ("" if args.phase_resistance is not None else
                                  " Pass --phase-resistance to add copper loss."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run folder, folder of runs, .parquet file or quoted glob, "
                                             "relative to the git workspace")
    ap.add_argument("--min-stride-s", type=float, default=0.5,
                    help="ignore a heel strike this soon after the previous one (default 0.5)")
    ap.add_argument("--max-stride-s", type=float, default=2.5,
                    help="longer heel-strike intervals are pauses, not strides (default 2.5)")
    ap.add_argument("--transmission-ratio", type=float, default=1.0,
                    help="motor output speed / ankle speed (default 1.0, an assumption)")
    ap.add_argument("--phase-resistance", type=float, default=None,
                    help="motor phase resistance in ohm; adds copper loss 1.5*R*Iq^2")
    ap.add_argument("--out", type=Path, default=None, help="results folder (default results/<date_time>)")
    ap.add_argument("--no-save", action="store_true", help="print only, write no files")
    args = ap.parse_args()

    root = workspace_root()
    recordings = resolve_inputs(args.paths, root)
    if not recordings:
        sys.exit("no recordings found")

    rows, all_strides = [], []
    for name, parts in recordings:
        try:
            row, strides = analyse(name, parts, root, args)
        except Exception as exc:                       # one bad run should not stop the rest
            row, strides = {"run": name, "note": f"error: {exc}"}, []
        rows.append(row)
        all_strides.extend(strides)

    print_report(rows, args)

    if not args.no_save:
        out = args.out or TEST_DIR / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out if out.is_absolute() else root / out
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
        pd.DataFrame(all_strides).to_csv(out / "strides.csv", index=False)
        with open(out / "report.json", "w") as f:
            json.dump({"inputs": args.paths, "settings": {k: v for k, v in vars(args).items()
                                                          if k not in ("paths", "out")},
                       "runs": rows}, f, indent=2, default=lambda o: None)
        print(f"\nSaved {out.relative_to(root) if out.is_relative_to(root) else out}/")


if __name__ == "__main__":
    main()
