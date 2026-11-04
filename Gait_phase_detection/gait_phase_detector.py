#!/usr/bin/env python3
"""
gait_phase_detector.py — live stance / swing from the exo sensors, with heel
strike taken from the shank IMU and checked against the FSRs.

WHY
───
The torque gate opens when the heel FSR crosses a fixed threshold (22446). On
the 2026-11-04 treadmill runs that was ~70 ms after the shank stopped swinging
forward, and the heel FSR's swing level changes between sessions (about 6k in
swing on one run, about 22k on another). This script calibrates the FSR
thresholds and the IMU heel-strike point from a short walk, then prints the
phase it detects so it can be checked before anything is put in the deploy.

HOW
───
Calibration (the first --calib-s seconds; walk normally the whole time):
  1. The gyro axis with the most variance is the swing axis. Its sign is set so
     the mid-swing peak (the one with the FSRs unloaded) is positive.
  2. Mid-swing peaks give the stride time T.
  3. FSR levels: the heel/toe maximum around each mid-swing peak (swing) and
     the level in mid-stance (stance). Each threshold sits FSR_FRAC of the way
     from swing to stance.
  4. For each stride, heel strike by the FSRs (first loaded sample after mid-
     swing) is compared with the gyro zero crossing after mid-swing (end of
     swing). If the crossing came first on every stride it becomes the heel
     strike trigger; otherwise the FSRs are used alone.

Live:
  SWING  -> STANCE  gyro crosses zero after mid-swing (if calibration kept it),
                    or the FSRs load, whichever comes first. A gyro heel strike
                    that the FSRs do not confirm in time is withdrawn.
  STANCE -> SWING   both FSRs unloaded (toe-off).

Sensors are read by sensor_dropout_monitor.Reader (same bus traffic and rates
as the deploy, never opens CAN), which also re-initialises an IMU that has gone
silent for 1 s.  Do not run this alongside the deploy or another sensor script.

USAGE
    python3 gait_phase_detector.py                 # 3 s countdown, 5 s calibration
    python3 gait_phase_detector.py --calib-s 8 --imu foot
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── Detection constants ────────────────────────────────────────────────────────
FSR_FRAC         = 0.3       # threshold position between swing and stance level
MIN_CONTRAST     = 1500      # counts; less than this and an FSR is not used
MIN_SWING_PEAK   = 1.5       # rad/s; a smaller mid-swing peak is not walking
MIN_STRIDE_S     = 0.6       # peaks closer than this belong to the same swing
STALE_S          = 0.1       # a stream this quiet is reported and not used

STANCE, SWING = "STANCE", "SWING"


class CalibrationError(RuntimeError):
    pass


@dataclass
class Calibration:
    axis: int
    sign: float
    stride_s: float
    arm_level: float            # mid-swing must exceed this before a zero crossing counts
    heel_thr: float
    toe_thr: float
    heel_levels: tuple          # (swing, stance)
    toe_levels: tuple
    leads_s: list               # per stride: FSR heel strike minus gyro zero crossing
    use_gyro: bool
    confirm_s: float            # time the FSRs get to confirm a gyro heel strike
    warnings: list = field(default_factory=list)

    @property
    def min_phase_s(self):
        return 0.2 * self.stride_s


# ── Calibration ────────────────────────────────────────────────────────────────
def _find_peaks(t, x, min_height, min_gap_s):
    """Indices of local maxima above min_height, at least min_gap_s apart
    (the higher one wins)."""
    cand = [i for i in range(1, len(x) - 1)
            if x[i] >= x[i - 1] and x[i] > x[i + 1] and x[i] > min_height]
    kept = []
    for i in sorted(cand, key=lambda i: -x[i]):
        if all(abs(t[i] - t[k]) >= min_gap_s for k in kept):
            kept.append(i)
    return sorted(kept)


def _window(t, x, lo, hi):
    return x[(t >= lo) & (t <= hi)]


def calibrate(gyro, fsr):
    """gyro: [(t, (gx, gy, gz))], fsr: [(t, (toe, heel))] from a walk."""
    if len(gyro) < 100 or len(fsr) < 100:
        raise CalibrationError(f"only {len(gyro)} gyro and {len(fsr)} FSR readings")
    tg = np.array([t for t, _ in gyro])
    G = np.array([v for _, v in gyro], dtype=float)
    tf = np.array([t for t, _ in fsr])
    F = np.array([v for _, v in fsr], dtype=float)
    toe, heel = F[:, 0], F[:, 1]

    axis = int(np.argmax(G.std(axis=0)))
    g = G[:, axis]
    load = np.interp(tg, tf, heel + toe)

    def load_at_peak(sign):
        s = sign * g
        return load[s > 0.6 * s.max()].mean()

    sign = 1.0 if load_at_peak(1.0) <= load_at_peak(-1.0) else -1.0
    s = sign * g
    if s.max() < MIN_SWING_PEAK:
        raise CalibrationError(f"largest swing was {s.max():.1f} rad/s; walk during calibration")

    peaks = _find_peaks(tg, s, 0.5 * s.max(), MIN_STRIDE_S)
    if len(peaks) < 3:
        raise CalibrationError(f"found {len(peaks)} swings in {tg[-1] - tg[0]:.1f} s; need 3. "
                               f"Walk steadily during calibration")
    tp = tg[peaks]
    T = float(np.median(np.diff(tp)))

    def levels(x, swing_q, stance_lo, stance_hi, stance_fn):
        sw = [_window(tf, x, p - 0.15 * T, p + 0.10 * T) for p in tp]
        st = [_window(tf, x, p + stance_lo * T, p + stance_hi * T) for p in tp]
        sw = np.concatenate([w for w in sw if len(w)])
        st = np.concatenate([w for w in st if len(w)] or [np.array([np.nan])])
        return float(np.quantile(sw, swing_q)), float(stance_fn(st))

    heel_levels = levels(heel, 1.0, 0.35, 0.55, np.median)
    toe_levels = levels(toe, 1.0, 0.30, 0.75, lambda x: np.quantile(x, 0.9))

    warnings = []

    def threshold(name, sw, st):
        if not st - sw >= MIN_CONTRAST:
            warnings.append(f"{name} FSR: swing {sw:.0f} vs stance {st:.0f} counts is too "
                            f"close to tell apart; {name} not used (check the strap)")
            return math.inf
        return sw + FSR_FRAC * (st - sw)

    heel_thr = threshold("heel", *heel_levels)
    toe_thr = threshold("toe", *toe_levels)
    if math.isinf(heel_thr) and math.isinf(toe_thr):
        raise CalibrationError("neither FSR changes between swing and stance")

    loaded = (heel > heel_thr) | (toe > toe_thr)
    leads = []
    for p in tp:
        hit = np.nonzero(loaded & (tf > p) & (tf < p + 0.5 * T))[0]
        if not len(hit):
            continue
        t_fsr = tf[hit[0]]
        cross = np.nonzero((tg > p) & (s <= 0) & (tg < t_fsr + 0.1))[0]
        leads.append(t_fsr - tg[cross[0]] if len(cross) else math.nan)

    valid = [x for x in leads if not math.isnan(x)]
    use_gyro = len(valid) >= 2 and len(valid) == len(leads) and min(valid) > 0
    if not use_gyro:
        warnings.append("gyro zero crossing did not come before the FSRs on every stride; "
                        "heel strike from the FSRs only")
    confirm_s = min(0.3, max(valid, default=0.0) + 0.1)
    return Calibration(axis, sign, T, 0.5 * float(np.median(s[peaks])), heel_thr, toe_thr,
                       heel_levels, toe_levels, leads, use_gyro, confirm_s, warnings)


# ── Live detection ─────────────────────────────────────────────────────────────
@dataclass
class Transition:
    t: float
    phase: str
    reason: str
    info: str = ""


class PhaseDetector:
    """Feed gyro and FSR readings in time order; each call returns a Transition
    or None."""

    def __init__(self, cal):
        self.cal = cal
        self.phase = None
        self.t_change = 0.0         # last printed transition
        self.t_toe_off = -math.inf  # last toe-off; a withdrawn heel strike keeps it
        self.armed = False          # mid-swing seen in this swing
        self.pending = None         # time of a gyro heel strike the FSRs have not confirmed
        self.lead = None            # how far the gyro was ahead of the FSRs this stance
        self.imu_ok = True
        self.stance_s, self.swing_s, self.leads = [], [], []
        self.by = {"shank gyro": 0, "FSR": 0}
        self.withdrawn = 0

    def gyro(self, t, values):
        s = self.cal.sign * values[self.cal.axis]
        if self.phase != SWING:
            return None
        if s > self.cal.arm_level:
            self.armed = True
        elif (self.armed and s <= 0 and self.cal.use_gyro and self.imu_ok
              and t - self.t_toe_off >= self.cal.min_phase_s):
            self.pending = t
            return self._enter(STANCE, t, "shank gyro")
        return None

    def fsr(self, t, values):
        toe, heel = values
        loaded = heel > self.cal.heel_thr or toe > self.cal.toe_thr
        if self.phase is None:
            self.phase, self.t_change = (STANCE if loaded else SWING), t
            return Transition(t, self.phase, "start")
        if self.phase == SWING:
            if loaded and t - self.t_toe_off >= self.cal.min_phase_s:
                return self._enter(STANCE, t, "FSR")
            return None
        if self.pending is not None:
            if loaded:
                self.lead, self.pending = t - self.pending, None
                if math.isfinite(self.t_toe_off):
                    self.swing_s.append(self.t_change - self.t_toe_off)
            elif t - self.pending > self.cal.confirm_s:
                self.pending = None
                self.withdrawn += 1
                self.by["shank gyro"] -= 1
                return self._enter(SWING, t, "withdrawn", "FSRs did not load", withdrawn=True)
            return None
        if not loaded and t - self.t_change >= self.cal.min_phase_s:
            return self._enter(SWING, t, "toe-off")
        return None

    def _enter(self, phase, t, reason, info="", withdrawn=False):
        if phase == STANCE:
            self.by[reason] += 1
            self.lead = None
            if math.isfinite(self.t_toe_off):
                swing = t - self.t_toe_off
                info = f"swing {swing:.2f} s"
                if reason == "FSR":             # a gyro heel strike is counted when confirmed
                    self.swing_s.append(swing)
        elif withdrawn:
            self.armed = False                  # wait for the next mid-swing
        else:
            stance = t - self.t_change
            self.t_toe_off = t
            self.armed = False
            self.stance_s.append(stance)
            info = f"stance {stance:.2f} s"
            if self.lead is not None:
                self.leads.append(self.lead)
                info += f" | gyro ahead of FSR by {self.lead * 1000:.0f} ms"
        self.phase, self.t_change = phase, t
        return Transition(t, phase, reason, info)


# ── Terminal output ────────────────────────────────────────────────────────────
def print_calibration(cal, imu):
    axis = "xyz"[cal.axis]
    print(f"\n[cal] stride {cal.stride_s:.2f} s ({60.0 / cal.stride_s:.0f} strides/min), "
          f"swing axis {imu} g{axis} (sign {cal.sign:+.0f}), mid-swing arm level "
          f"{cal.arm_level:.1f} rad/s")
    for name, (sw, st), thr in (("heel", cal.heel_levels, cal.heel_thr),
                                ("toe", cal.toe_levels, cal.toe_thr)):
        shown = "not used" if math.isinf(thr) else f"{thr:.0f}"
        print(f"[cal] {name:<4} FSR  swing {sw:6.0f}  stance {st:6.0f}  -> threshold {shown}")
    leads = " ".join("--" if math.isnan(x) else f"{x * 1000:.0f}" for x in cal.leads_s)
    print(f"[cal] gyro zero crossing ahead of FSR heel strike, per stride [ms]: {leads}")
    print(f"[cal] heel strike from: "
          + ("shank gyro, confirmed by the FSRs within "
             f"{cal.confirm_s * 1000:.0f} ms" if cal.use_gyro else "FSRs only"))
    for w in cal.warnings:
        print(f"[cal] note: {w}")


def print_summary(det, elapsed):
    def med(x):
        return f"{np.median(x):.2f} s" if x else "--"
    print(f"\nSummary over {elapsed:.1f} s")
    print(f"  heel strikes: {det.by['shank gyro']} by shank gyro, {det.by['FSR']} by FSR, "
          f"{det.withdrawn} withdrawn")
    print(f"  median stance {med(det.stance_s)}, median swing {med(det.swing_s)}")
    if det.leads:
        ms = np.array(det.leads) * 1000
        print(f"  gyro ahead of FSR: median {np.median(ms):.0f} ms, "
              f"25-75% {np.percentile(ms, 25):.0f}-{np.percentile(ms, 75):.0f} ms "
              f"({len(ms)} strides)")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calib-s", type=float, default=5.0,
                    help="seconds of walking used for calibration")
    ap.add_argument("--countdown", type=float, default=3.0,
                    help="seconds to start walking before calibration begins")
    ap.add_argument("--imu", choices=("shank", "foot"), default="shank",
                    help="IMU whose gyro marks heel strike")
    ap.add_argument("--stall-ms", type=float, default=20.0,
                    help="report an I2C call that blocks this long")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "Sensor_Tests"))
    import sensor_dropout_monitor as sdm                  # noqa: E402

    sdm.BUS_USERS = sdm.BUS_USERS + ("gait_phase_detector.py", "imu_timestamp_check.py",
                                     "true_dropout_monitor.py", "imu_only_dropout_monitor.py",
                                     "imu_diag.py")
    busy = sdm.other_bus_user()
    if busy:
        sys.exit(f"Another process is using I2C bus {sdm.I2C_BUS}; stop it first:\n  {busy}")

    gyro_stream = f"{args.imu}_gyro"
    watched = (gyro_stream, "fsr")
    try:
        reader = sdm.Reader(args.stall_ms / 1000.0, reconnect=True)
    except KeyboardInterrupt:
        print("\nAborted during init.")
        return
    reader.start()
    last = {s: time.perf_counter() for s in sdm.STREAMS}

    def drain():
        """New readings of every stream, oldest first; prints reader events."""
        while reader.events:
            t, kind, text = reader.events.popleft()
            print(sdm.paint(f"[{t - t0:8.3f}s] {kind:<6} {text}",
                            sdm.EVENT_COLOR.get(kind, sdm.BOLD)), flush=True)
        out = []
        for s in sdm.STREAMS:
            q = reader.samples[s]
            while q:
                t, v = q.popleft()
                last[s] = t
                out.append((t, s, v))
        out.sort(key=lambda x: x[0])
        return out

    t0 = time.perf_counter()
    det, t_live = None, None
    try:
        while det is None:
            print(f"\n[cal] start walking; calibration begins in {args.countdown:.0f} s", flush=True)
            t_end = time.perf_counter() + args.countdown
            while time.perf_counter() < t_end:
                drain()
                time.sleep(0.01)
            print(f"[cal] calibrating for {args.calib_s:.0f} s, keep walking normally...", flush=True)
            gyro, fsr = [], []
            t_end = time.perf_counter() + args.calib_s
            while time.perf_counter() < t_end:
                for t, s, v in drain():
                    if s == gyro_stream:
                        gyro.append((t, v))
                    elif s == "fsr":
                        fsr.append((t, v))
                time.sleep(0.01)
            try:
                cal = calibrate(gyro, fsr)
            except CalibrationError as exc:
                print(sdm.paint(f"[cal] calibration failed: {exc}. Trying again.", sdm.RED))
                continue
            print_calibration(cal, args.imu)
            det = PhaseDetector(cal)

        t_live = time.perf_counter()
        print("\n[live] phase changes below (t from now). Ctrl+C to stop.\n")
        stale = set()
        while True:
            for t, s, v in drain():
                if s == gyro_stream:
                    tr = det.gyro(t, v)
                elif s == "fsr":
                    tr = det.fsr(t, v)
                else:
                    continue
                if tr is not None:
                    color = "32" if tr.phase == STANCE else "36"
                    print(f"{tr.t - t_live:9.3f} s  " + sdm.paint(f"{tr.phase:<6}", color)
                          + f"  {tr.reason:<11} {tr.info}", flush=True)
            now = time.perf_counter()
            for s in watched:
                if now - last[s] > STALE_S and s not in stale:
                    stale.add(s)
                    print(sdm.paint(f"!! {s}: no reading for {(now - last[s]) * 1000:.0f} ms"
                                    + ("; heel strike from the FSRs until it returns"
                                       if s == gyro_stream else ""), sdm.RED), flush=True)
                elif now - last[s] <= STALE_S and s in stale:
                    stale.discard(s)
                    print(f"{s} updating again", flush=True)
            if (gyro_stream in stale) == det.imu_ok:
                det.imu_ok = not det.imu_ok
                det.armed = False
            if not reader.is_alive():
                print(sdm.paint("[live] sensor reader stopped", sdm.RED))
                break
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        reader.join(timeout=1.0)
        drain()
        if reader.is_alive():
            print(sdm.paint("[monitor] reader thread still blocked in an I2C call at exit", sdm.RED))
        else:
            reader.close()
        if det is not None and t_live is not None:
            print_summary(det, time.perf_counter() - t_live)


if __name__ == "__main__":
    main()
