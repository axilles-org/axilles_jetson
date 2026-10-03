#!/usr/bin/env python3
"""
test_recording.py

Regression tests for the calibration recorder. Run standalone:

    python test_recording.py

No hardware needed - a fake sensor hub with controllable I2C latency stands in.
These exist because the first version of record_phase ran a fixed SAMPLE COUNT
instead of a wall-clock duration, so when the bus could not keep up a 10 s phase
silently ran 30 s and the countdown went negative.
"""
import sys
from pathlib import Path

import time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import exo_frame as E

fails = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} {detail}")
    if not ok:
        fails.append(name)


class FakeHub:
    """Sensor hub with controllable per-read latency."""
    def __init__(self, delay=0.0, imu_period=None, drop=False):
        self.delay = delay
        self.imu_period = imu_period
        self.drop = drop
        self.t0 = time.perf_counter()
        self._n = 0

    def _v(self):
        t = time.perf_counter() - self.t0
        return [np.sin(t), np.cos(t), 0.5]

    def read_imu_foot(self):
        if self.delay:
            time.sleep(self.delay)
        if self.drop and self._n % 3 == 0:
            self._n += 1
            return None
        self._n += 1
        return {"accel": self._v(), "gyro": self._v(), "quat": [0, 0, 0, 1]}

    read_imu_shank = read_imu_foot

    def read_encoder(self):
        if self.delay:
            time.sleep(self.delay)
        return {"ankle_encoder_deg": 81.4}

    def read_fsr(self):
        if self.delay:
            time.sleep(self.delay)
        return {"toe_fsr_raw": 900.0, "heel_fsr_raw": 1200.0}


print("=" * 78)
print("record_phase: does the phase last the time it promises?")
print("=" * 78)

for label, delay in (("fast bus", 0.0), ("slow bus (5 ms/read)", 0.005),
                     ("very slow bus (20 ms/read)", 0.020)):
    hub = FakeHub(delay=delay)
    t0 = time.perf_counter()
    rec = E.record_phase(hub, seconds=2.0, fs=200.0, progress=False)
    wall = time.perf_counter() - t0
    n = len(rec["time"])
    check(f"{label}: wall clock", abs(wall - 2.0) < 0.35,
          f"{wall:.2f} s for a 2.0 s phase, {n} samples "
          f"({n / 2.0:.0f} Hz achieved)")
    check(f"{label}: time monotonic and in range",
          n > 0 and np.all(np.diff(rec["time"]) >= 0)
          and rec["time"][-1] <= 2.0 and rec["time"][0] >= 0,
          f"t spans [{rec['time'][0]:.3f}, {rec['time'][-1]:.3f}]")
    check(f"{label}: all arrays same length",
          len({len(v) for v in rec.values()}) == 1,
          f"{ {k: len(v) for k, v in rec.items()} if len({len(v) for v in rec.values()}) != 1 else 'consistent'}")

print("\n" + "=" * 78)
print("record_phase: dropped IMU packets stay NaN, not silently held")
print("=" * 78)
hub = FakeHub(delay=0.0, drop=True)
rec = E.record_phase(hub, seconds=1.0, fs=100.0, progress=False)
check("encoder always present", np.isfinite(rec["encoder"]).all(),
      f"{np.isfinite(rec['encoder']).sum()}/{len(rec['encoder'])} finite")
check("returns usable IMU data", np.isfinite(rec["foot_accel"]).any(),
      f"{np.isfinite(rec['foot_accel']).all(axis=1).sum()} finite IMU rows")

print("\n" + "=" * 78)
print("record_phase: a dead hub fails loudly instead of returning empty arrays")
print("=" * 78)


class DeadHub(FakeHub):
    def read_encoder(self):
        raise OSError("bus error")
    def read_fsr(self):
        raise OSError("bus error")
    def read_imu_foot(self):
        raise OSError("bus error")
    read_imu_shank = read_imu_foot


rec = E.record_phase(DeadHub(), seconds=0.5, fs=100.0, progress=False)
check("dead hub still returns timed samples", len(rec["time"]) > 0,
      f"{len(rec['time'])} samples, all sensor channels NaN "
      f"({np.isfinite(rec['foot_accel']).sum()} finite)")

print("\n" + "=" * 78)
print("_ffill_nan: vectorised version matches the reference")
print("=" * 78)


def reference_ffill(a):
    a = np.array(a, dtype=np.float64, copy=True)
    sq = a.ndim == 1
    if sq:
        a = a[:, None]
    for j in range(a.shape[1]):
        col = a[:, j]
        idx = np.where(np.isfinite(col))[0]
        if len(idx) == 0:
            col[:] = 0.0
            continue
        first = idx[0]
        col[:first] = col[first]
        last = first
        for i in range(first, len(col)):
            if np.isfinite(col[i]):
                last = i
            else:
                col[i] = col[last]
    return a[:, 0] if sq else a


rng = np.random.default_rng(0)
worst = 0.0
for trial in range(40):
    n = rng.integers(1, 200)
    m = rng.integers(1, 4)
    x = rng.normal(size=(n, m))
    x[rng.random((n, m)) < 0.4] = np.nan
    if trial % 7 == 0:                       # a fully-empty column
        x[:, 0] = np.nan
    if trial % 5 == 0:                       # leading NaN
        x[:3] = np.nan
    got, want = E._ffill_nan(x), reference_ffill(x)
    worst = max(worst, float(np.abs(got - want).max()))
check("matches reference over 40 random cases", worst == 0.0,
      f"max difference {worst:.2e}")

v = rng.normal(size=500)
v[rng.random(500) < 0.3] = np.nan
check("1-D input returns 1-D", E._ffill_nan(v).shape == (500,),
      f"shape {E._ffill_nan(v).shape}")
check("all-NaN column becomes zeros",
      np.all(E._ffill_nan(np.full((10, 2), np.nan)) == 0.0), "zeroed")
check("empty input does not crash", E._ffill_nan(np.empty((0, 3))).shape == (0, 3),
      "shape (0, 3)")

x = np.arange(2_000_000, dtype=float).reshape(-1, 4)
x[::3] = np.nan
t0 = time.perf_counter()
E._ffill_nan(x)
check("500k x 4 fill is fast", time.perf_counter() - t0 < 1.0,
      f"{time.perf_counter() - t0:.3f} s")

print("\n" + "=" * 78)
print("FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
print("=" * 78)
raise SystemExit(1 if fails else 0)
