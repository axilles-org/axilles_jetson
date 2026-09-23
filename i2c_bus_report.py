#!/usr/bin/env python3
"""
i2c_bus_report.py — what is on I²C bus 1, how fast the bus runs, and how fast
the data actually arrives from each sensor alone and in combination.

Four questions, four sections, in order:

  1. WHAT IS ON THE BUS
     A read-only address scan (0x03–0x77) plus the devices the kernel has
     already claimed.  A claimed device (INA3221, FUSB301) does NOT answer a
     userspace scan — its driver holds the address — so it is listed from
     /sys/bus/i2c/devices instead of being silently missing.

  2. HOW FAST THE BUS IS CLOCKED
     Two numbers, because they disagree more often than you would like:

       configured — the device-tree `clock-frequency` property.  What the bus
                    was ASKED for.
       measured   — what it actually delivers, timed here.

     The measurement is a size sweep of pure I²C reads (no register write, so
     it is safe against any device) against the simplest responder on the bus.
     Transaction cost is linear in length:

         t = fixed_overhead + bytes / throughput

     The slope is the bus; the intercept is the kernel's per-transfer cost.
     Each byte on the wire is 9 SCL clocks (8 data + 1 ACK), so

         SCL ≈ 9 / slope

     Measured SCL always lands a little under the configured clock — clock
     stretching, setup/hold times and the bytes not counted by the fit
     (address, START/STOP) all show up as loss.  ~350 kHz measured on a
     400 kHz bus is healthy; ~98 kHz would mean the bus is really at 100 kHz.

  3. HOW FAST THE DATA COMES FROM THE FULL SYSTEM
     Measured by running the real reader — BNO085/sensor_parse.py's SensorHub,
     unmodified — for a fixed wall-clock window.  These are the rates every
     other script in the repo gets.

  4. EACH SENSOR ALONE, THEN IN COMBINATION
     The same loop over a varying subset of the four devices: each one solo,
     then the IMUs together, encoder + FSRs, and so on up to all four.  Solo
     is that device's ceiling on this bus; the gap between solo and shared is
     bus contention and nothing else, because nothing else changed.

WHAT EACH RATE MEANS
────────────────────
  loop     poll cycles per second; the ceiling for everything below
  IMU      reports per second, counted per type (quat, accel and gyro
           separately — the device quantises each requested interval on its
           own, so they do not match), at the point the library parses them.
           NOT by watching the value change: a stationary gyro repeats the
           same bits and would score near zero.  A report the loop never came
           back for was overwritten in the device cache and never parsed, so
           this is usable rate, not the rate requested of the device.
  ADC      completed ADS1115 conversions.  The two channels alternate through
           one converter, so each FSR updates at half this figure.
  encoder  AS5600 reads issued; one per loop cycle, so this tracks the loop
           rate whether or not the magnet moved.

A BNO085 that errors mid-run makes the reader reconnect, blocking the loop for
a second or more.  Those stalls are timed and excluded, and reported, rather
than being averaged into a rate they have nothing to do with.

USAGE
    python3 i2c_bus_report.py                  # everything, bus 1, 5 s per run
    python3 i2c_bus_report.py --bus 7          # a different bus
    python3 i2c_bus_report.py --duration 10    # longer windows
    python3 i2c_bus_report.py --combos all     # all 15 subsets, not the curated 10
    python3 i2c_bus_report.py --combos a,ab,abef
    python3 i2c_bus_report.py --report-hz 200  # sweep at a different BNO085 rate
    python3 i2c_bus_report.py --no-matrix      # sections 1-3 only
    python3 i2c_bus_report.py --no-sensors     # sections 1-2 only, ~1 s

Read-only throughout.  Nothing on the bus is configured or written, except the
ADS1115 conversion trigger without which it does not sample at all.
"""

from __future__ import annotations

import argparse
import io
import statistics
import sys
import time
from contextlib import redirect_stdout
from itertools import combinations
from pathlib import Path

import smbus2

# ── Known addresses on the exo's sensor bus ───────────────────────────────────
KNOWN = {
    0x36: "AS5600   magnetic encoder",
    0x48: "ADS1115  4-ch ADC (FSR-1 on AIN0, FSR-2 on AIN1)",
    0x4A: "BNO085   IMU-A",
    0x4B: "BNO085   IMU-B",
}

# Preference order for the device used in the clock sweep: simplest protocol
# first.  The AS5600 has no packet layer at all, so its transaction time is
# the bus and nothing else.
SWEEP_PREFERENCE = (0x36, 0x48, 0x4A, 0x4B)

SWEEP_SIZES   = (1, 2, 4, 8, 16, 24, 32)
SWEEP_REPEATS = 300


# ── Section 1: what is on the bus ─────────────────────────────────────────────
def dt_node(bus: int):
    """Device-tree node name for an adapter, e.g. 'c240000.i2c'."""
    p = Path(f"/sys/bus/i2c/devices/i2c-{bus}/name")
    return p.read_text().strip() if p.exists() else None


def dt_clock_hz(bus: int):
    """Configured clock-frequency from the device tree, in Hz, or None.

    The property is a 4-byte big-endian cell, not text.
    """
    p = Path(f"/sys/bus/i2c/devices/i2c-{bus}/of_node/clock-frequency")
    if not p.exists():
        return None
    raw = p.read_bytes()
    try:
        return int.from_bytes(raw, "big") if len(raw) == 4 else int(raw.decode().strip())
    except ValueError:
        return None


def kernel_claimed(bus: int):
    """[(addr, driver_name)] for devices a kernel driver already owns.

    These do not answer a userspace scan, so they must be read from sysfs or
    they look like empty addresses.
    """
    out = []
    for d in sorted(Path("/sys/bus/i2c/devices").glob(f"{bus}-[0-9a-f]*")):
        try:
            addr = int(d.name.split("-")[1], 16)
        except ValueError:
            continue
        name = (d / "name").read_text().strip() if (d / "name").exists() else "?"
        out.append((addr, name))
    return out


def scan(bus_handle: smbus2.SMBus):
    """Addresses that ACK, found with the least intrusive probe available.

    A 1-byte read is harmless on every device here.  Addresses that refuse a
    read get one write_quick (address + no data) before being called empty —
    some devices ACK only that.
    """
    found = []
    for addr in range(0x03, 0x78):
        try:
            bus_handle.read_byte(addr)
            found.append(addr)
            continue
        except OSError:
            pass
        try:
            bus_handle.write_quick(addr)
            found.append(addr)
        except OSError:
            pass
    return found


# ── Section 2: how fast the bus is clocked ────────────────────────────────────
def sweep_clock(bus_handle: smbus2.SMBus, addr: int):
    """Time pure reads of several lengths; fit t = fixed + bytes/throughput.

    Pure reads (i2c_msg.read) send no register pointer, so this is safe on a
    device whose register map is unknown and the byte count in the fit is
    exactly the payload.

    The fit uses the MINIMUM time at each length, not the mean or median.
    Contention can only ever ADD time — this adapter is shared with whatever
    else touches the bus (the kernel's INA3221 polling, say), and a transfer
    that waited on the adapter mutex says nothing about the clock.  The
    fastest transfer of several hundred is the uncontended cost, which is the
    thing being measured.  Medians are kept alongside so the gap between them
    shows how busy the bus was.
    """
    timings, medians = {}, {}
    for n in SWEEP_SIZES:
        msg = smbus2.i2c_msg.read(addr, n)
        samples = []
        for _ in range(SWEEP_REPEATS):
            t0 = time.perf_counter()
            bus_handle.i2c_rdwr(msg)
            samples.append(time.perf_counter() - t0)
        timings[n] = min(samples)
        medians[n] = statistics.median(samples)

    xs = list(timings)
    ys = [timings[n] for n in xs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    intercept = my - slope * mx

    # Coefficient of determination — a bad fit means something else (another
    # process on the bus, CPU contention) is in the measurement.
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")

    return {
        "timings":     timings,
        "medians":     medians,
        "fixed_s":     intercept,
        "per_byte_s":  slope,
        "bytes_per_s": 1.0 / slope,
        "scl_hz":      9.0 / slope,    # 8 data bits + 1 ACK per byte
        "r2":          r2,
    }


# ── Sections 3 and 4: measuring what a given set of sensors delivers ──────────
#
# Both sections run the same poll loop and the same counters.  The only
# variable is WHICH devices are in the loop — that is what makes section 4 a
# controlled experiment: any difference between a device's solo rate and its
# rate alongside others is bus contention, nothing else.

DEVICES = {                       # key → (long label, column header)
    "a": ("IMU-A  BNO085 0x4A",  "IMU-A"),
    "b": ("IMU-B  BNO085 0x4B",  "IMU-B"),
    "f": ("FSRs   ADS1115 0x48", "FSR"),
    "e": ("Enc    AS5600 0x36",  "ENC"),
}

# Curated combinations: each device alone, then the groupings the exo actually
# runs.  --combos all replaces this with the full 15-member power set.
DEFAULT_COMBOS = [
    ("a",), ("b",), ("f",), ("e",),           # solo — the ceiling for each
    ("a", "b"),                               # both IMUs
    ("e", "f"),                               # encoder + FSRs, no IMU traffic
    ("a", "e", "f"),                          # one IMU + the cheap devices
    ("a", "b", "e"),
    ("a", "b", "f"),
    ("a", "b", "e", "f"),                     # everything — the real system
]

STALL_S = 0.050    # a gap longer than this is a reconnect, not a slow cycle


class _ReportCounter:
    """Counts BNO085 reports where they actually arrive.

    The obvious test for a fresh IMU sample — "did the cached value change?" —
    is wrong, and wrong in a way that looks like a bus problem.  A STATIONARY
    gyro emits the same bit pattern sample after sample, so change-detection
    scores it near zero however fast the device is really reporting, while the
    accelerometer's noisy low bit scores nearly every sample.  That produced a
    table showing gyro at 0.2 Hz next to accel at 33 Hz from the same device.

    adafruit_bno08x calls _process_report(report_id, bytes) once per report
    parsed out of a packet, so wrapping it counts deliveries directly, whatever
    the values are.  The wrapper is installed on the instance, shadowing the
    class method; _handle_packet looks it up through self, so it takes effect.
    A reconnect builds a new BNO08X object, which ensure() notices and re-wraps.
    """

    def __init__(self, imu):
        self._imu  = imu       # sensor_parse._FastIMU
        self._bno  = None
        self.counts = {}
        self.ensure()

    def ensure(self):
        """Re-wrap if this is a different BNO08X object than last seen."""
        bno = getattr(self._imu, "_bno", None)
        if bno is None or bno is self._bno:
            return
        self._bno = bno
        original = type(bno)._process_report.__get__(bno)   # never wrap a wrapper

        def counted(report_id, report_bytes, _orig=original, _c=self.counts):
            _c[report_id] = _c.get(report_id, 0) + 1
            return _orig(report_id, report_bytes)

        bno._process_report = counted

    def snapshot(self) -> dict:
        return dict(self.counts)


def _report_ids(sp) -> dict:
    """field name → BNO085 report id, from sensor_parse's own imports."""
    return {"quat":  sp.BNO_REPORT_GAME_ROTATION_VECTOR,
            "accel": sp.BNO_REPORT_ACCELEROMETER,
            "gyro":  sp.BNO_REPORT_GYROSCOPE}


def _load_sensor_parse():
    """Import BNO085/sensor_parse.py and hand back the module."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "BNO085"))
    import sensor_parse                       # noqa: E402  (path set above)
    return sensor_parse


class _Bench:
    """Every device the sweep will need, connected ONCE; each run polls a subset.

    The obvious implementation — build a fresh rig per combination — was wrong,
    and measurably so.  Re-initialising a BNO085 soft-resets it, and putting one
    through ten reset/enable cycles in two minutes leaves it delivering a
    fraction of its normal report rate.  The sweep then measures a progressively
    sicker device and blames the difference on bus contention.

    So the devices are connected once and never torn down; what varies between
    runs is only which of them the loop TALKS to.  I²C is host-driven, so a
    device nobody addresses puts nothing on the wire — not polling it is exactly
    equivalent to its absence, as far as bus contention is concerned — while
    every row is measured against identical device state.

    Built from sensor_parse's own classes rather than reimplementations, so
    polling all four is the work SensorHub does; that is why the last row of
    section 4 should agree with section 3.
    """

    def __init__(self, sp, needed, bus: int, report_hz: float, max_packets: int):
        self.imu_a = self.imu_b = self.ads = self.enc = None
        self._smbus = None

        if "a" in needed:
            self.imu_a = sp._FastIMU(bus, sp.IMU_A_ADDR, "IMU-A", report_hz,
                                     sp.DEFAULT_FEATURES, max_packets)
        if "b" in needed:
            self.imu_b = sp._FastIMU(bus, sp.IMU_B_ADDR, "IMU-B", report_hz,
                                     sp.DEFAULT_FEATURES, max_packets)
        if "f" in needed or "e" in needed:
            self._smbus = smbus2.SMBus(bus)          # ADC and encoder share it
        if "f" in needed:
            self.ads = sp._ADS1115(self._smbus, sp.ADS_ADDR)
        if "e" in needed:
            self.enc = sp._AS5600(self._smbus, sp.AS5600_ADDR)

        self.counters = {"imu_a": _ReportCounter(self.imu_a) if self.imu_a else None,
                         "imu_b": _ReportCounter(self.imu_b) if self.imu_b else None}

    def reader(self, combo):
        """(read, ads_channel, counters) for one cycle over `combo`, in SensorHub's order."""
        imu_a = self.imu_a if "a" in combo else None
        imu_b = self.imu_b if "b" in combo else None
        ads   = self.ads   if "f" in combo else None
        enc   = self.enc   if "e" in combo else None

        def read():
            frame = {"imu_a": None, "imu_b": None}
            if imu_a:
                frame["imu_a"] = imu_a.read()
            if imu_b:
                frame["imu_b"] = imu_b.read()
            if ads:
                ads.update()
                frame["fsr1"], frame["fsr2"] = ads.fsr1, ads.fsr2
            if enc:
                enc.update()
                frame["angle_deg"], frame["angle_raw"] = enc.angle_deg, enc.raw
            return frame

        counters = {"imu_a": self.counters["imu_a"] if imu_a else None,
                    "imu_b": self.counters["imu_b"] if imu_b else None}
        return read, (lambda: ads._ch if ads else None), counters

    def close(self):
        for imu in (self.imu_a, self.imu_b):
            if imu:
                imu.close()
        if self._smbus:
            try:
                self._smbus.close()
            except Exception:
                pass


def measure(read, ads_channel, devices, duration: float, counters=None,
            report_ids=None):
    """Poll for `duration` seconds and count what each signal delivered.

    read         callable → frame dict
    ads_channel  callable → the ADS1115's current channel, or None if absent
    devices      which device keys are in this loop
    counters     {"imu_a": _ReportCounter|None, "imu_b": ...}
    report_ids   field name → BNO085 report id, from _report_ids()

    Counting rules, one per sensor type:

      IMU      reports counted at the point the library parses them (see
               _ReportCounter), per report type — the device quantises each
               requested interval separately, so quat, accel and gyro do not
               arrive at the same rate.  A report the loop never came back for
               was overwritten in the device cache and is never parsed, so
               these are usable samples, not the rate requested of the device.
      ADS1115  the driver alternates AIN0/AIN1, so a channel flip is exactly
               one completed conversion.  Not inferred from the voltage, which
               could legitimately repeat.
      AS5600   one transaction per cycle by construction, so reads == cycles.
               Counting value changes instead would report 0 Hz whenever the
               joint happens to be still.

    Stalls (a BNO085 error forces a blocking reconnect) are timed out of the
    window rather than averaged into the rates.
    """
    counters   = counters or {}
    report_ids = report_ids or {}
    counts = {"loop": 0, "adc": 0, "enc": 0}
    for key in ("imu_a", "imu_b"):
        for field in ("quat", "accel", "gyro"):
            counts[f"{key}_{field}"] = 0

    active  = [(key, counters[key]) for key in ("imu_a", "imu_b")
               if counters.get(key) is not None]
    at_start = {key: c.snapshot() for key, c in active}

    prev_ch    = ads_channel()
    last_frame = None
    stalls, stall_s = 0, 0.0

    t_start = t_prev = time.perf_counter()
    try:
        while time.perf_counter() - t_start < duration:
            frame = read()
            t_now = time.perf_counter()
            dt, t_prev = t_now - t_prev, t_now
            if dt > STALL_S:
                stalls  += 1
                stall_s += dt

            counts["loop"] += 1
            if "e" in devices:
                counts["enc"] += 1

            for _, c in active:
                c.ensure()          # re-wrap after a reconnect; else a no-op

            ch = ads_channel()
            if ch is not None and ch != prev_ch:
                counts["adc"] += 1
                prev_ch = ch

            last_frame = frame
    except KeyboardInterrupt:
        print("\n  (interrupted — reporting the window measured so far)")

    for key, c in active:
        now, before = c.snapshot(), at_start[key]
        for field, rid in report_ids.items():
            counts[f"{key}_{field}"] = now.get(rid, 0) - before.get(rid, 0)

    elapsed = time.perf_counter() - t_start
    active  = max(elapsed - stall_s, 1e-9)
    return {"counts": counts, "elapsed": elapsed, "active": active,
            "stalls": stalls, "stall_s": stall_s, "frame": last_frame,
            "devices": tuple(devices)}


def rates(result: dict) -> dict:
    """counts → Hz over the non-stalled time."""
    a = result["active"]
    return {k: v / a for k, v in result["counts"].items()}


# ── Section 3: the real SensorHub ─────────────────────────────────────────────
def measure_sensor_hub(bus: int, duration: float):
    """Run sensor_parse.SensorHub itself, unmodified.

    Section 4 rebuilds the loop from the same parts in order to vary its
    membership; this runs the actual class, so these numbers are what every
    other script in the repo gets.
    """
    sp = _load_sensor_parse()
    chatter = io.StringIO()
    with redirect_stdout(chatter):
        hub = sp.SensorHub(bus=bus)
    counters = {"imu_a": _ReportCounter(hub._imu_a),
                "imu_b": _ReportCounter(hub._imu_b)}
    try:
        with redirect_stdout(chatter):
            res = measure(hub.read, lambda: hub._ads._ch, ("a", "b", "e", "f"),
                          duration, counters, _report_ids(sp))
    finally:
        hub.close()
    res["chatter"] = chatter.getvalue()
    return res


# ── Section 4: one device at a time, then combinations ────────────────────────
def _summarise_chatter(text: str) -> str:
    """Collapse the adafruit library's stdout into one short phrase.

    adafruit_bno08x prints the whole packet, unconditionally, whenever it
    cannot parse one (an SHTP advertisement after a reset, an unknown report
    id).  sensor_parse's drain catches the exception, so this is noise — but
    hundreds of lines of it would bury the table, and pretending it never
    happened would be worse.  Count it instead.
    """
    if not text:
        return ""
    notes = []
    packets = text.count("********** Packet")
    if packets:
        notes.append(f"{packets} unparsed packet(s)")
    reconnects = text.count("Reconnecting...")
    if reconnects:
        notes.append(f"{reconnects} reconnect(s)")
    return "   (" + ", ".join(notes) + ")" if notes else ""


def measure_matrix(bus: int, combos, duration: float, settle: float = 0.5,
                   report_hz=None, max_packets=None):
    """Measure each combination in turn against one set of live connections."""
    sp = _load_sensor_parse()
    hz = sp.REPORT_HZ   if report_hz   is None else report_hz
    mp = sp.MAX_PACKETS if max_packets is None else max_packets

    needed = set().union(*combos)
    print(f"    connecting {len(needed)} device(s) once, for every run...",
          end="", flush=True)
    chatter = io.StringIO()
    with redirect_stdout(chatter):
        bench = _Bench(sp, needed, bus, hz, mp)
    print(" done" + _summarise_chatter(chatter.getvalue()) + "\n")

    results = []
    try:
        for combo in combos:
            label = "+".join(DEVICES[d][1] for d in combo)
            print(f"    {label:<22} ", end="", flush=True)
            read, ads_channel, counters = bench.reader(combo)

            chatter = io.StringIO()
            try:
                with redirect_stdout(chatter):
                    # Settle: a device this run polls but the last one did not
                    # has a backlog queued.  Drain it before counting, or the
                    # first samples of the window are stale ones.
                    t_end = time.perf_counter() + settle
                    while time.perf_counter() < t_end:
                        read()
                    res = measure(read, ads_channel, combo, duration,
                                  counters, _report_ids(sp))
            except Exception as exc:
                print(f"FAILED ({type(exc).__name__}: {exc})")
                continue

            res["label"] = label
            results.append(res)
            r = rates(res)
            print(f"loop {r['loop']:8.1f} Hz"
                  + (f"   ({res['stalls']} stall(s), {res['stall_s']:.1f} s excluded)"
                     if res["stalls"] else "")
                  + _summarise_chatter(chatter.getvalue()))
    except KeyboardInterrupt:
        print("\n    interrupted — reporting the runs completed so far")
    finally:
        bench.close()
    return results


def parse_combos(spec: str):
    """'default' → the curated list; 'all' → the 15-member power set;
    otherwise a comma-separated spec like 'a,ab,abef' over the keys a/b/e/f."""
    if spec == "default":
        return DEFAULT_COMBOS
    if spec == "all":
        out = []
        for n in range(1, 5):
            out += list(combinations("abfe", n))
        return sorted(out, key=lambda c: (len(c), c))
    combos = []
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        bad = set(part) - set("abef")
        if bad:
            raise ValueError(f"unknown device key(s) {''.join(sorted(bad))!r} in "
                             f"{part!r}; valid keys are a, b, e, f")
        combos.append(tuple(dict.fromkeys(part)))   # de-duplicate, keep order
    if not combos:
        raise ValueError("no combinations given")
    return combos


# ── Report ────────────────────────────────────────────────────────────────────
def rule(title):
    print(f"\n{title}")
    print("─" * 78)


SIGNALS = [                       # (count key, column header, device required)
    ("imu_a_quat",  "A-quat", "a"),
    ("imu_a_accel", "A-acc",  "a"),
    ("imu_a_gyro",  "A-gyro", "a"),
    ("imu_b_quat",  "B-quat", "b"),
    ("imu_b_accel", "B-acc",  "b"),
    ("imu_b_gyro",  "B-gyro", "b"),
    ("adc",         "ADC",    "f"),
    ("enc",         "ENC",    "e"),
]


def print_rate_table(results):
    """One row per combination, one column per signal. '·' = device not in loop."""
    head = (f"    {'combination':<18}{'loop':>7}"
            + "".join(f"{h:>7}" for _, h, _ in SIGNALS))
    print(head)
    print("    " + "─" * (len(head) - 4))
    for res in results:
        r    = rates(res)
        # A run that reconnected is not comparable with the others: the BNO085
        # re-enables its features on the way back and does not always come back
        # in the same configuration (a lost rotation-vector feature shows up as
        # quat 0 Hz with accel and gyro at double rate).  Mark it, do not hide it.
        mark = "*" if res["stalls"] else " "
        line = f"    {res['label'] + mark:<18}{r['loop']:>7.1f}"
        for key, _, need in SIGNALS:
            line += f"{r[key]:>7.1f}" if need in res["devices"] else f"{'·':>7}"
        print(line)
    print("\n    All figures in Hz.  '·' means that device was not in the loop.")
    if any(res["stalls"] for res in results):
        print("    * this run hit a sensor error and reconnected.  The BNO085 does"
              "\n    not always come back with the same features enabled, so treat"
              "\n    a starred row as suspect and re-run it.")
    if any(res["devices"] == ("f",) for res in results):
        print("    FSR alone shows an enormous loop rate because the ADS1115 driver"
              "\n    is non-blocking: while a conversion is in flight the cycle does no"
              "\n    I2C at all, so the loop spins.  Its ADC column is the real figure.")


def print_contention(results):
    """Solo rate vs rate in the largest combination — what sharing costs."""
    # Only clean runs: a row that reconnected may be describing a differently
    # configured device, and a ratio built from one is meaningless.
    clean = [res for res in results if not res["stalls"]]
    solo = {res["devices"][0]: res for res in clean if len(res["devices"]) == 1}
    full = max(clean, key=lambda r: len(r["devices"]), default=None)
    if not solo or full is None or len(full["devices"]) < 2:
        print("\n  (not enough clean runs to compare solo against shared)")
        return

    print(f"\n  What sharing the bus costs — solo vs. inside {full['label']}:")
    print(f"\n    {'signal':<12}{'solo':>10}{'shared':>10}{'kept':>8}")
    print("    " + "─" * 38)
    rf = rates(full)
    for key, header, need in SIGNALS:
        if need not in full["devices"] or need not in solo:
            continue
        rs = rates(solo[need])
        if rs[key] <= 0:
            continue
        print(f"    {header:<12}{rs[key]:>9.1f} {rf[key]:>9.1f} "
              f"{100 * rf[key] / rs[key]:>7.0f}%")
    print("\n  The bus is a serial resource: every transaction one device spends is"
          "\n  a transaction the others do not get.  A signal that keeps most of its"
          "\n  solo rate is limited by the DEVICE (its report rate, its conversion"
          "\n  time); one that loses a lot is limited by the BUS.")


def print_frame(frame):
    if not frame:
        return
    print("\n  last frame:")
    for key in ("imu_a", "imu_b"):
        s = frame.get(key)
        if s is None:
            print(f"    {key:<10} no data")
            continue
        q, g = s.get("quat"), s.get("gyro")
        line = f"    {key:<10}"
        if q:
            line += f"quat ({q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f} {q[3]:+.3f})  "
        if g:
            line += f"gyro ({g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f}) rad/s"
        print(line)
    if "fsr1" in frame:
        print(f"    {'fsr1':<10} {frame['fsr1']:.4f} V")
        print(f"    {'fsr2':<10} {frame['fsr2']:.4f} V")
    if "angle_deg" in frame:
        print(f"    {'angle':<10} {frame['angle_deg']:.2f}°  (raw {frame['angle_raw']})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", type=int, default=1, help="I2C bus number (default 1)")
    ap.add_argument("--duration", type=float, default=5.0,
                    help="seconds per rate measurement (default 5)")
    ap.add_argument("--combos", default="default",
                    help="'default' (curated), 'all' (full power set), or a spec "
                         "like 'a,ab,abef' over the keys a=IMU-A b=IMU-B "
                         "e=encoder f=FSRs")
    ap.add_argument("--settle", type=float, default=0.5,
                    help="seconds of polling discarded before each section-4 run, "
                         "to drain a backlog left by the previous one (default 0.5)")
    ap.add_argument("--report-hz", type=float, default=None,
                    help="override the BNO085 report rate in section 4 "
                         "(default: sensor_parse.REPORT_HZ)")
    ap.add_argument("--max-packets", type=int, default=None,
                    help="override packets drained per IMU per cycle in section 4 "
                         "(default: sensor_parse.MAX_PACKETS)")
    ap.add_argument("--no-sensors", action="store_true",
                    help="sections 1-2 only; no sensor libraries touched")
    ap.add_argument("--no-matrix", action="store_true",
                    help="skip section 4 (the per-device / per-combination sweep)")
    args = ap.parse_args()

    dev = Path(f"/dev/i2c-{args.bus}")
    if not dev.exists():
        sys.exit(f"{dev} does not exist. Available: "
                 + ", ".join(sorted(p.name for p in Path('/dev').glob('i2c-*'))))

    try:
        combos = parse_combos(args.combos)
    except ValueError as exc:
        sys.exit(f"--combos: {exc}")

    try:
        handle = smbus2.SMBus(args.bus)
    except PermissionError:
        sys.exit(f"No permission to open {dev} — add your user to the 'i2c' group.")

    # ── 1 ─────────────────────────────────────────────────────────────────────
    rule(f"1. DEVICES ON I2C BUS {args.bus}")
    node = dt_node(args.bus)
    print(f"  adapter      /dev/i2c-{args.bus}  ({node or 'unnamed'})")

    responders = scan(handle)
    print("\n  responding to a userspace scan:")
    if responders:
        for a in responders:
            print(f"    0x{a:02X}   {KNOWN.get(a, 'unknown device')}")
    else:
        print("    (none)")

    claimed = kernel_claimed(args.bus)
    if claimed:
        print("\n  held by a kernel driver (will not answer the scan above):")
        for a, name in claimed:
            print(f"    0x{a:02X}   {name}")

    missing = [a for a in KNOWN if a not in responders]
    if missing:
        print("\n  expected but ABSENT: "
              + ", ".join(f"0x{a:02X} ({KNOWN[a].split()[0]})" for a in missing))

    # ── 2 ─────────────────────────────────────────────────────────────────────
    rule(f"2. CLOCK FREQUENCY OF BUS {args.bus}")
    cfg = dt_clock_hz(args.bus)
    print(f"  configured   {cfg/1000:.0f} kHz  (device-tree clock-frequency)"
          if cfg else "  configured   not stated in the device tree")

    sweep_addr = next((a for a in SWEEP_PREFERENCE if a in responders),
                      responders[0] if responders else None)
    if sweep_addr is None:
        print("  measured     cannot measure — no device answers on this bus")
    else:
        print(f"  measuring    timing {SWEEP_REPEATS} reads at each of "
              f"{len(SWEEP_SIZES)} lengths against 0x{sweep_addr:02X}...",
              end="", flush=True)
        r = sweep_clock(handle, sweep_addr)
        print(" done\n")
        print(f"    {'bytes':>6}  {'fastest':>10}  {'median':>10}")
        for n, t in r["timings"].items():
            print(f"    {n:>6}  {t*1e3:>7.3f} ms  {r['medians'][n]*1e3:>7.3f} ms")
        print()
        print(f"  fixed cost   {r['fixed_s']*1e3:.3f} ms per transaction "
              f"(kernel + START/address/STOP)")
        print(f"  per byte     {r['per_byte_s']*1e3:.4f} ms  "
              f"→ {r['bytes_per_s']/1024:.1f} KB/s payload")
        print(f"  measured     {r['scl_hz']/1000:.0f} kHz SCL   (fit R² = {r['r2']:.4f})")
        if r["r2"] < 0.99:
            print("  ⚠  poor fit — something else was using this bus while it was "
                  "measured.\n     Re-run; the clock figure above is not "
                  "trustworthy.")
        busy = statistics.median(
            [r["medians"][n] / r["timings"][n] for n in r["timings"]])
        if busy > 1.5:
            print(f"  note         the median transfer took {busy:.1f}x the fastest "
                  f"one — this bus\n               is carrying other traffic "
                  f"(kernel drivers on shared addresses).")
        if cfg:
            print(f"  efficiency   {100*r['scl_hz']/cfg:.0f}% of the configured "
                  f"{cfg/1000:.0f} kHz")
            if r["scl_hz"] < 0.6 * cfg:
                print("  ⚠  well under the configured clock — the adapter is not "
                      "running at the rate the device tree asks for.")
        print(f"\n  Small reads are dominated by the {r['fixed_s']*1e3:.3f} ms fixed "
              f"cost, so the NUMBER of\n  transactions matters as much as the "
              f"number of bytes.")

    handle.close()

    if args.no_sensors:
        print("\n  (sections 3-4 skipped: --no-sensors)\n")
        return

    # ── 3 ─────────────────────────────────────────────────────────────────────
    rule(f"3. DATA RATE OF THE FULL SYSTEM — sensor_parse.SensorHub  "
         f"({args.duration:g} s)")
    try:
        sp  = _load_sensor_parse()
        res = measure_sensor_hub(args.bus, args.duration)
    except ImportError as exc:
        print(f"  Cannot import SensorHub from BNO085/sensor_parse.py: {exc}")
        return
    except KeyboardInterrupt:
        print("  Aborted during sensor init.")
        return

    note = _summarise_chatter(res.get("chatter", ""))
    if note:
        print(f"  library reported{note}")
    print(f"\n  measured over {res['elapsed']:.2f} s", end="")
    if res["stalls"]:
        print(f", of which {res['stall_s']:.2f} s was lost to {res['stalls']} "
              f"stall(s)\n  (a sensor error forces a reconnect that blocks the "
              f"loop). Rates are over\n  the {res['active']:.2f} s the loop was "
              f"actually running.")
    else:
        print(" with no stalls.")
    print()
    res["label"] = "all four"
    print_rate_table([res])

    r = rates(res)
    print(f"\n  ADC counts completed conversions; the two channels alternate through"
          f"\n  one converter, so each FSR updates at {r['adc']/2:.1f} Hz.")
    print(f"  IMU figures are USABLE rates — a report the loop did not collect"
          f"\n  before the next arrived was overwritten in the device cache and is"
          f"\n  not counted.  They are capped by the loop rate and by the report"
          f"\n  rate requested in sensor_parse.py (REPORT_HZ = {sp.REPORT_HZ:g} Hz),"
          f"\n  though the device quantises that interval and can land either side"
          f"\n  of it.")
    print_frame(res["frame"])

    # ── 4 ─────────────────────────────────────────────────────────────────────
    if args.no_matrix:
        print("\n  (section 4 skipped: --no-matrix)\n")
        return

    rule(f"4. EACH SENSOR ALONE, THEN IN COMBINATION  "
         f"({args.duration:g} s each, {len(combos)} runs)")
    print("  Same loop, same code, one variable: which devices are in it.  A solo"
          "\n  run is that device's ceiling on this bus; the difference between solo"
          "\n  and shared is contention and nothing else.")
    print(f"\n  The devices are connected once and shared by every run — see _Bench"
          f"\n  for why re-connecting between runs corrupts the comparison.  About"
          f"\n  {len(combos)*(args.duration+args.settle)/60:.1f} min.  Ctrl+C stops "
          f"early and still reports.\n")

    results = measure_matrix(args.bus, combos, args.duration, args.settle,
                             args.report_hz, args.max_packets)
    if not results:
        print("\n  No combination completed.\n")
        return

    print()
    print_rate_table(results)
    print_contention(results)
    print()


if __name__ == "__main__":
    main()
