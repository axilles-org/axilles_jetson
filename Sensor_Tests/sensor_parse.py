#!/usr/bin/env python3
"""
sensor_parse.py — Unified fast sensor reader for:
  • IMU-A   BNO085  bus 1  0x4A   (Game Rotation Vector + Accel + Gyro)
  • IMU-B   BNO085  bus 1  0x4B   (Game Rotation Vector + Accel + Gyro)
  • ADC     ADS1115 bus 1  0x48   (AIN0 = FSR-1, AIN1 = FSR-2)
  • Encoder AS5600  bus 1  0x36   (12-bit magnetic angle, 0–360°)

WHAT LIMITS THE RATE  (measured on this Jetson, not estimated)
──────────────────────────────────────────────────────────────
Every device shares one I²C adapter (c240000.i2c) clocked at 98 kHz.  Timing
raw reads against the AS5600 gives the cost of any transaction on this bus:

        t = 0.386 ms  +  0.0914 ms per byte          (≈ 10.9 KB/s)

Two things follow, and they drive every design choice below.

1.  The fixed 0.386 ms dominates small reads, so the NUMBER of transactions
    matters as much as the number of bytes.
2.  The bus is a serial resource shared by all four devices.  Threading cannot
    create bandwidth — the kernel's i2c-tegra driver serialises every transfer
    behind a per-adapter mutex.  Measured: threads buy 3-4 % and make the ADC
    and encoder worse.  This module is deliberately single-threaded.

THREE FIXES APPLIED HERE
────────────────────────
• Drain: adafruit's _process_available_packets() spends THREE transactions per
  packet — it reads the 4-byte header to test _data_ready, reads it again in
  _read_packet, then re-reads header+payload.  _drain() below does TWO (one
  header probe, one full read).  Measured on one IMU: 40.7 Hz → 226.7 Hz.

• Bounded drain: the library drains until the FIFO is empty.  When the IMU
  produces faster than the bus can carry, that loop never exits and the ADC and
  encoder are starved to zero (measured: ADC 0 Hz, encoder 0 Hz at a 200 Hz
  report rate).  _drain() takes at most `max_packets` per call, so every device
  keeps getting bus time.

• ADC: the OS-ready bit was polled with an extra transaction every tick.  The
  conversion time is known (1/860 s), so readiness is now timed instead —
  2 transactions per sample rather than 3.

REPORT RATE: ASK FOR LESS, GET MORE
───────────────────────────────────
The BNO085 keeps only the newest report of each type (adafruit caches into a
dict, so a batch of five gyro samples collapses to one).  Usable sample rate is
therefore min(loop_rate, report_rate) — requesting a report rate above the loop
rate just spends bus time on samples that are overwritten before you read them.

The device also quantises the requested interval.  Asking for >= 70 Hz pushes
the accelerometer to 125 Hz and the extra traffic COSTS you loop rate:

    3 reports @ 60 Hz  ->  loop 79.2 Hz   7.7 KB/s     <- best
    3 reports @ 70 Hz  ->  loop 61.4 Hz   8.5 KB/s
    3 reports @ 90 Hz  ->  loop 62.7 Hz   8.5 KB/s
    3 reports @200 Hz  ->  loop 42.7 Hz   9.6 KB/s  (ADC 21 Hz, enc 43 Hz)

Hence REPORT_HZ = 60 by default.  Against the old default (a 50 ms library
interval nobody overrode) this is 21 Hz -> 79 Hz of usable IMU data, ~4x.

If you do not need orientation, enable only ACCEL+GYRO and pass report_hz=100:
that reaches ~98 Hz usable on 7.1 KB/s.  Note the Euler angles ARE the
quaternion — quat_to_euler() converts the Game Rotation Vector — so dropping
GAME_ROTATION_VECTOR drops roll/pitch/yaw with it.

200 Hz IS NOT REACHABLE ON THIS BUS.  Two IMUs at 200 Hz with three reports
each need ~230 % of a 100 kHz bus.  It needs a 400 kHz bus; bus 7
(c250000.i2c) is already clocked at 400 kHz and is currently unused.

──────────────────────────────────────────────────────────────────────────────
STANDALONE:
    python3 sensor_parse.py

IMPORT INTO OTHER SCRIPTS:
    from sensor_parse import SensorHub

    hub = SensorHub()
    for frame in hub.stream():
        qi_a, qj_a, qk_a, qr_a = frame["imu_a"]["quat"]
        qi_b, qj_b, qk_b, qr_b = frame["imu_b"]["quat"]
        fsr1_v = frame["fsr1"]          # volts  (AIN0, 0–3.3 V)
        fsr2_v = frame["fsr2"]          # volts  (AIN1, 0–3.3 V)
        angle  = frame["angle_deg"]     # degrees (0–360)
    hub.close()
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import warnings
from struct import unpack_from
import math
import smbus2

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="I2C frequency is not settable")

from adafruit_extended_bus import ExtendedI2C as ExtI2C
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    Packet,
    PacketError,
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_GAME_ROTATION_VECTOR,
)
# ── Hardware constants ─────────────────────────────────────────────────────────
I2C_BUS         = 1

IMU_A_ADDR      = 0x4A
IMU_B_ADDR      = 0x4B
ADS_ADDR        = 0x48
AS5600_ADDR     = 0x36

RECONNECT_DELAY = 1.0
BOOT_DELAY      = 0.8
FEATURE_RETRIES = 5

# Sensor report rate requested from the BNO085.  See the module docstring for
# why 60 beats 70-200 on a 98 kHz bus.
REPORT_HZ       = 60.0

# Packets consumed per IMU per read() call.  1 keeps the ADC and encoder fed;
# raising it favours the IMUs at their expense.
MAX_PACKETS     = 1

DEFAULT_FEATURES = (
    BNO_REPORT_GAME_ROTATION_VECTOR,   # source of roll/pitch/yaw
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
)

# ADS1115 registers
_ADS_REG_CONV   = 0x00   # conversion result
_ADS_REG_CFG    = 0x01   # configuration

# ADS1115 config words for single-shot at 860 SPS, PGA=±4.096 V
# Bit layout: OS MUX[2:0] PGA[2:0] MODE  DR[2:0] COMP_MODE COMP_POL COMP_LAT COMP_QUE[1:0]
# AIN0 (FSR-1): 1 100 001 1  111 0 0 0 11  → 0xC3E3
# AIN1 (FSR-2): 1 101 001 1  111 0 0 0 11  → 0xD3E3
_ADS_CFG_AIN0   = [0xC3, 0xE3]
_ADS_CFG_AIN1   = [0xD3, 0xE3]

# ADS1115 full-scale for PGA=±4.096 V (32767 counts = 4.096 V)
_ADS_LSB_MV     = 4.096 / 32767.0

# Conversion time at 860 SPS, plus a small margin for oscillator tolerance.
_ADS_CONV_S     = 1.0 / 860.0 + 0.00015

# AS5600 registers
_AS5600_REG_ANGLE = 0x0E   # ANGLE[11:8] MSB + 0x0F LSB (filtered output)

_SHTP_CHANNEL_CONTROL = 2
# Help function

def quat_to_euler(qi, qj, qk, qr):
    """Convert quaternion (i, j, k, real) → (roll, pitch, yaw) in degrees."""
    sinr = 2.0 * (qr * qi + qj * qk)
    cosr = 1.0 - 2.0 * (qi * qi + qj * qj)
    roll  = math.degrees(math.atan2(sinr, cosr))

    sinp  = max(-1.0, min(1.0, 2.0 * (qr * qj - qk * qi)))
    pitch = math.degrees(math.asin(sinp))

    siny = 2.0 * (qr * qk + qi * qj)
    cosy = 1.0 - 2.0 * (qj * qj + qk * qk)
    yaw  = math.degrees(math.atan2(siny, cosy))

    return roll, pitch, yaw

# ── BNO085 fast reader ─────────────────────────────────────────────────────────
class _FastIMU:
    """Single BNO085 read with a bounded, two-transaction packet drain."""

    def __init__(self, bus: int, address: int, label: str,
                 report_hz: float = REPORT_HZ,
                 features=DEFAULT_FEATURES,
                 max_packets: int = MAX_PACKETS):
        self._bus         = bus
        self._address     = address
        self._label       = label
        self._report_hz   = report_hz
        self._features    = tuple(features)
        self._max_packets = max(1, int(max_packets))
        self._i2c         = None
        self._bno         = None
        self._connect()

    def _connect(self):
        while True:
            try:
                if self._i2c is not None:
                    try:
                        self._i2c.deinit()
                    except Exception:
                        pass
                    time.sleep(0.3)
                self._i2c = ExtI2C(self._bus)
                self._bno = BNO08X_I2C(self._i2c, address=self._address)
                time.sleep(BOOT_DELAY)

                # The library default is 50 000 us (20 Hz) and nothing overrode
                # it, which is why the IMUs used to sit at 20 Hz.
                interval = max(1000, int(1_000_000.0 / self._report_hz))
                for feat in self._features:
                    for attempt in range(FEATURE_RETRIES):
                        try:
                            self._bno.enable_feature(feat, interval)
                            break
                        except Exception:
                            if attempt == FEATURE_RETRIES - 1:
                                raise
                            time.sleep(0.2)
                return
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[{self._label}] Connect failed ({exc}), "
                      f"retrying in {RECONNECT_DELAY}s...", flush=True)
                time.sleep(RECONNECT_DELAY)

    def _reconnect(self):
        print(f"\n[{self._label}] Reconnecting...", flush=True)
        self._connect()
        print(f"[{self._label}] Reconnected.\n", flush=True)

    def _drain(self):
        """Consume up to max_packets SHTP packets, 2 I2C transactions each.

        The library's version costs 3 transactions per packet and loops until
        the FIFO empties; both are fixed here.
        """
        bno = self._bno
        for _ in range(self._max_packets):
            buf = bno._data_buffer
            with bno.bus_device_obj as i2c:
                i2c.readinto(buf, end=4)               # header probe
            packet_byte_count = unpack_from("<H", buf)[0] & ~0x8000
            if packet_byte_count <= 4 or packet_byte_count == 0x7FFF:
                return                                  # nothing queued
            if packet_byte_count > len(buf):
                bno._data_buffer = bytearray(packet_byte_count)
                buf = bno._data_buffer
            with bno.bus_device_obj as i2c:
                i2c.readinto(buf, end=packet_byte_count)   # header + payload
            try:
                bno._handle_packet(Packet(buf))
            except (PacketError, KeyError, IndexError):
                pass                                    # skip a malformed packet

    def read(self):
        """One bounded drain → caches updated → return dict or None."""
        try:
            self._drain()
            accel = self._bno._readings.get(BNO_REPORT_ACCELEROMETER)
            gyro  = self._bno._readings.get(BNO_REPORT_GYROSCOPE)
            quat  = self._bno._readings.get(BNO_REPORT_GAME_ROTATION_VECTOR)
        except (OSError, RuntimeError, AttributeError, KeyError) as exc:
            print(f"\n[{self._label}] Error ({type(exc).__name__}: {exc}), "
                  f"reconnecting...", flush=True)
            self._reconnect()
            return None

        out = {}
        if BNO_REPORT_GAME_ROTATION_VECTOR in self._features:
            if quat is None:
                return None
            out["quat"] = quat
        if BNO_REPORT_ACCELEROMETER in self._features:
            if accel is None:
                return None
            out["accel"] = accel
        if BNO_REPORT_GYROSCOPE in self._features:
            if gyro is None:
                return None
            out["gyro"] = gyro
        return out or None

    def close(self):
        try:
            self._i2c.deinit()
        except Exception:
            pass


# ── ADS1115 non-blocking reader ────────────────────────────────────────────────
class _ADS1115:
    """
    Non-blocking two-channel ADS1115 reader.

    Triggers a single-shot conversion on one channel and returns immediately.
    Readiness is decided from elapsed time rather than by polling the OS bit,
    which removes one I2C transaction per sample (3 → 2).  Conversion takes
    ~1.16 ms at 860 SPS and is hidden behind the IMU and encoder reads.
    """

    def __init__(self, bus_handle: smbus2.SMBus, address: int = ADS_ADDR):
        self._bus     = bus_handle
        self._addr    = address
        self._ch      = 0          # channel currently converting: 0 or 1
        self._v       = [0.0, 0.0] # last valid voltages
        self._t_trig  = 0.0
        self._trigger(0)           # kick off first conversion immediately

    def _trigger(self, ch: int):
        """Write config to start single-shot conversion on AIN{ch}."""
        cfg = _ADS_CFG_AIN0 if ch == 0 else _ADS_CFG_AIN1
        self._bus.write_i2c_block_data(self._addr, _ADS_REG_CFG, cfg)
        self._ch     = ch
        self._t_trig = time.perf_counter()

    def _read_raw(self) -> int:
        """Read 16-bit signed conversion register."""
        data = self._bus.read_i2c_block_data(self._addr, _ADS_REG_CONV, 2)
        raw  = (data[0] << 8) | data[1]
        # Two's complement
        return raw if raw < 0x8000 else raw - 0x10000

    def update(self):
        """
        Call once per main loop tick.
        If the conversion has had time to finish, collect the result and trigger
        the opposite channel.  Non-blocking.
        """
        try:
            if time.perf_counter() - self._t_trig < _ADS_CONV_S:
                return
            raw = self._read_raw()
            self._v[self._ch] = max(0.0, raw * _ADS_LSB_MV)
            self._trigger(1 - self._ch)
        except OSError:
            pass   # I2C glitch — silently retry next tick

    @property
    def fsr1(self) -> float:
        """Voltage on AIN0 (FSR-1), volts."""
        return self._v[0]

    @property
    def fsr2(self) -> float:
        """Voltage on AIN1 (FSR-2), volts."""
        return self._v[1]


# ── AS5600 encoder reader ──────────────────────────────────────────────────────
class _AS5600:
    """12-bit magnetic angle from AS5600 — direct 2-byte register read."""

    def __init__(self, bus_handle: smbus2.SMBus, address: int = AS5600_ADDR):
        self._bus      = bus_handle
        self._addr     = address
        self._angle    = 0.0
        self._raw      = 0

    def update(self):
        """Read the 12-bit filtered angle register. Non-blocking."""
        try:
            data        = self._bus.read_i2c_block_data(
                              self._addr, _AS5600_REG_ANGLE, 2)
            self._raw   = ((data[0] & 0x0F) << 8) | data[1]
            self._angle = self._raw * (360.0 / 4096.0)
        except OSError:
            pass   # I2C glitch — keep last value

    @property
    def angle_deg(self) -> float:
        """Raw angle in degrees (0–360)."""
        return self._angle

    @property
    def raw(self) -> int:
        """Raw 12-bit angle count (0–4095)."""
        return self._raw


# ── SensorHub ──────────────────────────────────────────────────────────────────
class SensorHub:
    """
    Unified reader for all four sensors.

    Usage (context manager):
        with SensorHub() as hub:
            for frame in hub.stream():
                ...

    Usage (manual):
        hub = SensorHub()
        frame = hub.read()
        hub.close()

    Frame dict keys:
        imu_a       dict{"quat","accel","gyro"} or None
        imu_b       dict{"quat","accel","gyro"} or None
        fsr1        float  volts (AIN0)
        fsr2        float  volts (AIN1)
        angle_deg   float  degrees 0–360
        angle_raw   int    raw 12-bit count 0–4095

    Tunables:
        report_hz   BNO085 report rate.  60 measured best; see module docstring.
        features    Which BNO085 reports to enable.  Dropping
                    GAME_ROTATION_VECTOR also drops roll/pitch/yaw.
        max_packets Packets drained per IMU per read().  1 keeps the ADC and
                    encoder from being starved.
    """

    def __init__(self,
                 bus:         int = I2C_BUS,
                 imu_a:       int = IMU_A_ADDR,
                 imu_b:       int = IMU_B_ADDR,
                 ads_addr:    int = ADS_ADDR,
                 enc_addr:    int = AS5600_ADDR,
                 report_hz:   float = REPORT_HZ,
                 features           = DEFAULT_FEATURES,
                 max_packets: int = MAX_PACKETS):

        print(f"[Hub] Connecting IMU-A (bus {bus}, 0x{imu_a:02X}) "
              f"@ {report_hz:g} Hz...")
        self._imu_a = _FastIMU(bus, imu_a, "IMU-A", report_hz, features, max_packets)
        print(f"[Hub] IMU-A connected.")

        print(f"[Hub] Connecting IMU-B (bus {bus}, 0x{imu_b:02X}) "
              f"@ {report_hz:g} Hz...")
        self._imu_b = _FastIMU(bus, imu_b, "IMU-B", report_hz, features, max_packets)
        print(f"[Hub] IMU-B connected.")

        # Shared smbus2 handle for ADS1115 + AS5600
        print(f"[Hub] Opening smbus2 on bus {bus} for ADC + encoder...")
        self._smbus = smbus2.SMBus(bus)

        print(f"[Hub] Connecting ADS1115 (0x{ads_addr:02X})...")
        self._ads = _ADS1115(self._smbus, ads_addr)
        print(f"[Hub] ADS1115 connected.")

        print(f"[Hub] Connecting AS5600 (0x{enc_addr:02X})...")
        self._enc = _AS5600(self._smbus, enc_addr)
        print(f"[Hub] AS5600 connected.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._imu_a.close()
        self._imu_b.close()
        try:
            self._smbus.close()
        except Exception:
            pass

    def read(self) -> dict:
        """
        One poll cycle — reads all sensors and returns a frame dict.
        Call in a tight loop; never sleeps.
        """
        # IMUs: bounded two-transaction drain, so the ADC and encoder below
        # still get bus time even when the IMUs have data queued.
        sa = self._imu_a.read()
        sb = self._imu_b.read()

        # ADS1115: collect finished conversion + trigger next (non-blocking)
        self._ads.update()

        # AS5600: 2-byte register read
        self._enc.update()

        return {
            "imu_a":     sa,
            "imu_b":     sb,
            "fsr1":      self._ads.fsr1,
            "fsr2":      self._ads.fsr2,
            "angle_deg": self._enc.angle_deg,
            "angle_raw": self._enc.raw,
        }

    def stream(self):
        """Infinite generator — yields frames at max I²C rate, never sleeps."""
        while True:
            yield self.read()


# ── Standalone entry point ─────────────────────────────────────────────────────
def main():
    try:
        hub = SensorHub()
    except KeyboardInterrupt:
        print("\nAborted during init.")
        return

    print("\nAll sensors live. Ctrl+C to stop.\n")

    last_display = time.perf_counter()
    DISP_INTERVAL = 0.25
    count = 0
    t_start = time.perf_counter()

    # Print header
    print(f"{'Roll-A':>7} {'Pitch-A':>8} {'Yaw-A':>7}  |  "
          f"{'Roll-B':>7} {'Pitch-B':>8} {'Yaw-B':>7}  |  "
          f"{'FSR1(V)':>8} {'FSR2(V)':>8}  |  "
          f"{'Angle°':>8}")
    print("-" * 85)

    try:
        with hub:
            for frame in hub.stream():
                now = time.perf_counter()
                count += 1

                sa = frame["imu_a"]
                sb = frame["imu_b"]

                if now - last_display >= DISP_INTERVAL:
                    last_display = now

                    def rpy(s):
                        if s is None or "quat" not in s:
                            return "     ---      ---      ---"
                        qi, qj, qk, qr = s["quat"]
                        r, p, y = quat_to_euler(qi, qj, qk, qr)
                        return f"{r:7.2f}  {p:8.2f}  {y:7.2f}"

                    sys.stdout.write(
                        f"\r{rpy(sa)}  |  "
                        f"{rpy(sb)}  |  "
                        f"{frame['fsr1']:8.4f} {frame['fsr2']:8.4f}  |  "
                        f"{frame['angle_deg']:8.2f}   "
                    )
                    sys.stdout.flush()

    except KeyboardInterrupt:
        elapsed = time.perf_counter() - t_start
        print(f"\n\nStopped after {elapsed:.1f}s — {count} frames "
              f"({count/elapsed:.1f} Hz loop rate)")


if __name__ == "__main__":
    main()
