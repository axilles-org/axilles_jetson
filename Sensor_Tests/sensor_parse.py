#!/usr/bin/env python3
"""
sensor_parse.py — Unified fast sensor reader for:
  • IMU-A   BNO085  bus 7  0x4A   (Game Rotation Vector + Accel + Gyro)
  • IMU-B   BNO085  bus 7  0x4B   (Game Rotation Vector + Accel + Gyro)
  • ADC     ADS1115 bus 7  0x48   (AIN0 = FSR-1, AIN1 = FSR-2)
  • Encoder AS5600  bus 7  0x36   (12-bit magnetic angle, 0–360°)

Speed strategy
──────────────
• IMUs    : single _process_available_packets() call per sensor drains ALL
            buffered SHTP packets in one I2C burst, then reads _readings cache
            directly.  ~800 Hz per IMU.
• ADS1115 : non-blocking single-shot mode.  Conversion is triggered then the
            main loop continues; the result is collected on the NEXT pass
            (conversion takes ~1.16 ms at 860 SPS, hidden behind other I²C
            work).  Alternates AIN0 ↔ AIN1 each cycle.  ~430 Hz per channel.
• AS5600  : 2-byte angle register read on every loop tick.  I²C-rate limited,
            ~1000+ Hz raw reads.
• All sensors on bus 7 share a single smbus2 handle for ADC + encoder, and
  separate adafruit ExtendedI2C handles for each BNO085 (required by the
  Adafruit SHTP driver).

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

import math
import sys
import time
import warnings

import smbus2

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="I2C frequency is not settable")

from adafruit_extended_bus import ExtendedI2C as ExtI2C
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_GAME_ROTATION_VECTOR,
)
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

# ── Hardware constants ─────────────────────────────────────────────────────────
I2C_BUS         = 1

IMU_A_ADDR      = 0x4A
IMU_B_ADDR      = 0x4B
ADS_ADDR        = 0x48
AS5600_ADDR     = 0x36

RECONNECT_DELAY = 1.0
BOOT_DELAY      = 0.8
FEATURE_RETRIES = 5

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

# AS5600 registers
_AS5600_REG_ANGLE = 0x0E   # ANGLE[11:8] MSB + 0x0F LSB (filtered output)


# ── BNO085 fast reader ─────────────────────────────────────────────────────────
class _FastIMU:
    """Single BNO085 — one _process_available_packets() call per read."""

    def __init__(self, bus: int, address: int, label: str):
        self._bus     = bus
        self._address = address
        self._label   = label
        self._i2c     = None
        self._bno     = None
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
                for feat in (BNO_REPORT_GAME_ROTATION_VECTOR,
                             BNO_REPORT_ACCELEROMETER,
                             BNO_REPORT_GYROSCOPE):
                    for attempt in range(FEATURE_RETRIES):
                        try:
                            self._bno.enable_feature(feat)
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

    def read(self):
        """One I2C burst → all caches updated → return dict or None."""
        try:
            self._bno._process_available_packets()
            accel = self._bno._readings.get(BNO_REPORT_ACCELEROMETER)
            gyro  = self._bno._readings.get(BNO_REPORT_GYROSCOPE)
            quat  = self._bno._readings.get(BNO_REPORT_GAME_ROTATION_VECTOR)
        except (OSError, RuntimeError, AttributeError, KeyError) as exc:
            print(f"\n[{self._label}] Error ({type(exc).__name__}: {exc}), "
                  f"reconnecting...", flush=True)
            self._reconnect()
            return None
        if accel is None or gyro is None or quat is None:
            return None
        return {"quat": quat, "accel": accel, "gyro": gyro}

    def close(self):
        try:
            self._i2c.deinit()
        except Exception:
            pass


# ── ADS1115 non-blocking reader ────────────────────────────────────────────────
class _ADS1115:
    """
    Non-blocking two-channel ADS1115 reader.

    Triggers a single-shot conversion on one channel, returns immediately.
    On the next call, checks the OS (ready) bit — if set, reads the result
    and triggers the other channel.  Conversion time (~1.16 ms) is hidden
    behind IMU reads and AS5600 reads in the main loop.
    """

    def __init__(self, bus_handle: smbus2.SMBus, address: int = ADS_ADDR):
        self._bus  = bus_handle
        self._addr = address
        self._ch   = 0          # channel currently converting: 0 or 1
        self._v    = [0.0, 0.0] # last valid voltages
        self._trigger(0)        # kick off first conversion immediately

    def _trigger(self, ch: int):
        """Write config to start single-shot conversion on AIN{ch}."""
        cfg = _ADS_CFG_AIN0 if ch == 0 else _ADS_CFG_AIN1
        self._bus.write_i2c_block_data(self._addr, _ADS_REG_CFG, cfg)
        self._ch = ch

    def _is_ready(self) -> bool:
        """Read OS bit from config register — True when conversion done."""
        data = self._bus.read_i2c_block_data(self._addr, _ADS_REG_CFG, 2)
        return bool(data[0] & 0x80)   # OS bit = bit 15 of config word

    def _read_raw(self) -> int:
        """Read 16-bit signed conversion register."""
        data = self._bus.read_i2c_block_data(self._addr, _ADS_REG_CONV, 2)
        raw  = (data[0] << 8) | data[1]
        # Two's complement
        return raw if raw < 0x8000 else raw - 0x10000

    def update(self):
        """
        Call once per main loop tick.
        If the current single-shot conversion is done, collect the result,
        store it, and trigger the opposite channel for the next pass.
        Non-blocking — returns immediately whether or not data was ready.
        """
        try:
            if self._is_ready():
                raw = self._read_raw()
                self._v[self._ch] = max(0.0, raw * _ADS_LSB_MV)
                next_ch = 1 - self._ch
                self._trigger(next_ch)
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
    """

    def __init__(self,
                 bus:      int = I2C_BUS,
                 imu_a:    int = IMU_A_ADDR,
                 imu_b:    int = IMU_B_ADDR,
                 ads_addr: int = ADS_ADDR,
                 enc_addr: int = AS5600_ADDR):

        print(f"[Hub] Connecting IMU-A (bus {bus}, 0x{imu_a:02X})...")
        self._imu_a = _FastIMU(bus, imu_a, "IMU-A")
        print(f"[Hub] IMU-A connected.")

        print(f"[Hub] Connecting IMU-B (bus {bus}, 0x{imu_b:02X})...")
        self._imu_b = _FastIMU(bus, imu_b, "IMU-B")
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
        # IMUs: drain all buffered SHTP packets in one call each
        sa = self._imu_a.read()
        sb = self._imu_b.read()

        # ADS1115: collect last conversion + trigger next (non-blocking)
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
                        if s is None:
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
