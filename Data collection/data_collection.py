#!/usr/bin/env python3
"""
data_collection.py

Standalone CSV logger for:
- IMU foot  (BNO085): accel(3) + gyro(3) + quaternion
- IMU shank (BNO085): accel(3) + gyro(3) + quaternion
- Ankle encoder (AS5600): angle (deg)
- Toe FSR (ADS1115 AIN0): raw ADC counts
- Heel FSR (ADS1115 AIN1): raw ADC counts

Run:
    python3 "Data collection/data_collection.py"
"""

from __future__ import annotations

import csv
import signal
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import smbus2  # type: ignore[import-not-found]

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="I2C frequency is not settable",
)

from adafruit_bno08x import (  # type: ignore[import-not-found]
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GAME_ROTATION_VECTOR,
    BNO_REPORT_GYROSCOPE,
)
from adafruit_bno08x.i2c import BNO08X_I2C  # type: ignore[import-not-found]
from adafruit_extended_bus import ExtendedI2C as ExtI2C  # type: ignore[import-not-found]


# ========================= User Config =========================
# Shared publish/logging target rate (Hz)
SENSOR_HZ = 200.0

# Background worker rates are derived from SENSOR_HZ.
# Increase WORKER_RATE_SCALE slightly (>1.0) only if fresh samples lag.
WORKER_RATE_SCALE = 1.0
WORKER_HZ = SENSOR_HZ * WORKER_RATE_SCALE

# IMU hardware report generation rate (Hz).
IMU_REPORT_HZ = 25.0

# I2C configuration (explicit sensor addresses)
I2C_BUS = 1
IMU_FOOT_ADDR = 0x4A
IMU_SHANK_ADDR = 0x4B
ADS_ADDR = 0x48
AS5600_ADDR = 0x36

# IMU reconnect and setup behavior
RECONNECT_DELAY = 1.0
BOOT_DELAY = 0.8
FEATURE_RETRIES = 5

# FSR key mapping from standalone SensorHub frame
TOE_FSR_KEY = "toe_fsr_raw"
HEEL_FSR_KEY = "heel_fsr_raw"

# Output file naming
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_PREFIX = "data_collection"

# Auto-stop behavior (seconds). Set <= 0 to disable auto-stop.
AUTO_STOP_SECONDS = 10.0

# Optional diagnostics; disable to minimize runtime overhead.
DIAGNOSTIC_METRICS = False
# ==============================================================


# ADS1115 registers/config
_ADS_REG_CONV = 0x00
_ADS_REG_CFG = 0x01
_ADS_CFG_AIN0 = [0xC3, 0xE3]
_ADS_CFG_AIN1 = [0xD3, 0xE3]

# AS5600 registers
_AS5600_REG_ANGLE = 0x0E


class _FastIMU:
    """Single BNO085 using packet-drain read for high throughput."""

    def __init__(self, bus: int, address: int, label: str, report_hz: float):
        self._bus = bus
        self._address = address
        self._label = label
        self._report_hz = report_hz
        self._i2c = None
        self._bno = None
        self._connect()

    def _connect(self) -> None:
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

                report_interval_us = 0
                if self._report_hz > 0:
                    report_interval_us = max(1000, int(1_000_000.0 / self._report_hz))

                for feat in (
                    BNO_REPORT_GAME_ROTATION_VECTOR,
                    BNO_REPORT_ACCELEROMETER,
                    BNO_REPORT_GYROSCOPE,
                ):
                    for attempt in range(FEATURE_RETRIES):
                        try:
                            if report_interval_us > 0:
                                self._bno.enable_feature(feat, report_interval_us)
                            else:
                                self._bno.enable_feature(feat)
                            break
                        except TypeError:
                            # Some library versions only accept feature_id.
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
                print(
                    f"[{self._label}] Connect failed ({exc}), retrying in {RECONNECT_DELAY}s...",
                    flush=True,
                )
                time.sleep(RECONNECT_DELAY)

    def _reconnect(self) -> None:
        print(f"\n[{self._label}] Reconnecting...", flush=True)
        self._connect()
        print(f"[{self._label}] Reconnected.\n", flush=True)

    def read(self) -> Optional[dict]:
        try:
            self._bno._process_available_packets(max_packets=1)
            accel = self._bno._readings.get(BNO_REPORT_ACCELEROMETER)
            gyro = self._bno._readings.get(BNO_REPORT_GYROSCOPE)
            quat = self._bno._readings.get(BNO_REPORT_GAME_ROTATION_VECTOR)
        except KeyError:
            # Ignore unknown sensor report IDs emitted by some firmware versions.
            return None
        except (OSError, RuntimeError, AttributeError) as exc:
            print(
                f"\n[{self._label}] Error ({type(exc).__name__}: {exc}), reconnecting...",
                flush=True,
            )
            self._reconnect()
            return None

        if accel is None or gyro is None or quat is None:
            return None

        return {"quat": quat, "accel": accel, "gyro": gyro}

    def close(self) -> None:
        try:
            self._i2c.deinit()
        except Exception:
            pass


class _ADS1115:
    """Non-blocking 2-channel ADS1115 reader in single-shot mode."""

    def __init__(self, bus_handle: smbus2.SMBus, address: int = ADS_ADDR):
        self._bus = bus_handle
        self._addr = address
        self._ch = 0
        self._raw = [0, 0]
        self._trigger(0)

    def _trigger(self, ch: int) -> None:
        cfg = _ADS_CFG_AIN0 if ch == 0 else _ADS_CFG_AIN1
        self._bus.write_i2c_block_data(self._addr, _ADS_REG_CFG, cfg)
        self._ch = ch

    def _is_ready(self) -> bool:
        data = self._bus.read_i2c_block_data(self._addr, _ADS_REG_CFG, 2)
        return bool(data[0] & 0x80)

    def _read_raw(self) -> int:
        data = self._bus.read_i2c_block_data(self._addr, _ADS_REG_CONV, 2)
        raw = (data[0] << 8) | data[1]
        return raw if raw < 0x8000 else raw - 0x10000

    def update(self) -> None:
        try:
            if self._is_ready():
                raw = self._read_raw()
                self._raw[self._ch] = raw
                self._trigger(1 - self._ch)
        except OSError:
            pass

    @property
    def toe_fsr_raw(self) -> int:
        return self._raw[0]

    @property
    def heel_fsr_raw(self) -> int:
        return self._raw[1]


class _AS5600:
    """AS5600 angle reader."""

    def __init__(self, bus_handle: smbus2.SMBus, address: int = AS5600_ADDR):
        self._bus = bus_handle
        self._addr = address
        self._angle = 0.0

    def update(self) -> None:
        try:
            data = self._bus.read_i2c_block_data(self._addr, _AS5600_REG_ANGLE, 2)
            raw = ((data[0] & 0x0F) << 8) | data[1]
            self._angle = raw * (360.0 / 4096.0)
        except OSError:
            pass

    @property
    def angle_deg(self) -> float:
        return self._angle


class SensorHub:
    """Standalone I2C sensor hub for foot/shank IMUs, encoder, and FSR ADC."""

    def __init__(
        self,
        bus: int = I2C_BUS,
        imu_foot_addr: int = IMU_FOOT_ADDR,
        imu_shank_addr: int = IMU_SHANK_ADDR,
        ads_addr: int = ADS_ADDR,
        enc_addr: int = AS5600_ADDR,
    ):
        print(f"[Hub] Connecting IMU-foot (bus {bus}, 0x{imu_foot_addr:02X})...")
        self._imu_foot = _FastIMU(bus, imu_foot_addr, "IMU-foot", IMU_REPORT_HZ)
        print("[Hub] IMU-foot connected.")

        print(f"[Hub] Connecting IMU-shank (bus {bus}, 0x{imu_shank_addr:02X})...")
        self._imu_shank = _FastIMU(bus, imu_shank_addr, "IMU-shank", IMU_REPORT_HZ)
        print("[Hub] IMU-shank connected.")

        print(f"[Hub] Opening smbus2 on bus {bus} for ADS1115 + AS5600...")
        self._smbus = smbus2.SMBus(bus)

        print(f"[Hub] Connecting ADS1115 (0x{ads_addr:02X})...")
        self._ads = _ADS1115(self._smbus, ads_addr)
        print("[Hub] ADS1115 connected.")

        print(f"[Hub] Connecting AS5600 (0x{enc_addr:02X})...")
        self._enc = _AS5600(self._smbus, enc_addr)
        print("[Hub] AS5600 connected.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self._imu_foot.close()
        self._imu_shank.close()
        try:
            self._smbus.close()
        except Exception:
            pass

    def read_imus(self) -> dict:
        imu_foot = self._imu_foot.read()
        imu_shank = self._imu_shank.read()
        return {
            "imu_foot": imu_foot,
            "imu_shank": imu_shank,
        }

    def read_imu_foot(self) -> Optional[dict]:
        return self._imu_foot.read()

    def read_imu_shank(self) -> Optional[dict]:
        return self._imu_shank.read()

    def read_fsr(self) -> dict:
        self._ads.update()
        return {
            "toe_fsr_raw": self._ads.toe_fsr_raw,
            "heel_fsr_raw": self._ads.heel_fsr_raw,
        }

    def read_encoder(self) -> dict:
        self._enc.update()
        return {
            "ankle_encoder_deg": self._enc.angle_deg,
        }


def _period_from_hz(hz: float) -> float:
    if hz <= 0:
        return float("inf")
    return 1.0 / hz


def _empty_or(v: Optional[float]) -> str | float:
    return "" if v is None else v


def _update_imu_fields(latest: Dict[str, Optional[float]], imu_packet: Optional[dict], prefix: str) -> None:
    if imu_packet is None:
        return

    accel = imu_packet.get("accel")
    gyro = imu_packet.get("gyro")
    quat = imu_packet.get("quat")

    if accel is not None and len(accel) >= 3:
        latest[f"{prefix}_ax"] = accel[0]
        latest[f"{prefix}_ay"] = accel[1]
        latest[f"{prefix}_az"] = accel[2]

    if gyro is not None and len(gyro) >= 3:
        latest[f"{prefix}_gx"] = gyro[0]
        latest[f"{prefix}_gy"] = gyro[1]
        latest[f"{prefix}_gz"] = gyro[2]

    if quat is not None and len(quat) >= 4:
        qi, qj, qk, qr = quat[0], quat[1], quat[2], quat[3]
        latest[f"{prefix}_qi"] = qi
        latest[f"{prefix}_qj"] = qj
        latest[f"{prefix}_qk"] = qk
        latest[f"{prefix}_qr"] = qr


def _build_header() -> list[str]:
    header = ["timestamp_s"]

    for prefix in ("foot", "shank"):
        header.extend(
            [
                f"{prefix}_ax",
                f"{prefix}_ay",
                f"{prefix}_az",
                f"{prefix}_gx",
                f"{prefix}_gy",
                f"{prefix}_gz",
            ]
        )
        header.extend([f"{prefix}_qi", f"{prefix}_qj", f"{prefix}_qk", f"{prefix}_qr"])

    header.extend(["ankle_encoder_deg", "toe_fsr_raw", "heel_fsr_raw"])
    return header


def _periodic_worker(stop_event: threading.Event, period_s: float, task) -> None:
    if period_s == float("inf"):
        return

    next_t = time.perf_counter()
    while not stop_event.is_set():
        now = time.perf_counter()
        if now < next_t:
            # Event-based wait reduces CPU spin while preserving wake-up precision.
            stop_event.wait(timeout=min(0.001, next_t - now))
            continue

        task()
        next_t += period_s
        if next_t <= now:
            next_t = now + period_s


def main() -> None:
    mode = "quat"

    log_period = _period_from_hz(SENSOR_HZ)
    worker_period = _period_from_hz(WORKER_HZ)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{timestamp_tag}.csv"

    stop_requested = False

    def _handle_stop(_sig, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    header = _build_header()
    latest: Dict[str, Optional[float]] = {k: None for k in header}

    print(f"[Logger] Writing CSV to: {out_file}")
    print(
        f"[Logger] Rates -> publish: {SENSOR_HZ} Hz | worker: {WORKER_HZ:.1f} Hz | "
        f"IMU orientation mode: {mode}"
    )

    elapsed_s = 0.0
    imu_foot_polls = 0
    imu_shank_polls = 0
    enc_updates = 0
    fsr_updates = 0
    imu_foot_parses = 0
    imu_shank_parses = 0
    enc_parses = 0
    fsr_parses = 0
    logger_overruns = 0
    stop_reason = "signal"

    with SensorHub() as hub, out_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        timestamp_idx = header.index("timestamp_s")

        t0 = time.perf_counter()
        next_log_t = t0
        target_rows = 0
        if AUTO_STOP_SECONDS > 0:
            target_rows = max(1, int(round(AUTO_STOP_SECONDS * SENSOR_HZ)))

        cache_lock = threading.Lock()
        cache_imu_foot: Optional[dict] = None
        cache_imu_shank: Optional[dict] = None
        cache_enc: Optional[float] = None
        cache_toe: Optional[float] = None
        cache_heel: Optional[float] = None

        stop_event = threading.Event()

        def _imu_foot_task() -> None:
            nonlocal imu_foot_polls, cache_imu_foot
            imu_packet = hub.read_imu_foot()
            with cache_lock:
                imu_foot_polls += 1
                if imu_packet is not None:
                    cache_imu_foot = imu_packet

        def _imu_shank_task() -> None:
            nonlocal imu_shank_polls, cache_imu_shank
            imu_packet = hub.read_imu_shank()
            with cache_lock:
                imu_shank_polls += 1
                if imu_packet is not None:
                    cache_imu_shank = imu_packet

        def _enc_task() -> None:
            nonlocal enc_updates, cache_enc
            frame = hub.read_encoder()
            with cache_lock:
                cache_enc = frame.get("ankle_encoder_deg")
                enc_updates += 1

        def _fsr_task() -> None:
            nonlocal fsr_updates, cache_toe, cache_heel
            frame = hub.read_fsr()
            with cache_lock:
                cache_toe = frame.get(TOE_FSR_KEY)
                cache_heel = frame.get(HEEL_FSR_KEY)
                fsr_updates += 1

        workers = [
            threading.Thread(
                target=_periodic_worker,
                args=(stop_event, worker_period, _imu_foot_task),
                daemon=True,
            ),
            threading.Thread(
                target=_periodic_worker,
                args=(stop_event, worker_period, _imu_shank_task),
                daemon=True,
            ),
            threading.Thread(
                target=_periodic_worker,
                args=(stop_event, worker_period, _enc_task),
                daemon=True,
            ),
            threading.Thread(
                target=_periodic_worker,
                args=(stop_event, worker_period, _fsr_task),
                daemon=True,
            ),
        ]

        for worker in workers:
            worker.start()

        rows_written = 0

        try:
            while not stop_requested:
                now = time.perf_counter()
                if target_rows > 0 and rows_written >= target_rows:
                    stop_requested = True
                    stop_reason = "timeout"
                    break

                if now < next_log_t:
                    stop_event.wait(timeout=min(0.001, next_log_t - now))
                    continue

                with cache_lock:
                    _update_imu_fields(latest, cache_imu_foot, "foot")
                    _update_imu_fields(latest, cache_imu_shank, "shank")
                    latest["ankle_encoder_deg"] = cache_enc
                    latest["toe_fsr_raw"] = cache_toe
                    latest["heel_fsr_raw"] = cache_heel

                row = [_empty_or(latest.get(k)) for k in header]
                row[timestamp_idx] = now - t0
                writer.writerow(row)
                rows_written += 1
                imu_foot_parses += 1
                imu_shank_parses += 1
                enc_parses += 1
                fsr_parses += 1

                next_log_t += log_period
                if next_log_t <= now:
                    logger_overruns += 1
                    next_log_t = now + log_period
        finally:
            stop_event.set()
            for worker in workers:
                worker.join(timeout=1.0)

        elapsed_s = max(0.0, time.perf_counter() - t0)

    if stop_reason == "timeout":
        print(f"[Logger] Auto-stop reached at {AUTO_STOP_SECONDS:.3f} s")
    print(f"[Logger] Stopped. Rows written: {rows_written}")
    if elapsed_s > 0.0:
        imu_foot_poll_rate = imu_foot_polls / elapsed_s
        imu_shank_poll_rate = imu_shank_polls / elapsed_s
        enc_rate = enc_updates / elapsed_s
        fsr_rate = fsr_updates / elapsed_s
        logger_rate = rows_written / elapsed_s
    else:
        imu_foot_poll_rate = 0.0
        imu_shank_poll_rate = 0.0
        enc_rate = 0.0
        fsr_rate = 0.0
        logger_rate = 0.0

    file_bytes = out_file.stat().st_size if out_file.exists() else 0
    file_kib = file_bytes / 1024.0

    print("[Logger] Final summary:")
    print(f"  Duration: {elapsed_s:.3f} s")
    print(f"  Rows written: {rows_written}")
    print(f"  Logger cadence: {logger_rate:.2f} Hz (target {SENSOR_HZ:.2f} Hz)")
    print(
        f"  Published rates -> IMU-foot: {logger_rate:.2f} Hz, IMU-shank: {logger_rate:.2f} Hz, "
        f"Encoder: {logger_rate:.2f} Hz, FSR: {logger_rate:.2f} Hz"
    )

    if DIAGNOSTIC_METRICS:
        print(
            f"  Parsed samples -> IMU-foot: {imu_foot_parses}, IMU-shank: {imu_shank_parses}, "
            f"Encoder: {enc_parses}, FSR: {fsr_parses}"
        )
        print(
            f"  Worker poll rates -> IMU-foot: {imu_foot_poll_rate:.2f} Hz, IMU-shank: {imu_shank_poll_rate:.2f} Hz, "
            f"Encoder: {enc_rate:.2f} Hz, FSR: {fsr_rate:.2f} Hz"
        )
    print(f"  Logger overruns: {logger_overruns}")
    print(f"  CSV size: {file_bytes} bytes ({file_kib:.1f} KiB)")


if __name__ == "__main__":
    main()
