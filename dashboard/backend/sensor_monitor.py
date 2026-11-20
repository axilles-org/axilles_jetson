"""Sensor monitor: live sensor data for the dashboard while no controller is running.

Reads the ankle encoder (AS5600), both FSRs (ADS1115) and both IMUs (BNO085), the
same way the controllers do, and sends samples to the live relay (UDP 47270) tagged
``source: "monitor"``. The Live page then always has sensor data, and when a
controller starts its own stream takes over.

Hardware: I2C only. This file imports no CAN code and never touches the motor.
It does write to sensors: the ADS1115 config register for every conversion and,
when it starts, a soft reset and Set Feature for each BNO085. The controllers do
the same at start-up.

Bus sharing: the monitor never reads while a controller is running (checked
through /proc, so CLI runs count too). process_manager.start() calls suspend()
first and only launches once the monitor has closed every I2C handle.

Runs only while the Live page has a connection, or the Ankle Limits page polled
within IDLE_STOP_S.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

I2C_BUS = 1
REPO_ROOT = Path(__file__).resolve().parents[2]

AS5600_ADDR = 0x36
AS5600_REG_ANGLE = 0x0E
IMU_ADDRS = (("foot", 0x4A), ("shank", 0x4B))
IMU_REPORT_HZ = 100.0
IMU_MAX_POLLS = 4               # packets drained per IMU per loop, as in JetsonSensors
IMU_STALE_S = 0.5               # no report this long: re-send Set Feature
IMU_REENABLE_EVERY_S = 1.0
FSR_HZ = 100.0

LOOP_SLEEP_S = 0.002
ENCODER_PERIOD_S = 0.005        # 200 Hz: fast enough for Ankle Limits peaks
EMIT_PERIOD_S = 0.02            # 50 Hz to the Live page
FRESH_S = 0.5                   # a value older than this is sent as null, not frozen
DEVICE_OK_S = 1.0               # Health page: a device is OK if read within this
CONTROLLER_CHECK_S = 0.5
IDLE_STOP_S = 2.0
SUSPEND_WAIT_S = 5.0            # covers two IMUs failing to connect

RELAY_ADDR = ("127.0.0.1", 47270)   # server.LIVE_RELAY_PORT

DEVICES = ("imu_foot", "imu_shank", "encoder", "adc")


class TimedADS1115:
    """Toe (AIN0) then heel (AIN1), single-shot at 860 SPS, conversion timed rather
    than polled. Same as ML_model/src/exo/deploy/jetson_io.py TimedADS1115."""

    ADDR = 0x48
    REG_CONV = 0x00
    REG_CFG = 0x01
    CFG = ([0xC3, 0xE3], [0xD3, 0xE3])
    CONV_S = 0.0013

    def __init__(self, bus, sample_hz: float):
        self._bus = bus
        self._period = 1.0 / sample_hz
        self._next = 0.0
        self._ch: int | None = None
        self._t_trigger = 0.0
        self._raw = [0, 0]

    def step(self, now: float) -> tuple[int, int] | None:
        if self._ch is None:
            if now >= self._next:
                self._next = max(self._next + self._period, now)
                self._trigger(0)
            return None
        if now - self._t_trigger < self.CONV_S:
            return None
        d = self._bus.read_i2c_block_data(self.ADDR, self.REG_CONV, 2)
        raw = (d[0] << 8) | d[1]
        self._raw[self._ch] = raw - 0x10000 if raw >= 0x8000 else raw
        if self._ch == 0:
            self._trigger(1)
            return None
        self._ch = None
        return self._raw[0], self._raw[1]

    def _trigger(self, ch: int) -> None:
        self._ch = None             # if the write fails, start over next sample
        self._bus.write_i2c_block_data(self.ADDR, self.REG_CFG, self.CFG[ch])
        self._ch = ch
        self._t_trigger = time.perf_counter()


class _Hardware:
    """Open sensor handles for one monitor session. Each device fails on its own."""

    def __init__(self, status: dict[str, dict[str, Any]]):
        self.status = status
        self.values: dict[str, float] = {}
        self.updated: dict[str, float] = {}    # value group -> time.monotonic()
        self.bus = None
        self.fsr = None
        self.imus: list[tuple[str, Any, dict[str, float]]] = []
        self._next_encoder = 0.0

        try:
            from smbus2 import SMBus
            self.bus = SMBus(I2C_BUS)
            self.fsr = TimedADS1115(self.bus, FSR_HZ)
        except Exception as exc:        # noqa: BLE001
            for name in ("encoder", "adc"):
                self._fail(name, f"Cannot open I2C bus {I2C_BUS}: {exc}")

        try:
            sys.path.insert(0, str(REPO_ROOT / "pawan"))
            import bno085_lean
            from bno085_lean import ACCEL, GYRO, LeanBNO085
            # One attempt (~2 s) instead of six: a missing IMU must not hold the bus
            # for ~11 s while a launch waits. Only this process is affected.
            bno085_lean.CONNECT_RETRIES = 1
        except Exception as exc:        # noqa: BLE001
            for prefix, _ in IMU_ADDRS:
                self._fail(f"imu_{prefix}", f"IMU driver unavailable: {exc}")
            return
        self._accel_id, self._gyro_id = ACCEL, GYRO
        for prefix, addr in IMU_ADDRS:
            try:
                imu = LeanBNO085(I2C_BUS, addr, IMU_REPORT_HZ, (ACCEL, GYRO),
                                 label=f"IMU-{prefix}")
                now = time.monotonic()
                self.imus.append((prefix, imu, {"seen": now, "reenabled": now}))
            except Exception as exc:    # noqa: BLE001
                self._fail(f"imu_{prefix}", f"No reply at 0x{addr:02X}: {exc}")

    def _ok(self, name: str, now: float) -> None:
        self.status[name] = {"last_ok": now, "reason": None}

    def _fail(self, name: str, reason: str) -> None:
        prev = self.status.get(name, {})
        self.status[name] = {"last_ok": prev.get("last_ok"), "reason": reason}

    def step(self) -> float | None:
        """Poll every device once. Returns a new raw encoder angle, if one was read."""
        encoder = None
        self._step_fsr()
        for prefix, imu, st in self.imus:
            self._poll_imu(prefix, imu, st)
            self._step_fsr()
        now = time.monotonic()
        if self.bus is not None and now >= self._next_encoder:
            self._next_encoder = now + ENCODER_PERIOD_S
            try:
                d = self.bus.read_i2c_block_data(AS5600_ADDR, AS5600_REG_ANGLE, 2)
                encoder = (((d[0] & 0x0F) << 8) | d[1]) * (360.0 / 4096.0)
                self.values["encoder_raw_deg"] = encoder
                self.updated["encoder"] = now
                self._ok("encoder", now)
            except OSError as exc:
                self._fail("encoder", f"Encoder read failed: {exc}")
        return encoder

    def _step_fsr(self) -> None:
        if self.fsr is None:
            return
        try:
            out = self.fsr.step(time.perf_counter())
        except OSError as exc:
            self._fail("adc", f"FSR read failed: {exc}")
            return
        if out is not None:
            now = time.monotonic()
            self.values["toe_fsr_raw"], self.values["heel_fsr_raw"] = float(out[0]), float(out[1])
            self.updated["fsr"] = now
            self._ok("adc", now)

    def _poll_imu(self, prefix: str, imu, st: dict[str, float]) -> None:
        name = f"imu_{prefix}"
        before = dict(imu.counts)
        for _ in range(IMU_MAX_POLLS):
            try:
                if not imu.poll():
                    break
            except (OSError, KeyError, RuntimeError, IndexError) as exc:
                self._fail(name, f"IMU read failed: {exc}")
                break
        now = time.monotonic()
        for kind, report_id, letter in (("accel", self._accel_id, "a"),
                                        ("gyro", self._gyro_id, "g")):
            value = getattr(imu, kind)
            if value is not None and imu.counts.get(report_id, 0) > before.get(report_id, 0):
                for axis, v in zip("xyz", value):
                    self.values[f"{prefix}_{letter}{axis}"] = float(v)
                self.updated[f"{prefix}_{kind}"] = now
                st["seen"] = now
                self._ok(name, now)
        if ((imu.needs_enable or now - st["seen"] > IMU_STALE_S)
                and now - st["reenabled"] >= IMU_REENABLE_EVERY_S):
            st["reenabled"] = now
            try:
                imu.reenable()
            except (OSError, KeyError, RuntimeError, IndexError) as exc:
                self._fail(name, f"IMU re-enable failed: {exc}")
            if now - st["seen"] > IMU_STALE_S:
                self._fail(name, f"No IMU reports for {now - st['seen']:.1f} s")

    def sample(self, now: float) -> dict[str, Any]:
        """The latest values; anything not refreshed within FRESH_S is null."""
        groups = {"encoder": ["encoder_raw_deg"], "fsr": ["toe_fsr_raw", "heel_fsr_raw"]}
        for prefix, _ in IMU_ADDRS:
            groups[f"{prefix}_accel"] = [f"{prefix}_a{a}" for a in "xyz"]
            groups[f"{prefix}_gyro"] = [f"{prefix}_g{a}" for a in "xyz"]
        out: dict[str, Any] = {}
        for group, keys in groups.items():
            fresh = now - self.updated.get(group, float("-inf")) <= FRESH_S
            for k in keys:
                out[k] = self.values.get(k) if fresh else None
        return out

    def close(self) -> None:
        for _, imu, _ in self.imus:
            try:
                imu.close()
            except Exception:           # noqa: BLE001
                pass
        self.imus = []
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:           # noqa: BLE001
                pass
            self.bus = None


class SensorMonitor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._clients = 0
        self._last_touch = 0.0
        self._suspend_until = 0.0
        self._hw_closed = threading.Event()
        self._hw_closed.set()
        self._controller = False
        self._next_controller_check = 0.0
        self._state = "stopped"         # stopped | starting | running | paused
        self.status: dict[str, dict[str, Any]] = {}
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # ── who wants it ────────────────────────────────────────────────────────
    def client_connected(self) -> None:
        with self._lock:
            self._clients += 1
        self._ensure_thread()

    def client_disconnected(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)
            self._last_touch = time.monotonic()     # linger IDLE_STOP_S, e.g. a page reload

    def touch(self) -> None:
        with self._lock:
            self._last_touch = time.monotonic()
        self._ensure_thread()

    def _wanted(self) -> bool:
        return self._clients > 0 or time.monotonic() - self._last_touch <= IDLE_STOP_S

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, name="sensor-monitor", daemon=True)
            self._thread.start()

    # ── handing the bus to a controller ─────────────────────────────────────
    def suspend(self, seconds: float = 5.0) -> bool:
        """Stop reading for at least ``seconds`` and wait until every I2C handle is
        closed. False if the monitor did not let go within SUSPEND_WAIT_S (e.g. an
        IMU still connecting); the caller must not start a controller then."""
        with self._lock:
            self._suspend_until = time.monotonic() + seconds
        return self._hw_closed.wait(SUSPEND_WAIT_S)

    def _blocked(self, now: float) -> bool:
        if now >= self._next_controller_check:
            from .health import _controller_running
            self._controller = _controller_running()
            self._next_controller_check = now + CONTROLLER_CHECK_S
        return now < self._suspend_until or self._controller

    # ── state for other pages ───────────────────────────────────────────────
    def reading(self) -> bool:
        """True while the monitor holds the sensors (Health must not probe then)."""
        return not self._hw_closed.is_set()

    def device_status(self) -> dict[str, dict[str, Any]]:
        now = time.monotonic()
        out = {}
        with self._lock:
            for name in DEVICES:
                st = self.status.get(name, {})
                last = st.get("last_ok")
                ok = last is not None and now - last <= DEVICE_OK_S
                out[name] = {"ok": ok,
                             "reason": None if ok else st.get("reason") or "Starting up"}
        return out

    def state_text(self) -> str:
        return {"stopped": "Sensor monitor not running",
                "starting": "Starting sensors",
                "paused": "Paused: a run is using the sensors",
                "running": "Reading sensors"}[self._state]

    # ── the reader ──────────────────────────────────────────────────────────
    def _run(self) -> None:
        from . import ankle_limits

        hw: _Hardware | None = None
        next_emit = 0.0
        try:
            while True:
                now = time.monotonic()
                with self._lock:
                    wanted = self._wanted()
                if not wanted:
                    if hw is not None:
                        hw.close()
                        hw = None
                    with self._lock:
                        if not self._wanted():
                            # Cleared under the lock with the hardware already closed,
                            # so a new client starts a fresh thread, never a second reader.
                            self._thread = None
                            self._hw_closed.set()
                            self._state = "stopped"
                            return
                    continue
                with self._lock:
                    blocked = self._blocked(now)
                    if not blocked and hw is None:
                        self._hw_closed.clear()     # under the lock, so suspend() waits for us
                if blocked:
                    if hw is not None:
                        hw.close()
                        hw = None
                    self._hw_closed.set()
                    self._state = "paused"
                    time.sleep(0.1)
                    continue
                if hw is None:
                    self._state = "starting"
                    hw = _Hardware(self.status)     # IMU setup blocks about a second each
                    self._state = "running"
                    continue                        # re-check blocked before reading

                encoder = hw.step()
                if encoder is not None:
                    ankle_limits.record(encoder)
                if now >= next_emit:
                    next_emit = now + EMIT_PERIOD_S
                    self._emit(hw, now)
                time.sleep(LOOP_SLEEP_S)
        except Exception as exc:        # noqa: BLE001
            print(f"!! sensor monitor stopped: {exc!r}")
        finally:
            if hw is not None:
                hw.close()
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None
                self._hw_closed.set()
                self._state = "stopped"

    def _emit(self, hw: _Hardware, now: float) -> None:
        msg = {"source": "monitor", "t": now, **hw.sample(now),
               "monitor": {name: st["reason"] for name, st in self.device_status().items()
                           if not st["ok"]}}
        try:
            self._sock.sendto(json.dumps(msg).encode("utf-8"), RELAY_ADDR)
        except OSError:
            pass


monitor = SensorMonitor()
