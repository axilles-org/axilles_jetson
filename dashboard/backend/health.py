"""Connection checks for the Health Monitor page.

Read-only, and only runs while the page is asking. Each I2C device gets a one-byte
read (the same check as probe_i2c in fall_test_plan/test_4/test4_run.py). The motor
is checked by listening for the status frames it already broadcasts. This file
never sends a CAN frame and never writes to an I2C device.

I2C is skipped while a controller is running: a stray read from here could take
bytes out of its BNO085 stream.
"""
from __future__ import annotations

import os
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter

I2C_BUS = 1                     # TBE_controller/utilities.py
I2C_DEVICES = {"imu_foot": 0x4A, "imu_shank": 0x4B, "encoder": 0x36, "adc": 0x48}

CAN_CHANNEL = "can0"
MOTOR_STATUS_ID = 0x2968        # (0x29 << 8) | motor id, about every 20 ms
MOTOR_WAIT_S = 0.1

MIN_CHECK_INTERVAL_S = 0.5      # several open tabs share one check

CONTROLLER_SCRIPTS = ("jetson_deploy.py", "jetson_mock_deploy.py", "TBE_controller/main.py",
                      "test4_run.py", "fall_logger.py")

router = APIRouter(prefix="/api/health")

_lock = threading.Lock()
_last_result: dict[str, Any] | None = None
_last_check = 0.0


def _controller_running() -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            if "python" not in cmd:
                continue
            if any(name in cmd for name in CONTROLLER_SCRIPTS):
                return True
            if " main.py" in cmd and os.readlink(entry / "cwd").endswith("TBE_controller"):
                return True
        except OSError:
            continue                # the process ended while being read
    return False


def _check_i2c() -> dict[str, dict[str, Any]]:
    if _controller_running():
        return {name: {"ok": False, "reason": "Not checked: a run is using the sensors"}
                for name in I2C_DEVICES}
    try:
        from smbus2 import SMBus
        bus = SMBus(I2C_BUS)
    except Exception as exc:        # noqa: BLE001
        return {name: {"ok": False, "reason": f"Cannot open I2C bus {I2C_BUS}: {exc}"}
                for name in I2C_DEVICES}

    results: dict[str, dict[str, Any]] = {}
    with bus:
        for name, addr in I2C_DEVICES.items():
            try:
                bus.read_byte(addr)
                results[name] = {"ok": True, "reason": None}
            except Exception:       # noqa: BLE001
                results[name] = {"ok": False, "reason": f"No reply at 0x{addr:02X}"}
    return results


def _check_motor() -> dict[str, Any]:
    try:
        with socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as sock:
            status_filter = struct.pack("=II", MOTOR_STATUS_ID | socket.CAN_EFF_FLAG,
                                        socket.CAN_EFF_MASK | socket.CAN_EFF_FLAG)
            sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, status_filter)
            sock.bind((CAN_CHANNEL,))
            sock.settimeout(MOTOR_WAIT_S)
            frame = sock.recv(16)
    except TimeoutError:
        return {"ok": False, "reason": "No status frames from the motor"}
    except Exception as exc:        # noqa: BLE001
        return {"ok": False, "reason": f"Cannot listen on {CAN_CHANNEL}: {exc}"}

    _, length, data = struct.unpack("=IB3x8s", frame)
    if length < 8:
        return {"ok": False, "reason": "Short status frame"}
    if data[7] != 0:
        return {"ok": False, "reason": f"Motor error code {data[7]}"}
    return {"ok": True, "reason": None}


@router.get("")
def health() -> dict[str, Any]:
    global _last_result, _last_check
    with _lock:
        if _last_result is None or time.monotonic() - _last_check >= MIN_CHECK_INTERVAL_S:
            _last_result = {**_check_i2c(), "motor": _check_motor()}
            _last_check = time.monotonic()
        return _last_result
