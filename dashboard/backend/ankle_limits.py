"""Ankle range-of-motion measurement for the Ankle Limits page.

The wearer stands on both legs, the operator presses Set Standing, then the wearer
lifts the leg and flexes the ankle both ways. The largest offset from standing in
each direction is kept until Reset or until the backend stops. Nothing is saved.

Read-only, like health.py: one two-byte read of the AS5600 angle register per
sample (the same read as TBE_controller/data_obtainer.py readEncoder). Never sends
a CAN frame and never writes to an I2C device. Sampling runs only while the page
is asking, and pauses while a controller is running so it cannot take bytes from
that run's I2C traffic.

Directions are reported as "positive" (raw angle increasing) and "negative". The
page maps them to dorsi/plantar; exo_frame_calibration.json and the deploy config
both have ankle_encoder_sign = +1, i.e. increasing raw angle is dorsiflexion.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from .health import I2C_BUS, _controller_running

AS5600_ADDR = 0x36
AS5600_REG_ANGLE = 0x0E          # filtered 12-bit angle, MSB then LSB
COUNTS_PER_REV = 4096

SAMPLE_PERIOD_S = 0.01           # 100 Hz, fast enough not to miss a flexion peak
IDLE_STOP_S = 2.0                # stop sampling this long after the page stops polling
STALE_S = 0.5                    # a reading older than this is not shown as live
CONTROLLER_CHECK_S = 1.0

SETTLE_S = 2.0                   # no new maximum for this long -> settled
SETTLE_TOL_DEG = 0.5             # growth smaller than this does not restart the settle timer
MIN_EXCURSION_DEG = 2.0          # standing jitter alone never counts as settled

router = APIRouter(prefix="/api/ankle-limits")

_lock = threading.Lock()
_thread: threading.Thread | None = None
_last_request = 0.0

# Shared state, guarded by _lock.
_raw_deg: float | None = None
_raw_time = 0.0
_reason: str | None = "Not started"
_standing_deg: float | None = None
_max = {"positive": 0.0, "negative": 0.0}
_growth_time = {"positive": 0.0, "negative": 0.0}   # last time the max grew past tolerance
_growth_ref = {"positive": 0.0, "negative": 0.0}    # the max at that time


def _offset(raw: float, standing: float) -> float:
    """Signed raw - standing in degrees, unwrapped across the 0/360 seam."""
    return (raw - standing + 180.0) % 360.0 - 180.0


def _record(raw: float, now: float) -> None:
    global _raw_deg, _raw_time, _reason
    with _lock:
        _raw_deg, _raw_time, _reason = raw, now, None
        if _standing_deg is None:
            return
        d = _offset(raw, _standing_deg)
        side, size = ("positive", d) if d >= 0 else ("negative", -d)
        if size > _max[side]:
            _max[side] = size
            if size - _growth_ref[side] > SETTLE_TOL_DEG:
                _growth_ref[side] = size
                _growth_time[side] = now


def _set_reason(reason: str) -> None:
    global _reason
    with _lock:
        _reason = reason


def _sampler() -> None:
    try:
        from smbus2 import SMBus
    except ImportError as exc:
        _set_reason(f"Cannot open I2C bus {I2C_BUS}: {exc}")
        return

    bus = None
    next_check = 0.0
    paused = False
    try:
        while True:
            with _lock:
                if time.monotonic() - _last_request > IDLE_STOP_S:
                    return
            now = time.monotonic()
            if now >= next_check:
                paused = _controller_running()
                next_check = now + CONTROLLER_CHECK_S
                if paused and bus is not None:
                    bus.close()
                    bus = None
            if paused:
                _set_reason("Paused: a run is using the sensors")
                time.sleep(0.1)
                continue
            try:
                if bus is None:
                    bus = SMBus(I2C_BUS)
                data = bus.read_i2c_block_data(AS5600_ADDR, AS5600_REG_ANGLE, 2)
                raw = (((data[0] & 0x0F) << 8) | data[1]) * (360.0 / COUNTS_PER_REV)
                _record(raw, time.monotonic())
            except Exception as exc:    # noqa: BLE001
                _set_reason(f"Encoder read failed: {exc}")
                if bus is not None:
                    bus.close()
                    bus = None
                time.sleep(0.2)
                continue
            time.sleep(SAMPLE_PERIOD_S)
    finally:
        if bus is not None:
            bus.close()


def _ensure_sampler() -> None:
    global _thread, _last_request
    with _lock:
        _last_request = time.monotonic()
        if _thread is not None and _thread.is_alive():
            return
        _thread = threading.Thread(target=_sampler, name="ankle-limits", daemon=True)
        _thread.start()


def _side(side: str, now: float) -> dict[str, Any]:
    size = _max[side]
    settled = (_standing_deg is not None and size >= MIN_EXCURSION_DEG
               and now - _growth_time[side] >= SETTLE_S)
    return {"max_deg": size, "settled": settled}


def _snapshot() -> dict[str, Any]:
    now = time.monotonic()
    live = _raw_deg is not None and _reason is None and now - _raw_time <= STALE_S
    offset = (_offset(_raw_deg, _standing_deg)
              if live and _standing_deg is not None else None)
    return {
        "ok": live,
        "reason": None if live else (_reason or "No recent encoder reading"),
        "raw_deg": _raw_deg if live else None,
        "standing_deg": _standing_deg,
        "offset_deg": offset,
        "positive": _side("positive", now),
        "negative": _side("negative", now),
    }


def _clear_maxima(now: float) -> None:
    for side in _max:
        _max[side] = 0.0
        _growth_ref[side] = 0.0
        _growth_time[side] = now


@router.get("")
def get_state() -> dict[str, Any]:
    _ensure_sampler()
    with _lock:
        return _snapshot()


@router.post("/standing")
def set_standing() -> dict[str, Any]:
    global _standing_deg
    _ensure_sampler()
    with _lock:
        now = time.monotonic()
        if _raw_deg is None or _reason is not None or now - _raw_time > STALE_S:
            raise HTTPException(409, "No live encoder reading to use as standing")
        _standing_deg = _raw_deg
        _clear_maxima(now)
        return _snapshot()


@router.post("/reset")
def reset() -> dict[str, Any]:
    global _standing_deg
    with _lock:
        _standing_deg = None
        _clear_maxima(time.monotonic())
        return _snapshot()
