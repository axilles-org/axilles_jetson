"""Ankle range-of-motion measurement for the Ankle Limits page.

The wearer stands on both legs, the operator presses Set Standing, then the wearer
lifts the leg and flexes the ankle both ways. The largest offset from standing in
each direction is kept until Reset or until the backend stops. Nothing is saved.

Encoder readings come from sensor_monitor.py (200 Hz, while no controller is
running) and from a running controller's live stream (server.py relay), as raw
AS5600 degrees. The standing angle is also the zero of the Live page's encoder plot.

Directions are reported as "positive" (raw angle increasing) and "negative". The
page maps them to dorsi/plantar; exo_frame_calibration.json and the deploy config
both have ankle_encoder_sign = +1, i.e. increasing raw angle is dorsiflexion.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from .sensor_monitor import monitor

STALE_S = 0.5                    # a reading older than this is not shown as live

SETTLE_S = 2.0                   # no new maximum for this long -> settled
SETTLE_TOL_DEG = 0.5             # growth smaller than this does not restart the settle timer
MIN_EXCURSION_DEG = 2.0          # standing jitter alone never counts as settled

router = APIRouter(prefix="/api/ankle-limits")

_lock = threading.Lock()
_raw_deg: float | None = None
_raw_time = 0.0
_standing_deg: float | None = None
_max = {"positive": 0.0, "negative": 0.0}
_growth_time = {"positive": 0.0, "negative": 0.0}   # last time the max grew past tolerance
_growth_ref = {"positive": 0.0, "negative": 0.0}    # the max at that time


def offset(raw: float, standing: float) -> float:
    """Signed raw - standing in degrees, unwrapped across the 0/360 seam."""
    return (raw - standing + 180.0) % 360.0 - 180.0


def standing_deg() -> float | None:
    return _standing_deg


def record(raw: float) -> None:
    """One raw AS5600 reading in degrees [0, 360)."""
    global _raw_deg, _raw_time
    now = time.monotonic()
    with _lock:
        _raw_deg, _raw_time = raw, now
        if _standing_deg is None:
            return
        d = offset(raw, _standing_deg)
        side, size = ("positive", d) if d >= 0 else ("negative", -d)
        if size > _max[side]:
            _max[side] = size
            if size - _growth_ref[side] > SETTLE_TOL_DEG:
                _growth_ref[side] = size
                _growth_time[side] = now


def _live(now: float) -> bool:
    return _raw_deg is not None and now - _raw_time <= STALE_S


def _side(side: str, now: float) -> dict[str, Any]:
    size = _max[side]
    settled = (_standing_deg is not None and size >= MIN_EXCURSION_DEG
               and now - _growth_time[side] >= SETTLE_S)
    return {"max_deg": size, "settled": settled}


def _snapshot() -> dict[str, Any]:
    now = time.monotonic()
    live = _live(now)
    reason = None
    if not live:
        enc = monitor.device_status()["encoder"]
        reason = enc["reason"] if monitor.reading() else monitor.state_text()
    return {
        "ok": live,
        "reason": reason,
        "raw_deg": _raw_deg if live else None,
        "standing_deg": _standing_deg,
        "offset_deg": offset(_raw_deg, _standing_deg) if live and _standing_deg is not None else None,
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
    monitor.touch()
    with _lock:
        return _snapshot()


@router.post("/standing")
def set_standing() -> dict[str, Any]:
    global _standing_deg
    monitor.touch()
    with _lock:
        now = time.monotonic()
        if not _live(now):
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
