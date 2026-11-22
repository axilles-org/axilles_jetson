"""Runtime on/off switch for assistance, driven by the dashboard over local UDP.

The dashboard backend sends ``{"assist": bool, "scale": float, "delay_ms": float}``
a few times per second while a live job runs. ``delay_ms`` is present only once a
delay has been set from the Live page; until then the launch delay stays. The
switch is a dead-man: if no message arrives for ``timeout_s``, assistance turns
off. ``poll()`` never blocks, so it is safe to call from the control loop.
"""
from __future__ import annotations

import json
import math
import socket
import time

CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 47271


class AssistSwitch:
    def __init__(self, scale: float, max_scale: float, timeout_s: float = 1.0,
                 host: str = CONTROL_HOST, port: int = CONTROL_PORT,
                 delay_range_ms: tuple[float, float] = (10.0, 40.0)):
        self.max_scale = max_scale
        self.scale = min(scale, max_scale)
        self.min_delay_ms, self.max_delay_ms = (float(v) for v in delay_range_ms)
        if not 0 <= self.min_delay_ms <= self.max_delay_ms <= 1000:
            raise ValueError("delay range must satisfy 0 <= min <= max <= 1000 ms")
        self.delay_ms: float | None = None   # None: keep the launch delay
        self.timeout_s = timeout_s
        self.enabled = False
        self._last_msg = 0.0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.setblocking(False)

    def poll(self) -> bool:
        """Drain pending messages; returns whether assistance is enabled now."""
        while True:
            try:
                data = self._sock.recv(1024)
            except BlockingIOError:
                break
            try:
                msg = json.loads(data)
                self.enabled = bool(msg["assist"])
                if "scale" in msg:
                    self.scale = min(max(float(msg["scale"]), 0.0), self.max_scale)
                if msg.get("delay_ms") is not None:
                    delay = float(msg["delay_ms"])
                    if math.isfinite(delay):
                        self.delay_ms = min(max(delay, self.min_delay_ms), self.max_delay_ms)
                self._last_msg = time.monotonic()
            except (ValueError, KeyError, TypeError):
                continue
        if self.enabled and time.monotonic() - self._last_msg > self.timeout_s:
            self.enabled = False
        return self.enabled

    def close(self) -> None:
        self._sock.close()
