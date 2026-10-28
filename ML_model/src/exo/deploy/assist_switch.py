"""Runtime on/off switch for assistance, driven by the dashboard over local UDP.

The dashboard backend sends ``{"assist": bool, "scale": float}`` a few times per
second while a live job runs. The switch is a dead-man: if no message arrives for
``timeout_s``, assistance turns off. ``poll()`` never blocks, so it is safe to call
from the control loop.
"""
from __future__ import annotations

import json
import socket
import time

CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 47271


class AssistSwitch:
    def __init__(self, scale: float, max_scale: float, timeout_s: float = 1.0,
                 host: str = CONTROL_HOST, port: int = CONTROL_PORT):
        self.max_scale = max_scale
        self.scale = min(scale, max_scale)
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
                self._last_msg = time.monotonic()
            except (ValueError, KeyError, TypeError):
                continue
        if self.enabled and time.monotonic() - self._last_msg > self.timeout_s:
            self.enabled = False
        return self.enabled

    def close(self) -> None:
        self._sock.close()
