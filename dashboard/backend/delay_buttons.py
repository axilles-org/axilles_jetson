"""
Remote up/down buttons that step the live assistance delay by 5 ms per press.

Same wiring and debounce as test_ws/button_input.py (40-pin header, BOARD numbering):
    pin 17 -> 3.3 V (button VCC)
    pin 20 -> GND
    pin 29 -> "up" button signal   (GPIO PQ.05)
    pin 31 -> "down" button signal (GPIO PQ.06)
Both read LOW at rest and HIGH when pressed. The pins are inputs only.

A press goes through ProcessManager.step_delay(), the same assist_state a Live-page
Set writes, so the dashboard and the buttons have no priority over each other:
the later change wins and both show on the Live page. The controller still holds
any new delay until the stance in progress ends. Presses do nothing unless a live
ML run is going.

Runs in its own thread: the 5 ms polling would stall the asyncio event loop
(WebSocket relay, assist heartbeat), and it stays out of the controller process,
so no GPIO read ever happens inside the control loop.

Needs Jetson.GPIO in the backend's Python (`pip install Jetson.GPIO`; the user
must be in the `gpio` group). Without it the backend runs without the buttons.
"""

from __future__ import annotations

import logging
import threading
import time
import warnings
from typing import Callable

UP_PIN = 29
DOWN_PIN = 31
ACTIVE_HIGH = True   # set False if a button reacts on release instead of press
DEBOUNCE_S = 0.05
POLL_S = 0.005

log = logging.getLogger("uvicorn.error")


class Button:
    """Debounced button that reports True once per press (button_input.py)."""

    def __init__(self, read: Callable[[], bool]):
        self.read = read
        self.state = read()
        self.changed_at: float | None = None

    def pressed(self, now: float) -> bool:
        raw = self.read()
        if raw == self.state:
            self.changed_at = None
            return False
        if self.changed_at is None:
            self.changed_at = now
            return False
        if now - self.changed_at < DEBOUNCE_S:
            return False
        self.state = raw
        self.changed_at = None
        return raw  # True only on the press edge, not the release


class DelayButtons:
    def __init__(self, on_step: Callable[[int], object]):
        self.on_step = on_step          # called with +1 (up) or -1 (down)
        self._gpio = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        try:
            # Jetson.GPIO warns it can't read /dev/mem for an optional pinmux
            # check without root; reading the pins works fine without it.
            warnings.filterwarnings("ignore", message="Could not open /dev/mem")
            import Jetson.GPIO as GPIO
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup([UP_PIN, DOWN_PIN], GPIO.IN)
        except Exception as exc:   # ImportError off-Jetson, RuntimeError on bad permissions
            log.warning("delay buttons disabled (%s: %s)", type(exc).__name__, exc)
            return False
        self._gpio = GPIO
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="delay-buttons", daemon=True)
        self._thread.start()
        log.info("delay buttons on header pins %d (up) / %d (down)", UP_PIN, DOWN_PIN)
        return True

    def _pressed_reader(self, pin: int) -> Callable[[], bool]:
        GPIO = self._gpio
        level = GPIO.HIGH if ACTIVE_HIGH else GPIO.LOW
        return lambda: GPIO.input(pin) == level

    def _run(self) -> None:
        up = Button(self._pressed_reader(UP_PIN))
        down = Button(self._pressed_reader(DOWN_PIN))
        while not self._stop.is_set():
            try:
                now = time.monotonic()
                for button, steps in ((up, +1), (down, -1)):
                    if button.pressed(now):
                        state = self.on_step(steps)
                        if state is None:
                            log.info("delay button %s ignored: no live run, or its "
                                     "delay is not streamed yet", "up" if steps > 0 else "down")
                        else:
                            log.info("delay button %s -> %g ms (applies after the "
                                     "current stance)", "up" if steps > 0 else "down",
                                     state["delay_ms"])
            except Exception:
                log.exception("delay button read failed; retrying in 1 s")
                self._stop.wait(1.0)
            self._stop.wait(POLL_S)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._gpio is not None:
            self._gpio.cleanup([UP_PIN, DOWN_PIN])
            self._gpio = None
