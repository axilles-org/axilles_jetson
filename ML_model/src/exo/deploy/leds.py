"""Status lights on the exo's LED strip, for live ML runs.

The strip (APA102, 12 LEDs) hangs off an Arduino Micro running
led-strip/arduino/dotstar_bridge. The Arduino plays every timed pattern itself,
so a blink finishes even after this process exits. This module only tells it
which pattern to show and, during assistance, where the window is and its color.

    Arduino powers up        green blinking, 2 Hz, twice     (Arduino, by itself)
    ankle-zero calibration   yellow blinking, 3 Hz           CALIBRATING
    calibrated, assist off   green, steady                   READY
    assist on, walking       WINDOW_LEDS-wide window: wired end at heel strike,
                             far end by toe-off; blue at 0 N.m, red at max_nm
    assist on, swing         off
    assist on, standing      off (StanceWindow: no real step, or stance > 1.2 s)
    normal stop              red, steady 5 s, then off       STOP event
    motor fault or crash     red blinking, 5 Hz, for 10 s    ESTOP event

Blink rates and durations live in the Arduino sketch (BOOT_BLINK_HZ,
CALIB_BLINK_HZ, ESTOP_BLINK_HZ, ESTOP_MS, STOP_MS); changing them needs a
re-upload.

The control loop calls ``set_state()`` / ``event()``, which only store a value
under a lock. A background thread writes it to the serial port at SEND_HZ, so
the loop never waits on the Arduino. A missing Arduino or a serial error never
reaches the control loop: the thread prints once and retries the port.

Edit the constants below to match the strip. They are sent to the Arduino on
connect, and it saves them for its power-on blink.
"""
from __future__ import annotations

import collections
import math
import statistics
import threading
import time
from enum import IntEnum
from typing import Callable, Sequence

PORT = "/dev/serial/by-id/usb-Arduino_LLC_Arduino_Micro-if00"
NUM_LEDS = 12
COLOR_ORDER = "GBR"        # wire byte order after the brightness byte (this strip: GBR)
BRIGHTNESS = 0.5           # 0..1 of the APA102 5-bit global brightness
WINDOW_LEDS = 5

SEND_HZ = 50.0             # the Arduino blanks CALIBRATING / READY / ASSIST after 0.5 s of silence
RETRY_S = 3.0              # reopen the port this often while it is missing

# Walking vs standing. Values from the 2026-11-21 18:27 powered run: standing
# foot |gyro| peaked <= 0.32 rad/s, real swings 4.3-7.3 rad/s and 0.43-0.6 s,
# walking stances 0.75 s median (0.82 s p95). The stance flag also drops out
# for 0.02-0.11 s at push-off in every stride, at |gyro| 1-2.4 rad/s.
SWING_GYRO_RAD_S = 3.0     # foot |gyro| a swing must reach to count as a step
MIN_SWING_S = 0.2          # stance-flag gaps shorter than this are push-off dropouts
STANDING_STANCE_S = 1.2    # a stance longer than this is standing: window off
GYRO_FROZEN_S = 0.2        # foot gyro unchanged this long: judge swings on duration only

FIRST_STANCE_S = 0.75      # stance time assumed until one has been measured
STANCE_HISTORY = 3         # window speed uses the median of the last N stances
MIN_STANCE_S = 0.2         # shorter stances do not change the estimate

NO_WINDOW = 0xFF

Color = tuple[int, int, int]


class LedMode(IntEnum):
    OFF = 0
    CALIBRATING = 1
    READY = 2           # calibrated, assistance off: steady green
    ASSIST = 3
    STOP = 4            # one-shot
    ESTOP = 5           # one-shot


ONE_SHOT = {LedMode.STOP, LedMode.ESTOP}


def config_packet(num_leds: int = NUM_LEDS, order: str = COLOR_ORDER,
                  brightness: float = BRIGHTNESS, width: int = WINDOW_LEDS) -> bytes:
    order = order.upper()
    if sorted(order) != ["B", "G", "R"]:
        raise ValueError(f"color order must be a permutation of RGB, got {order!r}")
    if not 1 <= num_leds <= 64 or not 1 <= width <= num_leds:
        raise ValueError("need 1 <= width <= num_leds <= 64")
    level = round(min(max(float(brightness), 0.0), 1.0) * 31)
    return b"DC" + bytes((num_leds, *("RGB".index(c) for c in order), level, width))


def state_packet(mode: LedMode, pos: int | None = None,
                 color: Sequence[int] = (0, 0, 0)) -> bytes:
    p = NO_WINDOW if pos is None else min(max(int(pos), 0), 0xFE)
    r, g, b = (min(max(int(c), 0), 255) for c in color)
    return b"DP" + bytes((int(mode), p, r, g, b))


def torque_color(torque_nm: float, max_nm: float) -> Color:
    """Blue at 0 N.m to red at |torque| >= max_nm, linear in between."""
    t = abs(torque_nm) / max_nm if max_nm > 0 and math.isfinite(torque_nm) else 0.0
    r = round(255 * min(t, 1.0))
    return (r, 0, 255 - r)


def foot_gyro_rad_s(raw: dict[str, float]) -> float:
    """Foot gyro magnitude from a raw sensor frame (independent of IMU mounting)."""
    return math.sqrt(raw["foot_gx"] ** 2 + raw["foot_gy"] ** 2 + raw["foot_gz"] ** 2)


class StanceWindow:
    """Window position along the strip, from heel strike to toe-off, while walking.

    A heel strike counts only after a real swing: the stance flag off for at
    least MIN_SWING_S with the foot gyro reaching SWING_GYRO_RAD_S (duration
    alone if the gyro is frozen). So weight shifts while standing never start a
    window, and neither does the push-off dropout of the stance flag.

    The stance shown ends at toe-off: the first stance-flag gap that lasts
    MIN_SWING_S or reaches SWING_GYRO_RAD_S. Shorter gaps (push-off dropouts)
    keep the sweep going. A stance longer than STANDING_STANCE_S is standing:
    the window turns off until the next real step.

    The window starts on the first LED at heel strike and moves one LED per
    equal share of the expected stance time, the median of the last few
    measured stances. Returns None when no window should show, including until
    the first real step of a run.
    """

    def __init__(self, num_leds: int = NUM_LEDS, width: int = WINDOW_LEDS,
                 first_stance_s: float = FIRST_STANCE_S, history: int = STANCE_HISTORY):
        self.last_pos = num_leds - width
        self.first_stance_s = first_stance_s
        self._durations: collections.deque[float] = collections.deque(maxlen=history)
        self._in_stance = True          # a run that starts mid-stance waits for a real step
        self._heel_strike_t: float | None = None   # set while a walking stance shows
        self._off_t: float | None = None           # when the stance flag last went off
        self._swing_peak = 0.0
        self._last_gyro: float | None = None
        self._gyro_changed_t = 0.0

    @property
    def expected_stance_s(self) -> float:
        return statistics.median(self._durations) if self._durations else self.first_stance_s

    @property
    def walking(self) -> bool:
        return self._heel_strike_t is not None

    def update(self, stance: bool, t: float, foot_gyro: float | None = None) -> int | None:
        live = self._gyro_live(foot_gyro, t)
        if stance:
            if not self._in_stance and self._heel_strike_t is None and self._real_swing(t, live):
                self._heel_strike_t = t
            # A rising edge with a stance still shown is the end of a push-off dropout.
            if self._heel_strike_t is not None and t - self._heel_strike_t > STANDING_STANCE_S:
                self._heel_strike_t = None              # standing
        else:
            if self._in_stance:
                self._off_t = t
                self._swing_peak = 0.0
            if live:
                self._swing_peak = max(self._swing_peak, foot_gyro)
            if self._heel_strike_t is not None and (
                    t - self._off_t >= MIN_SWING_S
                    or (live and foot_gyro >= SWING_GYRO_RAD_S)):
                self._end_stance()                      # toe-off
        self._in_stance = stance

        if self._heel_strike_t is None:
            return None
        phase = (t - self._heel_strike_t) / self.expected_stance_s
        return min(self.last_pos, int(phase * (self.last_pos + 1)))

    def _gyro_live(self, gyro: float | None, t: float) -> bool:
        if gyro is None or not math.isfinite(gyro):
            return False
        if gyro != self._last_gyro:
            self._last_gyro = gyro
            self._gyro_changed_t = t
        return t - self._gyro_changed_t < GYRO_FROZEN_S

    def _real_swing(self, t: float, live: bool) -> bool:
        if self._off_t is None or t - self._off_t < MIN_SWING_S:
            return False
        return self._swing_peak >= SWING_GYRO_RAD_S or not live

    def _end_stance(self) -> None:
        duration = self._off_t - self._heel_strike_t
        if MIN_STANCE_S <= duration <= STANDING_STANCE_S:
            self._durations.append(duration)
        self._heel_strike_t = None


class LedStatus:
    def __init__(self, port: str = PORT, *, send_hz: float = SEND_HZ,
                 serial_factory: Callable[[], object] | None = None):
        self.port = port
        self._period = 1.0 / send_hz
        self._serial_factory = serial_factory or self._open_serial
        self._lock = threading.Lock()
        self._mode = LedMode.OFF
        self._pos: int | None = None
        self._color: Color = (0, 0, 0)
        self._events: list[LedMode] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="led-status", daemon=True)
        self._thread.start()

    # -- control-loop side (never blocks on the port) --------------------------
    def set_state(self, mode: LedMode, pos: int | None = None,
                  color: Color = (0, 0, 0)) -> None:
        if mode in ONE_SHOT:
            raise ValueError(f"{mode.name} is a one-shot: use event()")
        with self._lock:
            self._mode, self._pos, self._color = mode, pos, color

    def event(self, mode: LedMode) -> None:
        """Play a one-shot pattern; the steady state drops to OFF."""
        if mode not in ONE_SHOT:
            raise ValueError(f"{mode.name} is not a one-shot: use set_state()")
        with self._lock:
            self._events.append(mode)
            self._mode, self._pos, self._color = LedMode.OFF, None, (0, 0, 0)

    def close(self, timeout_s: float = 1.0) -> None:
        """Deliver any pending event, then release the port."""
        self._stop.set()
        self._thread.join(timeout_s)

    # -- writer thread --------------------------------------------------------
    def _open_serial(self):
        import serial
        # 115200 is nominal (native USB). Never 1200: that resets the Micro into
        # its bootloader. exclusive: led_strip.py cannot interleave frames.
        return serial.Serial(self.port, 115200, timeout=0, write_timeout=0.05,
                             exclusive=True)

    def _run(self) -> None:
        ser = None
        next_open = 0.0
        warned = False
        while True:
            stopping = self._stop.wait(self._period)
            with self._lock:
                events, self._events = self._events, []
                packet = state_packet(self._mode, self._pos, self._color)

            if ser is None and not stopping and time.monotonic() >= next_open:
                try:
                    ser = self._serial_factory()
                    ser.write(config_packet())
                    if warned:
                        print("LED strip connected")
                    warned = False
                except Exception as exc:                   # noqa: BLE001
                    ser = self._drop(ser)
                    next_open = time.monotonic() + RETRY_S
                    if not warned:
                        print(f"LED strip unavailable ({exc}); run continues without it")
                        warned = True

            if ser is not None:
                try:
                    for mode in events:
                        ser.write(state_packet(mode))
                    if not stopping:
                        ser.write(packet)
                    ser.reset_input_buffer()               # greeting text; nothing to read
                    if stopping:
                        ser.flush()
                except Exception as exc:                   # noqa: BLE001
                    ser = self._drop(ser)
                    next_open = time.monotonic() + RETRY_S
                    if not warned:
                        print(f"LED strip lost ({exc}); retrying every {RETRY_S:g} s")
                        warned = True

            if stopping:
                self._drop(ser)
                return

    @staticmethod
    def _drop(ser) -> None:
        if ser is not None:
            try:
                ser.close()
            except Exception:                              # noqa: BLE001
                pass
        return None
