#!/usr/bin/env python3
"""Wiring diagnostic for the LED strip on pins 19 (PZ.05) and 23 (PZ.03).

Phases (about 50 s total):
  1. 0-15 s   pin 23 held HIGH, to measure with a meter (expect ~3.3 V vs GND)
  2. 15-35 s  dim white sent over and over with the normal wiring
  3. 35-50 s  dim white with data/clock swapped in software (finds swapped wires)
  4. end      strip cleared with both mappings, pins released to inputs

White is used so the color order (--order) doesn't matter: any light counts.
"""
from __future__ import annotations

import argparse
import time

import gpiod
from gpiod.line import Direction, Value

from dotstar import DotStar

CHIP = "/dev/gpiochip0"
PIN19, PIN23 = "PZ.05", "PZ.03"


def hold_high(line_id: str, seconds: float) -> None:
    with gpiod.Chip(CHIP) as c:
        off = c.line_offset_from_id(line_id)
    req = gpiod.request_lines(CHIP, consumer="led-diag", config={
        off: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.ACTIVE)})
    try:
        time.sleep(seconds)
    finally:
        req.reconfigure_lines({off: gpiod.LineSettings(direction=Direction.INPUT)})
        req.release()


def stream_white(n: int, brightness: float, clock: str, data: str, seconds: float) -> int:
    frames = 0
    with DotStar(n, brightness=brightness, clock_line=clock, data_line=data) as strip:
        strip.fill((255, 255, 255))
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            strip.show()
            frames += 1
    return frames  # DotStar.__exit__ sends an all-off frame with this mapping


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=30)
    p.add_argument("--brightness", type=float, default=0.1)
    a = p.parse_args()

    t0 = time.monotonic()
    stamp = lambda: f"[{time.monotonic() - t0:5.1f}s]"

    print(stamp(), "phase 1: pin 23 HIGH for 15 s", flush=True)
    hold_high(PIN23, 15)

    print(stamp(), "phase 2: white, normal wiring (clock=pin 23, data=pin 19)", flush=True)
    f = stream_white(a.num, a.brightness, clock=PIN23, data=PIN19, seconds=20)
    print(stamp(), f"  sent {f} frames ({f / 20:.0f}/s)", flush=True)

    print(stamp(), "phase 3: white, SWAPPED (clock=pin 19, data=pin 23)", flush=True)
    f = stream_white(a.num, a.brightness, clock=PIN19, data=PIN23, seconds=15)
    print(stamp(), f"  sent {f} frames ({f / 15:.0f}/s)", flush=True)

    # Clear again with the normal mapping in case the swapped off-frame was not understood.
    with DotStar(a.num, clock_line=PIN23, data_line=PIN19):
        pass
    print(stamp(), "done: strip cleared, pins released", flush=True)


if __name__ == "__main__":
    main()
