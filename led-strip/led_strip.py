#!/usr/bin/env python3
"""Command-line control of the DotStar strip through the Arduino Micro bridge.

Examples:
    python3 led-strip/led_strip.py --num 30 test          # R, G, B, W for 2 s each
    python3 led-strip/led_strip.py --num 30 solid 255 0 0
    python3 led-strip/led_strip.py --num 30 rainbow
    python3 led-strip/led_strip.py --num 30 chase 0 0 255
    python3 led-strip/led_strip.py --num 30 off

Animations run until Ctrl+C. The strip is turned off on exit, except after
`solid`, which leaves the color on.
"""
from __future__ import annotations

import argparse
import sys
import time

from dotstar import DEFAULT_PORT, DotStar, wheel


def cmd_off(strip: DotStar, args) -> None:
    strip.off()


def cmd_solid(strip: DotStar, args) -> None:
    strip.fill((args.r, args.g, args.b))
    strip.show()


def cmd_test(strip: DotStar, args) -> None:
    """Show red, green, blue, white for 2 s each; prints what should be visible."""
    for name, color in (("RED", (255, 0, 0)), ("GREEN", (0, 255, 0)),
                        ("BLUE", (0, 0, 255)), ("WHITE", (255, 255, 255))):
        print(f"Strip should be {name}", flush=True)
        strip.fill(color)
        strip.show()
        time.sleep(2.0)
    print("If the colors were swapped, rerun with a different --order (e.g. BRG, GBR, RGB).")


def cmd_rainbow(strip: DotStar, args) -> None:
    n = len(strip)
    offset = 0
    while True:
        strip.set_all(wheel(offset + i * 256 // n) for i in range(n))
        strip.show()
        offset = (offset + 2) % 256
        time.sleep(args.delay)


def cmd_chase(strip: DotStar, args) -> None:
    n = len(strip)
    color = (args.r, args.g, args.b)
    tail = max(1, args.tail)
    i = 0
    while True:
        frame = [(0, 0, 0)] * n
        for k in range(tail):
            fade = (tail - k) / tail
            frame[(i - k) % n] = tuple(int(c * fade) for c in color)
        strip.set_all(frame)
        strip.show()
        i = (i + 1) % n
        time.sleep(args.delay)


def cmd_blink(strip: DotStar, args) -> None:
    color = (args.r, args.g, args.b)
    while True:
        strip.fill(color)
        strip.show()
        time.sleep(args.delay)
        strip.off()
        time.sleep(args.delay)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num", type=int, required=True, help="number of LEDs on the strip")
    p.add_argument("--port", default=DEFAULT_PORT, help="Arduino serial port")
    p.add_argument("--order", default="GBR", help="wire color order (default GBR, this strip)")
    p.add_argument("--brightness", type=float, default=0.2, help="0.0-1.0 (default 0.2)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("off", help="turn every LED off").set_defaults(func=cmd_off)
    sub.add_parser("test", help="cycle R, G, B, W to check wiring and color order").set_defaults(func=cmd_test)

    for name, func, helptext in (("solid", cmd_solid, "set every LED to one color and exit"),
                                 ("chase", cmd_chase, "a single dot running down the strip"),
                                 ("blink", cmd_blink, "blink the whole strip")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("r", type=int)
        sp.add_argument("g", type=int)
        sp.add_argument("b", type=int)
        if name != "solid":
            sp.add_argument("--delay", type=float, default=0.03 if name == "chase" else 0.5)
        if name == "chase":
            sp.add_argument("--tail", type=int, default=4)
        sp.set_defaults(func=func)

    sp = sub.add_parser("rainbow", help="scrolling rainbow")
    sp.add_argument("--delay", type=float, default=0.02)
    sp.set_defaults(func=cmd_rainbow)

    args = p.parse_args()

    strip = DotStar(args.num, port=args.port, order=args.order, brightness=args.brightness)
    keep_on = args.cmd == "solid"
    try:
        args.func(strip, args)
    except KeyboardInterrupt:
        pass
    finally:
        strip.close(turn_off=not keep_on)
    return 0


if __name__ == "__main__":
    sys.exit(main())
