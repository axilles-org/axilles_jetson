#!/usr/bin/env python3
"""Turn on the output driver of 40-pin header pads at runtime (Jetson Orin Nano).

The boot-time pinmux leaves every header GPIO with TRISTATE=1, which disconnects
the pad's output driver: Linux can set a GPIO "high" but the pin never drives.
This clears that one bit for the chosen pins by writing their pinmux register
through /dev/mem.

  * Runtime only: no boot file, device tree or extlinux change. A reboot
    restores the original settings.
  * Touches only the registers listed in PADS, and only bit 4 (tristate).
  * Before writing, the register must decode to the values that debugfs
    reported (function rsvd1, pull-down, input enabled, tristate on). Anything
    else aborts without writing.

Register layout and offsets: Linux drivers/pinctrl/tegra/pinctrl-tegra234.c
(PINGROUP table, PIN_PINGROUP_ENTRY_Y). Base 0x02430000 and size 0x19100 come
from /proc/device-tree/bus@0/pinmux@2430000/reg on this board.

Usage (needs root):
    sudo python3 led-strip/pinmux_enable.py show
    sudo python3 led-strip/pinmux_enable.py enable 19 21
    sudo python3 led-strip/pinmux_enable.py disable 19 21     # put tristate back
"""
from __future__ import annotations

import ctypes
import mmap
import os
import sys

BASE = 0x02430000
SIZE = 0x19100

# header pin -> (pad name, register offset from BASE)
PADS = {
    19: ("spi1_mosi_pz5", 0xD040),
    21: ("spi1_miso_pz4", 0xD018),
    23: ("spi1_sck_pz3", 0xD028),
    24: ("spi1_cs0_pz6", 0xD008),
}

PM_MASK = 0x3        # bits 1:0 function select
PUPD_SHIFT = 2       # bits 3:2  0 none, 1 pull-down, 2 pull-up
TRISTATE = 1 << 4
E_INPUT = 1 << 6
SFSEL = 1 << 10      # GPIO/SFIO select, owned by the kernel GPIO driver
SCHMITT = 1 << 12


def decode(v: int) -> str:
    pull = {0: "none", 1: "down", 2: "up"}.get((v >> PUPD_SHIFT) & 3, "?")
    return (f"0x{v:08x}  pm={v & PM_MASK} pull={pull} tristate={int(bool(v & TRISTATE))} "
            f"input={int(bool(v & E_INPUT))} sfsel={int(bool(v & SFSEL))} "
            f"schmitt={int(bool(v & SCHMITT))}")


class Regs:
    def __init__(self) -> None:
        page = mmap.PAGESIZE
        self._len = (SIZE + page - 1) // page * page
        self._fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self._mm = mmap.mmap(self._fd, self._len, mmap.MAP_SHARED,
                             mmap.PROT_READ | mmap.PROT_WRITE, offset=BASE)

    def _word(self, off: int) -> ctypes.c_uint32:
        # ctypes gives a real 32-bit load/store; byte-wise access to device
        # registers is not safe on Tegra.
        return ctypes.c_uint32.from_buffer(self._mm, off)

    def read(self, off: int) -> int:
        return self._word(off).value

    def write(self, off: int, value: int) -> None:
        self._word(off).value = value

    def close(self) -> None:
        self._mm.close()
        os.close(self._fd)


def expected_default(v: int) -> bool:
    """True if the register looks like the boot default debugfs showed."""
    return ((v & PM_MASK) == 1 and ((v >> PUPD_SHIFT) & 3) == 1
            and bool(v & E_INPUT) and (v >> 16) == 0)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("show", "enable", "disable"):
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    pins = [int(p) for p in sys.argv[2:]] or (list(PADS) if cmd == "show" else [])
    if not pins:
        print("name the header pins, e.g. enable 19 21")
        return 2
    bad = [p for p in pins if p not in PADS]
    if bad:
        print(f"unsupported pin(s) {bad}; supported: {sorted(PADS)}")
        return 2
    if os.geteuid() != 0:
        print("needs root: run with sudo")
        return 1

    regs = Regs()
    try:
        for pin in pins:
            name, off = PADS[pin]
            old = regs.read(off)
            print(f"pin {pin:2} {name:14} @0x{BASE + off:08x}  {decode(old)}")
            if cmd == "show":
                continue
            if not expected_default(old):
                print("   unexpected value, not writing this pin")
                continue
            new = old & ~TRISTATE if cmd == "enable" else old | TRISTATE
            if new == old:
                print("   already set, nothing to do")
                continue
            regs.write(off, new)
            print(f"   now                          {decode(regs.read(off))}")
    finally:
        regs.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
