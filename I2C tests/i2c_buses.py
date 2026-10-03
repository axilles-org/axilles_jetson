#!/usr/bin/env python3
"""
i2c_buses.py

Map every /dev/i2c-N to its device-tree node, its clock, and what is on it.

Run on the Jetson:

    python3 i2c_speed_probe.py   # first: establishes your sensors sit on a 100 kHz bus
    python3 i2c_buses.py         # then: finds which bus is already at 400 kHz

The speed probe measured the sensor bus at ~98 kHz, but the device tree also lists
nodes already configured at 400 kHz. If one of those is broken out on the header,
moving the sensors there is a 4x bandwidth increase with no device-tree edit and no
reboot - which is a far smaller risk than modifying the tree and rebooting a robot.

This prints, per bus: the kernel number, the device-tree node, the configured clock,
and which of the exo's addresses respond. Scanning is done with 1-byte reads, which
is the least intrusive probe available; addresses NOT belonging to the exo are only
counted, never poked repeatedly.

Read-only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

EXO = {0x4A: "BNO085 foot", 0x4B: "BNO085 shank",
       0x36: "AS5600 encoder", 0x48: "ADS1115 FSR"}


def node_clock(of_node: Path):
    """clock-frequency for a device-tree node, as an int, or None."""
    f = of_node / "clock-frequency"
    if not f.exists():
        return None
    try:
        raw = f.read_bytes()
        return (int.from_bytes(raw, "big") if len(raw) == 4
                else int(raw.decode().strip()))
    except Exception:
        return None


def enumerate_buses():
    """[(bus_number, node_name, clock_hz)] for every i2c adapter the kernel exposes."""
    out = []
    root = Path("/sys/bus/i2c/devices")
    if not root.exists():
        return out
    for entry in sorted(root.glob("i2c-*")):
        m = re.match(r"i2c-(\d+)$", entry.name)
        if not m:
            continue
        num = int(m.group(1))
        of_node = entry / "of_node"
        name, clock = "(no device-tree node)", None
        if of_node.exists():
            try:
                name = of_node.resolve().name
            except Exception:
                name = str(of_node)
            clock = node_clock(of_node)
        if clock is None:
            try:
                name_file = entry / "name"
                if name_file.exists():
                    name = f"{name}  [{name_file.read_text().strip()}]"
            except Exception:
                pass
        out.append((num, name, clock))
    return out


def scan(bus_num, addresses):
    """
    Which of the exo's addresses acknowledge on this bus.

    Deliberately probes ONLY the four known exo addresses rather than sweeping
    0x03-0x77 the way i2cdetect does. A full sweep on a Jetson touches the PMIC,
    EEPROMs and other board management devices on the system buses; a stray read is
    usually harmless but there is no reason to take that risk when the question is
    simply "where are my sensors".
    """
    try:
        import smbus2
    except ImportError:
        return None
    try:
        bus = smbus2.SMBus(bus_num)
    except Exception:
        return None
    found = []
    try:
        for addr in sorted(addresses):
            try:
                bus.read_byte(addr)
                found.append(addr)
            except Exception:
                continue
    finally:
        try:
            bus.close()
        except Exception:
            pass
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-scan", action="store_true",
                    help="list buses and clocks without probing addresses")
    args = ap.parse_args()

    print("=" * 78)
    print("  I2C BUSES ON THIS SYSTEM")
    print("=" * 78)

    buses = enumerate_buses()
    if not buses:
        print("  No I2C adapters found under /sys/bus/i2c/devices.")
        print("  Run this on the Jetson.")
        return 1

    print(f"\n  {'bus':>5}  {'clock':>9}  {'device-tree node':<28} devices")
    print("  " + "-" * 72)

    fast, sensor_bus = [], None
    for num, name, clock in buses:
        clock_s = f"{clock / 1000:.0f} kHz" if clock else "unknown"
        devices = ""
        if not args.no_scan:
            found = scan(num, EXO)
            if found is None:
                devices = "(smbus2 unavailable)"
            elif found:
                devices = ", ".join(f"0x{a:02X} {EXO[a]}" for a in found)
                sensor_bus = num
            else:
                devices = "-"
        print(f"  {num:>5}  {clock_s:>9}  {name:<28} {devices}")
        if clock and clock >= 400_000:
            fast.append((num, name))

    print("\n" + "=" * 78)
    print("  WHAT TO DO")
    print("=" * 78)

    if sensor_bus is not None:
        cur = next((c for n, _, c in buses if n == sensor_bus), None)
        print(f"  Your exo sensors are on bus {sensor_bus}"
              + (f", configured at {cur / 1000:.0f} kHz." if cur else "."))
        if cur and cur >= 400_000:
            print("  That bus is already at 400 kHz, so the clock is not your problem.")
            print("  Re-run i2c_speed_probe.py --bus %d to confirm what it delivers."
                  % sensor_bus)
            return 0

    if fast:
        print(f"\n  {len(fast)} bus(es) are already configured at 400 kHz:")
        for num, name in fast:
            print(f"    /dev/i2c-{num}   {name}")
        print("\n  Two routes, in increasing order of risk:")
        print("\n    A. Move the sensors onto one of those buses.")
        print("       Rewiring only - no device-tree edit, no reboot, fully reversible")
        print("       by moving the wires back. Check the carrier-board pinout to see")
        print("       which of those nodes is broken out on the 40-pin header, then")
        print("       update I2C_BUS in 'Data collection/data_collection.py'.")
        print("\n    B. Raise the sensor bus to 400 kHz in the device tree.")
        print("       No rewiring, but it needs a device-tree overlay and a reboot,")
        print("       and a bad tree can leave the board unbootable. Worth doing only")
        print("       if no fast bus is exposed on the header.")
        print("\n  Either way, re-run i2c_speed_probe.py afterwards to confirm you")
        print("  actually got 400 kHz, then imu_drain_test.py to see the IMU gain.")
    else:
        print("  No bus on this system is configured above 100 kHz, so moving the")
        print("  sensors would not help. The device-tree route is the only option.")

    print("\n  Independent of the bus: drop BNO_REPORT_GAME_ROTATION_VECTOR.")
    print("  Georgia Tech has no quaternion data, so no model feature can consume it,")
    print("  and your own measurement showed removing it took the foot IMU from")
    print("  64 to 100 calls/s. That is free bandwidth whatever you decide here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
