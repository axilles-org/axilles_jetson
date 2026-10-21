#!/usr/bin/env python3
"""
run_vescx_test.py

Simple CAN scanner to detect nodes on a `socketcan` interface (e.g. can0).

This script listens for frames on the selected interface and prints
unique arbitration IDs observed. It does not send any motor commands.
Use this first to find candidate VESC node IDs before running active tests.
"""
from __future__ import annotations

import argparse
import collections
import sys
import time

import can


def listen(interface: str, timeout: float, continuous: bool) -> int:
    try:
        bus = can.interface.Bus(channel=interface, bustype="socketcan")
    except Exception as exc:  # pragma: no cover - runtime environment
        print(f"Failed to open CAN interface {interface}: {exc}", file=sys.stderr)
        print(
            "Ensure the interface exists and is up, e.g.:",
            file=sys.stderr,
        )
        print("  sudo ip link set can0 type can bitrate 500000 up", file=sys.stderr)
        return 2

    seen = collections.Counter()
    start = time.time()
    print(f"Listening on {interface} for {'infinite' if continuous else f'{timeout}s'} (Ctrl-C to stop)")

    try:
        while True:
            remaining = None if continuous else max(0.0, timeout - (time.time() - start))
            # use a small recv timeout to be responsive to Ctrl-C
            recv_timeout = None if remaining is None else min(0.5, remaining)
            msg = bus.recv(timeout=recv_timeout)
            if msg is None:
                # loop to check timeout/interrupt
                if not continuous and (time.time() - start) >= timeout:
                    break
                continue

            aid = msg.arbitration_id
            seen[aid] += 1
            print(f"ID 0x{aid:X} len={getattr(msg, 'dlc', len(msg.data))} ext={msg.is_extended_id} data={msg.data.hex()} count={seen[aid]}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if seen:
            print("\nSummary of IDs seen:")
            for aid, cnt in seen.most_common():
                print(f"  0x{aid:X}  ({cnt} frames)")
        else:
            print("\nNo CAN frames observed.")
        try:
            bus.shutdown()
        except Exception:
            pass

    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Detect CAN nodes on a socketcan interface (e.g. can0)")
    p.add_argument("--interface", "-i", default="can0", help="socketcan interface to use (default: can0)")
    p.add_argument("--timeout", "-t", type=float, default=5.0, help="seconds to listen (default: 5). Ignored with --continuous")
    p.add_argument("--continuous", "-c", action="store_true", help="listen until interrupted")
    args = p.parse_args()

    code = listen(args.interface, args.timeout, args.continuous)
    sys.exit(code)


if __name__ == "__main__":
    main()
