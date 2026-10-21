#!/usr/bin/env python3
"""
Simple runtime test for a single BNO085 at a requested frequency.

Usage examples
--------------
python3 unit_tests/test_bno085_frequency.py --freq 100
python3 unit_tests/test_bno085_frequency.py --freq 200 --duration 10
python3 unit_tests/test_bno085_frequency.py --freq 100 --bus 7 --addr 0x4A
"""

import argparse
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BNO_PATH = WORKSPACE_ROOT / "Sensor_Tests" / "IMUs" / "BNO085"
if str(BNO_PATH) not in sys.path:
    sys.path.insert(0, str(BNO_PATH))

from bno085_live import IMUReader  # noqa: E402


def set_realtime_priority() -> None:
    """Try to set SCHED_FIFO real-time scheduling. Requires sudo or CAP_SYS_NICE."""
    try:
        import ctypes
        SCHED_FIFO = 1
        class SchedParam(ctypes.Structure):
            _fields_ = [("sched_priority", ctypes.c_int)]
        param = SchedParam(sched_priority=80)
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        ret = libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(param))
        if ret == 0:
            print("[INFO] Real-time scheduling (SCHED_FIFO, priority 80) enabled.")
        else:
            print("[WARN] Could not set real-time scheduling (run with sudo for best results).")
    except Exception as e:
        print(f"[WARN] Real-time scheduling unavailable: {e}")


def precise_sleep(target: float, busy_threshold: float = 0.002) -> None:
    """Sleep until target (perf_counter time).
    Coarse sleep to within busy_threshold seconds, then busy-wait.
    """
    remaining = target - time.perf_counter()
    if remaining <= 0:
        return
    if remaining > busy_threshold:
        time.sleep(remaining - busy_threshold)
    while time.perf_counter() < target:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test one BNO085 at a target frequency.")
    parser.add_argument("--freq",     type=float, default=None,                help="Target sample frequency in Hz.")
    parser.add_argument("--duration", type=float, default=5.0,                help="Test duration in seconds (default: 5).")
    parser.add_argument("--bus",      type=int,   default=1,                  help="I2C bus number (default: 1).")
    parser.add_argument("--addr",     type=lambda v: int(v, 0), default=0x4A, help="I2C address (default: 0x4A).")
    parser.add_argument("--no-rt",    action="store_true",                    help="Disable real-time scheduling attempt.")
    parser.add_argument(
        "--max-rate",
        action="store_true",
        help="Poll as fast as possible and report fresh (non-repeated) sample rate.",
    )
    return parser.parse_args()


def sample_key(sample: dict) -> tuple:
    """Tuple key for detecting repeated/cached payloads."""
    qi, qj, qk, qr = sample["quat"]
    ax, ay, az = sample["accel"]
    gx, gy, gz = sample["gyro"]
    return (qi, qj, qk, qr, ax, ay, az, gx, gy, gz)


def main() -> None:
    args = parse_args()

    if not args.max_rate and args.freq is None:
        raise SystemExit("[ERROR] Provide --freq for scheduled mode, or use --max-rate")
    if args.freq is not None and args.freq <= 0:
        raise SystemExit("[ERROR] --freq must be > 0")
    if args.duration <= 0:
        raise SystemExit("[ERROR] --duration must be > 0")

    if not args.no_rt:
        set_realtime_priority()

    period = (1.0 / args.freq) if args.freq is not None else None
    fresh_samples = 0
    repeated_samples = 0
    misses = 0
    total_polls = 0
    t_first = t_last = None
    last_key = None

    if args.max_rate:
        mode_text = "max-rate fresh-sample mode"
    else:
        mode_text = f"scheduled mode @ {args.freq:.2f} Hz"

    print(f"Starting BNO085 test: {mode_text}, duration={args.duration:.2f}s, bus={args.bus}, addr=0x{args.addr:02X}")

    start_time = time.perf_counter()
    next_tick = (start_time + period) if period is not None else None

    with IMUReader(bus=args.bus, address=args.addr) as imu:
        while True:
            if args.max_rate:
                now = time.perf_counter()
                if now - start_time >= args.duration:
                    break
            else:
                precise_sleep(next_tick)
                now = time.perf_counter()
                if now - start_time >= args.duration:
                    break

            total_polls += 1

            sample = imu.read()
            t_now = time.perf_counter()

            if sample is not None:
                current_key = sample_key(sample)
                if current_key != last_key:
                    fresh_samples += 1
                    if t_first is None:
                        t_first = t_now
                    t_last = t_now
                    last_key = current_key
                else:
                    repeated_samples += 1
            else:
                misses += 1

            if not args.max_rate:
                # Advance tick, clamp to avoid catch-up bursts after stalls
                next_tick += period
                if next_tick < t_now:
                    next_tick = t_now + period

    total_elapsed = time.perf_counter() - start_time
    expected_samples = int(args.freq * args.duration) if args.freq is not None else None

    if fresh_samples > 1:
        achieved_hz = (fresh_samples - 1) / (t_last - t_first)
    elif fresh_samples == 1:
        achieved_hz = 1.0 / total_elapsed
    else:
        achieved_hz = 0.0

    total_reads = fresh_samples + repeated_samples + misses
    print("\n--- Test Summary ---")
    if args.freq is not None:
        print(f"Target frequency:          {args.freq:.2f} Hz")
        print(f"Expected fresh samples:    {expected_samples}")
    print(f"Fresh samples read:        {fresh_samples}")
    print(f"Repeated/cached samples:   {repeated_samples}")
    print(f"Missed/empty reads:        {misses}")
    print(f"Total poll attempts:       {total_polls}")
    print(f"Elapsed time:              {total_elapsed:.3f}s")
    print(f"Fresh sample frequency:    {achieved_hz:.2f} Hz")
    print(f"Miss rate:                 {100*misses/total_reads:.1f}%" if total_reads > 0 else "Miss rate: N/A")


if __name__ == "__main__":
    main()