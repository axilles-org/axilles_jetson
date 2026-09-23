#!/usr/bin/env python3
"""
sweep_check.py

Step one of debugging the calibration: record an ankle sweep and look at the
encoder on its own, before any fitting happens.

Run on the Jetson, with the exo worn:

    python sweep_check.py                 # 20 s, default
    python sweep_check.py --seconds 30
    python sweep_check.py --sign -1       # if the converted trace comes out inverted

What it plots
-------------
Two traces of the same signal:

  RAW       ankle_encoder_deg exactly as the logger writes it, 0-360 degrees.
  CONVERTED wrap180(raw - zero), which is steps 1 and 2 of the conversion and
            nothing else.

The conversion being checked is:

    ankle = sign * ratio * wrap180(raw_deg - zero_deg)

    wrap180(x) = (x + 180) % 360 - 180

`zero_deg` is taken from the first `--neutral` seconds, which is why the protocol
asks you to start at neutral and hold briefly. `sign` and `ratio` default to +1 and
1.0, so the converted trace is the zeroed, unwrapped angle with nothing fitted on
top of it. If that trace does not look like your ankle moving, nothing downstream
can be trusted.

Outputs go to calibration/: a PNG, an npz of the raw arrays, and printed statistics.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

# Allow this script to be run directly from any working directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import exo_frame as E

OUT_DIR = ROOT / "calibration"
DEFAULT_SECONDS = 20.0
NEUTRAL_SECONDS = 2.0

INSTRUCTIONS = """
    1. Sit down, or stand on your LEFT leg.
    2. Put the right ankle in its NEUTRAL position and HOLD IT STILL.
       The first {neutral:.0f} seconds are used as the zero, so keep it steady.
    3. When the countdown ends, sweep the ankle slowly through its FULL range:
       toes up, toes down, repeat. About 5 full cycles over {sweep:.0f} seconds.
    4. Keep your shank still. Only the ankle moves.
"""


def wrap180(x):
    """Fold degrees into [-180, +180). This is the step that survives the 0/360 seam."""
    return (np.asarray(x, dtype=np.float64) + 180.0) % 360.0 - 180.0


def convert(raw_deg, zero_deg, sign=1, ratio=1.0):
    """ankle = sign * ratio * wrap180(raw - zero). The whole conversion, in one line."""
    return sign * ratio * wrap180(np.asarray(raw_deg, dtype=np.float64) - zero_deg)


def describe(raw, converted, zero, fs, neutral_n):
    """Print what the encoder actually did, including the failure signatures."""
    finite = np.isfinite(raw)
    n = int(finite.sum())
    print("\n" + "=" * 72)
    print("  ENCODER")
    print("=" * 72)
    print(f"  samples                : {n} of {len(raw)} finite")
    print(f"  zero (first {neutral_n / fs:.1f} s)   : {zero:.4f} deg")
    print(f"  raw range              : {np.nanmin(raw):.3f} .. {np.nanmax(raw):.3f} deg "
          f"(span {np.nanmax(raw) - np.nanmin(raw):.3f})")
    print(f"  converted range        : {np.nanmin(converted):+.3f} .. "
          f"{np.nanmax(converted):+.3f} deg "
          f"(span {np.nanmax(converted) - np.nanmin(converted):.3f})")

    # How often the value actually changes. The driver swallows OSError and returns
    # the previous angle, so a dead read looks like a perfectly steady reading.
    rv = raw[finite]
    changed = int(np.sum(np.diff(rv) != 0.0))
    dur = n / fs if fs > 0 else 0.0
    print(f"  updates                : {changed} changes in {dur:.1f} s "
          f"= {changed / dur if dur else 0:.1f} Hz")
    print(f"  standard deviation     : {np.nanstd(rv):.4f} deg")

    lsb = 360.0 / 4096.0
    steps = np.abs(np.diff(rv))
    nonzero = steps[steps > 0]
    if len(nonzero):
        print(f"  smallest step          : {nonzero.min():.4f} deg "
              f"(1 LSB = {lsb:.4f})")

    # A wrap shows up as a jump of nearly a full turn between consecutive samples.
    wraps = int(np.sum(np.abs(np.diff(rv)) > 180.0))
    print(f"  0/360 wraps in raw     : {wraps}"
          + ("  <- the unwrap step is doing real work here" if wraps else ""))

    print("\n  what to look for:")
    if changed == 0:
        print("    ! The encoder NEVER changed. It is not reading the joint at all.")
    elif changed / dur < 20:
        print(f"    ! Only {changed / dur:.1f} updates/s. Reads are failing and the")
        print("      driver is handing back the last good value.")
    if np.nanstd(rv) < 0.05:
        print("    ! Almost no variation. Either the ankle did not move, or the")
        print("      magnet is not turning with the joint.")
    span = float(np.nanmax(converted) - np.nanmin(converted))
    if span < 10.0:
        print(f"    ! Converted span is only {span:.1f} deg. A full ankle sweep should")
        print("      cover 30-50 deg. Check the magnet coupling.")
    else:
        print(f"    Converted span of {span:.1f} deg looks like a real ankle sweep.")
    if np.abs(np.nanmedian(converted[:neutral_n])) > 1.0:
        print("    ! The neutral hold is not centred on zero, so the hold was not")
        print("      steady. Redo it and keep still for the first seconds.")


def plot(t, raw, converted, zero, neutral_n, path, sign, ratio):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  (matplotlib unavailable: {exc})")
        return False

    from matplotlib.ticker import FuncFormatter, MultipleLocator

    # Both traces are angles in degrees; label the ticks with the degree sign so the
    # units are visible on the axis itself rather than only in the axis title.
    degrees = FuncFormatter(lambda v, _pos: f"{v:g}°")

    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("Ankle sweep: raw encoder vs converted angle  (both in degrees)",
                 fontsize=13)

    ax[0].plot(t, raw, color="tab:blue", lw=1.0)
    ax[0].axhline(zero, color="k", ls="--", lw=0.9, label=f"zero = {zero:.2f}°")
    ax[0].axvspan(t[0], t[min(neutral_n, len(t) - 1)], color="grey", alpha=0.18,
                  label="neutral hold (defines the zero)")
    ax[0].set_ylabel("RAW encoder (degrees)")
    ax[0].set_title("as logged: raw_12bit x 360/4096, range 0°..360°",
                    fontsize=9, loc="left")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, converted, color="tab:red", lw=1.0)
    ax[1].axhline(0.0, color="k", ls="--", lw=0.9)
    ax[1].axvspan(t[0], t[min(neutral_n, len(t) - 1)], color="grey", alpha=0.18)
    ax[1].set_ylabel("CONVERTED ankle (degrees)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_title(f"sign x ratio x wrap180(raw - zero), "
                    f"with sign={sign:+d} ratio={ratio:g}", fontsize=9, loc="left")
    ax[1].grid(True, alpha=0.3)

    for a in ax:
        a.yaxis.set_major_formatter(degrees)
    # 5 degree gridlines on the converted trace, so ankle range is readable by eye.
    span = float(np.nanmax(converted) - np.nanmin(converted))
    ax[1].yaxis.set_major_locator(MultipleLocator(5.0 if span <= 60 else 10.0))

    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                   help="sweep duration after the neutral hold")
    p.add_argument("--neutral", type=float, default=NEUTRAL_SECONDS,
                   help="seconds at the start used to define the zero")
    p.add_argument("--sign", type=int, choices=(-1, 1), default=1,
                   help="+1 if dorsiflexion increases raw degrees")
    p.add_argument("--ratio", type=float, default=1.0,
                   help="joint degrees per encoder degree")
    p.add_argument("--imu-hz", type=float, default=200.0)
    p.add_argument("--fs", type=float, default=200.0)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--replay", default=None,
                   help="re-analyse a saved sweep npz instead of recording")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.replay:
        d = np.load(args.replay)
        t, raw = d["time"], d["encoder"]
        fs = float(len(t) / (t[-1] - t[0])) if len(t) > 1 else args.fs
        print(f"Replaying {args.replay}: {len(t)} samples at {fs:.0f} Hz")
    else:
        total = args.neutral + args.seconds
        print("=" * 72)
        print("  ANKLE SWEEP CHECK")
        print("=" * 72)
        print(INSTRUCTIONS.format(neutral=args.neutral, sweep=args.seconds))
        print(f"  Recording {total:.0f} s total "
              f"({args.neutral:.0f} s neutral + {args.seconds:.0f} s sweep).")

        try:
            mod = E.load_sensor_hub(args.imu_hz)
            hub = mod.SensorHub()
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"\nCannot reach the sensors.\n\n{exc}")
            return 1

        try:
            E._wait_for_enter("\n  Press ENTER when the ankle is NEUTRAL and still...")
            import time
            for k in (3, 2, 1):
                print(f"    starting in {k}...", end="\r", flush=True)
                time.sleep(1.0)
            print("    RECORDING NOW              ")
            rec = E.record_phase(hub, total, args.fs)
        finally:
            hub.close()

        t, raw = rec["time"], rec["encoder"]
        fs = float(len(t) / (t[-1] - t[0])) if len(t) > 1 else args.fs

        npz = out_dir / f"sweep_{stamp}.npz"
        np.savez_compressed(npz, **rec)
        print(f"\n  raw arrays saved -> {npz}")

    neutral_n = max(1, int(args.neutral * fs))
    zero = float(np.nanmedian(raw[:neutral_n]))
    converted = convert(raw, zero, args.sign, args.ratio)

    describe(raw, converted, zero, fs, neutral_n)

    png = out_dir / f"sweep_{stamp}.png"
    if plot(t, raw, converted, zero, neutral_n, png, args.sign, args.ratio):
        print(f"\n  plot -> {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
