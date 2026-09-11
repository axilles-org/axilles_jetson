#!/usr/bin/env python3
"""
walk_check.py

Step two: record walking and answer whether a Georgia Tech comparison is even
possible, before fitting any rotation.

Run on the Jetson, with the exo worn:

    python3 walk_check.py                  # 60 s
    python3 walk_check.py --seconds 90
    python3 walk_check.py --replay calibration/walk_<stamp>.npz

Three questions, in order. Each one is pointless if the previous fails.

  1. Are the IMUs actually updating at the sampling rate? The drivers hand back the
     last good packet when a read fails, so a starved sensor looks like clean data.
     The sweep check already showed the encoder starved to 21.6 Hz on this bus, so
     this is not hypothetical.

  2. Do the FSRs give consistent heel strikes? The gait cycle is the only common
     reference between your recording and Georgia Tech's, so unreliable strides make
     everything downstream meaningless.

  3. Does your gait cycle look like theirs? Compared through |gyro|, the magnitude
     of angular velocity, which is the same number in every frame. That makes it
     valid BEFORE any rotation is known - and if these two curves disagree, no
     rotation exists that can reconcile them.

The encoder is deliberately not involved. The rotation fit uses IMU and heel strikes
only; the encoder supplies a separate model feature.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

import exo_frame as E

OUT_DIR = Path("calibration")

INSTRUCTIONS = """
    Walk at a comfortable, steady pace on level ground.
    Keep going until told to stop. Turns are fine; just keep walking.
"""

MIN_IMU_HZ = 100.0        # below this the gait cycle cannot be resolved
MIN_STRIDES = 20
MIN_SHAPE_CORR = 0.70     # |gyro| gait cycle, exo vs Georgia Tech

# Stride-time consistency is judged against the reference dataset rather than an
# absolute number. AB06's own treadmill strides vary by about 21% as detected here,
# so a fixed 20% cap would fail the gold standard itself and tell you nothing. The
# exo only has to be in the same league.
STRIDE_CV_FLOOR = 0.25    # never complain below this
STRIDE_CV_FACTOR = 1.5    # nor unless it is this much worse than the reference


def phase_shift(a, b):
    """
    Circular lag that best aligns two gait-cycle waveforms, and the correlation there.

    Maximises the signed correlation, not its magnitude. These are |gyro| traces,
    which are non-negative, so an anti-correlated alignment is never the right
    answer - and accepting one is exactly how a phase search can lock onto the wrong
    peak and produce a confident, badly wrong result.
    """
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0, 0.0
    n = len(a)
    corr = np.array([np.dot(a, np.roll(b, k)) for k in range(n)]) / denom
    k = int(np.argmax(corr))
    return (k if k <= n // 2 else k - n), float(corr[k])


def detect_strides(heel, fs, fraction=0.5):
    """Heel-strike indices from a per-recording adaptive threshold."""
    lo, hi = np.percentile(heel, [10.0, 90.0])
    thr = lo + fraction * (hi - lo)
    contact = (heel > thr).astype(int)
    hs = np.where(np.diff(contact) == 1)[0] + 1
    return hs, float(thr), contact


def report(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--fs", type=float, default=200.0)
    p.add_argument("--imu-hz", type=float, default=200.0)
    p.add_argument("--hf-root", default="mrsd-exo-ankle")
    p.add_argument("--subject", default="AB06")
    p.add_argument("--trial", default="treadmill_01_01")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--replay", default=None,
                   help="re-analyse a saved walk npz instead of recording")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- reference first, so a missing dataset costs no walking ----
    try:
        trials = E.preflight_reference(args.hf_root, args.subject)
    except FileNotFoundError as exc:
        print(f"Reference dataset problem:\n\n{exc}")
        return 1
    trial = args.trial if args.trial in trials else trials[0]

    # ---- record or replay ----
    if args.replay:
        d = np.load(args.replay)
        rec = {k: d[k] for k in d.files}
        print(f"Replaying {args.replay}: {len(rec['time'])} samples")
    else:
        print("=" * 72)
        print("  WALKING CHECK")
        print("=" * 72)
        print(INSTRUCTIONS)
        print(f"  Recording {args.seconds:.0f} s.")
        try:
            mod = E.load_sensor_hub(args.imu_hz)
            hub = mod.SensorHub()
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"\nCannot reach the sensors.\n\n{exc}")
            return 1
        try:
            E._wait_for_enter("\n  Press ENTER when you are ready to start walking...")
            import time
            for k in (3, 2, 1):
                print(f"    starting in {k}...", end="\r", flush=True)
                time.sleep(1.0)
            print("    RECORDING NOW              ")
            rec = E.record_phase(hub, args.seconds, args.fs)
        finally:
            hub.close()
        npz = out_dir / f"walk_{stamp}.npz"
        np.savez_compressed(npz, **rec)
        print(f"\n  raw arrays saved -> {npz}")

    t = rec["time"]
    dur = float(t[-1] - t[0])
    fs = len(t) / dur if dur > 0 else args.fs
    problems = []

    # ---- 1. sensor health ----
    report("1. ARE THE SENSORS KEEPING UP?")
    health = E.channel_health(rec)
    print(f"  {'channel':<14}{'updates':>10}{'finite':>9}   status")
    for key in ("foot_gyro", "foot_accel", "shank_gyro", "shank_accel",
                "encoder", "heel", "toe"):
        h = health[key]
        need = MIN_IMU_HZ if "foot" in key or "shank" in key else 20.0
        ok = h["update_hz"] >= need
        print(f"  {key:<14}{h['update_hz']:>9.1f}H{h['finite_fraction']:>8.0%}   "
              f"{'ok' if ok else 'TOO SLOW (need %.0f Hz)' % need}")
        if not ok and ("foot" in key or "shank" in key):
            problems.append(f"{key} only {h['update_hz']:.0f} Hz")
    print(f"\n  sampled at {fs:.0f} Hz for {dur:.1f} s")
    if problems:
        print("\n  The IMUs are being starved on the I2C bus. The gait cycle cannot")
        print("  be resolved at this rate, so the rotation fit would be meaningless.")

    # ---- 2. stride detection ----
    report("2. ARE THE HEEL STRIKES CLEAN?")
    heel = E._ffill_nan(rec["heel"])
    hs, thr, contact = detect_strides(heel, fs)
    bounds = E.stride_bounds(hs, len(heel), fs)
    print(f"  heel FSR range     : {heel.min():.0f} .. {heel.max():.0f} counts")
    print(f"  adaptive threshold : {thr:.0f}")
    print(f"  raw heel strikes   : {len(hs)}")
    print(f"  usable strides     : {len(bounds)} (0.5-2.5 s each)")
    if bounds:
        durs = np.array([(b - a) / fs for a, b in bounds])
        cv = float(durs.std() / durs.mean()) if durs.mean() else 1.0
        print(f"  stride time        : {durs.mean():.3f} +/- {durs.std():.3f} s "
              f"(cv {cv:.1%})")
        print(f"  stance fraction    : {contact.mean():.0%} of the recording")
        if len(bounds) < MIN_STRIDES:
            problems.append(f"only {len(bounds)} strides")

        # Judge against the reference's own consistency, not an absolute number.
        ref = E.read_hf_trial(args.hf_root, args.subject, trial)
        rb = E.stride_bounds(ref["heel_strikes"], len(ref["time"]), ref["fs"])
        ref_durs = np.array([(b - a) / ref["fs"] for a, b in rb])
        ref_cv = float(ref_durs.std() / ref_durs.mean()) if len(rb) else float("nan")
        limit = max(STRIDE_CV_FLOOR, STRIDE_CV_FACTOR * ref_cv)
        print(f"  reference cv       : {ref_cv:.1%} ({args.subject}/{trial}), "
              f"so the limit here is {limit:.1%}")
        if cv > limit:
            problems.append(f"stride time varies by {cv:.0%}")
            print(f"\n  Stride times vary by {cv:.0%}, well past the reference's own")
            print("  {:.0%}. Either the walking was uneven or the FSR threshold is"
                  .format(ref_cv))
            print("  catching spurious contacts.")
    else:
        problems.append("no usable strides")
        print("\n  No usable strides. The heel FSR is not resolving ground contact.")

    # ---- 3. shape comparison against Georgia Tech ----
    report("3. DOES YOUR GAIT CYCLE MATCH GEORGIA TECH?")
    print(f"  reference: {args.subject}/{trial}")
    corrs = {}
    if len(bounds) >= 3:
        hf = E.read_hf_trial(args.hf_root, args.subject, trial)
        hb = E.stride_bounds(hf["heel_strikes"], len(hf["time"]), hf["fs"])
        print(f"  their strides      : {len(hb)}      yours: {len(bounds)}")
        print(f"\n  |gyro| gait cycle, compared at the best phase alignment:")
        print("  (magnitude is the same in every frame, so this is valid before")
        print("   any rotation is known)\n")
        cycles = {}
        for seg in ("foot", "shank"):
            exo_c = E.mean_cycle(np.linalg.norm(E._ffill_nan(rec[f"{seg}_gyro"]),
                                                axis=1), bounds)
            gt_c = E.mean_cycle(np.linalg.norm(hf["gyro"][seg], axis=1), hb)
            lag, r = phase_shift(gt_c, exo_c)
            corrs[seg] = r
            cycles[seg] = (gt_c, np.roll(exo_c, lag), lag)
            flag = "ok" if r >= MIN_SHAPE_CORR else "TOO LOW"
            print(f"    {seg:<6} r = {r:+.3f} at {lag / len(gt_c):+.0%} phase   {flag}")
            print(f"           peak |gyro|  theirs {gt_c.max():5.2f}   "
                  f"yours {exo_c.max():5.2f} rad/s")
            if r < MIN_SHAPE_CORR:
                problems.append(f"{seg} gait shape r={r:+.2f}")
        _plot(t, rec, heel, thr, cycles, out_dir / f"walk_{stamp}.png", fs, args.subject)
        print(f"\n  plot -> {out_dir / f'walk_{stamp}.png'}")
    else:
        print("  Skipped: not enough strides to build a gait cycle.")

    # ---- verdict ----
    report("VERDICT")
    if not problems:
        print("  Everything needed for the transformation is present.")
        print("  The IMUs are keeping up, the strides are consistent, and your gait")
        print("  cycle matches Georgia Tech's. Fitting a rotation on this is sound.")
        print("\n  Next: python3 exo_frame.py calibrate")
        return 0
    print(f"  {len(problems)} problem(s) block the Georgia Tech comparison:")
    for x in problems:
        print(f"    - {x}")
    print("\n  Fix these before fitting a rotation; a fit on this data would produce")
    print("  a confident-looking matrix built on unusable inputs.")
    return 1


def _plot(t, rec, heel, thr, cycles, path, fs, subject) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.3])
    fig.suptitle("Walking check: raw traces, and your gait cycle vs Georgia Tech",
                 fontsize=13)

    for row, seg in enumerate(("foot", "shank")):
        ax = fig.add_subplot(gs[row, :])
        g = E._ffill_nan(rec[f"{seg}_gyro"])
        for j, a in enumerate("xyz"):
            ax.plot(t, g[:, j], lw=0.6, label=f"g{a}")
        ax.set_ylabel(f"{seg} gyro (rad/s)")
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    ax.plot(t, heel, color="tab:green", lw=0.7, label="heel FSR")
    ax.plot(t, E._ffill_nan(rec["toe"]), color="tab:red", lw=0.7, alpha=0.6,
            label="toe FSR")
    ax.axhline(thr, color="k", ls="--", lw=0.9, label=f"threshold {thr:.0f}")
    ax.set_ylabel("FSR (counts)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    phase = np.linspace(0, 100, len(next(iter(cycles.values()))[0]))
    for col, seg in enumerate(("foot", "shank")):
        gt_c, exo_c, lag = cycles[seg]
        ax = fig.add_subplot(gs[3, col])
        ax.plot(phase, gt_c, lw=1.6, label=f"{subject} (Georgia Tech)")
        ax.plot(phase, exo_c, lw=1.6, label="your exo (aligned)")
        ax.set_title(f"{seg}  |gyro| over the gait cycle", fontsize=10)
        ax.set_xlabel("Gait cycle (%)")
        ax.set_ylabel("|gyro| (rad/s)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


if __name__ == "__main__":
    sys.exit(main())
