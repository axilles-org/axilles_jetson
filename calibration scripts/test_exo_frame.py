#!/usr/bin/env python3
"""
test_exo_frame.py

Validation suite for exo_frame.py. Not part of the ingester - run it standalone:

    python test_exo_frame.py

A frame transform that is merely plausible is worthless, because a wrong sign or a
swapped axis produces data that looks perfectly healthy and trains a model that
drives the exo backwards. So the checks run from provable to empirical:

  1-4   Algebraic properties that must hold exactly, whatever the data says.
  5     Synthetic recovery against a known ground-truth rotation. The decisive one:
        the only test where the right answer is known in advance.
  6-11  Physical agreement between transformed HF data and real exo data.
  12-15 Integrity of the output and of the deployment path.

Exit code is 0 only if every critical check passes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exo_frame import (
    ExoFrame, FEATURES, G_TO_MS2, N_FEATURES, SEGMENTS,
    _angle_between, _fit_lag_and_rotation, average_rotations, is_rotation,
    kabsch, list_hf_trials, mean_cycle, principal_axis, read_exo_log,
    read_hf_trial, rotation_angle, stride_bounds,
)

HF_ROOT = ROOT / "mrsd-exo-ankle"
EXO_DIR = ROOT / "Data collection/data"
SUBJECT, TRIAL = "AB06", "treadmill_01_01"
EXCLUDE = ("20260404_214949",)

# A rotation recovered from noiseless synthetic data should be near-exact.
SYNTHETIC_TOL_DEG = 2.0
# Limited by the arccos used to measure the angle between rotations, whose
# derivative blows up at zero; machine epsilon there is about 1e-6 deg.
KABSCH_EXACT_TOL_DEG = 1e-4
# Cross-dataset agreement is necessarily softer: different people, speeds, hardware.
GRAVITY_TOL_DEG, SAGITTAL_TOL_DEG, MIN_SAGITTAL_CORR = 20.0, 25.0, 0.80
ANTIPHASE_BAND = 0.15
THRESHOLD_ROTATION_TOL_DEG = 3.0
MAX_LATENCY_US = 25.0


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    critical: bool = True


class Suite:
    def __init__(self):
        self.checks: list[Check] = []

    def run(self, name: str, fn: Callable[[], tuple[bool, str]], critical: bool = True):
        try:
            passed, detail = fn()
        except Exception as exc:          # a check that cannot run is a failed check
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        self.checks.append(Check(name, passed, detail, critical))

    def report(self) -> bool:
        w = max(len(c.name) for c in self.checks) + 2
        print("\n" + "=" * 78)
        print("VALIDATION REPORT")
        print("=" * 78)
        for c in self.checks:
            tag = "PASS" if c.passed else ("FAIL" if c.critical else "WARN")
            print(f"[{tag}] {c.name:<{w}} {c.detail}")
        hard = [c for c in self.checks if not c.passed and c.critical]
        soft = [c for c in self.checks if not c.passed and not c.critical]
        print("-" * 78)
        print(f"{len(self.checks) - len(hard) - len(soft)} passed, "
              f"{len(hard)} failed, {len(soft)} warnings")
        print("=" * 78)
        return not hard


def random_rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def cycles(trial: dict, seg: str, accel_scale: float = 1.0):
    b = stride_bounds(trial["heel_strikes"], len(trial["time"]), trial["fs"])
    return mean_cycle(trial["accel"][seg] * accel_scale, b), mean_cycle(trial["gyro"][seg], b)


def fit_pair(hf: dict, exo: dict, seg: str):
    ha, hg = cycles(hf, seg, G_TO_MS2)
    ea, eg = cycles(exo, seg)
    return _fit_lag_and_rotation(ha, hg, ea, eg)


# ========================= Checks =========================
def check_algebra(cal: ExoFrame):
    bad = [f"{s}(det={np.linalg.det(cal.rotations[s]):.6f})"
           for s in SEGMENTS if not is_rotation(cal.rotations[s])]
    if bad:
        return False, "not proper rotations: " + ", ".join(bad)
    return True, "orthonormal, right-handed (" + ", ".join(
        f"{s} det={np.linalg.det(cal.rotations[s]):+.9f}" for s in SEGMENTS) + ")"


def check_norms(cal: ExoFrame):
    v = np.random.default_rng(0).normal(size=(500, 3))
    worst = max(float(np.abs(np.linalg.norm(v @ cal.rotations[s].T, axis=1)
                             - np.linalg.norm(v, axis=1)).max()) for s in SEGMENTS)
    return worst < 1e-9, f"max |v| change over 500 random vectors = {worst:.2e}"


def check_round_trip(cal: ExoFrame):
    v = np.random.default_rng(1).normal(size=(500, 3))
    worst = max(float(np.abs(v @ cal.rotations[s].T @ cal.rotations[s] - v).max())
                for s in SEGMENTS)
    return worst < 1e-9, f"max round-trip error HF->exo->HF = {worst:.2e}"


def check_kabsch_exact():
    rng = np.random.default_rng(7)
    worst = 0.0
    for seed in range(20):
        rt = random_rotation(seed)
        p = rng.normal(size=(3, 64))
        worst = max(worst, rotation_angle(kabsch(p, rt @ p)[0], rt))
    return worst < KABSCH_EXACT_TOL_DEG, (
        f"max recovery error over 20 random rotations = {worst:.2e} deg "
        f"(tol {KABSCH_EXACT_TOL_DEG:.0e})")


def check_synthetic_recovery(exo: dict):
    """
    The decisive test: can the estimator recover a rotation it was not told?

    A real exo recording is rotated by a known random matrix and relabelled as HF
    data. Gait segmentation, phase alignment, unit handling and Kabsch then all have
    to cooperate to put it back.
    """
    errs = []
    for seed in (11, 23, 42):
        for i, seg in enumerate(SEGMENTS):
            rt = random_rotation(seed + i)
            fake = {**exo,
                    "accel": {s: exo["accel"][s] @ rt / G_TO_MS2 for s in SEGMENTS},
                    "gyro": {s: exo["gyro"][s] @ rt for s in SEGMENTS}}
            _, r, _ = fit_pair(fake, exo, seg)
            errs.append(rotation_angle(r, rt))
    worst = max(errs)
    return worst < SYNTHETIC_TOL_DEG, (
        f"{len(errs)} trials, mean={np.mean(errs):.4f} deg, worst={worst:.4f} deg "
        f"(tol {SYNTHETIC_TOL_DEG} deg)")


def check_units(hf: dict):
    mag = float(np.linalg.norm(hf["accel"]["shank"].mean(axis=0) * G_TO_MS2))
    return 8.5 < mag < 11.0, (
        f"HF shank mean |accel| after g->m/s^2 = {mag:.3f} m/s^2 (expect ~9.81)")


def check_gravity(hf: dict, exo: dict, cal: ExoFrame):
    parts, worst = [], 0.0
    for seg in SEGMENTS:
        gh = cal.rotations[seg] @ (hf["accel"][seg].mean(0) * G_TO_MS2)
        err = _angle_between(gh, exo["accel"][seg].mean(0))
        worst = max(worst, err)
        parts.append(f"{seg}={err:.1f} deg")
    return worst < GRAVITY_TOL_DEG, ", ".join(parts) + f" (tol {GRAVITY_TOL_DEG} deg)"


def check_sagittal_axis(hf: dict, exo: dict, cal: ExoFrame):
    parts, worst = [], 0.0
    for seg in SEGMENTS:
        ah, _ = principal_axis(hf["gyro"][seg])
        ae, _ = principal_axis(exo["gyro"][seg])
        err = _angle_between(cal.rotations[seg] @ ah, ae)
        err = min(err, 180.0 - err)          # an axis, not a direction
        worst = max(worst, err)
        parts.append(f"{seg}={err:.1f} deg")
    return worst < SAGITTAL_TOL_DEG, ", ".join(parts) + f" (tol {SAGITTAL_TOL_DEG} deg)"


def check_sagittal_corr(hf: dict, exo: dict, cal: ExoFrame):
    """The sagittal channel is what the model leans on, so it gets its own check."""
    parts, worst = [], 1.0
    for seg in SEGMENTS:
        ha, hg = cycles(hf, seg, G_TO_MS2)
        ea, eg = cycles(exo, seg)
        lag, _, _ = _fit_lag_and_rotation(ha, hg, ea, eg)
        rot = (cal.rotations[seg] @ hg.T).T
        r = float(np.corrcoef(rot[:, 2], np.roll(eg, lag, axis=0)[:, 2])[0, 1])
        worst = min(worst, r)
        parts.append(f"{seg} gyro_z r={r:+.3f}")
    return worst >= MIN_SAGITTAL_CORR, ", ".join(parts) + f" (min {MIN_SAGITTAL_CORR:+.2f})"


def check_intersegment_sign(hf: dict, exo: dict, cal: ExoFrame):
    """
    The foot and shank rotations must be mutually consistent, not just individually
    plausible.

    Each segment is fitted independently, so nothing forces their signs to agree -
    and a per-segment sign error is invisible in any single-segment check. The
    frame-independent invariant is how the two segments' sagittal gyros relate to
    each other: whatever that correlation is on the exo, the transformed HF data
    must reproduce its sign.

    Worth reading the numbers rather than just the verdict. In this hardware the
    Georgia Tech foot and shank IMUs are ANTI-correlated (about -0.89) while the
    exo's two are correlated (about +0.97), because the GT foot IMU is mounted
    opposite its shank IMU and the exo's two are mounted alike. So the transform is
    *supposed* to flip one segment and not the other: HF +Y lands on exo -Z for the
    foot and +Z for the shank. That asymmetry is the fix, not a bug.
    """
    b = stride_bounds(exo["heel_strikes"], len(exo["time"]), exo["fs"])
    ex_f = mean_cycle(exo["gyro"]["foot"], b)[:, 2]
    ex_s = mean_cycle(exo["gyro"]["shank"], b)[:, 2]
    r_exo = float(np.corrcoef(ex_f, ex_s)[0, 1])

    hb = stride_bounds(hf["heel_strikes"], len(hf["time"]), hf["fs"])
    hf_f = mean_cycle(hf["gyro"]["foot"], hb)
    hf_s = mean_cycle(hf["gyro"]["shank"], hb)
    r_raw = float(np.corrcoef(hf_f[:, 1], hf_s[:, 1])[0, 1])
    tf = (cal.rotations["foot"] @ hf_f.T).T
    ts = (cal.rotations["shank"] @ hf_s.T).T
    r_fixed = float(np.corrcoef(tf[:, 2], ts[:, 2])[0, 1])

    ok = np.sign(r_fixed) == np.sign(r_exo) and abs(r_fixed) > 0.5
    return ok, (f"exo foot/shank gyro_z r={r_exo:+.3f}; HF before transform "
                f"r={r_raw:+.3f}, after r={r_fixed:+.3f} (sign must match the exo)")


def check_not_antiphase(hf: dict, exo: dict):
    """
    Guards against a left/right leg mix-up, which shows up as a half-stride offset.

    A smaller offset is just the two heel-strike detectors triggering at different
    points, which the joint lag search absorbs.
    """
    parts, ok = [], True
    for seg in SEGMENTS:
        ha, hg = cycles(hf, seg, G_TO_MS2)
        ea, eg = cycles(exo, seg)
        lag, _, _ = _fit_lag_and_rotation(ha, hg, ea, eg)
        frac = abs(lag) / len(hg)
        ok &= abs(frac - 0.5) > ANTIPHASE_BAND
        parts.append(f"{seg} offset={lag / len(hg):+.0%}")
    return ok, ", ".join(parts) + f"; antiphase band is 50%+/-{ANTIPHASE_BAND:.0%}"


def check_raw_data_untouched():
    """
    FSR thresholds must not be able to alter the sensor data itself.

    Thresholding produces contact flags and stride boundaries. It must never reach
    an accelerometer or gyro sample.
    """
    paths = [p for p in sorted(EXO_DIR.glob("data_collection_*.csv"))
             if not any(e in p.name for e in EXCLUDE)]
    worst, checked = 0.0, 0
    for p in paths:
        base = read_exo_log(p, fsr_mode="fixed")
        for frac in (0.3, 0.5, 0.7):
            log = read_exo_log(p, fsr_mode="adaptive", adaptive_fraction=frac)
            for s in SEGMENTS:
                worst = max(worst,
                            float(np.abs(log["accel"][s] - base["accel"][s]).max()),
                            float(np.abs(log["gyro"][s] - base["gyro"][s]).max()))
            checked += 1
    return worst == 0.0, (f"raw accel/gyro identical across {checked} threshold "
                          f"settings on {len(paths)} recordings (max delta {worst:.1e})")


def check_calibration_threshold_stability():
    """
    The calibration you actually deploy must not depend on the FSR threshold.

    Thresholds do reach the fit indirectly, by deciding which strides get averaged.
    This compares the fully-fitted calibration - averaged over every recording pair,
    which is what `ExoFrame.fit` returns - between the two contact modes.

    Single-pair fits are noisier than this and deliberately not what is asserted on:
    a 10 s recording yields only 6-7 strides, so a threshold that gains or loses one
    stride visibly moves that pair's mean gait cycle. See the informational check
    below for those numbers.
    """
    fixed = ExoFrame.fit(HF_ROOT, EXO_DIR, fsr_mode="fixed", exclude=EXCLUDE,
                         verbose=False)
    adaptive = ExoFrame.fit(HF_ROOT, EXO_DIR, fsr_mode="adaptive", exclude=EXCLUDE,
                            verbose=False)
    drift = {s: rotation_angle(fixed.rotations[s], adaptive.rotations[s])
             for s in SEGMENTS}
    worst = max(drift.values())
    sign_ok = fixed.encoder_sign == adaptive.encoder_sign
    detail = (", ".join(f"{s}={d:.2f} deg" for s, d in drift.items()) +
              f"; encoder sign {'agrees' if sign_ok else 'DISAGREES'} "
              f"(tol {THRESHOLD_ROTATION_TOL_DEG} deg)")
    return (worst < THRESHOLD_ROTATION_TOL_DEG and sign_ok), detail


def check_single_pair_stability(hf: dict):
    """
    Informational: how much does one recording's fit move with the threshold?

    Reported with stride counts because that is the whole story - a recording whose
    stride count is stable across thresholds barely moves, and one that flickers
    between 6 and 7 strides moves several degrees. It is a small-sample effect of
    10 s recordings, not threshold contamination, and averaging over pairs removes
    most of it. Longer recordings would shrink it directly.
    """
    parts, worst = [], 0.0
    for p in sorted(EXO_DIR.glob("data_collection_*.csv")):
        if any(e in p.name for e in EXCLUDE):
            continue
        base = read_exo_log(p, fsr_mode="fixed")
        if base["fs"] < 150:
            continue
        counts = {len(stride_bounds(base["heel_strikes"], len(base["time"]), base["fs"]))}
        ref = {s: fit_pair(hf, base, s)[1] for s in SEGMENTS}
        local = 0.0
        for frac in (0.3, 0.4, 0.5, 0.6):
            log = read_exo_log(p, fsr_mode="adaptive", adaptive_fraction=frac)
            counts.add(len(stride_bounds(log["heel_strikes"], len(log["time"]), log["fs"])))
            for s in SEGMENTS:
                local = max(local, rotation_angle(fit_pair(hf, log, s)[1], ref[s]))
        worst = max(worst, local)
        parts.append(f"{p.name[-22:-4]}: {local:.2f} deg (strides {sorted(counts)})")
    return worst < THRESHOLD_ROTATION_TOL_DEG, "; ".join(parts)


def check_outlier_rejection():
    """A single bad fit must not be able to drag the averaged calibration."""
    base = random_rotation(3)
    bad = base @ np.diag([1.0, -1.0, -1.0])          # 180 deg away
    mean, rejected, spread = average_rotations([base] * 4 + [bad])
    err = rotation_angle(mean, base)
    return (rejected == [4] and err < KABSCH_EXACT_TOL_DEG), (
        f"rejected index {rejected}, mean is {err:.2e} deg from the true consensus, "
        f"spread {spread:.2e} deg")


def check_transform_output(hf: dict, cal: ExoFrame):
    x, y = cal.transform_hf(hf)
    problems = []
    if x.shape != (len(hf["time"]), N_FEATURES):
        problems.append(f"X shape {x.shape}")
    if not np.isfinite(x).all():
        problems.append("non-finite values in X")
    if y is None or not np.isfinite(y).all():
        problems.append("bad target")
    if problems:
        return False, "; ".join(problems)
    return True, (f"X{x.shape} finite, y range [{y.min():.2f}, {y.max():.2f}] N*m/kg")


def check_magnitudes(hf: dict, cal: ExoFrame):
    """Rotating must not change how big anything is."""
    x, _ = cal.transform_hf(hf)
    worst = 0.0
    for i, seg in enumerate(SEGMENTS):
        before = np.linalg.norm(hf["gyro"][seg], axis=1)
        after = np.linalg.norm(x[:, 3 + 6 * i:6 + 6 * i], axis=1)
        worst = max(worst, float(np.abs(before - after).max()))
    return worst < 1e-9, f"max gyro magnitude change through the transform = {worst:.2e} rad/s"


def check_train_deploy_parity(hf: dict, cal: ExoFrame, exo: dict):
    """
    Training rows and deployment rows must mean the same thing, column for column.

    Feeding the exo's own samples through `features_batch` and the HF data through
    `transform_hf` has to produce the same layout, or the model meets different
    inputs in training and on the robot - the whole failure this file exists to stop.
    """
    x_train, _ = cal.transform_hf(hf)
    cal.set_encoder_zero(exo["encoder_zero_deg"])
    x_deploy = cal.features_batch(
        exo["accel"]["foot"], exo["gyro"]["foot"],
        exo["accel"]["shank"], exo["gyro"]["shank"],
        exo["encoder_raw_deg"],
        np.where(exo["toe_contact"] > 0, cal.toe_threshold + 1, 0.0),
        np.where(exo["heel_contact"] > 0, cal.heel_threshold + 1, 0.0))
    if x_train.shape[1] != x_deploy.shape[1] != N_FEATURES:
        return False, f"width mismatch: train {x_train.shape[1]} vs deploy {x_deploy.shape[1]}"
    if not np.isfinite(x_deploy).all():
        return False, "non-finite values in the deployment matrix"
    return True, (f"both {N_FEATURES} columns wide and finite "
                  f"(train {x_train.shape[0]} rows, deploy {x_deploy.shape[0]} rows)")


def check_single_sample_matches_batch(cal: ExoFrame, exo: dict):
    """The hot path and the vectorised path must agree exactly."""
    cal.set_encoder_zero(exo["encoder_zero_deg"])
    n = 500
    fa, fg = exo["accel"]["foot"][:n], exo["gyro"]["foot"][:n]
    sa, sg = exo["accel"]["shank"][:n], exo["gyro"]["shank"][:n]
    enc = exo["encoder_raw_deg"][:n]
    toe = np.full(n, 25000.0)
    heel = np.full(n, 15000.0)

    batch = cal.features_batch(fa, fg, sa, sg, enc, toe, heel)
    buf = np.empty(N_FEATURES)
    worst = 0.0
    for i in range(n):
        one = cal.features(fa[i], fg[i], sa[i], sg[i], enc[i], toe[i], heel[i], out=buf)
        worst = max(worst, float(np.abs(one - batch[i]).max()))
    return worst == 0.0, f"max |features - features_batch| over {n} samples = {worst:.2e}"


def check_encoder_zero_recovery():
    """
    The per-run encoder zero must be recoverable from each log's startup window.

    The exo is re-zeroed every run, so the zero belongs to the file. At the head of
    each log the encoder is already reporting while the IMUs have not yet produced a
    packet; those rows are the reference pose.
    """
    rows = []
    for p in sorted(EXO_DIR.glob("data_collection_*.csv")):
        log = read_exo_log(p, fsr_mode="fixed")
        rows.append((p.name, log["encoder_zero_deg"], log["encoder_zero_n"],
                     log["encoder_zero_std"]))
    if not rows:
        return False, "no exo recordings found"

    weak = [r for r in rows if r[2] < 2 or r[3] > 1.0]
    detail = ", ".join(f"{n[-22:-4]}={z:.1f}(n={c})" for n, z, c, _ in rows)
    return not weak, detail + ("" if not weak else
                               f"; weak windows: {[w[0][-22:-4] for w in weak]}")


def check_latency(cal: ExoFrame):
    """Deployment must be fast enough to sit inside the 200 Hz control loop."""
    import time

    cal.set_encoder_zero(80.0)
    fa = np.array([10.9, -2.5, -1.1]); fg = np.array([0.01, 0.05, -0.14])
    sa = np.array([9.3, 1.9, -1.5]);   sg = np.array([-0.05, 0.01, -0.11])
    buf = np.empty(N_FEATURES)

    for _ in range(2000):
        cal.features(fa, fg, sa, sg, 81.3, 900.0, 12000.0, out=buf)
    n = 100_000
    t0 = time.perf_counter()
    for _ in range(n):
        cal.features(fa, fg, sa, sg, 81.3, 900.0, 12000.0, out=buf)
    us = (time.perf_counter() - t0) / n * 1e6
    budget_us = 1e6 / 200.0
    return us < MAX_LATENCY_US, (
        f"{us:.2f} us/sample ({us / budget_us:.3%} of the 200 Hz budget, "
        f"tol {MAX_LATENCY_US} us)")


# ========================= Entry point =========================
def main() -> int:
    print("=" * 78)
    print("exo_frame VALIDATION")
    print("=" * 78)

    hf = read_hf_trial(HF_ROOT, SUBJECT, TRIAL)
    paths = [p for p in sorted(EXO_DIR.glob("data_collection_*.csv"))
             if not any(e in p.name for e in EXCLUDE)]
    exos = [read_exo_log(p, fsr_mode="fixed") for p in paths]
    exos = [e for e in exos if e["fs"] >= 150
            and len(stride_bounds(e["heel_strikes"], len(e["time"]), e["fs"])) >= 3]
    if not exos:
        print(f"No usable exo recording in {EXO_DIR}.")
        return 1
    exo = max(exos, key=lambda e: len(stride_bounds(e["heel_strikes"],
                                                    len(e["time"]), e["fs"])))

    print(f"\nHF  : {hf['name']}  fs={hf['fs']:.0f} Hz  "
          f"strides={len(stride_bounds(hf['heel_strikes'], len(hf['time']), hf['fs']))}  "
          f"mass={hf['mass_kg']} kg")
    print(f"EXO : {exo['name']}  fs={exo['fs']:.0f} Hz  "
          f"strides={len(stride_bounds(exo['heel_strikes'], len(exo['time']), exo['fs']))}  "
          f"encoder_zero={exo['encoder_zero_deg']:.2f} deg "
          f"(n={exo['encoder_zero_n']}, sd={exo['encoder_zero_std']:.3f})")

    print("\nFitting calibration:")
    cal = ExoFrame.fit(HF_ROOT, EXO_DIR, exclude=EXCLUDE)
    for seg in SEGMENTS:
        print(f"\n  R[{seg}] =")
        for row in cal.rotations[seg]:
            print("      [" + "  ".join(f"{v:+.4f}" for v in row) + "]")

    s = Suite()
    s.run("rotation is orthonormal, det=+1", lambda: check_algebra(cal))
    s.run("rotation preserves vector norms", lambda: check_norms(cal))
    s.run("HF->exo->HF round trip is identity", lambda: check_round_trip(cal))
    s.run("Kabsch solver is exact on clean data", check_kabsch_exact)
    s.run("estimator recovers a known rotation", lambda: check_synthetic_recovery(exo))
    s.run("accel unit conversion lands on 1 g", lambda: check_units(hf))
    s.run("gravity direction agrees after transform", lambda: check_gravity(hf, exo, cal))
    s.run("sagittal axis agrees after transform",
          lambda: check_sagittal_axis(hf, exo, cal))
    s.run("sagittal waveform correlates", lambda: check_sagittal_corr(hf, exo, cal),
          critical=False)
    s.run("foot/shank rotations are mutually consistent",
          lambda: check_intersegment_sign(hf, exo, cal))
    s.run("recordings are not antiphase (same leg)", lambda: check_not_antiphase(hf, exo))
    s.run("FSR thresholds never touch raw samples", check_raw_data_untouched)
    s.run("calibration is stable across FSR modes", check_calibration_threshold_stability)
    s.run("single-pair fits are stable", lambda: check_single_pair_stability(hf),
          critical=False)
    s.run("outlier fits are rejected, not averaged", check_outlier_rejection)
    s.run("per-run encoder zero is recoverable", check_encoder_zero_recovery,
          critical=False)
    s.run("transform preserves magnitudes", lambda: check_magnitudes(hf, cal))
    s.run("transformed output is finite and complete",
          lambda: check_transform_output(hf, cal))
    s.run("train/deploy feature parity", lambda: check_train_deploy_parity(hf, cal, exo))
    s.run("features == features_batch", lambda: check_single_sample_matches_batch(cal, exo))
    s.run("deployment latency fits the control loop", lambda: check_latency(cal))

    return 0 if s.report() else 1


if __name__ == "__main__":
    sys.exit(main())
