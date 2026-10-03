#!/usr/bin/env python3
"""
test_calibration.py

Offline validation of the guided calibration in exo_frame.py. Run standalone:

    python test_calibration.py

No hardware is needed. Every phase is synthesised with KNOWN values - encoder zero,
sign, ratio, range of motion, and the sensor rotation itself - and pushed through
the real analysis path. The test passes only if those values come back. That makes
it the same kind of check as the rotation recovery test in test_exo_frame.py: the
only sort where the right answer is known in advance.

Three runs:

  1. Ground truth. The walking phase is built by taking real Georgia Tech AB06
     walking and rotating it by a known random matrix into a synthetic exo frame,
     so the calibration has to recover that rotation as well as the encoder terms.
  2. Negative control on the holds. Swapping dorsiflexion and plantarflexion must
     invert the recovered sign, or the sign is not being measured at all.
  3. Negative control on bandwidth. Replaying a real 25 Hz-report exo recording
     must fail the IMU rate check and must NOT emit a transform module.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow this script to be run directly from any working directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import exo_frame as E

HF_ROOT = ROOT / "mrsd-exo-ankle"
EXO_CSV = ROOT / "Data collection/data/data_collection_20260404_215743_tightshoes_200Hz_2kmph.csv"
OUT_DIR = ROOT / "calibration/_test_output"
SUBJECT, TRIAL = "AB06", "treadmill_01_01"

# Ground truth injected into the synthetic phases.
TRUE_ZERO = 81.4321
TRUE_SIGN = +1
TRUE_RATIO = 1.0
TRUE_DORSI_ROM = 11.0
TRUE_PLANTAR_ROM = 24.0
ROT_SEED = 17

ZERO_TOL_DEG = 0.30
RATIO_TOL = 0.06
ROTATION_TOL_DEG = 5.0   # sized for AB06 inter-trial spread, not solver error

FS = 200.0
COUNTS_LOW, COUNTS_HIGH = 1500.0, 24000.0


def _noise(shape, sd, seed):
    return np.random.default_rng(seed).normal(0.0, sd, shape)


def random_rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _phase(n, foot_accel, shank_accel, foot_gyro, shank_gyro,
           encoder, toe=COUNTS_LOW, heel=COUNTS_LOW):
    def col(v):
        a = np.asarray(v, dtype=np.float64)
        return np.tile(a, (n, 1)) if a.ndim == 1 else a

    def scal(v):
        return np.full(n, float(v)) if np.isscalar(v) else np.asarray(v, np.float64)

    return {"time": np.arange(n) / FS,
            "foot_accel": col(foot_accel), "foot_gyro": col(foot_gyro),
            "shank_accel": col(shank_accel), "shank_gyro": col(shank_gyro),
            "encoder": scal(encoder), "toe": scal(toe), "heel": scal(heel)}


def _raw_from_joint(joint_deg):
    """Invert ankle = sign * ratio * (raw - zero)."""
    return TRUE_ZERO + np.asarray(joint_deg, np.float64) / (TRUE_SIGN * TRUE_RATIO)


def build_static_phases(rot_foot, rot_shank):
    """Standing, the three holds, and the free sweep, in the synthetic exo frame."""
    # Gravity is ONE physical direction, so both segments must see the same base
    # vector rotated into their own frames. Giving each its own base vector would
    # make the synthetic gyro and the synthetic gravity imply different values of
    # R_foot<-shank, and the hip-swing analysis would be scored against data that
    # contradicts itself.
    g_base = np.array([10.888, -2.515, -1.132])
    g_base = g_base / np.linalg.norm(g_base) * 9.80665
    g_foot = rot_foot @ g_base
    g_shank = rot_shank @ g_base

    # The ankle axis, expressed in the synthetic foot frame.
    axis = rot_foot @ np.array([0.018, -0.180, 0.984])
    axis = axis / np.linalg.norm(axis)

    phases = {}
    n = int(10 * FS)
    phases["standing"] = _phase(
        n, g_foot + _noise((n, 3), 0.05, 1), g_shank + _noise((n, 3), 0.05, 2),
        _noise((n, 3), 0.02, 3), _noise((n, 3), 0.02, 4),
        TRUE_ZERO + _noise(n, 0.05, 5))

    def hold(joint_deg, seed):
        m = int(5 * FS)
        return _phase(m, g_foot + _noise((m, 3), 0.05, seed),
                      g_shank + _noise((m, 3), 0.05, seed + 1),
                      _noise((m, 3), 0.02, seed + 2), _noise((m, 3), 0.02, seed + 3),
                      float(_raw_from_joint(joint_deg)) + _noise(m, 0.05, seed + 4))

    phases["neutral"] = hold(0.0, 10)
    phases["dorsi"] = hold(+TRUE_DORSI_ROM, 20)
    phases["plantar"] = hold(-TRUE_PLANTAR_ROM, 30)

    # Free sweep: a slow sine in joint angle. The encoder follows it through the true
    # sign and ratio; the foot gyro is its derivative about the ankle axis.
    m = int(10 * FS)
    t = np.arange(m) / FS
    amp, freq = 12.0, 0.5
    theta = amp * np.sin(2 * np.pi * freq * t)
    dtheta = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
    # AS5600 is 12-bit, so quantise to exercise the noisy-encoder path.
    step = 360.0 / 4096.0
    raw = np.round(_raw_from_joint(theta) / step) * step
    gyro = np.radians(dtheta)[:, None] * axis[None, :] + _noise((m, 3), 0.01, 40)

    phases["sweep"] = _phase(m, g_foot + _noise((m, 3), 0.08, 41),
                             g_shank + _noise((m, 3), 0.05, 42),
                             gyro, _noise((m, 3), 0.02, 43), raw)

    # Hip swing: the whole leg turns about one axis, so foot and shank share a
    # single physical angular velocity. Expressed in their own frames that is
    # R_foot @ w and R_shank @ w, which is exactly what the analysis must undo.
    m = int(10 * FS)
    t = np.arange(m) / FS
    world_axis = np.array([0.0, 1.0, 0.0])          # medio-lateral in world terms
    rate = np.radians(45.0) * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t)
    w_world = rate[:, None] * world_axis[None, :]
    w_foot = w_world @ rot_foot.T + _noise((m, 3), 0.01, 60)
    w_shank = w_world @ rot_shank.T + _noise((m, 3), 0.01, 61)
    phases["hipswing"] = _phase(
        m, g_foot + _noise((m, 3), 0.10, 62), g_shank + _noise((m, 3), 0.10, 63),
        w_foot, w_shank, TRUE_ZERO + _noise(m, 0.2, 64))
    return phases


def build_walking_from_gt(rot_foot, rot_shank):
    """
    A synthetic exo walking phase built from real Georgia Tech walking.

    Rotating genuine 200 Hz data by a known matrix gives a phase with real gait
    content, real bandwidth, and a rotation whose true value is known, so the
    calibration's rotation fit can be scored rather than merely inspected.
    """
    trial = E.read_hf_trial(HF_ROOT, SUBJECT, TRIAL)
    n = len(trial["time"])

    foot_a = (trial["accel"]["foot"] * E.G_TO_MS2) @ rot_foot.T
    foot_g = trial["gyro"]["foot"] @ rot_foot.T
    shank_a = (trial["accel"]["shank"] * E.G_TO_MS2) @ rot_shank.T
    shank_g = trial["gyro"]["shank"] @ rot_shank.T

    step = 360.0 / 4096.0
    enc = np.round(_raw_from_joint(trial["ankle_angle_deg"]) / step) * step

    heel = COUNTS_LOW + (COUNTS_HIGH - COUNTS_LOW) * trial["heel_contact"]
    toe = COUNTS_LOW + (COUNTS_HIGH - COUNTS_LOW) * trial["toe_contact"]
    heel = heel + _noise(n, 120.0, 51)
    toe = toe + _noise(n, 120.0, 52)

    return {"time": trial["time"] - trial["time"][0],
            "foot_accel": foot_a, "foot_gyro": foot_g,
            "shank_accel": shank_a, "shank_gyro": shank_g,
            "encoder": enc, "toe": toe, "heel": heel}


def build_walking_replay(target_s=70.0):
    """Replay a real exo recording, tiled. Used only as the bandwidth negative control."""
    raw = pd.read_csv(EXO_CSV)
    df = raw.interpolate(limit_direction="both")
    t = df["timestamp_s"].to_numpy(np.float64)
    reps = int(np.ceil(target_s / (t[-1] - t[0])))

    def tile(a):
        return np.tile(a, (reps, 1)) if a.ndim == 2 else np.tile(a, reps)

    enc = tile(df["ankle_encoder_deg"].to_numpy(np.float64))
    foot_a = tile(df[[f"foot_a{c}" for c in "xyz"]].to_numpy(np.float64))
    n = len(foot_a)
    return {"time": np.arange(n) / FS,
            "foot_accel": foot_a,
            "foot_gyro": tile(df[[f"foot_g{c}" for c in "xyz"]].to_numpy(np.float64)),
            "shank_accel": tile(df[[f"shank_a{c}" for c in "xyz"]].to_numpy(np.float64)),
            "shank_gyro": tile(df[[f"shank_g{c}" for c in "xyz"]].to_numpy(np.float64)),
            "encoder": enc - np.median(enc) + TRUE_ZERO,
            "toe": tile(df["toe_fsr_raw"].to_numpy(np.float64)),
            "heel": tile(df["heel_fsr_raw"].to_numpy(np.float64))}


def report(label, ok, detail, failures):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<46} {detail}")
    if not ok:
        failures.append(label)


def main() -> int:
    print("=" * 92)
    print("CALIBRATION VALIDATION")
    print("=" * 92)

    for p in (HF_ROOT, EXO_CSV):
        if not p.exists():
            print(f"Missing required input: {p}")
            return 1

    failures: list = []
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    rot_foot = random_rotation(ROT_SEED)
    rot_shank = random_rotation(ROT_SEED + 1)

    phases = build_static_phases(rot_foot, rot_shank)
    phases["walking"] = build_walking_from_gt(rot_foot, rot_shank)

    print(f"\nInjected: zero={TRUE_ZERO} deg  sign={TRUE_SIGN:+d}  ratio={TRUE_RATIO}"
          f"  ROM={TRUE_DORSI_ROM}/{TRUE_PLANTAR_ROM} deg")
    print(f"Injected rotations: random seeds {ROT_SEED} (foot), {ROT_SEED + 1} (shank)")

    # ---------------- run 1: ground truth ----------------
    result = E.run_calibration(HF_ROOT, out_dir=OUT_DIR, reference_subject=SUBJECT,
                               interactive=False,
                               recorder=lambda spec: phases[spec.key])

    print("\n" + "=" * 92)
    print("RUN 1 - GROUND TRUTH RECOVERY")
    print("=" * 92)

    report("calibration reported success", result.ok,
           f"{sum(1 for c in result.checks if c.passed)}/{len(result.checks)} checks passed",
           failures)

    d = abs(result.encoder_zero_deg - TRUE_ZERO)
    report("encoder zero", d <= ZERO_TOL_DEG,
           f"got {result.encoder_zero_deg:.4f}, true {TRUE_ZERO:.4f}, err {d:.4f} deg",
           failures)
    report("encoder sign", result.encoder_sign == TRUE_SIGN,
           f"got {result.encoder_sign:+d}, true {TRUE_SIGN:+d}", failures)
    d = abs(result.encoder_ratio - TRUE_RATIO)
    report("encoder ratio", d <= RATIO_TOL,
           f"got {result.encoder_ratio:.4f}, true {TRUE_RATIO:.4f}, err {d:.4f}",
           failures)

    sweep = result.metrics["sweep"]
    for name, got, true in (("dorsiflexion ROM", sweep["dorsiflexion_rom_deg"],
                             TRUE_DORSI_ROM),
                            ("plantarflexion ROM", sweep["plantarflexion_rom_deg"],
                             TRUE_PLANTAR_ROM)):
        report(name, abs(got - true) <= 0.5,
               f"got {got:.2f}, true {true:.2f} deg", failures)

    # Two separate questions. Against the trial the synthetic walking was built from,
    # the fit must be exact - that tests the machinery. Against the average over three
    # trials it only has to be close, because AB06's own trials differ from each
    # other: treadmill_03_01 in particular walks with a different phase and sits about
    # 4 deg from the others. That spread is biology, not solver error, so it gets a
    # tolerance sized for it rather than a tight one.
    walk = phases["walking"]
    heel = walk["heel"]
    lo, hi = np.percentile(heel, [10.0, 90.0])
    hc = (heel > lo + 0.5 * (hi - lo)).astype(int)
    hs = np.where(np.diff(hc) == 1)[0] + 1
    bounds = E.stride_bounds(hs, len(heel), FS)
    src = E.read_hf_trial(HF_ROOT, SUBJECT, TRIAL)
    hb = E.stride_bounds(src["heel_strikes"], len(src["time"]), src["fs"])

    for seg, true_r in (("foot", rot_foot), ("shank", rot_shank)):
        _, r_exact, resid = E._fit_lag_and_rotation(
            E.mean_cycle(src["accel"][seg] * E.G_TO_MS2, hb),
            E.mean_cycle(src["gyro"][seg], hb),
            E.mean_cycle(walk[f"{seg}_accel"], bounds),
            E.mean_cycle(walk[f"{seg}_gyro"], bounds))
        err = E.rotation_angle(r_exact, true_r)
        report(f"{seg} rotation exact vs source trial", err <= 1e-3,
               f"{err:.2e} deg, residual {resid:.2e}", failures)

        err = E.rotation_angle(result.rotations[seg], true_r)
        report(f"{seg} rotation averaged over 3 trials", err <= ROTATION_TOL_DEG,
               f"{err:.3f} deg from injected (tol {ROTATION_TOL_DEG}, inter-trial "
               f"spread {result.metrics['walking']['rotation_spread_deg'][seg]:.2f})",
               failures)

    # The hip swing must recover the relationship between the two injected exo
    # frames: with the leg rigid, R_foot<-shank is exactly rot_foot @ rot_shank^T.
    hip = result.metrics["hipswing"]
    true_fs = rot_foot @ rot_shank.T
    err = E.rotation_angle(np.array(hip["r_foot_from_shank"]), true_fs)
    report("hip swing recovers R_foot<-shank", err <= 5.0,
           f"{err:.3f} deg from the injected relationship (tol 5.0)", failures)
    report("shank sagittal axis is measured",
           hip["shank_sagittal_var_ratio"] >= 0.90,
           f"axis explains {hip['shank_sagittal_var_ratio']:.1%} of shank swing "
           f"variance", failures)
    report("rigid-body fit is tight", hip["rigid_residual_fraction"] <= 0.05,
           f"{hip['rigid_residual_fraction']:.2%} residual", failures)

    # ---------------- run 2: swapped holds ----------------
    print("\n" + "=" * 92)
    print("RUN 2 - NEGATIVE CONTROL: dorsiflexion and plantarflexion holds swapped")
    print("=" * 92)
    m, _ = E.analyse_sweep(phases["neutral"], phases["plantar"], phases["dorsi"],
                           phases["sweep"], TRUE_ZERO)
    report("swapped holds invert the sign", m["encoder_sign"] == -TRUE_SIGN,
           f"got {m['encoder_sign']:+d}, expected {-TRUE_SIGN:+d}", failures)
    report("swapped holds keep the ratio", abs(m["encoder_ratio"] - TRUE_RATIO) <= RATIO_TOL,
           f"ratio {m['encoder_ratio']:.4f} is unchanged, as it must be", failures)

    # ---------------- run 3: low-bandwidth replay ----------------
    print("\n" + "=" * 92)
    print("RUN 3 - NEGATIVE CONTROL: replay of a real 25 Hz-report recording")
    print("=" * 92)
    low = dict(phases)
    low["walking"] = build_walking_replay()
    low_dir = OUT_DIR / "lowrate"
    lo = E.run_calibration(HF_ROOT, out_dir=low_dir, reference_subject=SUBJECT,
                           interactive=False, recorder=lambda spec: low[spec.key])
    rate_check = next((c for c in lo.checks if "full IMU rate" in c.name), None)
    report("low-rate walking is rejected", rate_check is not None and not rate_check.passed,
           rate_check.detail if rate_check else "rate check missing", failures)
    report("no module emitted on failure", not lo.ok
           and not list(low_dir.glob("exo_transform_2*.py")),
           "calibration failed and wrote no transform, as intended", failures)

    # ---------------- generated module ----------------
    print("\n" + "=" * 92)
    print("GENERATED TRANSFORM MODULE")
    print("=" * 92)
    mods = sorted(p for p in OUT_DIR.glob("exo_transform_2*.py"))
    if not mods:
        report("transform module written", False, "none found", failures)
    else:
        path = mods[-1]
        print(f"  {path.name}  ({path.stat().st_size / 1024:.1f} KB)")
        sys.path.insert(0, str(OUT_DIR.resolve()))
        mod = __import__(path.stem)

        report("features list matches exo_frame",
               list(mod.FEATURES) == list(E.FEATURES),
               f"{len(mod.FEATURES)} columns", failures)
        report("rotations match the calibration",
               all(np.allclose(mod.ROTATIONS[s], result.rotations[s], atol=1e-10)
                   for s in E.SEGMENTS), "bit-identical to 1e-10", failures)
        report("encoder terms match the calibration",
               abs(mod.ENCODER_ZERO_DEG - result.encoder_zero_deg) < 1e-9
               and mod.ENCODER_SIGN == result.encoder_sign
               and abs(mod.ENCODER_RATIO - result.encoder_ratio) < 1e-9,
               f"zero {mod.ENCODER_ZERO_DEG:.4f}, sign {mod.ENCODER_SIGN:+d}, "
               f"ratio {mod.ENCODER_RATIO:.4f}", failures)

        x = mod.features(np.array([10.9, -2.5, -1.1]), np.array([0.01, 0.05, -0.14]),
                         np.array([9.3, 1.9, -1.5]), np.array([-0.05, 0.01, -0.11]),
                         TRUE_ZERO + 5.0, 25000.0, 15000.0)
        report("features() returns a finite vector",
               len(x) == E.N_FEATURES and np.isfinite(x).all(),
               f"{len(x)} values", failures)
        expect = 5.0 * TRUE_SIGN * TRUE_RATIO
        report("ankle angle is correct at zero + 5 deg", abs(x[12] - expect) < 0.05,
               f"got {x[12]:+.4f}, expected {expect:+.4f}", failures)

        X, y = mod.ingest_gt_trial(HF_ROOT, SUBJECT, TRIAL)
        report("ingest_gt_trial produces training data",
               X.shape[1] == E.N_FEATURES and np.isfinite(X).all()
               and y is not None and np.isfinite(y).all(),
               f"X{X.shape}, y in [{y.min():.2f}, {y.max():.2f}] N*m/kg", failures)

        # The generated module must reproduce exo_frame's own transform exactly.
        ref = E.ExoFrame(rotations=result.rotations,
                         encoder_zero_deg=result.encoder_zero_deg,
                         encoder_sign=result.encoder_sign,
                         encoder_ratio=result.encoder_ratio,
                         heel_threshold=result.heel_threshold,
                         toe_threshold=result.toe_threshold)
        Xr, _ = ref.transform_hf_trial(HF_ROOT, SUBJECT, TRIAL)
        err = float(np.abs(X[:, :12] - Xr[:, :12]).max())
        report("generated module agrees with exo_frame", err < 1e-9,
               f"max IMU column delta {err:.2e}", failures)

        # Harness self-consistency: undoing the injected rotation must return exactly
        # the Georgia Tech data the phase was synthesised from. Deliberately uses the
        # true rotation, not the module's. The module carries the average over three
        # trials, which is a few degrees from truth by design, so inverting with it
        # would measure inter-trial spread rather than round-trip correctness - and
        # that spread is already reported above.
        gt = E.read_hf_trial(HF_ROOT, SUBJECT, TRIAL)["gyro"]["foot"]
        back = phases["walking"]["foot_gyro"] @ rot_foot
        report("synthetic walking round-trips exactly",
               float(np.abs(back - gt).max()) < 1e-12,
               f"max gyro delta {float(np.abs(back - gt).max()):.2e} rad/s", failures)

        recon = float(np.abs(phases["walking"]["foot_gyro"]
                             @ np.asarray(mod.ROTATION_FOOT) - gt).max())
        print(f"  [INFO] reconstruction through the averaged rotation is off by "
              f"{recon:.3f} rad/s, the cost of averaging over three trials")

        report("stable-name copy exists", (OUT_DIR / "exo_transform_latest.py").exists(),
               "exo_transform_latest.py", failures)

    print("\n" + "=" * 92)
    if failures:
        print(f"FAILED: {len(failures)} problem(s)")
        for f in failures:
            print(f"    - {f}")
    else:
        print("ALL CHECKS PASSED")
    print("=" * 92)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
