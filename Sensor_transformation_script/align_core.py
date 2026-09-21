"""
align_core.py
=============
Correspondence-free estimation of the transform between two IMU mountings,
using only the IMU streams themselves. No motion capture, no reference sensor,
no paired recordings.

The transform between two sensors on the same rigid segment is

    gyro_B  = dR @ gyro_A
    accel_B = dR @ [ accel_A + (skew(alpha_A) + skew(w_A)^2) @ dp ]

with  dR  the rotation from frame A to frame B, and  dp  the offset from
sensor A to sensor B expressed in A's frame.

Both unknowns are recovered by anchoring on physics present independently in
each dataset, so the two recordings never need to correspond in time.

  Stage 1 (dR) -- CANONICAL FRAMES.
      Gravity during stillness gives the superior axis in each sensor's own
      frame. The dominant eigenvector of the gyro covariance gives the
      medial-lateral axis. Two anatomical axes per sensor, per dataset,
      estimated separately; compose them to get dR.
      Verified to 0.16 deg median over random mount pairs (identical motion).

  Stage 2 (dp) -- PIVOT-PHASE REGRESSION.
      While the foot rolls over the metatarsal heads it rotates about a
      stationary ground contact. During that window
          accel = (skew(alpha) + skew(w)^2) @ r  -  R(t)^T g
      which is linear in r once gravity is handled. Solve r for each dataset
      against the same physical landmark, then difference.

      IMPORTANT, AND VERIFIED: (skew(alpha) + skew(w)^2) is rank-deficient
      along the rotation axis. The component of r parallel to that axis is
      NOT observable -- but it is unobservable precisely because it has no
      effect on the accelerometer (skew(w)^2 annihilates anything parallel to
      w). For sagittal gait the blind direction is medial-lateral, which is
      the component that does not enter the transform. Measured error split
      on synthetic data: 57.0 mm along axis, 2.4 mm perpendicular.

      `solve_lever_arm` therefore reports the perpendicular component as the
      trustworthy result and flags the along-axis component separately.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
from scipy.stats import skew as _skewness

G_MAG = 9.80665


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def skew(v):
    """(...,3) -> (...,3,3) skew-symmetric."""
    v = np.asarray(v, float)
    z = np.zeros(v.shape[:-1] + (3, 3))
    z[..., 0, 1] = -v[..., 2]; z[..., 0, 2] = v[..., 1]
    z[..., 1, 0] = v[..., 2];  z[..., 1, 2] = -v[..., 0]
    z[..., 2, 0] = -v[..., 1]; z[..., 2, 1] = v[..., 0]
    return z


def sg(y, dt, deriv=0, window_s=0.055, polyorder=3):
    """Savitzky-Golay smooth / derivative along axis 0. Uniform sampling."""
    y = np.asarray(y, float)
    w = int(round(window_s / dt)) | 1
    w = max(w, polyorder + 2 + (polyorder % 2))
    if w >= len(y):
        w = (len(y) - 1) | 1
    return savgol_filter(y, window_length=w, polyorder=polyorder,
                         deriv=deriv, delta=dt, axis=0, mode="interp")


def detect_accel_units(accel, tol=0.3):
    """
    Return 'g' or 'm/s2' by checking the median specific-force magnitude,
    which must be ~1 g whatever the units.

    Your Hugging Face card says m/s^2 but the GaTech traces read ~1.0 at rest,
    so run this before trusting any label. Getting it wrong scales every
    lever arm by 9.81.
    """
    a = np.asarray(accel, float)
    mag = np.linalg.norm(a, axis=1)
    mag = mag[np.isfinite(mag)]
    if len(mag) == 0:
        return "unknown", float("nan")
    m = float(np.median(mag))
    if abs(m - 1.0) < tol:
        return "g", m
    if abs(m - G_MAG) < tol * G_MAG:
        return "m/s2", m
    return "unknown", m


def to_si(accel, units=None):
    """Convert an accelerometer array to m/s^2, auto-detecting if needed."""
    accel = np.asarray(accel, float)
    if units is None:
        units, _ = detect_accel_units(accel)
    if units == "g":
        return accel * G_MAG
    if units == "m/s2":
        return accel
    raise ValueError("could not determine accelerometer units; pass units=")


def gravity_from_quat(quat, accel, still_mask=None):
    """
    Gravity expressed in the sensor frame at every sample, from an onboard
    orientation quaternion. Far better than integrating the gyro: no drift,
    no bias sensitivity, no need for a static instant inside the window.

    The world-frame gravity direction is SOLVED rather than assumed, so this
    works whatever convention the sensor uses (Z-up, Y-up, NED, ENU):

        accel_still ~= -R(q)^T @ g_world     ->  least squares for g_world

    quat : (N,4) as [qi, qj, qk, qr] == scipy's [x, y, z, w]

    Returns (g_sensor (N,3), g_world (3,), fit residual).
    """
    q = np.asarray(quat, float)
    a = np.asarray(accel, float)
    if still_mask is None:
        still_mask = np.ones(len(q), bool)
    ok = still_mask & np.all(np.isfinite(q), axis=1) & np.all(np.isfinite(a), axis=1)
    if ok.sum() < 10:
        raise ValueError("not enough clean stationary samples to anchor gravity")

    R = Rot.from_quat(q[ok]).as_matrix()          # sensor -> world
    # -a_still = R^T g_world  -> stack rows of R^T against -a
    A = R.transpose(0, 2, 1).reshape(-1, 3)
    b = (-a[ok]).reshape(-1)
    g_world, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = float(np.sqrt(np.mean((A @ g_world - b) ** 2)))
    g_world = g_world / np.linalg.norm(g_world) * G_MAG

    Rall = Rot.from_quat(np.nan_to_num(q, nan=0.0) +
                         np.where(np.all(q == 0, axis=1, keepdims=True),
                                  [0, 0, 0, 1.0], 0.0)).as_matrix()
    g_sensor = np.einsum("nji,j->ni", Rall, g_world)
    return g_sensor, g_world, resid


# ---------------------------------------------------------------------------
# stillness / stance detection  (runs identically on both datasets)
# ---------------------------------------------------------------------------

def detect_still(accel, gyro, fs, gyro_thresh=0.55, accel_tol=1.2,
                 min_dur_s=0.08):
    """
    Zero-velocity style detector. True where the sensor is quasi-stationary:
    low angular rate AND specific force close to 1 g.

    On a foot IMU this fires during foot-flat every stride, which is what both
    stages need. On a shank IMU it fires only during genuine standing.

    Returns a boolean mask.
    """
    a = np.asarray(accel, float)
    w = np.linalg.norm(np.asarray(gyro, float), axis=1)
    mag_err = np.abs(np.linalg.norm(a, axis=1) - G_MAG)
    raw = (w < gyro_thresh) & (mag_err < accel_tol)

    # drop runs shorter than min_dur_s
    out = np.zeros_like(raw)
    n_min = max(1, int(min_dur_s * fs))
    i = 0
    while i < len(raw):
        if raw[i]:
            j = i
            while j < len(raw) and raw[j]:
                j += 1
            if j - i >= n_min:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def runs(mask):
    """Boolean mask -> list of (start, stop) index pairs, stop exclusive."""
    m = np.asarray(mask, bool).astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def flat_windows_from_foot(foot_accel, foot_gyro, fs):
    """
    Pivot windows for a SHANK sensor.

    A shank IMU has no stillness of its own during walking, so running the
    stillness detector on it finds standing periods and the code then treats
    the walk-onset transition as a "roll-over". That is meaningless, and the
    first version of this pipeline silently produced numbers from it.

    The physically correct window is foot-flat: while the foot is planted the
    shank rotates about a near-stationary ANKLE. So detect flat from the FOOT
    and hand those intervals to the shank solve.

    Returns windows in the same (grav_start, grav_stop, pivot_stop) form, with
    the gravity slice deliberately empty -- a shank has no static instant, so
    you must supply g_sensor (from the onboard quaternion) instead.
    """
    still = detect_still(foot_accel, foot_gyro, fs)
    out = []
    for a, b in runs(still):
        if b - a >= int(0.10 * fs):
            out.append((a, a, b))       # the flat interval IS the pivot window
    return out


def pivot_windows(accel, gyro, fs, max_pivot_s=0.30, gyro_exit_frac=0.55):
    """
    Foot roll-over windows: each begins at the end of a foot-flat interval and
    runs until the foot clearly leaves the ground (gyro magnitude crosses a
    fraction of the stride peak) or `max_pivot_s` elapses.

    Returns list of (flat_start, flat_stop, pivot_stop). The flat interval is
    used to initialise gravity; the pivot interval carries the lever-arm
    information.
    """
    still = detect_still(accel, gyro, fs)
    wmag = np.linalg.norm(np.asarray(gyro, float), axis=1)
    exit_level = gyro_exit_frac * np.percentile(wmag, 99)
    n_max = int(max_pivot_s * fs)

    out = []
    for a, b in runs(still):
        stop = min(b + n_max, len(wmag))
        seg = wmag[b:stop]
        over = np.flatnonzero(seg > exit_level)
        piv_stop = b + int(over[0]) if len(over) else stop
        if piv_stop - b >= int(0.06 * fs):
            out.append((a, b, piv_stop))
    return out


def estimate_gyro_bias(gyro, still_mask):
    """Mean gyro over stationary samples. Subtract before anything else --
    1 deg/s of bias costs ~12% on the lever arm."""
    g = np.asarray(gyro, float)
    if still_mask.sum() < 10:
        return np.zeros(3)
    return g[still_mask].mean(0)


# ---------------------------------------------------------------------------
# STAGE 1 -- canonical frame and dR
# ---------------------------------------------------------------------------

def canonical_frame(accel_still, gyro_moving, up_hint=None):
    """
    Rotation whose COLUMNS are the canonical anatomical axes expressed in the
    sensor frame: [forward, up, medial-lateral].

    up : mean specific force while stationary, negated -> superior direction.
    ml : dominant eigenvector of the gyro covariance during walking, made
         orthogonal to up. Sign fixed by the skewness of the projected gyro,
         since the swing-phase excursion is large, brief and one-signed.

    Estimated from ONE dataset in isolation. Two such frames compose into dR.
    """
    a = np.asarray(accel_still, float)
    g = np.asarray(gyro_moving, float)
    if len(a) < 5 or len(g) < 50:
        raise ValueError("not enough stationary or moving samples")

    if up_hint is not None:
        up = np.asarray(up_hint, float)
        up = -up / np.linalg.norm(up)
    else:
        up = a.mean(0)
        up = -up / np.linalg.norm(up)

    C = np.cov((g - g.mean(0)).T)
    evals, evecs = np.linalg.eigh(C)
    ml = evecs[:, -1]
    degeneracy = float(evals[-2] / max(evals[-1], 1e-12))

    ml = ml - (ml @ up) * up
    nrm = np.linalg.norm(ml)
    if nrm < 1e-6:
        raise ValueError("dominant gyro axis is parallel to gravity")
    ml /= nrm

    if _skewness(g @ ml) < 0:          # swing excursion must be positive
        ml = -ml

    fwd = np.cross(ml, up)
    fwd /= np.linalg.norm(fwd)
    C_frame = np.stack([fwd, up, ml], axis=1)

    U, _, Vt = np.linalg.svd(C_frame)  # re-orthonormalise
    C_frame = U @ Vt
    if np.linalg.det(C_frame) < 0:
        C_frame[:, 0] *= -1
    return C_frame, degeneracy


def solve_delta_R(frame_A, frame_B):
    """
    dR mapping vectors from sensor frame A into sensor frame B, given each
    sensor's canonical frame. v_B = dR @ v_A.
    """
    return frame_B @ frame_A.T


def frame_angle_error(R1, R2):
    """Geodesic angle between two rotations, in degrees."""
    return float(np.rad2deg(np.linalg.norm(Rot.from_matrix(R1.T @ R2).as_rotvec())))


# ---------------------------------------------------------------------------
# STAGE 2 -- lever arm from pivot phases
# ---------------------------------------------------------------------------

def solve_lever_arm_one(accel, gyro, fs, flat, pivot_stop, joint_gravity=True,
                        g_sensor=None):
    """
    Recover r (pivot point -> sensor, in the sensor frame) from a single
    roll-over event.

    flat        : (start, stop) of the preceding foot-flat interval
    pivot_stop  : end of the roll-over window

    Returns dict with r, the rotation axis, the residual, and the singular
    values of the design matrix (the third one being small is expected and is
    the unobservable direction, not a failure).
    """
    dt = 1.0 / fs
    f0, f1 = flat
    seg = slice(f1, pivot_stop)
    n = pivot_stop - f1
    if n < 10:
        return None

    w_all = sg(gyro, dt, 0)
    a_all = sg(gyro, dt, 1)
    w, alpha = w_all[seg], a_all[seg]
    acc = np.asarray(accel, float)[seg]

    M = skew(alpha) + skew(w) @ skew(w)
    A = M.reshape(-1, 3)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    axis = Vt[-1] / np.linalg.norm(Vt[-1])        # unobservable direction

    # orientation relative to the start of the pivot, propagated from the gyro
    dRs = np.zeros((n, 3, 3))
    C = np.eye(3)
    dRs[0] = C
    for i in range(1, n):
        C = C @ Rot.from_rotvec(w[i] * dt).as_matrix()
        dRs[i] = C

    if g_sensor is not None:
        # gravity known per-sample from the onboard quaternion: the whole
        # problem collapses to ONE linear solve, no integration, no drift
        gS = np.asarray(g_sensor, float)[seg]
        A = M.reshape(-1, 3)
        b = (acc + gS).reshape(-1)
        r, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = float(np.sqrt(np.mean((A @ r - b) ** 2)))
        return dict(r=r, r_perp=r - (r @ axis) * axis, axis=axis,
                    resid=res, svals=S)

    if f1 <= f0:
        raise ValueError("no static slice for gravity init; pass g_sensor "
                         "(e.g. from the onboard quaternion)")
    g0 = -np.asarray(accel, float)[f0:f1].mean(0)
    g0 = g0 / np.linalg.norm(g0) * G_MAG

    def resid(x):
        r = x[:3]
        gv = Rot.from_rotvec(x[3:6]).as_matrix() @ g0 if joint_gravity else g0
        gS = np.einsum("nji,j->ni", dRs, gv)
        return (np.einsum("nij,j->ni", M, r) - gS - acc).ravel()

    sol = least_squares(resid, np.zeros(6 if joint_gravity else 3),
                        method="lm", max_nfev=8000)
    r = sol.x[:3]
    return dict(r=r,
                r_perp=r - (r @ axis) * axis,
                axis=axis,
                resid=float(np.sqrt(np.mean(sol.fun ** 2))),
                svals=S)


def solve_lever_arm(accel, gyro, fs, max_events=200, resid_percentile=70,
                    g_sensor=None, windows=None, min_events=15):
    """
    Aggregate `solve_lever_arm_one` over every roll-over in the recording.

    Keeps the better-fitting half of events, then takes a component-wise
    median. Reports the perpendicular part (trustworthy, ~mm) separately from
    the along-axis part (unobservable, and harmless).

    The per-event SPREAD is as useful as the estimate: it is the empirical
    donning/stride variability, and it is what should set your augmentation
    range rather than any guess.
    """
    wins = (pivot_windows(accel, gyro, fs) if windows is None else windows)
    wins = wins[:max_events]
    ests = [e for e in (solve_lever_arm_one(accel, gyro, fs, (a, b), c,
                                            g_sensor=g_sensor)
                        for a, b, c in wins) if e is not None]
    if not ests:
        raise RuntimeError("no usable pivot windows found -- check the "
                           "stillness detector thresholds against your data")
    if len(ests) < min_events:
        raise RuntimeError(
            f"only {len(ests)} usable pivot events (need >= {min_events} for a "
            f"stable median). Collect a longer recording: at ~1 stride/second "
            f"you want several minutes, not seconds.")

    res = np.array([e["resid"] for e in ests])
    keep = res <= np.percentile(res, resid_percentile)
    ests = [e for e, k in zip(ests, keep) if k]

    axes = np.array([e["axis"] for e in ests])
    axes *= np.sign(axes @ axes[0])[:, None]
    axis = axes.mean(0); axis /= np.linalg.norm(axis)

    R = np.array([e["r"] for e in ests])
    r_med = np.median(R, axis=0)
    r_perp = r_med - (r_med @ axis) * axis
    perp = R - np.outer(R @ axis, axis)

    return dict(
        r=r_med,
        r_perp=r_perp,
        axis=axis,
        n_events=len(ests),
        spread_perp_mm=float(np.median(np.linalg.norm(perp - r_perp, axis=1)) * 1000),
        spread_axis_mm=float(np.std(R @ axis) * 1000),
        resid_rms=float(np.median([e["resid"] for e in ests])),
        svals=np.median(np.array([e["svals"] for e in ests]), axis=0),
    )


# ---------------------------------------------------------------------------
# applying the transform
# ---------------------------------------------------------------------------

def transform_imu(accel_A, gyro_A, fs, dR, dp):
    """
    Map an IMU stream recorded at mounting A onto mounting B.

        gyro_B  = dR @ gyro_A
        accel_B = dR @ [ accel_A + (skew(alpha_A) + skew(w_A)^2) @ dp ]

    Gravity needs no special handling: it is already inside accel_A as
    specific force and rotates along with everything else.
    """
    dt = 1.0 / fs
    w = sg(gyro_A, dt, 0)
    alpha = sg(gyro_A, dt, 1)
    M = skew(alpha) + skew(w) @ skew(w)
    acc_B = (np.asarray(accel_A, float) + np.einsum("nij,j->ni", M, dp)) @ dR.T
    gyr_B = np.asarray(gyro_A, float) @ dR.T
    return acc_B, gyr_B


def sample_lever_arm(dp, axis, rng, perp_sd_m=0.008, axis_sd_m=0.025):
    """
    Draw an augmented lever arm. The along-axis component gets a WIDE spread
    because it is unobservable -- and that costs nothing, since it barely
    affects the output. The perpendicular component is known to a few mm, so
    keep its spread tight.
    """
    dp = np.asarray(dp, float)
    a = np.asarray(axis, float)
    perp = dp - (dp @ a) * a
    n1 = rng.normal(0, perp_sd_m, 3)
    n1 -= (n1 @ a) * a
    return perp + n1 + (dp @ a + rng.normal(0, axis_sd_m)) * a


def sample_rotation(dR, rng, sd_deg=3.0):
    """Perturb dR to cover residual orientation uncertainty."""
    return Rot.from_rotvec(rng.normal(0, np.deg2rad(sd_deg), 3)).as_matrix() @ dR


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------

def cycle_ensemble(sig, events, n_phase=100, min_len=40, max_len=600):
    """Segment by event index, resample each cycle to n_phase, stack."""
    sig = np.asarray(sig, float)
    out = []
    for a, b in zip(events[:-1], events[1:]):
        if not (min_len < b - a < max_len):
            continue
        src = np.linspace(0, 1, b - a)
        dst = np.linspace(0, 1, n_phase)
        out.append(np.stack([np.interp(dst, src, sig[a:b, j])
                             for j in range(sig.shape[1])], axis=1))
    return np.array(out)


def ensemble_distance(ens_A, ens_B):
    """
    Normalised RMS difference between two mean cycles, per channel and overall.
    Use it before and after the transform; the drop is your headline number.
    """
    mA, mB = ens_A.mean(0), ens_B.mean(0)
    scale = max(np.sqrt((mB ** 2).mean()), 1e-9)
    per_ch = np.sqrt(((mA - mB) ** 2).mean(0)) / scale
    return float(np.sqrt(((mA - mB) ** 2).mean()) / scale), per_ch


def assess(rec_label, fs, effective_odr, n_events, spread_perp_mm, resid_rms,
           degeneracy):
    """
    Blunt go / no-go on whether an estimate should be believed. Returns a list
    of (severity, message); empty means nothing objectionable.
    """
    out = []
    if effective_odr is not None and effective_odr < 0.6 * fs:
        out.append(("FATAL",
            f"effective ODR {effective_odr:.0f} Hz vs {fs:.0f} Hz logged. The "
            f"lever-arm solve needs angular ACCELERATION; a held-sample signal "
            f"differentiates into a train of step edges, inflating alpha ~3x "
            f"and randomising its direction. Nothing below this line is "
            f"meaningful until the firmware is fixed."))
    if n_events < 15:
        out.append(("FATAL",
            f"{n_events} pivot events. The per-event scatter is tens of mm, so "
            f"a median over <15 is noise. Collect minutes of walking."))
    elif n_events < 40:
        out.append(("WARN", f"{n_events} pivot events; {40}+ preferred."))
    if spread_perp_mm > 15:
        out.append(("WARN",
            f"perpendicular spread {spread_perp_mm:.0f} mm is large. Either the "
            f"pivot assumption is being violated or the signal is too coarse."))
    if resid_rms > 1.0:
        out.append(("WARN",
            f"fit residual {resid_rms:.2f} m/s^2 -- the rigid-pivot model is "
            f"not explaining the data well."))
    if degeneracy is not None and degeneracy > 0.7:
        out.append(("WARN",
            f"gyro axis degeneracy {degeneracy:.2f}: the medial-lateral axis is "
            f"not cleanly dominant, so dR is unreliable."))
    return out


# ===========================================================================
# MULTI-SUBJECT: stride templates + Kang-style joint refinement
# ===========================================================================
#
# Kang et al. (IEEE T-RO 2025, Sec. V.B) optimise all six transform
# parameters jointly, minimising the MSE between stride-aligned IMU data of the
# reference device and the transformed data of the new device. Here that is
# done on STRIDE TEMPLATES (mean gait cycle), per GaTech subject, against one
# exo recording.
#
# Key trick: resampling-to-phase and averaging are linear, so
#     mean_cycle[(a + M dp) dR^T] = (mean_cycle[a] + mean_cycle[M] dp) dR^T
# The template of M = skew(alpha)+skew(w)^2 is computed once from the raw,
# real-time signal (alpha in rad/s^2, NOT per-percent), after which every
# optimiser evaluation is a handful of small matrix products.

from scipy.signal import butter as _butter, filtfilt as _filtfilt


def lowpass(x, fs, fc=20.0, order=5):
    """Zero-lag Butterworth, as Kang et al. used (5th order, 20 Hz)."""
    x = np.asarray(x, float)
    if fc >= 0.45 * fs:
        return x
    b, a = _butter(order, fc / (fs / 2.0))
    return _filtfilt(b, a, x, axis=0)


def stride_ensemble(accel, gyro, fs, heel_strikes, n_phase=100, lp_hz=20.0,
                    min_stride_s=0.6, max_stride_s=2.2):
    """
    Per-stride arrays for one recording: (S, n_phase, 15) with channels
    [accel(3) | gyro(3) | M flattened(9)]. M is built from the low-passed,
    real-time gyro so alpha keeps physical units.
    """
    dt = 1.0 / fs
    a = lowpass(accel, fs, lp_hz)
    g = lowpass(gyro, fs, lp_hz)
    w = sg(g, dt, 0)
    al = sg(g, dt, 1)
    M = (skew(al) + skew(w) @ skew(w)).reshape(len(g), 9)
    X = np.hstack([a, g, M])
    return cycle_ensemble(X, np.asarray(heel_strikes, int), n_phase,
                          min_len=int(min_stride_s * fs),
                          max_len=int(max_stride_s * fs))


class Template:
    """Mean gait cycle of accel, gyro and M for one sensor."""

    def __init__(self, ens):
        ens = np.asarray(ens, float)
        if ens.ndim != 3 or len(ens) < 3:
            raise ValueError(f"need >= 3 strides for a template, got "
                             f"{0 if ens.ndim != 3 else len(ens)}")
        m = ens.mean(0)
        self.n_strides = len(ens)
        self.A = m[:, 0:3]
        self.G = m[:, 3:6]
        self.M = m[:, 6:15].reshape(-1, 3, 3)
        self.A_sd = ens[:, :, 0:3].std(0)
        self.G_sd = ens[:, :, 3:6].std(0)

    @classmethod
    def pooled(cls, ensembles):
        ens = [e for e in ensembles if e is not None and len(e)]
        return cls(np.concatenate(ens, axis=0))

    def transformed(self, dR, dp):
        G = self.G @ dR.T
        A = (self.A + np.einsum("nij,j->ni", self.M, dp)) @ dR.T
        return A, G

    def mirrored(self):
        """
        Reflect across the medial-lateral axis, for a LEFT/RIGHT leg mismatch.
        A proper rotation cannot represent a reflection, so this must happen
        before fitting. Accel is a vector (a' = P a); angular velocity is a
        pseudovector (w' = -P w); M transforms as P M P.
        """
        C = np.cov((self.G - self.G.mean(0)).T)
        n = np.linalg.eigh(C)[1][:, -1]
        P = np.eye(3) - 2.0 * np.outer(n, n)
        t = object.__new__(Template)
        t.n_strides = self.n_strides
        t.A = self.A @ P.T
        t.G = -(self.G @ P.T)
        t.M = np.einsum("ij,njk,kl->nil", P, self.M, P)
        t.A_sd, t.G_sd = self.A_sd, self.G_sd
        t.mirror_P = P
        return t


def kabsch(src_vecs, tgt_vecs):
    """Rotation R minimising sum ||R s_i - t_i||^2 (det +1 enforced)."""
    H = np.asarray(src_vecs, float).T @ np.asarray(tgt_vecs, float)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _template_rmse(A, G, tgt):
    return (float(np.sqrt(np.mean((A - tgt.A) ** 2))),
            float(np.sqrt(np.mean((G - tgt.G) ** 2))))


def kang_refine(src, tgt, dR0=None, dp0=None, max_shift=8, dp_bound=0.30,
                gyro_weight=None, dp_ridge=2.0):
    """
    Joint 6-DOF fit of the source->target transform on stride templates.

    src, tgt    : Template (source = one GaTech subject, target = your exo)
    dR0, dp0    : initial guess. Default: Kabsch on the gyro templates, dp = 0.
    max_shift   : search +-this many percent of gait cycle for a phase offset
                  between the two heel-strike definitions (force plate vs FSR)
    dp_ridge    : small penalty pulling dp toward dp0. The component of dp
                  along the rotation axis is unobservable, so without this the
                  optimiser lets it wander to the bound. The penalty is tiny
                  next to the perpendicular component's data term, so it only
                  pins the direction the data cannot see.
    gyro_weight : relative weight of gyro vs accel residuals. Default balances
                  them by the target's signal spread, so the fit is not purely
                  accelerometer-driven. (Kang summed raw MSEs; with m/s^2 vs
                  rad/s that lets accel dominate.)

    Returns dict with dR, dp, shift, and before/init/after RMSE in physical
    units (accel m/s^2, gyro rad/s).
    """
    n = len(tgt.A)
    wa = 1.0 / max(np.std(tgt.A), 1e-9)
    wg = (1.0 / max(np.std(tgt.G), 1e-9)) if gyro_weight is None else \
        gyro_weight * wa

    # ---- 1. phase shift via gyro-only Kabsch (closed form, cheap) ---------
    best = None
    for s in range(-max_shift, max_shift + 1):
        Gt = np.roll(tgt.G, s, axis=0)
        R = kabsch(src.G, Gt) if dR0 is None else dR0
        r = np.sqrt(np.mean((src.G @ R.T - Gt) ** 2))
        if best is None or r < best[0]:
            best = (r, s, R)
    _, shift, R_init = best

    class _T:                                   # target rolled by `shift`
        A = np.roll(tgt.A, shift, axis=0)
        G = np.roll(tgt.G, shift, axis=0)
    T = _T

    dp_init = np.zeros(3) if dp0 is None else np.asarray(dp0, float)

    # ---- 2. joint least squares over [rotvec, dp] ------------------------
    def resid(x):
        dR = Rot.from_rotvec(x[:3]).as_matrix() @ R_init
        A, G = src.transformed(dR, x[3:6])
        return np.concatenate([(wa * (A - T.A)).ravel(),
                               (wg * (G - T.G)).ravel(),
                               dp_ridge * (x[3:6] - dp_init)])

    x0 = np.concatenate([np.zeros(3), dp_init])
    lb = np.r_[[-np.pi] * 3, [-dp_bound] * 3]
    ub = np.r_[[np.pi] * 3, [dp_bound] * 3]
    sol = least_squares(resid, np.clip(x0, lb + 1e-9, ub - 1e-9),
                        bounds=(lb, ub), method="trf", x_scale="jac",
                        max_nfev=4000)
    dR = Rot.from_rotvec(sol.x[:3]).as_matrix() @ R_init
    dp = sol.x[3:6]

    before = _template_rmse(src.A, src.G, T)
    init = _template_rmse(*src.transformed(R_init, dp_init), T)
    after = _template_rmse(*src.transformed(dR, dp), T)

    # rotation axis of the SOURCE sensor, for splitting dp into its
    # observable (perpendicular) and unobservable (along-axis) parts
    ax = np.linalg.eigh(np.cov((src.G - src.G.mean(0)).T))[1][:, -1]
    return dict(dR=dR, dp=dp, shift=int(shift), axis=ax,
                dp_perp=dp - (dp @ ax) * ax,
                rmse_before=before, rmse_init=init, rmse_after=after,
                at_bound=bool(np.any(np.isclose(np.abs(dp), dp_bound, atol=1e-3))),
                n_src=src.n_strides, n_tgt=tgt.n_strides, success=bool(sol.success))


def rotation_mean(Rs):
    """Chordal L2 mean of rotation matrices."""
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def cluster_report(fits, mad_k=3.0):
    """
    dR describes hardware geometry, so it should cluster tightly across
    subjects. Flags outliers by angular distance from the mean rotation.
    """
    names = list(fits)
    Rs = np.array([fits[k]["dR"] for k in names])
    Rm = rotation_mean(Rs)
    ang = np.array([frame_angle_error(Rm, R) for R in Rs])
    dps = np.array([fits[k]["dp_perp"] for k in names])
    dpm = np.median(dps, axis=0)
    dpd = np.linalg.norm(dps - dpm, axis=1)

    def flags(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med)) * 1.4826 + 1e-9
        return (x - med) / mad > mad_k

    out = flags(ang) | flags(dpd)
    return dict(R_mean=Rm, dp_perp_median=dpm, angle_to_mean=dict(zip(names, ang)),
                dp_dist=dict(zip(names, dpd)),
                outliers=[n for n, o in zip(names, out) if o],
                angle_median=float(np.median(ang)),
                angle_p90=float(np.percentile(ang, 90)),
                dp_spread_mm=float(np.median(dpd) * 1000))