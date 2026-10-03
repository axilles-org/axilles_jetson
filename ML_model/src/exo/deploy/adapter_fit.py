"""Fit a :class:`SensorAdapterConfig` from one label-free exo walking trial.

Matches the exo's gait-cycle-averaged sensor waveforms to reference bands built
from the training data. Staged: (0) unit scales, (1) per-IMU rotation, (2) gait-
phase alignment, (3) per-IMU lever arm, (4) encoder offset/sign, (5) FSR
thresholds. See docs/SENSOR_ADAPTER.md for the method and the maths.
"""
from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass, field

import numpy as np

# macOS Accelerate BLAS emits a spurious matmul RuntimeWarning; results are fine.
warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

_PHASE_POINTS = 100

_MID_IMU = [
    "imu_foot_Accel_X", "imu_foot_Accel_Y", "imu_foot_Accel_Z",
    "imu_foot_Gyro_X", "imu_foot_Gyro_Y", "imu_foot_Gyro_Z",
    "imu_shank_Accel_X", "imu_shank_Accel_Y", "imu_shank_Accel_Z",
    "imu_shank_Gyro_X", "imu_shank_Gyro_Y", "imu_shank_Gyro_Z",
]

_EXO_FOOT_ACCEL = ("foot_ax", "foot_ay", "foot_az")
_EXO_FOOT_GYRO = ("foot_gx", "foot_gy", "foot_gz")
_EXO_SHANK_ACCEL = ("shank_ax", "shank_ay", "shank_az")
_EXO_SHANK_GYRO = ("shank_gx", "shank_gy", "shank_gz")


def resample_to_grid(t: np.ndarray, x: np.ndarray, rate_hz: float) -> np.ndarray:
    t = t - t[0]
    grid = np.arange(0.0, t[-1], 1.0 / rate_hz)
    if x.ndim == 1:
        return np.interp(grid, t, x)
    return np.stack([np.interp(grid, t, x[:, i]) for i in range(x.shape[1])], axis=1)


def _butter_lp(x: np.ndarray, rate_hz: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    b, a = butter(order, cutoff_hz / (0.5 * rate_hz), btype="low")
    if x.ndim == 1:
        return filtfilt(b, a, x)
    return np.stack([filtfilt(b, a, x[:, i]) for i in range(x.shape[1])], axis=1)


def heel_strike_indices(foot_gyro_sagittal: np.ndarray, rate_hz: float,
                        min_cycle_s: float = 0.6) -> np.ndarray:
    """Heel strikes = negative-going zero crossings of the detrended foot sagittal
    gyro, at least ``min_cycle_s`` apart. Robust enough to *segment* cycles; the
    exact phase origin is corrected later against the foot-accel impact."""
    g = _butter_lp(foot_gyro_sagittal - np.mean(foot_gyro_sagittal), rate_hz, 6.0)
    crossings = np.where((g[:-1] > 0) & (g[1:] <= 0))[0] + 1
    if crossings.size == 0:
        return crossings
    min_gap = int(min_cycle_s * rate_hz)
    kept = [int(crossings[0])]
    for c in crossings[1:]:
        if c - kept[-1] >= min_gap:
            kept.append(int(c))
    return np.asarray(kept, dtype=int)


def phase_average(signal: np.ndarray, events: np.ndarray,
                  return_std: bool = False, phase_shift: int = 0):
    """Average ``signal`` over the cycles delimited by ``events``, resampled to
    ``_PHASE_POINTS``. ``phase_shift`` circularly rolls the phase axis of the
    result (used to align the exo cycle origin to the GaTech one without dropping
    cycles by trimming event indices)."""
    if events.size < 3:
        raise ValueError("need at least 2 full gait cycles")
    grid = np.linspace(0.0, 1.0, _PHASE_POINTS)
    cycles = []
    for a, b in zip(events[:-1], events[1:]):
        seg = signal[a:b]
        if len(seg) < 5:
            continue
        src = np.linspace(0.0, 1.0, len(seg))
        if seg.ndim == 1:
            cycles.append(np.interp(grid, src, seg))
        else:
            cycles.append(np.stack([np.interp(grid, src, seg[:, i])
                                    for i in range(seg.shape[1])], axis=1))
    stack = np.stack(cycles, axis=0)
    if phase_shift:
        stack = np.roll(stack, -phase_shift, axis=1)
    if return_std:
        return stack.mean(axis=0), stack.std(axis=0)
    return stack.mean(axis=0)


# GaTech reference bands
@dataclass
class ReferenceBands:
    channels: list[str]
    mean: np.ndarray            # (P, C)
    std: np.ndarray             # (P, C)
    foot_accel_impact_phase: float   # cycle phase of the |a| heel-strike spike
    stance_duty: float
    stance_onset_phase: float


def _slow_trial_names(processed_dir: str, speed_lo: float, speed_hi: float) -> set[str] | None:
    """Trial stems whose treadmill speed falls in [speed_lo, speed_hi], from the
    dataset ``metadata.parquet`` two levels up. Returns None if metadata absent."""
    for cand in (os.path.join(processed_dir, "..", "hf_dataset", "metadata.parquet"),
                 os.path.join(processed_dir, "..", "metadata.parquet"),
                 os.path.join(processed_dir, "metadata.parquet")):
        if os.path.exists(cand):
            import pandas as pd
            m = pd.read_parquet(cand)
            if "speed_mean_mps" not in m:
                return None
            sel = m[(m.speed_mean_mps >= speed_lo) & (m.speed_mean_mps <= speed_hi)]
            return {f"{r.subject}_{r.trial}" for r in sel.itertuples()}
    return None


def build_reference_bands(processed_dir: str, rate_hz: float = 100.0,
                          max_trials: int = 60,
                          speed_range: tuple[float, float] | None = None
                          ) -> ReferenceBands:
    files = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no .npz trials in {processed_dir}")

    if speed_range is not None:
        keep = _slow_trial_names(processed_dir, *speed_range)
        if keep:
            matched = [f for f in files
                       if os.path.splitext(os.path.basename(f))[0] in keep]
            if len(matched) >= 5:
                files = matched
    files = files[:max_trials]

    cols = list(np.load(files[0])["feature_columns"])
    imu_idx = [cols.index(c) for c in _MID_IMU]
    stance_idx = cols.index("stance")
    ankle_idx = cols.index("gon_ankle_sagittal")
    foot_gyro_sag_idx = cols.index("imu_foot_Gyro_Y")
    foot_acc_idx = [cols.index(c) for c in _MID_IMU[0:3]]

    stacks, duty, onset, impact = [], [], [], []
    for f in files:
        feats = np.load(f)["features"]
        events = heel_strike_indices(feats[:, foot_gyro_sag_idx], rate_hz)
        if events.size < 4:
            continue
        sig = feats[:, imu_idx + [ankle_idx, stance_idx]]
        avg = phase_average(sig, events)
        stacks.append(avg)
        duty.append(float(np.mean(feats[:, stance_idx])))
        on = np.where(avg[:, -1] > 0.5)[0]
        if on.size:
            onset.append(on[0] / _PHASE_POINTS)
        amag = np.linalg.norm(phase_average(feats[:, foot_acc_idx], events), axis=1)
        impact.append(int(np.argmax(amag)) / _PHASE_POINTS)

    arr = np.stack(stacks, axis=0)
    return ReferenceBands(
        channels=_MID_IMU + ["gon_ankle_sagittal", "stance"],
        mean=arr.mean(axis=0),
        std=arr.std(axis=0) + 1e-6,
        foot_accel_impact_phase=float(np.mean(impact)),
        stance_duty=float(np.mean(duty)),
        stance_onset_phase=float(np.mean(onset)) if onset else 0.0,
    )


# rotation / rigid-body helpers
def _kabsch(src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Proper rotation (det +1) mapping ``src`` (N,3) onto ``dst`` (N,3)."""
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        return np.eye(3)
    w = weights / (weights.sum() + 1e-12)
    cov = (src * w[:, None]).T @ dst
    u, s, vt = np.linalg.svd(cov)
    if s[1] < 1e-9:                       # rank-deficient -> rotation underdetermined
        return np.eye(3)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot if np.isfinite(rot).all() else np.eye(3)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


# per-IMU staged fit
@dataclass
class ImuFit:
    rotation: np.ndarray            # 3x3
    gyro_scale: float
    accel_scale: float
    lever_arm_m: np.ndarray         # (3,)
    gyro_residual: float            # normalised MSE, gyro channels
    accel_residual_before: float    # rotation only
    accel_residual_after: float     # rotation + lever arm
    phase_avg_6ch: np.ndarray       # (P, 6) after full adaptation


def _lever_from_residual(a_rot_pa: np.ndarray, w_hat: np.ndarray,
                         ref_accel: np.ndarray, ref_accel_std: np.ndarray,
                         cycle_s: float) -> np.ndarray:
    """Linear LS for the lever arm ``r`` from the rotated-accel residual.

    ``accel_meas - accel_ref  ~=  -( [w]x[w]x r + [wdot]x r )`` is linear in ``r``;
    stack the 3x3 operator over phase samples and solve. ``w_hat`` is the rotated,
    scaled gyro phase-average (rad/s); ``wdot`` is its cycle-time derivative.
    """
    wdot = np.gradient(w_hat, cycle_s / _PHASE_POINTS, axis=0)
    rows, rhs = [], []
    for k in range(_PHASE_POINTS):
        Sw, Swd = _skew(w_hat[k]), _skew(wdot[k])
        rows.append(-(Sw @ Sw + Swd))
        rhs.append(ref_accel[k] - a_rot_pa[k])
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    w = np.repeat(1.0 / ref_accel_std.mean(axis=1), 3)
    r, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    return np.clip(r, -0.3, 0.3)


def _apply_lever(a_rot_pa: np.ndarray, w_hat: np.ndarray, r: np.ndarray,
                 cycle_s: float) -> np.ndarray:
    wdot = np.gradient(w_hat, cycle_s / _PHASE_POINTS, axis=0)
    corr = np.stack([_skew(w_hat[k]) @ _skew(w_hat[k]) @ r + _skew(wdot[k]) @ r
                     for k in range(_PHASE_POINTS)], axis=0)
    return a_rot_pa - corr


def _wahba_rotation(grav_exo, gyro_exo_pa, ref_accel, ref_gyro) -> np.ndarray:
    """Rotation from the gravity direction (2 DOF) + gyro swing axis (3rd DOF).

    The exo trial and the GaTech band may be at different walking speeds, so the
    gyro *amplitudes* differ. Normalise each side's gyro block by its own RMS
    before stacking, so the rotation is driven by the swing-axis *direction*, not
    the amplitude. Gravity is speed-independent and enters unnormalised (it is the
    dominant, best-conditioned constraint)."""
    ex_w = gyro_exo_pa / (np.sqrt(np.mean(gyro_exo_pa ** 2)) + 1e-9)
    rf_w = ref_gyro / (np.sqrt(np.mean(ref_gyro ** 2)) + 1e-9)
    ex_g = grav_exo / (np.linalg.norm(grav_exo, axis=1).mean() + 1e-9)
    rf_g = ref_accel / (np.linalg.norm(ref_accel, axis=1).mean() + 1e-9)

    # weight gravity ~2x the gyro block: it is the cleaner constraint
    src = np.vstack([ex_g * np.sqrt(2.0), ex_w])
    dst = np.vstack([rf_g * np.sqrt(2.0), rf_w])
    wts = np.concatenate([np.full(_PHASE_POINTS, 1.0),
                          np.linalg.norm(rf_w, axis=1) /
                          (np.linalg.norm(rf_w, axis=1).mean() + 1e-9)])
    return _kabsch(src, dst, wts)


def _fit_one_imu(accel: np.ndarray, gyro: np.ndarray, events: np.ndarray,
                 ref_accel: np.ndarray, ref_gyro: np.ndarray,
                 ref_accel_std: np.ndarray, ref_gyro_std: np.ndarray,
                 rate_hz: float, fit_lever_arm: bool = True,
                 phase_shift: int = 0, iters: int = 3) -> ImuFit:
    def pavg(x):
        return phase_average(x, events, phase_shift=phase_shift)

    cycle_s = float(np.mean(np.diff(events)) / rate_hz)

    # --- stage 0: unit scales ---
    gyro_scale = float((np.sqrt(np.mean(ref_gyro ** 2)) + 1e-9) /
                       (np.sqrt(np.mean(gyro ** 2)) + 1e-9))
    g = gyro * gyro_scale
    # accel scale from gravity: |low-freq accel| should be 1 g in GaTech units
    grav_mag = np.median(np.linalg.norm(_butter_lp(accel, rate_hz, 1.0), axis=1))
    ref_grav_mag = np.median(np.linalg.norm(ref_accel, axis=1))
    accel_scale = float(ref_grav_mag / grav_mag) if grav_mag > 1e-6 else 1.0
    a = accel * accel_scale

    exo_g_pa = pavg(g)
    exo_a_pa = pavg(a)
    accel_res_before = None
    lever = np.zeros(3)

    # Alternate rotation (gravity + gyro) and lever arm (accel residual); the
    # lever-arm term slightly perturbs the gravity estimate. Converges in ~3.
    rot = _wahba_rotation(exo_a_pa, exo_g_pa, ref_accel, ref_gyro)
    for _ in range(iters):
        g_rot_pa = exo_g_pa @ rot.T
        a_corr = _apply_lever(exo_a_pa, g_rot_pa, lever, cycle_s)
        rot = _wahba_rotation(a_corr, exo_g_pa, ref_accel, ref_gyro)
        g_rot_pa = exo_g_pa @ rot.T
        a_rot_pa = pavg(a @ rot.T)
        if accel_res_before is None:
            accel_res_before = float(np.mean(((a_rot_pa - ref_accel) / ref_accel_std) ** 2))
        if fit_lever_arm:
            lever = _lever_from_residual(a_rot_pa, g_rot_pa, ref_accel,
                                         ref_accel_std, cycle_s)

    g_rot_pa = exo_g_pa @ rot.T
    a_rot_pa = pavg(a @ rot.T)
    gyro_res = float(np.mean(((g_rot_pa - ref_gyro) / ref_gyro_std) ** 2))
    a_corrected_pa = _apply_lever(a_rot_pa, g_rot_pa, lever, cycle_s)
    accel_res_after = float(np.mean(((a_corrected_pa - ref_accel) / ref_accel_std) ** 2))

    phase_avg_6ch = np.concatenate([a_corrected_pa, g_rot_pa], axis=1)
    return ImuFit(rot, gyro_scale, accel_scale, lever, gyro_res,
                  accel_res_before, accel_res_after, phase_avg_6ch)


# encoder + FSR
def _fit_encoder(ankle_deg: np.ndarray, events: np.ndarray,
                 ref_ankle: np.ndarray, phase_shift: int = 0) -> tuple[float, float, float]:
    best = None
    for sign in (1.0, -1.0):
        avg = phase_average(np.radians(ankle_deg) * sign, events, phase_shift=phase_shift)
        neutral_rad = np.mean(avg) - np.mean(ref_ankle)
        err = float(np.mean(((avg - neutral_rad) - ref_ankle) ** 2))
        if best is None or err < best[0]:
            best = (err, float(np.degrees(neutral_rad) * sign), sign, avg)
    err, neutral_deg, sign, avg = best
    corr = float(np.corrcoef(avg, ref_ankle)[0, 1])
    return neutral_deg, sign, corr


def _debounced_stance(heel: np.ndarray, toe: np.ndarray, ht: float, tt: float,
                      rate_hz: float, min_state_s: float = 0.10) -> np.ndarray:
    raw = ((heel > ht) | (toe > tt)).astype(float)
    min_len = max(1, int(min_state_s * rate_hz))
    out = raw.copy()
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if j - i < min_len and 0 < i:            # too-short run -> absorb into previous
            out[i:j] = out[i - 1]
        i = j
    return out


def _bimodal_midpoint(x: np.ndarray) -> tuple[float, float, float]:
    """(low mode, high mode, midpoint) of a bimodal FSR signal via a 2-way split
    at the largest histogram gap in the mid-range."""
    lo_c = np.median(x[x < np.percentile(x, 40)])
    hi_c = np.median(x[x > np.percentile(x, 60)])
    return float(lo_c), float(hi_c), float(0.5 * (lo_c + hi_c))


def _fit_fsr_thresholds(heel: np.ndarray, toe: np.ndarray, events: np.ndarray,
                        rate_hz: float, ref_stance: np.ndarray,
                        target_duty: float, phase_shift: int = 0
                        ) -> tuple[float, float, dict]:
    """Choose FSR thresholds to match the GaTech stance waveform in the aligned
    phase frame. The exo FSRs are bimodal (no contact ~0, contact ~20-26k), so
    search a grid *centred on the bimodal midpoint* of each sensor and prefer the
    threshold nearest that midpoint among near-optimal fits (robust to noise)."""
    ref_on = np.where(ref_stance > 0.5)[0]
    onset_goal = float(ref_on[0] / _PHASE_POINTS) if ref_on.size else 0.3

    h_lo, h_hi, h_mid = _bimodal_midpoint(heel)
    t_lo, t_hi, t_mid = _bimodal_midpoint(toe)
    heel_grid = np.linspace(h_lo + 0.15 * (h_hi - h_lo), h_lo + 0.85 * (h_hi - h_lo), 25)
    toe_grid = np.linspace(t_lo + 0.15 * (t_hi - t_lo), t_lo + 0.85 * (t_hi - t_lo), 25)

    best = None
    for ht in heel_grid:
        for tt in toe_grid:
            stance = _debounced_stance(heel, toe, ht, tt, rate_hz)
            duty = float(np.mean(stance))
            if duty < 0.30 or duty > 0.80:
                continue
            avg = phase_average(stance, events, phase_shift=phase_shift)
            on = np.where(avg > 0.5)[0]
            on_phase = on[0] / _PHASE_POINTS if on.size else 1.0
            waveform_err = float(np.mean((avg - ref_stance) ** 2))
            onset_err = min(abs(on_phase - onset_goal), 1.0 - abs(on_phase - onset_goal))
            # small penalty for straying from the bimodal midpoint
            centre_pen = 0.10 * (abs(ht - h_mid) / (h_hi - h_lo) +
                                 abs(tt - t_mid) / (t_hi - t_lo))
            score = (2.0 * waveform_err + 1.0 * onset_err +
                     1.5 * abs(duty - target_duty) + centre_pen)
            if best is None or score < best[0]:
                best = (score, float(ht), float(tt), duty, on_phase)
    _, ht, tt, duty, on_phase = best
    diag = {"duty": round(duty, 3), "onset_phase": round(on_phase, 3),
            "onset_goal": round(onset_goal, 3),
            "heel_modes": [round(h_lo), round(h_hi)],
            "toe_modes": [round(t_lo), round(t_hi)]}
    return ht, tt, diag


# top-level
@dataclass
class AdapterFitResult:
    config_block: dict
    exo_phase_avg: np.ndarray        # (P, 14)
    reference: ReferenceBands
    n_cycles: int
    phase_shift_frames: int
    diagnostics: dict = field(default_factory=dict)


def _load_exo_walk(csv_path: str, rate_hz: float):
    import pandas as pd

    df = pd.read_csv(csv_path).sort_values("timestamp_s")
    df = df.interpolate().ffill().bfill()
    t = df["timestamp_s"].to_numpy()

    def block(cols):
        return resample_to_grid(t, df[list(cols)].to_numpy(), rate_hz)

    return {
        "foot_accel": block(_EXO_FOOT_ACCEL),
        "foot_gyro": block(_EXO_FOOT_GYRO),
        "shank_accel": block(_EXO_SHANK_ACCEL),
        "shank_gyro": block(_EXO_SHANK_GYRO),
        "ankle_deg": resample_to_grid(t, df["ankle_encoder_deg"].to_numpy(), rate_hz),
        "heel_fsr": resample_to_grid(t, df["heel_fsr_raw"].to_numpy(), rate_hz),
        "toe_fsr": resample_to_grid(t, df["toe_fsr_raw"].to_numpy(), rate_hz),
    }


def _exo_foot_sagittal_gyro_axis(foot_gyro: np.ndarray) -> int:
    """The sagittal (pitch) axis is the gyro component with the largest
    gait-periodic swing; pick it by variance."""
    return int(np.argmax(np.var(foot_gyro, axis=0)))


def _best_phase_lag(exo: dict, events: np.ndarray, ref: ReferenceBands,
                    sag_axis: int) -> int:
    """Phase-origin offset (in phase points) between the exo cycle definition and
    GaTech's. Cross-correlate several large, clean gait signatures jointly:
    foot sagittal gyro, shank sagittal gyro, and the encoder angle."""
    ref_foot_g = ref.mean[:, 4]
    ref_shank_g = ref.mean[:, 10]
    ref_ank = ref.mean[:, 12]
    shank_sag = int(np.argmax(np.var(exo["shank_gyro"], axis=0)))

    def norm_pa(sig):
        pa = phase_average(sig, events)
        return (pa - pa.mean()) / (pa.std() + 1e-9)

    exo_sigs = [norm_pa(exo["foot_gyro"][:, sag_axis]),
                norm_pa(exo["shank_gyro"][:, shank_sag]),
                norm_pa(exo["ankle_deg"])]
    ref_sigs = [(ref_foot_g - ref_foot_g.mean()) / (ref_foot_g.std() + 1e-9),
                (ref_shank_g - ref_shank_g.mean()) / (ref_shank_g.std() + 1e-9),
                (ref_ank - ref_ank.mean()) / (ref_ank.std() + 1e-9)]

    best = None
    for signs in [(sf, ss) for sf in (1, -1) for ss in (1, -1)]:
        sf, ss = signs
        s = [sf * exo_sigs[0], ss * exo_sigs[1], exo_sigs[2]]
        cc = np.zeros(_PHASE_POINTS)
        for L in range(_PHASE_POINTS):
            cc[L] = sum(np.dot(np.roll(si, -L), ri) for si, ri in zip(s, ref_sigs))
        L = int(np.argmax(cc))
        if best is None or cc[L] > best[0]:
            best = (cc[L], L)
    return int(best[1] % _PHASE_POINTS)


def fit_adapter(csv_path: str, processed_dir: str,
                control_rate_hz: float = 100.0,
                fit_lever_arm: bool = True,
                match_slow_speed: bool = True) -> AdapterFitResult:
    rate = control_rate_hz
    exo = _load_exo_walk(csv_path, rate)
    # restrict the reference band to GaTech's slowest trials (the exo walk is
    # slower than any of them); the rotation fits shape, not amplitude
    speed_range = (0.80, 0.98) if match_slow_speed else None
    ref = build_reference_bands(processed_dir, rate, speed_range=speed_range)

    # segment cycles from the exo foot sagittal gyro
    sag_axis = _exo_foot_sagittal_gyro_axis(exo["foot_gyro"])
    events = heel_strike_indices(exo["foot_gyro"][:, sag_axis], rate)
    if events.size < 4:
        raise ValueError(f"only {events.size} heel strikes - recording too short")

    # stage 2: phase-origin correction (multi-signal cross-correlation)
    phase_lag = _best_phase_lag(exo, events, ref, sag_axis)
    shift_frames = phase_lag

    def epavg(x):
        return phase_average(x, events, phase_shift=phase_lag)

    # stages 0 + 1 + 3 per IMU (per-IMU unit scales, rotation, lever arm; iterated)
    foot = _fit_one_imu(
        exo["foot_accel"], exo["foot_gyro"], events,
        ref.mean[:, 0:3], ref.mean[:, 3:6], ref.std[:, 0:3], ref.std[:, 3:6],
        rate, fit_lever_arm, phase_shift=phase_lag)
    shank = _fit_one_imu(
        exo["shank_accel"], exo["shank_gyro"], events,
        ref.mean[:, 6:9], ref.mean[:, 9:12], ref.std[:, 6:9], ref.std[:, 9:12],
        rate, fit_lever_arm, phase_shift=phase_lag)

    # stage 4: encoder
    neutral_deg, enc_sign, enc_corr = _fit_encoder(
        exo["ankle_deg"], events, ref.mean[:, 12], phase_shift=phase_lag)

    # stage 5: FSR - match the GaTech stance waveform in the aligned phase frame
    heel_thr, toe_thr, fsr_diag = _fit_fsr_thresholds(
        exo["heel_fsr"], exo["toe_fsr"], events, rate,
        ref.mean[:, 13], ref.stance_duty, phase_shift=phase_lag)

    ankle_pa = epavg(np.radians(exo["ankle_deg"]) * enc_sign)
    ankle_pa = ankle_pa - (np.mean(ankle_pa) - np.mean(ref.mean[:, 12]))
    stance_pa = epavg(
        _debounced_stance(exo["heel_fsr"], exo["toe_fsr"], heel_thr, toe_thr, rate))
    exo_phase_avg = np.concatenate(
        [foot.phase_avg_6ch, shank.phase_avg_6ch,
         ankle_pa[:, None], stance_pa[:, None]], axis=1)

    block = {
        "foot_rotation": np.round(foot.rotation, 6).tolist(),
        "shank_rotation": np.round(shank.rotation, 6).tolist(),
        "foot_accel_scale": round(float(foot.accel_scale), 6),
        "foot_gyro_scale": round(float(foot.gyro_scale), 6),
        "shank_accel_scale": round(float(shank.accel_scale), 6),
        "shank_gyro_scale": round(float(shank.gyro_scale), 6),
        "foot_lever_arm_m": np.round(foot.lever_arm_m, 5).tolist(),
        "shank_lever_arm_m": np.round(shank.lever_arm_m, 5).tolist(),
        "ankle_encoder_neutral_deg": round(float(neutral_deg), 3),
        "ankle_encoder_sign": float(enc_sign),
        "heel_fsr_threshold": round(float(heel_thr), 1),
        "toe_fsr_threshold": round(float(toe_thr), 1),
        "fsr_debounce_s": 0.10,
    }
    diagnostics = {
        "n_cycles": int(events.size - 1),
        "phase_shift_frames": shift_frames,
        "foot": {
            "gyro_residual": round(foot.gyro_residual, 3),
            "accel_residual_rotation_only": round(foot.accel_residual_before, 3),
            "accel_residual_with_lever_arm": round(foot.accel_residual_after, 3),
            "lever_arm_cm": np.round(foot.lever_arm_m * 100, 2).tolist(),
        },
        "shank": {
            "gyro_residual": round(shank.gyro_residual, 3),
            "accel_residual_rotation_only": round(shank.accel_residual_before, 3),
            "accel_residual_with_lever_arm": round(shank.accel_residual_after, 3),
            "lever_arm_cm": np.round(shank.lever_arm_m * 100, 2).tolist(),
        },
        "encoder_corr": round(enc_corr, 3),
        "fsr": fsr_diag,
        "reference_stance_duty": round(ref.stance_duty, 3),
    }
    return AdapterFitResult(
        config_block=block,
        exo_phase_avg=exo_phase_avg,
        reference=ref,
        n_cycles=int(events.size - 1),
        phase_shift_frames=shift_frames,
        diagnostics=diagnostics,
    )


def summarize(result: AdapterFitResult) -> str:
    d = result.diagnostics
    b = result.config_block
    f, s = d["foot"], d["shank"]
    lines = [
        f"fitted from {d['n_cycles']} gait cycles   phase-origin shift {d['phase_shift_frames']}%",
        "",
        f"  foot   accel_scale {b['foot_accel_scale']:<9}  gyro_scale {b['foot_gyro_scale']:<9}"
        f"  lever {f['lever_arm_cm']} cm",
        f"  shank  accel_scale {b['shank_accel_scale']:<9}  gyro_scale {b['shank_gyro_scale']:<9}"
        f"  lever {s['lever_arm_cm']} cm",
        f"  ankle_encoder_neutral_deg {b['ankle_encoder_neutral_deg']}   "
        f"sign {b['ankle_encoder_sign']}   (cycle corr {d['encoder_corr']})",
        f"  heel_fsr_threshold {b['heel_fsr_threshold']}   toe_fsr_threshold {b['toe_fsr_threshold']}"
        f"   (duty {d['fsr']['duty']} vs {d['reference_stance_duty']}, "
        f"onset {d['fsr']['onset_phase']})",
        "",
        "  residual (norm MSE)          gyro    accel rot-only   accel +lever",
        f"    foot                      {f['gyro_residual']:>6}    {f['accel_residual_rotation_only']:>10}"
        f"     {f['accel_residual_with_lever_arm']:>10}",
        f"    shank                     {s['gyro_residual']:>6}    {s['accel_residual_rotation_only']:>10}"
        f"     {s['accel_residual_with_lever_arm']:>10}",
    ]
    return "\n".join(lines)
