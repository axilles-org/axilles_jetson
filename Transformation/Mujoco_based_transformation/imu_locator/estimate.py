"""Estimate where an IMU sits on a segment from mocap kinematics + raw IMU data.

Pipeline (per candidate segment):
  1. low-pass the IMU with the same cutoff used on the mocap
  2. time offset: cross-correlate |gyro| with |omega_segment| (rotation invariant)
  3. orientation Q (sensor->segment) + gyro bias: Kabsch fit of gyro to omega
  4. segment choice: lowest normalised gyro residual among candidates
  5. position r + accel bias: linear least squares on
         Q a_meas - f = [alpha]x r + [omega]x[omega]x r + Q b_a
  6. joint nonlinear refinement of (Q, r, b_g, b_a) with optional position prior
  7. uncertainty: Gauss-Newton covariance (autocorrelation-corrected) and
     spread across independent time chunks
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import correlate, correlation_lags
from scipy.spatial.transform import Rotation as Rot

from .kinematics import SegmentKin
from .model import C_MJ2OS
from .signal_utils import interp_rows, lowpass, sample_rate


@dataclass
class ImuData:
    t: np.ndarray
    acc: np.ndarray   # (N,3) specific force, m/s^2, sensor frame
    gyr: np.ndarray   # (N,3) rad/s, sensor frame


@dataclass
class ImuEstimate:
    segment: str
    time_offset: float                  # t_mocap = t_imu + time_offset
    R: np.ndarray                       # sensor -> segment (MuJoCo frame)
    r: np.ndarray                       # position in segment frame (MuJoCo) [m]
    gyro_bias: np.ndarray
    acc_bias: np.ndarray
    rms_gyro: float
    rms_acc: float
    r_std: np.ndarray = None            # 1-sigma, Gauss-Newton
    rot_std_deg: np.ndarray = None
    r_chunk_std: np.ndarray = None      # 1-sigma across time chunks
    rot_chunk_std_deg: float = None
    candidates: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    acc_sign: int = 1                   # -1: accelerometer inverted relative to gyro
    rms_acc_by_sign: dict = None
    rel_acc: float = None               # rms_acc / accel signal level (quality metric)
    sat_frac: float = 0.0               # fraction of frames excluded for saturation

    def to_dict(self) -> dict:
        q = Rot.from_matrix(self.R).as_quat(scalar_first=True)
        R_os = C_MJ2OS @ self.R @ C_MJ2OS.T
        f = lambda a: None if a is None else np.round(np.asarray(a, float), 5).tolist()
        return dict(
            segment=self.segment,
            time_offset_s=round(self.time_offset, 5),
            position_m_mujoco=f(self.r),
            position_m_opensim=f(C_MJ2OS @ self.r),
            quat_wxyz_sensor_to_segment_mujoco=f(q),
            R_sensor_to_segment_mujoco=f(self.R),
            R_sensor_to_segment_opensim=f(R_os),
            sensor_axes_in_segment_mujoco={ax: f(self.R[:, i]) for i, ax in enumerate("xyz")},
            gyro_bias_rad_s=f(self.gyro_bias),
            acc_bias_m_s2=f(self.acc_bias),
            rms_residual_gyro_rad_s=round(self.rms_gyro, 5),
            rms_residual_acc_m_s2=round(self.rms_acc, 5),
            position_std_m_mujoco=f(self.r_std),
            rotation_std_deg=f(self.rot_std_deg),
            position_chunk_std_m_mujoco=f(self.r_chunk_std),
            rotation_chunk_std_deg=None if self.rot_chunk_std_deg is None else round(self.rot_chunk_std_deg, 3),
            acc_sign=self.acc_sign,
            segment_ranking=self.candidates,
            warnings=self.warnings,
        )


# ---------------------------------------------------------------- helpers
def _skew(v):
    Z = np.zeros(v.shape[:-1] + (3, 3))
    Z[..., 0, 1], Z[..., 0, 2] = -v[..., 2], v[..., 1]
    Z[..., 1, 0], Z[..., 1, 2] = v[..., 2], -v[..., 0]
    Z[..., 2, 0], Z[..., 2, 1] = -v[..., 1], v[..., 0]
    return Z


def _slice(seg: SegmentKin, idx) -> SegmentKin:
    return SegmentKin(t=seg.t[idx], omega=seg.omega[idx], alpha=seg.alpha[idx],
                      f=seg.f[idx], R=seg.R[idx], p=seg.p[idx])


def estimate_time_offset(t_ref, w_ref, t_imu, w_imu, max_lag=0.3, guess=0.0):
    """Offset such that t_ref = t_imu + offset, from |omega| cross-correlation."""
    fs = sample_rate(t_ref)
    dt = 1.0 / fs
    ga = np.arange(t_ref[0], t_ref[-1], dt)
    gb = np.arange(t_imu[0], t_imu[-1], dt)
    a = np.interp(ga, t_ref, np.linalg.norm(w_ref, axis=1))
    b = np.interp(gb, t_imu, np.linalg.norm(w_imu, axis=1))
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    c = correlate(a, b, mode="full", method="fft")
    lags = correlation_lags(len(a), len(b), mode="full")
    # normalise by overlap length so partial overlaps are not favoured
    n_ov = np.array([min(len(a), len(b) + L) - max(0, L) for L in lags], float)
    c = c / np.maximum(n_ov, 1)
    offsets = ga[0] - gb[0] + lags * dt
    ok = n_ov > 0.3 * min(len(a), len(b))
    if max_lag is not None:
        ok &= np.abs(offsets - guess) <= max_lag
    if not ok.any():
        raise ValueError("no admissible lag; increase max_lag or check timestamps")
    c_ok = np.where(ok, c, -np.inf)
    k = int(np.argmax(c_ok))
    off = offsets[k]
    if 0 < k < len(c) - 1 and np.isfinite(c_ok[k - 1]) and np.isfinite(c_ok[k + 1]):
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            off += 0.5 * (y0 - y2) / den * dt
    return float(off), float(c[k])


def fit_orientation(omega, gyr):
    """Kabsch: gyr ~= Q^T omega + b.  Returns Q (sensor->segment), b, info."""
    A = omega - omega.mean(0)
    B = gyr - gyr.mean(0)
    H = B.T @ A
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Q = Vt.T @ np.diag([1, 1, d]) @ U.T          # omega_c ~= Q gyr_c
    Q_improper = Vt.T @ np.diag([1, 1, -d]) @ U.T
    b = gyr.mean(0) - omega.mean(0) @ Q
    res = gyr - omega @ Q - b
    res_imp = B - A @ Q_improper
    sv = np.linalg.svd(A, compute_uv=False) / np.sqrt(len(A))
    return Q, b, dict(rms=float(np.sqrt((res ** 2).mean() * 3)),
                      rel=float(np.sqrt((res ** 2).sum() / (B ** 2).sum())),
                      rel_improper=float(np.sqrt((res_imp ** 2).sum() / (B ** 2).sum())),
                      omega_sv=sv)


def fit_orientation_joint(omega, gyr, f, acc, fs, fc_grav=1.5, w_grav=1.0, prefiltered=False):
    """Wahba fit of Q using two vector sets at once:
         gyro  : centred gyr  <->  centred omega      (gives the rotation axes)
         gravity: low-passed acc <-> low-passed f     (gives rotation about the
                  dominant axis, which gyro alone cannot fix in planar motion)
    Each set is normalised by its energy; w_grav scales the gravity set.
    Lever-arm terms are small at < fc_grav and are ignored here."""
    A1, B1 = omega - omega.mean(0), gyr - gyr.mean(0)
    A2, B2 = (f, acc) if prefiltered else (lowpass(f, fs, fc_grav), lowpass(acc, fs, fc_grav))
    H = B1.T @ A1 / (A1 ** 2).sum() + w_grav * B2.T @ A2 / (A2 ** 2).sum()
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Q = Vt.T @ np.diag([1, 1, d]) @ U.T
    b = gyr.mean(0) - omega.mean(0) @ Q
    res = gyr - omega @ Q - b
    return Q, b, dict(rel=float(np.sqrt((res ** 2).sum() / (B1 ** 2).sum())),
                      grav_rms=float(np.sqrt(((B2 - A2 @ Q) ** 2).sum(1).mean())))


def detect_acc_sign(omega, gyr, f, acc, fs, fc_grav=1.0):
    """+1 if the accelerometer shares the gyro's frame with the usual
    specific-force convention, -1 if it is inverted (mirrored) relative to the
    gyro.  Test: orientation from gravity alone must also explain the gyro."""
    A2 = lowpass(f, fs, fc_grav)
    oc, gc = omega - omega.mean(0), gyr - gyr.mean(0)
    out = {}
    for sgn in (1, -1):
        B2 = lowpass(sgn * acc, fs, fc_grav)
        U, S, Vt = np.linalg.svd(B2.T @ A2)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        Q = Vt.T @ np.diag([1, 1, d]) @ U.T
        out[sgn] = float(np.sqrt(((gc @ Q.T - oc) ** 2).sum() / (oc ** 2).sum()))
    return (1 if out[1] <= out[-1] else -1), out


def fit_position(seg: SegmentKin, acc, Q, omega=None, alpha=None):
    """Linear LSQ for r and c = Q b_a given orientation Q.
    omega/alpha (segment frame) override the mocap ones for the lever-arm terms."""
    om = seg.omega if omega is None else omega
    al = seg.alpha if alpha is None else alpha
    K = _skew(al) + _skew(om) @ _skew(om)
    N = len(acc)
    A = np.concatenate([K, np.broadcast_to(np.eye(3), (N, 3, 3))], axis=2).reshape(-1, 6)
    y = (acc @ Q.T - seg.f).reshape(-1)
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    r, c = x[:3], x[3:]
    sv = np.linalg.svd(K.reshape(-1, 3), compute_uv=False) / np.sqrt(N)
    return r, Q.T @ c, sv


def refine(seg, acc, gyr, Q0, r0, bg0, ba0, prior=None, inflate=1.0, dgyr=None, fix_rot=False):
    """Joint nonlinear LSQ over (rotvec, r, b_g, b_a).

    prior: (r_prior (3,), sigma (3,) or scalar) Gaussian prior on position.
    inflate: residual autocorrelation factor (~ fs / (2 fc)).
    dgyr: time derivative of gyr. If given, the lever-arm terms use the IMU's own
          (bias-corrected) gyro instead of the mocap omega/alpha. The gyro is far
          less noisy than twice-differentiated mocap, which removes the
          errors-in-variables bias that noisy omega causes in omega x (omega x r).
    """
    om, al, f = seg.omega, seg.alpha, seg.f

    def predict(p):
        Q = Q0 @ Rot.from_rotvec(p[:3]).as_matrix()
        r = p[3:6]
        if dgyr is None:
            a_pt = f + np.cross(al, r) + np.cross(om, np.cross(om, r))
            pa = a_pt @ Q
        else:
            w_s, r_s = gyr - p[6:9], Q.T @ r
            pa = f @ Q + np.cross(dgyr, r_s) + np.cross(w_s, np.cross(w_s, r_s))
        return pa + p[9:12], om @ Q + p[6:9], Q

    pa, pg, _ = predict(np.r_[np.zeros(3), r0, bg0, ba0])
    sa = max(np.sqrt(((acc - pa) ** 2).mean()), 1e-3)
    sg = max(np.sqrt(((gyr - pg) ** 2).mean()), 1e-4)
    w = 1.0 / np.sqrt(inflate)

    def fun(p):
        pa, pg, _ = predict(p)
        parts = [w * ((gyr - pg) / sg).ravel(), w * ((acc - pa) / sa).ravel()]
        if prior is not None:
            parts.append((p[3:6] - prior[0]) / prior[1])
        return np.concatenate(parts)

    p0 = np.r_[np.zeros(3), r0, bg0, ba0]
    if fix_rot:     # orientation held at Q0; solve r and biases only
        sol = least_squares(lambda x: fun(np.r_[np.zeros(3), x]), p0[3:], method="lm", x_scale="jac")
        p = np.r_[np.zeros(3), sol.x]
        sol.jac = np.hstack([np.zeros((sol.jac.shape[0], 3)), sol.jac])
    else:
        sol = least_squares(fun, p0, method="lm", x_scale="jac")
        p = sol.x
    pa, pg, Q = predict(p)
    rms_a = float(np.sqrt(((acc - pa) ** 2).mean()))
    rms_g = float(np.sqrt(((gyr - pg) ** 2).mean()))
    # re-scale with final residual level for the covariance
    J = sol.jac.copy()
    n_data = 6 * len(acc)
    J[: 3 * len(acc)] *= sg / max(rms_g, 1e-9)
    J[3 * len(acc): n_data] *= sa / max(rms_a, 1e-9)
    free = slice(3, 12) if fix_rot else slice(0, 12)
    std = np.zeros(12)
    try:
        cov = np.linalg.inv(J[:, free].T @ J[:, free])
        std[free] = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        std[free] = np.nan
    return dict(Q=Q, r=p[3:6], bg=p[6:9], ba=p[9:12], rms_a=rms_a, rms_g=rms_g,
                r_std=std[3:6], rot_std_deg=np.degrees(std[:3]))


# ---------------------------------------------------------------- driver
def _ranking(cands):
    return [dict(segment=c["segment"], rel_gyro_residual=round(c["rel_gyro_residual"], 4),
                 offset_s=round(c["offset"], 4),
                 norm_corr=None if not np.isfinite(c["corr"]) else round(c["corr"], 4))
            for c in cands]


def align(seg: SegmentKin, imu_f: ImuData, offset: float, trim: float):
    """IMU resampled onto mocap frames (overlapping part, edges trimmed)."""
    t_imu_q = seg.t - offset
    ok = (t_imu_q >= imu_f.t[0] + trim) & (t_imu_q <= imu_f.t[-1] - trim)
    ok &= (seg.t >= seg.t[0] + trim) & (seg.t <= seg.t[-1] - trim)
    idx = np.nonzero(ok)[0]
    if len(idx) < 50:
        raise ValueError("too little overlap between mocap and IMU after sync")
    acc = interp_rows(imu_f.t, imu_f.acc, t_imu_q[idx])
    gyr = interp_rows(imu_f.t, imu_f.gyr, t_imu_q[idx])
    return _slice(seg, idx), acc, gyr


def locate_imu(segments: dict[str, SegmentKin], imu: ImuData, fc: float | None = 6.0,
               segment: str | None = None, max_lag: float | None = 0.3,
               offset_guess: float = 0.0, fixed_offset: float | None = None,
               prior=None, trim: float = 0.5, n_chunks: int = 5,
               lever_from_gyro: bool = True, orientation: str = "joint", acc_sign="auto",
               refine_orientation: bool = True, sat_acc: float | None = None,
               sat_gyr: float | None = None, sat_pad: float = 0.15) -> ImuEstimate:
    """sat_acc [m/s^2] / sat_gyr [rad/s]: per-axis sensor range. Samples at the
    limit (padded by sat_pad seconds) are excluded from the fit."""
    fs_imu = sample_rate(imu.t)
    valid = None
    if sat_acc or sat_gyr:
        bad = np.zeros(len(imu.t), bool)
        if sat_acc:
            bad |= (np.abs(imu.acc) >= sat_acc).any(1)
        if sat_gyr:
            bad |= (np.abs(imu.gyr) >= sat_gyr).any(1)
        n = int(round(sat_pad * fs_imu))
        if bad.any() and n > 0:
            bad = np.convolve(bad.astype(float), np.ones(2 * n + 1), mode="same") > 0
        valid = ~bad
    imu_f = ImuData(imu.t, lowpass(imu.acc, fs_imu, fc), lowpass(imu.gyr, fs_imu, fc))
    names = [segment] if segment else list(segments)

    cands = []
    for name in names:
        seg = segments[name]
        if fixed_offset is not None:
            off, corr = fixed_offset, np.nan
        else:
            off, corr = estimate_time_offset(seg.t, seg.omega, imu_f.t, imu_f.gyr,
                                             max_lag=max_lag, guess=offset_guess)
        s, acc, gyr = align(seg, imu_f, off, trim)
        Q, bg, info = fit_orientation(s.omega, gyr)
        cands.append(dict(segment=name, offset=off, corr=corr, rel_gyro_residual=info["rel"],
                          _Q=Q, _bg=bg, _info=info))
    cands.sort(key=lambda c: c["rel_gyro_residual"])
    best = cands[0]
    seg = segments[best["segment"]]
    warnings = []

    # sub-sample offset refinement on the gyro residual
    off = best["offset"]
    if fixed_offset is None:
        dt = 1.0 / sample_rate(seg.t)
        grid = off + np.linspace(-2 * dt, 2 * dt, 41)
        errs = []
        for o in grid:
            s, _, gyr = align(seg, imu_f, o, trim)
            errs.append(fit_orientation(s.omega, gyr)[2]["rel"])
        off = float(grid[int(np.argmin(errs))])

    s, acc, gyr = align(seg, imu_f, off, trim)
    Q, bg, info = fit_orientation(s.omega, gyr)
    if info["rel_improper"] < 0.7 * info["rel"]:
        warnings.append("an improper (mirrored) axis set fits the gyro much better: "
                        "check IMU axis signs / handedness")
    if info["rel"] > 0.35:
        warnings.append(f"poor gyro fit (relative residual {info['rel']:.2f}): wrong segment, "
                        "bad sync, units, or mocap/IMU mismatch")
    sv = info["omega_sv"]
    if sv[2] < 0.05 * sv[0]:
        warnings.append("segment rotation is nearly planar: orientation about the dominant "
                        "rotation axis relies on the accelerometer (gravity) term")

    if acc_sign == "auto":
        # full fit both ways; the physically right sign gives the smaller accel residual
        kw = dict(fc=fc, segment=best["segment"], fixed_offset=off, prior=prior, trim=trim,
                  n_chunks=n_chunks, lever_from_gyro=lever_from_gyro, orientation=orientation,
                  refine_orientation=refine_orientation, sat_acc=sat_acc, sat_gyr=sat_gyr,
                  sat_pad=sat_pad)
        e_pos = locate_imu(segments, imu, acc_sign=1, **kw)
        e_neg = locate_imu(segments, imu, acc_sign=-1, **kw)
        chosen = e_pos if e_pos.rms_acc <= e_neg.rms_acc else e_neg
        chosen.candidates = _ranking(cands)
        chosen.time_offset = off
        chosen.rms_acc_by_sign = {1: e_pos.rms_acc, -1: e_neg.rms_acc}
        if chosen is e_neg:
            chosen.warnings.append(f"accelerometer is sign-inverted relative to the gyro "
                                   f"(acc rms {e_neg.rms_acc:.2f} inverted vs {e_pos.rms_acc:.2f} as-is); using -acc")
        return chosen
    acc = acc_sign * acc
    imu_f = ImuData(imu_f.t, acc_sign * imu_f.acc, imu_f.gyr)
    # everything that filters or differentiates runs on the contiguous signal first ...
    fs_m = sample_rate(s.t)
    dgyr = np.gradient(gyr, s.t, axis=0) if lever_from_gyro else None
    f_lp, a_lp = lowpass(s.f, fs_m, 1.5), lowpass(acc, fs_m, 1.5)
    # ... then frames near IMU saturation are dropped
    sat_frac = 0.0
    if valid is not None:
        keep = np.interp(s.t - off, imu.t, valid.astype(float)) > 0.999
        sat_frac = 1.0 - keep.mean()
        if keep.sum() < 200:
            raise ValueError("almost all frames are near IMU saturation")
        if sat_frac > 0:
            s = _slice(s, keep)
            acc, gyr, f_lp, a_lp = acc[keep], gyr[keep], f_lp[keep], a_lp[keep]
            dgyr = None if dgyr is None else dgyr[keep]
            warnings.append(f"excluded {100 * sat_frac:.1f}% of frames near IMU saturation")
    if orientation == "joint":
        Q, bg, jinfo = fit_orientation_joint(s.omega, gyr, f_lp, a_lp, fs_m, prefiltered=True)
    if lever_from_gyro:
        r, ba, ksv = fit_position(s, acc, Q, omega=(gyr - bg) @ Q.T, alpha=dgyr @ Q.T)
    else:
        r, ba, ksv = fit_position(s, acc, Q)
    fs = sample_rate(seg.t)
    inflate = max(1.0, fs / (2 * fc)) if fc else 1.0
    ref = refine(s, acc, gyr, Q, r, bg, ba, prior=prior, inflate=inflate, dgyr=dgyr,
                 fix_rot=not refine_orientation)

    # chunk consistency (independent estimates on contiguous chunks)
    r_chunks, R_chunks = [], []
    if n_chunks and n_chunks > 1:
        for idx in np.array_split(np.arange(len(s.t)), n_chunks):
            if len(idx) < 200:
                continue
            sc = _slice(s, idx)
            rc = refine(sc, acc[idx], gyr[idx], ref["Q"], ref["r"], ref["bg"], ref["ba"],
                        prior=prior, inflate=inflate,
                        dgyr=None if dgyr is None else dgyr[idx], fix_rot=not refine_orientation)
            r_chunks.append(rc["r"])
            R_chunks.append(rc["Q"])
    r_chunk_std = rot_chunk_std = None
    if len(r_chunks) >= 3:
        r_chunk_std = np.std(r_chunks, axis=0, ddof=1)
        ang = [np.degrees(Rot.from_matrix(ref["Q"].T @ Rc).magnitude()) for Rc in R_chunks]
        rot_chunk_std = float(np.sqrt(np.mean(np.square(ang))))

    worst = np.nanmax(np.maximum(ref["r_std"], r_chunk_std if r_chunk_std is not None else 0))
    if worst > 0.02:
        ax = "xyz"[int(np.nanargmax(np.maximum(ref["r_std"], r_chunk_std if r_chunk_std is not None else 0)))]
        warnings.append(f"position poorly determined along segment {ax}-axis (~{worst * 100:.1f} cm): "
                        "add richer 3D movements or a prior (IMU sits on the skin surface)")

    ranking = _ranking(cands)
    return ImuEstimate(segment=best["segment"], time_offset=off, R=ref["Q"], r=ref["r"],
                       gyro_bias=ref["bg"], acc_bias=ref["ba"], rms_gyro=ref["rms_g"],
                       rms_acc=ref["rms_a"], r_std=ref["r_std"], rot_std_deg=ref["rot_std_deg"],
                       r_chunk_std=r_chunk_std, rot_chunk_std_deg=rot_chunk_std,
                       candidates=ranking, warnings=warnings, acc_sign=int(acc_sign),
                       rel_acc=ref["rms_a"] / max(np.sqrt(acc.var(0).sum() / 3), 1e-9),
                       sat_frac=float(sat_frac))
