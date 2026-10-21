#!/usr/bin/env python3
"""
exo_frame.py

One-file frame transform between the Georgia Tech ankle biomechanics data
(Hugging Face `aicognition/mrsd-exo-ankle`) and the axilles exoskeleton's own
sensors. Import it from a data ingester for training, and from the controller for
deployment; both produce the same feature vector.

    from exo_frame import ExoFrame

    # ---- offline, once: fit the calibration and save it ----
    cal = ExoFrame.fit("mrsd-exo-ankle", "Data collection/data")
    cal.save("exo_frame_calibration.json")

    # ---- training: Georgia Tech data -> exo frame ----
    cal = ExoFrame.load("exo_frame_calibration.json")
    X, y = cal.transform_hf_trial("mrsd-exo-ankle", "AB06", "treadmill_01_01")

    # ---- deployment: one sample, no allocation ----
    cal = ExoFrame.load("exo_frame_calibration.json")
    cal.set_encoder_zero(raw_encoder_deg_at_run_start)
    x = cal.features(foot_a, foot_g, shank_a, shank_g, enc_deg, toe_fsr, heel_fsr)

CLI:
    python exo_frame.py fit          # fit and save the calibration
    python exo_frame.py build        # convert every downloaded subject to parquet
    python exo_frame.py bench        # measure deployment-path latency

Deployment note: the transform maps Georgia Tech data INTO the exo's frame, so the
robot's own samples need no rotation at run time. The hot path is therefore just
unit handling, an encoder offset and two FSR comparisons, and it neither allocates
nor imports pandas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

__all__ = ["ExoFrame", "FEATURES", "TARGET", "G_TO_MS2"]


# ========================= Constants =========================
# Georgia Tech accelerometers report g; the BNO085 reports m/s^2. Gyros are rad/s
# on both sides. Measured: HF trunk/shank/thigh |mean accel| = 1.01/1.03/1.02 g,
# exo foot/shank = 11.2/9.6 m/s^2.
G_TO_MS2 = 9.80665

SEGMENTS = ("foot", "shank")

# Feature layout. Training and deployment must agree on this exactly, which is the
# whole reason both paths live in one file.
FEATURES = (
    "foot_ax", "foot_ay", "foot_az",
    "foot_gx", "foot_gy", "foot_gz",
    "shank_ax", "shank_ay", "shank_az",
    "shank_gx", "shank_gy", "shank_gz",
    "ankle_angle_deg",
    "heel_contact",
    "toe_contact",
)
N_FEATURES = len(FEATURES)
TARGET = "ankle_moment_nm_per_kg"

# Ground contact thresholds, from TBE_controller/utilities.py, so offline strides
# and the online controller mean the same thing.
HEEL_STRIKE_THRESHOLD = 10000.0
TOE_OFF_THRESHOLD = 20000.0
FSR_FILTER_CUTOFF = 20.0

# Sampling rate the exo logs at and the control loop runs at.
TARGET_HZ = 200.0

GAIT_CYCLE_POINTS = 101
MIN_STRIDE_S, MAX_STRIDE_S = 0.5, 2.5

# A fit further than this from the consensus is a failed fit, not spread to average
# in. Real per-recording variation is a few degrees.
OUTLIER_REJECT_DEG = 30.0


# ========================= Small math helpers =========================
def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("Cannot normalise a zero vector.")
    return v / n


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))))


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle in degrees between two rotation matrices."""
    c = np.clip((np.trace(np.asarray(a) @ np.asarray(b).T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def is_rotation(r: np.ndarray, tol: float = 1e-6) -> bool:
    """True if r is a proper rotation: orthonormal with determinant +1."""
    r = np.asarray(r, dtype=np.float64)
    return (r.shape == (3, 3)
            and np.allclose(r.T @ r, np.eye(3), atol=tol)
            and abs(np.linalg.det(r) - 1.0) < tol)


def kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Least-squares rotation R minimising ||R p - q||_F over (3, N) point sets.

    The determinant correction keeps the result a rotation rather than a reflection.
    That matters here: a reflection would mirror the exo's left/right sense and
    silently invert the sagittal channel.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape or p.shape[0] != 3:
        raise ValueError(f"Expected matching (3, N) inputs, got {p.shape} and {q.shape}.")
    u, _, vt = np.linalg.svd(p @ q.T)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, float(np.linalg.norm(r @ p - q) / np.sqrt(p.shape[1]))


def principal_axis(gyro: np.ndarray) -> tuple[np.ndarray, float]:
    """Dominant angular-velocity direction and the variance fraction it explains."""
    g = np.asarray(gyro, dtype=np.float64)
    g = g - g.mean(axis=0)
    w, v = np.linalg.eigh(g.T @ g / len(g))
    axis = v[:, -1]
    if axis[int(np.argmax(np.abs(axis)))] < 0.0:
        axis = -axis
    return _unit(axis), float(w[-1] / w.sum())


def _chordal_mean(rots: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of rotations, projected back onto SO(3)."""
    u, _, vt = np.linalg.svd(np.mean(np.stack(rots), axis=0))
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def average_rotations(rots: Sequence[np.ndarray],
                      reject_deg: float = OUTLIER_REJECT_DEG
                      ) -> tuple[np.ndarray, list[int], float]:
    """
    Outlier-rejecting mean of several rotation estimates.

    Averaging rotations is only meaningful if they agree. A fit that landed on the
    wrong phase alignment sits ~170 deg away, and a plain mean drags the result
    somewhere matching no recording at all - invisible afterwards, because the
    average is still a valid rotation. So: provisional mean, discard anything beyond
    `reject_deg`, re-average the survivors.
    """
    rots = list(rots)
    if not rots:
        raise ValueError("No rotations to average.")
    if len(rots) == 1:
        return rots[0], [], 0.0

    provisional = _chordal_mean(rots)
    dev = [rotation_angle(r, provisional) for r in rots]
    keep = [i for i, d in enumerate(dev) if d <= reject_deg]

    # If the provisional mean was itself dragged into no-man's-land, everything can
    # look like an outlier. Fall back to the largest mutually-consistent group.
    if not keep:
        best: list[int] = []
        for i, ref in enumerate(rots):
            grp = [j for j, r in enumerate(rots) if rotation_angle(r, ref) <= reject_deg]
            if len(grp) > len(best):
                best = grp
        keep = best

    mean = _chordal_mean([rots[i] for i in keep])
    spread = max(rotation_angle(rots[i], mean) for i in keep)
    return mean, [i for i in range(len(rots)) if i not in keep], float(spread)


# ========================= Gait cycle =========================
def stride_bounds(heel_strikes: np.ndarray, n: int, fs: float) -> list[tuple[int, int]]:
    """Heel-strike-to-heel-strike index pairs of plausible duration."""
    hs = np.asarray(heel_strikes, dtype=int)
    lo, hi = int(MIN_STRIDE_S * fs), int(MAX_STRIDE_S * fs)
    return [(int(a), int(b)) for a, b in zip(hs[:-1], hs[1:])
            if b <= n and lo <= (b - a) <= hi]


def mean_cycle(sig: np.ndarray, bounds: list[tuple[int, int]],
               n_points: int = GAIT_CYCLE_POINTS) -> np.ndarray:
    """
    Stride-averaged waveform on a 0-100% phase axis.

    This is what makes the two datasets comparable at all: they share no clock,
    subject, speed or frame, but both are right-leg walking, so heel strike is a
    common phase reference.
    """
    sig = np.asarray(sig, dtype=np.float64)
    phase = np.linspace(0.0, 1.0, n_points)
    out = []
    for a, b in bounds:
        seg = sig[a:b]
        if len(seg) < 3:
            continue
        src = np.linspace(0.0, 1.0, len(seg))
        if seg.ndim == 1:
            out.append(np.interp(phase, src, seg))
        else:
            out.append(np.column_stack([np.interp(phase, src, seg[:, j])
                                        for j in range(seg.shape[1])]))
    if not out:
        raise ValueError("No usable strides to average over.")
    return np.mean(np.stack(out), axis=0)


def _stack(accel: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """
    Pack a mean gait cycle into a (3, 2N) matrix for Kabsch.

    Each block is scaled to unit Frobenius norm so the fit is driven by direction
    rather than magnitude. Without this the ~10 m/s^2 gravity term swamps the
    ~2 rad/s gyro term and the two datasets' different walking speeds bias the fit.
    """
    a, g = np.asarray(accel).T, np.asarray(gyro).T
    na, ng = np.linalg.norm(a), np.linalg.norm(g)
    if na == 0.0 or ng == 0.0:
        raise ValueError("Degenerate gait cycle: an accel or gyro block is all zero.")
    return np.hstack([a / na, g / ng])


def _fit_lag_and_rotation(hf_a: np.ndarray, hf_g: np.ndarray,
                          ex_a: np.ndarray, ex_g: np.ndarray
                          ) -> tuple[int, np.ndarray, float]:
    """
    Choose phase offset and rotation together, by search over all lags.

    The exo's FSR-derived heel strike fires ~30% of a stride away from the Georgia
    Tech kinematic one, so the cycles must be aligned before fitting.

    Estimating the lag first from a rotation-invariant proxy does NOT work, and
    failed on real data here: |gyro| discards sign, the foot's profile has a
    near-symmetric double peak, and on one pair the correlation peak landed at
    r = -0.71 and produced a rotation 170 deg from truth while looking like a clean
    fit. Scoring each lag by the Kabsch residual it produces - the quantity actually
    being minimised - removes that failure mode for ~100 small SVDs.
    """
    p = _stack(hf_a, hf_g)
    n = hf_g.shape[0]
    best = None
    for lag in range(n):
        r, res = kabsch(p, _stack(np.roll(ex_a, lag, axis=0), np.roll(ex_g, lag, axis=0)))
        if best is None or res < best[2]:
            best = (lag, r, res)
    lag, r, res = best
    return (lag - n if lag > n // 2 else lag), r, res


# ========================= Exo log reading =========================
_IMU_COLS = tuple(f"{s}_{a}{x}" for s in SEGMENTS for a in ("a", "g") for x in "xyz")


def _lowpass(x: np.ndarray, fs: float, cutoff: float = FSR_FILTER_CUTOFF) -> np.ndarray:
    """First-order low pass, matching TBE_controller/data_obtainer.lowPassFilter."""
    dt = 1.0 / fs
    a = 2 * np.pi * cutoff * dt / (2 * np.pi * cutoff * dt + 1)
    y = np.empty_like(x, dtype=np.float64)
    acc = float(x[0])
    for i in range(len(x)):
        acc = a * float(x[i]) + (1.0 - a) * acc
        y[i] = acc
    return y


def encoder_zero_from_log(df) -> tuple[float, int, float]:
    """
    Recover one recording's encoder zero from its startup window.

    The exo is zeroed at the start of every run, so each file has its own reference.
    The logger writes the latest cached value every tick, so at the head of the file
    the encoder is already reporting while the IMUs have not yet produced a packet.
    Those rows are the pose the run was zeroed at.

    Returns (zero_deg, n_samples, std_deg). n is typically 3-7 and std under 0.15
    deg. It is a short window - about 40 ms - so the caller should look at n and std
    rather than trust the number blindly; see the note in `fit` about the one
    recording where this goes wrong.
    """
    first_imu = df["foot_ax"].notna().to_numpy()
    lead = int(np.argmax(first_imu)) if first_imu.any() else len(df)
    window = df["ankle_encoder_deg"].iloc[:lead].dropna()
    if len(window) == 0:
        # Nothing in the blank window: fall back to the first reading available.
        window = df["ankle_encoder_deg"].dropna().iloc[:1]
    if len(window) == 0:
        raise ValueError("No encoder samples at all in this recording.")
    return float(window.mean()), int(len(window)), float(window.std(ddof=0) or 0.0)


def read_exo_log(path, fsr_mode: str = "fixed", adaptive_fraction: float = 0.5) -> dict:
    """
    Read one `data_collection_*.csv` into plain arrays.

    Offline only - imports pandas. The deployment path never calls this.

    `fsr_mode="adaptive"` places contact thresholds between each recording's own
    loaded and unloaded levels instead of using the controller's absolute counts.
    Use it when a log was taken with different FSR seating than the thresholds were
    tuned for; see the note in `fit`.
    """
    import pandas as pd

    path = Path(path)
    raw = pd.read_csv(path)
    zero, zero_n, zero_std = encoder_zero_from_log(raw)

    t = raw["timestamp_s"].to_numpy(dtype=np.float64)
    fs = float(1.0 / np.median(np.diff(t)))

    df = raw.interpolate(limit_direction="both")
    accel = {s: df[[f"{s}_a{x}" for x in "xyz"]].to_numpy(np.float64) for s in SEGMENTS}
    gyro = {s: df[[f"{s}_g{x}" for x in "xyz"]].to_numpy(np.float64) for s in SEGMENTS}

    heel = _lowpass(df["heel_fsr_raw"].to_numpy(np.float64), fs)
    toe = _lowpass(df["toe_fsr_raw"].to_numpy(np.float64), fs)

    if fsr_mode == "adaptive":
        def level(x):
            lo, hi = np.percentile(x, [10.0, 90.0])
            return float(lo + adaptive_fraction * (hi - lo))
        heel_thr, toe_thr = level(heel), level(toe)
    elif fsr_mode == "fixed":
        heel_thr, toe_thr = HEEL_STRIKE_THRESHOLD, TOE_OFF_THRESHOLD
    else:
        raise ValueError(f"Unknown fsr_mode {fsr_mode!r}; expected 'fixed' or 'adaptive'.")

    heel_contact = (heel > heel_thr).astype(np.float64)
    toe_contact = (toe >= toe_thr).astype(np.float64)
    hs = np.where(np.diff(heel_contact.astype(int)) == 1)[0] + 1

    return {
        "name": path.stem, "fs": fs, "time": t,
        "accel": accel, "gyro": gyro,
        "encoder_raw_deg": df["ankle_encoder_deg"].to_numpy(np.float64),
        "encoder_zero_deg": zero, "encoder_zero_n": zero_n, "encoder_zero_std": zero_std,
        "heel_contact": heel_contact, "toe_contact": toe_contact,
        "heel_strikes": hs, "heel_threshold": heel_thr, "toe_threshold": toe_thr,
    }


def read_hf_trial(root, subject: str, trial: str) -> dict:
    """Read one Georgia Tech trial in its native frames and units. Offline only."""
    import pandas as pd

    root = Path(root)
    base = root / "subjects" / subject
    imu = pd.read_parquet(base / f"{trial}__imu.parquet")
    idf = pd.read_parquet(base / f"{trial}__id.parquet")
    gon = pd.read_parquet(base / f"{trial}__gon.parquet")
    gc = pd.read_parquet(base / f"{trial}__gcRight.parquet")

    t = imu["time_s"].to_numpy(np.float64)
    n = len(t)

    # HeelStrike/ToeOff are 0-100% ramps that wrap at the event; a large negative
    # jump is the event. The -50 threshold is the dataset card's own convention.
    hs = np.where(np.diff(gc["HeelStrike"].to_numpy()) < -50.0)[0] + 1
    to = np.where(np.diff(gc["ToeOff"].to_numpy()) < -50.0)[0] + 1

    heel_contact = np.zeros(n)
    toe_contact = np.zeros(n)
    for i in hs:
        nxt = to[to > i]
        end = int(nxt[0]) if len(nxt) else n
        heel_contact[i:end] = 1.0
        toe_contact[i + (end - i) // 3:end] = 1.0

    meta = pd.read_parquet(root / "metadata.parquet")
    row = meta[(meta["subject"] == subject) & (meta["trial"] == trial)]

    return {
        "name": f"{subject}/{trial}", "fs": float(1.0 / np.median(np.diff(t))), "time": t,
        "accel": {s: imu[[f"{s}_Accel_{x}" for x in "XYZ"]].to_numpy(np.float64)
                  for s in SEGMENTS},
        "gyro": {s: imu[[f"{s}_Gyro_{x}" for x in "XYZ"]].to_numpy(np.float64)
                 for s in SEGMENTS},
        "ankle_angle_deg": np.interp(t, gon["time_s"].to_numpy(),
                                     gon["ankle_sagittal"].to_numpy()),
        "ankle_moment_nm": idf["ankle_angle_r_moment"].to_numpy(np.float64),
        "heel_contact": heel_contact, "toe_contact": toe_contact,
        "heel_strikes": hs, "toe_offs": to,
        "mass_kg": float(row["weight_kg"].iloc[0]) if len(row) else None,
    }


def list_hf_trials(root, subject: str) -> list[str]:
    d = Path(root) / "subjects" / subject
    if not d.is_dir():
        raise FileNotFoundError(f"No such subject directory: {d}")
    return sorted({p.name.split("__")[0] for p in d.glob("*__imu.parquet")})


# ========================= The transform =========================
class ExoFrame:
    """
    Maps Georgia Tech sensor axes onto the exoskeleton's, and builds feature vectors.

    Hold one instance for the life of a run. `features` is the deployment hot path:
    no rotation is needed there because the transform already brought the training
    data into the exo's frame, so it is only unit handling, an encoder offset and
    two comparisons. It does not allocate when handed an output buffer.
    """

    __slots__ = ("rotations", "encoder_zero_deg", "encoder_sign", "encoder_ratio",
                 "heel_threshold", "toe_threshold", "meta", "_rot_f", "_rot_s")

    def __init__(self,
                 rotations: dict[str, np.ndarray],
                 encoder_zero_deg: Optional[float] = None,
                 encoder_sign: int = 1,
                 encoder_ratio: float = 1.0,
                 heel_threshold: float = HEEL_STRIKE_THRESHOLD,
                 toe_threshold: float = TOE_OFF_THRESHOLD,
                 meta: Optional[dict] = None):
        for seg in SEGMENTS:
            if seg not in rotations:
                raise KeyError(f"No rotation supplied for segment {seg!r}.")
            if not is_rotation(np.asarray(rotations[seg], dtype=np.float64)):
                raise ValueError(f"Rotation for {seg!r} is not orthonormal with det=+1.")

        self.rotations = {s: np.ascontiguousarray(rotations[s], dtype=np.float64)
                          for s in SEGMENTS}
        # Cached for the hot path so attribute lookups stay off the critical line.
        self._rot_f = self.rotations["foot"]
        self._rot_s = self.rotations["shank"]

        self.encoder_zero_deg = encoder_zero_deg
        self.encoder_sign = int(encoder_sign)
        self.encoder_ratio = float(encoder_ratio)
        self.heel_threshold = float(heel_threshold)
        self.toe_threshold = float(toe_threshold)
        self.meta = meta or {}

    # ---------------- deployment ----------------
    def set_encoder_zero(self, raw_deg: float) -> None:
        """
        Set the run's encoder reference, read at the moment the exo is zeroed.

        The AS5600 is absolute but arbitrarily clocked to the joint and the exo is
        re-zeroed every run, so this belongs to the run, not to the calibration.
        """
        self.encoder_zero_deg = float(raw_deg)

    def ankle_angle(self, raw_deg: float) -> float:
        """Raw AS5600 degrees -> ankle angle, dorsiflexion positive."""
        if self.encoder_zero_deg is None:
            raise ValueError(
                "Encoder zero is not set. The AS5600 is absolute but re-zeroed every "
                "run, so call set_encoder_zero() with the reading taken at run start "
                "(or let read_exo_log recover it from the log's startup window)."
            )
        # Unwrap the 0/360 seam so a zero near the wrap point still behaves.
        d = (raw_deg - self.encoder_zero_deg + 180.0) % 360.0 - 180.0
        return self.encoder_sign * self.encoder_ratio * d

    def features(self,
                 foot_accel: Sequence[float], foot_gyro: Sequence[float],
                 shank_accel: Sequence[float], shank_gyro: Sequence[float],
                 encoder_deg: float, toe_fsr: float, heel_fsr: float,
                 out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Build one feature vector from live exo samples. Deployment hot path.

        Inputs are the exo's own units: accel m/s^2, gyro rad/s, encoder raw degrees,
        FSRs raw counts (filter them upstream the way the controller does). Pass
        `out` - a (15,) float64 array - to avoid allocating per sample.
        """
        v = np.empty(N_FEATURES, dtype=np.float64) if out is None else out
        v[0] = foot_accel[0]; v[1] = foot_accel[1]; v[2] = foot_accel[2]
        v[3] = foot_gyro[0];  v[4] = foot_gyro[1];  v[5] = foot_gyro[2]
        v[6] = shank_accel[0]; v[7] = shank_accel[1]; v[8] = shank_accel[2]
        v[9] = shank_gyro[0];  v[10] = shank_gyro[1]; v[11] = shank_gyro[2]
        v[12] = self.ankle_angle(encoder_deg)
        v[13] = 1.0 if heel_fsr > self.heel_threshold else 0.0
        v[14] = 1.0 if toe_fsr >= self.toe_threshold else 0.0
        return v

    def features_batch(self, foot_accel, foot_gyro, shank_accel, shank_gyro,
                       encoder_deg, toe_fsr, heel_fsr) -> np.ndarray:
        """Vectorised `features` over (N, 3) blocks and (N,) scalars. Returns (N, 15)."""
        if self.encoder_zero_deg is None:
            raise ValueError("Encoder zero is not set; call set_encoder_zero() first.")
        n = len(encoder_deg)
        out = np.empty((n, N_FEATURES), dtype=np.float64)
        out[:, 0:3] = foot_accel
        out[:, 3:6] = foot_gyro
        out[:, 6:9] = shank_accel
        out[:, 9:12] = shank_gyro
        d = (np.asarray(encoder_deg, np.float64) - self.encoder_zero_deg + 180.0) % 360.0 - 180.0
        out[:, 12] = self.encoder_sign * self.encoder_ratio * d
        out[:, 13] = np.asarray(heel_fsr) > self.heel_threshold
        out[:, 14] = np.asarray(toe_fsr) >= self.toe_threshold
        return out

    # ---------------- training ----------------
    def rotate_hf(self, accel_g: np.ndarray, gyro: np.ndarray,
                  segment: str) -> tuple[np.ndarray, np.ndarray]:
        """Rotate one segment's HF data into the exo frame, lifting accel g -> m/s^2."""
        r = self.rotations[segment]
        a = np.asarray(accel_g, np.float64) * G_TO_MS2
        return a @ r.T, np.asarray(gyro, np.float64) @ r.T

    def transform_hf(self, trial: dict) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        A loaded HF trial -> (X, y) in the exo's frame and units.

        X follows FEATURES exactly, so it is interchangeable with `features_batch`
        output. `ankle_angle_deg` comes from the Georgia Tech goniometer, which is
        already an anatomical angle and so needs the sign convention but no rotation.
        y is the right-ankle moment normalised by body mass, or None if absent.
        """
        n = len(trial["time"])
        x = np.empty((n, N_FEATURES), dtype=np.float64)

        fa, fg = self.rotate_hf(trial["accel"]["foot"], trial["gyro"]["foot"], "foot")
        sa, sg = self.rotate_hf(trial["accel"]["shank"], trial["gyro"]["shank"], "shank")
        x[:, 0:3], x[:, 3:6] = fa, fg
        x[:, 6:9], x[:, 9:12] = sa, sg
        x[:, 12] = self.encoder_sign * trial["ankle_angle_deg"]
        x[:, 13] = trial["heel_contact"]
        x[:, 14] = trial["toe_contact"]

        y = None
        if trial.get("ankle_moment_nm") is not None and trial.get("mass_kg"):
            y = trial["ankle_moment_nm"] / trial["mass_kg"]
        return x, y

    def transform_hf_trial(self, root, subject: str, trial: str):
        """Convenience: read a trial from disk and transform it in one call."""
        return self.transform_hf(read_hf_trial(root, subject, trial))

    # ---------------- persistence ----------------
    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "rotations": {s: self.rotations[s].tolist() for s in SEGMENTS},
            "encoder_sign": self.encoder_sign,
            "encoder_ratio": self.encoder_ratio,
            "heel_threshold": self.heel_threshold,
            "toe_threshold": self.toe_threshold,
            "features": list(FEATURES),
            "target": TARGET,
            "meta": self.meta,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ExoFrame":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if list(d.get("features", FEATURES)) != list(FEATURES):
            raise ValueError("Calibration was saved with a different feature layout.")
        return cls(rotations={s: np.array(d["rotations"][s], np.float64) for s in SEGMENTS},
                   encoder_sign=d.get("encoder_sign", 1),
                   encoder_ratio=d.get("encoder_ratio", 1.0),
                   heel_threshold=d.get("heel_threshold", HEEL_STRIKE_THRESHOLD),
                   toe_threshold=d.get("toe_threshold", TOE_OFF_THRESHOLD),
                   meta=d.get("meta", {}))

    # ---------------- fitting ----------------
    @classmethod
    def fit(cls,
            hf_root="mrsd-exo-ankle",
            exo_dir="Data collection/data",
            reference_subject: str = "AB06",
            max_reference_trials: int = 3,
            fsr_mode: str = "fixed",
            min_exo_fs: float = 150.0,
            exclude: Sequence[str] = (),
            verbose: bool = True) -> "ExoFrame":
        """
        Fit the HF -> exo rotations from real recordings.

        Method: average each signal over the gait cycle on both sides, then solve for
        the rotation and the phase offset jointly (see `_fit_lag_and_rotation`).
        Every available (HF trial, exo recording) pair is fitted and the results are
        combined with outlier rejection, because a tight spread across independent
        pairs is the main evidence the fit describes the hardware rather than one
        recording's quirks.

        `exclude` drops exo recordings by filename substring. Note that
        `data_collection_20260404_214949` is worth excluding: its encoder spans 99 deg
        with a zero of 169 deg, against 12-29 deg spans and 79-81 deg zeros in the
        other four, which suggests it wrapped the 0/360 seam or the magnet moved.
        """
        hf_root = Path(hf_root)
        exo_paths = [p for p in sorted(Path(exo_dir).glob("data_collection_*.csv"))
                     if not any(e in p.name for e in exclude)]

        exos = []
        for p in exo_paths:
            log = read_exo_log(p, fsr_mode=fsr_mode)
            if log["fs"] < min_exo_fs:
                continue
            if len(stride_bounds(log["heel_strikes"], len(log["time"]), log["fs"])) >= 3:
                exos.append(log)
        if not exos:
            raise RuntimeError(f"No exo recording in {exo_dir} has enough clean strides.")

        hfs = [read_hf_trial(hf_root, reference_subject, t)
               for t in list_hf_trials(hf_root, reference_subject)[:max_reference_trials]]
        if not hfs:
            raise RuntimeError(f"No trials found for {reference_subject}.")

        rotations, notes = {}, []
        for seg in SEGMENTS:
            cands, labels, residuals = [], [], []
            for hf in hfs:
                hb = stride_bounds(hf["heel_strikes"], len(hf["time"]), hf["fs"])
                hf_a = mean_cycle(hf["accel"][seg] * G_TO_MS2, hb)
                hf_g = mean_cycle(hf["gyro"][seg], hb)
                for ex in exos:
                    eb = stride_bounds(ex["heel_strikes"], len(ex["time"]), ex["fs"])
                    ex_a = mean_cycle(ex["accel"][seg], eb)
                    ex_g = mean_cycle(ex["gyro"][seg], eb)
                    lag, r, res = _fit_lag_and_rotation(hf_a, hf_g, ex_a, ex_g)
                    cands.append(r)
                    residuals.append(res)
                    labels.append(f"{hf['name']} x {ex['name']}")

            mean_r, rejected, spread = average_rotations(cands)
            rotations[seg] = mean_r
            note = (f"{seg}: {len(cands) - len(rejected)}/{len(cands)} pairs, "
                    f"spread {spread:.2f} deg, best residual {min(residuals):.4f}")
            notes.append(note)
            if verbose:
                print(f"  {note}")
                for i in rejected:
                    print(f"    rejected outlier: {labels[i]}")

        obj = cls(rotations=rotations,
                  heel_threshold=exos[0]["heel_threshold"],
                  toe_threshold=exos[0]["toe_threshold"],
                  meta={"hf_root": str(hf_root), "reference_subject": reference_subject,
                        "exo_recordings": [e["name"] for e in exos],
                        "fsr_mode": fsr_mode, "notes": notes})

        sign, corr = obj._resolve_encoder_sign(hfs, exos, verbose=verbose)
        obj.encoder_sign = sign
        obj.meta["encoder_sign_corr"] = corr
        return obj

    def _resolve_encoder_sign(self, hfs: list[dict], exos: list[dict],
                              verbose: bool = True) -> tuple[int, float]:
        """
        Decide whether increasing raw encoder angle means dorsiflexion, from data.

        The exo's ankle angle and the Georgia Tech goniometer describe the same joint,
        so their gait-cycle waveforms should match once the sign is right. Both signs
        are scored by correlation against the HF `ankle_sagittal` cycle and the better
        one wins. This is a measurement, not an assumption - but a weak margin means
        the recordings do not settle it, and the caller is told so.
        """
        hf_cycles, ex_cycles = [], []
        for hf in hfs:
            hb = stride_bounds(hf["heel_strikes"], len(hf["time"]), hf["fs"])
            hf_cycles.append(mean_cycle(hf["ankle_angle_deg"], hb))
        for ex in exos:
            eb = stride_bounds(ex["heel_strikes"], len(ex["time"]), ex["fs"])
            centred = ex["encoder_raw_deg"] - ex["encoder_zero_deg"]
            ex_cycles.append(mean_cycle(centred, eb))

        hf_mean = np.mean(np.stack(hf_cycles), axis=0)
        hf_mean = hf_mean - hf_mean.mean()

        # The two heel-strike detectors differ, so compare at the best alignment.
        best = -2.0
        for ex_c in ex_cycles:
            e = ex_c - ex_c.mean()
            for lag in range(len(e)):
                r = float(np.corrcoef(hf_mean, np.roll(e, lag))[0, 1])
                best = max(best, abs(r))
                if abs(r) == best:
                    signed = r
        sign = 1 if signed > 0 else -1

        if verbose:
            msg = f"  encoder sign = {sign:+d} (|r| = {best:.3f} vs HF goniometer)"
            if best < 0.5:
                msg += "  <-- WEAK, verify against a calibration recording"
            print(msg)
        return sign, float(signed)


# ========================= Guided calibration =========================
#
# Everything below runs on the robot. The hardware imports are kept inside the
# functions that need them, so `import exo_frame` stays numpy-only on a laptop.

REPO_ROOT = Path(__file__).resolve().parent
CALIB_DIR = REPO_ROOT / "calibration"

# Quiet-standing acceptance. A subject who cannot hold below this is not standing
# still enough for the reading to serve as a zero.
STILL_GYRO_MAX = 0.35            # rad/s
STILL_MIN_FRACTION = 0.80        # of the standing window
STILL_MAX_ENCODER_SD = 1.5       # deg

# Sweep acceptance.
SWEEP_MIN_ROM_DEG = 12.0         # combined dorsi + plantar excursion
SWEEP_MIN_R2 = 0.75              # encoder rate vs IMU rate regression
SWEEP_MAX_SHANK_GYRO = 0.60      # rad/s mean; the shank is meant to stay put
SWEEP_MIN_HOLD_SEPARATION = 4.0  # deg between neutral and each held pose

# Hip-swing acceptance. The point of this phase is that with the knee and ankle
# held, foot and shank move as one rigid body, so both IMUs see the same physical
# angular velocity. That only holds if the ankle really did stay put.
HIPSWING_MAX_ENCODER_SD = 2.5      # deg; larger means the ankle moved and the
                                   # rigid-body assumption is broken
HIPSWING_MIN_GYRO = 0.50           # rad/s mean; smaller means the leg barely moved
HIPSWING_MAX_RESIDUAL = 0.25       # Kabsch residual as a fraction of |gyro|
HIPSWING_MIN_VAR_RATIO = 0.70      # the swing should be planar

# How far the two independent estimates of R_foot<-shank may disagree. One comes
# from standing gravity plus the swing axis, the other from Kabsch on the swing
# gyro alone; the latter is rank-limited, so this is generous by design.
HIPSWING_METHOD_TOL_DEG = 25.0

# Stability of the derived Georgia Tech foot<-shank relationship across calibration
# runs. That relationship is a property of their hardware, so it must not move.
GT_RELATION_DRIFT_DEG = 10.0

# Walking acceptance.
WALK_MIN_STRIDES = 20
WALK_MAX_ROTATION_SPREAD_DEG = 8.0
WALK_MIN_SAGITTAL_CORR = 0.70

# A measured IMU update rate below this means the report-rate override did not take.
MIN_EFFECTIVE_IMU_HZ = 100.0


@dataclass
class PhaseSpec:
    key: str
    seconds: float
    title: str
    instruction: str


# The protocol. Standing and walking are as specified. The sweep is split into
# labelled sub-phases because an unlabelled sweep cannot determine which direction
# is dorsiflexion: PCA recovers the axis but not its sign. The held poses make the
# sign a direct measurement, and the free sweep afterwards supplies the rate
# regression that gives the encoder-to-joint ratio and cross-checks that sign.
PROTOCOL = (
    PhaseSpec("standing", 10.0, "QUIET STANDING",
              "Stand still, weight even on both feet, ankle relaxed and neutral.\n"
              "Do not shift or sway. Look straight ahead."),
    PhaseSpec("neutral", 5.0, "NEUTRAL HOLD",
              "Sit down, or stand on your LEFT leg only.\n"
              "Let the right foot rest in its NEUTRAL position and hold it still."),
    PhaseSpec("dorsi", 5.0, "DORSIFLEXION HOLD",
              "Pull your toes UP toward your shin, as far as is comfortable.\n"
              "HOLD that position still until told to stop."),
    PhaseSpec("plantar", 5.0, "PLANTARFLEXION HOLD",
              "Point your toes DOWN and away from you, as far as is comfortable.\n"
              "HOLD that position still until told to stop."),
    PhaseSpec("sweep", 10.0, "SLOW SWEEPS",
              "Sweep the ankle smoothly up and down, about 5 full cycles.\n"
              "Keep it SLOW, and keep your shank still. Only the ankle moves."),
    PhaseSpec("hipswing", 10.0, "HIP SWINGS",
              "Stand on your LEFT leg, holding a support for balance.\n"
              "Swing the RIGHT leg forward and back from the HIP, about 5 cycles.\n"
              "Keep the knee STRAIGHT and the ankle STILL - the whole leg swings\n"
              "as one piece. Do not let the ankle flap."),
    PhaseSpec("walking", 60.0, "LEVEL WALKING",
              "Walk at a comfortable, steady pace on level ground.\n"
              "Keep walking until told to stop."),
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    critical: bool = True


@dataclass
class CalibrationResult:
    stamp: str
    rotations: dict
    encoder_zero_deg: float
    encoder_sign: int
    encoder_ratio: float
    heel_threshold: float
    toe_threshold: float
    checks: list
    metrics: dict

    @property
    def ok(self) -> bool:
        return all(bool(c.passed) for c in self.checks if c.critical)

    def to_json(self) -> dict:
        return {
            "stamp": self.stamp,
            "ok": self.ok,
            "rotations": {s: np.asarray(self.rotations[s]).tolist() for s in SEGMENTS},
            "encoder_zero_deg": self.encoder_zero_deg,
            "encoder_sign": self.encoder_sign,
            "encoder_ratio": self.encoder_ratio,
            "heel_threshold": self.heel_threshold,
            "toe_threshold": self.toe_threshold,
            "features": list(FEATURES),
            "target": TARGET,
            "metrics": self.metrics,
            "checks": [{"name": c.name, "passed": bool(c.passed), "detail": c.detail,
                        "critical": bool(c.critical)} for c in self.checks],
        }


def _previous_gt_relations(out_dir) -> list:
    """
    Derived Georgia Tech foot<-shank relationships from earlier calibration runs.

    Only runs that passed are considered; a failed run's rotations are not something
    to measure drift against.
    """
    found = []
    for path in sorted(Path(out_dir).glob("calibration_*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            rel = d.get("metrics", {}).get("derived", {}).get("r_foot_from_shank_gt")
            if rel and d.get("ok"):
                found.append((path.name, np.array(rel, dtype=np.float64)))
        except (ValueError, OSError):
            continue
    return found


def _json_default(o):
    """
    Coerce numpy scalars and arrays for json.dumps.

    Comparisons on numpy values return np.bool_, and reductions return np.float64;
    both look like Python types until json refuses them, so this is applied at the
    one place every calibration result is written.
    """
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


# ---------------- hardware access ----------------
def load_sensor_hub(imu_report_hz: float = 200.0, verbose: bool = True):
    """
    Import SensorHub from 'Data collection/data_collection.py' and re-rate the IMUs.

    That module hard-codes IMU_REPORT_HZ = 25.0, and SensorHub reads the global when
    it constructs each _FastIMU, so the override must be applied to the module object
    before SensorHub is instantiated. Patching the global rather than editing the file
    leaves your recording script untouched.
    """
    import importlib.util

    path = REPO_ROOT / "Data collection" / "data_collection.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Run calibration from the repository root on the "
            "Jetson, with the exoskeleton connected."
        )

    spec = importlib.util.spec_from_file_location("_exo_data_collection", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)      # requires smbus2 + adafruit_bno08x
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot load the sensor drivers ({exc}).\n"
            "Calibration reads the IMUs, encoder and FSRs directly, so it has to run "
            "on the Jetson with the exoskeleton connected and smbus2 plus "
            "adafruit_bno08x installed.\n"
            "To exercise the analysis without hardware, pass a `recorder` callable to "
            "run_calibration(); test_calibration.py does exactly that."
        ) from exc

    previous = getattr(mod, "IMU_REPORT_HZ", None)
    mod.IMU_REPORT_HZ = float(imu_report_hz)
    if verbose and previous is not None and float(previous) != float(imu_report_hz):
        print(f"  IMU report rate overridden for this run: "
              f"{previous:.0f} Hz -> {imu_report_hz:.0f} Hz")
        print(f"  WARNING: 'Data collection/data_collection.py' still records at "
              f"{previous:.0f} Hz.")
        print(f"  Logs taken with it carry only ~{previous * 1.25:.0f} Hz of real IMU "
              f"bandwidth, so they cannot resolve heel-strike transients.")
    return mod


_SLEEP_MARGIN: Optional[float] = None


def _sleep_margin() -> float:
    """
    How early to stop sleeping and start spinning, measured rather than assumed.

    A short wait costs far more than it asks for on some platforms: on Windows an
    Event.wait(1 ms) measures around 4 ms and time.sleep(1 ms) around 16 ms, while
    Linux delivers close to the requested 1 ms. Sizing the spin window from the
    machine's actual behaviour keeps the loop on rate everywhere, instead of hitting
    target on the Jetson and silently running at a third of it elsewhere.

    Measured once and cached; the result is clamped so a noisy measurement cannot
    make the loop spin for an absurd fraction of every period.
    """
    global _SLEEP_MARGIN
    if _SLEEP_MARGIN is None:
        import threading
        import time

        ev = threading.Event()
        worst = 0.0
        for _ in range(12):
            t = time.perf_counter()
            ev.wait(0.001)
            worst = max(worst, time.perf_counter() - t)
        _SLEEP_MARGIN = float(min(0.020, max(0.0015, worst)))
    return _SLEEP_MARGIN


def record_phase(hub, seconds: float, fs: float = TARGET_HZ,
                 progress: bool = True) -> dict:
    """
    Record every sensor for `seconds` of wall-clock time and return raw arrays.

    Two things this gets right that the obvious implementation does not.

    The loop is bounded by elapsed TIME, not by sample count. Reading four I2C
    devices takes however long it takes, and a fixed `for i in range(seconds * fs)`
    silently runs long whenever the bus cannot keep up - a 10 s phase becoming 30 s,
    with the countdown sailing past zero into negative numbers. Here the phase ends
    when the clock says so and the arrays are truncated to whatever was actually
    captured.

    And the sensors are polled on background threads rather than serially in the
    sampling loop, mirroring data_collection.py. Four blocking I2C transactions per
    tick cannot fit in a 5 ms budget; with workers feeding a cache, the sampling loop
    only has to copy the latest values.

    Ticks where an IMU has not yet produced a packet are left as NaN rather than
    back-filled, so the analysis can measure the true update rate instead of being
    fooled by held values.
    """
    import threading
    import time

    period = 1.0 / fs if fs > 0 else 0.005
    # Room for the target rate plus headroom, so a fast bus is never truncated.
    n_max = int(round(seconds * fs * 1.2)) + 16

    out = {
        "time": np.full(n_max, np.nan),
        "foot_accel": np.full((n_max, 3), np.nan),
        "foot_gyro": np.full((n_max, 3), np.nan),
        "shank_accel": np.full((n_max, 3), np.nan),
        "shank_gyro": np.full((n_max, 3), np.nan),
        "encoder": np.full(n_max, np.nan),
        "toe": np.full(n_max, np.nan), "heel": np.full(n_max, np.nan),
    }

    stop = threading.Event()
    lock = threading.Lock()
    cache = {"foot": None, "shank": None, "enc": None, "toe": None, "heel": None}

    def worker(task):
        # Same cadence discipline as data_collection._periodic_worker: wake often
        # enough to stay fresh without spinning the CPU flat out.
        next_t = time.perf_counter()
        while not stop.is_set():
            now = time.perf_counter()
            if now < next_t:
                stop.wait(timeout=min(0.001, next_t - now))
                continue
            try:
                task()
            except Exception:
                # A dropped I2C read must not kill the phase; the sample simply
                # keeps its previous cached value and the rate check will notice.
                pass
            next_t += period
            if next_t <= now:
                next_t = now + period

    def read_foot():
        pkt = hub.read_imu_foot()
        if pkt is not None:
            with lock:
                cache["foot"] = pkt

    def read_shank():
        pkt = hub.read_imu_shank()
        if pkt is not None:
            with lock:
                cache["shank"] = pkt

    def read_aux():
        enc = hub.read_encoder()["ankle_encoder_deg"]
        fsr = hub.read_fsr()
        with lock:
            cache["enc"] = enc
            cache["toe"] = fsr["toe_fsr_raw"]
            cache["heel"] = fsr["heel_fsr_raw"]

    threads = [threading.Thread(target=worker, args=(read_foot,), daemon=True),
               threading.Thread(target=worker, args=(read_shank,), daemon=True),
               threading.Thread(target=worker, args=(read_aux,), daemon=True)]
    for t in threads:
        t.start()

    margin = _sleep_margin()

    i = 0
    t0 = time.perf_counter()
    next_t = t0
    next_print = t0
    try:
        while True:
            now = time.perf_counter()
            elapsed = now - t0
            if elapsed >= seconds or i >= n_max:
                break
            remaining = next_t - now
            if remaining > 0.0:
                # Sleep while there is more margin than this machine's sleep can
                # overshoot, then spin out the rest. See _sleep_margin.
                if remaining > margin:
                    stop.wait(timeout=remaining - margin)
                    continue
                while time.perf_counter() < next_t:
                    pass

            with lock:
                f, s = cache["foot"], cache["shank"]
                enc, toe, heel = cache["enc"], cache["toe"], cache["heel"]

            out["time"][i] = elapsed
            if f is not None:
                out["foot_accel"][i] = np.asarray(f["accel"][:3], dtype=np.float64)
                out["foot_gyro"][i] = np.asarray(f["gyro"][:3], dtype=np.float64)
            if s is not None:
                out["shank_accel"][i] = np.asarray(s["accel"][:3], dtype=np.float64)
                out["shank_gyro"][i] = np.asarray(s["gyro"][:3], dtype=np.float64)
            if enc is not None:
                out["encoder"][i] = enc
            if toe is not None:
                out["toe"][i] = toe
            if heel is not None:
                out["heel"][i] = heel
            i += 1

            next_t += period
            if next_t <= now:
                # Behind schedule: reset rather than accumulate debt, or the loop
                # would try to "catch up" forever and never sleep again.
                next_t = now + period

            # Throttled: at 200 Hz an unthrottled print is 200 lines a second, which
            # floods the terminal and steals time from the sampling loop itself.
            if progress and now >= next_print:
                print(f"\r    {max(0.0, seconds - elapsed):5.1f} s remaining ",
                      end="", flush=True)
                next_print = now + 0.2
    except KeyboardInterrupt:
        stop.set()
        for t in threads:
            t.join(timeout=1.0)
        raise
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=1.0)

    if i == 0:
        raise RuntimeError(
            f"Recorded no samples in {seconds:.0f} s. The sensor hub returned nothing "
            "at all - check the I2C wiring and addresses before retrying."
        )

    for key in out:
        out[key] = out[key][:i]

    achieved = i / seconds if seconds > 0 else 0.0
    if progress:
        print(f"\r    done: {i} samples in {seconds:.0f} s "
              f"({achieved:.0f} Hz achieved, {fs:.0f} Hz target)")
        if achieved < 0.8 * fs:
            print(f"    NOTE: the sampling loop reached only {achieved:.0f} Hz. "
                  f"The I2C bus is the limit, not the IMU report rate.")
    return out


def channel_health(rec: dict) -> dict:
    """
    Per-channel update rate and variability for one recording.

    This exists because a sensor that has quietly stopped reporting looks exactly
    like a sensor reading a perfectly steady value: the drivers in
    data_collection.py swallow OSError and hand back the last good sample, so a dead
    I2C read shows up as suspiciously clean data rather than as an error. An update
    rate far below the sampling rate, or a standard deviation of exactly zero across
    thousands of samples, is the signature.
    """
    t = rec["time"]
    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    out = {"duration_s": dur, "samples": int(len(t)),
           "sample_hz": float(len(t) / dur) if dur > 0 else 0.0}

    for name in ("foot_accel", "foot_gyro", "shank_accel", "shank_gyro",
                 "encoder", "toe", "heel"):
        v = np.asarray(rec[name], dtype=np.float64)
        finite = np.isfinite(v).all(axis=1) if v.ndim == 2 else np.isfinite(v)
        vv = v[finite]
        if len(vv) < 2:
            out[name] = {"update_hz": 0.0, "std": 0.0, "finite_fraction": 0.0}
            continue
        changed = (np.any(np.diff(vv, axis=0) != 0.0, axis=1) if vv.ndim == 2
                   else (np.diff(vv) != 0.0))
        out[name] = {
            "update_hz": float(np.sum(changed) / dur) if dur > 0 else 0.0,
            "std": float(np.mean(np.std(vv, axis=0)) if vv.ndim == 2 else np.std(vv)),
            "finite_fraction": float(finite.mean()),
        }
    return out


def plot_phase(rec: dict, title: str, path) -> bool:
    """
    Save a multi-panel plot of one recorded phase. Returns False if unavailable.

    Uses the Agg backend so it works over SSH with no display; the file is written
    for later inspection rather than shown.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    t = rec["time"]
    health = channel_health(rec)

    fig, ax = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    fig.suptitle(f"{title}   ({health['samples']} samples, "
                 f"{health['sample_hz']:.0f} Hz)", fontsize=13)

    for row, (key, label) in enumerate((("foot_gyro", "Foot gyro (rad/s)"),
                                        ("foot_accel", "Foot accel (m/s^2)"),
                                        ("shank_gyro", "Shank gyro (rad/s)"),
                                        ("shank_accel", "Shank accel (m/s^2)"))):
        v = np.asarray(rec[key], dtype=np.float64)
        for j, axis_name in enumerate("xyz"):
            ax[row].plot(t, v[:, j], lw=0.8, label=axis_name)
        hz = health[key]["update_hz"]
        ax[row].set_ylabel(label)
        ax[row].legend(loc="upper right", ncol=3, fontsize=8)
        ax[row].grid(True, alpha=0.3)
        ax[row].set_title(f"updates at {hz:.0f} Hz", fontsize=8, loc="left")

    enc = np.asarray(rec["encoder"], dtype=np.float64)
    ax[4].plot(t, enc, color="tab:orange", lw=1.0, label="encoder (deg)")
    ax[4].set_ylabel("Encoder (deg)")
    ax[4].grid(True, alpha=0.3)
    h = health["encoder"]
    ax[4].set_title(f"updates at {h['update_hz']:.1f} Hz, sd {h['std']:.4f} deg, "
                    f"range {np.nanmax(enc) - np.nanmin(enc):.2f} deg",
                    fontsize=8, loc="left")

    fsr = ax[4].twinx()
    fsr.plot(t, rec["toe"], color="tab:red", lw=0.7, alpha=0.6, label="toe FSR")
    fsr.plot(t, rec["heel"], color="tab:green", lw=0.7, alpha=0.6, label="heel FSR")
    fsr.set_ylabel("FSR (counts)")
    lines = ax[4].get_lines() + fsr.get_lines()
    ax[4].legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)
    ax[4].set_xlabel("Time (s)")

    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def plot_sweep_diagnostic(sweep: dict, metrics: dict, path) -> bool:
    """
    Plot what the encoder-to-joint ratio regression actually saw.

    The ratio comes from regressing encoder rate against the foot's sagittal angular
    velocity. When that fit is poor the single R^2 number says nothing about why, so
    this draws both rate traces against each other and as a scatter. A cloud with no
    slope means the encoder and the IMU disagree about the motion; a tilted line with
    scatter means they agree but one is noisy.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    t = sweep["time"]
    ok = np.isfinite(sweep["encoder"]) & np.isfinite(sweep["foot_gyro"]).all(axis=1)
    tt, enc = t[ok], sweep["encoder"][ok]
    axis = np.asarray(metrics.get("foot_sagittal_axis", [0, 0, 1]), dtype=np.float64)
    omega = np.degrees(sweep["foot_gyro"][ok] @ axis)
    d_enc = np.gradient(enc, tt) if len(tt) > 2 else np.zeros_like(enc)

    fig, ax = plt.subplots(3, 1, figsize=(13, 10))
    fig.suptitle("Sweep diagnostic: encoder rate vs IMU sagittal rate", fontsize=13)

    ax[0].plot(tt, enc, color="tab:orange", lw=1.0)
    ax[0].set_ylabel("Encoder (deg)")
    ax[0].grid(True, alpha=0.3)
    ax[0].set_title(f"range {enc.max() - enc.min():.2f} deg", fontsize=8, loc="left")

    ax[1].plot(tt, d_enc, lw=0.8, label="d(encoder)/dt")
    ax[1].plot(tt, omega, lw=0.8, label="IMU rate about sagittal axis")
    ax[1].set_ylabel("deg/s")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(True, alpha=0.3)
    ax[1].set_title("these two should trace the same shape", fontsize=8, loc="left")

    ax[2].scatter(omega, d_enc, s=3, alpha=0.3)
    r2 = metrics.get("encoder_rate_r2", float("nan"))
    ratio = metrics.get("encoder_ratio", float("nan"))
    if np.isfinite(ratio) and ratio != 0:
        xs = np.linspace(omega.min(), omega.max(), 10)
        ax[2].plot(xs, xs / (metrics.get("encoder_sign", 1) * ratio), "r-", lw=1.2,
                   label=f"fit: ratio={ratio:.3f}")
        ax[2].legend(loc="upper left", fontsize=8)
    ax[2].set_xlabel("IMU sagittal rate (deg/s)")
    ax[2].set_ylabel("d(encoder)/dt (deg/s)")
    ax[2].grid(True, alpha=0.3)
    ax[2].set_title(f"R^2 = {r2:.3f}  (a shapeless cloud means the encoder is not "
                    f"tracking the ankle)", fontsize=8, loc="left")

    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def _wait_for_enter(message: str) -> None:
    """
    Block until ENTER, tolerating a stdin that cannot be read.

    If the calibration is launched with stdin redirected or closed, `input` raises
    rather than blocking, and an unguarded prompt would abort the run between two
    phases with the subject mid-pose.
    """
    try:
        input(message)
    except (EOFError, KeyboardInterrupt):
        print(f"{message}  [stdin unavailable, continuing]")


def prompt_phase(spec: PhaseSpec, interactive: bool = True) -> None:
    import time

    print("\n" + "=" * 72)
    print(f"  PHASE {spec.key.upper()}: {spec.title}   ({spec.seconds:.0f} s)")
    print("=" * 72)
    for line in spec.instruction.splitlines():
        print(f"    {line}")
    if interactive:
        _wait_for_enter("\n  Press ENTER when you are in position and ready...")
    for k in (3, 2, 1):
        print(f"    starting in {k}...", end="\r", flush=True)
        time.sleep(1.0)
    print("    RECORDING NOW              ")


def phase_complete(spec: PhaseSpec, index: int, total: int,
                   interactive: bool = True) -> None:
    """
    Stop after a phase and wait to be released into the next one.

    Each phase is gated at both ends deliberately. Rolling straight from one
    recording into the next prompt gives no room to change posture, catch balance
    after standing on one leg, or simply rest, and a rushed phase is a phase that
    fails its quality checks.
    """
    print(f"\n  -- {spec.title} complete ({index} of {total}) --")
    if interactive:
        _wait_for_enter("  Press ENTER when you are ready for the next phase..."
                        if index < total else
                        "  Press ENTER to analyse the recordings...")


# ---------------- per-phase analysis ----------------
def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[np.isfinite(a).all(axis=1)] if a.ndim == 2 else a[np.isfinite(a)]


def _effective_hz(block: np.ndarray, duration_s: float) -> float:
    """Update rate implied by how often a held value actually changes."""
    v = _finite(block)
    if len(v) < 2 or duration_s <= 0:
        return 0.0
    changed = np.any(np.diff(v, axis=0) != 0.0, axis=1) if v.ndim == 2 \
        else (np.diff(v) != 0.0)
    return float(np.sum(changed) / duration_s)


def _unwrap_deg(x: np.ndarray, ref: float) -> np.ndarray:
    """Fold degrees into ref +/- 180 so the 0/360 seam cannot split a hold."""
    return (np.asarray(x, dtype=np.float64) - ref + 180.0) % 360.0 - 180.0 + ref


def analyse_standing(rec: dict) -> tuple[dict, list]:
    """Encoder zero and the gravity reference for both IMUs."""
    checks: list = []
    dur = float(np.nanmax(rec["time"]) - np.nanmin(rec["time"]))

    fg = _finite(rec["foot_gyro"])
    sg = _finite(rec["shank_gyro"])
    speed = np.linalg.norm(np.vstack([fg, sg]), axis=1)
    still = float(np.mean(speed < STILL_GYRO_MAX)) if len(speed) else 0.0

    enc = _finite(rec["encoder"])
    if len(enc) == 0:
        raise ValueError("Standing phase recorded no encoder samples.")
    enc_u = _unwrap_deg(enc, float(np.median(enc)))
    zero = float(np.median(enc_u)) % 360.0
    enc_sd = float(np.std(enc_u))

    gravity = {}
    for seg in SEGMENTS:
        a = _finite(rec[f"{seg}_accel"])
        if len(a) == 0:
            raise ValueError(f"Standing phase recorded no {seg} accelerometer samples.")
        gravity[seg] = _unit(a.mean(axis=0))

    imu_hz = min(_effective_hz(rec["foot_accel"], dur),
                 _effective_hz(rec["shank_accel"], dur))

    checks.append(Check(
        "subject stood still", still >= STILL_MIN_FRACTION,
        f"{still:.0%} of samples below {STILL_GYRO_MAX} rad/s "
        f"(need {STILL_MIN_FRACTION:.0%})"))
    checks.append(Check(
        "encoder steady during standing", enc_sd <= STILL_MAX_ENCODER_SD,
        f"encoder sd = {enc_sd:.3f} deg over {len(enc)} samples "
        f"(max {STILL_MAX_ENCODER_SD})"))
    checks.append(Check(
        "IMU report rate override took effect", imu_hz >= MIN_EFFECTIVE_IMU_HZ,
        f"measured {imu_hz:.0f} Hz effective (need {MIN_EFFECTIVE_IMU_HZ:.0f}); "
        f"below this the BNO085 is still on its old report interval"))
    for seg in SEGMENTS:
        mag = float(np.linalg.norm(_finite(rec[f"{seg}_accel"]).mean(axis=0)))
        checks.append(Check(
            f"{seg} accel reads 1 g at rest", 8.5 <= mag <= 11.0,
            f"|mean accel| = {mag:.2f} m/s^2 (expect ~9.81)"))

    return {"encoder_zero_deg": zero, "encoder_zero_sd": enc_sd,
            "encoder_zero_n": int(len(enc)), "still_fraction": still,
            "gravity": {s: gravity[s].tolist() for s in SEGMENTS},
            "effective_imu_hz": imu_hz}, checks


def analyse_sweep(neutral: dict, dorsi: dict, plantar: dict, sweep: dict,
                  encoder_zero_deg: float) -> tuple[dict, list]:
    """
    Encoder sign, encoder-to-joint ratio, and the foot's sagittal axis.

    Sign comes from the labelled holds and needs no IMU at all: if the encoder reads
    higher in dorsiflexion than in neutral, then increasing raw degrees means
    dorsiflexion. Ratio comes from regressing encoder rate against the foot's
    sagittal angular velocity during the free sweep - both are rates, so there is no
    integration drift and no need to estimate absolute tilt.

    Two things that are easy to get wrong here:

    The regression is deliberately run with the gyro as the predictor and the encoder
    rate as the response, because the AS5600 is quantised at 0.088 deg and
    differentiating it is noisy; putting the noisy variable on the predictor side
    would bias the slope toward zero. That direction means the fitted slope is
    d(raw)/d(joint) = 1 / (sign * ratio), so the ratio is the RECIPROCAL of the
    slope, not the slope.

    And the sweep cannot independently confirm the sign. The sagittal axis comes from
    PCA, whose sign is arbitrary, so the sign of the slope carries that arbitrary
    choice rather than any fact about the ankle. The axis is therefore oriented using
    the holds, and the genuine independent confirmation of sign happens later, in
    analyse_walking, against the Georgia Tech goniometer.
    """
    checks: list = []

    def hold_level(rec: dict, label: str) -> float:
        e = _finite(rec["encoder"])
        if len(e) == 0:
            raise ValueError(f"{label} hold recorded no encoder samples.")
        return float(np.median(_unwrap_deg(e, encoder_zero_deg)))

    e_neutral = hold_level(neutral, "neutral")
    e_dorsi = hold_level(dorsi, "dorsiflexion")
    e_plantar = hold_level(plantar, "plantarflexion")

    d_dorsi = e_dorsi - e_neutral
    d_plantar = e_plantar - e_neutral
    sign = 1 if d_dorsi > 0 else -1
    rom = abs(d_dorsi) + abs(d_plantar)

    fg = _finite(sweep["foot_gyro"])
    axis, var_ratio = principal_axis(fg)

    t = sweep["time"]
    ok = np.isfinite(sweep["encoder"]) & np.isfinite(sweep["foot_gyro"]).all(axis=1) \
        & np.isfinite(t)
    tt = t[ok]
    enc = _unwrap_deg(sweep["encoder"][ok], encoder_zero_deg)
    if len(tt) < 20:
        raise ValueError("Sweep phase has too few usable samples to fit a ratio.")

    d_enc = np.gradient(enc, tt)                   # raw encoder deg/s

    # Orient the PCA axis so that positive angular velocity along it means increasing
    # ankle angle in the direction the holds established as dorsiflexion. Without
    # this the axis sign is a coin flip and the fitted slope's sign is meaningless.
    gyro_ok = sweep["foot_gyro"][ok]                        # exactly len(tt) rows
    if float(np.dot(gyro_ok @ axis, sign * d_enc)) < 0.0:
        axis = -axis

    omega_deg = np.degrees(gyro_ok @ axis)                  # joint deg/s from the IMU

    # Least squares through the origin: a stationary ankle must give zero rate.
    # Gyro is the predictor (it is the clean signal); see the note in the docstring.
    denom = float(np.dot(omega_deg, omega_deg))
    slope = float(np.dot(omega_deg, d_enc) / denom) if denom > 0 else 0.0
    resid = d_enc - slope * omega_deg
    ss_tot = float(np.sum((d_enc - d_enc.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0

    # slope is d(raw)/d(joint) = 1 / (sign * ratio), so invert it for the ratio.
    ratio = 1.0 / abs(slope) if abs(slope) > 1e-9 else float("nan")

    shank_speed = float(np.mean(np.linalg.norm(_finite(sweep["shank_gyro"]), axis=1))) \
        if len(_finite(sweep["shank_gyro"])) else float("nan")

    checks.append(Check(
        "held poses separate from neutral",
        abs(d_dorsi) >= SWEEP_MIN_HOLD_SEPARATION
        and abs(d_plantar) >= SWEEP_MIN_HOLD_SEPARATION,
        f"dorsi {d_dorsi:+.1f} deg, plantar {d_plantar:+.1f} deg from neutral "
        f"(each must exceed {SWEEP_MIN_HOLD_SEPARATION})"))
    checks.append(Check(
        "the two holds lie on opposite sides", d_dorsi * d_plantar < 0,
        f"dorsi {d_dorsi:+.1f} and plantar {d_plantar:+.1f} must have opposite signs; "
        f"same sign means the poses were confused or the encoder wrapped"))
    checks.append(Check(
        "ankle range of motion is usable", rom >= SWEEP_MIN_ROM_DEG,
        f"total ROM = {rom:.1f} deg (need {SWEEP_MIN_ROM_DEG})"))
    checks.append(Check(
        "shank stayed still during sweeps", shank_speed <= SWEEP_MAX_SHANK_GYRO,
        f"mean shank |gyro| = {shank_speed:.3f} rad/s (max {SWEEP_MAX_SHANK_GYRO}); "
        f"a moving shank contaminates the ratio"))
    checks.append(Check(
        "encoder tracks the IMU during sweeps",
        r2 >= SWEEP_MIN_R2 and np.isfinite(ratio),
        f"rate regression R^2 = {r2:.3f}, ratio = {ratio:.4f} joint deg per encoder "
        f"deg (need R^2 >= {SWEEP_MIN_R2})"))
    checks.append(Check(
        "encoder moved during the sweep",
        float(np.ptp(enc)) >= SWEEP_MIN_ROM_DEG * 0.5,
        f"encoder spanned {float(np.ptp(enc)):.2f} deg across the sweep "
        f"(need {SWEEP_MIN_ROM_DEG * 0.5:.1f}); a small span with a large IMU "
        f"excursion means the magnet is not following the joint"))
    checks.append(Check(
        "encoder ratio is physically plausible",
        bool(np.isfinite(ratio) and 0.2 <= ratio <= 5.0),
        f"ratio = {ratio:.4f}; a magnet mounted directly on the joint axis should be "
        f"near 1.0, and anything outside 0.2-5.0 suggests a linkage or a bad fit"))
    checks.append(Check(
        "sweep motion was planar", var_ratio >= 0.70,
        f"sagittal axis explains {var_ratio:.0%} of sweep gyro variance", False))

    return {"encoder_sign": int(sign), "encoder_ratio": float(ratio),
            "encoder_rate_r2": float(r2), "rate_slope": float(slope),
            "neutral_deg": e_neutral, "dorsi_deg": e_dorsi, "plantar_deg": e_plantar,
            "dorsiflexion_rom_deg": abs(d_dorsi),
            "plantarflexion_rom_deg": abs(d_plantar),
            "foot_sagittal_axis": axis.tolist(),
            "foot_sagittal_var_ratio": float(var_ratio),
            "shank_mean_gyro": shank_speed}, checks


def preflight_reference(hf_root, subject: str) -> list[str]:
    """
    Check the Georgia Tech reference is present and usable, BEFORE any recording.

    This used to be discovered inside analyse_walking, which meant a missing dataset
    surfaced only after the full 105 s protocol had been performed - and because the
    exception escaped before the raw recordings were written, every one of those
    seconds was lost. Nothing about this check needs the hardware, so it belongs at
    the very start.
    """
    root = Path(hf_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at '{root.resolve()}'.\n"
            f"The rotation fit needs the Georgia Tech parquet trials. Either copy the "
            f"mrsd-exo-ankle folder next to exo_frame.py, or point at it with "
            f"--hf-root /path/to/mrsd-exo-ankle."
        )

    subjects_dir = root / "subjects"
    if not subjects_dir.is_dir():
        raise FileNotFoundError(
            f"'{root.resolve()}' exists but has no 'subjects/' directory, so it is not "
            f"a mrsd-exo-ankle snapshot. Contents: "
            f"{sorted(p.name for p in root.iterdir())[:8]}"
        )

    available = sorted(p.name for p in subjects_dir.iterdir() if p.is_dir())
    if subject not in available:
        raise FileNotFoundError(
            f"Subject '{subject}' is not in '{subjects_dir.resolve()}'.\n"
            f"Available: {available if available else '(none)'}\n"
            f"Pick one with --reference-subject, or download {subject} into the "
            f"snapshot."
        )

    trials = list_hf_trials(root, subject)
    if not trials:
        raise FileNotFoundError(
            f"Subject '{subject}' has no *__imu.parquet trials in "
            f"{(subjects_dir / subject).resolve()}. The download is incomplete."
        )

    missing = [f"{t}__{k}" for t in trials[:3] for k in ("imu", "id", "gon", "gcRight")
               if not (subjects_dir / subject / f"{t}__{k}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"Subject '{subject}' is missing parquet files needed by the fit: "
            f"{missing[:6]}. Re-download the snapshot."
        )
    return trials


def two_direction_rotation(g_from: np.ndarray, w_from: np.ndarray,
                           g_to: np.ndarray, w_to: np.ndarray) -> np.ndarray:
    """
    Rotation taking `from`-frame coordinates to `to`-frame coordinates.

    Built from two physical directions observed in both frames - here gravity while
    standing, and the hip-swing axis. Two non-parallel directions pin all three
    degrees of freedom, which matters because the swing on its own cannot: a planar
    swing produces angular velocity along a single axis, leaving rotation about that
    axis completely unconstrained. Gravity supplies the missing direction.
    """
    def basis(g, w):
        e1 = _unit(g)
        e2 = w - np.dot(w, e1) * e1
        if np.linalg.norm(e2) < 1e-8:
            raise ValueError("Gravity and the swing axis are parallel; "
                             "cannot build a frame from this recording.")
        e2 = _unit(e2)
        return np.column_stack([e1, e2, np.cross(e1, e2)])

    return basis(g_to, w_to) @ basis(g_from, w_from).T


def analyse_hipswing(standing: dict, hipswing: dict,
                     encoder_zero_deg: float) -> tuple[dict, list]:
    """
    Shank sagittal axis, and the relationship between the two exo IMU frames.

    With the knee and ankle held, the foot and shank swing as one rigid body, so
    both IMUs observe the same physical angular velocity expressed in their own
    frames. That makes the swing axis a shared direction, and combined with gravity
    from the standing phase it determines R_foot<-shank outright.

    This is the only phase that measures the shank directly. Without it the shank's
    sagittal axis is never observed on its own - it is only implied by the walking
    fit against Georgia Tech - which leaves the two segments calibrated on very
    unequal evidence.

    Note what this does NOT do. R_foot<-shank cannot constrain the two Georgia Tech
    rotations, because that would also require the Georgia Tech foot<-shank
    relationship, and their walking data never contains a rigid foot-shank epoch to
    measure it from: fitting one segment's gyro onto the other's at mid-stance
    leaves a residual around 70% of signal. The relationship is therefore recorded
    as a derived quantity and checked for drift across runs, not imposed.
    """
    checks: list = []

    enc = _finite(hipswing["encoder"])
    enc_sd = float(np.std(_unwrap_deg(enc, encoder_zero_deg))) if len(enc) else float("nan")

    fg = _finite(hipswing["foot_gyro"])
    sg = _finite(hipswing["shank_gyro"])
    if len(fg) < 20 or len(sg) < 20:
        raise ValueError("Hip-swing phase has too few usable IMU samples.")

    foot_speed = float(np.mean(np.linalg.norm(fg, axis=1)))
    shank_speed = float(np.mean(np.linalg.norm(sg, axis=1)))

    foot_axis, foot_var = principal_axis(fg)
    shank_axis, shank_var = principal_axis(sg)

    # PCA signs are arbitrary. Both segments share one angular velocity, so orient
    # them to agree with each other.
    n = min(len(fg), len(sg))
    if float(np.dot(fg[:n] @ foot_axis, sg[:n] @ shank_axis)) < 0.0:
        shank_axis = -shank_axis

    g_foot = _unit(_finite(standing["foot_accel"]).mean(axis=0))
    g_shank = _unit(_finite(standing["shank_accel"]).mean(axis=0))

    r_fs = two_direction_rotation(g_shank, shank_axis, g_foot, foot_axis)

    # Best-achievable fit, used to judge whether the motion was rigid at all. Kabsch
    # on a planar swing is rank-limited, so this measures rigidity, not R_fs.
    r_kabsch, _ = kabsch(sg[:n].T, fg[:n].T)
    residual = float(np.linalg.norm(r_kabsch @ sg[:n].T - fg[:n].T) / np.sqrt(n))
    residual_frac = residual / foot_speed if foot_speed > 0 else float("nan")
    method_gap = rotation_angle(r_fs, r_kabsch)

    # How well the rotation we actually keep reproduces the measured swing. This
    # tests two of the three degrees of freedom; it cannot test rotation about the
    # swing axis, because rotating a vector about itself changes nothing. That
    # remaining degree of freedom is fixed by standing gravity, and is watched over
    # time by the cross-run drift check rather than from within this phase.
    applied = float(np.linalg.norm(r_fs @ sg[:n].T - fg[:n].T) / np.sqrt(n))
    applied_frac = applied / foot_speed if foot_speed > 0 else float("nan")

    checks.append(Check(
        "ankle stayed still during hip swings", enc_sd <= HIPSWING_MAX_ENCODER_SD,
        f"encoder sd = {enc_sd:.2f} deg (max {HIPSWING_MAX_ENCODER_SD}); a moving "
        f"ankle breaks the rigid-body assumption this phase depends on"))
    checks.append(Check(
        "the leg actually swung", shank_speed >= HIPSWING_MIN_GYRO,
        f"mean shank |gyro| = {shank_speed:.3f} rad/s "
        f"(need {HIPSWING_MIN_GYRO}), foot {foot_speed:.3f}"))
    checks.append(Check(
        "foot and shank moved as one body", residual_frac <= HIPSWING_MAX_RESIDUAL,
        f"one rotation explains the pair to {residual_frac:.1%} of signal "
        f"(max {HIPSWING_MAX_RESIDUAL:.0%}); higher means the knee or ankle moved"))
    checks.append(Check(
        "swing was planar", min(foot_var, shank_var) >= HIPSWING_MIN_VAR_RATIO,
        f"swing axis explains {foot_var:.0%} of foot and {shank_var:.0%} of shank "
        f"gyro variance", False))
    checks.append(Check(
        "R_foot<-shank reproduces the swing", applied_frac <= HIPSWING_MAX_RESIDUAL,
        f"applying it to the shank gyro predicts the foot gyro to {applied_frac:.1%} "
        f"of signal (max {HIPSWING_MAX_RESIDUAL:.0%}). Tests two of three axes: "
        f"rotation about the swing axis cannot be tested from a planar swing, and "
        f"is fixed by standing gravity instead"))

    return {"shank_sagittal_axis": shank_axis.tolist(),
            "applied_residual_fraction": float(applied_frac),
            "shank_sagittal_var_ratio": float(shank_var),
            "foot_sagittal_axis": foot_axis.tolist(),
            "foot_sagittal_var_ratio": float(foot_var),
            "r_foot_from_shank": r_fs.tolist(),
            "r_foot_from_shank_kabsch": r_kabsch.tolist(),
            "method_gap_deg": float(method_gap),
            "rigid_residual_fraction": float(residual_frac),
            "encoder_sd_deg": enc_sd,
            "foot_mean_gyro": foot_speed,
            "shank_mean_gyro": shank_speed}, checks


def constrain_rotations(stacked: dict, r_fs_gt: np.ndarray,
                        r_fs_exo: np.ndarray) -> tuple[dict, float]:
    """
    Re-fit both rotations subject to R_foot @ R_fs_gt == R_fs_exo @ R_shank.

    Derivation. For any physical vector, going Georgia-Tech-shank -> exo-foot by
    either route must agree, which gives that constraint and hence
    R_foot = R_fs_exo @ R_shank @ R_fs_gt^-1. Substituting into the combined
    least-squares objective, and using that R_fs_exo is orthogonal so it can be
    moved across the norm, turns the two separate Procrustes problems into one:

        minimise  || R_shank [P_s , R_fs_gt^-1 P_f] - [Q_s , R_fs_exo^-1 Q_f] ||

    So a single Kabsch on the stacked pair yields R_shank, and R_foot follows
    exactly. Both rotations end up informed by both segments' walking data.

    This is only sound when `r_fs_gt` is genuinely known. It is a property of the
    Georgia Tech hardware that their walking data cannot reveal - fitting one of
    their segments' gyro onto the other's at mid-stance leaves ~70% residual - so it
    has to come from an earlier calibration that was trusted. That is why this is
    opt-in and never the default.
    """
    p_s, q_s = stacked["shank"]
    p_f, q_f = stacked["foot"]
    inv_gt = np.asarray(r_fs_gt, dtype=np.float64).T
    inv_exo = np.asarray(r_fs_exo, dtype=np.float64).T

    p = np.hstack([p_s, inv_gt @ p_f])
    q = np.hstack([q_s, inv_exo @ q_f])
    r_shank, residual = kabsch(p, q)
    r_foot = np.asarray(r_fs_exo, dtype=np.float64) @ r_shank @ inv_gt
    return {"foot": r_foot, "shank": r_shank}, float(residual)


def analyse_walking(rec: dict, hf_root, reference_subject: str = "AB06",
                    max_reference_trials: int = 3,
                    adaptive_fraction: float = 0.5,
                    encoder_zero_deg: Optional[float] = None,
                    encoder_sign: int = 1,
                    encoder_ratio: float = 1.0) -> tuple[dict, list, dict]:
    """
    Fit the Georgia Tech to exo rotations from the walking phase.

    Contact thresholds are taken adaptively from this recording rather than from the
    controller's fixed counts, because a calibration run is exactly the moment to
    measure what this particular FSR seating produces.
    """
    checks: list = []
    dur = float(np.nanmax(rec["time"]) - np.nanmin(rec["time"]))
    fs = len(rec["time"]) / dur if dur > 0 else TARGET_HZ

    accel, gyro = {}, {}
    for seg in SEGMENTS:
        accel[seg] = _ffill_nan(rec[f"{seg}_accel"])
        gyro[seg] = _ffill_nan(rec[f"{seg}_gyro"])

    heel = _lowpass(_ffill_nan(rec["heel"]), fs)
    toe = _lowpass(_ffill_nan(rec["toe"]), fs)

    def level(x):
        lo, hi = np.percentile(x, [10.0, 90.0])
        return float(lo + adaptive_fraction * (hi - lo))

    heel_thr, toe_thr = level(heel), level(toe)
    heel_contact = (heel > heel_thr).astype(np.float64)
    toe_contact = (toe >= toe_thr).astype(np.float64)
    hs = np.where(np.diff(heel_contact.astype(int)) == 1)[0] + 1
    bounds = stride_bounds(hs, len(heel_contact), fs)

    checks.append(Check(
        "enough strides for a stable fit", len(bounds) >= WALK_MIN_STRIDES,
        f"{len(bounds)} usable strides in {dur:.0f} s (need {WALK_MIN_STRIDES})"))

    imu_hz = min(_effective_hz(rec["foot_accel"], dur),
                 _effective_hz(rec["shank_accel"], dur))
    checks.append(Check(
        "walking captured at full IMU rate", imu_hz >= MIN_EFFECTIVE_IMU_HZ,
        f"measured {imu_hz:.0f} Hz effective (need {MIN_EFFECTIVE_IMU_HZ:.0f})"))

    if len(bounds) < 3:
        raise ValueError(
            f"Only {len(bounds)} usable strides in the walking phase; cannot fit a "
            "rotation. Check the FSR seating and walk for longer.")

    hfs = [read_hf_trial(hf_root, reference_subject, t)
           for t in list_hf_trials(hf_root, reference_subject)[:max_reference_trials]]
    if not hfs:
        raise RuntimeError(f"No Georgia Tech trials found for {reference_subject}.")

    rotations, spreads, corrs, stacked = {}, {}, {}, {}
    for seg in SEGMENTS:
        ex_a = mean_cycle(accel[seg], bounds)
        ex_g = mean_cycle(gyro[seg], bounds)
        cands, lags = [], []
        for hf in hfs:
            hb = stride_bounds(hf["heel_strikes"], len(hf["time"]), hf["fs"])
            hf_a = mean_cycle(hf["accel"][seg] * G_TO_MS2, hb)
            hf_g = mean_cycle(hf["gyro"][seg], hb)
            lag, r, _ = _fit_lag_and_rotation(hf_a, hf_g, ex_a, ex_g)
            cands.append(r)
            lags.append(lag)
        mean_r, rejected, spread = average_rotations(cands)
        rotations[seg] = mean_r
        spreads[seg] = spread

        # Kept at the first trial's alignment so a constrained re-fit can reuse them.
        hb0 = stride_bounds(hfs[0]["heel_strikes"], len(hfs[0]["time"]), hfs[0]["fs"])
        stacked[seg] = (
            _stack(mean_cycle(hfs[0]["accel"][seg] * G_TO_MS2, hb0),
                   mean_cycle(hfs[0]["gyro"][seg], hb0)),
            _stack(np.roll(ex_a, lags[0], axis=0), np.roll(ex_g, lags[0], axis=0)))

        hb = stride_bounds(hfs[0]["heel_strikes"], len(hfs[0]["time"]), hfs[0]["fs"])
        hf_g0 = mean_cycle(hfs[0]["gyro"][seg], hb)
        rotated = (mean_r @ hf_g0.T).T
        corrs[seg] = float(np.corrcoef(rotated[:, 2],
                                       np.roll(ex_g, lags[0], axis=0)[:, 2])[0, 1])

    worst_spread = max(spreads.values())
    worst_corr = min(corrs.values())
    checks.append(Check(
        "rotation fits agree across trials", worst_spread <= WALK_MAX_ROTATION_SPREAD_DEG,
        ", ".join(f"{s}={spreads[s]:.2f} deg" for s in SEGMENTS)
        + f" (max {WALK_MAX_ROTATION_SPREAD_DEG})"))
    checks.append(Check(
        "sagittal waveform matches Georgia Tech", worst_corr >= WALK_MIN_SAGITTAL_CORR,
        ", ".join(f"{s} gyro_z r={corrs[s]:+.3f}" for s in SEGMENTS)
        + f" (min {WALK_MIN_SAGITTAL_CORR:+.2f})"))

    # Foot and shank are fitted independently, so nothing forces their signs to
    # agree. Whatever relationship the exo's two sagittal gyros have, the rotated
    # Georgia Tech data has to reproduce its sign.
    ex_gf = mean_cycle(gyro["foot"], bounds)[:, 2]
    ex_gs = mean_cycle(gyro["shank"], bounds)[:, 2]
    r_exo = float(np.corrcoef(ex_gf, ex_gs)[0, 1])
    hb = stride_bounds(hfs[0]["heel_strikes"], len(hfs[0]["time"]), hfs[0]["fs"])
    tf = (rotations["foot"] @ mean_cycle(hfs[0]["gyro"]["foot"], hb).T).T
    ts = (rotations["shank"] @ mean_cycle(hfs[0]["gyro"]["shank"], hb).T).T
    r_fixed = float(np.corrcoef(tf[:, 2], ts[:, 2])[0, 1])
    # This is only informative when the exo's own two segments are actually related.
    # On this hardware they correlate at about +0.96 because both IMUs are mounted
    # alike, and then a sign mismatch is a genuine per-segment error. If the two
    # segments happen to be weakly related, the sign of a near-zero correlation is
    # noise and asserting on it would fail at random, so the check steps down to
    # advisory rather than pretending to know something it does not.
    informative = abs(r_exo) >= 0.5
    checks.append(Check(
        "foot and shank rotations are consistent",
        (np.sign(r_fixed) == np.sign(r_exo) and abs(r_fixed) >= 0.4)
        if informative else True,
        f"exo foot/shank gyro_z r={r_exo:+.3f}, transformed GT r={r_fixed:+.3f}"
        + ("; signs must match" if informative else
           "; exo segments only weakly related, so this cannot confirm the signs"),
        critical=informative))

    # Independent confirmation of the encoder sign. The holds established it from the
    # encoder alone; this checks it against the Georgia Tech goniometer, which is
    # labelled dorsiflexion-positive. Nothing in this comparison touches the holds,
    # so it is a real second opinion rather than a restatement of the first.
    ankle_sign_corr = float("nan")
    if encoder_zero_deg is not None:
        enc_raw = _ffill_nan(rec["encoder"])
        d = (enc_raw - encoder_zero_deg + 180.0) % 360.0 - 180.0
        exo_ankle = encoder_sign * encoder_ratio * d
        exo_cycle = mean_cycle(exo_ankle, bounds)
        exo_cycle = exo_cycle - exo_cycle.mean()

        hb0 = stride_bounds(hfs[0]["heel_strikes"], len(hfs[0]["time"]), hfs[0]["fs"])
        gt_cycle = mean_cycle(hfs[0]["ankle_angle_deg"], hb0)
        gt_cycle = gt_cycle - gt_cycle.mean()

        # The two heel-strike detectors fire at different points, so compare at the
        # alignment that maximises absolute agreement and read the sign there.
        best = 0.0
        for lag in range(len(exo_cycle)):
            r = float(np.corrcoef(gt_cycle, np.roll(exo_cycle, lag))[0, 1])
            if abs(r) > abs(best):
                best = r
        ankle_sign_corr = best

    checks.append(Check(
        "encoder sign confirmed against Georgia Tech goniometer",
        bool(np.isfinite(ankle_sign_corr) and ankle_sign_corr > 0.5),
        f"ankle angle gait cycle vs GT ankle_sagittal: r={ankle_sign_corr:+.3f} "
        f"(must be positive and above 0.5; negative means the held poses were "
        f"swapped and the sign is inverted)"))

    return {"strides": len(bounds), "effective_imu_hz": imu_hz,
            "ankle_sign_corr": ankle_sign_corr,
            "stacked": stacked,
            "heel_threshold": heel_thr, "toe_threshold": toe_thr,
            "rotation_spread_deg": spreads, "sagittal_corr": corrs,
            "intersegment_corr_exo": r_exo, "intersegment_corr_fitted": r_fixed,
            "rotations": {s: rotations[s].tolist() for s in SEGMENTS}}, checks, rotations


def _ffill_nan(a: np.ndarray) -> np.ndarray:
    """
    Forward then backward fill NaN along axis 0, matching the logger's hold.

    Vectorised via a running maximum over the indices of valid samples, because the
    per-element Python version ran 200 Hz x 60 s x every column on the walking phase.
    """
    arr = np.array(a, dtype=np.float64, copy=True)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[:, None]

    n, m = arr.shape
    if n == 0:
        return arr[:, 0] if squeeze else arr

    valid = np.isfinite(arr)
    # Index of the most recent valid sample at or before each row, per column.
    idx = np.where(valid, np.arange(n)[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)

    # Leading NaN have no earlier sample, so they take the first valid one instead.
    any_valid = valid.any(axis=0)
    first = np.argmax(valid, axis=0)
    idx = np.where(idx < 0, first[None, :], idx)

    out = arr[idx, np.arange(m)[None, :]]
    if not any_valid.all():
        out[:, ~any_valid] = 0.0          # a channel that never reported at all
    return out[:, 0] if squeeze else out


# ---------------- orchestration ----------------
def run_calibration(hf_root="mrsd-exo-ankle",
                    out_dir=CALIB_DIR,
                    reference_subject: str = "AB06",
                    imu_report_hz: float = 200.0,
                    fs: float = TARGET_HZ,
                    interactive: bool = True,
                    recorder=None,
                    constrain: bool = False,
                    protocol=PROTOCOL) -> CalibrationResult:
    """
    Run the full guided calibration and write the results plus a transform module.

    `recorder(spec) -> dict` can be injected to replay recorded phases instead of
    reading hardware, which is how the offline tests exercise this path.
    """
    import datetime

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  EXOSKELETON CALIBRATION")
    print("=" * 72)
    total = sum(p.seconds for p in protocol)
    print(f"  {len(protocol)} phases, {total:.0f} s of recording plus setup time.")
    print(f"  Reference dataset : {hf_root} (subject {reference_subject})")
    print(f"  Output            : {out_dir}/")
    print("\n  Wear the exoskeleton on the RIGHT leg before starting.")
    print("  You will be told what to do and for how long before each phase.")

    # Verified before a single second is recorded. Discovering a missing
    # dataset after the protocol has been performed wastes the whole session.
    ref_trials = preflight_reference(hf_root, reference_subject)
    print(f"  Reference verified: {len(ref_trials)} trial(s) for {reference_subject}")

    plot_dir = out_dir / f"plots_{stamp}"

    if recorder is None:
        mod = load_sensor_hub(imu_report_hz)
        hub = mod.SensorHub()
        # Measure the timer granularity now rather than inside the first
        # phase, so it costs nothing once the subject is in position.
        print(f"  Timer granularity : {_sleep_margin() * 1000:.1f} ms")

        def recorder(spec):
            prompt_phase(spec, interactive)
            return record_phase(hub, spec.seconds, fs)
    else:
        hub = None

    try:
        recordings = {}
        for i, spec in enumerate(protocol, start=1):
            recordings[spec.key] = recorder(spec)
            if hub is not None:
                phase_complete(spec, i, len(protocol), interactive)
    finally:
        if hub is not None:
            hub.close()

    # Written before anything is analysed. An analysis failure must never cost
    # the recordings, which are the expensive part of the session.
    raw_path = out_dir / f"raw_{stamp}.npz"
    flat = {f"{k}__{f}": v for k, rec in recordings.items() for f, v in rec.items()}
    np.savez_compressed(raw_path, **flat)
    print("")
    print(f"  raw recordings saved -> {raw_path}")

    plot_dir.mkdir(parents=True, exist_ok=True)
    plotted = sum(plot_phase(recordings[s.key], f"{s.key.upper()}: {s.title}",
                             plot_dir / f"{s.key}.png") for s in protocol)
    print(f"  {plotted} phase plot(s) -> {plot_dir}/" if plotted
          else "  (matplotlib unavailable, no plots written)")

    print("")
    print("=" * 72)
    print("  CHANNEL HEALTH")
    print("=" * 72)
    print("  phase       samples   rate  footIMU shankIMU  encoder    enc sd")
    health_all = {}
    for spec in protocol:
        h = channel_health(recordings[spec.key])
        health_all[spec.key] = h
        print(f"  {spec.key:<10}{h['samples']:>9}{h['sample_hz']:>6.0f}H"
              f"{h['foot_gyro']['update_hz']:>8.0f}H"
              f"{h['shank_gyro']['update_hz']:>8.0f}H"
              f"{h['encoder']['update_hz']:>8.1f}H"
              f"{h['encoder']['std']:>10.4f}")

    print("\n" + "=" * 72)
    print("  ANALYSING")
    print("=" * 72)

    checks: list = []
    metrics: dict = {"health": health_all}

    stand_m, stand_c = analyse_standing(recordings["standing"])
    metrics["standing"] = stand_m
    checks += stand_c
    print(f"  encoder zero      : {stand_m['encoder_zero_deg']:.3f} deg "
          f"(sd {stand_m['encoder_zero_sd']:.3f}, n={stand_m['encoder_zero_n']})")

    sweep_m, sweep_c = analyse_sweep(recordings["neutral"], recordings["dorsi"],
                                     recordings["plantar"], recordings["sweep"],
                                     stand_m["encoder_zero_deg"])
    metrics["sweep"] = sweep_m
    checks += sweep_c
    print(f"  encoder sign      : {sweep_m['encoder_sign']:+d} "
          f"(dorsiflexion {'increases' if sweep_m['encoder_sign'] > 0 else 'decreases'} "
          f"raw degrees)")
    print(f"  encoder ratio     : {sweep_m['encoder_ratio']:.4f} joint deg per "
          f"encoder deg (R^2 {sweep_m['encoder_rate_r2']:.3f})")
    print(f"  ankle ROM         : {sweep_m['dorsiflexion_rom_deg']:.1f} deg dorsi / "
          f"{sweep_m['plantarflexion_rom_deg']:.1f} deg plantar")

    # Drawn whether or not the regression succeeded: when it fails, this plot is the
    # fastest way to see whether the encoder and the IMU disagree about the motion or
    # simply agree noisily.
    if plot_sweep_diagnostic(recordings["sweep"], sweep_m,
                             plot_dir / "sweep_diagnostic.png"):
        print(f"  sweep diagnostic  -> {plot_dir / 'sweep_diagnostic.png'}")

    hip_m, hip_c = analyse_hipswing(recordings["standing"], recordings["hipswing"],
                                    stand_m["encoder_zero_deg"])
    metrics["hipswing"] = hip_m
    checks += hip_c
    print(f"  shank axis        : {np.round(hip_m['shank_sagittal_axis'], 3).tolist()} "
          f"({hip_m['shank_sagittal_var_ratio']:.0%} of swing variance)")
    print(f"  rigid-body fit    : {hip_m['rigid_residual_fraction']:.1%} residual, "
          f"methods agree to {hip_m['method_gap_deg']:.1f} deg")

    walk_m, walk_c, rotations = analyse_walking(
        recordings["walking"], hf_root, reference_subject,
        encoder_zero_deg=stand_m["encoder_zero_deg"],
        encoder_sign=sweep_m["encoder_sign"],
        encoder_ratio=sweep_m["encoder_ratio"])
    metrics["walking"] = walk_m
    checks += walk_c
    print(f"  strides captured  : {walk_m['strides']}")
    for seg in SEGMENTS:
        print(f"  R[{seg}] spread    : {walk_m['rotation_spread_deg'][seg]:.2f} deg, "
              f"sagittal r={walk_m['sagittal_corr'][seg]:+.3f}")

    # The Georgia Tech foot<-shank relationship, derived rather than measured:
    #   R_fs_exo = R_foot @ R_fs_gt @ R_shank^-1   =>   R_fs_gt = R_foot^-1 @ R_fs_exo @ R_shank
    # It is a fixed property of their hardware, so it must come out the same on
    # every calibration of every exo. Drift means an IMU moved, or a phase was done
    # badly. It is recorded and checked, never imposed - imposing it would require
    # trusting a value nothing in this run can verify.
    r_fs_exo = np.array(hip_m["r_foot_from_shank"], dtype=np.float64)
    stacked = walk_m.pop("stacked")

    prior = _previous_gt_relations(out_dir)

    if constrain:
        if not prior:
            raise RuntimeError(
                "--constrain-segments needs a Georgia Tech foot<-shank relationship "
                "from an earlier passing calibration, and none was found in "
                f"{out_dir}. Run the calibration normally first; that run records the "
                "relationship, and once you have runs that agree you can constrain "
                "against it."
            )
        name, r_fs_gt_prior = prior[-1]
        constrained, resid = constrain_rotations(stacked, r_fs_gt_prior, r_fs_exo)
        moved = {s: rotation_angle(constrained[s], rotations[s]) for s in SEGMENTS}
        print(f"\n  constrained against {name}: residual {resid:.4f}, rotations moved "
              + ", ".join(f"{s} {moved[s]:.2f} deg" for s in SEGMENTS))
        checks.append(Check(
            "constraint did not distort the fit", max(moved.values()) <= 10.0,
            ", ".join(f"{s} moved {moved[s]:.2f} deg" for s in SEGMENTS)
            + " (max 10.0); a large move means the stored relationship disagrees "
              "with this run's independent fit"))
        metrics["constrained"] = {"source": name, "residual": resid,
                                  "moved_deg": moved}
        rotations = constrained

    r_fs_gt = rotations["foot"].T @ r_fs_exo @ rotations["shank"]
    metrics["derived"] = {"r_foot_from_shank_gt": r_fs_gt.tolist()}
    if prior:
        drifts = [rotation_angle(r_fs_gt, p) for _, p in prior]
        worst = max(drifts)
        checks.append(Check(
            "Georgia Tech segment relationship is stable across runs",
            worst <= GT_RELATION_DRIFT_DEG,
            f"differs from {len(prior)} earlier run(s) by up to {worst:.1f} deg "
            f"(max {GT_RELATION_DRIFT_DEG}); this is fixed hardware, so drift means "
            f"an IMU moved or a phase was done badly", False))
        print(f"  GT relation drift : {worst:.1f} deg vs {len(prior)} earlier run(s)")
    else:
        print(f"  GT relation       : first run, recorded as the baseline")

    result = CalibrationResult(
        stamp=stamp, rotations=rotations,
        encoder_zero_deg=stand_m["encoder_zero_deg"],
        encoder_sign=sweep_m["encoder_sign"],
        encoder_ratio=sweep_m["encoder_ratio"],
        heel_threshold=walk_m["heel_threshold"],
        toe_threshold=walk_m["toe_threshold"],
        checks=checks, metrics=metrics)

    # ---- verdict ----
    print("\n" + "=" * 72)
    print("  CALIBRATION CHECKS")
    print("=" * 72)
    width = max(len(c.name) for c in checks) + 2
    for c in checks:
        tag = "PASS" if c.passed else ("FAIL" if c.critical else "WARN")
        print(f"  [{tag}] {c.name:<{width}} {c.detail}")
    hard = [c for c in checks if not c.passed and c.critical]
    soft = [c for c in checks if not c.passed and not c.critical]
    print("-" * 72)
    print(f"  {len(checks) - len(hard) - len(soft)} passed, {len(hard)} failed, "
          f"{len(soft)} warnings")

    json_path = out_dir / f"calibration_{stamp}.json"
    json_path.write_text(json.dumps(result.to_json(), indent=2,
                                default=_json_default), encoding="utf-8")

    print(f"  calibration    -> {json_path}")

    if hard:
        print("\n  CALIBRATION FAILED. No transform module was written.")
        print("  Fix the failures above and run again; the raw recordings are kept")
        print("  so a borderline run can be re-analysed without re-walking.")
        return result

    module_path = out_dir / f"exo_transform_{stamp}.py"
    emit_transform_module(result, module_path, hf_root=str(hf_root),
                          reference_subject=reference_subject)
    latest = out_dir / "exo_transform_latest.py"
    latest.write_text(module_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"  transform      -> {module_path}")
    print(f"                 -> {latest}  (stable import name)")
    print("\n  CALIBRATION SUCCEEDED.")
    print(f"  Use it with:  from calibration.exo_transform_latest import features")
    return result


def emit_transform_module(result: CalibrationResult, path,
                          hf_root: str = "mrsd-exo-ankle",
                          reference_subject: str = "AB06") -> None:
    """
    Write a self-contained transform module from a calibration result.

    The calibration values are baked in as literals, so the generated file imports
    numpy and nothing else, reads no JSON at run time, and does not depend on
    exo_frame.py. That makes it safe to copy onto the robot on its own.
    """
    path = Path(path)

    def fmt(mat) -> str:
        rows = ",\n".join("        [" + ", ".join(f"{v:+.12f}" for v in row) + "]"
                          for row in np.asarray(mat))
        return "np.array([\n" + rows + "\n    ], dtype=np.float64)"

    passed = [c for c in result.checks if c.passed]
    warned = [c for c in result.checks if not c.passed]
    check_lines = "\n".join(
        f"#   {'PASS' if c.passed else 'WARN'}  {c.name}: {c.detail}"
        for c in result.checks)

    src = f'''#!/usr/bin/env python3
"""
exo_transform_{result.stamp}.py

Generated by exo_frame.run_calibration on {result.stamp}. Do not edit by hand;
re-run the calibration instead.

Self-contained: imports numpy only, holds every calibration value as a literal, and
has no dependency on exo_frame.py. Copy it onto the robot as-is.

Two entry points, both producing the same {N_FEATURES} columns in the same order:

    features(...)          one live exo sample  -> feature vector   (deployment)
    ingest_gt_trial(...)   a Georgia Tech trial -> (X, y)           (training)

Calibration summary
-------------------
#   encoder zero  : {result.encoder_zero_deg:.4f} deg (raw AS5600, at neutral standing)
#   encoder sign  : {result.encoder_sign:+d}
#   encoder ratio : {result.encoder_ratio:.6f} joint deg per encoder deg
#   contact       : heel > {result.heel_threshold:.1f}, toe >= {result.toe_threshold:.1f} counts
#   reference     : {hf_root} subject {reference_subject}
#   checks        : {len(passed)} passed, {len(warned)} not passed

{check_lines}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["FEATURES", "TARGET", "features", "features_batch",
           "ingest_exo_log", "ingest_gt_trial", "rotate_gt", "ankle_angle",
           "CALIBRATION"]

# ========================= Calibration constants =========================
CALIBRATION_STAMP = "{result.stamp}"

G_TO_MS2 = {G_TO_MS2!r}

# Georgia Tech sensor axes -> this exoskeleton's sensor axes. Apply as R @ v.
ROTATION_FOOT = {fmt(result.rotations["foot"])}

ROTATION_SHANK = {fmt(result.rotations["shank"])}

ROTATIONS = {{"foot": ROTATION_FOOT, "shank": ROTATION_SHANK}}

# Raw AS5600 degrees at neutral standing. The exo is re-zeroed every run, so
# override this with set_encoder_zero() if a run starts from a different reference.
ENCODER_ZERO_DEG = {result.encoder_zero_deg!r}
ENCODER_SIGN = {result.encoder_sign!r}
ENCODER_RATIO = {result.encoder_ratio!r}

HEEL_THRESHOLD = {result.heel_threshold!r}
TOE_THRESHOLD = {result.toe_threshold!r}
FSR_FILTER_CUTOFF = {FSR_FILTER_CUTOFF!r}

SEGMENTS = {SEGMENTS!r}
FEATURES = {FEATURES!r}
TARGET = {TARGET!r}
N_FEATURES = len(FEATURES)

CALIBRATION = {{
    "stamp": CALIBRATION_STAMP,
    "encoder_zero_deg": ENCODER_ZERO_DEG,
    "encoder_sign": ENCODER_SIGN,
    "encoder_ratio": ENCODER_RATIO,
    "heel_threshold": HEEL_THRESHOLD,
    "toe_threshold": TOE_THRESHOLD,
    "metrics": {result.metrics!r},
}}

_encoder_zero = ENCODER_ZERO_DEG


# ========================= Deployment =========================
def set_encoder_zero(raw_deg: float) -> None:
    """Override the encoder reference for this run, read at the moment of zeroing."""
    global _encoder_zero
    _encoder_zero = float(raw_deg)


def ankle_angle(raw_deg):
    """Raw AS5600 degrees -> ankle angle in degrees, dorsiflexion positive."""
    d = (np.asarray(raw_deg, dtype=np.float64) - _encoder_zero + 180.0) % 360.0 - 180.0
    return ENCODER_SIGN * ENCODER_RATIO * d


def features(foot_accel, foot_gyro, shank_accel, shank_gyro,
             encoder_deg, toe_fsr, heel_fsr, out=None):
    """
    One live exo sample -> one feature vector. This is the control-loop hot path.

    Inputs are the exo's own units: accel m/s^2, gyro rad/s, encoder raw degrees,
    FSRs raw counts (filter them upstream as the controller does). No rotation is
    applied: the training data was brought into this frame, not the other way round.
    Pass `out`, a ({N_FEATURES},) float64 array, to avoid allocating per sample.
    """
    v = np.empty(N_FEATURES, dtype=np.float64) if out is None else out
    v[0] = foot_accel[0]; v[1] = foot_accel[1]; v[2] = foot_accel[2]
    v[3] = foot_gyro[0]; v[4] = foot_gyro[1]; v[5] = foot_gyro[2]
    v[6] = shank_accel[0]; v[7] = shank_accel[1]; v[8] = shank_accel[2]
    v[9] = shank_gyro[0]; v[10] = shank_gyro[1]; v[11] = shank_gyro[2]
    d = (encoder_deg - _encoder_zero + 180.0) % 360.0 - 180.0
    v[12] = ENCODER_SIGN * ENCODER_RATIO * d
    v[13] = 1.0 if heel_fsr > HEEL_THRESHOLD else 0.0
    v[14] = 1.0 if toe_fsr >= TOE_THRESHOLD else 0.0
    return v


def features_batch(foot_accel, foot_gyro, shank_accel, shank_gyro,
                   encoder_deg, toe_fsr, heel_fsr):
    """Vectorised `features` over (N, 3) blocks and (N,) scalars -> (N, {N_FEATURES})."""
    n = len(encoder_deg)
    out = np.empty((n, N_FEATURES), dtype=np.float64)
    out[:, 0:3] = foot_accel
    out[:, 3:6] = foot_gyro
    out[:, 6:9] = shank_accel
    out[:, 9:12] = shank_gyro
    out[:, 12] = ankle_angle(encoder_deg)
    out[:, 13] = np.asarray(heel_fsr) > HEEL_THRESHOLD
    out[:, 14] = np.asarray(toe_fsr) >= TOE_THRESHOLD
    return out


# ========================= Training-side ingestion =========================
def rotate_gt(accel_g, gyro, segment):
    """Rotate one segment of Georgia Tech data into this exo's frame.

    Accelerations are lifted from g into m/s^2 first; gyro is rad/s on both sides.
    """
    r = ROTATIONS[segment]
    return (np.asarray(accel_g, np.float64) * G_TO_MS2) @ r.T, \\
        np.asarray(gyro, np.float64) @ r.T


def ingest_gt_trial(root, subject, trial):
    """
    One Georgia Tech trial -> (X, y) in this exo's frame, units and column order.

    X is (N, {N_FEATURES}) matching FEATURES; y is the right-ankle moment normalised by
    body mass, in N*m/kg, or None when the trial carries no moment or mass.
    Requires pandas, which is imported here so the deployment path stays numpy-only.
    """
    import pandas as pd

    base = Path(root) / "subjects" / subject
    imu = pd.read_parquet(base / f"{{trial}}__imu.parquet")
    idf = pd.read_parquet(base / f"{{trial}}__id.parquet")
    gon = pd.read_parquet(base / f"{{trial}}__gon.parquet")
    gc = pd.read_parquet(base / f"{{trial}}__gcRight.parquet")

    t = imu["time_s"].to_numpy(np.float64)
    n = len(t)
    x = np.empty((n, N_FEATURES), dtype=np.float64)

    for i, seg in enumerate(SEGMENTS):
        a, g = rotate_gt(imu[[f"{{seg}}_Accel_{{c}}" for c in "XYZ"]].to_numpy(),
                         imu[[f"{{seg}}_Gyro_{{c}}" for c in "XYZ"]].to_numpy(), seg)
        x[:, 6 * i:6 * i + 3] = a
        x[:, 6 * i + 3:6 * i + 6] = g

    # The goniometer is already an anatomical angle, so it needs the sign convention
    # but no rotation.
    x[:, 12] = ENCODER_SIGN * np.interp(t, gon["time_s"].to_numpy(),
                                        gon["ankle_sagittal"].to_numpy())

    hs = np.where(np.diff(gc["HeelStrike"].to_numpy()) < -50.0)[0] + 1
    to = np.where(np.diff(gc["ToeOff"].to_numpy()) < -50.0)[0] + 1
    heel = np.zeros(n)
    toe = np.zeros(n)
    for i in hs:
        nxt = to[to > i]
        end = int(nxt[0]) if len(nxt) else n
        heel[i:end] = 1.0
        toe[i + (end - i) // 3:end] = 1.0
    x[:, 13] = heel
    x[:, 14] = toe

    meta = pd.read_parquet(Path(root) / "metadata.parquet")
    row = meta[(meta["subject"] == subject) & (meta["trial"] == trial)]
    mass = float(row["weight_kg"].iloc[0]) if len(row) else None
    moment = idf["ankle_angle_r_moment"].to_numpy(np.float64)
    y = moment / mass if mass else None
    return x, y


def ingest_exo_log(path):
    """
    One exo `data_collection_*.csv` -> (X, timestamps) using this calibration.

    The per-run encoder zero is recovered from the file's startup window, where the
    encoder is already reporting but the IMUs have not yet produced a packet.
    Requires pandas.
    """
    import pandas as pd

    raw = pd.read_csv(path)
    lead = int(np.argmax(raw["foot_ax"].notna().to_numpy()))
    window = raw["ankle_encoder_deg"].iloc[:lead].dropna()
    if len(window) == 0:
        window = raw["ankle_encoder_deg"].dropna().iloc[:1]
    zero = float(window.mean())

    df = raw.interpolate(limit_direction="both")
    t = df["timestamp_s"].to_numpy(np.float64)
    fs = float(1.0 / np.median(np.diff(t)))

    def lp(x):
        dt = 1.0 / fs
        a = 2 * np.pi * FSR_FILTER_CUTOFF * dt / (2 * np.pi * FSR_FILTER_CUTOFF * dt + 1)
        y = np.empty_like(x)
        acc = float(x[0])
        for i in range(len(x)):
            acc = a * float(x[i]) + (1.0 - a) * acc
            y[i] = acc
        return y

    prev = _encoder_zero
    try:
        set_encoder_zero(zero)
        x = features_batch(
            df[[f"foot_a{{c}}" for c in "xyz"]].to_numpy(np.float64),
            df[[f"foot_g{{c}}" for c in "xyz"]].to_numpy(np.float64),
            df[[f"shank_a{{c}}" for c in "xyz"]].to_numpy(np.float64),
            df[[f"shank_g{{c}}" for c in "xyz"]].to_numpy(np.float64),
            df["ankle_encoder_deg"].to_numpy(np.float64),
            lp(df["toe_fsr_raw"].to_numpy(np.float64)),
            lp(df["heel_fsr_raw"].to_numpy(np.float64)))
    finally:
        set_encoder_zero(prev)
    return x, t


if __name__ == "__main__":
    print(f"exo transform, calibration {{CALIBRATION_STAMP}}")
    print(f"{{N_FEATURES}} features: {{', '.join(FEATURES)}}")
    for _s in SEGMENTS:
        print(f"\\nR[{{_s}}] =")
        for _row in ROTATIONS[_s]:
            print("    [" + "  ".join(f"{{_v:+.6f}}" for _v in _row) + "]")
    print(f"\\nencoder: zero={{ENCODER_ZERO_DEG:.4f}} deg, sign={{ENCODER_SIGN:+d}}, "
          f"ratio={{ENCODER_RATIO:.6f}}")
    print(f"contact: heel>{{HEEL_THRESHOLD:.1f}}, toe>={{TOE_THRESHOLD:.1f}}")
'''
    path.write_text(src, encoding="utf-8")


# ========================= CLI =========================
def _cmd_fit(args) -> int:
    print("Fitting HF -> exo rotations")
    cal = ExoFrame.fit(args.hf_root, args.exo_dir, fsr_mode=args.fsr_mode,
                       exclude=args.exclude)
    for seg in SEGMENTS:
        print(f"\n  R[{seg}] =")
        for row in cal.rotations[seg]:
            print("      [" + "  ".join(f"{v:+.4f}" for v in row) + "]")
    cal.save(args.out)
    print(f"\nSaved -> {args.out}")
    return 0


def _cmd_build(args) -> int:
    import pandas as pd

    cal = (ExoFrame.load(args.out) if Path(args.out).exists()
           else ExoFrame.fit(args.hf_root, args.exo_dir, fsr_mode=args.fsr_mode,
                             exclude=args.exclude))
    if not Path(args.out).exists():
        cal.save(args.out)

    root, dest = Path(args.hf_root), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    subjects = sorted(p.name for p in (root / "subjects").iterdir() if p.is_dir())

    rows = 0
    for subject in subjects:
        (dest / subject).mkdir(exist_ok=True)
        for name in list_hf_trials(root, subject):
            trial = read_hf_trial(root, subject, name)
            x, y = cal.transform_hf(trial)
            df = pd.DataFrame(x, columns=list(FEATURES))
            if y is not None:
                df[TARGET] = y
            df.to_parquet(dest / subject / f"{name}.parquet", index=False)
            rows += len(df)
            print(f"  {subject}/{name:18s} {len(df):6d} rows")
    print(f"\n{rows} rows -> {dest}")
    return 0


def _cmd_bench(args) -> int:
    import time

    cal = ExoFrame.load(args.out) if Path(args.out).exists() else ExoFrame.fit(
        args.hf_root, args.exo_dir, verbose=False)
    cal.set_encoder_zero(80.0)

    fa = np.array([10.9, -2.5, -1.1]); fg = np.array([0.01, 0.05, -0.14])
    sa = np.array([9.3, 1.9, -1.5]);   sg = np.array([-0.05, 0.01, -0.11])
    buf = np.empty(N_FEATURES)

    for label, fn in (("allocating", lambda: cal.features(fa, fg, sa, sg, 81.3, 900.0, 12000.0)),
                      ("into buffer", lambda: cal.features(fa, fg, sa, sg, 81.3, 900.0,
                                                           12000.0, out=buf))):
        for _ in range(2000):
            fn()
        n = 200_000
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        dt = (time.perf_counter() - t0) / n
        print(f"  features ({label:11s}): {dt * 1e6:7.3f} us/sample  "
              f"-> {1e-3 / dt:8.1f} kHz")

    n = 200_000
    big = {k: np.tile(v, (n, 1)) for k, v in
           (("fa", fa), ("fg", fg), ("sa", sa), ("sg", sg))}
    enc = np.full(n, 81.3); tf_ = np.full(n, 900.0); hf_ = np.full(n, 12000.0)
    t0 = time.perf_counter()
    cal.features_batch(big["fa"], big["fg"], big["sa"], big["sg"], enc, tf_, hf_)
    dt = (time.perf_counter() - t0) / n
    print(f"  features_batch            : {dt * 1e6:7.3f} us/sample  "
          f"-> {1e-3 / dt:8.1f} kHz")
    return 0


def _cmd_calibrate(args) -> int:
    try:
        result = run_calibration(hf_root=args.hf_root, out_dir=args.calib_dir,
                                 reference_subject=args.reference_subject,
                                 imu_report_hz=args.imu_hz,
                                 interactive=not args.no_prompt,
                                 constrain=args.constrain_segments)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"\nCalibration could not start.\n\n{exc}")
        return 1
    return 0 if result.ok else 1


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("fit", "build", "bench", "calibrate"))
    p.add_argument("--hf-root", default="mrsd-exo-ankle")
    p.add_argument("--exo-dir", default="Data collection/data")
    p.add_argument("--out", default="exo_frame_calibration.json")
    p.add_argument("--dest", default="training_data")
    p.add_argument("--fsr-mode", choices=("fixed", "adaptive"), default="fixed")
    p.add_argument("--exclude", nargs="*", default=["20260404_214949"],
                   help="exo filename substrings to skip")
    p.add_argument("--calib-dir", default="calibration",
                   help="where calibration runs and generated transforms are written")
    p.add_argument("--reference-subject", dest="reference_subject", default="AB06",
                   help="Georgia Tech subject used as the rotation reference")
    p.add_argument("--imu-hz", type=float, default=200.0,
                   help="IMU report rate to force during calibration")
    p.add_argument("--no-prompt", action="store_true",
                   help="skip the ENTER confirmation before each phase")
    p.add_argument("--constrain-segments", action="store_true",
                   help="force the foot and shank rotations to stay mutually "
                        "consistent, using the Georgia Tech segment "
                        "relationship recorded by an earlier passing run")
    args = p.parse_args()
    return {"fit": _cmd_fit, "build": _cmd_build, "bench": _cmd_bench,
            "calibrate": _cmd_calibrate}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
