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


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("fit", "build", "bench"))
    p.add_argument("--hf-root", default="mrsd-exo-ankle")
    p.add_argument("--exo-dir", default="Data collection/data")
    p.add_argument("--out", default="exo_frame_calibration.json")
    p.add_argument("--dest", default="training_data")
    p.add_argument("--fsr-mode", choices=("fixed", "adaptive"), default="fixed")
    p.add_argument("--exclude", nargs="*", default=["20260404_214949"],
                   help="exo filename substrings to skip")
    args = p.parse_args()
    return {"fit": _cmd_fit, "build": _cmd_build, "bench": _cmd_bench}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
