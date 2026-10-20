"""Measured vs model-predicted IMU signals, for checking a fit by eye."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .estimate import ImuData, align  # noqa: E402
from .kinematics import predict_imu  # noqa: E402
from .signal_utils import lowpass, sample_rate  # noqa: E402


def plot_fit(seg, imu: ImuData, est, fc, path, title="", seconds=6.0):
    fs = sample_rate(imu.t)
    imu_f = ImuData(imu.t, lowpass(est.acc_sign * imu.acc, fs, fc), lowpass(imu.gyr, fs, fc))
    s, acc, gyr = align(seg, imu_f, est.time_offset, 0.5)
    # same measurement model as the estimator: mocap f, IMU-gyro lever arms
    _, pg = predict_imu(s, est.r, est.R)
    pg = pg + est.gyro_bias
    w_s, r_s = gyr - est.gyro_bias, est.R.T @ est.r
    dw = np.gradient(gyr, s.t, axis=0)
    pa = s.f @ est.R + np.cross(dw, r_s) + np.cross(w_s, np.cross(w_s, r_s)) + est.acc_bias
    t0 = s.t[len(s.t) // 2]
    m = (s.t >= t0) & (s.t < t0 + seconds)
    fig, ax = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for i, c in enumerate("xyz"):
        ax[0, i].plot(s.t[m], acc[m, i], "k", lw=1.2, label="measured")
        ax[0, i].plot(s.t[m], pa[m, i], "C1--", lw=1.2, label="model")
        ax[0, i].set_title(f"acc {c} [m/s²]")
        ax[1, i].plot(s.t[m], gyr[m, i], "k", lw=1.2)
        ax[1, i].plot(s.t[m], pg[m, i], "C0--", lw=1.2)
        ax[1, i].set_title(f"gyro {c} [rad/s]")
        ax[1, i].set_xlabel("mocap time [s]")
    ax[0, 0].legend(frameon=False)
    fig.suptitle(f"{title} on {est.segment}: rms acc {est.rms_acc:.3f} m/s², "
                 f"gyro {est.rms_gyro:.3f} rad/s")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
