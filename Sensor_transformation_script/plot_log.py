"""
plot_log.py -- see your IMU data raw vs splined.

    python plot_log.py <logfile> [--seg foot] [--secs 6]
"""
import argparse, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import align_io as aio

ap = argparse.ArgumentParser()
ap.add_argument("log"); ap.add_argument("--seg", default="foot")
ap.add_argument("--secs", type=float, default=6.0)
ap.add_argument("--out", default="log_check.png")
a = ap.parse_args()

raw = aio.load_exo(a.log, interpolate=False, verbose=False)
fix = aio.load_exo(a.log, interpolate=True, verbose=False)
fs = raw.fs; n = int(a.secs * fs)
odr = aio.effective_odr(raw.accel[a.seg], fs)

fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
t = np.arange(n) / fs
ax[0].plot(t, raw.gyro[a.seg][:n, 2], lw=.9, label="raw (held)")
ax[0].plot(t, fix.gyro[a.seg][:n, 2], lw=1.3, label="splined")
ax[0].set_ylabel("gyro z (rad/s)"); ax[0].legend(); ax[0].grid(alpha=.3)

ax[1].plot(t, np.gradient(raw.gyro[a.seg][:n, 2], 1/fs), lw=.8, label="raw")
ax[1].plot(t, np.gradient(fix.gyro[a.seg][:n, 2], 1/fs), lw=1.2, label="splined")
ax[1].set_ylabel("alpha z (rad/s²)"); ax[1].legend(); ax[1].grid(alpha=.3)
ax[1].set_title("this is what the lever-arm solve consumes", fontsize=9)

for lbl, rec in (("raw", raw), ("splined", fix)):
    x = rec.gyro[a.seg][:, 2]
    f = np.fft.rfftfreq(len(x), 1/fs); P = np.abs(np.fft.rfft(x - x.mean()))**2
    ax[2].loglog(f[1:], P[1:], lw=.9, label=lbl)
ax[2].axvline(odr/2, color="k", ls="--", lw=.8, label=f"Nyquist {odr/2:.0f} Hz")
ax[2].set_xlabel("Hz"); ax[2].set_ylabel("power"); ax[2].legend(); ax[2].grid(alpha=.3)

fig.suptitle(f"{a.seg}  |  logged {fs:.0f} Hz, effective {odr:.0f} Hz")
fig.tight_layout(); fig.savefig(a.out, dpi=120)
print(f"effective ODR {odr:.1f} Hz -> {a.out}")