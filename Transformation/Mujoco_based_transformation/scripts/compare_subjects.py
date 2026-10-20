"""Compare foot and shank IMU placement across subjects (Camargo dataset).

Reads results/<subject>/{summary,per_trial}.json from locate_camargo.py and
writes results/comparison/{subject_comparison.md, .csv, .png}.

  python scripts/compare_subjects.py --subjects AB06 AB07 AB08 AB09 AB10
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imu_locator.osim import segment_lengths as _segment_lengths  # noqa: E402

IMUS = ("foot", "shank")
MODES = ("treadmill", "levelground", "ramp", "stair")


def segment_lengths(subject_dir):
    return _segment_lengths(next((Path(subject_dir) / "osimxml").glob("*.osim")))


def angles(name, R):
    """Intuitive orientation angles [deg] from sensor->segment R (segment x fwd, y left, z up)."""
    ax, az = R[:, 0], R[:, 2]
    if name == "foot":
        return dict(pitch_down=np.degrees(np.arcsin(-ax[2])),          # sensor x below horizontal
                    yaw_medial=np.degrees(np.arctan2(ax[1], ax[0])),     # sensor x toward +y (medial, right foot)
                    face_tilt=np.degrees(np.arccos(np.clip(az[2], -1, 1))))  # sensor z from vertical
    return dict(axis_tilt=np.degrees(np.arccos(np.clip(-ax[2], -1, 1))),   # sensor x from the shank axis
                face_dir_medial=np.degrees(np.arctan2(az[1], az[0])),   # face normal around the shin; 0 = straight forward
                face_tilt_up=np.degrees(np.arcsin(az[2])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["AB06", "AB07", "AB08", "AB09", "AB10"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/comparison")
    ap.add_argument("--camargo-root", default=".", help="folder with the raw Camargo subject folders")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    for s in a.subjects:
        summ = json.loads((Path(a.results) / s / "summary.json").read_text())
        per = json.loads((Path(a.results) / s / "per_trial.json").read_text())
        data[s] = dict(summary=summ, per=per, lengths=segment_lengths(Path(a.camargo_root) / s))

    rows, stats = [], {}
    for name in IMUS:
        Rs, P, Pn = [], [], []
        for s in a.subjects:
            v, L = data[s]["summary"][name], data[s]["lengths"]
            p = np.array(v["position_m_mujoco_median"])
            iqr = np.array(v["position_spread_m_iqr"])
            R = np.array(v["R_sensor_to_segment_mujoco"])
            ref_len = L["foot"] if name == "foot" else L["tibia"]
            # activity dependence: range of per-activity medians over kept trials
            kept = [r for r in data[s]["per"][name] if r["trial"] not in v["rejected_trials"]]
            mode_med = [np.median([r["r"] for r in kept if r["mode"] == m], axis=0)
                        for m in MODES if any(r["mode"] == m for r in kept)]
            act_range = np.ptp(np.array(mode_med), axis=0) if len(mode_med) > 1 else np.zeros(3)
            Rs.append(R)
            P.append(p)
            Pn.append(p / ref_len)
            rows.append(dict(
                imu=name, subject=s, segment=v["segment"],
                trials_kept=v["n_trials"], trials_total=v["n_trials_total"],
                x_cm=p[0] * 100, y_cm=p[1] * 100, z_cm=p[2] * 100,
                iqr_x_cm=iqr[0] * 100, iqr_y_cm=iqr[1] * 100, iqr_z_cm=iqr[2] * 100,
                activity_range_x_cm=act_range[0] * 100, activity_range_y_cm=act_range[1] * 100,
                activity_range_z_cm=act_range[2] * 100,
                ref_length_cm=ref_len * 100,
                x_frac=p[0] / ref_len, z_frac=p[2] / ref_len,
                orientation_spread_deg=v["rotation_spread_deg_rms"],
                clock_offset_ms=v["time_offset_ms_median"],
                quat_wxyz=" ".join(f"{q:.5f}" for q in v["quat_wxyz_sensor_to_segment_mujoco"]),
                **{k: float(val) for k, val in angles(name, R).items()},
                _R=R))
        Rm = Rot.from_matrix(np.array(Rs)).mean()
        for r in rows:
            if r["imu"] == name:
                r["dev_from_mean_orientation_deg"] = float(
                    np.degrees((Rm.inv() * Rot.from_matrix(r["_R"])).magnitude()))
        P, Pn = np.array(P), np.array(Pn)
        stats[name] = dict(mean=P.mean(0), sd=P.std(0, ddof=1), mean_n=Pn.mean(0), sd_n=Pn.std(0, ddof=1),
                           R_mean=Rm.as_matrix(), angles_mean=angles(name, Rm.as_matrix()))

    # ------------------------------------------------------------- CSV
    fields = [k for k in rows[0] if not k.startswith("_")]
    with open(out / "subject_comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in fields})

    # ------------------------------------------------------------- figure
    n_sub = len(a.subjects)
    fig, axes = plt.subplots(2, 3, figsize=(13, 1.6 + 0.52 * n_sub), sharey=True)
    # one cm-per-width scale for every panel, so equal distances mean equal cm
    half = 0.0
    for name in IMUS:
        for k in range(3):
            key = "xyz"[k]
            sub = [r for r in rows if r["imu"] == name]
            c = stats[name]["mean"][k] * 100
            half = max(half, *[abs(r[f"{key}_cm"] - c) + r[f"iqr_{key}_cm"] / 2 for r in sub],
                       stats[name]["sd"][k] * 100)
    half = float(np.ceil(half * 1.15 * 2) / 2)
    subj = a.subjects
    ypos = np.arange(len(subj))[::-1]
    ink, muted, blue = "#0b0b0b", "#8a8984", "#2a78d6"
    for i, name in enumerate(IMUS):
        for j, (key, lab) in enumerate([("x", "x (forward)"), ("y", "y (left)"), ("z", "z (up)")]):
            ax = axes[i, j]
            sub = [r for r in rows if r["imu"] == name]
            val = np.array([r[f"{key}_cm"] for r in sub])
            iq = np.array([r[f"iqr_{key}_cm"] for r in sub])
            m, sd = stats[name]["mean"]["xyz".index(key)] * 100, stats[name]["sd"]["xyz".index(key)] * 100
            ax.axvspan(m - sd, m + sd, color="#e8e7e3", zorder=0, lw=0)
            ax.axvline(m, color=muted, lw=1, zorder=1)
            ax.hlines(ypos, val - iq / 2, val + iq / 2, color=blue, lw=2, zorder=2)
            ax.plot(val, ypos, "o", ms=8, color=blue, mec="white", mew=2, zorder=3)
            for yv, v in zip(ypos, val if n_sub <= 8 else []):   # many rows: numbers live in the tables
                ax.annotate(f"{v:.1f}", (v, yv), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=8, color=ink)
            ax.set_xlim(m - half, m + half)
            ax.set_title(f"{name} — {lab} [cm]", fontsize=10, color=ink, loc="left")
            ax.set_yticks(ypos, subj)
            ax.set_ylim(-0.7, len(subj) - 0.3)
            ax.grid(axis="x", color="#ebeae6", lw=0.8)
            ax.tick_params(colors="#52514e", labelsize=9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color("#c9c8c2")
    fig.suptitle("IMU position in segment frame per subject (dot = median over trials, bar = IQR; "
                 f"grey band = cross-subject mean ± SD; every panel spans {2 * half:.0f} cm)", fontsize=11, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    fig.savefig(out / "subject_comparison.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------- Markdown
    f1 = lambda v: f"{v:.1f}"
    L = ["# Foot and shank IMU placement: subject comparison", "",
         f"Subjects: {', '.join(subj)} (Camargo et al. 2021). Each subject's MuJoCo model is converted "
         "from their own scaled `.osim`. Positions are medians over the trials that passed the quality "
         "gate, in each segment's frame (**x forward, y left, z up**, cm).", "",
         "![comparison](subject_comparison.png)", ""]
    for name in IMUS:
        seg = rows[[r["imu"] for r in rows].index(name)]["segment"]
        origin = "heel (calcaneus origin)" if name == "foot" else "knee joint centre"
        L += [f"## {name.capitalize()} IMU (`{seg}`, origin = {origin})", "",
              "| Subject | Trials kept | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | "
              "Activity range (x, y, z) cm | Orientation spread | Off mean orientation |",
              "|---|---|---|---|---|---|---|"]
        for r in [r for r in rows if r["imu"] == name]:
            L.append(f"| {r['subject']} | {r['trials_kept']}/{r['trials_total']} | "
                     f"({f1(r['x_cm'])}, {f1(r['y_cm'])}, {f1(r['z_cm'])}) | "
                     f"({f1(r['iqr_x_cm'])}, {f1(r['iqr_y_cm'])}, {f1(r['iqr_z_cm'])}) | "
                     f"({f1(r['activity_range_x_cm'])}, {f1(r['activity_range_y_cm'])}, {f1(r['activity_range_z_cm'])}) | "
                     f"{f1(r['orientation_spread_deg'])}° | {f1(r['dev_from_mean_orientation_deg'])}° |")
        st = stats[name]
        L.append(f"| **Mean ± SD** | | ({f1(st['mean'][0] * 100)} ± {f1(st['sd'][0] * 100)}, "
                 f"{f1(st['mean'][1] * 100)} ± {f1(st['sd'][1] * 100)}, {f1(st['mean'][2] * 100)} ± "
                 f"{f1(st['sd'][2] * 100)}) | | | | |")
        L.append("")
        if name == "foot":
            L += ["Normalised by foot length (heel → MTP joint):", "",
                  "| Subject | Foot length cm | x / foot length | z (cm above heel origin) | "
                  "Sensor x pitch down | Sensor x yaw (medial +) | Face tilt from vertical |",
                  "|---|---|---|---|---|---|---|"]
            for r in [r for r in rows if r["imu"] == name]:
                L.append(f"| {r['subject']} | {f1(r['ref_length_cm'])} | {r['x_frac']:.2f} | {f1(r['z_cm'])} | "
                         f"{f1(r['pitch_down'])}° | {f1(r['yaw_medial'])}° | {f1(r['face_tilt'])}° |")
            am = st["angles_mean"]
            L.append(f"| **Mean ± SD** | | {st['mean_n'][0]:.2f} ± {st['sd_n'][0]:.2f} | | "
                     f"{f1(am['pitch_down'])}° | {f1(am['yaw_medial'])}° | {f1(am['face_tilt'])}° |")
        else:
            L += ["Normalised by tibia length (knee → ankle joint centre):", "",
                  "| Subject | Tibia length cm | Fraction down the shank | x (cm in front of the tibia axis) | "
                  "Sensor x tilt from shank axis | Face direction (0 = forward, medial +) | Face tilt up |",
                  "|---|---|---|---|---|---|---|"]
            for r in [r for r in rows if r["imu"] == name]:
                L.append(f"| {r['subject']} | {f1(r['ref_length_cm'])} | {-r['z_frac']:.2f} | {f1(r['x_cm'])} | "
                         f"{f1(r['axis_tilt'])}° | {f1(r['face_dir_medial'])}° | {f1(r['face_tilt_up'])}° |")
            am = st["angles_mean"]
            L.append(f"| **Mean ± SD** | | {-st['mean_n'][2]:.2f} ± {st['sd_n'][2]:.2f} | | "
                     f"{f1(am['axis_tilt'])}° | {f1(am['face_dir_medial'])}° | {f1(am['face_tilt_up'])}° |")
        L.append("")

    def reason_kind(txt):
        return ("implausible" if "implausibly" in txt else "accel" if "accelerometer" in txt
                else "orientation" if "orientation" in txt else "gyro")

    L += ["## Data quality", "",
          "Rejection reasons: **implausible** = gyro matches the mocap far better than a strapped IMU can "
          "(likely not a raw recording); **accel** = accelerometer not explained (saturation, damped or "
          "corrupt channel); **gyro** = poor gyro fit; **orientation** = fit flipped relative to the "
          "subject's other trials.", "",
          "| Subject | Foot rejected (implausible/accel/gyro/orient.) | Shank rejected (implausible/accel/gyro/orient.) | "
          "Thigh accel sign | Clock offset foot / shank |",
          "|---|---|---|---|---|"]
    for s in subj:
        cells = []
        for n in IMUS:
            rs = data[s]["summary"][n].get("rejection_reasons", {})
            k = [reason_kind(t) for t in rs.values()]
            cells.append(f"{len(rs)} ({k.count('implausible')}/{k.count('accel')}/{k.count('gyro')}/"
                         f"{k.count('orientation')})")
        sf, ss = data[s]["summary"]["foot"], data[s]["summary"]["shank"]
        th = data[s]["summary"].get("thigh", {}).get("acc_sign")
        L.append(f"| {s} | {cells[0]} | {cells[1]} | {'inverted' if th == -1 else 'normal'} | "
                 f"{sf['time_offset_ms_median']:.1f} / {ss['time_offset_ms_median']:.1f} ms |")
    signs = {n: sorted({data[s]["summary"][n]["acc_sign"] for s in subj}) for n in IMUS}
    L += ["",
          f"Foot and shank accelerometer sign: {', '.join(f'{n} {signs[n]}' for n in IMUS)} across all "
          f"{len(subj)} subjects (+1 = normal, +1 g upward at rest). The thigh accelerometer's sign "
          "changes between subjects (column above), so the thigh sensor configuration changed during "
          "data collection.", ""]
    (out / "subject_comparison.md").write_text("\n".join(L))
    print(f"wrote {out}/subject_comparison.md, .csv, .png")


if __name__ == "__main__":
    main()
