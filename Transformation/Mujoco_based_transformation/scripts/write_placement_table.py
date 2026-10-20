"""Write the IMU placement table (Markdown + CSV) from locate_camargo.py output.

  python scripts/write_placement_table.py --results results/AB09 --subject AB09
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imu_locator.osim import segment_lengths  # noqa: E402

def body_place(name, p, L):
    """Plain-language location from position p [m] and segment lengths L [m] (right side)."""
    side = lambda y: "centred" if abs(y) < 0.01 else ("medial" if y > 0 else "lateral")
    if name == "foot":
        return (f"Top of the foot, {p[0] / L['foot'] * 100:.0f}% of the way from heel to toe joint "
                f"(midfoot/laces), {side(p[1])}")
    if name == "shank":
        return f"Front of the shin, {-p[2] / L['tibia'] * 100:.0f}% of the way down, {side(p[1])}"
    if name == "thigh":
        return f"Front of the thigh, {-p[2] / L['femur'] * 100:.0f}% of the way down"
    return "Not reliable (lumbar locked in IK, no torso markers)"
ORIGIN = {"calcn_r": "heel", "tibia_r": "knee centre", "femur_r": "hip centre", "torso": "lumbar joint"}
def axes_text(name, R):
    """Plain-language sensor axes from sensor->segment R."""
    if name == "foot":
        pitch = np.degrees(np.arcsin(-R[2, 0]))
        return f"x forward, ~{pitch:.0f}° down the instep; y left (medial); z out of the top of the foot"
    if name == "shank":
        face = np.degrees(np.arctan2(R[1, 2], R[0, 2]))
        turn = "straight forward" if abs(face) < 5 else f"turned {abs(face):.0f}° {'medially' if face > 0 else 'laterally'}"
        return f"x down the shank; y left (medial); z out of the shin, {turn}"
    if name == "thigh":
        return "x up the thigh; y backward (into the thigh); z right (lateral)"
    return "not reliable"


MODES = ["treadmill", "levelground", "ramp", "stair"]


def cm(v):
    return "(" + ", ".join(f"{x * 100:.1f}" for x in v) + ")"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/AB09")
    ap.add_argument("--subject", default="AB09")
    ap.add_argument("--camargo-root", default=".", help="folder with the raw Camargo subject folders")
    a = ap.parse_args()
    res = Path(a.results)
    summ = json.loads((res / "summary.json").read_text())
    lengths = segment_lengths(next((Path(a.camargo_root) / a.subject / "osimxml").glob("*.osim")))
    place = {n: body_place(n, np.array(v["position_m_mujoco_median"]), lengths) for n, v in summ.items()}
    per = json.loads((res / "per_trial.json").read_text())

    L = [f"# {a.subject} IMU placement", "",
         "Estimated from mocap IK + raw IMU data with the MuJoCo model converted from "
         f"`{a.subject}.osim`. Median over trials (treadmill, levelground, ramp, stair).", "",
         "Position is in each segment's own frame, in cm: **x forward, y left, z up**, "
         "measured from the segment origin. IQR is the spread between trials.", "",
         "## Position", "",
         "| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |",
         "|---|---|---|---|---|---|"]
    rows = []
    for name, s in summ.items():
        seg = s["segment"]
        reliable = name != "trunk"
        pos = cm(s["position_m_mujoco_median"]) if reliable else "not reliable"
        iqr = cm(s["position_spread_m_iqr"]) if reliable else "—"
        L.append(f"| {name.capitalize()} | `{seg}` ({ORIGIN.get(seg, '')}) | {s['n_trials']}/{s['n_trials_total']} | "
                 f"{pos} | {iqr} | {place[name]} |")
        rows.append(dict(
            imu=name, segment=seg, origin=ORIGIN.get(seg, ""), reliable=reliable,
            x_cm=round(s["position_m_mujoco_median"][0] * 100, 2),
            y_cm=round(s["position_m_mujoco_median"][1] * 100, 2),
            z_cm=round(s["position_m_mujoco_median"][2] * 100, 2),
            iqr_x_cm=round(s["position_spread_m_iqr"][0] * 100, 2),
            iqr_y_cm=round(s["position_spread_m_iqr"][1] * 100, 2),
            iqr_z_cm=round(s["position_spread_m_iqr"][2] * 100, 2),
            quat_wxyz_sensor_to_segment=" ".join(f"{q:.5f}" for q in s["quat_wxyz_sensor_to_segment_mujoco"]),
            orientation_spread_deg=s["rotation_spread_deg_rms"],
            clock_offset_ms=s["time_offset_ms_median"],
            acc_sign=s["acc_sign"],
            n_trials=s["n_trials"], n_trials_total=s["n_trials_total"], place_on_body=place[name]))

    L += ["", "## Orientation (sensor axes in the segment frame)", "",
          "| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |",
          "|---|---|---|---|---|"]
    for name, s in summ.items():
        q = ", ".join(f"{v:.4f}" for v in s["quat_wxyz_sensor_to_segment_mujoco"])
        spread = f"{s['rotation_spread_deg_rms']:.1f}°" if name != "trunk" else "—"
        L.append(f"| {name.capitalize()} | {axes_text(name, np.array(s["R_sensor_to_segment_mujoco"]))} | {q if name != 'trunk' else '—'} | {spread} | "
                 f"{s['time_offset_ms_median']:.1f} ms |")
    L += ["", "*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.", ""]

    L += ["## Foot and shank by activity", "",
          "| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |",
          "|---|---|---|---|---|"]
    for name in ("foot", "shank"):
        for mode in MODES:
            rr = np.array([r["r"] for r in per.get(name, []) if r["mode"] == mode])
            if len(rr) == 0:
                continue
            sd = cm(rr.std(0, ddof=1)) if len(rr) > 1 else "—"
            L.append(f"| {name.capitalize()} | {mode} | {len(rr)} | {cm(np.median(rr, 0))} | {sd} |")

    L += ["", "## Notes", "",
          "- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank "
          "IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.",
          "- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, "
          "deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.",
          "- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × "
          "signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see "
          "`summary.json` → `rejected_trials`.",
          "- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.",
          "- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.", ""]
    (res / "imu_placement_table.md").write_text("\n".join(L))

    with open(res / "imu_placement_table.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {res / 'imu_placement_table.md'} and {res / 'imu_placement_table.csv'}")


if __name__ == "__main__":
    main()
