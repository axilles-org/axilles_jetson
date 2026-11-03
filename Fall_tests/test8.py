#!/usr/bin/env python3
"""
test8.py -- scoring for Fall Test Plan Test 8 (frame transformation accuracy
and reliability). Checks the DEPLOYED input path: SensorAdapter ->
FeaturePipeline -> the normalisation baked into the deployed ONNX model.

Run with the ML environment (numpy, yaml, onnx, onnxruntime, joblib):
    PY=~/miniconda3/bin/python

Recordings are made with fall_logger.py (system python3), named by condition:
    angle block (step 3), 5 s still per position, both IMUs logged:
        python3 fall_logger.py --test 8 --condition t8_angle_<foot|shank>_<x|y|z>_<0|20|45|90> \
            --sensors imu_foot,imu_shank --seconds 5
    donning (step 4), 20 s: 0-10 s quiet standing, 10-20 s slow hip swings:
        python3 fall_logger.py --test 8 --condition t8_don_<01..10> --seconds 20
    unpowered walk (step 5):
        python3 fall_logger.py --test 8 --condition t8_walk_2.0 --speed 2.0 --seconds 120

Subcommands (each prints PASS/FAIL and rows used/dropped/missing, writes summary.csv)
    $PY test8.py manifest                 steps 1-2: export + compare feature_manifest.csv
    $PY test8.py angle   [runs ...]       step 3: static angle error   (default runs/*t8_angle_*)
    $PY test8.py donning [runs ...]       step 4: orientation spread   (default runs/*t8_don_*)
    $PY test8.py inputs  runs/<walk>      step 5: input-range check against training data
    $PY test8.py gate    runs/<walk>      positive control: inject NaN / inf / missing /
                                          out-of-range frames into ExoController and check
                                          whether they reach the model and whether the
                                          torque command is disabled
manifest/angle/donning/gate write a new folder runs/T8_<timestamp>_<step>/;
inputs writes T8 rows into the walk run's own summary.csv.
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "ML_model" / "src"))

from analyze import Run, Summary, table  # noqa: E402

DEFAULT_MODEL_DIR = ROOT / "ML_model" / "runs" / "deploy"
OUT_ROOT = HERE / "runs"          # --out-root overrides (e.g. for dry runs)

# ── criteria (plan) ─────────────────────────────────────────────────────────
ANGLE_MAX_ERR_DEG = 3.0        # static angle error within +-3 deg at each block angle
DON_MAX_SPREAD_DEG = 5.0       # calibration spread <= 5 deg per IMU axis
DON_REQUIRED = 10              # ten donnings
# ── thresholds this script had to choose (the plan does not give numbers) ───
RANGE_Z_MAX = 4.0              # run-level range check: p0.5..p99.5 of each input
                               # must lie within training mean +- 4 std
GATE_Z_MAX = 6.0               # per-frame: an input beyond 6 std counts as out of range
STILL_GYRO_MAX = 0.10          # rad/s mean |gyro| for a "still" window
SWING_GYRO_MIN = 0.50          # rad/s rms |gyro| for a usable swing window
ACCEL_CLIP = 78.0              # m/s^2, BNO085 +-8 g full scale (78.45)
GYRO_CLIP = 34.5               # rad/s, BNO085 +-2000 deg/s full scale (34.9)
ADC_CLIP = 32760               # ADS1115 positive full scale 32767

IMU_KEYS = {
    "foot": (("foot_ax", "foot_ay", "foot_az"), ("foot_gx", "foot_gy", "foot_gz")),
    "shank": (("shank_ax", "shank_ay", "shank_az"), ("shank_gx", "shank_gy", "shank_gz")),
}
RAW_UNITS = {"a": "m/s^2", "g": "rad/s"}     # BNO085 output, passed through by jetson_io


# ── shared helpers ──────────────────────────────────────────────────────────
def sha(path_or_bytes, n=12):
    b = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    return hashlib.sha256(b).hexdigest()[:n]


def feature_hash(names):
    return sha("\n".join(names).encode())


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


class OutDir:
    """A new run folder for results that combine several recordings."""

    def __init__(self, step, model_dir, notes="", sources=()):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.id = f"T8_{ts}_{step}"
        self.dir = OUT_ROOT / self.id
        self.dir.mkdir(parents=True, exist_ok=False)
        names = load_model(model_dir)["names"]
        meta = {
            "test": "8", "run_id": self.id, "date": datetime.now().isoformat(timespec="seconds"),
            "condition": step, "software_commit": git_commit(),
            "model_dir": str(model_dir), "model_hash": sha(Path(model_dir) / "best.onnx"),
            "feature_list_hash": feature_hash(names), "source_runs": [str(s) for s in sources],
            "notes": notes,
        }
        (self.dir / "run.json").write_text(json.dumps(meta, indent=2))


def load_model(model_dir):
    """Everything the deployed controller uses to build its input."""
    import yaml
    model_dir = Path(model_dir)
    from exo.config import Config
    cfg = Config.load(model_dir / "config.yaml")
    dmeta = json.loads((model_dir / "deploy_metadata.json").read_text())
    with open(model_dir / "config.yaml") as f:
        raw_cfg = yaml.safe_load(f)
    return {"cfg": cfg, "raw_cfg": raw_cfg, "dmeta": dmeta, "names": dmeta["feature_names"]}


def onnx_norm(model_dir):
    """(mean, scale) actually subtracted/divided inside the deployed ONNX graph."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(Path(model_dir) / "best.onnx"))
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    first = [n for n in m.graph.node[:2]]
    if not (first[0].op_type == "Sub" and first[1].op_type == "Div"):
        raise RuntimeError("ONNX graph does not start with Sub/Div normalisation")
    return (init[first[0].input[1]].reshape(-1).astype(np.float64),
            init[first[1].input[1]].reshape(-1).astype(np.float64))


def make_adapter(model_dir, ankle_neutral=None):
    import dataclasses
    from exo.deploy.sensor_adapter import SensorAdapter
    cfg = load_model(model_dir)["cfg"].deploy.sensor_adapter
    if ankle_neutral is not None:
        cfg = dataclasses.replace(cfg, ankle_encoder_neutral_deg=float(ankle_neutral))
    return SensorAdapter(cfg)


def vec3(conv):
    try:
        return np.array([float(x) for x in conv.split(";")], dtype=np.float64)
    except ValueError:
        return np.full(3, np.nan)


def channel_xyz(run, ch, t_lo=None, t_hi=None):
    """(t_ns, Nx3) ok rows of a vector channel, optionally in [t_lo, t_hi] s from start."""
    d = run.ch.get(ch)
    if d is None:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3))
    idx = np.where(d["ok"])[0]
    t = d["t"][idx]
    keep = np.ones(len(idx), bool)
    if t_lo is not None:
        keep &= t >= run.t0 + t_lo * 1e9
    if t_hi is not None:
        keep &= t <= run.t0 + t_hi * 1e9
    idx = idx[keep]
    return d["t"][idx], np.array([vec3(d["conv"][i]) for i in idx]).reshape(-1, 3)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v * np.nan


def angle_deg(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(unit(a), unit(b)))))))


def rotvec_deg(R):
    """Axis-angle vector (deg) of rotation matrix R."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return np.degrees(th) * w


def nearest_rotation(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def find_runs(pattern, given):
    if given:
        return [Path(p) for p in given]
    return sorted(Path(p) for p in glob.glob(str(HERE / "runs" / pattern))
                  if Path(p, "samples.csv").exists())


def finish(s, label):
    path = s.write()
    print(f"\n{label}: {'PASS' if s.ok else 'FAIL'}  -> {path}")
    return 0 if s.ok else 1


# ── steps 1-2: feature manifest ─────────────────────────────────────────────
MANIFEST_HEADER = ["index", "name", "source", "unit", "sign", "scale",
                   "norm_mean", "norm_std", "notes"]


def expected_source(name):
    """Training contract: the tx_mujoco dataset is already expressed in the exo IMU
    frame, so feature imu_<seg>_<Accel|Gyro>_<X> must come from exo channel
    <seg>_<a|g><x> with sign +1 and scale 1."""
    m = re.fullmatch(r"imu_(foot|shank)_(Accel|Gyro)_([XYZ])", name)
    if m:
        seg, q, ax = m.groups()
        k = "a" if q == "Accel" else "g"
        return f"{seg}_{k}{ax.lower()}", RAW_UNITS[k], 1.0, 1.0
    if name == "gon_ankle_sagittal":
        cal = ROOT / "exo_frame_calibration.json"
        sgn = float(json.loads(cal.read_text()).get("encoder_sign", 1.0)) if cal.exists() else 1.0
        return "ankle_encoder_deg", "rad", sgn, math.pi / 180.0
    if name == "stance":
        return "heel_fsr_raw|toe_fsr_raw", "binary", 1.0, 1.0
    return "?", "?", float("nan"), float("nan")


def training_manifest(model_dir):
    import joblib
    md = Path(model_dir)
    mm = json.loads((md / "model_meta.json").read_text())
    names = mm["feature_names"]
    raw_cfg = load_model(md)["raw_cfg"]
    fs = raw_cfg["data"]["feature_set"]
    fs_names = raw_cfg["data"]["feature_sets"][fs]
    sb = joblib.load(md / "scaler_bundle.pkl")
    cols = list(sb["feature_columns"])
    rows, notes = [], []
    if fs_names != names:
        notes.append(f"config feature_set '{fs}' differs from model_meta feature_names")
    for i, n in enumerate(names):
        src, u, sgn, sc = expected_source(n)
        j = cols.index(n) if n in cols else None
        mean = float(sb["x_mean"][j]) if j is not None else float("nan")
        std = float(sb["x_scale"][j]) if j is not None else float("nan")
        rows.append([i, n, src, u, sgn, sc, mean, std,
                     "" if j is not None else "not in training scaler"])
    return rows, notes


def probe_adapter(model_dir):
    """Drive the deployed SensorAdapter with unit inputs and read off, for every
    output feature, which raw channel feeds it and with what sign/scale."""
    adapter = make_adapter(model_dir)
    cfg = adapter.cfg
    raw_keys = [k for seg in IMU_KEYS.values() for grp in seg for k in grp]

    def base():
        f = {k: 0.0 for k in raw_keys}
        f.update(ankle_encoder_deg=cfg.ankle_encoder_neutral_deg, heel_fsr_raw=0.0, toe_fsr_raw=0.0)
        return f

    def run(frame, dt=0.01):
        adapter.reset()
        return adapter.transform_frame(frame, dt)

    out0 = run(base())
    coef = {}                                    # feature -> {raw_key: d_out/d_in}
    for k in raw_keys + ["ankle_encoder_deg"]:
        f = base()
        f[k] += 1.0
        o = run(f)
        for n, v in o.items():
            d = v - out0[n]
            if abs(d) > 1e-12:
                coef.setdefault(n, {})[k] = d
    # stance: one long step so the debounce passes
    for k, thr in (("heel_fsr_raw", cfg.heel_fsr_threshold), ("toe_fsr_raw", cfg.toe_fsr_threshold)):
        f = base()
        f[k] = thr + 1.0
        o = run(f, dt=1.0)
        if o["stance"] != out0["stance"]:
            coef.setdefault("stance", {})[k] = o["stance"] - out0["stance"]
    return coef


def jetson_manifest(model_dir):
    from exo.data.feature_pipeline import FeaturePipeline
    lm = load_model(model_dir)
    names = lm["names"]
    pipe = FeaturePipeline(names)
    order = list(pipe.feature_names)
    mean, scale = onnx_norm(model_dir)
    coef = probe_adapter(model_dir)
    rows, notes = [], []
    if order != names:
        notes.append("FeaturePipeline order differs from deploy_metadata feature_names")
    if len(mean) != len(names):
        notes.append(f"ONNX normaliser has {len(mean)} channels, model input list has {len(names)}")
    for i, n in enumerate(order):
        c = coef.get(n, {})
        note = ""
        if n == "stance":
            src = "|".join(sorted(c)) if c else "none"
            u, sgn, sc = "binary", 1.0, 1.0
        elif c:
            src, d = max(c.items(), key=lambda kv: abs(kv[1]))
            others = {k: v for k, v in c.items() if k != src and abs(v) > 1e-6 * abs(d)}
            if others:
                note = "mixes " + ",".join(f"{k}:{v:+.4f}" for k, v in others.items())
            sgn, sc = float(np.sign(d)), float(abs(d))
            if src == "ankle_encoder_deg":
                u = "rad" if abs(sc - math.pi / 180) < 1e-9 else f"deg*{sc:.6g}"
            else:
                ru = RAW_UNITS[src.split("_")[1][0]]
                u = ru if abs(sc - 1.0) < 1e-9 else f"{ru}*{sc:.6g}"
        else:
            src, u, sgn, sc, note = "none", "?", float("nan"), float("nan"), "no raw input moves it"
        rows.append([i, n, src, u, sgn, sc,
                     float(mean[i]) if i < len(mean) else float("nan"),
                     float(scale[i]) if i < len(scale) else float("nan"), note])
    return rows, notes


def write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_HEADER)
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:+g}", f"{r[5]:.9g}",
                        f"{r[6]:.9g}", f"{r[7]:.9g}", r[8]])


def cmd_manifest(args):
    md = Path(args.model)
    out = OutDir("manifest", md, args.notes)
    s = Summary("T8", out)
    tr, tr_notes = training_manifest(md)
    js, js_notes = jetson_manifest(md)
    write_manifest(out.dir / "feature_manifest_training.csv", tr)
    write_manifest(out.dir / "feature_manifest_jetson.csv", js)
    for n in tr_notes + js_notes:
        print("NOTE:", n)

    def same(a, b, rel):
        if isinstance(a, str) or isinstance(b, str):
            return a == b
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= rel * max(1.0, abs(a), abs(b))

    count_ok = len(tr) == len(js)
    s.add("all", "feature_count", f"{len(tr)} vs {len(js)}", "", "equal", count_ok)
    diffs = []
    fields = [(1, "name", 0), (2, "source", 0), (3, "unit", 0), (4, "sign", 0),
              (5, "scale", 1e-9), (6, "norm_mean", 1e-6), (7, "norm_std", 1e-6)]
    for i in range(max(len(tr), len(js))):
        a = tr[i] if i < len(tr) else None
        b = js[i] if i < len(js) else None
        for col, fname, tol in fields:
            va = a[col] if a else "<absent>"
            vb = b[col] if b else "<absent>"
            if not same(va, vb, tol):
                diffs.append((i, fname, va, vb))
    by_field = {f: sum(1 for d in diffs if d[1] == f) for _, f, _ in fields}
    for f, n in by_field.items():
        s.add("all", f"{f}_differences", n, "", "0", n == 0)
    blocking = (not count_ok) or any(by_field[f] for f in ("name", "unit")) or \
        any(d[1] == "name" for d in diffs)
    s.add("all", "training_feature_hash", feature_hash([r[1] for r in tr]))
    s.add("all", "jetson_feature_hash", feature_hash([r[1] for r in js]))
    s.add("all", "manifests_identical", "yes" if not diffs and count_ok else "no", "",
          "identical", not diffs and count_ok)
    s.add("all", "powered_tcn_blocked", "yes" if blocking else "no", "",
          "plan: block on count/name/order/unit difference")
    for i, f, va, vb in diffs:
        s.add(f"feature_{i}", f"diff_{f}", f"training={va} jetson={vb}")
    print(f"training manifest: {len(tr)} features, hash {feature_hash([r[1] for r in tr])}")
    print(f"jetson   manifest: {len(js)} features, hash {feature_hash([r[1] for r in js])}")
    if diffs:
        table([(i, f, str(va)[:28], str(vb)[:28]) for i, f, va, vb in diffs],
              ("idx", "field", "training", "jetson"))
    else:
        print("no differences in name, order, source, unit, sign, scale or normalisation")
    print(f"rows: {len(tr)} training, {len(js)} jetson, 0 dropped")
    return finish(s, "TEST 8 MANIFEST")


# ── step 3: angle block ─────────────────────────────────────────────────────
ANGLE_RE = re.compile(r"t8_angle_(foot|shank)_([xyz])_(\d+)")


def static_vectors(run, seg, adapter):
    """Mean raw and converted accel (and mean |gyro|) of one IMU over a still run."""
    _, a = channel_xyz(run, f"imu_{seg}_accel")
    _, g = channel_xyz(run, f"imu_{seg}_gyro")
    a = a[np.isfinite(a).all(1)]
    g = g[np.isfinite(g).all(1)]
    raw = a.mean(0) if len(a) else np.full(3, np.nan)
    gm = g.mean(0) if len(g) else np.zeros(3)
    ch = getattr(adapter, seg)
    ch.reset()
    conv, _ = ch.transform(raw, gm, 0.01)
    still = float(np.linalg.norm(g, axis=1).mean()) if len(g) else float("nan")
    return raw, conv, len(a), still


def cmd_angle(args):
    runs = find_runs("*t8_angle_*", args.runs)
    if not runs:
        sys.exit("no t8_angle_* runs found")
    adapter = make_adapter(args.model)
    rec = {}
    vals = []
    for p in runs:
        run = Run(p)
        m = ANGLE_RE.search(run.meta["condition"])
        if not m:
            print(f"skip {p}: condition {run.meta['condition']!r} not t8_angle_<imu>_<axis>_<deg>")
            continue
        seg, ax, deg = m.group(1), m.group(2), int(m.group(3))
        raw, conv, n, still = static_vectors(run, seg, adapter)
        rec[(seg, ax, deg)] = (run, raw, conv, n, still)
        vals.append([run.id, seg, ax, deg, n, *np.round(raw, 5), *np.round(conv, 5),
                     round(float(np.linalg.norm(conv)), 4), round(still, 4)])
    out = OutDir("angle", args.model, args.notes, runs)
    with open(out.dir / "angle_values.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "imu", "axis", "block_deg", "n_samples", "raw_x", "raw_y", "raw_z",
                    "conv_x", "conv_y", "conv_z", "conv_norm", "mean_abs_gyro_rad_s"])
        w.writerows(vals)
    s = Summary("T8", out)
    rows = []
    for seg in ("foot", "shank"):
        for ax in "xyz":
            got = sorted(d for (sg, a, d) in rec if sg == seg and a == ax)
            if not got:
                continue
            if 0 not in got:
                s.add(f"{seg}_{ax}", "reference_0deg", "missing", "", "present", False)
                rows.append((seg, ax, "-", "no 0 deg run", "", "", "FAIL"))
                continue
            g0 = rec[(seg, ax, 0)][2]
            e_ax = np.eye(3)["xyz".index(ax)]
            for d in got:
                _, _, gd, n, still = rec[(seg, ax, d)]
                still_ok = still <= STILL_GYRO_MAX
                if d == 0:
                    s.add(f"{seg}_{ax}_0", "gravity_norm", float(np.linalg.norm(gd)), "m/s^2")
                    rows.append((seg, ax, 0, "ref", f"{np.linalg.norm(gd):.3f}", "",
                                 "" if still_ok else "MOVING"))
                    continue
                meas = angle_deg(g0, gd)
                err = meas - d
                ok = abs(err) <= ANGLE_MAX_ERR_DEG and still_ok
                n_rot = np.cross(unit(g0), unit(gd))
                mis = (math.degrees(math.acos(min(1.0, abs(float(np.dot(unit(n_rot), e_ax))))))
                       if np.linalg.norm(n_rot) > 1e-6 else float("nan"))   # no rotation seen
                ch = f"{seg}_{ax}_{d}"
                s.add(ch, "measured_angle", meas, "deg")
                s.add(ch, "angle_error", err, "deg", f"+-{ANGLE_MAX_ERR_DEG}", ok)
                s.add(ch, "rotation_axis_misalignment", mis, "deg")
                s.add(ch, "mean_abs_gyro", still, "rad/s", f"<={STILL_GYRO_MAX}")
                rows.append((seg, ax, d, f"{meas:.2f}", f"{err:+.2f}", f"{mis:.1f}",
                             "PASS" if ok else "FAIL"))
    checked = {(sg, a) for (sg, a, _) in rec}
    for seg in ("foot", "shank"):
        for ax in "xyz":
            full = all((seg, ax, d) in rec for d in (0, 20, 45, 90))
            s.add(f"{seg}_{ax}", "positions_recorded",
                  sum((seg, ax, d) in rec for d in (0, 20, 45, 90)), "", "4", full)
    print(f"angle-block runs: {len(rec)} used, {len(runs) - len(rec)} skipped; "
          f"axes checked: {sorted(checked)}")
    table(rows, ("imu", "axis", "block_deg", "measured", "error", "axis_mis", "result"))
    print("(axis_mis: angle between the measured rotation axis and the named model axis)")
    return finish(s, "TEST 8 ANGLE BLOCK")


# ── step 4: donnings ────────────────────────────────────────────────────────
def donning_frame(run, seg, adapter, still_win, swing_win):
    """Body-fixed directions (vertical, swing axis, forward) in the converted IMU
    frame, as the columns of a rotation matrix. Swing axis may be missing."""
    ch = getattr(adapter, seg)
    _, a = channel_xyz(run, f"imu_{seg}_accel", *still_win)
    _, gs = channel_xyz(run, f"imu_{seg}_gyro", *still_win)
    _, gw = channel_xyz(run, f"imu_{seg}_gyro", *swing_win)
    a = a[np.isfinite(a).all(1)]
    gs = gs[np.isfinite(gs).all(1)]
    gw = gw[np.isfinite(gw).all(1)]
    rot = lambda v: ch.R @ (ch.s_g * v)
    g = ch.R @ (ch.s_a * a.mean(0)) if len(a) else np.full(3, np.nan)
    still = float(np.linalg.norm(gs, axis=1).mean()) if len(gs) else float("nan")
    axis, swing_rms = None, 0.0
    if len(gw):
        W = np.array([rot(v) for v in gw])
        swing_rms = float(np.sqrt((W ** 2).sum(1).mean()))
        if swing_rms >= SWING_GYRO_MIN:
            axis = np.linalg.svd(W, full_matrices=False)[2][0]
            axis = axis if axis[np.argmax(np.abs(axis))] > 0 else -axis
    return g, axis, still, swing_rms, len(a)


def cmd_donning(args):
    runs = find_runs("*t8_don_*", args.runs)
    if not runs:
        sys.exit("no t8_don_* runs found")
    adapter = make_adapter(args.model)
    still_win, swing_win = tuple(args.still), tuple(args.swing)
    out = OutDir("donning", args.model, args.notes, runs)
    s = Summary("T8", out)
    s.add("all", "donnings", len(runs), "", f">={DON_REQUIRED}", len(runs) >= DON_REQUIRED)
    per = {"foot": [], "shank": []}
    enc = []
    vals = []
    for p in runs:
        run = Run(p)
        d = run.ch.get("encoder")
        if d is not None:
            m = d["ok"] & (d["t"] >= run.t0 + still_win[0] * 1e9) & (d["t"] <= run.t0 + still_win[1] * 1e9)
            e = np.array([float(d["conv"][i]) for i in np.where(m)[0]])
            enc.append(float(np.median(e)) if len(e) else float("nan"))
        for seg in per:
            g, axis, still, swing, n = donning_frame(run, seg, adapter, still_win, swing_win)
            per[seg].append((run.id, g, axis, still, swing))
            vals.append([run.id, seg, n, *np.round(g, 5),
                         *(np.round(axis, 5) if axis is not None else ["", "", ""]),
                         round(still, 4), round(swing, 4)])
    with open(out.dir / "donning_values.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "imu", "n_still", "g_x", "g_y", "g_z", "swing_axis_x",
                    "swing_axis_y", "swing_axis_z", "still_mean_abs_gyro", "swing_rms_gyro"])
        w.writerows(vals)
    rows = []
    for seg, lst in per.items():
        bad_still = [rid for rid, g, a, st, sw in lst if not st <= STILL_GYRO_MAX]
        s.add(seg, "not_still_donnings", len(bad_still), "", "0", not bad_still)
        full = [x for x in lst if x[2] is not None and np.isfinite(x[1]).all()]
        if len(full) == len(lst) and len(lst) >= 2:
            # full 3-axis orientation from gravity + swing axis
            Ms = []
            for _, g, a, _, _ in full:
                e1 = unit(g)
                e2 = unit(a - np.dot(a, e1) * e1)
                Ms.append(np.column_stack([e1, e2, np.cross(e1, e2)]))
            Mref = nearest_rotation(sum(Ms))
            rv = np.array([rotvec_deg(M @ Mref.T) for M in Ms])
            for k, ax in enumerate("XYZ"):
                spread = float(rv[:, k].max() - rv[:, k].min())
                ok = spread <= DON_MAX_SPREAD_DEG
                s.add(f"{seg}_{ax}", "orientation_spread", spread, "deg",
                      f"<={DON_MAX_SPREAD_DEG}", ok)
                rows.append((seg, ax, f"{spread:.2f}", "rotation about axis", "PASS" if ok else "FAIL"))
            s.add(seg, "max_deviation_from_mean", float(np.linalg.norm(rv, axis=1).max()), "deg")
        else:
            # gravity only: tilt of each axis from vertical; rotation about gravity unobservable
            gs = np.array([unit(g) for _, g, _, _, _ in lst])
            tilt = np.degrees(np.arccos(np.clip(gs, -1, 1)))
            s.add(seg, "method", "gravity-only (swing missing or weak); heading not observable")
            for k, ax in enumerate("XYZ"):
                spread = float(np.nanmax(tilt[:, k]) - np.nanmin(tilt[:, k]))
                ok = spread <= DON_MAX_SPREAD_DEG
                s.add(f"{seg}_{ax}", "tilt_from_vertical_spread", spread, "deg",
                      f"<={DON_MAX_SPREAD_DEG}", ok)
                rows.append((seg, ax, f"{spread:.2f}", "tilt from vertical", "PASS" if ok else "FAIL"))
    enc = np.array(enc, dtype=float)
    if len(enc) and np.isfinite(enc).any():
        s.add("encoder", "standing_angle_spread", float(np.nanmax(enc) - np.nanmin(enc)), "deg")
        s.add("encoder", "standing_angle_median", float(np.nanmedian(enc)), "deg")
    print(f"donnings: {len(runs)} (plan: {DON_REQUIRED}); still window {still_win} s, "
          f"swing window {swing_win} s")
    table(rows, ("imu", "axis", "spread_deg", "metric", "result"))
    if len(enc):
        print(f"encoder at quiet standing: median {np.nanmedian(enc):.2f} deg, "
              f"spread {np.nanmax(enc) - np.nanmin(enc):.2f} deg")
    return finish(s, "TEST 8 DONNING")


# ── step 5: model inputs on a walk ──────────────────────────────────────────
CH_KEYS = {
    "imu_foot_accel": IMU_KEYS["foot"][0], "imu_foot_gyro": IMU_KEYS["foot"][1],
    "imu_shank_accel": IMU_KEYS["shank"][0], "imu_shank_gyro": IMU_KEYS["shank"][1],
}


def loop_frames(run):
    """Per control loop: the raw frame dict the controller would have received
    (built from the sample IDs in control.csv), plus raw values for clip checks."""
    head, a = run.control()
    chans = [h[len("sid_"):] for h in head[3:]]
    lut = {}
    for name, d in run.ch.items():
        for i, sid in enumerate(d["sid"]):
            lut[int(sid)] = (name, i)
    frames, raws, missing = [], [], 0
    for row in a:
        f, r, miss = {}, {}, False
        for ch, sid in zip(chans, row[3:]):
            hit = lut.get(int(sid)) if sid >= 0 else None
            ok = hit is not None and run.ch[hit[0]]["ok"][hit[1]]
            miss |= not ok
            if ch in CH_KEYS:
                v = vec3(run.ch[ch]["conv"][hit[1]]) if ok else np.full(3, np.nan)
                for k, x in zip(CH_KEYS[ch], v):
                    f[k] = float(x)
                r[ch] = v
            elif ch == "encoder":
                f["ankle_encoder_deg"] = float(run.ch[ch]["conv"][hit[1]]) if ok else float("nan")
            elif ch in ("fsr_heel", "fsr_toe"):
                v = float(run.ch[ch]["raw"][hit[1]]) if ok else float("nan")
                f[f"{ch.split('_')[1]}_fsr_raw"] = v
                r[ch] = v
        missing += miss
        frames.append(f)
        raws.append(r)
    dt = np.diff(a[:, 1]) / 1e9
    return frames, raws, missing, float(np.median(dt)) if len(dt) else 0.01, a


def ankle_neutral(run, frames, model_dir, mode, value):
    if value is not None:
        return value, "given"
    if mode == "auto":
        ref = json.loads((Path(model_dir) / "ankle_reference.json").read_text())["gatech_walking_median_deg"]
        sign = load_model(model_dir)["cfg"].deploy.sensor_adapter.ankle_encoder_sign
        n = int(10.0 * 100)
        e = np.array([f["ankle_encoder_deg"] for f in frames[:n]], dtype=float)
        return float(np.nanmedian(e)) - ref * sign, "auto (first 10 s, as jetson_deploy --ankle-zero auto)"
    return load_model(model_dir)["cfg"].deploy.sensor_adapter.ankle_encoder_neutral_deg, "config.yaml"


def model_inputs(frames, adapter, names, dt):
    from exo.data.feature_pipeline import FeaturePipeline
    pipe = FeaturePipeline(names)
    adapter.reset()
    X = np.zeros((len(frames), len(names)))
    for i, f in enumerate(frames):
        X[i] = pipe.transform_row(adapter.transform_frame(f, dt))
    return X


def cmd_inputs(args):
    run = Run(args.run)
    md = Path(args.model)
    names = load_model(md)["names"]
    mean, scale = onnx_norm(md)
    frames, raws, missing_loops, dt, _ = loop_frames(run)
    neutral, how = ankle_neutral(run, frames, md, args.ankle_zero, args.ankle_neutral)
    X = model_inputs(frames, make_adapter(md, neutral), names, dt)
    s = Summary("T8", run)
    n = len(X)
    nonfinite = ~np.isfinite(X)
    Z = (X - mean) / scale
    s.add("all", "loops", n)
    s.add("all", "ankle_neutral_deg", neutral, "deg", how)
    s.add("all", "missing_input_loops", missing_loops, "", "0", missing_loops == 0)
    s.add("all", "nonfinite_input_frames", int(nonfinite.any(1).sum()), "", "0",
          not nonfinite.any())
    # clipping at the raw sensor level
    clip = {}
    for ch, lim in (("imu_foot_accel", ACCEL_CLIP), ("imu_shank_accel", ACCEL_CLIP),
                    ("imu_foot_gyro", GYRO_CLIP), ("imu_shank_gyro", GYRO_CLIP),
                    ("fsr_heel", ADC_CLIP), ("fsr_toe", ADC_CLIP)):
        v = np.array([r.get(ch, np.nan) for r in raws], dtype=object)
        cnt = 0
        for x in v:
            x = np.atleast_1d(np.asarray(x, dtype=float))
            cnt += int(np.any(np.abs(x[np.isfinite(x)]) >= lim))
        clip[ch] = cnt
        s.add(ch, "clipped_frames", cnt, "", "0", cnt == 0)
    rows, range_ok = [], True
    gate_frames = (nonfinite.any(1) | (np.abs(np.nan_to_num(Z, nan=np.inf)) > GATE_Z_MAX).any(1))
    for j, name in enumerate(names):
        x = X[np.isfinite(X[:, j]), j]
        z = Z[np.isfinite(Z[:, j]), j]
        if not len(x):
            rows.append((name, "-", "-", "-", "-", "-", "FAIL"))
            range_ok = False
            continue
        lo, hi = np.percentile(z, [0.5, 99.5])
        out_pct = 100.0 * np.mean(np.abs(z) > RANGE_Z_MAX)
        ok = name == "stance" or (lo >= -RANGE_Z_MAX and hi <= RANGE_Z_MAX)
        range_ok &= ok
        s.add(name, "min", float(x.min()))
        s.add(name, "max", float(x.max()))
        s.add(name, "train_mean", float(mean[j]))
        s.add(name, "train_std", float(scale[j]))
        s.add(name, "z_p0.5", float(lo))
        s.add(name, "z_p99.5", float(hi), "", f"within +-{RANGE_Z_MAX}", ok)
        s.add(name, "outside_training_range_pct", out_pct, "%", f"|z|>{RANGE_Z_MAX}")
        rows.append((name, f"{x.min():.3f}", f"{x.max():.3f}", f"{mean[j]:.3f}+-{scale[j]:.3f}",
                     f"[{lo:+.2f},{hi:+.2f}]", f"{out_pct:.2f}", "PASS" if ok else "FAIL"))
    check_ok = range_ok and not nonfinite.any() and not any(clip.values()) and missing_loops == 0
    s.add("all", "input_range_check", "PASS" if check_ok else "FAIL", "",
          "pre-actuation check", check_ok)
    s.add("all", "frames_gated_per_frame", int(gate_frames.sum()), "",
          f"|z|>{GATE_Z_MAX} or non-finite")
    print(f"=== TEST 8 INPUTS  {run.id}  ({run.dur:.1f} s, condition {run.meta['condition']}) ===")
    print(f"ankle neutral {neutral:.2f} deg ({how}); dt {dt * 1e3:.2f} ms; loops {n}")
    print(f"missing-input loops {missing_loops}, non-finite input frames {int(nonfinite.any(1).sum())}, "
          f"clipped frames {clip}")
    table(rows, ("input", "min", "max", "train mean+-std", "z p0.5..p99.5", f"%|z|>{RANGE_Z_MAX:g}", "result"))
    print(f"input-range check: {'PASS' if check_ok else 'FAIL'}; frames a per-frame gate "
          f"(|z|>{GATE_Z_MAX:g} or non-finite) would block: {int(gate_frames.sum())}")
    n_ok = sum(int(d["ok"].sum()) for d in run.ch.values())
    print(f"rows: {run.n_rows} total, {n_ok} used, {run.n_rows - n_ok} dropped (failed reads), "
          f"missing channels: {[c for c in run.channels if c not in run.ch] or 'none'}")
    return finish(s, "TEST 8 INPUTS")


# ── positive control: does a bad input reach the model / is actuation disabled? ─
def cmd_gate(args):
    import dataclasses
    from exo.deploy.runtime import ExoController
    md = Path(args.model)
    lm = load_model(md)
    names = lm["names"]
    mean, scale = onnx_norm(md)
    run = Run(args.run)
    frames, _, _, dt, _ = loop_frames(run)
    neutral, how = ankle_neutral(run, frames, md, args.ankle_zero, args.ankle_neutral)
    dcfg = dataclasses.replace(
        lm["cfg"].deploy, backend="onnx",
        sensor_adapter=dataclasses.replace(lm["cfg"].deploy.sensor_adapter,
                                           ankle_encoder_neutral_deg=neutral))
    T = lm["dmeta"]["window_length"]
    frames = [f for f in frames if all(math.isfinite(v) for v in f.values())]

    def controller():
        c = ExoController(md, dcfg, subject_mass_kg=args.mass)
        c.reset()
        return c

    # baseline: find a loop after the window is full where assistance is active
    c = controller()
    base = [c.step(f, dt)["command_nm"] for f in frames[:T + 600]]
    active = [i for i in range(T + 50, len(base)) if abs(base[i]) > 0.5]
    if not active:
        sys.exit("baseline never commands torque > 0.5 Nm; cannot run the positive control")
    k = active[0]
    # kind "invalid": plan criterion "no missing, infinite or NaN input reaches the model"
    # kind "range":   plan criterion "actuation remains disabled whenever the
    #                 input-range check fails" -> torque must go to zero
    cases = [
        ("nan_foot_ax", "invalid", lambda f: {**f, "foot_ax": float("nan")}),
        ("inf_shank_gz", "invalid", lambda f: {**f, "shank_gz": float("inf")}),
        ("missing_key_foot_gx", "invalid", lambda f: {kk: v for kk, v in f.items() if kk != "foot_gx"}),
        ("out_of_range_foot_ax_500", "range", lambda f: {**f, "foot_ax": 500.0}),
        ("out_of_range_encoder_+90deg", "range",
         lambda f: {**f, "ankle_encoder_deg": f["ankle_encoder_deg"] + 90.0}),
    ]
    out = OutDir("gate", md, args.notes, [args.run])
    s = Summary("T8", out)
    s.add("all", "baseline_command_at_injection", base[k], "Nm")
    rows = []
    for name, kind_, bad in cases:
        c = controller()
        for f in frames[:k]:
            c.step(f, dt)
        reached = blocked = None
        err = ""
        try:
            o = c.step(bad(frames[k]), dt)
            w = c.buffer.window()[:, -1]
            z = (w - mean) / scale
            reached = bool((~np.isfinite(w)).any() or (np.abs(z) > GATE_Z_MAX).any())
            cmds = [abs(o["command_nm"])] + [abs(c.step(f, dt)["command_nm"])
                                             for f in frames[k + 1:k + 20]]
            blocked = all(x == 0.0 for x in cmds)
        except Exception as e:          # the controller raising is what jetson_deploy would see
            err = f"{type(e).__name__}: {e}"[:60]
        # an unhandled exception is a FAIL: the controller should reject the frame itself
        if kind_ == "invalid":
            ok = not err and reached is False
        else:
            ok = not err and (reached is False or blocked is True)
        yn = lambda b: "" if b is None else ("yes" if b else "no")
        s.add(name, "bad_input_reaches_model", yn(reached), "", "no",
              (None if reached is None else not reached) if kind_ == "invalid" else None)
        s.add(name, "torque_disabled_next_20_loops", yn(blocked), "", "yes",
              blocked if (kind_ == "range" and blocked is not None) else None)
        if err:
            s.add(name, "controller_exception", err, "", "handled", False)
        s.add(name, "case_result", "PASS" if ok else "FAIL", "", kind_, ok)
        rows.append((name, kind_, yn(reached) or "-", yn(blocked) or "-", err,
                     "PASS" if ok else "FAIL"))
    print(f"=== TEST 8 GATE (positive control) source {run.id}, injection at loop {k} "
          f"(baseline command {base[k]:.2f} Nm), ankle neutral {neutral:.2f} deg ({how}) ===")
    table(rows, ("case", "criterion", "reaches model", "torque off 20 loops", "exception", "result"))
    print("invalid: must not reach the model.  range: torque must be disabled.")
    print("software only: no CAN, no motor")
    return finish(s, "TEST 8 GATE")


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL_DIR), help="deployed model run dir")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out-root", default=None, help="folder for new T8_* result folders")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    p = sub.add_parser("angle")
    p.add_argument("runs", nargs="*")
    p = sub.add_parser("donning")
    p.add_argument("runs", nargs="*")
    p.add_argument("--still", nargs=2, type=float, default=[1.0, 9.0], metavar=("T0", "T1"))
    p.add_argument("--swing", nargs=2, type=float, default=[11.0, 19.0], metavar=("T0", "T1"))
    for name in ("inputs", "gate"):
        p = sub.add_parser(name)
        p.add_argument("run")
        p.add_argument("--ankle-zero", choices=["config", "auto"], default="config")
        p.add_argument("--ankle-neutral", type=float, default=None, help="encoder deg at neutral")
        if name == "gate":
            p.add_argument("--mass", type=float, default=72.0)
    args = ap.parse_args()
    global OUT_ROOT
    if args.out_root:
        OUT_ROOT = Path(args.out_root)
    fn = {"manifest": cmd_manifest, "angle": cmd_angle, "donning": cmd_donning,
          "inputs": cmd_inputs, "gate": cmd_gate}[args.cmd]
    sys.exit(fn(args))


if __name__ == "__main__":
    main()
