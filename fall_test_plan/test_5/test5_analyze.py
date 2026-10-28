#!/usr/bin/env python3
"""
test5_analyze.py — Test 5: end-to-end latency, sensor read to torque command (R-05).

    python3 fall_test_plan/test_5/test5_analyze.py                    # latest Test 4 run
    python3 fall_test_plan/test_5/test5_analyze.py <Test 4 session or load folder>
    python3 fall_test_plan/test_5/test5_analyze.py <...> --out <dir>

Analysis only. It reads the logs a Test 4 load run left behind and never opens
the CAN bus or the I2C bus, so it cannot move the motor.

Inputs (from a Test 4 load run folder):
    loops.csv        per loop: start/end time and the time of every stage
    can_sends.csv    every bus.send() call: start, end, loop it belongs to
    can_monitor.npz  every frame on the CAN bus with its kernel timestamp
                     (our commands as they hit the wire, and the motor's frames)
    meta.json        run settings (torque cap, slew, watchdog, clock pair)

For every torque command in the timed window it rebuilds the path from the end
of the loop's sensor read to the frame leaving on the CAN bus:

    sensor read done ─ preprocessing ─ TCN inference ─ safety checks + packing ─ CAN send ─ on the bus
                       (controller.step               (session ramp, watchdog,   (bus.send()
                        minus the TCN)                 torque cap, slew, MIT       + transmit)
                                                       frame packing)

Script output: median, 99th-percentile and maximum time of each stage and of
the whole path; the slowest stage; the CAN reply age; and a per-command check
that a valid safety-limit result existed before the frame was transmitted.

Writes report.md, report.json, latency_per_command.csv and *.png into
fall_test_plan/test_5/results/<date_time>/ (or --out).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
TEST4_RESULTS = TEST_DIR.parent / "test_4" / "results"

# ── pass criteria (Test 5, R-05) ─────────────────────────────────────────────
MEDIAN_MAX_MS = 10.0
P99_MAX_MS = 20.0
MAX_MAX_MS = 50.0

# ── CAN frame format (same as ML_model/src/exo/deploy/motor.py) ──────────────
MOTOR_ID = 0x68
CMD_ARB_ID = (8 << 8) | MOTOR_ID          # MIT-mode torque command, extended id
MIT_T_MIN, MIT_T_MAX = -22.0, 22.0
MIT_T_LSB = (MIT_T_MAX - MIT_T_MIN) / 4096.0   # one 12-bit torque step, ~0.011 N·m

# Tolerances for the safety recomputation. loops.csv stores torques as float32.
TORQUE_TOL_NM = 1e-4
# The watchdog compares the work time measured just before send_torque(); the
# send itself starts after the clamp, slew and packing, a few tens of µs later.
WATCHDOG_TOL_MS = 1.0
CLOCK_TOL_MS = 0.05                       # wall/monotonic pairing error allowed

STAGES = [
    ("preprocess_ms", "Preprocessing (adapter, features, assist)"),
    ("inference_ms", "TCN inference"),
    ("safety_ms", "Safety checks + frame packing"),
    ("can_send_ms", "CAN send (to bus)"),
]
SAFETY_CHECKS = [
    ("one_frame", "exactly one command frame sent, no send error"),
    ("finite", "assist, target and sent torque are finite numbers"),
    ("watchdog_decided", "watchdog decision matches the work time measured before the send"),
    ("watchdog_zeroes", "a watchdog trip forced the target to 0 N·m"),
    ("session_ramp", "target = assist × session ramp (when the watchdog did not trip)"),
    ("within_cap", "sent torque within the torque cap"),
    ("pipeline", "sent torque = sign, cap and slew limit applied to the target"),
    ("wire_within_cap", "torque decoded from the frame on the bus within the cap"),
    ("wire_matches", "torque decoded from the frame on the bus = logged sent torque"),
    ("wire_no_gains", "frame on the bus has kp = kd = 0 (pure torque command)"),
]

# chart colors: categorical slots 1-4 of the reference palette (validated, fixed
# order), the same palette as Test 4's plots
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SERIES_DARK = "#184f95"                   # slot 1's own ramp, a darker step
SENSOR_GRAY = "#c3c2b7"
INK, INK_2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e6e5e0"
STATUS = {"PASS": "#0ca30c", "FAIL": "#d03b3b"}


def _pct(x, q):
    x = np.asarray(x, dtype=float)
    return float(np.percentile(x, q)) if len(x) else float("nan")


def _summary(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "p50": _pct(x, 50),
            "p99": _pct(x, 99), "max": float(x.max()), "min": float(x.min())}


# ── loading ─────────────────────────────────────────────────────────────────

def _usable(d):
    if not all((d / f).exists() for f in ("meta.json", "loops.csv", "can_sends.csv")):
        return False
    return json.loads((d / "meta.json").read_text()).get("phase") == "load"


def find_source(path):
    """A Test 4 session folder or load-run folder -> the load-run folder.
    No path: the newest Test 4 session that has one."""
    if path is not None:
        p = Path(path).expanduser().resolve()
        for d in (p, p / "load"):
            if _usable(d):
                return d
        sys.exit(f"no Test 4 load run (meta.json, loops.csv, can_sends.csv) in {p}")
    sessions = sorted((p for p in TEST4_RESULTS.iterdir() if p.is_dir()), reverse=True)
    for s in sessions:
        for d in (s / "load", s):
            if _usable(d):
                return d
    sys.exit(f"no Test 4 load run found under {TEST4_RESULTS}")


def load_run(path):
    meta = json.loads((path / "meta.json").read_text())
    loops = pd.read_csv(path / "loops.csv").sort_values("seq").reset_index(drop=True)
    sends = pd.read_csv(path / "can_sends.csv")
    sends["arb"] = sends["arb_id"].apply(lambda s: int(s, 16))
    mon = path / "can_monitor.npz"
    return meta, loops, sends, (dict(np.load(mon)) if mon.exists() else None)


# ── CAN frames ──────────────────────────────────────────────────────────────

def decode_mit(data):
    """MIT command frame bytes (n x 8) -> torque (N·m), kp count, kd count."""
    d = np.asarray(data, dtype=np.int64)
    kp_i = (d[:, 0] << 4) | (d[:, 1] >> 4)
    kd_i = ((d[:, 1] & 0xF) << 8) | d[:, 2]
    t_i = ((d[:, 6] & 0xF) << 8) | d[:, 7]
    return t_i * MIT_T_LSB + MIT_T_MIN, kp_i, kd_i


def match_wire(meta, sends, mon):
    """Pair each successful command send() with its frame on the bus.

    The Jetson's mttcan driver echoes a frame to other sockets only once it has
    been transmitted, and frames leave in FIFO order, so the k-th successful
    send is the k-th command frame the monitor saw. This holds only if the two
    counts agree; otherwise the on-bus time is left out."""
    ok = (sends[(sends["arb"] == CMD_ARB_ID) & (sends["ok"] == 1)]
          .sort_values("idx").reset_index(drop=True))
    info = {"ok_sends": int(len(ok)), "matched": False}
    if mon is None or not len(mon["frame_ts"]):
        info["status"] = "no CAN monitor data: path ends at send() return"
        return ok, info
    fts, farb, ffl = mon["frame_ts"], mon["frame_arb"], mon["frame_flags"]
    is_cmd = (farb == CMD_ARB_ID) & ((ffl & 1) > 0) & ((ffl & 4) == 0)
    info["wire_frames"] = int(is_cmd.sum())
    if is_cmd.sum() != len(ok):
        info["status"] = (f"{int(is_cmd.sum())} command frames on the bus vs {len(ok)} "
                          "successful sends: path ends at send() return")
        return ok, info
    wall0, mono0 = meta["clock_pair"]
    ok["t_wire"] = fts[is_cmd] - (wall0 - mono0)          # into the loop's clock
    ok["wire_nm"], ok["wire_kp"], ok["wire_kd"] = decode_mit(mon["frame_data"][is_cmd])
    # A frame can reach the bus before send() returns (the thread is preempted
    # after the syscall), but never before send() was called.
    lag = (ok["t_wire"] - ok["t_send_start"]).to_numpy() * 1e3
    info.update({"matched": True, "status": "matched (FIFO, equal counts)",
                 "send_start_to_wire_min_ms": float(lag.min())})
    if lag.min() < -CLOCK_TOL_MS:
        info["warning"] = (f"a frame appears on the bus {-lag.min():.3f} ms before send() "
                           "was called: the wall/monotonic clock pairing is off")
    if "clock_pair_end" in meta:
        w1, m1 = meta["clock_pair_end"]
        info["clock_drift_over_run_ms"] = float(((w1 - m1) - (wall0 - mono0)) * 1e3)
    return ok, info


def motor_status(mon, meta, t_cmd_wire, t_send_start):
    """Reply age. The servo-mode AK80-9 broadcasts a status frame (0x29xx) at a
    fixed rate instead of answering each command, so two ages are reported:
      next_status_ms     command on the bus -> the motor's next status frame
      feedback_age_ms    age of the newest status frame when send() started
                         (how old the motor feedback is that a controller could
                         have used for this command)"""
    if mon is None or not len(mon["frame_ts"]):
        return None, None, {"status": "no CAN monitor data"}
    fts, farb, ffl = mon["frame_ts"], mon["frame_arb"], mon["frame_flags"]
    sel = (((ffl & 2) > 0) & ((ffl & 4) == 0) & (farb != CMD_ARB_ID)
           & ((farb & 0xFF) == MOTOR_ID))
    st = fts[sel]
    if not len(st):
        return None, None, {"status": "no frames from the motor"}
    off = meta["clock_pair"][0] - meta["clock_pair"][1]
    out = {"status": "ok", "ids": sorted({f"0x{int(a):X}" for a in farb[sel]})}

    nxt = None
    if t_cmd_wire is not None:
        tw = t_cmd_wire + off
        j = np.searchsorted(st, tw, side="right")
        nxt = np.full(len(tw), np.nan)
        has = j < len(st)
        nxt[has] = (st[j[has]] - tw[has]) * 1e3
    ts = t_send_start + off
    k = np.searchsorted(st, ts, side="right") - 1
    age = np.full(len(ts), np.nan)
    has = k >= 0
    age[has] = (ts[has] - st[k[has]]) * 1e3

    win = (st >= ts.min()) & (st <= ts.max())
    period = np.diff(st[win]) * 1e3
    data = mon["frame_data"][sel][win].astype(np.int64)
    current = ((data[:, 4] << 8) | data[:, 5]).astype(np.int16) * 0.01
    out.update({
        "frames_in_window": int(win.sum()),
        "rate_hz": float(win.sum() / (ts.max() - ts.min())) if ts.max() > ts.min() else None,
        "period_ms": {**_summary(period), "std": float(period.std()) if len(period) else None},
        "max_abs_current_a": float(np.abs(current).max()) if len(current) else None,
        "error_codes": sorted({int(e) for e in data[:, 7]}) if len(data) else [],
        "max_temp_c": int(data[:, 6].max()) if len(data) else None,
    })
    return nxt, age, out


# ── per-command table ───────────────────────────────────────────────────────

def build_table(meta, loops, ok_sends, wire_ok):
    """One row per timed loop that sent exactly one command."""
    timed = loops[loops["timed"] == 1]
    per_loop = ok_sends.groupby("loop_seq").size()
    n_cmd = timed["seq"].map(per_loop).fillna(0).astype(int)
    one = timed[(n_cmd == 1).to_numpy()]
    cols = ["loop_seq", "t_send_start", "t_send_end"] + (
        ["t_wire", "wire_nm", "wire_kp", "wire_kd"] if wire_ok else [])
    df = one.merge(ok_sends[cols], left_on="seq", right_on="loop_seq", how="inner")

    t0 = timed["t_start"].iloc[0]
    sensor_done = df["t_start"] + (df["imu_ms"] + df["fsr_enc_ms"]) / 1e3
    out = pd.DataFrame({
        "seq": df["seq"],
        "t_s": df["t_start"] - t0,
        "sensor_read_ms": df["imu_ms"] + df["fsr_enc_ms"],
        "preprocess_ms": df["step_ms"] - df["predict_ms"],
        "inference_ms": df["predict_ms"],
        "safety_ms": (df["t_send_start"] - sensor_done) * 1e3 - df["step_ms"],
        "send_call_ms": (df["t_send_end"] - df["t_send_start"]) * 1e3,
        "e2e_send_ms": (df["t_send_end"] - sensor_done) * 1e3,
        "pre_send_work_ms": (df["t_send_start"] - df["t_start"]) * 1e3,
        "watchdog": df["watchdog"],
        "stance": df["stance"],
        "target_nm": df["target_nm"],
        "sent_nm": df["sent_nm"],
    })
    if wire_ok:
        out["transmit_ms"] = (df["t_wire"] - df["t_send_end"]) * 1e3
        out["e2e_wire_ms"] = (df["t_wire"] - sensor_done) * 1e3
        out["can_send_ms"] = out["send_call_ms"] + out["transmit_ms"]
        out["e2e_ms"] = out["e2e_wire_ms"]
        out["wire_nm"] = df["wire_nm"]
    else:
        out["can_send_ms"] = out["send_call_ms"]
        out["e2e_ms"] = out["e2e_send_ms"]
    out["_t_send_start"] = df["t_send_start"].to_numpy()
    out["_t_wire"] = df["t_wire"].to_numpy() if wire_ok else np.nan
    return out, int(len(timed)), int((n_cmd != 1).sum())


# ── safety-limit result per command ─────────────────────────────────────────

def safety_checks(meta, loops, ok_sends, wire_ok):
    """Re-derive, from the logs alone, the safety result each command needed
    before it was sent, and compare with what was actually sent and what was
    actually on the bus. Returns a per-timed-loop boolean frame and settings."""
    a, mo = meta["args"], meta.get("motor", {})
    cap = float(mo.get("torque_cap_nm", a["torque_cap"]))
    sign = float(mo.get("command_sign", a["command_sign"]))
    slew = float(a["slew"]) if mo.get("slew_active") else None
    computed = mo.get("torque_mode", a["torque_mode"]) == "computed"
    dt = 1.0 / float(a["rate"])

    # expected sent torque, loop by loop, as MotorInterface.send_torque does it.
    # motor_start() sends 0 N·m first, so the slew limit starts from 0.
    ts = loops["t_start"].to_numpy()
    target = loops["target_nm"].to_numpy(dtype=float)
    sent = loops["sent_nm"].to_numpy(dtype=float)
    tick = np.empty(len(ts))
    tick[0] = dt
    tick[1:] = np.minimum(np.diff(ts), 0.05)
    expected = np.clip(sign * (target if computed else np.zeros_like(target)), -cap, cap)
    if slew:
        prev = np.concatenate([[0.0], sent[:-1]])      # the motor's last_torque_nm
        step = slew * tick
        expected = np.clip(expected, prev - step, prev + step)

    lp = loops.assign(expected_nm=expected)
    timed = lp[lp["timed"] == 1]
    per_loop = ok_sends.groupby("loop_seq").size()
    first = ok_sends.drop_duplicates("loop_seq").set_index("loop_seq")
    t_send = timed["seq"].map(first["t_send_start"])
    pre_send_ms = (t_send - timed["t_start"]).to_numpy() * 1e3

    wd = timed["watchdog"].to_numpy() == 1
    assist = timed["assist_cmd_nm"].to_numpy(dtype=float)
    ramp = timed["session_ramp"].to_numpy(dtype=float)
    tgt = timed["target_nm"].to_numpy(dtype=float)
    snt = timed["sent_nm"].to_numpy(dtype=float)
    wdl = float(a["watchdog_ms"])

    c = pd.DataFrame({"seq": timed["seq"].to_numpy()})
    c["one_frame"] = ((timed["seq"].map(per_loop).fillna(0).to_numpy() == 1)
                      & (timed["can_cmd_ok"].to_numpy() == 1)
                      & (timed["can_err"].to_numpy() == 0))
    c["finite"] = np.isfinite(assist) & np.isfinite(tgt) & np.isfinite(snt)
    with np.errstate(invalid="ignore"):
        c["watchdog_decided"] = np.where(wd, pre_send_ms > wdl,
                                         pre_send_ms <= wdl + WATCHDOG_TOL_MS)
    c["watchdog_zeroes"] = ~wd | (tgt == 0.0)
    c["session_ramp"] = wd | (np.abs(tgt - assist * ramp)
                              <= TORQUE_TOL_NM * np.maximum(1.0, np.abs(tgt)))
    c["within_cap"] = np.abs(snt) <= cap + TORQUE_TOL_NM
    c["pipeline"] = np.abs(snt - timed["expected_nm"].to_numpy()) <= TORQUE_TOL_NM
    if wire_ok:
        w = first.loc[timed["seq"].to_numpy()]
        wnm = w["wire_nm"].to_numpy(dtype=float)
        c["wire_within_cap"] = np.abs(wnm) <= cap + MIT_T_LSB
        c["wire_matches"] = np.abs(wnm - snt) <= MIT_T_LSB
        c["wire_no_gains"] = (w["wire_kp"].to_numpy() == 0) & (w["wire_kd"].to_numpy() == 0)
    checks = [k for k, _ in SAFETY_CHECKS if k in c]
    c["valid"] = c[checks].all(axis=1)
    settings = {"torque_cap_nm": cap, "command_sign": sign, "slew_nm_per_s": slew,
                "watchdog_ms": wdl, "torque_mode": "computed" if computed else "zero",
                "arm_ramp_s": a.get("arm_ramp"), "checks_run": checks}
    return c, settings


# ── metrics + verdict ───────────────────────────────────────────────────────

def analyze(src):
    meta, loops, sends, mon = load_run(src)
    ok_sends, wire = match_wire(meta, sends, mon)
    wire_ok = wire["matched"]
    tab, n_timed, n_bad_loops = build_table(meta, loops, ok_sends, wire_ok)
    checks, safety_cfg = safety_checks(meta, loops, ok_sends, wire_ok)

    t_wire = tab["_t_wire"].to_numpy() if wire_ok else None
    nxt, age, motor = motor_status(mon, meta, t_wire, tab["_t_send_start"].to_numpy())
    if nxt is not None:
        tab["next_status_ms"] = nxt
    if age is not None:
        tab["feedback_age_ms"] = age

    stages = {k: {"label": lab, **_summary(tab[k])} for k, lab in STAGES}
    sub = {"send_call_ms": {"label": "  bus.send() call", **_summary(tab["send_call_ms"])}}
    if wire_ok:
        sub["transmit_ms"] = {"label": "  send() return -> on the bus", **_summary(tab["transmit_ms"])}
    e2e = _summary(tab["e2e_ms"])
    slowest_p50 = max(stages, key=lambda k: stages[k]["p50"])
    slowest_p99 = max(stages, key=lambda k: stages[k]["p99"])
    tail = tab[tab["e2e_ms"] >= np.percentile(tab["e2e_ms"], 99)]
    tail_share = {k: float(tail[k].sum() / tail["e2e_ms"].sum()) for k, _ in STAGES}

    n_cmd = int(len(checks))
    n_valid = int(checks["valid"].sum())
    fails = {k: int((~checks[k]).sum()) for k, _ in SAFETY_CHECKS if k in checks}

    endpoint = "on the CAN bus" if wire_ok else "send() return"
    crit = [
        ("Median sensor-read-to-CAN-send delay", e2e["p50"], MEDIAN_MAX_MS),
        ("99th-percentile delay", e2e["p99"], P99_MAX_MS),
        ("Maximum delay", e2e["max"], MAX_MAX_MS),
    ]
    criteria = [{"criterion": name, "value": f"{v:.2f} ms", "limit": f"<= {lim:g} ms",
                 "measured_ms": v, "limit_ms": lim, "result": "PASS" if v <= lim else "FAIL"}
                for name, v, lim in crit]
    safe_ok = n_valid == n_cmd == n_timed and n_cmd > 0
    criteria.append({"criterion": "Every torque command has a valid safety-limit result "
                                  "before transmission",
                     "value": f"{n_valid} / {n_timed} commands",
                     "limit": "all", "result": "PASS" if safe_ok else "FAIL"})
    overall = all(c["result"] == "PASS" for c in criteria)

    return {
        "source": str(src), "meta": meta, "wire": wire, "endpoint": endpoint,
        "timed_loops": n_timed, "loops_without_one_command": n_bad_loops,
        "commands_scored": int(len(tab)),
        "window_s": float(tab["t_s"].iloc[-1]) if len(tab) else 0.0,
        "e2e_ms": e2e, "e2e_send_return_ms": _summary(tab["e2e_send_ms"]),
        "sensor_read_ms": _summary(tab["sensor_read_ms"]),
        "stages": stages, "send_detail": sub,
        "slowest_stage": {"by_median": stages[slowest_p50]["label"],
                          "by_p99": stages[slowest_p99]["label"],
                          "share_of_slowest_1pct": {stages[k]["label"]: v
                                                    for k, v in tail_share.items()}},
        "safety": {"settings": safety_cfg, "commands": n_cmd, "valid": n_valid,
                   "failures_by_check": fails,
                   "watchdog_trips": int(tab["watchdog"].sum())},
        "motor_reply": {**motor,
                        "next_status_ms": _summary(nxt) if nxt is not None else None,
                        "feedback_age_ms": _summary(age) if age is not None else None},
        "criteria": criteria, "pass": overall,
        "_tab": tab, "_checks": checks,
    }


# ── plots (presentation format, 16:9) ───────────────────────────────────────

def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=12)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _header(fig, title, subtitle):
    fig.text(0.04, 0.95, title, fontsize=20, color=INK, weight="bold", va="top")
    fig.text(0.04, 0.885, subtitle, fontsize=12.5, color=INK_2, va="top")


def _footer(fig, r):
    m = r["meta"]
    bits = [f"Source: Test 4 run {Path(r['source']).parent.name}/{Path(r['source']).name}",
            f"{r['commands_scored']} commands over {r['window_s']:.0f} s",
            f"end point: torque command {r['endpoint']}"]
    if m.get("sim"):
        bits.append("SIMULATED HARDWARE — not a test result")
    elif m.get("sim_sensors"):
        bits.append("simulated sensor reads (path after the read is real)")
    fig.text(0.04, 0.025, "  ·  ".join(bits), fontsize=10, color=MUTED, va="bottom")


def _limit_lines(ax, horizontal=False, xmax=None):
    for v, lab in ((MEDIAN_MAX_MS, "median limit 10 ms"), (P99_MAX_MS, "p99 limit 20 ms"),
                   (MAX_MAX_MS, "max limit 50 ms")):
        if horizontal:
            ax.axhline(v, color=MUTED, linewidth=1.2, linestyle="--", zorder=1)
            ax.annotate(lab, xy=(1.0, v), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                        textcoords="offset points", ha="right", fontsize=11, color=INK_2)
        else:
            ax.axvline(v, color=MUTED, linewidth=1.2, linestyle="--", zorder=1)
            ax.annotate(lab, xy=(v, 0.5), xycoords=("data", "axes fraction"), xytext=(5, 0),
                        textcoords="offset points", rotation=90, va="center", fontsize=11,
                        color=INK_2)


def make_plots(r, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    tab = r["_tab"]
    e2e = tab["e2e_ms"].to_numpy()
    files = []
    size = (13.33, 7.5)

    def save(fig, name):
        files.append(out_dir / name)
        fig.savefig(files[-1], dpi=150, facecolor=SURFACE)
        plt.close(fig)

    # 1. scorecard: each criterion, measured vs limit, on one shared ms axis
    fig = plt.figure(figsize=size, facecolor=SURFACE)
    _header(fig, "Test 5 — Sensor read to torque command: pass criteria",
            f"Overall: {'PASS' if r['pass'] else 'FAIL'}   ·   delay from the end of the sensor "
            f"read to the torque command {r['endpoint']}")
    ax = fig.add_axes([0.25, 0.36, 0.52, 0.44])
    _style(ax)
    lat = [c for c in r["criteria"] if "limit_ms" in c]
    y = np.arange(len(lat))
    xmax = max(MAX_MAX_MS * 1.1, max(c["measured_ms"] for c in lat) * 1.15)
    ax.barh(y, [c["measured_ms"] for c in lat], height=0.5, color=SERIES[0], zorder=2)
    for yi, c in zip(y, lat):
        ax.plot([c["limit_ms"]] * 2, [yi - 0.36, yi + 0.36], color=INK, linewidth=2.5, zorder=3)
        ax.annotate(f"limit {c['limit_ms']:g}", xy=(c["limit_ms"], yi - 0.36), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=11,
                    color=INK_2)
    ax.set_yticks(y, ["Median", "99th percentile", "Maximum"], fontsize=14, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel("ms, sensor read done → torque command " + r["endpoint"],
                  fontsize=12, color=INK_2)
    ax.grid(axis="y", visible=False)
    for yi, c in zip(y, lat):          # value column, then status: colored dot + word
        fig.text(0.865, ax.transData.transform((0, yi))[1] / fig.dpi / size[1],
                 f"{c['measured_ms']:.2f} ms", color=INK, fontsize=14, weight="bold",
                 va="center", ha="right")
        fig.text(0.885, ax.transData.transform((0, yi))[1] / fig.dpi / size[1],
                 "●", color=STATUS[c["result"]], fontsize=16, va="center")
        fig.text(0.905, ax.transData.transform((0, yi))[1] / fig.dpi / size[1],
                 c["result"], color=INK, fontsize=14, weight="bold", va="center")
    sc = r["criteria"][-1]
    s = r["safety"]
    fig.text(0.04, 0.24, "Safety-limit result before every transmission", fontsize=14,
             color=INK, weight="bold")
    fig.text(0.04, 0.19, f"{sc['value']} valid  ·  {s['watchdog_trips']} watchdog trips, each "
             "forced 0 N·m", fontsize=12.5, color=INK_2)
    fig.text(0.04, 0.145, f"{len(s['settings']['checks_run'])} checks per command: watchdog, "
             f"session ramp, ±{s['settings']['torque_cap_nm']:g} N·m cap, slew limit, and the "
             "frame decoded from the bus", fontsize=12.5, color=INK_2)
    fig.text(0.885, 0.25, "●", color=STATUS[sc["result"]], fontsize=16, va="center")
    fig.text(0.905, 0.25, sc["result"], color=INK, fontsize=14, weight="bold", va="center")
    _footer(fig, r)
    save(fig, "criteria_scorecard.png")

    # 2. distribution of the end-to-end delay
    fig = plt.figure(figsize=size, facecolor=SURFACE)
    _header(fig, "End-to-end delay: sensor read → torque command",
            f"Every command in the timed window (n = {len(e2e)}); log scale shows the rare slow "
            "commands")
    ax = fig.add_axes([0.08, 0.15, 0.88, 0.66])
    _style(ax)
    hi = max(MAX_MAX_MS * 1.1, e2e.max() * 1.05)
    ax.hist(e2e, bins=np.arange(0, hi + 0.25, 0.25), color=SERIES[0], zorder=2)
    ax.set_yscale("log")
    ax.set_xlim(0, hi)
    _limit_lines(ax)
    e = r["e2e_ms"]
    ytop = ax.get_ylim()[1]
    for v, lab, row in ((e["p50"], f"median {e['p50']:.2f} ms", 0),
                        (e["p99"], f"p99 {e['p99']:.2f} ms", 1),
                        (e["max"], f"max {e['max']:.2f} ms", 0)):
        ax.plot([v, v], [1, ytop], color=INK, linewidth=1.2, zorder=3)
        ax.annotate(lab, xy=(v, ytop), xytext=(5, -4 - 22 * row),
                    textcoords="offset points", va="top", ha="left", fontsize=12, color=INK,
                    weight="bold", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none"))
    ax.set_xlabel("delay (ms)", fontsize=13, color=INK_2)
    ax.set_ylabel("commands", fontsize=13, color=INK_2)
    _footer(fig, r)
    save(fig, "e2e_latency_histogram.png")

    # 3. where the time goes: median and p99 per stage
    fig = plt.figure(figsize=size, facecolor=SURFACE)
    _header(fig, "Where the delay comes from",
            f"Median and 99th percentile of each stage. Slowest stage: "
            f"{r['slowest_stage']['by_median']}")
    ax = fig.add_axes([0.33, 0.15, 0.62, 0.66])
    _style(ax)
    rows = [(v["label"], v["p50"], v["p99"]) for v in r["stages"].values()]
    rows.append(("Whole path (sensor read → " + ("bus" if "bus" in r["endpoint"] else "send")
                 + ")", e["p50"], e["p99"]))
    y = np.arange(len(rows))
    h = 0.36
    ax.barh(y - h / 2 - 0.01, [x[1] for x in rows], height=h, color=SERIES[0], label="median",
            zorder=2)
    ax.barh(y + h / 2 + 0.01, [x[2] for x in rows], height=h, color=SERIES[1],
            label="99th percentile", zorder=2)
    for yi, (_, p50, p99) in zip(y, rows):
        ax.annotate(f"{p50:.2f}", xy=(p50, yi - h / 2), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=11.5, color=INK)
        ax.annotate(f"{p99:.2f}", xy=(p99, yi + h / 2), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=11.5, color=INK)
    ax.set_yticks(y, [x[0] for x in rows], fontsize=13, color=INK)
    ax.get_yticklabels()[-1].set_weight("bold")
    ax.invert_yaxis()
    ax.axhline(len(rows) - 1.5, color=MUTED, linewidth=0.8)
    ax.set_xlim(0, max(x[2] for x in rows) * 1.15)
    ax.set_xlabel("ms", fontsize=13, color=INK_2)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=12, labelcolor=INK, loc="upper right")
    _footer(fig, r)
    save(fig, "stage_breakdown.png")

    # 4. anatomy of a typical and of the slowest command (real loops)
    typ = tab.iloc[int(np.argmin(np.abs(e2e - np.median(e2e))))]
    worst = tab.iloc[int(np.argmax(e2e))]
    fig = plt.figure(figsize=size, facecolor=SURFACE)
    _header(fig, "Inside one control loop: from loop start to torque command",
            "Two real commands from the run: the one at the median delay and the slowest one. "
            f"Safety checks ({typ['safety_ms']:.2f} ms) and CAN send ({typ['can_send_ms']:.2f} ms) "
            "are thin slivers at this scale")
    ax = fig.add_axes([0.20, 0.27, 0.76, 0.52])
    _style(ax)
    seg_names = ["Sensor read" + (" (simulated)" if r["meta"].get("sim_sensors") else "")] + \
        [lab for _, lab in STAGES]
    seg_colors = [SENSOR_GRAY] + SERIES
    span = max(worst["sensor_read_ms"] + worst["e2e_ms"], 1.0)
    for yi, row, name in ((0, typ, "Median command"), (1, worst, "Slowest command")):
        parts = [row["sensor_read_ms"]] + [row[k] for k, _ in STAGES]
        x0 = 0.0
        for w, col in zip(parts, seg_colors):
            ax.add_patch(Rectangle((x0, yi - 0.28), w, 0.56, facecolor=col, edgecolor=SURFACE,
                                   linewidth=2, zorder=2))
            if w >= span * 0.07:
                ax.text(x0 + w / 2, yi, f"{w:.1f}", ha="center", va="center", fontsize=12,
                        color=INK, zorder=3)
            x0 += w
        s0 = row["sensor_read_ms"]
        ax.annotate("", xy=(x0, yi - 0.40), xytext=(s0, yi - 0.40),
                    arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.2))
        ax.text((s0 + x0) / 2, yi - 0.44, f"sensor read → command {r['endpoint']}: "
                f"{row['e2e_ms']:.2f} ms", ha="center", va="bottom", fontsize=12, color=INK,
                weight="bold")
        ax.text(-span * 0.01, yi, f"{name}\nt = {row['t_s']:.1f} s", ha="right", va="center",
                fontsize=13, color=INK)
    ax.set_ylim(1.6, -0.75)
    ax.set_xlim(0, span * 1.03)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("ms since the loop started", fontsize=13, color=INK_2)
    handles = [Rectangle((0, 0), 1, 1, facecolor=c) for c in seg_colors]
    fig.legend(handles, seg_names, loc="lower left", bbox_to_anchor=(0.04, 0.075), ncol=3,
               frameon=False, fontsize=12, labelcolor=INK)
    _footer(fig, r)
    save(fig, "loop_anatomy.png")

    # 5. delay of every command over the run
    fig = plt.figure(figsize=size, facecolor=SURFACE)
    _header(fig, "End-to-end delay over the run",
            "Each dot is one torque command; the line is the median of the last 100 commands "
            "(~1 s)")
    ax = fig.add_axes([0.07, 0.15, 0.89, 0.66])
    _style(ax)
    t = tab["t_s"].to_numpy()
    ax.plot(t, e2e, ".", markersize=3, color=SERIES[0], alpha=0.55, zorder=2,
            label="torque command")
    roll = pd.Series(e2e).rolling(100, min_periods=10).median()
    ax.plot(t, roll, color=SERIES_DARK, linewidth=2, zorder=3, label="rolling median")
    wd = tab["watchdog"].to_numpy() == 1
    if wd.any():
        ax.plot(t[wd], e2e[wd], "D", markersize=9, color=SERIES[1], markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=4, label=f"watchdog trip → 0 N·m ({int(wd.sum())})")
    _limit_lines(ax, horizontal=True)
    ax.set_ylim(0, max(MAX_MAX_MS * 1.1, e2e.max() * 1.05))
    ax.set_xlim(0, t.max())
    ax.set_xlabel("time in the timed window (s)", fontsize=13, color=INK_2)
    ax.set_ylabel("delay (ms)", fontsize=13, color=INK_2)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=".", linestyle="", markersize=12, color=SERIES[0],
                      label="torque command"),
               Line2D([], [], color=SERIES_DARK, linewidth=2, label="rolling median")]
    if wd.any():
        handles.append(Line2D([], [], marker="D", linestyle="", markersize=9, color=SERIES[1],
                              label=f"watchdog trip → 0 N·m ({int(wd.sum())})"))
    leg = ax.legend(handles=handles, frameon=True, fontsize=12, labelcolor=INK,
                    loc="upper left", bbox_to_anchor=(0.0, 0.84), ncol=len(handles))
    leg.get_frame().set(facecolor=SURFACE, edgecolor="none")
    _footer(fig, r)
    save(fig, "e2e_latency_timeline.png")

    # 6. CAN reply age (reported separately, not scored)
    mr = r["motor_reply"]
    panels = [(k, lab) for k, lab in
              (("next_status_ms", "Command on the bus → motor's next status frame"),
               ("feedback_age_ms", "Age of the newest motor feedback when the command is sent"))
              if k in tab]
    if panels:
        per = mr.get("period_ms", {}).get("p50")
        fig = plt.figure(figsize=size, facecolor=SURFACE)
        _header(fig, "CAN reply age (reported separately)",
                "The motor broadcasts its status " + (f"every {per:.1f} ms" if per else
                "at a fixed rate") + " rather than answering each command, so the age is set "
                "by the motor, not the controller")
        for i, (k, lab) in enumerate(panels):
            ax = fig.add_axes([0.07 + i * 0.47, 0.15, 0.42, 0.62])
            _style(ax)
            v = tab[k].dropna().to_numpy()
            top = max(25.0, np.percentile(v, 99.9) * 1.1)
            ax.hist(v, bins=np.arange(0, top + 0.5, 0.5), color=SERIES[0], zorder=2)
            if per:
                ax.axvline(per, color=MUTED, linestyle="--", linewidth=1.2)
                ax.annotate(f"status period {per:.1f} ms", xy=(per, 0.97),
                            xycoords=("data", "axes fraction"), xytext=(-5, 0),
                            textcoords="offset points", ha="right", va="top", fontsize=11,
                            color=INK_2)
            sm = mr[k]
            ax.set_title(lab, loc="left", fontsize=13, color=INK)
            ax.text(0.02, 0.97, f"median {sm['p50']:.1f} ms\np99 {sm['p99']:.1f} ms\n"
                    f"max {sm['max']:.1f} ms", transform=ax.transAxes, va="top", fontsize=12,
                    color=INK)
            ax.set_xlim(0, top)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
            ax.set_xlabel("ms", fontsize=13, color=INK_2)
            ax.set_ylabel("commands" if i == 0 else "", fontsize=13, color=INK_2)
        _footer(fig, r)
        save(fig, "can_reply_age.png")
    return files


# ── report ──────────────────────────────────────────────────────────────────

def _f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_report(r, out_dir, plots):
    m, a = r["meta"], r["meta"]["args"]
    L = ["# Test 5 — End-to-end latency: sensor read to torque command (R-05)", "",
         f"**Verdict: {'PASS' if r['pass'] else 'FAIL'}**", "",
         f"- Source: Test 4 load run `{r['source']}` (status `{m.get('status')}`, "
         f"{a['duration']:g} s timed window; the test plan asks for 600 s)"]
    if m.get("sim"):
        L.append("- **SIMULATED HARDWARE**: virtual CAN bus and fake motor. Not a test result.")
    elif m.get("sim_sensors"):
        L.append(f"- **SIMULATED SENSORS**: the sensor reads were placeholder waits "
                 f"({m.get('sim_sensor_read_ms')} ms). The delay measured here starts when the "
                 "read ends, so it covers only real work (controller, TCN, safety checks, CAN). "
                 "Repeat on a Test 4 run with the sensors connected for the final result.")
    if not a.get("no_gait_inject"):
        L.append("- Gait injection was on: the controller ran on a recorded walk while the "
                 "sensors were read every loop. The timing path is the same.")
    L += [f"- Controller: TCN `{m.get('tcn', {}).get('run_dir', '?')}`, backend "
          f"`{m.get('tcn', {}).get('backend', '?')}`; torque mode "
          f"`{r['safety']['settings']['torque_mode']}`; {a['rate']:g} Hz, scheduler "
          f"`{a['scheduler']}`, logger `{a['logger']}`",
          f"- Jetson: {m['system'].get('model')}; "
          f"{(m['system'].get('nvpmodel') or '').splitlines()[0] if m['system'].get('nvpmodel') else ''}; "
          f"governor {m['system'].get('governors')}",
          f"- Delay measured from the end of the loop's sensor read to the torque command "
          f"**{r['endpoint']}** ({r['wire']['status']})", "",
          "| Criterion | Measured | Limit | Result |", "|---|---|---|---|"]
    L += [f"| {c['criterion']} | {c['value']} | {c['limit']} | **{c['result']}** |"
          for c in r["criteria"]]

    L += ["", "## Script output", "",
          f"{r['commands_scored']} commands in a {r['window_s']:.1f} s window "
          f"({r['timed_loops']} timed loops, {r['loops_without_one_command']} without exactly "
          "one command).", "",
          "| Stage | median | p99 | max | mean |", "|---|---|---|---|---|"]
    for v in list(r["stages"].values()) + list(r["send_detail"].values()):
        L.append(f"| {v['label']} | {_f(v['p50'], 3)} | {_f(v['p99'], 3)} | "
                 f"{_f(v['max'], 3)} | {_f(v['mean'], 3)} |")
    e, es = r["e2e_ms"], r["e2e_send_return_ms"]
    L += [f"| **Sensor read → command {r['endpoint']}** | **{_f(e['p50'], 3)}** | "
          f"**{_f(e['p99'], 3)}** | **{_f(e['max'], 3)}** | **{_f(e['mean'], 3)}** |"]
    if r["wire"]["matched"]:
        L.append(f"| Sensor read → send() return | {_f(es['p50'], 3)} | {_f(es['p99'], 3)} | "
                 f"{_f(es['max'], 3)} | {_f(es['mean'], 3)} |")
    sr = r["sensor_read_ms"]
    L += [f"| (context) sensor read itself | {_f(sr['p50'], 3)} | {_f(sr['p99'], 3)} | "
          f"{_f(sr['max'], 3)} | {_f(sr['mean'], 3)} |", "",
          "All times in ms. The stages add up exactly to the whole path.", "",
          f"**Slowest stage:** {r['slowest_stage']['by_median']} by median, "
          f"{r['slowest_stage']['by_p99']} by p99. Share of the time in the slowest 1 % of "
          "commands: " + ", ".join(f"{k} {100 * v:.0f} %" for k, v in
                                   r["slowest_stage"]["share_of_slowest_1pct"].items()) + "."]

    tab = r["_tab"]
    L += ["", "Five slowest commands:", "",
          "| t (s) | total | preprocessing | TCN | safety + packing | CAN send | watchdog |",
          "|---|---|---|---|---|---|---|"]
    for _, w in tab.nlargest(5, "e2e_ms").iterrows():
        L.append(f"| {w['t_s']:.1f} | {w['e2e_ms']:.2f} | {w['preprocess_ms']:.2f} | "
                 f"{w['inference_ms']:.2f} | {w['safety_ms']:.2f} | {w['can_send_ms']:.2f} | "
                 f"{'trip' if w['watchdog'] else ''} |")

    s = r["safety"]
    st = s["settings"]
    L += ["", "## Safety-limit result per command", "",
          f"{s['valid']} of {s['commands']} commands passed every check "
          f"(cap ±{st['torque_cap_nm']:g} N·m, sign {st['command_sign']:+g}, slew "
          f"{st['slew_nm_per_s']} N·m/s, watchdog {st['watchdog_ms']:g} ms, session ramp "
          f"{st['arm_ramp_s']} s). {s['watchdog_trips']} watchdog trips, each forcing 0 N·m.", "",
          "| Check | Failures |", "|---|---|"]
    L += [f"| {lab} | {s['failures_by_check'][k]} |" for k, lab in SAFETY_CHECKS
          if k in s["failures_by_check"]]
    L += ["", "The checks recompute each command's safety result from the logged inputs and "
          "compare it with what `send_torque()` returned and with the frame decoded from the "
          "bus. The watchdog uses the work time measured before the send, so its decision is "
          "made before transmission."]

    mr = r["motor_reply"]
    L += ["", "## CAN reply age (reported separately, not scored)", ""]
    if mr.get("status") != "ok":
        L.append(f"Not available: {mr.get('status')}.")
    else:
        pm = mr["period_ms"]
        L += [f"The motor sends status frames ({', '.join(mr['ids'])}) at "
              f"{_f(mr['rate_hz'])} Hz, every {_f(pm.get('p50'))} ms (std {_f(pm.get('std'), 3)} "
              "ms), whatever the command rate. It does not answer each command, so a command "
              "is matched to the next status frame after it.", "",
              "| Reply age | median | p99 | max |", "|---|---|---|---|"]
        for k, lab in (("next_status_ms", "Command on the bus → next motor status frame"),
                       ("feedback_age_ms", "Age of newest motor feedback when send() starts")):
            v = mr.get(k)
            if v and v.get("n"):
                L.append(f"| {lab} | {_f(v['p50'])} | {_f(v['p99'])} | {_f(v['max'])} |")
        L += ["", f"Motor status during the window: max |current| "
              f"{_f(mr['max_abs_current_a'])} A, max temperature {mr['max_temp_c']} °C, error "
              f"codes {mr['error_codes']}. The controller does not use motor feedback yet, so "
              "the reply age does not add to the command delay."]

    if r["wire"].get("warning"):
        L += ["", f"**Warning:** {r['wire']['warning']}"]
    L += ["", "## Plots", ""] + [f"![{p.stem}]({p.name})" for p in plots]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default=None,
                    help="Test 4 session or load-run folder (default: the newest one)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: fall_test_plan/test_5/results/<date_time>)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    src = find_source(args.source)
    out_dir = Path(args.out) if args.out else (
        TEST_DIR / "results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Test 5: analysing Test 4 run {src}")

    r = analyze(src)
    plots = [] if args.no_plots else make_plots(r, out_dir)
    write_report(r, out_dir, plots)
    keep = [c for c in r["_tab"].columns if not c.startswith("_")]
    r["_tab"][keep].merge(r["_checks"], on="seq", how="left").to_csv(
        out_dir / "latency_per_command.csv", index=False, float_format="%.4f")
    with open(out_dir / "report.json", "w") as f:
        json.dump({"overall_pass": r["pass"], "generated": datetime.now().isoformat(
                   timespec="seconds"),
                   **{k: v for k, v in r.items() if not k.startswith("_") and k != "meta"},
                   "source_meta": {k: r["meta"].get(k) for k in
                                   ("status", "sim", "sim_sensors", "sim_sensor_read_ms",
                                    "started", "tcn", "motor")}},
                  f, indent=2, default=_jsonable)

    e = r["e2e_ms"]
    print(f"  {r['commands_scored']} commands, sensor read -> command {r['endpoint']}: "
          f"median {e['p50']:.2f}  p99 {e['p99']:.2f}  max {e['max']:.2f} ms")
    for v in r["stages"].values():
        print(f"    {v['label']:<44} p50 {v['p50']:6.3f}  p99 {v['p99']:6.3f}  "
              f"max {v['max']:6.3f} ms")
    print(f"  slowest stage: {r['slowest_stage']['by_median']} (median), "
          f"{r['slowest_stage']['by_p99']} (p99)")
    mr = r["motor_reply"]
    if mr.get("status") == "ok" and mr.get("next_status_ms"):
        print(f"  CAN reply age (not scored): next status p50 {mr['next_status_ms']['p50']:.1f} "
              f"ms, feedback age p50 {mr['feedback_age_ms']['p50']:.1f} ms "
              f"(motor broadcasts every {mr['period_ms']['p50']:.1f} ms)")
    print()
    for c in r["criteria"]:
        print(f"  [{c['result']:>4}] {c['criterion']}: {c['value']} (limit {c['limit']})")
    fails = {k: v for k, v in r["safety"]["failures_by_check"].items() if v}
    if fails:
        print(f"  -> safety checks with failures: {fails}")
    m = r["meta"]
    print(f"\nTest 5 {'PASS' if r['pass'] else 'FAIL'}"
          + ("  (SIMULATED HARDWARE — not a test result)" if m.get("sim") else
             "  (SIMULATED SENSORS in the source run)" if m.get("sim_sensors") else ""))
    print(f"report: {out_dir / 'report.md'}")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
