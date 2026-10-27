#!/usr/bin/env python3
"""
test4_analyze.py — metrics, pass/fail and plots for Test 4.

    python3 fall_test_plan/test_4/test4_analyze.py <session folder>
    python3 fall_test_plan/test_4/test4_analyze.py <run folder> [<run folder> ...] --out <dir>

A session folder (from `test4_run.py --phase all`) holds idle/, load/ and
optionally flush/. Each run folder has meta.json, loops.csv, can_sends.csv and
can_monitor.npz.

Script output (per run, timed window only — pre-roll excluded):
    mean loop rate, median and 99th-percentile period, largest period,
    % of loops longer than 10 ms and 20 ms, CAN command rate
plus where the loop time goes (per stage), CAN frame accounting, bus errors,
memory, and the pass/fail verdict of the load run.

Writes report.md, report.json and *.png into the output folder.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── pass criteria (Test 4, R-05) ─────────────────────────────────────────────
TARGET_RATE_HZ = 100.0
P99_PERIOD_MAX_MS = 20.0
MAX_PERIOD_MAX_MS = 50.0
CAN_RATE_MIN_HZ = 100.0
LONG_LOOP_MS = (10.0, 20.0)
# Rates are compared at the precision they can be measured to. Over a 600 s run
# the window endpoints jitter by ~0.1 ms, so 99.9999 Hz and 100.0000 Hz are the
# same measurement; anything below 99.995 Hz is a real shortfall.
RATE_RESOLUTION_HZ = 0.01
CLOCK_DRIFT_WARN_MS = 1.0

CMD_ARB_ID = (8 << 8) | 0x68

STAGES = [("imu_ms", "IMU read (2x BNO085)"), ("fsr_enc_ms", "FSR + encoder read"),
          ("predict_ms", "TCN inference"), ("step_other_ms", "adapter + features + assist"),
          ("send_ms", "CAN send"), ("log_ms", "logging"), ("telemetry_ms", "Teleplot"),
          ("flush_ms", "disk flush")]

# chart colors: categorical slots 1-3 of the reference palette (validated)
RUN_COLORS = {"idle": "#2a78d6", "load": "#eb6834", "flush": "#1baf7a"}
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def _summary(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "p50": _pct(x, 50),
            "p99": _pct(x, 99), "max": float(x.max())}


# ── loading ─────────────────────────────────────────────────────────────────

def load_run(path):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    run = {"path": path, "meta": meta, "name": path.name,
           "loops": pd.read_csv(path / "loops.csv")}
    sends = path / "can_sends.csv"
    run["sends"] = pd.read_csv(sends) if sends.exists() else None
    mon = path / "can_monitor.npz"
    run["monitor"] = dict(np.load(mon)) if mon.exists() else None
    return run


def find_runs(paths):
    runs = []
    for p in map(Path, paths):
        if (p / "meta.json").exists():
            runs.append(p)
        else:
            runs += [d for d in (p / "idle", p / "load", p / "flush") if (d / "meta.json").exists()]
    return [load_run(r) for r in runs]


def run_kind(run):
    m = run["meta"]
    if m["phase"] == "idle":
        return "idle"
    return "flush" if m["args"].get("flush_every", 0) > 0 else "load"


# ── metrics ─────────────────────────────────────────────────────────────────

def loop_metrics(run):
    lp = run["loops"]
    t = lp[lp["timed"] == 1]
    ts = t["t_start"].to_numpy()
    end = run["meta"].get("final_loop_start")
    if end is not None and len(ts) and end > ts[-1]:
        ts = np.append(ts, end)            # closes the period of the last timed loop
    out = {"timed_loops": int(len(t))}
    if len(t) < 3:
        out["error"] = "fewer than 3 timed loops"
        return out, np.array([]), np.array([])
    periods = np.diff(ts) * 1e3
    span = ts[-1] - ts[0]
    busy = (t["t_end"] - t["t_start"]).to_numpy() * 1e3
    # time a period spends beyond the loop work and the requested sleep
    n = len(periods)
    overshoot = periods - busy[:n] - t["sleep_req_ms"].to_numpy()[:n]
    out.update({
        "window_s": float(span),
        "mean_rate_hz": float((len(ts) - 1) / span),
        "period_ms": {**_summary(periods), "median": _pct(periods, 50),
                      "std": float(periods.std())},
        "pct_over_10ms": float(100.0 * np.mean(periods > LONG_LOOP_MS[0])),
        "pct_over_20ms": float(100.0 * np.mean(periods > LONG_LOOP_MS[1])),
        "count_over_50ms": int(np.sum(periods > MAX_PERIOD_MAX_MS)),
        # same thresholds on the loop's own work time (start -> end, no sleep)
        "work_pct_over_10ms": float(100.0 * np.mean(busy > LONG_LOOP_MS[0])),
        "work_pct_over_20ms": float(100.0 * np.mean(busy > LONG_LOOP_MS[1])),
        "busy_ms": _summary(busy),
        "sched_overshoot_ms": _summary(overshoot),
        "watchdog_trips": int(t["watchdog"].sum()),
    })
    stages = {}
    for col, label in STAGES:
        if col == "step_other_ms":
            vals = (t["step_ms"] - t["predict_ms"]).to_numpy()
        elif col == "flush_ms":
            vals = t[col].to_numpy()
            vals = vals[vals > 0]                 # only the loops that flushed
        else:
            vals = t[col].to_numpy()
        if np.any(vals):
            stages[col] = {"label": label, **_summary(vals)}
    out["stages"] = stages

    # the five longest periods and what the loop that started each one was doing
    idx = np.argsort(periods)[::-1][:5]          # period k belongs to timed loop k
    worst = []
    for k in idx:
        row = t.iloc[k]
        worst.append({"t_s": float(ts[k] - ts[0]), "period_ms": float(periods[k]),
                      "busy_ms": float(busy[k]),
                      **{c: float(row[c]) for c, _ in STAGES if c in row and row[c] > 0},
                      "step_ms": float(row["step_ms"])})
    out["worst_periods"] = worst
    return out, ts, periods


def integrity(run):
    m, lp = run["meta"], run["loops"]
    seq = lp["seq"].to_numpy()
    checks = {
        "status": m.get("status"),
        "completed": m.get("status") == "completed",
        "rows_match": int(len(lp)) == int(m.get("loops_recorded", -1)),
        "seq_contiguous": bool(len(seq) and seq[0] == 0 and np.all(np.diff(seq) == 1)),
        "can_send_errors": int(m.get("can_send_error_count", 0)),
        "recorder_grow_events": int(m.get("recorder_grow_events", 0)),
        "exception": m.get("exception"),
    }
    if run["sends"] is not None:
        checks["send_rows_match"] = int(len(run["sends"])) == int(m.get("can_sends_recorded", -1))
    if "monitor" in m:
        checks["monitor_ok"] = m["monitor"].get("returncode") == 0
    return checks


def can_metrics(run, ts_loops):
    """In-process send times, on-wire frames from the monitor, link counters."""
    m, lp, sends, mon = run["meta"], run["loops"], run["sends"], run["monitor"]
    out = {}
    if sends is None or not len(ts_loops):
        return out
    timed_seq = set(lp.loc[lp["timed"] == 1, "seq"].tolist())
    is_cmd = sends["arb_id"].apply(lambda s: int(s, 16)) == CMD_ARB_ID
    ok_cmd = sends[is_cmd & (sends["ok"] == 1)].reset_index(drop=True)
    in_window = ok_cmd["loop_seq"].isin(timed_seq).to_numpy()
    t_send = ok_cmd.loc[in_window, "t_send_start"].to_numpy()
    timed = lp[lp["timed"] == 1]
    out["inproc"] = {
        "cmd_frames_total": int(len(ok_cmd)),
        "cmd_frames_timed": int(len(t_send)),
        "loops_without_exactly_one_cmd": int(np.sum(timed["can_cmd_ok"] != 1)),
        "send_errors": int((sends["ok"] == 0).sum()),
    }
    if len(t_send) > 2:
        iv = np.diff(t_send) * 1e3
        out["inproc"].update({
            "rate_hz": float((len(t_send) - 1) / (t_send[-1] - t_send[0])),
            "interval_ms": _summary(iv)})

    if mon is not None and "recv_error_wall" in mon:
        out["monitor_recv_errors"] = int(len(mon["recv_error_wall"]))
        out["monitor_last_error"] = str(mon["recv_last_error"])
    if mon is not None and len(mon["frame_ts"]):
        fts, farb, ffl = mon["frame_ts"], mon["frame_arb"], mon["frame_flags"]
        ext, rx, err = (ffl & 1) > 0, (ffl & 2) > 0, (ffl & 4) > 0
        wire_cmd = fts[(farb == CMD_ARB_ID) & ext & ~err]
        wire = {"cmd_frames_total": int(len(wire_cmd)),
                "error_frames": int(err.sum()),
                "accounting_ok": int(len(wire_cmd)) == int(len(ok_cmd))}
        wall0, mono0 = m["clock_pair"]
        if "clock_pair_end" in m:
            w1, m1 = m["clock_pair_end"]
            wire["wall_clock_drift_ms"] = float(((w1 - m1) - (wall0 - mono0)) * 1e3)
        to_wall = lambda t: np.asarray(t) + (wall0 - mono0)  # noqa: E731
        if wire["accounting_ok"]:
            # FIFO: the k-th successful send is the k-th command frame on the wire
            w = wire_cmd[in_window]
            wire["window_select"] = "index (exact)"
            lat = (w - to_wall(ok_cmd.loc[in_window, "t_send_end"].to_numpy())) * 1e3
            wire["send_to_wire_ms"] = _summary(lat)
        else:
            lo, hi = to_wall(t_send[0]) - 0.001, to_wall(t_send[-1]) + 0.02
            w = wire_cmd[(wire_cmd >= lo) & (wire_cmd <= hi)]
            wire["window_select"] = "clock (approximate)"
        if len(w) > 2:
            iv = np.diff(w) * 1e3
            wire.update({"cmd_frames_timed": int(len(w)),
                         "rate_hz": float((len(w) - 1) / (w[-1] - w[0])),
                         "interval_ms": _summary(iv), "_intervals": iv,
                         "_t": w[1:] - w[0]})
        # replies: frames from other nodes (the motor) during the timed window
        lo, hi = to_wall(ts_loops[0]), to_wall(ts_loops[-1])
        rx_sel = rx & ~err & (farb != CMD_ARB_ID)
        in_win = rx_sel & (fts >= lo) & (fts <= hi)
        ids, counts = np.unique(farb[rx_sel], return_counts=True)
        wire["rx_frames_timed"] = int(in_win.sum())
        wire["rx_rate_hz"] = float(in_win.sum() / (hi - lo)) if hi > lo else float("nan")
        wire["rx_ids"] = {f"0x{int(i):X}": int(c) for i, c in zip(ids, counts)}
        out["wire"] = wire
    elif mon is not None:
        out["wire"] = {"cmd_frames_total": 0, "accounting_ok": len(ok_cmd) == 0}

    a, b = m.get("can_link_before"), m.get("can_link_after")
    if a and b and "error" not in a and "error" not in b:
        def flat(link):
            s = link.get("stats64", {})
            x = link.get("linkinfo", {}).get("info_xstats", {})
            return {"tx_packets": s.get("tx", {}).get("packets"),
                    "tx_errors": s.get("tx", {}).get("errors"),
                    "tx_dropped": s.get("tx", {}).get("dropped"),
                    "rx_packets": s.get("rx", {}).get("packets"),
                    "rx_errors": s.get("rx", {}).get("errors"),
                    "rx_dropped": s.get("rx", {}).get("dropped"),
                    "rx_over_errors": s.get("rx", {}).get("over_errors"),
                    **{k: x.get(k) for k in ("bus_error", "error_warning", "error_passive",
                                             "bus_off", "restarts", "arbitration_lost")}}
        fa, fb = flat(a), flat(b)
        out["link_delta"] = {k: (fb[k] - fa[k]) if None not in (fa[k], fb[k]) else None
                             for k in fa}
        out["link_state_after"] = b.get("linkinfo", {}).get("info_data", {}).get("state")
    return out


def system_metrics(run):
    mon = run["monitor"]
    out = {"oom_check": run["meta"].get("oom_check")}
    if mon is None or not len(mon["sys_wall"]):
        return out
    rss = mon["sys_rss_kb"][mon["sys_rss_kb"] > 0] / 1024.0
    mem = mon["sys_memavail_kb"][mon["sys_memavail_kb"] > 0] / 1024.0
    temp = mon["sys_cpu_temp_mc"][mon["sys_cpu_temp_mc"] > 0] / 1000.0
    fmin = mon["sys_cpu_freq_min_khz"][mon["sys_cpu_freq_min_khz"] > 0] / 1000.0
    fmax = mon["sys_cpu_freq_max_khz"][mon["sys_cpu_freq_max_khz"] > 0] / 1000.0
    if len(rss):
        out["rss_mb"] = {"start": float(rss[0]), "end": float(rss[-1]), "max": float(rss.max())}
    if len(mem):
        out["mem_available_min_mb"] = float(mem.min())
    if len(temp):
        out["cpu_temp_max_c"] = float(temp.max())
    if len(fmin) and len(fmax):
        out["cpu_freq_mhz"] = {"min": float(fmin.min()), "max": float(fmax.max())}
    out["load1_max"] = float(mon["sys_load1"].max())
    return out


# ── verdict ─────────────────────────────────────────────────────────────────

def verdict(lm, cm, integ, sysm):
    crit = []

    def add(name, value, limit, ok, note=""):
        crit.append({"criterion": name, "value": value, "limit": limit,
                     "result": "PASS" if ok is True else "FAIL" if ok is False else "UNVERIFIED",
                     "note": note})

    tol = RATE_RESOLUTION_HZ / 2
    rate = lm.get("mean_rate_hz", float("nan"))
    add("Mean loop rate", f"{rate:.4f} Hz", f">= {TARGET_RATE_HZ:.0f} Hz",
        rate >= TARGET_RATE_HZ - tol)
    p99 = lm.get("period_ms", {}).get("p99", float("nan"))
    add("99th-percentile period", f"{p99:.2f} ms", f"<= {P99_PERIOD_MAX_MS:.0f} ms",
        p99 <= P99_PERIOD_MAX_MS)
    mx = lm.get("period_ms", {}).get("max", float("nan"))
    add("Largest period", f"{mx:.2f} ms", f"<= {MAX_PERIOD_MAX_MS:.0f} ms", mx <= MAX_PERIOD_MAX_MS)

    wire = cm.get("wire", {})
    if "rate_hz" in wire:
        drift = wire.get("wall_clock_drift_ms", 0.0)
        add("CAN torque-command rate (on wire)", f"{wire['rate_hz']:.4f} Hz",
            f">= {CAN_RATE_MIN_HZ:.0f} Hz", wire["rate_hz"] >= CAN_RATE_MIN_HZ - tol,
            "" if abs(drift) <= CLOCK_DRIFT_WARN_MS else
            f"the wall clock moved {drift:+.1f} ms against the monotonic clock during the run "
            "(NTP step?); wire timestamps are wall-clock, compare with the in-process rate")
    elif "rate_hz" in cm.get("inproc", {}):
        r = cm["inproc"]["rate_hz"]
        add("CAN torque-command rate (in-process only)", f"{r:.4f} Hz",
            f">= {CAN_RATE_MIN_HZ:.0f} Hz", r >= CAN_RATE_MIN_HZ - tol,
            "no bus monitor data: send() calls counted, not frames on the wire")
    else:
        add("CAN torque-command rate", "no data", f">= {CAN_RATE_MIN_HZ:.0f} Hz", False)

    restart_ok = integ["completed"] and integ["seq_contiguous"] and not integ["exception"]
    add("No controller restart", integ["status"], "single run, completed", restart_ok,
        integ["exception"] or "")

    loss = []
    if not integ["rows_match"]:
        loss.append("loop rows != loops recorded")
    if not integ["seq_contiguous"]:
        loss.append("loop sequence has gaps")
    if not integ.get("send_rows_match", True):
        loss.append("send rows != sends recorded")
    if integ["can_send_errors"]:
        loss.append(f"{integ['can_send_errors']} CAN send errors "
                    "(the deploy runner would have stopped at the first)")
    if wire and not wire.get("accounting_ok", True):
        loss.append(f"{wire.get('cmd_frames_total')} command frames on the wire vs "
                    f"{cm['inproc']['cmd_frames_total']} sent")
    if not integ.get("monitor_ok", True):
        loss.append("bus monitor did not exit cleanly")
    if cm.get("monitor_recv_errors"):
        loss.append(f"bus monitor saw {cm['monitor_recv_errors']} receive errors "
                    f"({cm['monitor_last_error']})")
    ld = cm.get("link_delta", {})
    for k in ("tx_errors", "tx_dropped", "rx_dropped", "rx_over_errors", "bus_off"):
        if ld.get(k):
            loss.append(f"CAN {k} +{ld[k]}")
    add("No log or frame loss", "none" if not loss else "; ".join(loss), "none", not loss)

    oom = (sysm.get("oom_check") or {})
    add("No out-of-memory event", "none" if oom.get("ok") else
        ("; ".join(oom.get("matches", [])[:2]) or oom.get("error", "not checked")),
        "none", oom.get("ok"))
    overall = all(c["result"] == "PASS" for c in crit)
    return crit, overall


def diagnose(lm, dt_ms):
    """Plain-language hints about why the rate criteria failed."""
    hints = []
    if "busy_ms" not in lm:
        return hints
    busy50 = lm["busy_ms"]["p50"]
    over50 = lm["sched_overshoot_ms"]["p50"]
    stages = sorted(lm["stages"].items(), key=lambda kv: -kv[1]["p50"])
    top = ", ".join(f"{v['label']} {v['p50']:.2f} ms" for _, v in stages[:4])
    if lm["mean_rate_hz"] < TARGET_RATE_HZ - RATE_RESOLUTION_HZ / 2:
        if busy50 + over50 < dt_ms * 0.98:
            hints.append(f"The loop work fits ({busy50:.2f} ms median of {dt_ms:.0f} ms) but the "
                         f"rate is still under target: each period loses {over50:.3f} ms to sleep "
                         "overshoot. Try --scheduler deadline.")
        else:
            hints.append(f"The loop work alone takes {busy50:.2f} ms (median) of the "
                         f"{dt_ms:.0f} ms budget. Largest stages (median): {top}.")
    if lm["period_ms"]["p99"] > P99_PERIOD_MAX_MS or lm["period_ms"]["max"] > MAX_PERIOD_MAX_MS:
        worst = sorted(lm["stages"].items(), key=lambda kv: -kv[1]["max"])[:3]
        hints.append("Largest single-stage spikes: " + ", ".join(
            f"{v['label']} {v['max']:.1f} ms (p99 {v['p99']:.1f})" for _, v in worst) + ".")
    return hints


# ── plots ───────────────────────────────────────────────────────────────────

def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


def _limits(ax, vertical=False):
    for v, lab in ((10, "10 ms"), (P99_PERIOD_MAX_MS, "p99 limit 20 ms"),
                   (MAX_PERIOD_MAX_MS, "max limit 50 ms")):
        (ax.axvline if vertical else ax.axhline)(v, color=MUTED, linewidth=0.8, linestyle="--")
        if not vertical:
            ax.annotate(lab, xy=(1.0, v), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                        textcoords="offset points", ha="right", fontsize=7, color=INK_2)


def make_plots(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    runs = [r for r in results if len(r["_periods"])]
    if not runs:
        return files

    # 1. period over time, one panel per run (same y scale)
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 2.6 * len(runs)), squeeze=False,
                             sharey=True, facecolor=SURFACE)
    ymax = max(60.0, max(r["_periods"].max() for r in runs) * 1.05)
    for ax, r in zip(axes[:, 0], runs):
        _style(ax)
        t = r["_ts"][1:] - r["_ts"][0]
        ax.plot(t, r["_periods"], ".", markersize=2, color=RUN_COLORS[r["kind"]])
        _limits(ax)
        ax.set_ylim(0, ymax)
        ax.set_title(f"{r['kind']}: loop period (start to start)", fontsize=9, color=INK,
                     loc="left")
        ax.set_ylabel("ms", fontsize=8, color=INK_2)
    axes[-1, 0].set_xlabel("time in timed window (s)", fontsize=8, color=INK_2)
    fig.tight_layout()
    files.append(out_dir / "periods_timeline.png")
    fig.savefig(files[-1], dpi=130)
    plt.close(fig)

    # 2. period distribution
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    _style(ax)
    hi = max(r["_periods"].max() for r in runs)
    bins = np.linspace(0, max(55.0, hi * 1.02), 220)
    for r in runs:
        ax.hist(r["_periods"], bins=bins, histtype="step", linewidth=1.5,
                color=RUN_COLORS[r["kind"]], label=f"{r['kind']} (n={len(r['_periods'])})")
    _limits(ax, vertical=True)
    ax.set_yscale("log")
    ax.set_xlabel("loop period (ms); dashed: 10 ms, 20 ms p99 limit, 50 ms max limit",
                  fontsize=8, color=INK_2)
    ax.set_ylabel("loops", fontsize=8, color=INK_2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    files.append(out_dir / "period_histogram.png")
    fig.savefig(files[-1], dpi=130)
    plt.close(fig)

    # 3. where the time goes (load run)
    load = next((r for r in runs if r["kind"] == "load"), None)
    if load and load["loop"].get("stages"):
        st = list(load["loop"]["stages"].values())
        labels = [s["label"] for s in st]
        y = np.arange(len(st))
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(st) + 1.2), facecolor=SURFACE)
        _style(ax)
        ax.barh(y - 0.2, [s["p50"] for s in st], height=0.36, color=RUN_COLORS["idle"],
                label="median")
        ax.barh(y + 0.2, [s["p99"] for s in st], height=0.36, color=RUN_COLORS["load"],
                label="99th percentile")
        ax.set_yticks(y, labels, fontsize=8, color=INK)
        ax.invert_yaxis()
        ax.axvline(1000.0 / TARGET_RATE_HZ, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_title("load: time spent in each loop stage", fontsize=9, color=INK, loc="left")
        ax.set_xlabel(f"ms per loop (dashed: whole {1000.0 / TARGET_RATE_HZ:.0f} ms budget)",
                      fontsize=8, color=INK_2)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        fig.tight_layout()
        files.append(out_dir / "stage_breakdown.png")
        fig.savefig(files[-1], dpi=130)
        plt.close(fig)

    # 4. CAN command intervals on the wire (load run)
    if load and "_intervals" in load["can"].get("wire", {}):
        w = load["can"]["wire"]
        fig, ax = plt.subplots(figsize=(10, 2.8), facecolor=SURFACE)
        _style(ax)
        ax.plot(w["_t"], w["_intervals"], ".", markersize=2, color=RUN_COLORS["load"])
        _limits(ax)
        ax.set_ylim(0, max(60.0, w["_intervals"].max() * 1.05))
        ax.set_title("load: interval between torque commands on the CAN bus", fontsize=9,
                     color=INK, loc="left")
        ax.set_xlabel("time in timed window (s)", fontsize=8, color=INK_2)
        ax.set_ylabel("ms", fontsize=8, color=INK_2)
        fig.tight_layout()
        files.append(out_dir / "can_intervals.png")
        fig.savefig(files[-1], dpi=130)
        plt.close(fig)
    return files


# ── report ──────────────────────────────────────────────────────────────────

def _f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_report(results, overall, crit, out_dir, plots):
    L = ["# Test 4 — Control loop and CAN command rate", ""]
    load = next((r for r in results if r["kind"] == "load"), None)
    if load:
        m = load["meta"]
        L += [f"**Verdict (load run): {'PASS' if overall else 'FAIL'}**", "",
              f"- Run: `{load['path']}`" + (" — **SIMULATED HARDWARE**" if m.get("sim") else ""),
              *([f"- **SIMULATED SENSORS**: no sensors attached; sensor reads are placeholder "
                 f"waits ({m.get('sim_sensor_read_ms')} ms). CAN bus, motor, TCN, safety checks "
                 "and logging are real. Re-run with the sensors connected for the final result."]
                if m.get("sim_sensors") and not m.get("sim") else []),
              f"- Controller: TCN `{m.get('tcn', {}).get('run_dir', '?')}`, backend "
              f"`{m.get('tcn', {}).get('backend', '?')}`; torque mode "
              f"`{m.get('motor', {}).get('torque_mode', '?')}`; scheduler "
              f"`{m['args']['scheduler']}`; gait injection "
              f"{'off' if m['args']['no_gait_inject'] else 'on'}",
              f"- Jetson: {m['system'].get('model')}; {m['system'].get('nvpmodel', '').splitlines()[0] if m['system'].get('nvpmodel') else ''}; "
              f"governor {m['system'].get('governors')}; git {m['system'].get('git_commit', '')[:8]} "
              f"({m['system'].get('git_branch')})", "",
              "| Criterion | Measured | Limit | Result |", "|---|---|---|---|"]
        L += [f"| {c['criterion']} | {c['value']} | {c['limit']} | **{c['result']}** |"
              for c in crit]
        notes = [c for c in crit if c["note"]]
        L += [""] + [f"- {c['criterion']}: {c['note']}" for c in notes]
        if load["hints"]:
            L += ["", "**Diagnosis**", ""] + [f"- {h}" for h in load["hints"]]

    L += ["", "## Script output", "",
          "| Metric | " + " | ".join(r["kind"] for r in results) + " |",
          "|---|" + "---|" * len(results)]

    def row(label, fn):
        vals = []
        for r in results:
            try:
                vals.append(fn(r))
            except (KeyError, TypeError, IndexError):
                vals.append("—")
        L.append(f"| {label} | " + " | ".join(vals) + " |")

    row("Timed loops", lambda r: str(r["loop"]["timed_loops"]))
    row("Window (s)", lambda r: _f(r["loop"]["window_s"], 1))
    row("Mean loop rate (Hz)", lambda r: _f(r["loop"]["mean_rate_hz"], 4))
    row("Median period (ms)", lambda r: _f(r["loop"]["period_ms"]["median"], 3))
    row("99th-pct period (ms)", lambda r: _f(r["loop"]["period_ms"]["p99"], 3))
    row("Largest period (ms)", lambda r: _f(r["loop"]["period_ms"]["max"], 3))
    row("Periods > 10 ms (%)", lambda r: _f(r["loop"]["pct_over_10ms"], 3))
    row("Periods > 20 ms (%)", lambda r: _f(r["loop"]["pct_over_20ms"], 3))
    row("Loop work > 10 ms (%)", lambda r: _f(r["loop"]["work_pct_over_10ms"], 3))
    row("Loop work > 20 ms (%)", lambda r: _f(r["loop"]["work_pct_over_20ms"], 3))
    row("Periods > 50 ms", lambda r: str(r["loop"]["count_over_50ms"]))
    row("Loop work p50 / p99 (ms)", lambda r: f"{_f(r['loop']['busy_ms']['p50'])} / "
                                               f"{_f(r['loop']['busy_ms']['p99'])}")
    row("CAN cmd rate, wire (Hz)", lambda r: _f(r["can"]["wire"]["rate_hz"]))
    row("CAN cmd rate, in-process (Hz)", lambda r: _f(r["can"]["inproc"]["rate_hz"]))
    row("CAN cmd interval p99 / max (ms)",
        lambda r: f"{_f(r['can']['wire']['interval_ms']['p99'])} / "
                  f"{_f(r['can']['wire']['interval_ms']['max'])}")
    row("Motor reply rate (Hz)", lambda r: _f(r["can"]["wire"]["rx_rate_hz"]))
    row("Bus error frames", lambda r: str(r["can"]["wire"]["error_frames"]))
    row("Watchdog trips", lambda r: str(r["loop"]["watchdog_trips"]))
    row("RSS start / max (MB)", lambda r: f"{_f(r['sys']['rss_mb']['start'], 0)} / "
                                          f"{_f(r['sys']['rss_mb']['max'], 0)}")
    row("Min MemAvailable (MB)", lambda r: _f(r["sys"]["mem_available_min_mb"], 0))
    row("Max CPU temp (°C)", lambda r: _f(r["sys"]["cpu_temp_max_c"], 1))
    row("CPU freq min / max (MHz)", lambda r: f"{_f(r['sys']['cpu_freq_mhz']['min'], 0)} / "
                                              f"{_f(r['sys']['cpu_freq_mhz']['max'], 0)}")

    if load and load["loop"].get("stages"):
        L += ["", "## Where the loop time goes (load run, ms)", "",
              "| Stage | mean | p50 | p99 | max |", "|---|---|---|---|---|"]
        for s in load["loop"]["stages"].values():
            L.append(f"| {s['label']} | {_f(s['mean'], 3)} | {_f(s['p50'], 3)} | "
                     f"{_f(s['p99'], 3)} | {_f(s['max'], 3)} |")
        L += ["", "Five longest periods (load run):", "",
              "| t (s) | period | loop work | step | TCN | IMU | FSR+enc | send |",
              "|---|---|---|---|---|---|---|---|"]
        for w in load["loop"]["worst_periods"]:
            L.append(f"| {_f(w['t_s'], 1)} | {_f(w['period_ms'])} | {_f(w['busy_ms'])} | "
                     f"{_f(w.get('step_ms'))} | {_f(w.get('predict_ms', 0.0))} | "
                     f"{_f(w.get('imu_ms', 0.0))} | {_f(w.get('fsr_enc_ms', 0.0))} | "
                     f"{_f(w.get('send_ms', 0.0))} |")

    flush = next((r for r in results if r["kind"] == "flush"), None)
    if flush and "stages" in flush["loop"] and "flush_ms" in flush["loop"]["stages"]:
        fl = flush["loop"]["stages"]["flush_ms"]
        L += ["", "## Disk-flush run (marked)", "",
              f"In-loop write + fsync every {flush['meta']['args']['flush_every']} s: "
              f"{fl['n']} loops did a flush, which cost p50 {_f(fl['p50'])} ms, "
              f"max {_f(fl['max'])} ms. Largest period {_f(flush['loop']['period_ms']['max'])} ms "
              f"vs {_f(load['loop']['period_ms']['max']) if load else '—'} ms without flushing. "
              f"Criteria on this run: " + ", ".join(
                  f"{c['criterion']} {c['result']}" for c in flush["crit"]) + "."]

    if load:
        cm = load["can"]
        L += ["", "## CAN detail (load run)", "", "```",
              json.dumps({k: v for k, v in cm.items()}, indent=2, default=_jsonable), "```"]
    L += ["", "## Plots", ""] + [f"![{p.stem}]({p.name})" for p in plots]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return f"<{len(o)} values>"
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def _strip_private(d):
    if isinstance(d, dict):
        return {k: _strip_private(v) for k, v in d.items() if not k.startswith("_")}
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="session folder, or run folders")
    ap.add_argument("--out", default=None, help="output folder (default: first path)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    runs = find_runs(args.paths)
    if not runs:
        sys.exit(f"no runs (meta.json) found in {args.paths}")
    out_dir = Path(args.out or args.paths[0])
    out_dir.mkdir(parents=True, exist_ok=True)

    order = {"idle": 0, "load": 1, "flush": 2}
    results = []
    for run in sorted(runs, key=lambda r: order[run_kind(r)]):
        lm, ts, periods = loop_metrics(run)
        res = {"kind": run_kind(run), "path": run["path"], "meta": run["meta"], "loop": lm,
               "integrity": integrity(run), "can": can_metrics(run, ts), "sys": system_metrics(run),
               "_ts": ts, "_periods": periods}
        if res["kind"] != "idle":
            res["crit"], res["pass"] = verdict(lm, res["can"], res["integrity"], res["sys"])
            res["hints"] = diagnose(lm, 1000.0 / run["meta"]["rate_hz"])
        results.append(res)

    load = next((r for r in results if r["kind"] == "load"), None)
    crit, overall = (load["crit"], load["pass"]) if load else ([], False)
    plots = [] if args.no_plots else make_plots(results, out_dir)
    write_report(results, overall, crit, out_dir, plots)
    with open(out_dir / "report.json", "w") as f:
        json.dump({"overall_pass": overall if load else None,
                   "runs": [_strip_private({k: v for k, v in r.items() if k != "meta"})
                            for r in results]},
                  f, indent=2, default=_jsonable)

    # console summary
    for r in results:
        lm = r["loop"]
        if "mean_rate_hz" not in lm:
            print(f"{r['kind']:>5}: {lm.get('error')}")
            continue
        wire = r["can"].get("wire", {}).get("rate_hz")
        inproc = r["can"].get("inproc", {}).get("rate_hz")
        print(f"{r['kind']:>5}: {lm['mean_rate_hz']:7.2f} Hz  period p50 "
              f"{lm['period_ms']['median']:6.2f}  p99 {lm['period_ms']['p99']:6.2f}  max "
              f"{lm['period_ms']['max']:6.2f} ms  >10ms {lm['pct_over_10ms']:5.2f}%  "
              f">20ms {lm['pct_over_20ms']:5.2f}%  CAN "
              + (f"{wire:7.2f} Hz (wire)" if wire else
                 f"{inproc:7.2f} Hz (in-process)" if inproc else "—"))
    if load:
        print()
        for c in crit:
            print(f"  [{c['result']:>10}] {c['criterion']}: {c['value']} (limit {c['limit']})")
        for h in load["hints"]:
            print(f"  -> {h}")
        print(f"\nTest 4 {'PASS' if overall else 'FAIL'}"
              + ("  (SIMULATED HARDWARE — not a test result)" if load["meta"].get("sim") else
                 "  (SIMULATED SENSORS — sensor read times are placeholders)"
                 if load["meta"].get("sim_sensors") else ""))
    print(f"report: {out_dir / 'report.md'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
