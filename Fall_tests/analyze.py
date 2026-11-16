#!/usr/bin/env python3
"""
analyze.py -- scoring scripts for Fall Test Plan Tests 1-3.

    python3 analyze.py test1 runs/<run> [runs/<run> ...]
    python3 analyze.py test2 runs/<run> [...]          # positive-control runs use run.json "freeze"
    python3 analyze.py test3 runs/<run> [...]
    python3 analyze.py test3 --off runs/<walk_off> --on runs/<walk_on>   # adds the actuator delta check

Each command prints a PASS/FAIL summary, the rows used / dropped / missing, and
writes its metrics into <run>/summary.csv (one row per metric; rows from other
tests in the same file are kept).
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Required rates (R-02). Expected spacing for gap / missing checks comes from these.
REQ_HZ = {"imu": 100.0, "fsr": 100.0, "encoder": 200.0}
GAP_FACTOR = 3.0
MISSING_MAX_PCT = 3.0
FAILED_MAX_PCT = 1.0
REPEAT_MAX = 5                 # T2: no channel unchanged for more than 5 samples
MOVE_DEG_S = 5.0               # T2: motion = encoder speed above this ...
# Revised criterion (agreed with test lead): the encoder is judged by time, not
# sample count. At 0.088 deg/count and ~370 Hz a live encoder cannot change at
# least every 6 samples below ~5.4 deg/s, which every motion reversal crosses.
ENC_UNCHANGED_MAX_MS = 50.0
FSR_LOADED_V = 0.05            # ... or an FSR above this voltage (loaded pad)
T3_P99_MAX_MS = 10.0
T3_DELTA_MAX_MS = 2.0
COPIED_MAX_PCT = 0.1           # revised: copied-value loops allowed (plan: 0)
SUMMARY_HEADER = ["test", "run_id", "channel", "metric", "value", "unit", "limit", "result"]


def kind(ch):
    return "imu" if ch.startswith("imu") else ("fsr" if ch.startswith("fsr") else "encoder")


# ── loading ─────────────────────────────────────────────────────────────────
class Run:
    def __init__(self, path):
        self.dir = Path(path)
        self.meta = json.loads((self.dir / "run.json").read_text())
        self.id = self.meta["run_id"]
        self.channels = self.meta["channels"]
        cols = defaultdict(lambda: defaultdict(list))
        self.n_rows = 0
        with open(self.dir / "samples.csv", newline="") as f:
            for r in csv.DictReader(f):
                self.n_rows += 1
                c = cols[r["channel"]]
                c["sid"].append(int(r["sample_id"]))
                c["t"].append(int(r["t_read_ns"]))
                c["ok"].append(r["status"] == "ok")
                c["raw"].append(r["raw"])
                c["conv"].append(r["converted"])
        self.ch = {}
        for name, c in cols.items():
            self.ch[name] = {
                "sid": np.array(c["sid"], dtype=np.int64),
                "t": np.array(c["t"], dtype=np.int64),
                "ok": np.array(c["ok"], dtype=bool),
                "raw": c["raw"], "conv": c["conv"],
            }
        self.t0, self.t1 = self.meta["t_start_ns"], self.meta["t_end_ns"]
        self.dur = (self.t1 - self.t0) / 1e9
        events = self.dir / "events.csv"
        self.restarts = 0
        if events.exists():
            with open(events, newline="") as f:
                self.restarts = sum(1 for r in csv.DictReader(f) if r["event"].startswith("restart"))

    def control(self):
        with open(self.dir / "control.csv", newline="") as f:
            rows = list(csv.reader(f))
        head, body = rows[0], rows[1:]
        a = np.array([[int(x) for x in r[1:]] for r in body], dtype=np.int64)
        return head[1:], a


class Summary:
    def __init__(self, test, run):
        self.test, self.run, self.rows, self.ok = test, run, [], True

    def add(self, channel, metric, value, unit="", limit="", passed=None):
        # passed may be a numpy bool, so test truthiness, never `is False`
        res = "" if passed is None else ("PASS" if passed else "FAIL")
        if passed is not None and not passed:
            self.ok = False
        if isinstance(value, float):
            value = f"{value:.4g}"
        self.rows.append([self.test, self.run.id, channel, metric, value, unit, limit, res])

    def write(self):
        path = self.run.dir / "summary.csv"
        keep = []
        if path.exists():
            with open(path, newline="") as f:
                keep = [r for r in csv.reader(f)][1:]
            keep = [r for r in keep if r[0] != self.test]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(SUMMARY_HEADER)
            w.writerows(keep + self.rows)
        return path


def table(rows, head):
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
    line = lambda r: "  ".join(str(x).rjust(w) for x, w in zip(r, widths))
    print(line(head))
    print("-" * (sum(widths) + 2 * len(widths)))
    for r in rows:
        print(line(r))


def longest_true_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


# ── Test 1 ──────────────────────────────────────────────────────────────────
def test1(run):
    s = Summary("T1", run)
    out = []
    for ch in run.channels:
        d = run.ch.get(ch)
        if d is None:
            s.add(ch, "samples", 0, "", ">0", False)
            out.append((ch, 0, "-", "-", "-", "-", "-", "FAIL"))
            continue
        k = kind(ch)
        spacing = 1.0 / REQ_HZ[k]
        t_ok = d["t"][d["ok"]] / 1e9
        n_ok, n = len(t_ok), len(d["t"])
        rate = n_ok / run.dur
        # include the run edges so a sensor that stops early shows as a gap
        edges = np.concatenate(([run.t0 / 1e9], t_ok, [run.t1 / 1e9]))
        gaps = np.diff(edges)
        max_gap = gaps.max() if len(gaps) else float("inf")
        missed = np.clip(np.floor(gaps / spacing) - 1, 0, None).sum()
        expected = run.dur / spacing
        miss_pct = 100.0 * missed / expected
        fail_pct = 100.0 * (n - n_ok) / n if n else 100.0
        fail_run = longest_true_run(~d["ok"])
        p_rate = rate >= REQ_HZ[k]
        p_gap = max_gap <= GAP_FACTOR * spacing
        p_miss = miss_pct < MISSING_MAX_PCT
        p_fail = fail_pct < FAILED_MAX_PCT
        s.add(ch, "mean_rate", rate, "Hz", f">={REQ_HZ[k]:.0f}", p_rate)
        s.add(ch, "largest_gap", max_gap * 1e3, "ms", f"<={GAP_FACTOR * spacing * 1e3:.0f}", p_gap)
        s.add(ch, "missing_samples", miss_pct, "%", f"<{MISSING_MAX_PCT}", p_miss)
        s.add(ch, "failed_reads", fail_pct, "%", f"<{FAILED_MAX_PCT}", p_fail)
        s.add(ch, "longest_failed_run", fail_run, "reads")
        allp = p_rate and p_gap and p_miss and p_fail
        out.append((ch, n_ok, f"{rate:.1f}", f"{max_gap * 1e3:.1f}",
                    f"{miss_pct:.2f}", f"{fail_pct:.3f}", fail_run, "PASS" if allp else "FAIL"))
    s.add("all", "driver_restarts", run.restarts, "", "0", run.restarts == 0)
    table(out, ("channel", "ok_reads", "Hz", "max_gap_ms", "missing_%", "failed_%",
                "fail_run", "result"))
    print(f"driver restarts: {run.restarts}")
    return s


# ── Test 2 ──────────────────────────────────────────────────────────────────
def motion_masks(run):
    """Activity intervals, each (t_ns array, bool array):
        "ankle"    encoder speed > MOVE_DEG_S      (plan's motion rule)
        "fsr_toe" / "fsr_heel"  that pad loaded (> FSR_LOADED_V)
    """
    masks = {}
    enc = run.ch.get("encoder")
    if enc is not None and enc["ok"].sum() > 10:
        t = enc["t"][enc["ok"]]
        ang = np.unwrap(np.radians([float(x) for x, o in zip(enc["conv"], enc["ok"]) if o]))
        ang = np.degrees(ang)
        # speed over a 50 ms window so encoder quantisation doesn't fake zero speed
        win = int(0.05e9)
        idx = np.searchsorted(t, t - win)
        dt = (t - t[idx]) / 1e9
        spd = np.where(dt > 0, np.abs(ang - ang[idx]) / np.where(dt > 0, dt, 1), 0)
        masks["ankle"] = (t, spd > MOVE_DEG_S)
    for ch in ("fsr_toe", "fsr_heel"):
        d = run.ch.get(ch)
        if d is not None and d["ok"].sum() > 10:
            t = d["t"][d["ok"]]
            v = np.array([float(x) for x, o in zip(d["conv"], d["ok"]) if o])
            masks[ch] = (t, v > FSR_LOADED_V)
    return masks


def active_for(ch):
    """Each channel is judged only while its own input is active: IMUs and the
    encoder while the ankle moves, and each FSR while that pad is loaded (an
    unloaded pad legitimately sits at a constant noise floor)."""
    if ch.startswith("fsr"):
        return ch
    return "ankle"


def sample_mask(masks, key, tq):
    if key not in masks:
        return np.zeros(len(tq), dtype=bool)
    t, flag = masks[key]
    i = np.clip(np.searchsorted(t, tq, side="right") - 1, 0, len(t) - 1)
    return flag[i]


def longest_span_ms(rep, t):
    """Longest time a value stayed unchanged; rep[i] means sample i+1 == sample i."""
    cur_start = 0
    best_ms = 0.0
    run = False
    for i, r in enumerate(rep):
        if r and not run:
            cur_start, run = i, True
        if run and (not r or i == len(rep) - 1):
            end = i if not r else i + 1
            best_ms = max(best_ms, (t[end] - t[cur_start]) / 1e6)
            run = False
    return best_ms


def test2(run):
    s = Summary("T2", run)
    masks = motion_masks(run)
    freeze = run.meta.get("freeze")
    out, flagged = [], []
    for ch in run.channels:
        d = run.ch.get(ch)
        if d is None:
            continue
        t_all = d["t"]
        noninc = int((np.diff(t_all) <= 0).sum())
        ok = d["ok"]
        t = t_all[ok]
        raw = [r for r, o in zip(d["raw"], ok) if o]
        key = active_for(ch)
        mv = sample_mask(masks, key, t)
        same = np.array([a == b for a, b in zip(raw, raw[1:])], dtype=bool)
        rep = same & mv[1:] & mv[:-1]      # unchanged sample while this channel is active
        n_mv = int(mv.sum())
        rep_pct = 100.0 * rep.sum() / max(1, n_mv - 1)
        longest = longest_true_run(rep)
        span = longest_span_ms(rep, t)
        p_ts = noninc == 0
        if ch == "encoder":
            p_rep = span <= ENC_UNCHANGED_MAX_MS
            limit = f"<={ENC_UNCHANGED_MAX_MS:g} ms"
        else:
            p_rep = longest <= REPEAT_MAX
            limit = f"<={REPEAT_MAX} samples"
        tested = n_mv > 1
        if tested and not p_rep:
            flagged.append(ch)
        s.add(ch, "non_increasing_timestamps", noninc, "", "0", p_ts)
        s.add(ch, "active_condition", key)
        s.add(ch, "active_samples", n_mv, "")
        s.add(ch, "repeated_while_active", rep_pct, "%")
        verdict = (p_rep if tested else None) if freeze is None else None
        enc = ch == "encoder"
        s.add(ch, "longest_repeat_run", longest, "samples",
              "info" if enc else f"<={REPEAT_MAX}", None if enc else verdict)
        s.add(ch, "longest_unchanged_time", span, "ms",
              f"<={ENC_UNCHANGED_MAX_MS:g}" if enc else "info", verdict if enc else None)
        res = ("PASS" if (p_ts and p_rep) else "FAIL") if tested else ("PASS" if p_ts else "FAIL") + " (not active)"
        out.append((ch, key, noninc, n_mv, f"{rep_pct:.2f}", longest, f"{span:.1f}", res))
    table(out, ("channel", "judged_while", "non_incr_ts", "active_n", "repeat_%",
                "longest_run", "longest_ms", "result"))
    print(f"limits: unchanged <= {REPEAT_MAX} samples (IMU, FSR); "
          f"<= {ENC_UNCHANGED_MAX_MS:g} ms (encoder)")
    if sum(r[3] for r in out) == 0:
        print("WARNING: no channel was active (ankle still, FSRs unloaded) -- "
              "the repeat check had nothing to test.")
    if freeze:
        detected = flagged == [freeze]
        s.add("all", "positive_control_detected", ";".join(flagged) or "none",
              "", freeze, detected)
        print(f"positive control: froze {freeze}, script flagged {flagged or 'nothing'} "
              f"-> {'DETECTED' if detected else 'NOT DETECTED'}")
    return s


# ── Test 3 ──────────────────────────────────────────────────────────────────
def test3(run):
    s = Summary("T3", run)
    head, a = run.control()
    chans = [h[len("sid_"):] for h in head[3:]]
    sids = a[:, 3:]
    # sample id -> read time
    t_of = {}
    for d in run.ch.values():
        for sid, t in zip(d["sid"], d["t"]):
            t_of[int(sid)] = int(t)
    n_loops = len(a)
    missing = (sids < 0).any(axis=1)
    copied = np.zeros(n_loops, dtype=bool)
    copied[1:] = ((sids[1:] == sids[:-1]) & (sids[1:] >= 0)).any(axis=1)
    copied_by_ch = ((sids[1:] == sids[:-1]) & (sids[1:] >= 0)).sum(axis=0)
    good = ~missing
    times = np.full(sids.shape, np.nan)
    for j in range(sids.shape[1]):
        times[:, j] = [t_of.get(int(x), np.nan) if x >= 0 else np.nan for x in sids[:, j]]
    skew_ms = (np.nanmax(times[good], axis=1) - np.nanmin(times[good], axis=1)) / 1e6
    med, p99, mx = (np.median(skew_ms), np.percentile(skew_ms, 99), skew_ms.max()) \
        if len(skew_ms) else (np.nan,) * 3
    copied_pct = 100.0 * copied.sum() / max(1, n_loops)
    p_copy = copied_pct <= COPIED_MAX_PCT
    p_p99 = p99 <= T3_P99_MAX_MS
    s.add("all", "loops", n_loops)
    s.add("all", "missing_value_loops", int(missing.sum()), "", "0", not missing.any())
    s.add("all", "copied_value_loops", int(copied.sum()), "")
    s.add("all", "copied_value_loops_pct", copied_pct, "%", f"<={COPIED_MAX_PCT}", p_copy)
    for ch, c in zip(chans, copied_by_ch):
        s.add(ch, "copied_value_loops", int(c))
    s.add("all", "skew_median", med, "ms")
    s.add("all", "skew_p99", p99, "ms", f"<={T3_P99_MAX_MS}", p_p99)
    s.add("all", "skew_max", mx, "ms")
    # pairwise: median |t_i - t_j|
    pair_rows = []
    for i in range(len(chans)):
        for j in range(i + 1, len(chans)):
            dd = np.abs(times[good, i] - times[good, j]) / 1e6
            m = float(np.nanmedian(dd)) if len(dd) else np.nan
            p = float(np.nanpercentile(dd, 99)) if len(dd) else np.nan
            s.add(f"{chans[i]}|{chans[j]}", "pair_skew_median", m, "ms")
            s.add(f"{chans[i]}|{chans[j]}", "pair_skew_p99", p, "ms")
            pair_rows.append((chans[i], chans[j], f"{m:.2f}", f"{p:.2f}"))
    print(f"loops {n_loops}  missing-value loops {int(missing.sum())}  "
          f"copied-value loops {int(copied.sum())} ({copied_pct:.3f}%) "
          f"-> {'PASS' if p_copy else 'FAIL'} (<= {COPIED_MAX_PCT}%)")
    print("copied per channel: " + ", ".join(f"{c}={int(n)}" for c, n in zip(chans, copied_by_ch)))
    print(f"read-time skew: median {med:.2f} ms, p99 {p99:.2f} ms, max {mx:.2f} ms "
          f"-> {'PASS' if p_p99 else 'FAIL'} (p99 <= {T3_P99_MAX_MS} ms)")
    table(pair_rows, ("ch_a", "ch_b", "median_ms", "p99_ms"))
    s.p99 = p99
    return s


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=["test1", "test2", "test3"])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--off", help="T3: actuator-off walking run")
    ap.add_argument("--on", help="T3: actuator-on walking run")
    args = ap.parse_args()
    fn = {"test1": test1, "test2": test2, "test3": test3}[args.test]
    paths = list(args.runs) + [p for p in (args.off, args.on) if p]
    if not paths:
        ap.error("give at least one run folder")
    results = {}
    all_ok = True
    for p in paths:
        run = Run(p)
        print(f"\n=== {args.test.upper()}  {run.id}  ({run.dur:.1f} s, "
              f"condition {run.meta['condition']}) ===")
        s = fn(run)
        results[p] = s
        n_ok = sum(int(d['ok'].sum()) for d in run.ch.values())
        print(f"rows: {run.n_rows} total, {n_ok} used, {run.n_rows - n_ok} dropped (failed reads), "
              f"missing channels: {[c for c in run.channels if c not in run.ch] or 'none'}")
        if args.test == "test3" and args.off and args.on and p == args.on:
            delta = s.p99 - results[args.off].p99
            ok = delta <= T3_DELTA_MAX_MS
            s.add("all", "p99_increase_vs_actuator_off", delta, "ms",
                  f"<={T3_DELTA_MAX_MS}", ok)
            print(f"actuator-on p99 - actuator-off p99 = {delta:+.2f} ms "
                  f"-> {'PASS' if ok else 'FAIL'}")
        path = s.write()
        print(f"{'PASS' if s.ok else 'FAIL'}  -> {path}")
        all_ok &= s.ok
    print(f"\nOVERALL {args.test.upper()}: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
