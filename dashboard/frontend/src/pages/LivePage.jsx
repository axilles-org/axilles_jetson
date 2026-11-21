import React, { useEffect, useRef, useState } from "react";
import LiveChart from "../components/LiveChart.jsx";
import { connectLiveSocket, getAssist, getJobStatus, setAssist, stopJob } from "../lib/api.js";

const SCALE_STEP = 0.01;
const SCALE_MIN = 0;
const SCALE_MAX = 0.3; // jetson_deploy.py --max-scale default; the controller caps it too

const clampScale = (x) => Math.min(SCALE_MAX, Math.max(SCALE_MIN, Math.round(x * 100) / 100));

// process_manager.LIVE_DELAY_MIN_MS/MAX_MS; the backend and controller enforce them too
const DELAY_STEP_MS = 5;
const DELAY_MIN_MS = 10;
const DELAY_MAX_MS = 40;

const clampDelay = (x) =>
  Math.min(DELAY_MAX_MS, Math.max(DELAY_MIN_MS, Math.round(x / DELAY_STEP_MS) * DELAY_STEP_MS));

function AssistPanel({ latest }) {
  const [job, setJob] = useState(null);
  // What the backend is currently heartbeating to the controller
  const [desired, setDesired] = useState({ assist: false, scale: null, delay_ms: null });
  // Value chosen with the arrows, sent only when "Set" is pressed
  const [pending, setPending] = useState(null);
  const [pendingDelay, setPendingDelay] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const poll = () => {
      getJobStatus().then(setJob).catch(() => {});
      getAssist()
        .then((s) => {
          setDesired(s);
          setPending((p) => (p === null ? clampScale(s.scale) : p));
        })
        .catch(() => {});
    };
    poll();
    // 500 ms so a remote-button press shows up promptly in the delay box
    const id = setInterval(poll, 500);
    return () => clearInterval(id);
  }, []);

  // A new run starts from its own launch delay, not the last run's arrows.
  useEffect(() => setPendingDelay(null), [job?.job_id]);

  // The backend's delay changed (Set here, or a remote-button press): show it.
  // No priority between the two, so this replaces arrow clicks not yet Set.
  useEffect(() => {
    if (typeof desired.delay_ms === "number") setPendingDelay(desired.delay_ms);
  }, [desired.delay_ms]);

  // Delay the controller is working towards: the one set here, else the launch delay.
  const requestedDelay = desired.delay_ms ?? latest.assistance_delay_ms ?? null;
  useEffect(() => {
    if (pendingDelay === null && typeof requestedDelay === "number") {
      setPendingDelay(clampDelay(requestedDelay));
    }
  }, [pendingDelay, requestedDelay]);

  if (!job?.live || job.status !== "running") return null;

  const calibrated = latest.calibrated === 1;
  const ready = calibrated && latest.buffer_ready === 1;
  const on = latest.assist_on === 1; // what the controller reports it is doing
  const appliedScale = latest.assist_scale ?? desired.scale;
  const state = !calibrated
    ? "Calibrating: walk normally to zero the ankle angle"
    : !ready
    ? "Filling the 3 s window"
    : on
    ? `Ramp ${(latest.session_ramp ?? 0).toFixed(2)}`
    : "Passive: predicting, 0 N·m sent";

  async function send(assist, scale, delayMs = null) {
    setError(null);
    setBusy(true);
    try {
      // null leaves the backend's scale / delay unchanged
      setDesired(await setAssist(assist, scale == null ? null : Number(scale), delayMs));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const unsaved = pending !== null && desired.scale !== null && pending !== clampScale(desired.scale);

  const appliedDelay = latest.assistance_delay_ms;
  const delayUnsaved =
    pendingDelay !== null && typeof requestedDelay === "number" && pendingDelay !== requestedDelay;
  // Set, but the controller holds it until the stance in progress ends
  const delayWaiting =
    typeof latest.pending_delay_ms === "number" &&
    typeof appliedDelay === "number" &&
    latest.pending_delay_ms !== appliedDelay;

  return (
    <aside className="card assist-panel">
      <h3>Assistance</h3>

      <div className={`assist-indicator ${on ? "on" : "off"}`}>{on ? "ASSIST ON" : "ASSIST OFF"}</div>
      <div className="assist-state">{state}</div>

      <div className="assist-toggle">
        <button
          className="assist-btn on"
          onClick={() => send(true, desired.scale)}
          disabled={busy || !ready || on || desired.scale === null}
        >
          ON
        </button>
        <button className="assist-btn off" onClick={() => send(false, desired.scale)} disabled={busy}>
          OFF
        </button>
      </div>

      <div className="assist-scale">
        <div className="assist-label">Scale</div>
        <div className="assist-stepper">
          <span className={`assist-scale-value ${unsaved ? "unsaved" : ""}`}>
            {pending === null ? "—" : pending.toFixed(2)}
          </span>
          <div className="assist-arrows">
            <button
              aria-label="Increase scale"
              onClick={() => setPending((p) => clampScale(p + SCALE_STEP))}
              disabled={pending === null || pending >= SCALE_MAX}
            >
              ▲
            </button>
            <button
              aria-label="Decrease scale"
              onClick={() => setPending((p) => clampScale(p - SCALE_STEP))}
              disabled={pending === null || pending <= SCALE_MIN}
            >
              ▼
            </button>
          </div>
        </div>
        <button
          className="assist-btn set"
          onClick={() => send(desired.assist, pending)}
          disabled={busy || pending === null || !unsaved}
        >
          Set
        </button>
        <div className="assist-dim">
          applied {appliedScale == null ? "—" : Number(appliedScale).toFixed(2)}
          {unsaved && <span className="assist-unsaved"> · not set yet</span>}
        </div>
      </div>

      <div className="assist-scale">
        <div className="assist-label">Assistance delay</div>
        <div className="assist-stepper">
          <span className={`assist-scale-value ${delayUnsaved ? "unsaved" : ""}`}>
            {pendingDelay === null ? "—" : pendingDelay}
            <span className="assist-unit"> ms</span>
          </span>
          <div className="assist-arrows">
            <button
              aria-label="Increase assistance delay"
              onClick={() => setPendingDelay((d) => clampDelay(d + DELAY_STEP_MS))}
              disabled={pendingDelay === null || pendingDelay >= DELAY_MAX_MS}
            >
              ▲
            </button>
            <button
              aria-label="Decrease assistance delay"
              onClick={() => setPendingDelay((d) => clampDelay(d - DELAY_STEP_MS))}
              disabled={pendingDelay === null || pendingDelay <= DELAY_MIN_MS}
            >
              ▼
            </button>
          </div>
        </div>
        <button
          className="assist-btn set"
          onClick={() => send(desired.assist, null, pendingDelay)}
          disabled={busy || pendingDelay === null || !delayUnsaved}
        >
          Set
        </button>
        <div className="assist-dim">
          applied {typeof appliedDelay === "number" ? `${appliedDelay} ms` : "—"}
          {delayWaiting && (
            <span className="assist-unsaved">
              {" "}
              · {latest.pending_delay_ms} ms after this stance
            </span>
          )}
          {delayUnsaved && <span className="assist-unsaved"> · not set yet</span>}
        </div>
      </div>

      <div className="assist-dim">
        sent {(latest.sent_torque_nm ?? 0).toFixed(2)} N·m
        <br />
        predicted {(latest.predicted_nm_per_kg ?? 0).toFixed(2)} N·m/kg
      </div>

      <button className="assist-btn stop" onClick={() => stopJob().catch((e) => setError(e.message))}>
        Stop run
      </button>
      {error && <p className="assist-error">{error}</p>}
    </aside>
  );
}

const WINDOW_S = 10; // seconds of history on screen
const STALE_MS = 1500; // no sample for this long: say so

const MONITOR_DEVICES = {
  imu_foot: "Foot IMU",
  imu_shank: "Shank IMU",
  encoder: "Encoder",
  adc: "FSR ADC",
};

// Charts are always shown. Sensor data comes from the dashboard's sensor monitor
// while no controller runs, then from the controller's own stream. A series reads
// the first of its keys present in a sample:
//   zero: true      controller-only output, drawn as 0 until a controller sends it
//   optional: true  added to the chart once it first arrives (TCN calibrated angle)
//   right: true     second y-axis
const imuAxes = (prefix, kind, letter) =>
  ["x", "y", "z"].map((axis, i) => ({
    label: `${prefix} ${kind} ${axis}`,
    keys: [`${prefix}_${letter}${axis}`],
    color: ["#ff5c5c", "#3ddc84", "#5b9dff"][i],
  }));

const CHARTS = [
  {
    id: "ankle",
    title: "Ankle Angle / Encoder",
    unit: "deg",
    rightUnit: "deg (TCN)",
    series: [
      { label: "encoder", keys: ["ankle_deg"], color: "#5b9dff" },
      { label: "TCN calibrated", keys: ["ankle_angle_deg"], color: "#c792ea", right: true, optional: true },
    ],
  },
  {
    id: "velocity",
    title: "Ankle Velocity",
    unit: "deg/s",
    series: [{ label: "velocity", keys: ["ankle_velocity_dps"], color: "#ffb238" }],
  },
  {
    id: "torque",
    title: "Torque",
    unit: "N·m",
    series: [
      { label: "TBE command", keys: ["torque_cmd"], color: "#ff5c5c", zero: true },
      { label: "ML command", keys: ["assist_cmd_nm"], color: "#ffb238", zero: true },
      { label: "ML sent", keys: ["sent_torque_nm"], color: "#5b9dff", zero: true },
    ],
  },
  {
    id: "prediction",
    title: "ML Prediction",
    unit: "N·m/kg",
    series: [{ label: "predicted", keys: ["predicted_nm_per_kg"], color: "#c792ea", zero: true }],
  },
  {
    id: "fsr",
    title: "FSR (Heel / Toe)",
    unit: "counts",
    series: [
      { label: "heel", keys: ["heel_fsr_raw", "heel_fsr"], color: "#3ddc84" },
      { label: "toe", keys: ["toe_fsr_raw", "toe_fsr"], color: "#5b9dff" },
    ],
  },
  {
    id: "gait",
    title: "Gait State",
    unit: "",
    series: ["stance", "ramp", "session_ramp", "assist_on", "calibrated"].map((k, i) => ({
      label: k,
      keys: [k],
      color: ["#3ddc84", "#ffb238", "#5b9dff", "#ff5c5c", "#c792ea"][i],
      zero: true,
    })),
  },
  { id: "foot_acc", title: "Foot IMU Accel", unit: "m/s²", series: imuAxes("foot", "accel", "a") },
  { id: "foot_gyro", title: "Foot IMU Gyro", unit: "rad/s", series: imuAxes("foot", "gyro", "g") },
  { id: "shank_acc", title: "Shank IMU Accel", unit: "m/s²", series: imuAxes("shank", "accel", "a") },
  { id: "shank_gyro", title: "Shank IMU Gyro", unit: "rad/s", series: imuAxes("shank", "gyro", "g") },
];

const ALL_SERIES = CHARTS.flatMap((c) => c.series.map((s) => ({ ...s, id: `${c.id}:${s.label}` })));

function pick(msg, keys) {
  for (const k of keys) {
    if (typeof msg[k] === "number" && Number.isFinite(msg[k])) return msg[k];
  }
  return null;
}

function makeRing() {
  return { t: [], values: Object.fromEntries(ALL_SERIES.map((s) => [s.id, []])) };
}

export default function LivePage() {
  const [connected, setConnected] = useState(false);
  const [, setTick] = useState(0);
  const ringRef = useRef(makeRing());
  const startRef = useRef(null);
  const seenRef = useRef(new Set()); // optional series that have arrived
  const latestRef = useRef({});
  const lastMsgRef = useRef(0);
  const frameRef = useRef(null);

  useEffect(() => {
    const redraw = () => {
      if (frameRef.current !== null) return;
      frameRef.current = requestAnimationFrame(() => {
        frameRef.current = null;
        setTick((n) => n + 1);
      });
    };

    const ws = connectLiveSocket((msg) => {
      setConnected(true);
      latestRef.current = msg;
      lastMsgRef.current = Date.now();

      const t = typeof msg.server_t === "number" ? msg.server_t : Date.now() / 1000;
      if (startRef.current === null) startRef.current = t;
      const ring = ringRef.current;
      ring.t.push(t - startRef.current);
      for (const s of ALL_SERIES) {
        let v = pick(msg, s.keys);
        if (v !== null && s.optional) seenRef.current.add(s.id);
        if (v === null && s.zero) v = 0;
        ring.values[s.id].push(v);
      }

      // drop samples older than the window, in batches
      const cutoff = ring.t[ring.t.length - 1] - WINDOW_S;
      if (ring.t[0] < cutoff - 1) {
        let n = 0;
        while (n < ring.t.length && ring.t[n] < cutoff) n++;
        ring.t.splice(0, n);
        for (const arr of Object.values(ring.values)) arr.splice(0, n);
      }
      redraw();
    });

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);

    // re-render now and then so a stalled stream shows as stalled
    const staleTimer = setInterval(() => setTick((n) => n + 1), 500);

    return () => {
      ws.close();
      clearInterval(staleTimer);
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    };
  }, []);

  const ring = ringRef.current;
  const latest = latestRef.current;
  const stale = Date.now() - lastMsgRef.current > STALE_MS;
  const fromMonitor = latest.source === "monitor";

  let statusText;
  if (!connected) statusText = "Waiting for the dashboard backend…";
  else if (stale) statusText = "Connected · no sensor data";
  else if (fromMonitor) statusText = "Sensors only · no controller running";
  else statusText = `Controller running${latest.run_id ? ` · run ${latest.run_id}` : ""}`;

  const problems =
    !stale && fromMonitor && latest.monitor
      ? Object.entries(latest.monitor).map(([dev, reason]) => `${MONITOR_DEVICES[dev] ?? dev}: ${reason}`)
      : [];

  const ankleSubtitle = stale
    ? null
    : latest.ankle_zeroed === 1
    ? "zeroed to standing angle"
    : "raw (no standing angle set on Ankle Limits)";

  return (
    <div className="live-layout">
      <div className="live-main">
        <div className="status-row">
          <span className={`status-dot ${connected && !stale ? "connected" : ""}`} />
          <span>{statusText}</span>
          {!fromMonitor && !stale && latest.controller && (
            <span style={{ color: "var(--text-dim)" }}>· {latest.controller}</span>
          )}
        </div>
        {problems.length > 0 && <p className="live-problems">{problems.join(" · ")}</p>}

        <div className="grid">
          {CHARTS.map((chart) => {
            const series = chart.series.filter(
              (s) => !s.optional || seenRef.current.has(`${chart.id}:${s.label}`)
            );
            return (
              <LiveChart
                key={chart.id}
                title={chart.title}
                subtitle={chart.id === "ankle" ? ankleSubtitle : null}
                unit={chart.unit}
                rightUnit={chart.rightUnit}
                series={series}
                data={[ring.t, ...series.map((s) => ring.values[`${chart.id}:${s.label}`])]}
              />
            );
          })}
        </div>
      </div>
      <AssistPanel latest={latest} />
    </div>
  );
}
