import React, { useEffect, useRef, useState } from "react";
import LiveChart from "../components/LiveChart.jsx";
import { connectLiveSocket, getAssist, getJobStatus, setAssist, stopJob } from "../lib/api.js";

const SCALE_STEP = 0.01;
const SCALE_MIN = 0;
const SCALE_MAX = 0.3; // jetson_deploy.py --max-scale default; the controller caps it too

const clampScale = (x) => Math.min(SCALE_MAX, Math.max(SCALE_MIN, Math.round(x * 100) / 100));

function AssistPanel({ latest }) {
  const [job, setJob] = useState(null);
  // What the backend is currently heartbeating to the controller
  const [desired, setDesired] = useState({ assist: false, scale: null });
  // Value chosen with the arrows, sent only when "Set" is pressed
  const [pending, setPending] = useState(null);
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
    const id = setInterval(poll, 1000);
    return () => clearInterval(id);
  }, []);

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

  async function send(assist, scale) {
    setError(null);
    setBusy(true);
    try {
      // null leaves the backend's scale unchanged
      setDesired(await setAssist(assist, scale == null ? null : Number(scale)));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const unsaved = pending !== null && desired.scale !== null && pending !== clampScale(desired.scale);

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

const WINDOW_SIZE = 600; // ~4-6s of history depending on control rate

// Different controllers log different field names (TBE fixed-profile vs the
// ML/TCN controller). Rather than hardcode one schema, group whatever fields
// actually show up in the live stream into sensible chart panels by name.
const CHART_GROUPS = [
  {
    title: "Ankle Angle / Encoder",
    unit: "deg",
    colors: ["#5b9dff", "#3ddc84"],
    fields: ["ankle_angle", "ankle_encoder_deg", "ankle_angle_deg"],
  },
  {
    title: "Ankle Velocity",
    unit: "deg/s",
    colors: ["#ffb238"],
    fields: ["ankle_velocity"],
  },
  {
    title: "Torque",
    unit: "Nm",
    colors: ["#ff5c5c", "#ffb238", "#5b9dff"],
    fields: ["torque_cmd", "assist_cmd_nm", "sent_torque_nm"],
  },
  {
    title: "ML Prediction",
    unit: "Nm/kg",
    colors: ["#c792ea"],
    fields: ["predicted_nm_per_kg"],
  },
  {
    title: "FSR (Heel / Toe)",
    unit: "raw",
    colors: ["#3ddc84", "#5b9dff"],
    fields: ["heel_fsr", "toe_fsr", "heel_fsr_raw", "toe_fsr_raw"],
  },
  {
    title: "Gait State",
    unit: "",
    colors: ["#3ddc84", "#ffb238", "#5b9dff", "#ff5c5c", "#c792ea"],
    fields: ["stance", "ramp", "session_ramp", "assist_on", "calibrated"],
  },
];

function makeRing() {
  return { t: [], values: {} };
}

export default function LivePage() {
  const [connected, setConnected] = useState(false);
  const [activeRunId, setActiveRunId] = useState(null);
  const [controllerLabel, setControllerLabel] = useState(null);
  const [tick, setTick] = useState(0);
  const ringRef = useRef(makeRing());
  const startTimeRef = useRef(null);
  const seenFieldsRef = useRef(new Set());
  const latestRef = useRef({});

  useEffect(() => {
    const ws = connectLiveSocket((msg) => {
      setConnected(true);
      latestRef.current = msg;
      setActiveRunId(msg.run_id);
      if (msg.controller) setControllerLabel(msg.controller);

      if (startTimeRef.current === null) startTimeRef.current = msg.t;
      const relT = msg.t - startTimeRef.current;

      const ring = ringRef.current;
      ring.t.push(relT);

      for (const [key, value] of Object.entries(msg)) {
        if (typeof value !== "number") continue;
        if (key === "t") continue;
        seenFieldsRef.current.add(key);
        if (!ring.values[key]) {
          // backfill so this series stays aligned with ring.t in length
          ring.values[key] = new Array(ring.t.length - 1).fill(null);
        }
        ring.values[key].push(value);
      }
      // any field not present in this sample stays aligned via null padding
      for (const key of Object.keys(ring.values)) {
        if (!(key in msg)) ring.values[key].push(null);
      }

      if (ring.t.length > WINDOW_SIZE) {
        ring.t.shift();
        Object.keys(ring.values).forEach((k) => ring.values[k].shift());
      }

      setTick((n) => n + 1);
    });

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);

    return () => ws.close();
  }, []);

  const ring = ringRef.current;
  const seen = seenFieldsRef.current;

  const activeGroups = CHART_GROUPS.map((group) => {
    const presentFields = group.fields.filter((f) => seen.has(f));
    return { ...group, presentFields };
  }).filter((g) => g.presentFields.length > 0);

  return (
    <div className="live-layout">
      <div className="live-main">
        <div className="status-row">
          <span className={`status-dot ${connected ? "connected" : ""}`} />
          <span>{connected ? "Connected to live relay" : "Waiting for connection..."}</span>
          {activeRunId && <span style={{ color: "var(--text-dim)" }}>· run {activeRunId}</span>}
          {controllerLabel && <span style={{ color: "var(--text-dim)" }}>· {controllerLabel}</span>}
        </div>

        {!connected && (
          <p style={{ color: "var(--text-dim)" }}>
            No live data yet. Start a controller (TBE_controller/main.py or
            ML_model/scripts/jetson_deploy.py) or the fake-data simulator to see
            charts update here.
          </p>
        )}

        <div className="grid">
          {activeGroups.map((group) => (
            <LiveChart
              key={group.title}
              title={group.title}
              unit={group.unit}
              colors={group.colors}
              seriesData={{
                t: ring.t,
                values: Object.fromEntries(group.presentFields.map((f) => [f, ring.values[f] || []])),
              }}
            />
          ))}
        </div>
      </div>
      <AssistPanel latest={latestRef.current} />
    </div>
  );
}
