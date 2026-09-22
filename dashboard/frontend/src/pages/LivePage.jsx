import React, { useEffect, useRef, useState } from "react";
import LiveChart from "../components/LiveChart.jsx";
import { connectLiveSocket } from "../lib/api.js";

const WINDOW_SIZE = 600; // ~4-6s of history depending on control rate

// Different controllers log different field names (TBE fixed-profile vs the
// ML/TCN controller). Rather than hardcode one schema, group whatever fields
// actually show up in the live stream into sensible chart panels by name.
const CHART_GROUPS = [
  {
    title: "Ankle Angle / Encoder",
    unit: "deg",
    colors: ["#5b9dff"],
    fields: ["ankle_angle", "ankle_encoder_deg"],
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
    colors: ["#3ddc84", "#ffb238"],
    fields: ["stance", "ramp", "session_ramp"],
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

  useEffect(() => {
    const ws = connectLiveSocket((msg) => {
      setConnected(true);
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
    <div>
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
  );
}
