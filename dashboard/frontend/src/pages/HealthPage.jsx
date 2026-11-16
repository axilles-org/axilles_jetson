import React, { useEffect, useState } from "react";
import { getHealth } from "../lib/api.js";

const POLL_MS = 1000;

const DEVICES = [
  { id: "imu_foot", name: "Foot IMU", part: "BNO085, I2C 0x4A" },
  { id: "imu_shank", name: "Shank IMU", part: "BNO085, I2C 0x4B" },
  { id: "encoder", name: "Ankle encoder", part: "AS5600, I2C 0x36" },
  { id: "adc", name: "FSR ADC", part: "ADS1115, I2C 0x48" },
  { id: "motor", name: "Motor", part: "AK80-9, CAN 0x68" },
];

function HealthBox({ device, result }) {
  const ok = result?.ok === true;
  const reason = result?.reason ?? "No answer from the dashboard backend";
  return (
    <div className={`health-box ${ok ? "ok" : "fail"}`}>
      <div className="health-name">{device.name}</div>
      <div className="dim">{device.part}</div>
      <div className="health-state">{ok ? "CONNECTED" : "NOT CONNECTED"}</div>
      {!ok && <div className="health-reason">{reason}</div>}
    </div>
  );
}

export default function HealthPage() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let active = true;
    let timer = null;

    async function poll() {
      let next = null;
      try {
        next = await getHealth();
      } catch {
        next = null; // every box shows red if the backend does not answer
      }
      if (!active) return;
      setHealth(next);
      timer = setTimeout(poll, POLL_MS);
    }

    poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, []);

  return (
    <div className="health-grid">
      {DEVICES.map((device) => (
        <HealthBox key={device.id} device={device} result={health?.[device.id]} />
      ))}
    </div>
  );
}
