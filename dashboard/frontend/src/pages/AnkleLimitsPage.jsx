import React, { useEffect, useState } from "react";
import { getAnkleLimits, setAnkleStanding, resetAnkleLimits } from "../lib/api.js";

const POLL_MS = 100;
const SWAP_KEY = "ankleLimits.swap";

// Default: increasing raw encoder angle is dorsiflexion (ankle_encoder_sign = +1 in
// exo_frame_calibration.json and the deploy config). "Swap directions" flips that.
function readSwap() {
  try {
    return localStorage.getItem(SWAP_KEY) === "1";
  } catch {
    return false;
  }
}

function writeSwap(value) {
  try {
    localStorage.setItem(SWAP_KEY, value ? "1" : "0");
  } catch {
    // remembered for this page only
  }
}

const fmt = (deg) => (deg == null ? "—" : `${deg.toFixed(1)}°`);
const fmtSigned = (deg) => (deg == null ? "—" : `${deg >= 0 ? "+" : ""}${deg.toFixed(1)}°`);

function LimitBox({ title, side, active }) {
  const settled = active && side?.settled;
  return (
    <div className={`limit-box ${settled ? "settled" : ""}`}>
      <div className="limit-title">{title}</div>
      <div className="limit-value">{active ? fmt(side?.max_deg ?? 0) : "—"}</div>
      <div className="limit-state">
        {!active ? "Set standing first" : settled ? "SETTLED" : "Measuring…"}
      </div>
    </div>
  );
}

export default function AnkleLimitsPage() {
  const [state, setState] = useState(null);
  const [swap, setSwap] = useState(readSwap);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let active = true;
    let timer = null;

    async function poll() {
      let next = null;
      try {
        next = await getAnkleLimits();
      } catch {
        next = null;
      }
      if (!active) return;
      setState(next);
      timer = setTimeout(poll, POLL_MS);
    }

    poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, []);

  async function act(fn) {
    setBusy(true);
    setError(null);
    try {
      setState(await fn());
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  function toggleSwap() {
    setSwap((s) => {
      writeSwap(!s);
      return !s;
    });
  }

  const ok = state?.ok === true;
  const standingSet = state?.standing_deg != null;
  const dorsi = swap ? state?.negative : state?.positive;
  const plantar = swap ? state?.positive : state?.negative;
  // Offset shown dorsiflexion-positive, whichever way the encoder turns.
  const offset = state?.offset_deg == null ? null : swap ? -state.offset_deg : state.offset_deg;

  return (
    <div className="ankle-limits">
      <div className={`encoder-box ${ok ? "" : "fail"}`}>
        <div className="limit-title">Current encoder value</div>
        <div className="encoder-value">{ok ? fmt(state.raw_deg) : "—"}</div>
        {ok ? (
          <div className="dim">
            {standingSet
              ? `Standing ${fmt(state.standing_deg)} · offset ${fmtSigned(offset)} (dorsi +)`
              : "Standing not set"}
          </div>
        ) : (
          <div className="health-reason">
            {state?.reason ?? "No answer from the dashboard backend"}
          </div>
        )}
      </div>

      <div className="ankle-controls">
        <button onClick={() => act(setAnkleStanding)} disabled={busy || !ok}>
          {standingSet ? "Re-set Standing" : "Set Standing"}
        </button>
        <button onClick={() => act(resetAnkleLimits)} disabled={busy}>
          Reset
        </button>
        <label className="swap-toggle">
          <input type="checkbox" checked={swap} onChange={toggleSwap} />
          Swap directions
        </label>
      </div>
      {error && <div className="health-reason bad-text">{error}</div>}

      <div className="limit-grid">
        <LimitBox title="Max Dorsiflexion" side={dorsi} active={standingSet} />
        <LimitBox title="Max Plantarflexion" side={plantar} active={standingSet} />
      </div>
    </div>
  );
}
