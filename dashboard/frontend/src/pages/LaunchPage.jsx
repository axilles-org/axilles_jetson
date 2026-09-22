import React, { useEffect, useRef, useState } from "react";
import DropZone from "../components/DropZone.jsx";
import {
  listModels,
  listSamples,
  uploadReplayCsv,
  launchJob,
  stopJob,
  getJobStatus,
} from "../lib/api.js";

const POLL_MS = 1000;

function fmtSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

export default function LaunchPage() {
  const [models, setModels] = useState([]);
  const [samples, setSamples] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState("tbe");
  const [mode, setMode] = useState("mock"); // mock | dry-run (armed is never offered here)
  const [mass, setMass] = useState(72);
  const [duration, setDuration] = useState(30);
  const [backend, setBackend] = useState("onnx");
  const [assistScale, setAssistScale] = useState(0.1);
  const [replayFile, setReplayFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [job, setJob] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => {
    listModels().then(setModels);
    listSamples().then(setSamples);
    getJobStatus().then(setJob);
  }, []);

  useEffect(() => {
    if (job && job.status === "running") {
      pollRef.current = setInterval(async () => {
        const s = await getJobStatus();
        setJob(s);
      }, POLL_MS);
    } else if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [job?.status]);

  const selectedModel = models.find((m) => m.id === selectedModelId);
  const isMl = selectedModel?.controller === "ml";
  const isRunning = job?.status === "running";

  async function handleReplayDrop(file) {
    setError(null);
    setUploading(true);
    try {
      const result = await uploadReplayCsv(file);
      setReplayFile(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleLaunch() {
    setError(null);
    try {
      const controller = !isMl ? "tbe" : mode === "mock" ? "ml-mock" : "ml-dry-run";
      const body = {
        controller,
        run_dir: isMl ? selectedModel.run_dir : null,
        mass: isMl ? Number(mass) : null,
        replay_path: replayFile?.path ?? null,
        duration: Number(duration),
        backend: isMl ? backend : null,
        assist_scale: isMl ? Number(assistScale) : null,
      };
      const result = await launchJob(body);
      setJob(result);
    } catch (e) {
      setError(e.message);
    }
  }

  async function handleStop() {
    setError(null);
    try {
      const result = await stopJob();
      setJob(result);
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <div>
      <div className="grid">
        <div className="card">
          <h3>Model</h3>
          <select
            value={selectedModelId}
            onChange={(e) => setSelectedModelId(e.target.value)}
            disabled={isRunning}
            style={{ width: "100%", padding: 8, marginBottom: 12 }}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} {m.kind !== "fixed-profile" ? `(${m.kind})` : ""}
              </option>
            ))}
          </select>

          {selectedModel && selectedModel.kind === "tcn" && (
            <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
              <div>channels: {JSON.stringify(selectedModel.model_arch?.num_channels)}</div>
              <div>test RMSE: {selectedModel.test_metrics?.rmse_nm_per_kg?.toFixed(4)} Nm/kg</div>
              <div>
                backends available: {selectedModel.backends?.join(", ") || "none found"}
              </div>
              {selectedModel.wandb_run_url ? (
                <a href={selectedModel.wandb_run_url} target="_blank" rel="noreferrer">
                  W&amp;B run &rarr;
                </a>
              ) : (
                <div>no W&amp;B run linked</div>
              )}
            </div>
          )}

          {isMl && (
            <>
              <h3 style={{ marginTop: 16 }}>Mode</h3>
              <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
                <label>
                  <input
                    type="radio"
                    checked={mode === "mock"}
                    onChange={() => setMode("mock")}
                    disabled={isRunning}
                  />{" "}
                  mock (no motor code path at all)
                </label>
                <label>
                  <input
                    type="radio"
                    checked={mode === "dry-run"}
                    onChange={() => setMode("dry-run")}
                    disabled={isRunning}
                  />{" "}
                  dry-run (full path, torque never sent)
                </label>
              </div>
              <p style={{ fontSize: 12, color: "var(--text-dim)" }}>
                Armed (real torque) runs are CLI-only by design — not available from
                this page.
              </p>
            </>
          )}
        </div>

        <div className="card">
          <h3>Parameters</h3>
          {isMl && (
            <>
              <label style={{ display: "block", marginBottom: 8, fontSize: 13 }}>
                Subject mass (kg)
                <input
                  type="number"
                  value={mass}
                  onChange={(e) => setMass(e.target.value)}
                  disabled={isRunning}
                  style={{ width: "100%", padding: 6, marginTop: 4 }}
                />
              </label>
              <label style={{ display: "block", marginBottom: 8, fontSize: 13 }}>
                Backend
                <select
                  value={backend}
                  onChange={(e) => setBackend(e.target.value)}
                  disabled={isRunning}
                  style={{ width: "100%", padding: 6, marginTop: 4 }}
                >
                  {(selectedModel?.backends || ["onnx", "jit"]).map((b) => (
                    <option key={b} value={b}>
                      {b}
                    </option>
                  ))}
                </select>
              </label>
              <label style={{ display: "block", marginBottom: 8, fontSize: 13 }}>
                Assistance scale
                <input
                  type="number"
                  step="0.05"
                  value={assistScale}
                  onChange={(e) => setAssistScale(e.target.value)}
                  disabled={isRunning}
                  style={{ width: "100%", padding: 6, marginTop: 4 }}
                />
              </label>
            </>
          )}
          <label style={{ display: "block", marginBottom: 8, fontSize: 13 }}>
            Duration (s)
            <input
              type="number"
              value={duration}
              onChange={(e) => setDuration(e.target.value)}
              disabled={isRunning}
              style={{ width: "100%", padding: 6, marginTop: 4 }}
            />
          </label>

          {isMl && (
            <>
              <h3 style={{ marginTop: 16 }}>Replay CSV (optional)</h3>

              {samples.length > 0 && (
                <div style={{ marginBottom: 10 }}>
                  <select
                    value={replayFile?.isSample ? replayFile.path : ""}
                    onChange={(e) => {
                      const s = samples.find((x) => x.path === e.target.value);
                      if (s) setReplayFile({ path: s.path, filename: s.filename, isSample: true });
                    }}
                    disabled={isRunning}
                    style={{ width: "100%", padding: 6, marginBottom: 6 }}
                  >
                    <option value="">Use a bundled sample...</option>
                    {samples.map((s) => (
                      <option key={s.path} value={s.path}>
                        {s.filename} ({fmtSize(s.size_bytes)})
                      </option>
                    ))}
                  </select>
                  <div style={{ fontSize: 12, color: "var(--text-dim)" }}>or drop your own below</div>
                </div>
              )}

              <DropZone
                accept=".csv"
                label={
                  uploading
                    ? "Uploading..."
                    : replayFile && !replayFile.isSample
                    ? `${replayFile.filename} (uploaded)`
                    : "Drop a data_collection_*.csv here, or use live sensors if omitted"
                }
                onFile={handleReplayDrop}
                disabled={isRunning || uploading}
              />
              {replayFile && !isRunning && (
                <button style={{ marginTop: 8 }} onClick={() => setReplayFile(null)}>
                  Clear ({replayFile.filename})
                </button>
              )}
            </>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="status-row" style={{ justifyContent: "space-between" }}>
          <div className="status-row">
            <span className={`status-dot ${isRunning ? "connected" : ""}`} />
            <span>
              {isRunning
                ? `Running (job ${job.job_id})`
                : job
                ? `Last job: ${job.status}`
                : "No job running"}
            </span>
            {job?.run_id && (
              <span style={{ color: "var(--text-dim)" }}>· run {job.run_id}</span>
            )}
          </div>
          <div>
            {!isRunning ? (
              <button onClick={handleLaunch}>Launch</button>
            ) : (
              <button onClick={handleStop}>Stop</button>
            )}
          </div>
        </div>

        {error && <p style={{ color: "var(--bad)" }}>{error}</p>}

        {job && (
          <pre
            style={{
              background: "#080a0e",
              padding: 12,
              borderRadius: 8,
              maxHeight: 320,
              overflowY: "auto",
              fontSize: 12,
              marginTop: 12,
            }}
          >
            {(job.log_tail || []).join("\n") || "(no output yet)"}
          </pre>
        )}
      </div>
    </div>
  );
}
