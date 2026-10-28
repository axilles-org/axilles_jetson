import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listRuns, deleteRun, diffRuns, importRun, exportRunUrl } from "../lib/api.js";
import RunDiff from "../components/RunDiff.jsx";
import DropZone from "../components/DropZone.jsx";

function fmtTime(epochSeconds) {
  if (!epochSeconds) return "-";
  return new Date(epochSeconds * 1000).toLocaleString();
}

function fmtDuration(started, updated) {
  if (!started || !updated) return "-";
  const s = updated - started;
  return `${s.toFixed(1)}s`;
}

const STATUS_OPTIONS = ["all", "completed", "running", "failed", "crashed", "stopped", "imported"];

export default function RunsPage() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState([]);
  const [diff, setDiff] = useState(null);
  const [importError, setImportError] = useState(null);
  const [importing, setImporting] = useState(false);

  const [modelFilter, setModelFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [searchText, setSearchText] = useState("");

  const navigate = useNavigate();

  async function refresh() {
    setLoading(true);
    const data = await listRuns();
    setRuns(data);
    setLoading(false);
  }

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, []);

  async function handleDelete(e, runId) {
    e.stopPropagation();
    if (!confirm(`Delete run ${runId}? This cannot be undone.`)) return;
    await deleteRun(runId);
    refresh();
  }

  async function handleImport(file) {
    setImportError(null);
    setImporting(true);
    try {
      await importRun(file);
      await refresh();
    } catch (e) {
      setImportError(e.message);
    } finally {
      setImporting(false);
    }
  }

  const modelNames = useMemo(() => {
    const names = new Set(runs.map((r) => r.model?.name).filter(Boolean));
    return ["all", ...Array.from(names)];
  }, [runs]);

  const filteredRuns = useMemo(() => {
    return runs.filter((r) => {
      if (modelFilter !== "all" && r.model?.name !== modelFilter) return false;
      if (statusFilter !== "all" && r.status !== statusFilter) return false;
      if (searchText.trim()) {
        const haystack = `${r.run_id} ${r.notes || ""} ${r.architecture_hash || ""}`.toLowerCase();
        if (!haystack.includes(searchText.trim().toLowerCase())) return false;
      }
      return true;
    });
  }, [runs, modelFilter, statusFilter, searchText]);

  function toggleSelect(e, runId) {
    e.stopPropagation();
    setDiff(null);
    setSelected((prev) => {
      if (prev.includes(runId)) return prev.filter((id) => id !== runId);
      if (prev.length >= 2) return [prev[1], runId]; // keep last 2 selections
      return [...prev, runId];
    });
  }

  async function handleCompare() {
    if (selected.length !== 2) return;
    const result = await diffRuns(selected[0], selected[1]);
    setDiff(result);
  }

  return (
    <div>
      <div className="card" style={{ marginBottom: 16 }}>
        <h3>Import a run</h3>
        <DropZone
          accept=".zip"
          label={importing ? "Importing..." : "Drop an exported run .zip here, or click to browse"}
          onFile={handleImport}
          disabled={importing}
        />
        {importError && <p style={{ color: "var(--bad)" }}>{importError}</p>}
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="status-row" style={{ justifyContent: "space-between", flexWrap: "wrap", gap: 10 }}>
          <h3 style={{ margin: 0 }}>
            Trial Runs ({filteredRuns.length}
            {filteredRuns.length !== runs.length ? ` of ${runs.length}` : ""})
          </h3>
          <div>
            <span style={{ color: "var(--text-dim)", fontSize: 13, marginRight: 10 }}>
              {selected.length === 0 && "select two runs to compare"}
              {selected.length === 1 && "select one more run to compare"}
              {selected.length === 2 && "ready to compare"}
            </span>
            <button disabled={selected.length !== 2} onClick={handleCompare}>
              Compare selected
            </button>
          </div>
        </div>

        <div className="status-row" style={{ gap: 10, flexWrap: "wrap" }}>
          <select value={modelFilter} onChange={(e) => setModelFilter(e.target.value)}>
            {modelNames.map((n) => (
              <option key={n} value={n}>
                {n === "all" ? "All models" : n}
              </option>
            ))}
          </select>
          <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            {STATUS_OPTIONS.map((s) => (
              <option key={s} value={s}>
                {s === "all" ? "All statuses" : s}
              </option>
            ))}
          </select>
          <input
            type="text"
            placeholder="Search run ID, notes, hash..."
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            style={{ padding: 6, flex: 1, minWidth: 180 }}
          />
        </div>

        {loading && runs.length === 0 ? (
          <p style={{ color: "var(--text-dim)" }}>Loading...</p>
        ) : runs.length === 0 ? (
          <p style={{ color: "var(--text-dim)" }}>
            No runs yet. Runs appear here once a controller (or the fake-data
            simulator) has been started at least once, or after importing one.
          </p>
        ) : filteredRuns.length === 0 ? (
          <p style={{ color: "var(--text-dim)" }}>No runs match the current filters.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th></th>
                <th>Run ID</th>
                <th>Started</th>
                <th>Duration</th>
                <th>Samples</th>
                <th>Status</th>
                <th>Model</th>
                <th>Assistance</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {filteredRuns.map((r) => (
                <tr key={r.run_id} className="clickable" onClick={() => navigate(`/runs/${r.run_id}`)}>
                  <td onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={selected.includes(r.run_id)}
                      onChange={(e) => toggleSelect(e, r.run_id)}
                    />
                  </td>
                  <td>{r.run_id}</td>
                  <td>{fmtTime(r.started_at)}</td>
                  <td>{fmtDuration(r.started_at, r.updated_at)}</td>
                  <td>{r.sample_count}</td>
                  <td>
                    <span className={`badge ${r.status}`}>{r.status}</span>
                  </td>
                  <td>
                    {r.model?.name || "-"}
                    {r.architecture_hash && (
                      <span style={{ color: "var(--text-dim)", fontFamily: "monospace", fontSize: 11 }}>
                        {" "}
                        ({r.architecture_hash})
                      </span>
                    )}
                  </td>
                  <td>{r.meta?.assistance_level ?? "-"}</td>
                  <td onClick={(e) => e.stopPropagation()}>
                    <a href={exportRunUrl(r.run_id)} download style={{ marginRight: 10 }}>
                      Export
                    </a>
                    <button onClick={(e) => handleDelete(e, r.run_id)}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {diff && <RunDiff diff={diff} />}
    </div>
  );
}
