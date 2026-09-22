import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import uPlot from "uplot";
import { getRun, getRunData } from "../lib/api.js";

function StaticChart({ title, t, series, colors, unit }) {
  const containerRef = React.useRef(null);
  const plotRef = React.useRef(null);

  useEffect(() => {
    if (!containerRef.current || t.length === 0) return;
    const data = [t, ...series.map((s) => s.data)];

    if (plotRef.current) {
      plotRef.current.destroy();
    }
    plotRef.current = new uPlot(
      {
        width: containerRef.current.clientWidth,
        height: 260,
        series: [{}, ...series.map((s, i) => ({ label: s.label, stroke: colors[i], width: 1.5 }))],
        axes: [
          { stroke: "#8b93a7", grid: { stroke: "#232838" } },
          { stroke: "#8b93a7", grid: { stroke: "#232838" }, label: unit },
        ],
        legend: { show: true },
      },
      data,
      containerRef.current
    );

    return () => {
      if (plotRef.current) {
        plotRef.current.destroy();
        plotRef.current = null;
      }
    };
  }, [t, series, colors, unit]);

  return (
    <div className="card">
      <h3>{title}</h3>
      <div ref={containerRef} />
    </div>
  );
}

export default function RunDetailPage() {
  const { runId } = useParams();
  const [meta, setMeta] = useState(null);
  const [data, setData] = useState(null);

  useEffect(() => {
    getRun(runId).then(setMeta);
    getRunData(runId).then(setData);
  }, [runId]);

  if (!meta || !data) {
    return <p style={{ color: "var(--text-dim)" }}>Loading run {runId}...</p>;
  }

  if (data.error || !data.rows || data.rows.length === 0) {
    return (
      <div>
        <Link className="back-link" to="/runs">
          &larr; Back to runs
        </Link>
        <p style={{ color: "var(--text-dim)" }}>No data recorded for this run.</p>
      </div>
    );
  }

  const rows = data.rows;
  const t0 = rows[0].t;
  const t = rows.map((r) => r.t - t0);
  const col = (name) => rows.map((r) => r[name] ?? null);

  return (
    <div>
      <Link className="back-link" to="/runs">
        &larr; Back to runs
      </Link>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3>Run {runId}</h3>
        <div className="grid">
          <div>
            <div className="stat">{data.rows.length}</div>
            <div className="stat-label">samples (downsampled for display)</div>
          </div>
          <div>
            <div className="stat">{meta.status}</div>
            <div className="stat-label">status</div>
          </div>
          <div>
            <div className="stat">{meta.meta?.assistance_level ?? "-"}</div>
            <div className="stat-label">assistance level</div>
          </div>
          <div>
            <div className="stat">{meta.meta?.peak_torque ?? "-"} Nm</div>
            <div className="stat-label">peak torque</div>
          </div>
        </div>
      </div>

      {meta.model && (
        <div className="card" style={{ marginBottom: 16 }}>
          <h3>Model</h3>
          <div className="grid">
            <div>
              <div className="stat" style={{ fontSize: 16 }}>
                {meta.model.name || "-"}
              </div>
              <div className="stat-label">model name</div>
            </div>
            <div>
              <div className="stat" style={{ fontSize: 16, fontFamily: "monospace" }}>
                {meta.architecture_hash || "-"}
              </div>
              <div className="stat-label">architecture fingerprint</div>
            </div>
            <div>
              {meta.model.wandb_run_url ? (
                <a
                  href={meta.model.wandb_run_url}
                  target="_blank"
                  rel="noreferrer"
                  style={{ color: "var(--accent)" }}
                >
                  Open in W&amp;B &rarr;
                </a>
              ) : (
                <span style={{ color: "var(--text-dim)" }}>no W&amp;B run linked</span>
              )}
              <div className="stat-label">training history</div>
            </div>
            <div>
              <div className="stat" style={{ fontSize: 13, fontFamily: "monospace" }}>
                {meta.model.checkpoint_ref || "-"}
              </div>
              <div className="stat-label">checkpoint</div>
            </div>
          </div>
          {meta.model.architecture && Object.keys(meta.model.architecture).length > 0 && (
            <>
              <h3 style={{ marginTop: 20 }}>Architecture / Config</h3>
              <table>
                <tbody>
                  {Object.entries(meta.model.architecture).map(([k, v]) => (
                    <tr key={k}>
                      <td style={{ color: "var(--text-dim)" }}>{k}</td>
                      <td>{JSON.stringify(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      )}

      <div className="grid">
        <StaticChart
          title="Ankle Angle"
          unit="deg"
          t={t}
          colors={["#5b9dff"]}
          series={[{ label: "ankle_angle", data: col("ankle_angle") }]}
        />
        <StaticChart
          title="Ankle Velocity"
          unit="deg/s"
          t={t}
          colors={["#ffb238"]}
          series={[{ label: "ankle_velocity", data: col("ankle_velocity") }]}
        />
        <StaticChart
          title="Torque Command"
          unit="Nm"
          t={t}
          colors={["#ff5c5c"]}
          series={[{ label: "torque_cmd", data: col("torque_cmd") }]}
        />
        <StaticChart
          title="FSR (Heel / Toe)"
          unit="raw"
          t={t}
          colors={["#3ddc84", "#5b9dff"]}
          series={[
            { label: "heel_fsr", data: col("heel_fsr") },
            { label: "toe_fsr", data: col("toe_fsr") },
          ]}
        />
      </div>
    </div>
  );
}
