import React, { useEffect, useMemo, useState } from "react";
import LineChart from "../components/LineChart.jsx";
import {
  getAblation,
  getSubjectStrides,
  getTransformVerification,
  listAblations,
  listTransformVerifications,
} from "../lib/api.js";

const UNITS = { accel: "m/s²", gyro: "rad/s" };
const PHASE = Array.from({ length: 100 }, (_, i) => i);

const TRANSFORM_METRICS = [
  { key: "rmse", label: "RMSE", lowerIsBetter: true },
  { key: "pearson_r", label: "Pearson r", lowerIsBetter: false },
  { key: "peak_timing_error_pct", label: "Peak timing err (%)", lowerIsBetter: true },
  { key: "dtw_distance", label: "DTW", lowerIsBetter: true },
];

function fmt(v, digits = 3) {
  return v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(digits);
}

function changeClass(before, after, lowerIsBetter) {
  if (before === after) return "";
  return (after < before) === lowerIsBetter ? "good" : "bad";
}

function Select({ label, value, options, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

function TransformSummary({ rows }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Channel</th>
          {TRANSFORM_METRICS.map((m) => (
            <th key={m.key}>{m.label} before → after</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={`${r.segment}-${r.sensor}`}>
            <td>
              {r.segment} {r.sensor}
            </td>
            {TRANSFORM_METRICS.map(({ key, lowerIsBetter }) => {
              const before = r[`${key}_before`];
              const after = r[`${key}_after`];
              return (
                <td key={key}>
                  {fmt(before, 2)} →{" "}
                  <span className={changeClass(before, after, lowerIsBetter)}>{fmt(after, 2)}</span>
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function StrideOverlay({ verificationId, subjects }) {
  const [subject, setSubject] = useState(subjects[0]);
  const [segment, setSegment] = useState("foot");
  const [sensor, setSensor] = useState("gyro");
  const [axis, setAxis] = useState("Z");
  const [strides, setStrides] = useState(null);

  useEffect(() => {
    getSubjectStrides(verificationId, subject).then(setStrides).catch(() => setStrides(null));
  }, [verificationId, subject]);

  const series = useMemo(() => {
    const s = strides?.[segment]?.[sensor];
    if (!s) return [];
    const col = (arr) => arr.map((row) => row["XYZ".indexOf(axis)]);
    return [
      { label: "exo", data: col(s.exo), color: "#5b9dff" },
      { label: "GaTech original", data: col(s.before), color: "#8b93a7", dash: [6, 4] },
      { label: "GaTech transformed", data: col(s.after), color: "#ffb238" },
    ];
  }, [strides, segment, sensor, axis]);

  return (
    <>
      <div className="controls">
        <Select label="Subject" value={subject} options={subjects} onChange={setSubject} />
        <Select label="Segment" value={segment} options={["foot", "shank"]} onChange={setSegment} />
        <Select label="Sensor" value={sensor} options={["accel", "gyro"]} onChange={setSensor} />
        <Select label="Axis" value={axis} options={["X", "Y", "Z"]} onChange={setAxis} />
      </div>
      {series.length > 0 ? (
        <LineChart x={PHASE} series={series} xLabel="% gait cycle" yLabel={UNITS[sensor]} />
      ) : (
        <p className="dim">No strides for this selection.</p>
      )}
    </>
  );
}

function TransformSection() {
  const [ids, setIds] = useState([]);
  const [id, setId] = useState(null);
  const [data, setData] = useState(null);

  useEffect(() => {
    listTransformVerifications().then((list) => {
      setIds(list.map((v) => v.id));
      if (list.length) setId(list[0].id);
    });
  }, []);

  useEffect(() => {
    if (id) getTransformVerification(id).then(setData);
  }, [id]);

  if (!ids.length) {
    return (
      <div className="card">
        <h3>Transform verification</h3>
        <p className="dim">
          No results yet. Run <code>Sensor_transformation_script/verify_transform.py</code>.
        </p>
      </div>
    );
  }

  const outliers = data ? Object.entries(data.meta.outliers) : [];
  return (
    <div className="card">
      <h3>Transform verification — GaTech mean stride vs exo mean stride</h3>
      <div className="controls">
        <Select label="Fit" value={id ?? ""} options={ids} onChange={setId} />
        {data && (
          <span className="dim">
            exo log <code>{data.meta.exo_log}</code> · {data.meta.subjects.length} subjects ·
            outliers{" "}
            {outliers.map(([seg, subs]) => `${seg}: ${subs.join(", ") || "none"}`).join(" · ")}
          </span>
        )}
      </div>
      {data && (
        <>
          <p className="dim">Median over subjects of the X/Y/Z-axis mean. Green = moved toward the exo signal.</p>
          <TransformSummary rows={data.summary} />
          <StrideOverlay key={id} verificationId={id} subjects={data.meta.subjects} />
        </>
      )}
    </div>
  );
}

function AblationSection() {
  const [ids, setIds] = useState([]);
  const [id, setId] = useState(null);
  const [results, setResults] = useState([]);

  useEffect(() => {
    listAblations().then((list) => {
      setIds(list.map((a) => a.id));
      if (list.length) setId(list[0].id);
    });
  }, []);

  useEffect(() => {
    if (id) getAblation(id).then(setResults);
  }, [id]);

  const rmse = (r) => r.test_streaming.rmse_nm_per_kg;
  const sorted = [...results].sort((a, b) => rmse(a) - rmse(b));
  const baseline = results.find((r) => r.variant === "original");
  const maxRmse = Math.max(...results.map(rmse), 0);

  return (
    <div className="card">
      <h3>Transform ablation — held-out test subjects</h3>
      {!ids.length ? (
        <p className="dim">
          No results yet. Run <code>ML_model/scripts/transform_ablation.py</code>.
        </p>
      ) : (
        <>
          <div className="controls">
            <Select label="Run" value={id ?? ""} options={ids} onChange={setId} />
            <span className="dim">{results.length} variants finished</span>
          </div>
          <table>
            <thead>
              <tr>
                <th>Variant</th>
                <th>Transformed</th>
                <th>Dropped</th>
                <th>RMSE (N·m/kg)</th>
                <th>MAE</th>
                <th>R²</th>
                <th>vs original</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => {
                const delta = baseline ? (100 * (rmse(r) - rmse(baseline))) / rmse(baseline) : null;
                return (
                  <tr key={r.variant}>
                    <td>{r.variant}</td>
                    <td>{r.tx_groups.join(", ") || "—"}</td>
                    <td>{r.dropped_groups.join(", ") || "—"}</td>
                    <td>
                      <div className="bar-cell">
                        <div className="bar" style={{ width: `${(100 * rmse(r)) / maxRmse}%` }} />
                        <span>{fmt(rmse(r), 4)}</span>
                      </div>
                    </td>
                    <td>{fmt(r.test_streaming.mae_nm_per_kg, 4)}</td>
                    <td>{fmt(r.test_streaming.r2)}</td>
                    <td className={delta === null || r === baseline ? "" : delta < 0 ? "good" : "bad"}>
                      {delta === null || r === baseline ? "—" : `${delta > 0 ? "+" : ""}${delta.toFixed(1)}%`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

export default function ExperimentsPage() {
  return (
    <div className="stack">
      <TransformSection />
      <AblationSection />
    </div>
  );
}
