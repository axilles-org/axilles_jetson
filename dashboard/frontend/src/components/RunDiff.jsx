import React from "react";

function DiffTable({ title, diff }) {
  const keys = Object.keys(diff);
  if (keys.length === 0) {
    return (
      <div style={{ marginBottom: 16 }}>
        <h3>{title}</h3>
        <p style={{ color: "var(--text-dim)", fontSize: 13 }}>No differences.</p>
      </div>
    );
  }
  return (
    <div style={{ marginBottom: 16 }}>
      <h3>{title}</h3>
      <table>
        <thead>
          <tr>
            <th>Field</th>
            <th>Run A</th>
            <th>Run B</th>
          </tr>
        </thead>
        <tbody>
          {keys.map((k) => (
            <tr key={k}>
              <td>{k}</td>
              <td style={{ color: "var(--warn)" }}>{JSON.stringify(diff[k].a)}</td>
              <td style={{ color: "var(--good)" }}>{JSON.stringify(diff[k].b)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function RunDiff({ diff }) {
  if (!diff) return null;
  if (diff.error) {
    return <p style={{ color: "var(--bad)" }}>{diff.error}</p>;
  }

  return (
    <div className="card">
      <div className="status-row">
        <span
          className="badge"
          style={{
            background: diff.architecture_changed ? "rgba(255,178,56,0.15)" : "rgba(61,220,132,0.15)",
            color: diff.architecture_changed ? "var(--warn)" : "var(--good)",
          }}
        >
          {diff.architecture_changed ? "Architecture changed" : "Same architecture"}
        </span>
        <span style={{ color: "var(--text-dim)", fontSize: 12, fontFamily: "monospace" }}>
          {diff.architecture_hash_a} &rarr; {diff.architecture_hash_b}
        </span>
      </div>
      <DiffTable title="Architecture / Model Config" diff={diff.architecture_diff} />
      <DiffTable title="Trial Metadata" diff={diff.meta_diff} />
    </div>
  );
}
