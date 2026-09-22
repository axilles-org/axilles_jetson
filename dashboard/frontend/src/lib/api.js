const API_BASE = "/api";

export async function listRuns() {
  const res = await fetch(`${API_BASE}/runs`);
  return res.json();
}

export async function getRun(runId) {
  const res = await fetch(`${API_BASE}/runs/${runId}`);
  return res.json();
}

export async function getRunData(runId, maxPoints = 5000) {
  const res = await fetch(`${API_BASE}/runs/${runId}/data?max_points=${maxPoints}`);
  return res.json();
}

export async function deleteRun(runId) {
  const res = await fetch(`${API_BASE}/runs/${runId}`, { method: "DELETE" });
  return res.json();
}

export async function diffRuns(runIdA, runIdB) {
  const res = await fetch(`${API_BASE}/runs/${runIdA}/diff/${runIdB}`);
  return res.json();
}

export function exportRunUrl(runId) {
  return `${API_BASE}/runs/${runId}/export`;
}

export async function importRun(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/runs/import`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail || "import failed");
  return res.json();
}

export async function listModels() {
  const res = await fetch(`${API_BASE}/models`);
  return res.json();
}

export async function listSamples() {
  const res = await fetch(`${API_BASE}/samples`);
  return res.json();
}

export async function uploadReplayCsv(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/uploads/replay`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail || "upload failed");
  return res.json();
}

export async function launchJob(body) {
  const res = await fetch(`${API_BASE}/launch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json()).detail || "launch failed");
  return res.json();
}

export async function stopJob() {
  const res = await fetch(`${API_BASE}/launch/stop`, { method: "POST" });
  if (!res.ok) throw new Error((await res.json()).detail || "stop failed");
  return res.json();
}

export async function getJobStatus() {
  const res = await fetch(`${API_BASE}/launch/status`);
  return res.json();
}

export function connectLiveSocket(onMessage) {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.host}/ws/live`);
  ws.onmessage = (evt) => {
    try {
      onMessage(JSON.parse(evt.data));
    } catch {
      // ignore malformed packet
    }
  };
  return ws;
}
