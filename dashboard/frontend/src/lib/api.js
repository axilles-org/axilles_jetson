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

export async function getAssist() {
  const res = await fetch(`${API_BASE}/launch/assist`);
  return res.json();
}

export async function setAssist(assist, scale) {
  const res = await fetch(`${API_BASE}/launch/assist`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ assist, scale }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || "assist change failed");
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

async function getJson(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error((await res.json()).detail || `${path} failed`);
  return res.json();
}

export const listTransformVerifications = () => getJson("/experiments/transform");
export const getTransformVerification = (id) => getJson(`/experiments/transform/${id}`);
export const getSubjectStrides = (id, subject) =>
  getJson(`/experiments/transform/${id}/strides/${subject}`);
export const listAblations = () => getJson("/experiments/ablations");
export const getAblation = (id) => getJson(`/experiments/ablations/${id}`);
