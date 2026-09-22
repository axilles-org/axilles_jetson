"""
Dashboard backend.

Two jobs:
  1. Live relay: listen on a local UDP socket for samples that RunLogger
     fires off during a trial, and fan them out to any connected browser
     over WebSocket. If no trial is running, this is just idle.
  2. History API: list past runs (from SQLite) and serve a run's full
     Parquet data for plotting/comparison in the browser.

Run with:
    uvicorn dashboard.backend.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import sqlite3
import uuid
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from . import process_manager

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RUNS_ROOT = DATA_ROOT / "runs"
DB_PATH = DATA_ROOT / "runs.db"
UPLOADS_ROOT = DATA_ROOT / "uploads"
SAMPLE_DATA_ROOT = Path(__file__).resolve().parent / "sample_data"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ML_MODEL_RUNS = REPO_ROOT / "ML_model" / "runs"

LIVE_RELAY_HOST = "127.0.0.1"
LIVE_RELAY_PORT = 47270

app = FastAPI(title="Axilles Exo Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lab-network tool; tighten if ever exposed beyond LAN
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------- #
# Live relay: UDP -> WebSocket fan-out
# ---------------------------------------------------------------------- #

class LiveHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def register(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def unregister(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, message: str) -> None:
        async with self._lock:
            dead = []
            for ws in self._clients:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)


hub = LiveHub()


async def udp_relay_task() -> None:
    """Non-blocking UDP listener that forwards RunLogger samples to WebSocket clients."""
    loop = asyncio.get_running_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    sock.bind((LIVE_RELAY_HOST, LIVE_RELAY_PORT))

    while True:
        try:
            data = await loop.sock_recv(sock, 65536)
            await hub.broadcast(data.decode("utf-8"))
        except Exception:
            await asyncio.sleep(0.01)


@app.on_event("startup")
async def on_startup() -> None:
    asyncio.create_task(udp_relay_task())


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    await ws.accept()
    await hub.register(ws)
    try:
        while True:
            # Client doesn't need to send anything; keep the connection open.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.unregister(ws)


# ---------------------------------------------------------------------- #
# History API
# ---------------------------------------------------------------------- #

def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            started_at REAL,
            updated_at REAL,
            status TEXT,
            sample_count INTEGER,
            notes TEXT,
            meta_json TEXT,
            model_json TEXT,
            architecture_hash TEXT
        )
        """
    )
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    if "model_json" not in existing_cols:
        conn.execute("ALTER TABLE runs ADD COLUMN model_json TEXT")
    if "architecture_hash" not in existing_cols:
        conn.execute("ALTER TABLE runs ADD COLUMN architecture_hash TEXT")
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    out["meta"] = json.loads(out.pop("meta_json") or "{}")
    out["model"] = json.loads(out.pop("model_json") or "null")
    return out


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    conn = _db()
    try:
        rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    conn = _db()
    try:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return {"error": "not found"}
        return _row_to_dict(row)
    finally:
        conn.close()


def _diff_dict(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Flat key-level diff between two (usually small, hyperparameter-shaped)
    dicts. Returns only keys that differ or exist on one side only."""
    keys = set(a.keys()) | set(b.keys())
    changes = {}
    for k in sorted(keys):
        va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
        if va != vb:
            changes[k] = {"a": va, "b": vb}
    return changes


@app.get("/api/runs/{run_id_a}/diff/{run_id_b}")
def diff_runs(run_id_a: str, run_id_b: str) -> dict[str, Any]:
    """
    Compare two runs' model architecture and trial meta so you can see what
    changed between trials without leaving the dashboard — e.g. "run B used
    a different hidden_size and a higher assistance_level than run A."
    """
    conn = _db()
    try:
        row_a = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id_a,)).fetchone()
        row_b = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id_b,)).fetchone()
    finally:
        conn.close()

    if row_a is None or row_b is None:
        return {"error": "one or both runs not found"}

    a, b = _row_to_dict(row_a), _row_to_dict(row_b)
    model_a = (a.get("model") or {}).get("architecture", {}) or {}
    model_b = (b.get("model") or {}).get("architecture", {}) or {}

    return {
        "run_a": run_id_a,
        "run_b": run_id_b,
        "architecture_hash_a": a.get("architecture_hash"),
        "architecture_hash_b": b.get("architecture_hash"),
        "architecture_changed": a.get("architecture_hash") != b.get("architecture_hash"),
        "architecture_diff": _diff_dict(model_a, model_b),
        "meta_diff": _diff_dict(a.get("meta") or {}, b.get("meta") or {}),
    }


@app.get("/api/runs/{run_id}/data")
def get_run_data(run_id: str, max_points: int = 5000) -> dict[str, Any]:
    """
    Load all Parquet parts for a run, concatenate, and (if long) downsample
    evenly to max_points so the browser isn't handed millions of rows.
    """
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        return {"error": "not found"}

    parts = sorted(run_dir.glob("part-*.parquet"))
    frames = [pd.read_parquet(p) for p in parts]

    fallback = run_dir / "fallback.jsonl"
    if fallback.exists():
        frames.append(pd.read_json(fallback, lines=True))

    if not frames:
        return {"columns": [], "rows": []}

    df = pd.concat(frames, ignore_index=True)
    if "t" in df.columns:
        df = df.sort_values("t")

    if len(df) > max_points:
        step = len(df) // max_points
        df = df.iloc[::step]

    return {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    }


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str) -> dict[str, Any]:
    run_dir = RUNS_ROOT / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    conn = _db()
    try:
        conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


# ---------------------------------------------------------------------- #
# Run import (drag-and-drop a zipped run bundle from another machine)
# ---------------------------------------------------------------------- #

@app.post("/api/runs/import")
async def import_run(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Accepts a .zip containing one run's Parquet parts (part-*.parquet /
    fallback.jsonl) plus its metadata as a top-level metadata.json
    ({run_id, started_at, ..., meta, model}). Produced by /api/runs/{id}/export.
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "expected a .zip run bundle")

    tmp_zip = UPLOADS_ROOT / f"import_{uuid.uuid4().hex[:8]}.zip"
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    with tmp_zip.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        with zipfile.ZipFile(tmp_zip) as zf:
            names = zf.namelist()
            if "metadata.json" not in names:
                raise HTTPException(400, "run bundle missing metadata.json")
            metadata = json.loads(zf.read("metadata.json"))
            run_id = metadata.get("run_id")
            if not run_id:
                raise HTTPException(400, "metadata.json missing run_id")

            dest = RUNS_ROOT / run_id
            dest.mkdir(parents=True, exist_ok=True)
            for name in names:
                if name == "metadata.json" or name.endswith("/"):
                    continue
                zf.extract(name, dest)

        conn = _db()
        try:
            conn.execute(
                """
                INSERT INTO runs (run_id, started_at, updated_at, status, sample_count, notes, meta_json, model_json, architecture_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    updated_at=excluded.updated_at, status=excluded.status,
                    sample_count=excluded.sample_count, notes=excluded.notes,
                    meta_json=excluded.meta_json, model_json=excluded.model_json,
                    architecture_hash=excluded.architecture_hash
                """,
                (
                    run_id,
                    metadata.get("started_at"),
                    metadata.get("updated_at"),
                    metadata.get("status", "imported"),
                    metadata.get("sample_count", 0),
                    metadata.get("notes", ""),
                    json.dumps(metadata.get("meta", {})),
                    json.dumps(metadata.get("model")),
                    metadata.get("architecture_hash"),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return {"ok": True, "run_id": run_id}
    finally:
        tmp_zip.unlink(missing_ok=True)


@app.get("/api/runs/{run_id}/export")
def export_run(run_id: str):
    """Bundles a run's Parquet data + metadata into a .zip for import elsewhere."""
    from fastapi.responses import FileResponse

    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(404, "run not found")

    row = get_run(run_id)
    if row.get("error"):
        raise HTTPException(404, "run metadata not found")

    export_path = UPLOADS_ROOT / f"{run_id}.zip"
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(row, default=str))
        for p in run_dir.glob("*"):
            if p.is_file():
                zf.write(p, arcname=p.name)

    return FileResponse(export_path, filename=f"{run_id}.zip", media_type="application/zip")


# ---------------------------------------------------------------------- #
# Models: list available checkpoints + fixed-profile controllers to pick from
# ---------------------------------------------------------------------- #

@app.get("/api/models")
def list_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = [
        {
            "id": "tbe",
            "controller": "tbe",
            "name": "TBE fixed torque profile",
            "kind": "fixed-profile",
            "run_dir": None,
        }
    ]

    if ML_MODEL_RUNS.exists():
        for run_dir in sorted(ML_MODEL_RUNS.iterdir()):
            if not run_dir.is_dir():
                continue
            deploy_meta_path = run_dir / "deploy_metadata.json"
            if not deploy_meta_path.exists():
                continue  # not a valid deployable checkpoint

            model_meta = {}
            meta_path = run_dir / "model_meta.json"
            if meta_path.exists():
                model_meta = json.loads(meta_path.read_text())

            test_metrics = {}
            metrics_path = run_dir / "test_metrics.json"
            if metrics_path.exists():
                test_metrics = json.loads(metrics_path.read_text())

            wandb_run = {}
            wandb_path = run_dir / "wandb_run.json"
            if wandb_path.exists():
                wandb_run = json.loads(wandb_path.read_text())

            backends = []
            if (run_dir / "best.ts").exists():
                backends.append("jit")
            if (run_dir / "best.onnx").exists():
                backends.append("onnx")
            if (run_dir / "best.engine").exists():
                backends.append("trt")

            # Optional one-line friendly name, e.g. "Ankle Assist v1" — falls
            # back to the raw checkpoint folder name (a training timestamp
            # stamp) if a team hasn't set one. Add/edit it any time with:
            #   echo "My Model Name" > ML_model/runs/<checkpoint>/display_name.txt
            display_name_path = run_dir / "display_name.txt"
            display_name = (
                display_name_path.read_text().strip()
                if display_name_path.exists()
                else run_dir.name
            )

            models.append({
                "id": run_dir.name,
                "controller": "ml",
                "name": display_name,
                "checkpoint_id": run_dir.name,
                "kind": "tcn",
                "run_dir": str(run_dir.relative_to(REPO_ROOT / "ML_model")),
                "backends": backends,
                "model_arch": model_meta.get("model", {}),
                "test_metrics": test_metrics,
                "wandb_run_url": wandb_run.get("run_url"),
            })

    return models


# ---------------------------------------------------------------------- #
# Sample replay CSVs bundled with the dashboard (dashboard/backend/sample_data/)
# ---------------------------------------------------------------------- #

@app.get("/api/samples")
def list_samples() -> list[dict[str, Any]]:
    if not SAMPLE_DATA_ROOT.exists():
        return []
    out = []
    for p in sorted(SAMPLE_DATA_ROOT.glob("*.csv")):
        out.append({
            "filename": p.name,
            "path": str(p),
            "size_bytes": p.stat().st_size,
        })
    return out


# ---------------------------------------------------------------------- #
# Replay CSV upload (drag-and-drop into the Launch page)
# ---------------------------------------------------------------------- #

@app.post("/api/uploads/replay")
async def upload_replay_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "expected a .csv file")

    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    dest = UPLOADS_ROOT / safe_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"ok": True, "path": str(dest), "filename": file.filename}


# ---------------------------------------------------------------------- #
# Launch / stop a controller process
# ---------------------------------------------------------------------- #

@app.get("/api/launch/status")
def launch_status() -> dict[str, Any] | None:
    return process_manager.manager.status()


@app.post("/api/launch")
def launch(body: dict[str, Any]) -> dict[str, Any]:
    """
    body: {
      controller: "tbe" | "ml-mock" | "ml-dry-run",
      run_dir: str | null,      # required for ml-*, relative to ML_model/
      mass: float | null,
      replay_path: str | null,  # from /api/uploads/replay
      duration: float,
      backend: "jit" | "onnx" | "trt" | null,
      assist_scale: float | null,
    }
    """
    try:
        argv, cwd = process_manager.build_argv(
            controller=body.get("controller"),
            run_dir=body.get("run_dir"),
            mass=body.get("mass"),
            replay_path=body.get("replay_path"),
            duration=body.get("duration", 60.0),
            backend=body.get("backend"),
            assist_scale=body.get("assist_scale"),
        )
        job_id = uuid.uuid4().hex[:12]
        return process_manager.manager.start(argv, cwd, job_id)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/launch/stop")
def stop_launch() -> dict[str, Any]:
    try:
        return process_manager.manager.stop()
    except RuntimeError as e:
        raise HTTPException(400, str(e))
