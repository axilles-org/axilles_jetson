"""
RunLogger: drop-in telemetry recorder for the TBE controller loop.

Design constraints (matches the existing 150Hz MIT-mode control loop in
TBE_controller/main.py):
  - log_sample() must never block on disk or network I/O. It only appends
    to an in-memory list and optionally does a non-blocking UDP send.
  - All actual I/O (Parquet flush, SQLite write) happens on a background
    thread, or at explicit start()/stop() boundaries outside the hot loop.
  - If anything in here raises, the control loop must keep running —
    telemetry is never allowed to take down the exo. All I/O is wrapped.

Model / experiment linkage:
  A hardware trial (this RunLogger's run_id) and an ML experiment (a W&B
  run) are two different things that need to be joined, not merged:
    - This RunLogger owns the hardware side: raw sensor traces at full
      rate, written to Parquet, because W&B is the wrong tool for a
      150Hz firehose.
    - W&B (if used) owns the ML side: training curves, sweeps, and
      checkpoint artifacts for whatever policy/model produced the
      commanded torque this trial.
  Pass `model_info=ModelInfo(...)` to RunLogger to record which model
  produced this trial's behavior, a W&B URL to jump to its training
  history, and an architecture fingerprint so two runs' model configs
  can be diffed later even without opening W&B. For the current
  fixed-profile TBE controller there is no learned model — pass nothing,
  or describe the torque-profile parameters as the "architecture" so the
  same diffing tooling still finds parameter changes across trials.

Usage in main.py:

    from dashboard.backend.run_logger import RunLogger, ModelInfo

    run_logger = RunLogger(
        meta={"assistance_level": ASSISTANCE_LEVEL, "peak_torque": PEAK_TORQUE},
        model_info=ModelInfo(
            name="tbe-fixed-profile",
            wandb_run_url="https://wandb.ai/axilles-org/tbe-controller/runs/abcd1234",
            wandb_run_id="abcd1234",
            architecture={"tau_phase": list(TAU_PHASE_ARRAY), "tau_val": list(TAU_VAL_ARRAY)},
            checkpoint_ref=None,
        ),
    )
    run_logger.start()
    ...
    while True:
        ...
        run_logger.log_sample(
            t=loop_start,
            heel_fsr=sensor_data.filtered_heel_fsr,
            toe_fsr=sensor_data.filtered_toe_fsr,
            ankle_angle=sensor_data.encoder_data,
            ankle_velocity=sensor_data.filtered_encoder_velocity,
            torque_cmd=sensor_data._torque_plot,
            calibrated=calibration.calibrated,
            stride_time=controller.stride_time,
        )
        ...
    # on KeyboardInterrupt / shutdown:
    run_logger.stop()
"""

from __future__ import annotations

import hashlib
import json
import queue
import socket
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RUNS_ROOT = Path(__file__).resolve().parent.parent / "data" / "runs"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "runs.db"

# Local UDP relay to the dashboard backend's WebSocket bridge (see server.py).
# Fire-and-forget, same pattern as the existing Teleplot socket in data_obtainer.py.
LIVE_RELAY_HOST = "127.0.0.1"
LIVE_RELAY_PORT = 47270

# How many samples to buffer in memory before flushing to disk.
FLUSH_EVERY_N_SAMPLES = 500


def _new_run_id() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def architecture_hash(architecture: dict[str, Any]) -> str:
    """
    Short, stable fingerprint of a model/config dict. Two runs with the same
    architecture always hash the same, regardless of key order — this is
    what the Runs UI uses to flag "architecture changed" between trials
    without needing to open W&B or diff JSON by eye.
    """
    return hashlib.sha256(_canonical_json(architecture).encode("utf-8")).hexdigest()[:12]


@dataclass
class ModelInfo:
    """
    Links a hardware trial to the ML side of the world: which model/policy
    produced the commanded behavior, where its training history lives in
    W&B, and a content-addressed fingerprint of its architecture/config so
    runs can be diffed even without network access to W&B.

    name:            short human label, e.g. "tbe-fixed-profile" or "ppo-ankle-v3"
    wandb_run_url:   full URL to the W&B run — dashboard just links out to it
    wandb_run_id:    W&B run id alone, for API lookups if ever needed
    wandb_project:   W&B project name, for constructing/verifying URLs
    architecture:    small JSON-able dict describing the model/config —
                     hyperparameters, layer sizes, torque-profile params,
                     whatever defines "what this trial was running." Kept
                     deliberately small (this is NOT the full checkpoint).
    checkpoint_ref:  pointer to the actual weights (W&B artifact path, S3
                     URI, or local path) — not embedded here, just referenced.
    """

    name: str = ""
    wandb_run_url: str | None = None
    wandb_run_id: str | None = None
    wandb_project: str | None = None
    architecture: dict[str, Any] = field(default_factory=dict)
    checkpoint_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "wandb_run_url": self.wandb_run_url,
            "wandb_run_id": self.wandb_run_id,
            "wandb_project": self.wandb_project,
            "architecture": self.architecture,
            "architecture_hash": architecture_hash(self.architecture) if self.architecture else None,
            "checkpoint_ref": self.checkpoint_ref,
        }


@dataclass
class RunLogger:
    meta: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=_new_run_id)
    notes: str = ""
    model_info: ModelInfo | None = None

    def __post_init__(self) -> None:
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        self._run_dir = RUNS_ROOT / self.run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._sample_count = 0
        self._part_index = 0

        self._relay_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._relay_addr = (LIVE_RELAY_HOST, LIVE_RELAY_PORT)

        # Background writer thread so Parquet flushes never happen in the
        # caller's (control loop's) thread.
        self._write_queue: "queue.Queue[list[dict[str, Any]] | None]" = queue.Queue()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)

    # ------------------------------------------------------------------ #
    # Public API — called from the control loop
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._started_at = time.time()
        self._writer_thread.start()
        self._write_run_metadata(status="running")

    def log_sample(self, **fields: Any) -> None:
        """Non-blocking. Appends to memory buffer, relays over UDP for live view."""
        fields.setdefault("_recv_t", time.time())
        try:
            with self._lock:
                self._buffer.append(fields)
                self._sample_count += 1
                if len(self._buffer) >= FLUSH_EVERY_N_SAMPLES:
                    batch = self._buffer
                    self._buffer = []
                    self._write_queue.put(batch)
        except Exception:
            pass  # telemetry must never break the control loop

        self._relay(fields)

    def stop(self, status: str = "completed") -> None:
        try:
            with self._lock:
                if self._buffer:
                    self._write_queue.put(self._buffer)
                    self._buffer = []
            self._write_queue.put(None)  # sentinel to stop writer thread
            self._writer_thread.join(timeout=5.0)
        finally:
            self._write_run_metadata(status=status)
            self._relay_sock.close()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _relay(self, fields: dict[str, Any]) -> None:
        try:
            payload = {"run_id": self.run_id, **fields}
            msg = json.dumps(payload, default=str).encode("utf-8")
            self._relay_sock.sendto(msg, self._relay_addr)
        except OSError:
            pass  # dashboard backend not running — fine, file logging still happens

    def _writer_loop(self) -> None:
        import pandas as pd

        while True:
            batch = self._write_queue.get()
            if batch is None:
                break
            try:
                df = pd.DataFrame(batch)
                part_path = self._run_dir / f"part-{self._part_index:05d}.parquet"
                df.to_parquet(part_path, index=False)
                self._part_index += 1
            except Exception as e:
                # Last-resort fallback so we never silently lose a whole batch:
                # append as newline-delimited JSON if Parquet write fails
                # (e.g. missing pyarrow, malformed batch).
                fallback = self._run_dir / "fallback.jsonl"
                with open(fallback, "a") as f:
                    for row in batch:
                        f.write(json.dumps(row, default=str) + "\n")
                print(f"[RunLogger] Parquet write failed ({e}); wrote fallback JSONL.")

    def _write_run_metadata(self, status: str) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        try:
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
            # Cheap migration for DBs created before model_json/architecture_hash existed.
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
            if "model_json" not in existing_cols:
                conn.execute("ALTER TABLE runs ADD COLUMN model_json TEXT")
            if "architecture_hash" not in existing_cols:
                conn.execute("ALTER TABLE runs ADD COLUMN architecture_hash TEXT")

            model_dict = self.model_info.to_dict() if self.model_info else None
            arch_hash = model_dict["architecture_hash"] if model_dict else None

            conn.execute(
                """
                INSERT INTO runs (run_id, started_at, updated_at, status, sample_count, notes, meta_json, model_json, architecture_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    status=excluded.status,
                    sample_count=excluded.sample_count,
                    notes=excluded.notes,
                    model_json=excluded.model_json,
                    architecture_hash=excluded.architecture_hash
                """,
                (
                    self.run_id,
                    self._started_at,
                    time.time(),
                    status,
                    self._sample_count,
                    self.notes,
                    json.dumps(self.meta, default=str),
                    json.dumps(model_dict, default=str) if model_dict else None,
                    arch_hash,
                ),
            )
            conn.commit()
        finally:
            conn.close()
