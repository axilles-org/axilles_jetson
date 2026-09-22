"""
Launches controller processes (mock/dry-run only — see module docstring on
`ALLOWED_MODES`) from the dashboard and tracks their lifecycle.

Safety posture: this manager will never construct an argv containing --arm.
Sending real torque to real hardware stays a deliberate, CLI-only action;
the browser can only launch modes that compute + log but never transmit.
Only one job may run at a time — a second launch request while one is
active is rejected, not queued or force-killed, so a trial in progress is
never silently interrupted by an unrelated browser click.
"""

from __future__ import annotations

import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ML_MODEL_DIR = REPO_ROOT / "ML_model"
TBE_DIR = REPO_ROOT / "TBE_controller"

# Modes the dashboard is allowed to launch. "mock" never touches the motor at
# all (jetson_mock_deploy.py). "dry-run" exercises the full command path
# (ramp, clamping, MIT packing) through MotorInterface but with dry_run=True,
# so nothing is ever written to a CAN bus. Neither can command the actuator.
ALLOWED_MODES = ("mock", "dry-run")

MAX_LOG_LINES = 2000


@dataclass
class JobState:
    job_id: str
    argv: list[str]
    cwd: str
    status: str = "starting"  # starting | running | completed | failed | stopped
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    return_code: int | None = None
    log_lines: list[str] = field(default_factory=list)
    run_id: str | None = None  # filled in once the child prints "Run ID: ..."

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "argv": self.argv,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "return_code": self.return_code,
            "run_id": self.run_id,
            "log_tail": self.log_lines[-200:],
        }


class ProcessManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._state: JobState | None = None
        self._reader_thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict[str, Any] | None:
        with self._lock:
            return self._state.to_dict() if self._state else None

    def start(self, argv: list[str], cwd: Path, job_id: str) -> dict[str, Any]:
        for flag in argv:
            if flag == "--arm" or flag.startswith("--arm="):
                raise PermissionError(
                    "Refusing to launch with --arm: the dashboard can only start "
                    "mock/dry-run controller processes. Armed (torque-sending) "
                    "runs must be started from the CLI."
                )

        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError(
                    f"A job is already running (job_id={self._state.job_id}, "
                    f"pid={self._proc.pid}). Stop it before starting another."
                )

            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,  # own process group, so we can signal cleanly
            )
            state = JobState(job_id=job_id, argv=argv, cwd=str(cwd), status="running")
            self._proc = proc
            self._state = state

            self._reader_thread = threading.Thread(
                target=self._read_output, args=(proc, state), daemon=True
            )
            self._reader_thread.start()

            return state.to_dict()

    def _read_output(self, proc: subprocess.Popen, state: JobState) -> None:
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                state.log_lines.append(line)
                if len(state.log_lines) > MAX_LOG_LINES:
                    state.log_lines = state.log_lines[-MAX_LOG_LINES:]
                if state.run_id is None and "Run ID:" in line:
                    state.run_id = line.split("Run ID:", 1)[1].strip()
        finally:
            proc.wait()
            with self._lock:
                state.return_code = proc.returncode
                state.ended_at = time.time()
                state.status = "completed" if proc.returncode == 0 else "failed"

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                raise RuntimeError("No job is currently running.")
            proc = self._proc
            state = self._state

        # SIGINT first (the scripts already handle KeyboardInterrupt to zero
        # torque / close resources cleanly); escalate to SIGTERM/SIGKILL only
        # if it doesn't exit promptly.
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

        with self._lock:
            state.status = "stopped"
            state.ended_at = time.time()
            state.return_code = proc.returncode
            return state.to_dict()


manager = ProcessManager()


def build_argv(
    controller: str,
    run_dir: str | None,
    mass: float | None,
    replay_path: str | None,
    duration: float,
    backend: str | None,
    assist_scale: float | None,
) -> tuple[list[str], Path]:
    """
    Build a safe argv for one of the two controller entrypoints. `controller`
    is "tbe" (TBE_controller/main.py, no CLI args) or "ml-mock"
    (ML_model/scripts/jetson_mock_deploy.py, mock only — no --arm exists on
    this script, it never sends torque by construction).
    """
    if controller == "tbe":
        # -u: unbuffered stdout/stderr, so log lines reach the dashboard as
        # they're printed instead of only appearing in a burst at exit
        # (Python fully buffers stdout by default when it isn't a TTY, which
        # a subprocess pipe never is).
        return ["python3", "-u", str(TBE_DIR / "main.py")], TBE_DIR

    if controller in ("ml-mock", "ml-dry-run"):
        if not run_dir:
            raise ValueError("run_dir is required for the ML controller")

        # ml-mock: jetson_mock_deploy.py has no --arm flag at all — torque
        # transmission is structurally impossible, not just unrequested.
        # It never touches MotorInterface/the CAN bus (no motor code path
        # exists in this script), whereas ml-dry-run exercises the full
        # command path (ramp, clamp, MIT packing) with dry_run=True. Both
        # are fully wired to RunLogger/ModelInfo, so either shows up live
        # and in history.
        script = "scripts/jetson_mock_deploy.py" if controller == "ml-mock" else "scripts/jetson_deploy.py"
        argv = [
            "python3",
            "-u",
            script,
            "--run", run_dir,
            "--mass", str(mass or 70.0),
            "--duration", str(duration),
            "--no-teleplot",
        ]
        if controller == "ml-dry-run":
            argv += ["--dry-run"]
        if replay_path:
            argv += ["--replay", replay_path]
        if backend:
            argv += ["--backend", backend]
        # --assist-scale only exists on jetson_deploy.py; jetson_mock_deploy.py
        # has no such flag (it never computes a gated/scaled command, only
        # would_command_nm for review) and errors out if it's passed.
        if controller == "ml-dry-run" and assist_scale is not None:
            argv += ["--assist-scale", str(assist_scale)]
        return argv, ML_MODEL_DIR

    raise ValueError(f"unknown controller {controller!r}")
