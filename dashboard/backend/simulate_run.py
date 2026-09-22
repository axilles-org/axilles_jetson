"""
Fake-data simulator for testing the dashboard without the exo hardware.

Generates a plausible gait-cycle waveform (ankle angle, velocity, torque,
heel/toe FSR) at 150Hz and pushes it through the exact same RunLogger used
by TBE_controller/main.py — so it exercises the real Parquet writer, the
real SQLite run table, and the real UDP live relay end to end.

Usage:
    python dashboard/backend/simulate_run.py                 # one 15s run
    python dashboard/backend/simulate_run.py --duration 30
    python dashboard/backend/simulate_run.py --crash          # simulate a crash mid-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "TBE_controller"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.run_logger import RunLogger, ModelInfo  # noqa: E402

FREQ_HZ = 150.0
DT = 1.0 / FREQ_HZ
STRIDE_PERIOD_S = 1.1  # ~55 strides/min, reasonable walking cadence


def gait_waveforms(t: float) -> dict:
    """One plausible-looking ankle gait cycle. Not biomechanically exact —
    just enough shape/phase relationship to look real on the charts."""
    phase = (t % STRIDE_PERIOD_S) / STRIDE_PERIOD_S  # 0..1 within a stride

    # Ankle angle: dorsiflexion during stance, plantarflexion push-off, swing reset
    ankle_angle = (
        10 * np.sin(2 * np.pi * phase)
        - 15 * np.exp(-((phase - 0.55) ** 2) / 0.002)  # push-off plantarflexion spike
        + np.random.normal(0, 0.3)
    )

    ankle_velocity = 300 * np.cos(2 * np.pi * phase) + np.random.normal(0, 5)

    # Torque profile mirrors TAU_PHASE_ARRAY/TAU_VAL_ARRAY shape in utilities.py:
    # ramps up through toe-in/toe-off window, zero elsewhere.
    torque_cmd = 6.0 * np.exp(-((phase - 0.65) ** 2) / 0.01) + np.random.normal(0, 0.05)
    torque_cmd = max(0.0, torque_cmd)

    # FSR: heel strike near phase 0, toe-off/loading near phase 0.6-0.8
    heel_fsr = 20000 * np.exp(-((phase - 0.05) ** 2) / 0.003) + np.random.normal(0, 200)
    toe_fsr = 18000 * np.exp(-((phase - 0.55) ** 2) / 0.01) + np.random.normal(0, 200)

    return {
        "ankle_angle": float(ankle_angle),
        "ankle_velocity": float(ankle_velocity),
        "torque_cmd": float(torque_cmd),
        "heel_fsr": max(0.0, float(heel_fsr)),
        "toe_fsr": max(0.0, float(toe_fsr)),
    }


def run_simulation(
    duration_s: float,
    crash: bool,
    assistance_level: float,
    notes: str,
    hidden_size: int,
    wandb_run_id: str | None,
) -> None:
    run_logger = RunLogger(
        meta={
            "assistance_level": assistance_level,
            "peak_torque": 7.0,
            "motor_control_freq": FREQ_HZ,
            "source": "simulate_run.py (fake data, no hardware)",
        },
        notes=notes,
        model_info=ModelInfo(
            name="sim-fake-policy",
            wandb_run_url=(
                f"https://wandb.ai/axilles-org/tbe-controller/runs/{wandb_run_id}"
                if wandb_run_id
                else None
            ),
            wandb_run_id=wandb_run_id,
            wandb_project="tbe-controller" if wandb_run_id else None,
            architecture={
                "hidden_size": hidden_size,
                "num_layers": 2,
                "activation": "relu",
                "assistance_level": assistance_level,
            },
            checkpoint_ref=f"wandb-artifact://tbe-controller/model-{wandb_run_id}:latest"
            if wandb_run_id
            else None,
        ),
    )
    run_logger.start()
    print(f"[sim] Run ID: {run_logger.run_id}")
    print(f"[sim] Streaming fake gait data for {duration_s:.0f}s at {FREQ_HZ:.0f}Hz "
          f"(watch it at http://localhost:5173)")

    t_start = time.perf_counter()
    calibrated_at = 2.0  # fake a short calibration period at the start, like the real controller

    try:
        while True:
            loop_start = time.perf_counter()
            elapsed = loop_start - t_start
            if elapsed > duration_s:
                break

            if crash and elapsed > duration_s / 2:
                raise RuntimeError("Simulated crash for testing 'crashed' run status")

            values = gait_waveforms(elapsed)
            run_logger.log_sample(
                t=loop_start,
                calibrated=elapsed > calibrated_at,
                stride_time=STRIDE_PERIOD_S if elapsed > calibrated_at else float("nan"),
                **values,
            )

            sleep_time = DT - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        run_logger.stop(status="completed")
        print(f"[sim] Done. {run_logger._sample_count} samples logged to "
              f"dashboard/data/runs/{run_logger.run_id}/")

    except KeyboardInterrupt:
        run_logger.stop(status="completed")
        print("[sim] Interrupted, run saved as completed.")
    except Exception as e:
        run_logger.stop(status="crashed")
        print(f"[sim] Crashed as requested: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=15.0, help="seconds of fake data")
    parser.add_argument("--crash", action="store_true", help="simulate a mid-run crash")
    parser.add_argument("--assistance", type=float, default=0.1)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--hidden-size", type=int, default=64, help="fake model arch param, for diff testing")
    parser.add_argument("--wandb-run-id", type=str, default=None, help="fake W&B run id to link, for testing")
    args = parser.parse_args()

    run_simulation(
        args.duration, args.crash, args.assistance, args.notes, args.hidden_size, args.wandb_run_id
    )
