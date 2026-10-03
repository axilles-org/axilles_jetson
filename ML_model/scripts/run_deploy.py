"""Drive the exo controller from a recorded CSV or a live sensor source.

Replay mode resamples the recording to the control rate and streams it through
``ExoController``, printing the commanded torque and stance state.

    python scripts/run_deploy.py --run runs/<run_dir> --mass 72 \
        --replay ~/Downloads/data_collection_20260404_215743_tightshoes_200Hz_2kmph.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exo.config import Config, replace
from exo.deploy.runtime import ExoController

_RAW_COLUMNS = [
    "foot_ax", "foot_ay", "foot_az", "foot_gx", "foot_gy", "foot_gz",
    "shank_ax", "shank_ay", "shank_az", "shank_gx", "shank_gy", "shank_gz",
    "ankle_encoder_deg", "toe_fsr_raw", "heel_fsr_raw",
]


def load_replay(path: str, control_rate_hz: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.set_index("timestamp_s").sort_index()
    df = df[[c for c in _RAW_COLUMNS if c in df.columns]].interpolate().ffill().bfill()
    duration = df.index[-1] - df.index[0]
    grid = np.arange(0.0, duration, 1.0 / control_rate_hz)
    resampled = pd.DataFrame(
        {c: np.interp(grid, df.index - df.index[0], df[c].to_numpy()) for c in df.columns},
        index=grid,
    )
    return resampled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mass", type=float, required=True)
    ap.add_argument("--replay", required=True, help="recorded exo CSV")
    ap.add_argument("--backend", choices=["jit", "onnx", "trt"], default=None,
                    help="override deploy.backend")
    ap.add_argument("--print-every", type=int, default=50)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = Config.load(run_dir / "config.yaml")
    if args.backend:
        cfg = replace(cfg, **{"deploy.backend": args.backend})
    controller = ExoController(run_dir, cfg.deploy, subject_mass_kg=args.mass)

    frames = load_replay(args.replay, cfg.deploy.control_rate_hz)
    dt = 1.0 / cfg.deploy.control_rate_hz
    controller.reset()

    peak = 0.0
    for i, (_, row) in enumerate(frames.iterrows()):
        out = controller.step(row.to_dict(), dt)
        peak = max(peak, out["command_nm"])
        if args.print_every and i % args.print_every == 0:
            print(f"[{i:5d}] cmd={out['command_nm']:6.2f} Nm  "
                  f"pred={out['predicted_nm_per_kg']:+.3f} Nm/kg  "
                  f"stance={out['stance']:.0f}  ramp={out['ramp']:.2f}  "
                  f"ready={out['buffer_ready']:.0f}")

    print(f"\nprocessed {len(frames)} frames  peak command {peak:.2f} Nm")


if __name__ == "__main__":
    main()
