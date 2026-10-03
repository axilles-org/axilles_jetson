"""Recover SensorAdapter parameters from an exo walking recording.

Complements ``calibrate_exo.py`` (static hold): this matches the exo's gait-cycle
-averaged per-channel waveforms to the GaTech training bands to recover the yaw
DOF of each IMU rotation, the gyro/accel unit scales, the encoder neutral offset
and sign, and the two FSR thresholds - from a label-free walking trial.

    python scripts/fit_adapter.py \
        --walk ~/Downloads/data_collection_20260404_215743_tightshoes_200Hz_2kmph.csv \
        --processed ~/PycharmProjects/exo-data/processed \
        --out configs/sensor_adapter.yaml --plot

Paste the printed ``sensor_adapter:`` block under ``deploy:`` in the config.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exo.config import Config
from exo.deploy.adapter_fit import fit_adapter, summarize


def _plot(result, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = result.reference
    chans = ref.channels
    exo = result.exo_phase_avg
    phase = np.linspace(0, 100, exo.shape[0])

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for i, ax in enumerate(axes.flat):
        if i >= len(chans):
            ax.axis("off")
            continue
        m, s = ref.mean[:, i], ref.std[:, i]
        ax.fill_between(phase, m - s, m + s, alpha=0.25, color="tab:blue",
                        label="GaTech band")
        ax.plot(phase, m, color="tab:blue", lw=1)
        ax.plot(phase, exo[:, i], color="tab:red", ls="--", lw=1.5, label="exo (adapted)")
        ax.set_title(chans[i], fontsize=9)
        ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=8, loc="upper right")
    d = result.diagnostics
    fig.suptitle(
        f"SensorAdapter fit - {result.n_cycles} cycles  |  "
        f"gyro res foot {d['foot']['gyro_residual']} / shank {d['shank']['gyro_residual']}  |  "
        f"accel res foot {d['foot']['accel_residual_with_lever_arm']} / "
        f"shank {d['shank']['accel_residual_with_lever_arm']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"plot -> {path}")


def _write_into_config(config_path: str, block: dict) -> None:
    """Replace the deploy.sensor_adapter block in an existing config YAML.

    Round-trips through pyyaml, so comments in the config are not preserved; keep
    a copy under version control before patching.
    """
    doc = yaml.safe_load(Path(config_path).read_text())
    doc.setdefault("deploy", {})["sensor_adapter"] = block
    Path(config_path).write_text(yaml.safe_dump(doc, sort_keys=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", required=True, help="exo walking-trial CSV")
    ap.add_argument("--processed", default=None,
                    help="GaTech processed .npz dir (default: cfg.paths.processed())")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="configs/sensor_adapter.yaml")
    ap.add_argument("--write-config", action="store_true",
                    help="patch deploy.sensor_adapter in --config in place")
    ap.add_argument("--rate", type=float, default=None, help="control rate Hz")
    ap.add_argument("--no-slow-match", action="store_true",
                    help="use all GaTech speeds for the reference band")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    processed = args.processed or str(cfg.paths.processed())
    rate = args.rate or float(cfg.deploy.control_rate_hz)

    result = fit_adapter(args.walk, processed, control_rate_hz=rate,
                         match_slow_speed=not args.no_slow_match)
    print(summarize(result))

    block = {"sensor_adapter": result.config_block}
    Path(args.out).write_text(yaml.safe_dump(block, sort_keys=False))
    Path(args.out).with_suffix(".diagnostics.json").write_text(
        json.dumps(result.diagnostics, indent=2))
    print(f"\nwritten -> {args.out}\n")
    print(yaml.safe_dump(block, sort_keys=False))

    if args.write_config:
        _write_into_config(args.config, result.config_block)
        print(f"patched deploy.sensor_adapter in {args.config}")

    if args.plot:
        _plot(result, str(Path(args.out).with_suffix(".png")))


if __name__ == "__main__":
    main()
