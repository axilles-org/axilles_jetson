"""Mock deployment on the Jetson: live sensors -> ML prediction -> log + plot.

NO torque is sent to the motor (no CAN bus is opened). This validates the full
inference path on real hardware before any powered test:

    foot IMU + shank IMU + ankle encoder + heel/toe FSR
      -> SensorAdapter (exo frame -> training convention)
      -> FeaturePipeline -> 3 s rolling window
      -> TCN (TorchScript / ONNX)
      -> predicted ankle plantarflexion moment (N.m/kg)

The AssistanceController is run too (its stance gate / ramp / limits are
exercised) but its output is recorded only, as ``would_command_nm``.

For the powered version see scripts/jetson_deploy.py.

Run on the Jetson:
    python scripts/jetson_mock_deploy.py \
        --run runs/tcn_mid_stance_lastN_20260831_212705 \
        --mass 72 --axilles ~/axilles_jetson \
        --duration 120 --out logs/mock_run.csv

Then review:
    python scripts/plot_mock_run.py logs/mock_run.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exo.config import Config, replace
from exo.deploy.jetson_io import FRAME_KEYS, ReplaySensors, Teleplot
from exo.deploy.runtime import ExoController


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="trained run dir (best.ts / best.onnx)")
    ap.add_argument("--mass", type=float, required=True, help="wearer body mass (kg)")
    ap.add_argument("--axilles", default="~/axilles_jetson",
                    help="path to the axilles_jetson repo (hardware I/O)")
    ap.add_argument("--replay", default=None,
                    help="replay a data_collection CSV instead of live sensors")
    ap.add_argument("--backend", choices=["jit", "onnx", "trt"], default=None)
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--out", default="logs/mock_run.csv")
    ap.add_argument("--no-teleplot", action="store_true")
    ap.add_argument("--print-every", type=int, default=50)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = Config.load(run_dir / "config.yaml")
    if args.backend:
        cfg = replace(cfg, **{"deploy.backend": args.backend})
    rate = args.rate or float(cfg.deploy.control_rate_hz)
    dt = 1.0 / rate

    controller = ExoController(run_dir, cfg.deploy, subject_mass_kg=args.mass)
    controller.reset()

    if args.replay:
        sensors = ReplaySensors(args.replay, rate)
    else:
        from exo.deploy.jetson_io import JetsonSensors
        sensors = JetsonSensors(args.axilles)
    tp = Teleplot(enabled=not args.no_teleplot)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["t_s", *FRAME_KEYS, "predicted_nm_per_kg", "predicted_nm",
              "would_command_nm", "stance", "ramp", "buffer_ready"]
    csv_file = out_path.open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    print(f"mock deploy | {rate:.0f} Hz | backend={cfg.deploy.backend} | "
          f"mass={args.mass} kg | NO TORQUE SENT | logging -> {out_path}")
    print("fill the 3 s window by walking, then watch predicted_nm_per_kg / stance\n")

    t0 = time.perf_counter()
    i = 0
    try:
        while (t_now := time.perf_counter() - t0) < args.duration:
            loop_start = time.perf_counter()

            raw = sensors.read()
            out = controller.step(raw, dt)
            pred_kg = out["predicted_nm_per_kg"]
            pred_nm = pred_kg * args.mass

            writer.writerow({
                "t_s": round(t_now, 4), **{k: round(raw[k], 5) for k in FRAME_KEYS},
                "predicted_nm_per_kg": round(pred_kg, 5), "predicted_nm": round(pred_nm, 4),
                "would_command_nm": round(out["command_nm"], 4),
                "stance": int(out["stance"]), "ramp": round(out["ramp"], 3),
                "buffer_ready": int(out["buffer_ready"])})

            t_ms = int(time.time() * 1000)
            tp.send("ml_pred_nm_per_kg,ML", pred_kg, t_ms)
            tp.send("ml_pred_nm,ML", pred_nm, t_ms)
            tp.send("ml_would_cmd_nm,ML", out["command_nm"], t_ms)
            tp.send("stance,ML", out["stance"], t_ms)
            tp.send("heel_fsr,FSR", raw["heel_fsr_raw"], t_ms)
            tp.send("toe_fsr,FSR", raw["toe_fsr_raw"], t_ms)
            tp.send("ankle_deg,Encoder", raw["ankle_encoder_deg"], t_ms)

            if args.print_every and i % args.print_every == 0:
                print(f"[{t_now:6.1f}s] pred={pred_kg:+.3f} Nm/kg ({pred_nm:+6.2f} Nm)  "
                      f"would_cmd={out['command_nm']:6.2f} Nm  stance={int(out['stance'])}  "
                      f"ramp={out['ramp']:.2f}  ready={int(out['buffer_ready'])}")

            if args.replay and getattr(sensors, "exhausted", False):
                break
            i += 1
            sleep = dt - (time.perf_counter() - loop_start)
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        csv_file.close()
        sensors.shutdown()
        print(f"\n{i} frames over {time.perf_counter() - t0:.1f} s. log: {out_path}\n"
              f"  review: python scripts/plot_mock_run.py {out_path}")


if __name__ == "__main__":
    main()
