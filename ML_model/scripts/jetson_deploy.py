"""Powered ML deployment: live sensors -> ExoController -> AK80-9 torque.

Torque is off unless --arm. Safety: session ramp (--arm-ramp), loop-overrun
watchdog, and any exception / Ctrl-C -> zero torque -> exit MIT mode. --dry-run
runs the motor interface without a CAN bus. Run scripts/jetson_mock_deploy.py and
review it first. See docs/DEPLOYMENT.md.

    python scripts/jetson_deploy.py --run runs/<dir> --mass 72 \
        --axilles ~/axilles_jetson --arm --assist-scale 0.1 --torque-cap 2.0
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exo.config import Config, replace
from exo.deploy.jetson_io import FRAME_KEYS, ReplaySensors, Teleplot
from exo.deploy.motor import MIT_T_MAX, MotorInterface
from exo.deploy.runtime import ExoController

_STOP = False


def _handle_signal(signum, frame):
    global _STOP
    _STOP = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mass", type=float, required=True)
    ap.add_argument("--axilles", default="~/axilles_jetson")
    ap.add_argument("--replay", default=None, help="CSV to replay instead of live sensors")
    ap.add_argument("--backend", choices=["jit", "onnx", "trt"], default=None)
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--out", default="logs/powered_run.csv")

    ap.add_argument("--arm", action="store_true",
                    help="ACTUALLY send torque (default: compute + log only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="MotorInterface without a CAN bus (bench check)")
    ap.add_argument("--assist-scale", type=float, default=None,
                    help="override deploy.assistance_scale")
    ap.add_argument("--torque-cap", type=float, default=2.0,
                    help="hard N.m ceiling on the command (<= 5)")
    ap.add_argument("--arm-ramp", type=float, default=5.0,
                    help="seconds to fade assistance 0 -> full after arming")
    ap.add_argument("--command-sign", type=float, default=1.0,
                    help="maps assist magnitude to the actuator direction")
    ap.add_argument("--watchdog-ms", type=float, default=25.0,
                    help="zero torque if a loop takes longer than this")

    ap.add_argument("--no-teleplot", action="store_true")
    ap.add_argument("--print-every", type=int, default=50)
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    run_dir = Path(args.run)
    cfg = Config.load(run_dir / "config.yaml")
    overrides = {}
    if args.backend:
        overrides["deploy.backend"] = args.backend
    if args.assist_scale is not None:
        overrides["deploy.assistance_scale"] = args.assist_scale
    if overrides:
        cfg = replace(cfg, **overrides)

    rate = args.rate or float(cfg.deploy.control_rate_hz)
    dt = 1.0 / rate
    torque_cap = min(args.torque_cap, MIT_T_MAX)

    controller = ExoController(run_dir, cfg.deploy, subject_mass_kg=args.mass)
    controller.reset()

    # sensors
    if args.replay:
        sensors = ReplaySensors(args.replay, rate)
    else:
        from exo.deploy.jetson_io import JetsonSensors
        sensors = JetsonSensors(args.axilles)

    # motor. dry_run drives the interface (clamp/sign/pack, session ramp) without
    # a CAN bus; --arm without --dry-run actually transmits.
    motor = MotorInterface(torque_limit_nm=torque_cap, command_sign=args.command_sign,
                           dry_run=args.dry_run)
    live_torque = args.arm and not args.dry_run
    exercise_command = args.arm or args.dry_run       # compute + ramp the command
    if live_torque or args.dry_run:
        motor.enter()

    tp = Teleplot(enabled=not args.no_teleplot)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["t_s", *FRAME_KEYS, "predicted_nm_per_kg", "predicted_nm",
              "assist_cmd_nm", "sent_torque_nm", "session_ramp", "stance",
              "ramp", "buffer_ready", "loop_ms"]
    csv_file = out_path.open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    mode = ("ARMED - SENDING TORQUE" if live_torque else
            "dry-run: command computed, NOT transmitted" if args.dry_run else
            "compute + log only, NO torque")
    print(f"powered ML deploy | {rate:.0f} Hz | backend={cfg.deploy.backend} | "
          f"mass={args.mass} kg | scale={cfg.deploy.assistance_scale} | "
          f"cap={torque_cap} N.m | {mode}")
    if exercise_command:
        print(f"assistance ramps 0 -> full over {args.arm_ramp:.1f} s. Ctrl-C = stop.\n")

    t0 = time.perf_counter()
    i = 0
    sent = 0.0
    try:
        while not _STOP and (t_now := time.perf_counter() - t0) < args.duration:
            loop_start = time.perf_counter()

            raw = sensors.read()
            out = controller.step(raw, dt)
            pred_kg = out["predicted_nm_per_kg"]
            assist_cmd = out["command_nm"]                 # N.m from AssistanceController

            session_ramp = min(1.0, t_now / args.arm_ramp) if exercise_command else 0.0
            target = assist_cmd * session_ramp

            loop_ms = (time.perf_counter() - loop_start) * 1000.0
            if loop_ms > args.watchdog_ms:
                target = 0.0                               # overrun -> no torque

            sent = motor.send_torque(target) if exercise_command else 0.0

            row = {"t_s": round(t_now, 4),
                   **{k: round(raw[k], 5) for k in FRAME_KEYS},
                   "predicted_nm_per_kg": round(pred_kg, 5),
                   "predicted_nm": round(pred_kg * args.mass, 4),
                   "assist_cmd_nm": round(assist_cmd, 4),
                   "sent_torque_nm": round(sent, 4),
                   "session_ramp": round(session_ramp, 3),
                   "stance": int(out["stance"]), "ramp": round(out["ramp"], 3),
                   "buffer_ready": int(out["buffer_ready"]),
                   "loop_ms": round(loop_ms, 2)}
            writer.writerow(row)

            t_ms = int(time.time() * 1000)
            tp.send("ml_pred_nm_per_kg,ML", pred_kg, t_ms)
            tp.send("assist_cmd_nm,ML", assist_cmd, t_ms)
            tp.send("sent_torque_nm,Motor", sent, t_ms)
            tp.send("session_ramp,ML", session_ramp, t_ms)
            tp.send("stance,ML", out["stance"], t_ms)
            tp.send("loop_ms,Perf", loop_ms, t_ms)

            if args.print_every and i % args.print_every == 0:
                print(f"[{t_now:6.1f}s] pred={pred_kg:+.3f} Nm/kg  assist={assist_cmd:6.2f}  "
                      f"sent={sent:+.2f} Nm  sess_ramp={session_ramp:.2f}  "
                      f"stance={int(out['stance'])}  loop={loop_ms:.1f}ms")

            if args.replay and getattr(sensors, "exhausted", False):
                break
            i += 1
            sleep = dt - (time.perf_counter() - loop_start)
            if sleep > 0:
                time.sleep(sleep)

    except Exception as exc:                               # noqa: BLE001
        print(f"\n!! exception: {exc!r} -> zeroing torque")
        raise
    finally:
        try:
            motor.stop()
        finally:
            motor.shutdown()
        csv_file.close()
        sensors.shutdown()
        tp.close()
        print(f"\n{i} frames over {time.perf_counter() - t0:.1f} s. "
              f"last torque {motor.last_torque_nm:+.2f} N.m. log: {out_path}")


if __name__ == "__main__":
    main()
