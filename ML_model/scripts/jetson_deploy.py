"""Powered ML deployment: live sensors -> ExoController -> AK80-9 torque.

Torque is off unless --arm. Safety: session ramp (--arm-ramp), loop-overrun
watchdog, a motor thread that zeroes torque if the loop hangs or the motor reports
a problem, and any exception / Ctrl-C -> zero torque. --dry-run runs the motor
interface without a CAN bus. Run scripts/jetson_mock_deploy.py and
review it first. See docs/DEPLOYMENT.md.

    python scripts/jetson_deploy.py --run runs/<dir> --mass 72 \
        --arm --assist-scale 0.1 --torque-cap 2.0

--start-passive (with --arm): the motor is engaged at zero torque and assistance
stays off until the dashboard switches it on (exo.deploy.assist_switch). With
--ankle-zero auto the first --calib-s seconds of walking zero the encoder against
the GaTech walking median; assistance cannot turn on before that.
"""
from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# repo root (two levels up from ML_model/scripts) so dashboard/ is importable,
# same pattern used by TBE_controller/main.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exo.config import Config, replace
from exo.deploy.assist_switch import AssistSwitch
from exo.deploy.dashboard_info import build_model_info
from exo.deploy.prepare_backend import prepare_backend
from exo.deploy.jetson_io import FRAME_KEYS, ReplaySensors, Teleplot
from exo.deploy.motor import SAFE_T_MAX, MotorInterface
from exo.deploy.runtime import ExoController

from dashboard.backend.run_logger import RunLogger

_STOP = False


def _handle_signal(signum, frame):
    global _STOP
    _STOP = True


def send_assistance(motor, target, *, dt, enabled, stance, ready, faulted, overrun):
    """Apply the final gate after all controller ramps, including the motor slew."""
    permitted = enabled and stance and ready and not faulted and not overrun
    return motor.send_torque(target if permitted else 0.0, dt=dt,
                             immediate_zero=not permitted)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mass", type=float, required=True)
    ap.add_argument("--height", type=float, default=None,
                    help="wearer height (m); with --gender, re-exports the model for this wearer")
    ap.add_argument("--gender", choices=["M", "F"], default=None)
    ap.add_argument("--axilles", default=str(Path(__file__).resolve().parents[2]),
                    help="path to the axilles_jetson repo (default: this checkout)")
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
    ap.add_argument("--assistance-mode", choices=["plantarflexion", "bidirectional"],
                    default=None, help="override deploy.assistance_mode; both modes are stance-only")
    ap.add_argument("--dorsi-scale-ratio", type=float, default=None,
                    help="dorsiflexion scale relative to the dashboard assistance scale")
    ap.add_argument("--dorsi-cap", type=float, default=None,
                    help="dorsiflexion command limit, also bounded by --torque-cap")
    ap.add_argument("--torque-cap", type=float, default=2.0,
                    help="hard N.m ceiling on the command (<= 5)")
    ap.add_argument("--arm-ramp", type=float, default=5.0,
                    help="seconds to fade assistance 0 -> full after arming")
    ap.add_argument("--command-sign", type=float, default=1.0,
                    help="maps assist magnitude to the actuator direction (+1: plantarflexion "
                         "assist is positive motor torque; -1 gave dorsiflexion on a wearer)")
    ap.add_argument("--slew", type=float, default=20.0,
                    help="max N.m/s change while assisting; swing, assist-off, "
                         "watchdog and faults bypass slew to zero")
    ap.add_argument("--watchdog-ms", type=float, default=25.0,
                    help="zero torque if a loop takes longer than this")
    ap.add_argument("--sender-stale-ms", type=float, default=100.0,
                    help="the motor thread zeroes torque and ends the run if the loop "
                         "gives it no target for this long (saved runs had gaps up to 79 ms)")
    ap.add_argument("--disable-on-exit", action="store_true",
                    help="also send the motor disable command on exit (not yet "
                         "tested on this motor)")
    ap.add_argument("--start-passive", action="store_true",
                    help="assistance off until switched on from the dashboard")
    ap.add_argument("--max-scale", type=float, default=0.3,
                    help="ceiling on the assistance scale the dashboard may set")
    ap.add_argument("--ankle-zero", choices=["config", "auto"], default="config",
                    help="auto: zero the encoder from the first --calib-s s of walking")
    ap.add_argument("--calib-s", type=float, default=10.0)

    ap.add_argument("--no-teleplot", action="store_true")
    ap.add_argument("--print-every", type=int, default=50)
    args = ap.parse_args()
    if not 0.0 < args.torque_cap <= SAFE_T_MAX:
        ap.error(f"--torque-cap must be above 0 and at most {SAFE_T_MAX}")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    run_dir = Path(args.run)
    cfg = Config.load(run_dir / "config.yaml")
    overrides = {}
    if args.backend:
        overrides["deploy.backend"] = args.backend
    if args.assist_scale is not None:
        overrides["deploy.assistance_scale"] = args.assist_scale
    for arg, key in ((args.assistance_mode, "assistance_mode"),
                     (args.dorsi_scale_ratio, "dorsiflexion_scale_ratio"),
                     (args.dorsi_cap, "dorsiflexion_limit_nm")):
        if arg is not None:
            overrides[f"deploy.{key}"] = arg
    if overrides:
        cfg = replace(cfg, **overrides)

    prepare_backend(run_dir, cfg, args.mass, args.height, args.gender)

    rate = args.rate or float(cfg.deploy.control_rate_hz)
    dt = 1.0 / rate
    torque_cap = args.torque_cap

    controller = ExoController(run_dir, cfg.deploy, subject_mass_kg=args.mass)
    controller.reset()

    # Trial tracking: same RunLogger used by TBE_controller/main.py, so ML
    # and fixed-profile trials show up side by side in the dashboard. Model
    # architecture + (if present) the W&B training run are read straight out
    # of the checkpoint directory, so nothing needs to be re-entered by hand.
    run_logger = RunLogger(
        meta={
            "controller": "ML_control (TCN)",
            "run_dir": str(run_dir),
            "subject_mass_kg": args.mass,
            "assistance_scale": cfg.deploy.assistance_scale,
            "assistance_mode": cfg.deploy.assistance_mode,
            "dorsiflexion_scale_ratio": cfg.deploy.dorsiflexion_scale_ratio,
            "dorsiflexion_limit_nm": cfg.deploy.dorsiflexion_limit_nm,
            "torque_cap_nm": torque_cap,
            "arm_ramp_s": args.arm_ramp,
            "command_sign": args.command_sign,
            "slew_nm_per_s": args.slew,
            "control_rate_hz": rate,
            "armed": bool(args.arm),
            "dry_run": bool(args.dry_run),
            "replay_source": args.replay,
            "start_passive": bool(args.start_passive),
            "ankle_zero": args.ankle_zero,
        },
        model_info=build_model_info(run_dir, cfg),
    )
    run_logger.start()
    print(f"Run ID: {run_logger.run_id}")

    # sensors
    if args.replay:
        sensors = ReplaySensors(args.replay, rate)
    else:
        from exo.deploy.jetson_io import JetsonSensors
        sensors = JetsonSensors(args.axilles)

    # motor. dry_run drives the interface (clamp/sign/pack, session ramp) without
    # a CAN bus; --arm without --dry-run actually transmits.
    motor = MotorInterface(torque_limit_nm=torque_cap, command_sign=args.command_sign,
                           dry_run=args.dry_run, slew_nm_per_s=args.slew,
                           disable_on_shutdown=args.disable_on_exit)
    live_torque = args.arm and not args.dry_run
    exercise_command = args.arm or args.dry_run       # compute + ramp the command
    if live_torque or args.dry_run:
        motor.enter()
        motor.start_sender(stale_s=args.sender_stale_ms / 1000.0)

    switch = (AssistSwitch(cfg.deploy.assistance_scale, args.max_scale)
              if args.start_passive else None)
    sign = cfg.deploy.sensor_adapter.ankle_encoder_sign
    reference_deg = None
    if args.ankle_zero == "auto":
        reference_deg = json.loads((run_dir / "ankle_reference.json").read_text())[
            "gatech_walking_median_deg"]
    calib_samples: list[float] = []
    calibrated = reference_deg is None

    tp = Teleplot(enabled=not args.no_teleplot)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["t_s", *FRAME_KEYS, "predicted_nm_per_kg", "predicted_nm",
              "assist_cmd_nm", "sent_torque_nm", "session_ramp", "stance",
              "ramp", "buffer_ready", "ankle_angle_deg", "calibrated",
              "assist_on", "assist_scale", "loop_ms", "motor_iq_a", "model_fault",
              "tick_ms", "sensor_read_ms", "preprocess_ms", "inference_ms",
              "prediction_t_s", "command_queued_t_s", "last_can_tx_t_s",
              "last_can_tx_nm", "motor_status_age_ms", "sensor_age_ms",
              "foot_accel_age_ms", "foot_gyro_age_ms", "shank_accel_age_ms",
              "shank_gyro_age_ms", "raw_stance", "watchdog_tripped"]
    csv_file = out_path.open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    mode = ("ARMED - SENDING TORQUE" if live_torque else
            "dry-run: command computed, NOT transmitted" if args.dry_run else
            "compute + log only, NO torque")
    print(f"powered ML deploy | {rate:.0f} Hz | backend={cfg.deploy.backend} | "
          f"mass={args.mass} kg | scale={cfg.deploy.assistance_scale} | "
          f"cap={torque_cap} N.m | sign={args.command_sign:+.0f} | "
          f"slew={args.slew} N.m/s | assistance={cfg.deploy.assistance_mode} | {mode}")
    if switch:
        print("PASSIVE: zero torque until assistance is switched on from the dashboard.")
    elif exercise_command:
        print(f"assistance ramps 0 -> full over {args.arm_ramp:.1f} s. Ctrl-C = stop.\n")
    if not calibrated:
        print(f"CALIBRATING: walk normally for {args.calib_s:.0f} s to zero the ankle angle.")

    t0 = time.perf_counter()
    i = 0
    sent = 0.0
    session_ramp = 0.0
    prev_start = None
    model_fault_seen = False
    try:
        while not _STOP and (t_now := time.perf_counter() - t0) < args.duration:
            loop_start = time.perf_counter()
            # measured tick (the loop can run slower than --rate); capped so a
            # stall doesn't license one large slew step
            tick_dt = min(loop_start - prev_start, 0.05) if prev_start else dt
            prev_start = loop_start

            sensor_start = time.perf_counter()
            raw = sensors.read()
            sensor_read_ms = (time.perf_counter() - sensor_start) * 1000.0
            sensor_ages = sensors.ages_ms() if hasattr(sensors, "ages_ms") else {}
            switch_enabled = switch.poll() if switch else exercise_command
            if switch:
                controller.assistance.scale = switch.scale
            if not calibrated:
                calib_samples.append(raw["ankle_encoder_deg"])
                if t_now >= args.calib_s:
                    neutral = float(np.median(calib_samples)) - reference_deg * sign
                    controller.set_ankle_neutral(neutral)
                    calibrated = True
                    print(f"ankle zero set: encoder {neutral:.1f} deg = GaTech 0")
            out = controller.step(raw, tick_dt)
            prediction_t = time.perf_counter()
            pred_kg = out["predicted_nm_per_kg"]
            assist_cmd = out["command_nm"]                 # N.m from AssistanceController
            if out["model_fault"] and not model_fault_seen:
                model_fault_seen = True
                print(f"!! non-finite model output at {t_now:.1f} s: assistance stays off")

            assist_on = (exercise_command and switch_enabled and calibrated
                         and bool(out["buffer_ready"]) and not bool(out["model_fault"]))
            session_ramp = (min(1.0, session_ramp + tick_dt / args.arm_ramp)
                            if assist_on else 0.0)
            target = assist_cmd * session_ramp

            loop_ms = (time.perf_counter() - loop_start) * 1000.0
            overrun = loop_ms > args.watchdog_ms
            sent = (send_assistance(
                motor, target, dt=tick_dt, enabled=assist_on, stance=bool(out["stance"]),
                ready=bool(out["buffer_ready"]), faulted=bool(out["model_fault"]),
                overrun=overrun) if exercise_command else 0.0)
            queued_t = time.perf_counter()
            tx_t, tx_nm = motor.transmission_snapshot()

            row = {"t_s": round(t_now, 4),
                   **{k: round(raw[k], 5) for k in FRAME_KEYS},
                   "predicted_nm_per_kg": round(pred_kg, 5),
                   "predicted_nm": round(pred_kg * args.mass, 4),
                   "assist_cmd_nm": round(assist_cmd, 4),
                   "sent_torque_nm": round(sent, 4),
                   "session_ramp": round(session_ramp, 3),
                   "stance": int(out["stance"]), "ramp": round(out["ramp"], 3),
                   "buffer_ready": int(out["buffer_ready"]),
                   "ankle_angle_deg": round((raw["ankle_encoder_deg"]
                                             - controller.adapter.cfg.ankle_encoder_neutral_deg)
                                            * sign, 2),
                   "calibrated": int(calibrated),
                   "assist_on": int(assist_on),
                   "assist_scale": round(controller.assistance.scale, 3),
                   "loop_ms": round(loop_ms, 2),
                   "motor_iq_a": round(motor.status["current_a"], 2) if motor.status else 0.0,
                   "model_fault": int(out["model_fault"]),
                   "tick_ms": round(tick_dt * 1000.0, 3),
                   "sensor_read_ms": round(sensor_read_ms, 3),
                   "preprocess_ms": round(out["preprocess_ms"], 3),
                   "inference_ms": round(out["inference_ms"], 3),
                   "prediction_t_s": round(prediction_t - t0, 6),
                   "command_queued_t_s": round(queued_t - t0, 6),
                   "last_can_tx_t_s": round(tx_t - t0, 6) if tx_t is not None else None,
                   "last_can_tx_nm": tx_nm if tx_t is not None else None,
                   "motor_status_age_ms": ((queued_t - motor.last_status_at) * 1000.0
                                           if motor.last_status_at is not None else None),
                   "sensor_age_ms": max(sensor_ages.values()) if sensor_ages else None,
                   **{f"{name}_age_ms": sensor_ages.get(name) for name in
                      ("foot_accel", "foot_gyro", "shank_accel", "shank_gyro")},
                   "raw_stance": int(controller.assistance.stance_detector(
                       raw["heel_fsr_raw"], raw["toe_fsr_raw"])),
                   "watchdog_tripped": int(overrun)}
            writer.writerow(row)
            run_logger.log_sample(t=loop_start, **row)

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
        motor.stop()  # zero before potentially blocking telemetry flush
        run_logger.stop(status="crashed")
        raise
    else:
        motor.stop()
        run_logger.stop(status="completed")
    finally:
        try:
            motor.stop()
        finally:
            motor.shutdown()
        csv_file.close()
        sensors.shutdown()
        if switch:
            switch.close()
        tp.close()
        print(f"\n{i} frames over {time.perf_counter() - t0:.1f} s. "
              f"last torque {motor.last_torque_nm:+.2f} N.m. log: {out_path}")


if __name__ == "__main__":
    main()
