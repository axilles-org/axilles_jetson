# Test 4 — Control loop and CAN command rate (R-05)

**Objective:** confirm that the high-level controller and the CAN torque commands
keep up at least 100 Hz under full load: all sensors, TCN inference, CAN traffic,
safety checks and buffered logging.

This folder runs the test **on the bench, without walking**.

| File | What it does |
|---|---|
| `test4_run.py` | Runs the controller loop and records every loop and every CAN send. Phases: `check`, `idle`, `load`, `all` |
| `can_monitor.py` | Runs as a separate process. Timestamps every frame on the CAN bus and samples memory, temperature and CPU frequency |
| `test4_analyze.py` | Computes the metrics, applies the pass criteria, and writes `report.md`, `report.json` and plots |
| `sim_hw.py` | Simulated sensors and virtual CAN for `--sim` (checks the pipeline only) |

---

## Why no walking is needed

The procedure's 10-minute walk is there to put the loop under full load. On this
controller that load does not depend on whether anyone is walking:

- **Sensors.** The BNO085s report at a fixed rate, and the FSR (ADS1115) and
  encoder (AS5600) reads are the same I2C transactions whether the foot moves or
  not.
- **TCN.** It does the same arithmetic on every 300-sample window, whatever the
  values in it.
- **CAN.** The powered runner sends one MIT torque frame every loop, in stance
  and swing alike, and the motor answers each one.
- **Logging and safety checks.** These run every loop either way.

So the test reads the real sensors every loop and runs the real controller code.
It also **injects a recorded walk** (`Data collection/data/…_tightshoes_200Hz_2kmph.csv`)
into the controller, so the stance gate, ramps and torque path follow a gait
pattern exactly as they would in walking.

By default every frame carries **0 N·m** (`--torque-mode zero`). The bus traffic
and the motor's replies are identical to a walking run, but the shaft never
moves, so nobody has to wear the exo.

What this does not cover: the electrical load of the motor actually producing
torque. If you want that too, mount the actuator safely and run with
`--torque-mode computed` (capped at 2 N·m by default).

### How the procedure maps onto this test

| Procedure step | Here |
|---|---|
| 1. 10 s idle reference | `idle` phase: the same loop, scheduler and logger with no work. Shows how much timing jitter the Jetson adds on its own. The CAN monitor also records any background bus traffic. |
| 2. 10 min walk with everything enabled | `load` phase, 60 s timed window by default (`--duration 600` for the plan's full 10 minutes): live sensors, gait injection, TCN, safety checks, a CAN frame every loop, the deploy runner's logging (CSV row, dashboard `RunLogger`, console line), Teleplot |
| 3. Record each loop start/end and every CAN send | `loops.csv` (start, end and per-stage time for every loop), `can_sends.csv` (every `send()` call), `can_monitor.npz` (every frame on the wire, kernel timestamp) |
| 4. Flush the log after the run; optional marked flush run | The test's own timing log stays in RAM and is written after the timed window, so measuring adds no disk I/O. `--flush-run` adds a marked run that writes and fsyncs the log every second inside the loop. |

---

## Before you start

1. **Hardware.** Connect the exo electronics: IMUs at 0x4A and 0x4B, the ADS1115
   at 0x48 and the AS5600 at 0x36, all on I2C bus 1. Power the AK80-9 (CAN id
   0x68).
2. **CAN link** (needs sudo; do this after every reboot):
   ```bash
   sudo ip link set can0 down
   sudo ip link set can0 up type can bitrate 1000000 restart-ms 100
   ```
3. **Python.** Use the default `python3` (miniconda, Python 3.13). It has
   onnxruntime, torch, python-can and the Adafruit BNO08x driver. The system
   `/usr/bin/python3` has no onnxruntime or torch, so it cannot run the TCN.
4. **ML controller code.** On `main` (and on this branch), `ML_model/src/exo/data/`
   has no `.py` sources, so the TCN controller cannot be imported. The complete
   controller is on `axilles-59-ml-control-model`. Until that is merged, check it
   out next to this repo and point the test at it:
   ```bash
   git worktree add ../exo_ws_ml axilles-59-ml-control-model
   # then add to every command below:  --ml-root ../exo_ws_ml/ML_model
   ```
   The TCN defaults to `<ml-root>/runs/deploy`, with the backend from that run's
   `config.yaml` (`onnx`). Override with `--run-dir` / `--backend`.

   The same checkout supplies the dashboard `RunLogger`
   (`<checkout>/dashboard/backend/run_logger.py`), which `jetson_deploy.py`
   calls every loop. Each load run therefore also appears as a run in that
   checkout's dashboard database, tagged `Test 4 harness`.
5. **Power mode.** Results depend on the Jetson power mode and the CPU frequency
   governor. With `schedutil`, CPU clocks were seen dropping to 730 MHz during a
   run. Record which mode you test in, and use the one the exo will run in. The
   report lists the power mode, governor, and min/max CPU frequency seen during
   the run.

---

## Running the test

All commands are run from the repository root.

```bash
# 1. preflight: CAN link, the four I2C sensors, listen for motor traffic, TCN load time
#    (sends nothing to the motor; add --motor-check to also send 20 zero-torque frames)
python3 fall_test_plan/test_4/test4_run.py --phase check --ml-root ../exo_ws_ml/ML_model

# 2. the whole test: check -> 10 s idle -> 60 s load -> analysis
#    (add --duration 600 for the test plan's full 10-minute run)
python3 fall_test_plan/test_4/test4_run.py --phase all --ml-root ../exo_ws_ml/ML_model

# same, plus the marked disk-flush run (120 s, write + fsync every 1 s)
python3 fall_test_plan/test_4/test4_run.py --phase all --flush-run --ml-root ../exo_ws_ml/ML_model
```

Results go to `fall_test_plan/test_4/results/<date_time>/` (`idle/`, `load/`,
`flush/`, `report.md` and the plots). The console prints the verdict at the end.
The exit code is 0 only if the test passes.

Ctrl-C at any point sends 0 N·m frames, closes the CAN bus, and
still writes everything recorded so far. The run is then marked `interrupted`.
With `--phase all`, the outer process waits for the running step to finish its
own shutdown and never kills it.

**Dead-man switch.** The bus monitor runs as a separate process. If the
controller dies without shutting down after commanding the motor (`kill -9`,
a crash), the monitor sends 0 N·m itself within about 20–100 ms,
then saves its data and exits. Without this, the motor would hold its last
torque command until its own CAN timeout (about 2 s on this motor).
If set-up is stuck (for example an IMU reconnect loop), press Ctrl-C a second
time. Once shutdown has started, further Ctrl-Cs are ignored until the logs are
written.

Closing the terminal stops the run the same way, and the output goes to
`console_after_hangup.log` in the run folder. The rest of an `all` session
(flush run, analysis) is then skipped. For a 10-minute run, start it inside
`tmux` (or with `nohup ... &`) so that it keeps going if the terminal closes.
`kill -9` still loses the RAM log: only `deploy_log.csv` survives that.

### Motor only, no sensors attached (`--sim-sensors`)

Use this when the Jetson and the motor are connected but the exo's sensors are
not. The CAN bus, motor, TCN, safety checks and logging are all real. Only the
sensor reads are replaced by fixed waits: 1.4 ms for both IMUs and 3.0 ms for
the FSRs and encoder. Change them with `--sim-imu-ms` and `--sim-fsr-enc-ms`.
These waits are placeholders, not measurements. The report is stamped
**SIMULATED SENSORS**, and the full test has to be repeated with the sensors
connected.

```bash
python3 fall_test_plan/test_4/test4_run.py --phase all --sim-sensors --ml-root ../exo_ws_ml/ML_model
```

### Without hardware (pipeline check only)

```bash
python3 fall_test_plan/test_4/test4_run.py --phase all --sim --duration 30 --ml-root ../exo_ws_ml/ML_model
```

This uses simulated sensors (fixed placeholder read times) and a virtual CAN bus
with a fake motor that replies. The TCN is still the real model on the real CPU.
The report is stamped **SIMULATED HARDWARE**. Never use it as a test result.

### Re-analysing

```bash
python3 fall_test_plan/test_4/test4_analyze.py fall_test_plan/test_4/results/<session>
python3 fall_test_plan/test_4/test4_analyze.py <run dir> [<run dir> ...] --out <dir>
```

### Useful options (`test4_run.py --help` lists them all)

| Option | Default | Meaning |
|---|---|---|
| `--duration` | 60 | Timed window of the load run, in s. The test plan asks for 600 (10 min). Rare stalls (> 50 ms) were seen only in the 10-minute run, so use 600 for the result you report. |
| `--idle-duration` | 10 | Idle reference, in s |
| `--preroll` | 5 (load), 1 (idle) | Settling time before the timed window. The load window also waits for the TCN's 300-sample buffer to fill. Pre-roll loops are logged but not scored. |
| `--rate` | 100 | Control rate, in Hz |
| `--scheduler` | `relative` | `relative` = `sleep(dt - elapsed)`, as `jetson_deploy.py` and TBE `main.py` do. `deadline` = fixed 10 ms deadlines (see below). |
| `--logger` | `deploy` | `deploy` adds `jetson_deploy.py`'s per-loop logging: `csv.DictWriter` row, `RunLogger.log_sample` (which writes Parquet from a thread every 500 rows), and a console line every 50 loops. `buffered` = RAM only (lighter than the real runner). |
| `--torque-mode` | `zero` | `computed` sends the controller's torque. The motor moves. |
| `--torque-cap`, `--command-sign`, `--slew`, `--watchdog-ms`, `--arm-ramp` | 2.0, -1, 10, 25, 5 | Same safety settings as `jetson_deploy.py` |
| `--backend` | from the run's config | `onnx`, `jit` or `trt` |
| `--no-gait-inject` | off | Feed the controller live sensor values instead of the recorded walk |
| `--flush-run`, `--flush-duration`, `--flush-every` | off, 120, 1.0 | The marked disk-flush run |
| `--no-teleplot` | off | Teleplot is on by default because the deploy runner streams it |
| `--imu-bus`, `--can-channel` | 1, can0 | |
| `--sim-sensors`, `--sim-imu-ms`, `--sim-fsr-enc-ms` | off, 1.4, 3.0 | Real CAN and motor, simulated sensor reads |
| `--motor-check` | off | Preflight also sends 20 zero-torque frames |
| `--sim` | off | No hardware |

---

## What the report contains

`report.md` covers the idle, load and flush runs side by side. Only the timed
window is scored.

- **Mean loop rate:** (loops − 1) / (last loop start − first loop start).
- **Period:** time from one loop start to the next. The report gives the median,
  99th percentile and largest period, the % of periods longer than 10 ms and
  20 ms, and the number longer than 50 ms. The last timed loop's period is
  closed by the start time of the loop that ended the run.
- **Loop work:** time from a loop's start to its end, without the sleep. The
  report also gives the % of loops whose work took longer than 10 ms and 20 ms.
  This is the other reading of "loops longer than 10 ms", and it is free of
  scheduler jitter.
- **CAN command rate:** taken from the bus monitor, i.e. torque frames as they
  actually left the CAN controller. The Jetson's mttcan driver echoes a frame to
  other sockets only once it has been transmitted. The rate from the `send()`
  calls inside the controller is reported too.
- **Where the time goes:** mean, median, p99 and max per loop stage (IMU read,
  FSR + encoder read, TCN inference, adapter and features, CAN send, log row,
  Teleplot, disk flush), plus the stage times of the five longest periods.
- **Integrity:**
  - every loop has exactly one log row, with no gaps in the sequence;
  - every successful `send()` appears as a frame on the bus;
  - no CAN send errors, bus errors, dropped frames or bus-off (interface counters
    are read before and after the run);
  - the motor's reply rate;
  - RSS and MemAvailable over the run;
  - a kernel-log (`journalctl -k`) search for OOM-killer events during the run.
- **Plots:**
  - `periods_timeline.png`: the period of every loop over the run;
  - `period_histogram.png`: the spread of periods for idle, load and flush;
  - `stage_breakdown.png`: time per loop stage in the load run;
  - `can_intervals.png`: the gap between torque frames on the bus.

### Pass criteria (scored on the load run)

| Criterion | Pass if |
|---|---|
| Mean loop rate | ≥ 100 Hz, judged to 0.01 Hz (anything ≥ 99.995 Hz counts as 100) |
| 99th-percentile period | ≤ 20 ms |
| Largest period | ≤ 50 ms |
| CAN torque-command rate (on the wire) | ≥ 100 Hz, judged to 0.01 Hz |
| No controller restart | the run completed in one process, with no gaps in the loop sequence and no exception |
| No log or frame loss | log rows = loops, frames on the bus = successful sends, no send errors, no CAN drops or bus-off |
| No out-of-memory event | no OOM line in the kernel log during the run |

Why the 0.01 Hz resolution: with exact 10 ms deadlines the measured mean
scatters by about ±0.00002 Hz around 100 Hz (99.99998 one run, 100.00002 the
next). Comparing raw floats would make the result a coin flip. A real shortfall,
such as one lost loop per 2000, is still caught. The constants are at the top of
`test4_analyze.py` if the test owner wants another tolerance.

If the controller hits a CAN send error, the test records it and keeps going so
the whole window is measured. The real `jetson_deploy.py` would stop at the first
send error, so any send error fails the test.

---

## Reading the results

- **`relative` scheduling cannot reach a 100 Hz mean.** `sleep(dt - elapsed)`
  always oversleeps by about 0.06 ms, so every period is a little over 10 ms. On
  this Jetson the *empty* idle loop runs at about 99.3 Hz, with 100 % of periods
  over 10 ms. This is how the current runners schedule, so it is the default and
  the measured number is honest. Run once more with `--scheduler deadline` to
  separate scheduling from workload. That mode keeps absolute 10 ms deadlines
  and makes up a late wake-up on the next loop, so the empty loop holds
  100.0000 Hz.
- **"% of periods over 10 ms" at a 100 Hz target** is mostly sub-millisecond
  jitter: about 35–45 % even for an empty loop with deadlines. Use "loop work
  > 10 ms" to see real overruns.
- **Diagnosis lines** under the verdict say whether the loop work itself exceeds
  the 10 ms budget, and which stages dominate.
- **Expect the TCN step to be the largest cost.** Benchmarked on this Jetson
  without sensors, `ExoController.step` with the ONNX backend took 6.1 ms median
  (TCN 3.1 ms, the rest in the adapter and feature pipeline) and 39 ms at worst.
  The TorchScript backend took 12.7 ms median. Sensor I/O adds to that. The last
  powered run in `ML_model/logs/powered_run.csv` logged about 52 Hz (p99 period
  45 ms). Use the stage table to decide what to optimise.

## If something fails

The preflight (`--phase check`) tells hardware problems apart from software
problems:

| Message | Likely cause |
|---|---|
| `CAN interface can0 is down` / wrong bitrate | Run the two `ip link` commands above |
| `I2C bus 1 0x.. not responding` | That sensor is unplugged, unpowered, or on another bus (try `--imu-bus 7` for the IMUs) |
| `motor reply: 0 replies` | Motor unpowered, CAN H/L swapped, missing 120 Ω termination, or the motor id is not 0x68 |
| `Cannot import the ML controller` | `--ml-root` points at a checkout without `exo/data`. See step 4 above. |
| `bus monitor saw N receive errors (Network is down)` | The CAN link went down during the run (bus-off). Check wiring and termination. |

## Notes on the current code (found while building this test)

- **Safety: the MIT "enter/exit/zero" frames are full-scale commands on this
  motor.** The AK80-9 runs CubeMars *servo-mode* firmware (its status frames
  are id `0x2968`). That firmware takes the 8-byte "special" frames
  `FF FF FF FF FF FF FF FC/FD/FE` as ordinary MIT commands with every field at
  maximum: kp ≈ 500, kd ≈ 5, target +12.56 rad at +65 rad/s, near-max torque.
  On the bench, each "enter" frame produced 41–46 A and drove the shaft to full
  speed within 64 ms. Each "exit" frame produced up to 59 A and full speed,
  held until the motor's ~2 s CAN timeout.
  `ML_model/src/exo/deploy/motor.py` sends them from `enter()`, `exit()` /
  `shutdown()` and `zero_position()`, and `ML_model/scripts/jetson_deploy.py`
  calls `enter()` at start and `shutdown()` at the end. That is the torque
  that continued after stopping the walking test. This test never sends those
  frames: it starts and stops with 0 N·m frames. `motor.py` has since been fixed
  the same way: `enter()` and `exit()` send 0 N·m, and `zero_position()` uses
  the servo-mode set-origin command. `tests/test_motor.py` checks that no
  special frame is ever sent.

- `TBE_controller/data_obtainer.py` has `self.can_bus.send(msg)` commented out
  in `sendTorqueData`, on every branch. The TBE controller computes torque but
  never transmits it. This test drives the ML controller's `MotorInterface`,
  which does transmit.
- `TBE_controller/data_obtainer.py` (`SensorData`) opens its own handle on
  `can0` and sends a standard-id enter-MIT frame at start-up. The test keeps this
  behaviour because it is part of the real sensor path. The monitor reports such
  frames under `rx_ids` / other ids.
- `Sensor_Tests/IMUs/BNO085/bno085_live_dual_fast.py` defaults to I2C bus 7. The
  exo uses bus 1, which the ML branch also passes. The test passes `--imu-bus`,
  default 1.
