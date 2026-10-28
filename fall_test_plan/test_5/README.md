# Test 5 — End-to-end latency: sensor read to torque command (R-05)

**Objective:** measure the software delay between the last sensor read and the
outgoing torque command.

This test is **analysis only**. It reads the logs of a Test 4 load run and never
opens the CAN bus or the I2C bus, so it cannot move the motor. It does not
measure when torque appears at the ankle: that needs a separate force or torque
sensor.

| File | What it does |
|---|---|
| `test5_analyze.py` | Reads a Test 4 load run, computes the per-stage and end-to-end delay for every torque command, checks each command's safety-limit result, and writes the report, a per-command CSV and the plots |

---

## Running it

From the repository root:

```bash
# newest Test 4 run
python3 fall_test_plan/test_5/test5_analyze.py

# a specific Test 4 session (or its load/ folder)
python3 fall_test_plan/test_5/test5_analyze.py fall_test_plan/test_4/results/<session>
```

Results go to `fall_test_plan/test_5/results/<date_time>/`. Use `--out <dir>` to
write somewhere else, and `--no-plots` to skip the figures. The console prints
the verdict. The exit code is 0 only if the test passes.

To produce the input, run Test 4 with the sensors connected (see
`fall_test_plan/test_4/README.md`). The test plan asks for the 10-minute run:

```bash
python3 fall_test_plan/test_4/test4_run.py --phase all --duration 600 --ml-root ../exo_ws_ml/ML_model
```

Test 4's default `--torque-mode zero` sends 0 N·m in every frame, so the motor
does not move.

---

## What is measured

Test 4 logs, for every loop, the loop start time, how long each stage took
(`loops.csv`), the start and end of every `bus.send()` call (`can_sends.csv`),
and the kernel timestamp of every frame on the CAN bus (`can_monitor.npz`). The
loop runs its stages in this order:

```
loop start ─ sensor read ─┬─ preprocessing ─ TCN inference ─ safety checks + packing ─ CAN send ─┬─ (logging, Teleplot, sleep)
                          │                                                                       │
                          └──────────────── end-to-end delay (what this test scores) ─────────────┘
```

| Stage | From the logs | What runs in it |
|---|---|---|
| Preprocessing | `step_ms − predict_ms` | `controller.step` without the TCN: sensor adapter, feature pipeline, window buffer, stance gate and assist profile |
| TCN inference | `predict_ms` | `backend.predict` (ONNX by default) |
| Safety checks + packing | `send() start − (sensor read end + step_ms)` | session ramp, watchdog, torque cap, slew limit, MIT frame packing. Also includes the gait-injection lookup, a few µs |
| CAN send | `frame on the bus − send() start` | the `bus.send()` call plus transmission. The report splits it into the call and "send() return → on the bus" |

- **End of the sensor read** = loop start + IMU read + FSR/encoder read. The few
  µs between loop start and the read are counted as delay, so the result is
  slightly pessimistic.
- **End point** = the torque frame on the CAN bus, which is when the command
  really leaves the Jetson. The Jetson's mttcan driver echoes a frame to the
  monitor only after transmitting it. The k-th successful `send()` is matched to
  the k-th command frame on the bus. That is exact when the two counts agree,
  which the report confirms. If they disagree, or there is no monitor data, the
  end point falls back to `send()` returning, and the report says so. Both
  numbers are always reported.
- The four stages add up exactly to the end-to-end delay.
- "send() return → on the bus" can be negative. If the thread is preempted after
  the system call, the frame goes out before `send()` returns. The CAN send stage
  is measured from the start of `send()`, so it is always positive.

### Safety-limit result per command

For every command, the test re-derives from the logs the safety result the
controller had to compute before sending, and checks it against what was
actually sent and what was actually on the bus:

| Check | Pass if |
|---|---|
| One frame | exactly one command frame for the loop, no send error |
| Finite | assist, target and sent torque are finite numbers |
| Watchdog decision | it tripped exactly when the work before the send exceeded `--watchdog-ms` (25 ms) |
| Watchdog zeroes | a trip set the target to 0 N·m |
| Session ramp | target = assist × session ramp |
| Within cap | \|sent\| ≤ torque cap (2 N·m by default) |
| Pipeline | sent = sign, cap and slew limit applied to the target, as `MotorInterface.send_torque` does |
| Frame on the bus | the torque decoded from the transmitted frame is within the cap and equals the logged sent torque (to one 12-bit step, 0.011 N·m), and kp = kd = 0 |

The watchdog uses the loop time measured *before* `send_torque()`, and the cap
and slew limit are applied inside `send_torque()` before the frame is packed. So
a result that passes these checks existed before transmission.

### CAN reply age (reported separately, not scored)

The AK80-9 runs CubeMars servo-mode firmware. It **broadcasts** a status frame
(`0x2968`) every 20 ms whatever the command rate, and does not reply to each
command. Two ages are reported:

- **command on the bus → the motor's next status frame**: the reply age as the
  procedure defines it;
- **age of the newest motor feedback when `send()` starts**: how old the motor
  state is that a controller could have used for that command.

Both come out roughly uniform between 0 and 20 ms, because the motor's broadcast
period sets them, not our software. The controller does not use motor feedback
yet, so this age does not add to the command delay.

---

## Pass criteria

| Criterion | Pass if |
|---|---|
| Median sensor-read-to-CAN-send delay | ≤ 10 ms |
| 99th-percentile delay | ≤ 20 ms |
| Maximum delay | ≤ 50 ms |
| Every torque command has a valid safety-limit result before transmission | all checks above pass for every command in the timed window |

Only Test 4's timed window is scored. Pre-roll loops and the start/stop 0 N·m
frames are excluded.

---

## Outputs

| File | Contents |
|---|---|
| `report.md` | verdict, criteria table, stage table (median, p99, max, mean), slowest stage, five slowest commands, safety check counts, CAN reply age |
| `report.json` | the same numbers, machine-readable |
| `latency_per_command.csv` | one row per command: every stage time, end-to-end delay, reply ages, torques and the result of every safety check |
| `criteria_scorecard.png` | measured vs. limit for each criterion, with PASS/FAIL |
| `e2e_latency_histogram.png` | distribution of the end-to-end delay (log scale), with the limits |
| `stage_breakdown.png` | median and p99 of each stage and of the whole path |
| `loop_anatomy.png` | timeline of two real loops: the one at the median delay and the slowest one |
| `e2e_latency_timeline.png` | delay of every command over the run, with watchdog trips marked |
| `can_reply_age.png` | the two reply-age distributions |

The figures are 16:9 at 2000×1125 px, sized for slides.

---

## Notes

- **Simulated sensors.** All Test 4 runs so far used `--sim-sensors`: the sensor
  reads were fixed waits. The delay scored here starts when the read ends, so it
  covers only real work (controller, TCN, safety checks, CAN). The reports and
  plots are still stamped, and the final result should come from a Test 4 run
  with the sensors connected.
- **Run length.** The test plan asks for the 10-minute loaded run. In Test 4,
  rare stalls (> 50 ms periods) were seen only in 10-minute runs. A 60 s source
  run can therefore miss the worst case for the maximum-delay criterion.
- **Gait injection.** With Test 4's default gait injection, the controller is fed
  a recorded walk while the sensors are still read every loop. The timing path
  is identical, so this does not change the delay.
