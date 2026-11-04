# Motor safety tests

Offline tests for fixes S4–S10. **They cannot run the motor.**

```bash
motor_safety_tests/run.sh
```

| Protection | Where |
|---|---|
| `can0` does not exist for the tests (empty network namespace) | `run.sh` |
| The tests refuse to start if any CAN interface is visible | `conftest.py` |
| The `can` and `smbus2` modules are replaced with fakes | `conftest.py`, `fakes.py` |
| Opening a CAN socket raises an error | `conftest.py` |

| File | Checks |
|---|---|
| `test_s4_no_special_frames.py` | No enter/exit frames, sensor path leaves CAN alone, stuck ADC raises |
| `test_s5_packing.py` | Every file uses ±22 N·m and matches the V3 manual's bytes |
| `test_s6_s7_s10_motor_thread.py` | Motor thread, status check and disable frame on a fake bus |
| `test_s8_torque_cap.py` | Bad torque caps are refused |
| `test_s9_nonfinite.py` | NaN or infinity never becomes torque, including with the real model |
| `test_end_to_end_dry_run.py` | The real `jetson_deploy.py` in `--dry-run` on a recorded CSV |

These prove the code's logic, not the motor's behaviour. Nothing here has been run on
the AK80-9, including the disable frame, which is off unless `--disable-on-exit` is set.
