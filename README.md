# axilles_jetson

Ankle exoskeleton control on a Jetson: sensor drivers, two controllers, and
a trial dashboard.

## Layout

| Folder | What it is |
|---|---|
| `TBE_controller/` | Fixed torque-profile controller — the working baseline |
| `ML_model/` | TCN torque-prediction model: training, deployment scripts, checkpoints |
| `dashboard/` | Live telemetry + trial history/comparison for either controller — see [`dashboard/README.md`](dashboard/README.md) |
| `BNO085/` | IMU driver and bench-test scripts |
| `Cubemars_AK80-9/` | AK80-9 actuator driver and bench-test scripts |
| `test_encoder.py`, `test_fsr.py` | Standalone sensor bench tests |

## Where to start

- Running the fixed-profile controller: `TBE_controller/main.py`
- Running or training the ML controller: see [`ML_model/README.md`](ML_model/README.md)
- Watching a trial live, browsing past runs, or launching a controller from
  a browser: see [`dashboard/README.md`](dashboard/README.md)
