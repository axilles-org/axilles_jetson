# Axilles Jetson

This repository contains the sensor, motor, data-collection, calibration, and
machine-learning code for the Axilles exoskeleton project.

## Main folders

- `Sensor_Tests/` - Sensor-specific tests and readers.
  - `IMUs/BNO085/` contains the BNO085 reader, dual-reader variants, diagnostic
    tools, and raw packet dumper.
  - `Encoder/` contains encoder tests.
  - `FSRs/` contains force-sensitive resistor tests.
  - `sensor_parse.py` is the combined sensor hub for the IMUs, FSRs, and
    encoder.
- `Data collection/` - Runtime sensor logging and recorded CSV data.
- `I2C tests/` - I2C bus discovery, speed, throughput, and diagnostic tools.
- `calibration scripts/` - Calibration workflows and offline validation scripts.
- `Cubemars_AK80-9/` - Cubemars AK80-9 motor/CAN tools and motor-related
  integration code.
- `TBE_controller/` - Controller, sensor acquisition, logging, and utility
  modules used by the TBE controller.
- `ML_model/` - Training, evaluation, deployment, model tests, and
  configuration for the machine-learning pipeline.
- `Sensor_transformation_script/` - Sensor-frame transformation data and
  related artifacts.
- `pawan/` - Experimental and rate-testing sensor scripts.
- `mrsd-exo-ankle/` - Downloaded/reference Georgia Tech dataset and subjects.
- `dashboard/` - Dashboard backend and related visualization code.
- `calibration/` - Calibration outputs, plots, and recorded calibration data.

## Important top-level files

- `exo_frame.py` - Sensor-frame transformation, calibration, and feature
  extraction utilities.
- `requirements.txt` - Python dependencies for the repository.

## Files moved so far

The following files were moved from the repository root or the former
`BNO085/` directory into more focused folders.

### Former `BNO085/` directory

Moved to `Sensor_Tests/IMUs/BNO085/`:

- `bno085_live.py`
- `bno085_live_dual.py`
- `bno085_live_dual_fast.py`
- `bno085_test_imu.py`
- `bno085_diagnostic.py`
- `bno_raw_dump.py`

Moved to `Sensor_Tests/`:

- `sensor_parse.py`
- `live_all_sensor_plot.py`

### Former repository-root sensor and I2C scripts

Moved to `Sensor_Tests/`:

- `test_encoder.py` -> `Sensor_Tests/Encoder/test_encoder.py`
- `test_fsr.py` -> `Sensor_Tests/FSRs/test_fsr.py`
- `imu_drain_test.py` -> `Sensor_Tests/IMUs/imu_drain_test.py`

Moved to `I2C tests/`:

- `i2c_bench.py`
- `i2c_buses.py`
- `i2c_bus_report.py`
- `i2c_speed_probe.py`

### Former repository-root calibration scripts

Moved to `calibration scripts/`:

- `sweep_check.py`
- `test_calibration.py`
- `test_exo_frame.py`
- `test_recording.py`
- `walk_check.py`

## Running scripts after the reorganization

Paths containing spaces should be quoted when used from a shell. For example:

```bash
python3 "Sensor_Tests/IMUs/BNO085/bno085_live.py"
python3 "Sensor_Tests/sensor_parse.py"
python3 "I2C tests/i2c_bus_report.py"
python3 "calibration scripts/test_exo_frame.py"
```

Hardware-dependent scripts require the Jetson, connected sensors or motors, and
the relevant third-party Python packages. The path handling in the moved
scripts is based on their file locations where applicable, so they do not
generally need to be launched from the repository root.
