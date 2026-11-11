# Deploying the ML controller on the exo (Jetson)

This describes how the ML ankle-moment predictor plugs into the team's Jetson
stack (`axilles_jetson`), and the **mock deployment** — a no-torque run that
validates the whole inference path on real hardware before any powered test.

---

## Subject-conditioned FiLM in the dashboard

On **Launch**, select `model_film_subject_embedding_current_action_terrain`.
This is the five-epoch subject-FiLM checkpoint from
`runs/terrain/20261029_011205/film`, with a 300-sample, 14-channel history and
current-time output. It is labelled **Awaiting hardware testing**. Context-FiLM
and MoE remain outside the dashboard catalog.

The package includes the original weights, scalers, training demographics,
ankle calibration reference, TorchScript and ONNX exports. Enter the wearer's
height, mass and gender to select the nearest training embedding and re-export
before launch. ONNX runs on CPU; TensorRT is built on the Jetson when its
dependencies are installed. TensorRT has not been validated on this Mac.

This older experiment trained in **GaTech coordinates**, using noncausal offline
preprocessing, rather than the newer transformed-data timing-augmentation setup.
Its original exo-to-GaTech sensor adapter is retained. Do not replace it with
the identity adapter from `runs/deploy`. The package defaults to assistance scale
0.02 and a 2 N·m cap; dashboard launch settings are explicit overrides.

The recorded pooled last-timestep test scores with nearest-subject embeddings
are prediction R² **0.859834** and RMSE **0.188815 N·m/kg**. These describe offline
prediction, not hardware assistance performance. The catalog displays these
streaming scores rather than the separate trainer test score.

To recreate the package in a fresh destination, run from `ML_model`:

```bash
python scripts/package_film.py --destination /tmp/film-packages
```

Repackaging needs the source training data to compute its training-only ankle
reference. Launching the committed package does not require that dataset.

---

## 1. How it relates to the existing TBE controller

The Jetson repo `axilles_jetson` runs a **time-based estimator (TBE)**: it detects
heel strike from the heel FSR, estimates the stride time, and replays a fixed
6-point ankle torque profile (`TAU_PHASE_ARRAY` / `TAU_VAL_ARRAY` × `PEAK_TORQUE`)
scaled by `ASSISTANCE_LEVEL`, sending it to the AK80-9 motor over CAN in MIT mode.

The ML controller **predicts** the wearer's ankle plantarflexion moment from a
2.5 s window of foot + shank IMU, ankle angle, and a contact flag, then applies a
fraction of it as assistance during stance. Same actuator, same FSR/encoder
hardware, different torque source.

We integrate as a **standalone runner** in this repo, not by editing the TBE
files. It borrows only the hardware-I/O classes from `axilles_jetson`:

| Borrowed from `axilles_jetson` | Used for |
|---|---|
| `BNO085/bno085_live_dual_fast.py` → `FastDualIMUReader` | foot IMU (0x4A) + shank IMU (0x4B), accel m/s², gyro rad/s |
| `TBE_controller/data_obtainer.py` → `SensorData` | ADS1115 FSRs (heel/toe, low-pass filtered), AS5600 encoder (deg) |

Wrapped in `src/exo/deploy/jetson_io.py` (`JetsonSensors`). `SensorData` also
opens a CAN bus; the **mock** runner never actuates it, and the **powered** runner
uses its own `src/exo/deploy/motor.py` (`MotorInterface`) instead, so torque
handling stays in this repo.

### Sensor mapping

The exo raw frame the `SensorAdapter` expects (`foot_ax..foot_gz`,
`shank_ax..shank_gz`, `ankle_encoder_deg`, `heel_fsr_raw`, `toe_fsr_raw`) is
exactly `FastDualIMUReader` IMU-A + IMU-B + `SensorData.encoder_data` +
`SensorData.filtered_heel_fsr` / `filtered_toe_fsr`. This is the same layout as
the `data_collection_*.csv` files the adapter was fitted from.

The `SensorAdapter` (see [`SENSOR_ADAPTER.md`](SENSOR_ADAPTER.md)) then converts
that frame into the training convention: per-IMU rotation + scale + lever-arm
correction, encoder degrees → radians, FSR counts → debounced binary stance.

### Two independent stance gates (by design)

1. Inside the model input — the `stance` feature, from `SensorAdapter` FSR
   thresholds. This is a *context cue* for the network.
2. Inside `AssistanceController` — a separate `StanceDetector` on the raw FSRs,
   gating whether assistance is applied at all.

A small FSR-timing error degrades gracefully rather than causing a bad command.

---

## 2. Mock deployment — live sensors, **zero torque**

```bash
# on the Jetson
python scripts/jetson_mock_deploy.py \
    --run     runs/tcn_mid_stance_lastN_20260831_212705 \
    --mass    <wearer kg> \
    --axilles ~/axilles_jetson \
    --backend onnx \
    --duration 120 \
    --out     logs/mock_run.csv
```

What it does each control step (default 100 Hz):

1. read foot IMU, shank IMU, encoder, heel/toe FSR;
2. `ExoController.step(frame, dt)` → SensorAdapter → FeaturePipeline → rolling
   window → TCN → predicted moment (N·m/kg);
3. the `AssistanceController` is run too (its stance gate, ramp, torque/rate
   limits are exercised) but its output is **only logged** — labelled
   `would_command_nm`;
4. every frame → a CSV row and a Teleplot UDP packet.

The first ~3 s produce no prediction (`buffer_ready = 0`) while the window fills —
walk during that time.

### Live view

Point Teleplot at UDP `127.0.0.1:47269` (same as TBE). Streams:
`ml_pred_nm_per_kg`, `ml_pred_nm`, `ml_would_cmd_nm`, `stance`, `heel_fsr`,
`toe_fsr`, `ankle_deg`.

---

## 3. Reviewing accuracy

There are **no torque labels on the exo**, so we cannot score RMSE directly.
Instead we verify the properties that must hold if the pipeline is correct:

```bash
python scripts/plot_mock_run.py logs/mock_run.csv \
    --processed /path/to/exo-data/processed     # optional GaTech reference band
```

Checks, with pass criteria:

| Check | Expected |
|---|---|
| Prediction is periodic, locked to the gait cycle | `std / mean\|·\|` ≳ 0.8 |
| Near zero in swing | `mean\|pred\|` swing ≪ stance (ratio < 0.3) |
| Negative (plantarflexion) through stance | `mean(pred \| stance)` < −0.2 N·m/kg |
| Peak magnitude physiologically plausible | peak in ≈ [−1.5, −0.6] N·m/kg |
| Cycle-average peak timing | ≈ 45–60 % of the gait cycle |
| Cycle-average curve | inside the GaTech held-out band |

The script prints these and writes `mock_run.review.png` (4 panels: live trace,
gait-cycle average vs band, prediction-vs-stance overlay, numeric verdicts).

**Offline dry-run:** the same `ExoController` path can be replayed from a recorded
`data_collection_*.csv` on any machine (see `scripts/run_deploy.py`) before taking
it to the Jetson.

---

## 4. Powered deployment — `scripts/jetson_deploy.py`

The powered runner is the mock runner plus a CAN torque command. It is
**self-contained in this repo** — `src/exo/deploy/motor.py` speaks the AK80-9 MIT
protocol directly (no dependency on `axilles_jetson` for actuation).

### Safety layers (six, in order)

1. **`AssistanceController.torque_limit_nm`** (config) — the controller's own cap.
   A NaN or infinite model output turns assistance off until reset.
2. **`--torque-cap`** (CLI, default 2.0) — a second ceiling, above 0 and ≤ 5 N·m.
3. **`MotorInterface`** hard-clamps every command to `min(cap, SAFE_T_MAX=5.0)`
   before packing — the last line. A non-finite command is refused.
4. **Watchdog** — if a control loop takes longer than `--watchdog-ms` (25),
   that frame's torque is zeroed.
5. **Motor thread** — sends the torque frames. If the loop gives it no target for
   `--sender-stale-ms` (100), it sends 0 N·m and ends the run, so a hung sensor
   read or inference does not leave torque on.
6. **Status check** — the same thread reads status frames and ends the run on a
   motor error code, `|Iq|` above `max(2 A, 1.5 × cap / Kt)`, or no status for
   100 ms. This is the only current limit we can apply without CubeMarsTool.

Plus: torque is **off unless `--arm`**; a session ramp (`--arm-ramp`, default 5 s)
fades assistance 0 → full after arming; `Ctrl-C` / `SIGTERM` / any exception →
zero torque.

These do not cover the whole process dying or the CAN link failing. The motor then
holds its last torque until its own CAN timeout (about 3 s in the saved Test 4
frames), so keep the physical e-stop in reach.

### Progression

```bash
# a) bench dry-run from a recording — MotorInterface runs, no CAN, no torque.
#    Verifies the ramp, clamp, watchdog and packing.
python scripts/jetson_deploy.py --run runs/<dir> --mass 72 \
    --replay ~/Downloads/data_collection_...csv --dry-run --arm \
    --assist-scale 0.1 --torque-cap 2.0 --arm-ramp 2

# b) live on the Jetson, still NO torque (equivalent to the mock)
python scripts/jetson_deploy.py --run runs/<dir> --mass 72 --axilles ~/axilles_jetson

# c) live, POWERED — assistance ramps in over 5 s, hard-capped at 2 N·m
python scripts/jetson_deploy.py --run runs/<dir> --mass 72 --axilles ~/axilles_jetson \
    --arm --assist-scale 0.1 --torque-cap 2.0
```

The log (`logs/powered_run.csv`) records `assist_cmd_nm` (what the controller
wanted), `sent_torque_nm` (the queued motor target, after ramp + clamp; not measured torque),
`session_ramp`, and `loop_ms` per frame. Teleplot streams the same.

### Tuning the envelope

Commands are capped at **±5 N·m** (`SAFE_T_MAX`) inside the motor's ±22 N·m MIT
encoding range; TBE delivers ≈ 0.7 N·m effective. Starting point: `--assist-scale 0.1 --torque-cap 2.0`.

Note: at `assist-scale 0.1` the raw command at push-off (~9–10 N·m for a 72 kg
wearer) is far above a 2 N·m cap, so the delivered torque is a **flat-topped 2 N·m
block** through push-off, not a scaled copy of the prediction. To let the ML
*shape* show through, drop `--assist-scale` to ≈ 0.02 (→ ~1.9 N·m peak, just under
the cap). Decide on the bench.

### Before arming

1. **`--command-sign`** — verify which sign assists plantarflexion on your build.
   The TBE repo negates torque in `sendTorqueData` and once swapped cable
   polarity. Bench-test with a tiny constant command first.
2. **Refit the SensorAdapter** if a ~4 km/h exo walk and a static-hold recording
   are available — [`SENSOR_ADAPTER.md`](SENSOR_ADAPTER.md) §6.
3. **Mock run passes review** (§3) with a real trained checkpoint.

---

## 5. File map

| File | Role |
|---|---|
| `scripts/jetson_mock_deploy.py` | live-sensor, no-torque runner (§2) |
| `scripts/jetson_deploy.py` | **powered runner** — CAN torque with 4 safety layers (§4) |
| `scripts/plot_mock_run.py` | accuracy review of a mock/powered log (§3) |
| `scripts/run_deploy.py` | offline CSV replay through `ExoController` |
| `src/exo/deploy/runtime.py` | `ExoController`, `InferenceBackend`, `ObservationBuffer` |
| `src/exo/deploy/sensor_adapter.py` | exo frame → training convention |
| `src/exo/deploy/assistance.py` | stance gate, ramp, torque/rate limits |
| `src/exo/deploy/motor.py` | `MotorInterface` — AK80-9 MIT-mode CAN, torque clamp |
| `src/exo/deploy/jetson_io.py` | `JetsonSensors`, `ReplaySensors`, `Teleplot` |
| `src/exo/deploy/export.py` | `best.ts` / `best.onnx` with both scalers baked in |


## November 2 recording and bidirectional stance assistance

`runs/deploy/config.yaml` now selects `assistance_mode: bidirectional` with
`assistance_scale: 0.02`. Other configurations retain plantarflexion-only behavior
unless explicitly changed. CLI overrides are `--assistance-mode bidirectional`,
`--dorsi-scale-ratio` (relative to the assistance scale), and `--dorsi-cap`.
The configured dorsiflexion cap is 2 Nm; the final motor cap still applies.

With `plantarflexion_sign: -1`, negative predicted biological moment requests
plantarflexion and positive moment requests dorsiflexion. The motor mapping
(`exo_command_sign` and `--command-sign`) must be checked physically in both
directions. Direction selection uses predicted moment, not ankle angle.
The slew limiter smooths direction changes. Both directions use the same stance
flag as the model input. Swing, disabled assistance, warm-up, model faults and
watchdog overruns zero the final target without a slew tail. This does not enable
swing dorsiflexion or guarantee instantaneous cessation of physical torque.

Sensor report ages are diagnostic only. The previous 100 ms stale-report stop
and 2 s missing-report startup stop have been removed. Late or missing accel,
gyro, FSR or encoder reports retain their last values (zero before the first
report) without stopping the run or introducing an assistance gate. Acquisition
thread failures still stop the run. Receipt counters distinguish identical new
readings from missing reports; continued operation does not establish freshness.

New log fields separate sensor-read, preprocessing and inference time; include
per-IMU stream ages, raw and model-input stance, watchdog state, prediction and
command queue timestamps, latest CAN send completion and motor status age.
`last_can_tx_*` describes the latest completed bus send, which may correspond to
an earlier loop target. It is not a measured actuator response timestamp.

The imported source is archived under `runs/imported/backup_ml_control_model_4745a55`
with a checksum manifest. The latest CSV is available in the dashboard as
`20261102_182941_1d037a`; its run ID association is inferred from the unique
8,446-row match in the source database. The other November 2 database records
are archived, but their unavailable raw Parquet streams have not been fabricated.
Existing local recordings and the local database were preserved.

Reproduce the analysis without hardware, from `ML_model/`:

```sh
python3 analysis/analyze_assistance_delay.py
```

Results are in `runs/hw_analysis/20261102_delay/`. The command counterfactual
reuses recorded predictions to test routing; it intentionally bypasses sensor
freshness offline and is not a validation of assistance on a wearer. The old CSV
cannot establish whether identical values were accompanied by fresh reports.
