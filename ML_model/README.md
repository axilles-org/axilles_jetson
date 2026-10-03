# MRSD Ankle Exoskeleton — Torque Prediction & Assistive Control

A causal temporal convolutional network (TCN) predicts the human ankle
plantarflexion moment from wearable IMUs, an ankle angle, and a ground-contact
flag. On the exoskeleton the predicted moment is scaled to a fraction and applied
as assistance during the stance phase of gait.

Training data: the Camargo et al. 2021 (Georgia Tech EPIC lab) lower-limb
biomechanics dataset, treadmill trials. Deployment target: NVIDIA Jetson driving
an AK80-9 actuator over CAN.

The workflow has five stages, each a thin CLI:

| # | Stage | Script | What it does |
|---|---|---|---|
| 1 | **Train** | `scripts/train.py` | fit the TCN on GaTech data |
| 2 | **Test** | `scripts/evaluate.py` | held-out per-subject / streaming metrics |
| 3 | **Validate on the exo** | `scripts/fit_adapter.py`, `scripts/run_deploy.py` | fit the sensor-frame transform, replay a real exo recording end to end |
| 4 | **Mock deployment** | `scripts/jetson_mock_deploy.py`, `scripts/plot_mock_run.py` | live Jetson sensors → predictions, **no torque** |
| 5 | **Powered deployment** | `scripts/jetson_deploy.py` | live sensors → CAN torque, ramped and capped |

---

## Setup

```
pip install -e ".[dev]"          # + [train] for wandb, [onnx], [deploy] for python-can
```

Or run scripts without installing: `PYTHONPATH=src python scripts/<name>.py`.

Interpreter for development on macOS:
`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3` (torch 2.9,
scikit-learn, scipy). Training runs on CUDA; the code stays runnable on MPS/CPU.

```
python -m pytest -q       # 62 tests
ruff check .              # lint (E, F, I, UP)
```

## Layout

```
configs/default.yaml     single source of truth (data, model, training, deploy)
src/exo/
  config.py              typed dataclass config tree, loaded from YAML
  data/                  HF hub download, raw ingest, scalers, feature pipeline,
                         windowed dataset, deployment domain-randomisation
  models/                TCN model + reusable layers
  training/              trainer, metrics, evaluation, subject embedding
  deploy/                sensor adapter, TorchScript/ONNX export, runtime loop,
                         assistance controller, AK80-9 motor, Jetson I/O
  viz/                   plots
scripts/                 one CLI per pipeline stage
tests/                   pytest suite
docs/DATASET.md          dataset schema + contribution guide
docs/SENSOR_ADAPTER.md   how the exo sensor-frame transform is fitted (+ maths)
docs/DEPLOYMENT.md       Jetson integration, the mock run, powered-deploy safety
docs/DOMAIN_RANDOMIZATION.md  training-time DR for latency / frame error / assist gap
```

## What the model predicts

Target `ankle_angle_r_moment` — right-ankle net plantarflexion/dorsiflexion
moment from OpenSim inverse dynamics, in **N·m**. Learned in z-scored space;
reported and deployed in **N·m/kg** (inverse z-score, then ÷ body mass). The
controller consumes the **last timestep** of each 3 s window — "the ankle moment
now, given the last 3 s of sensor history."

In the GaTech convention the plantarflexion (push-off) moment is **negative**;
`deploy.plantarflexion_sign` and `deploy.exo_command_sign` map it to the exo
actuator convention.

**Feature set (`mid`, 14 channels):** foot IMU 6-axis, shank IMU 6-axis, ankle
angle (rad), binary `stance`. `stance` comes from the dataset's gait-phase
signals during ingest and, on the exo, from the two foot FSRs — so training and
deployment inputs occupy the same 14 slots. `configs/default.yaml` also defines
`full` and `minimal` sets for experiments.

---

## 1. Train

### Data

The canonical dataset is a **Hugging Face dataset repo** of raw sensor Parquet
([`aicognition/mrsd-exo-ankle`](https://huggingface.co/datasets/aicognition/mrsd-exo-ankle),
`configs/default.yaml -> paths.hub`, CC BY 4.0). Feature engineering and z-scoring
happen locally and reproducibly; nothing pre-processed is hosted.

```
python scripts/prepare.py            # fetch snapshot + build the processed cache
python scripts/prepare.py --force    # rebuild the cache
python scripts/prepare.py --no-fetch # use the local dataset as-is
```

The processed `.npz` cache is keyed by (ingest config + dataset revision + code
version), so changing a setting transparently rebuilds it. `train.py` runs
`prepare` automatically when the cache is cold (`--no-fetch` to skip the network).

### Run training

```
python scripts/train.py                                   # full run, config defaults
python scripts/train.py --epochs 1 --no-wandb             # smoke test
python scripts/train.py --config configs/default.yaml --run-name tcn_mid_v2
```

| Flag | Meaning |
|---|---|
| `--config` | config YAML (default `configs/default.yaml`) |
| `--epochs` | override `train.epochs` |
| `--run-name` | output dir name under `runs/` |
| `--no-wandb` | disable Weights & Biases logging |
| `--no-fetch` | do not touch the network; use the local cache |

**Outputs** in `runs/<name>/`:

```
best.pt              best-val checkpoint (weights + subject embedding + norm stats)
config.yaml          the fully resolved config used for this run
metrics.json         train/val curves, aggregate test metrics
```

Key training choices (all in `configs/default.yaml -> train`):

- **Loss on the last `loss_last_n` timesteps** (default 50) — matches what
  deployment reads and stops the model memorising the low-receptive-field early
  window.
- **Per-subject embedding** + cosine cold-start for unseen users.
- **Subject-disjoint splits** (`data.subjects`): train / val / test share no
  subjects.
- **Performance**: AMP (bf16 on CUDA Ampere+), TF32, optional `torch.compile`,
  fused AdamW, memory-mapped dataset with a cached window index.
- **Domain randomisation** (`train.augment`): five knobs for the deployment gap —
  latency, IMU-frame error, encoder offset, stance-edge jitter, "assistance
  felt". All default **off**; see `docs/DOMAIN_RANDOMIZATION.md` for a
  recommended profile and the evaluation protocol.

### Benchmark

```
python scripts/bench.py --train                       # samples/s: eager vs amp vs compiled
python scripts/bench.py --infer --run runs/<name>     # per-window p50/p99 latency
```

The 100 Hz control loop budgets 10 ms/step; `--infer` asserts each backend is
under budget.

---

## 2. Test

`evaluate.py` reports held-out accuracy on the test subjects two ways: full-signal
reconstruction (per-subject and aggregate RMSE / MAE / normalised MAE / R² in
N·m/kg) **and** a streaming last-timestep-only pass that exactly mimics
deployment.

```
python scripts/evaluate.py --run runs/<name>                     # test split, with plots
python scripts/evaluate.py --run runs/<name> --split val --no-plots
python scripts/evaluate.py --run runs/<name> --checkpoint best.pt
```

| Flag | Meaning |
|---|---|
| `--run` | run directory to evaluate (required) |
| `--split` | `test` (default) / `val` / `train` |
| `--checkpoint` | checkpoint filename inside the run dir (default `best.pt`) |
| `--no-plots` | metrics only, no figures |

Architecture and normalisation stats are read from the checkpoint, so evaluation
can never drift from how the model was trained. Plots (predicted vs measured
moment, per-subject overlays) land in `runs/<name>/eval/`.

---

## 3. Validate on the exoskeleton

The model is trained on GaTech's IMU mounting and goniometer; the exo has its own
IMU frames, an encoder in degrees, and two FSRs. `SensorAdapter`
(`configs/default.yaml -> deploy.sensor_adapter`) bridges the two:

- per-IMU **rotation**, unit **scaling**, and a **lever-arm** correction on the
  accelerometer (an offset IMU senses extra centripetal / tangential terms — a
  plain rigid transform is not enough),
- ankle encoder degrees → radians with a neutral offset and sign,
- FSR counts → debounced binary stance via per-sensor thresholds.

### 3a. Fit the sensor-frame transform

```
python scripts/fit_adapter.py \
    --walk ~/Downloads/data_collection_20260404_215743_tightshoes_200Hz_2kmph.csv \
    --processed data/processed \
    --config configs/default.yaml --write-config --plot
```

| Flag | Meaning |
|---|---|
| `--walk` | a label-free exo walking recording (CSV) |
| `--processed` | processed GaTech `.npz` dir — supplies the reference bands |
| `--config` | config to read the current adapter block from |
| `--write-config` | patch `deploy.sensor_adapter` in the config in place |
| `--out` | output dir (default alongside the walk file) |
| `--rate` | control rate in Hz (default 100) |
| `--no-slow-match` | do **not** restrict the reference to GaTech's slowest trials |
| `--plot` | write a 4×4 channel panel (band vs adapted exo) |

The method — units from gravity, staged Wahba rotation (gravity + gyro swing
axis, iterated with the lever arm), multi-signal phase alignment, linear
least-squares lever arm, encoder offset/sign, FSR thresholds by stance-waveform
matching — is written up in
[`docs/SENSOR_ADAPTER.md`](docs/SENSOR_ADAPTER.md). It runs **once, offline**;
nothing is optimised at runtime.

Outputs: `sensor_adapter.yaml`, `.diagnostics.json`, `.png`. A fitted block from
the 2 km/h recording is already baked into `configs/default.yaml` — see the
speed caveat in that file; refit from a ~4 km/h walk plus a 5 s static hold
(`scripts/calibrate_exo.py`) during hardware bring-up to tighten every residual.

### 3b. Replay a real exo recording end to end

```
python scripts/run_deploy.py --run runs/<name> --mass 72 \
    --replay ~/Downloads/data_collection_20260404_215743_tightshoes_200Hz_2kmph.csv
```

Resamples the recording to the control rate and streams it through the full
`ExoController` (adapter → feature pipeline → rolling window → TCN → assistance
gate), printing the commanded torque and stance state each step. No labels exist
on the exo data, so this is a pipeline/plausibility check — it confirms the
prediction is periodic, gait-locked, plantarflexion in stance, near zero in
swing, and physiologically sized.

### Export for the Jetson

```
python scripts/export_jit.py --run runs/<name> --mass 72 --height 1.75 --gender M
python scripts/export_jit.py --run runs/<name> --mass 72 --onnx
```

| Flag | Meaning |
|---|---|
| `--run` | run directory to export (required) |
| `--mass` | deployment subject mass in kg (baked into the module) |
| `--height`, `--gender` | demographics for the subject-embedding cold-start |
| `--checkpoint` | checkpoint filename (default `best.pt`) |
| `--onnx` | also export an ONNX graph for a TensorRT build on the Jetson |

Produces `best.ts` (TorchScript) + `deploy_metadata.json`, optionally `best.onnx`.
**Both input and output scalers are baked into the module**, so the robot feeds
raw physical-unit windows and receives N·m/kg directly. TensorRT / INT8 engines
are built from the ONNX on the Jetson itself.

---

## 4. Mock deployment (no torque)

Run this on the Jetson before any powered test. It exercises the entire inference
path on real hardware — foot IMU + shank IMU + ankle encoder + heel/toe FSR →
`SensorAdapter` → feature pipeline → 3 s window → TCN → predicted torque — and
**opens no CAN bus**; the motor is never touched.

```
python scripts/jetson_mock_deploy.py --run runs/<name> --mass 72 \
    --axilles ~/axilles_jetson --duration 120 --out logs/mock_$(date +%s).csv
```

| Flag | Meaning |
|---|---|
| `--run` | exported run directory (needs `best.ts` + `deploy_metadata.json`) |
| `--mass` | subject mass in kg |
| `--axilles` | path to the team's `axilles_jetson` repo (for the IMU/FSR drivers) |
| `--replay` | bench dry-run from a `data_collection_*.csv` instead of live sensors |
| `--backend` | `jit` (default) / `onnx` / `trt` |
| `--rate` | control rate in Hz (default 100) |
| `--duration` | seconds to run (default: until Ctrl-C) |
| `--out` | log CSV path |
| `--no-teleplot` | do not stream to Teleplot (UDP 127.0.0.1:47269) |
| `--print-every` | console print decimation |

Each frame logs the raw sensor values, `predicted_nm_per_kg`, `predicted_nm`, and
`would_command_nm` (computed, **not** actuated), and streams live to Teleplot.

### Review the run

```
python scripts/plot_mock_run.py logs/mock_1234.csv --processed data/processed
```

Writes `logs/mock_1234.review.png` (4 panels: live trace with stance shading,
gait-cycle average vs the GaTech test band, prediction-vs-stance overlay, numeric
checks) and prints pass/fail verdicts:

| Check | Pass condition |
|---|---|
| Periodicity | cycle-to-cycle std / mean\|·\| below threshold |
| Swing vs stance | prediction near zero in swing, active in stance |
| Peak magnitude | push-off peak in ≈ [-1.5, -0.6] N·m/kg |
| Peak phase | peak at ≈ 45–60 % of the gait cycle |

Do not proceed to powered deployment until this review passes with a real trained
checkpoint. Full procedure and the pre-arming checklist: `docs/DEPLOYMENT.md`.

---

## 5. Powered deployment

Live sensors → `ExoController` → AK80-9 torque over CAN. **Torque is off unless
`--arm`.**

```
# bench, motor interface exercised without a CAN bus:
python scripts/jetson_deploy.py --run runs/<name> --mass 72 --dry-run --replay <csv>

# first powered session — conservative envelope, short:
python scripts/jetson_deploy.py --run runs/<name> --mass 72 --axilles ~/axilles_jetson \
    --arm --assist-scale 0.1 --torque-cap 2.0 --arm-ramp 5.0 --duration 60 \
    --command-sign 1 --out logs/powered_$(date +%s).csv
```

| Flag | Meaning | Default |
|---|---|---|
| `--run`, `--mass`, `--axilles` | as for the mock run | — |
| `--arm` | actually transmit torque (omit → computes only) | off |
| `--dry-run` | run `MotorInterface` with no CAN bus | off |
| `--assist-scale` | fraction of predicted human moment applied as assistance | — |
| `--torque-cap` | hard torque limit, N·m (clamped ≤ 5) | 2.0 |
| `--arm-ramp` | seconds to fade assistance in at session start | 5.0 |
| `--command-sign` | actuator sign convention (**verify on the bench**) | 1 |
| `--watchdog-ms` | loop-overrun budget; exceeded → torque zeroed that step | 25.0 |
| `--rate` | control rate in Hz | 100 |
| `--duration` | seconds to run | until Ctrl-C |
| `--backend` | `jit` / `onnx` / `trt` | jit |

**Four independent safety layers:**

1. `assist_scale` × session ramp (`--arm-ramp`) — soft start, small gain.
2. `--torque-cap` CLI clamp.
3. `MotorInterface` hard clamp to ±5 N·m (MIT mode), well under the ±18 N·m
   hardware max.
4. Loop-overrun watchdog — any step slower than `--watchdog-ms` commands zero
   torque; any exception or Ctrl-C zeros torque and exits MIT mode.

**Recommended progression** (`docs/DEPLOYMENT.md` §4):

1. `--dry-run` on the bench — confirm `--command-sign`, watch the log.
2. `--arm --assist-scale 0.02` — just enough to feel the ML torque *shape*, leg
   unloaded / on a stand.
3. `--arm --assist-scale 0.1 --torque-cap 2.0` — first walking session, short.
4. Raise `assist-scale` / `torque-cap` gradually, reviewing each log with
   `plot_mock_run.py`.

Log fields: `t_s`, all raw sensor channels, `predicted_nm_per_kg`,
`predicted_nm`, `assist_cmd_nm`, `sent_torque_nm`, `session_ramp`, `stance`,
`ramp`, `buffer_ready`, `loop_ms`.

---

## Relation to the team's TBE controller

The time-based estimator (`axilles_jetson/TBE_controller/`) replays a fixed
torque-vs-phase profile keyed to stride time. This repo's runners borrow only its
sensor drivers (`FastDualIMUReader`, `SensorData`) for Jetson I/O; torque is
commanded here through `src/exo/deploy/motor.py`. The two controllers are
independent — see `docs/DEPLOYMENT.md` §1.

## Dataset maintenance

```
scripts/csv_to_parquet.py     raw CSV  -> Parquet
scripts/upload_dataset.py     push Parquet to the HF repo
scripts/ingest_raw.py         low-level raw-trial ingest (prepare.py wraps this)
```

Schema and contribution guide: [`docs/DATASET.md`](docs/DATASET.md).
