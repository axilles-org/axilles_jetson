# Fixed-horizon ankle-moment prediction

The three experiments use the deployed embedding TCN architecture and transformed
IMU dataset. Each receives a 300-sample history ending at observation time `t`.
Its last output estimates **biological ankle moment at `t + horizon`**. These are
moment forecasts, not learned actuator actions. No future stance or sensor samples
are included in the input window.

| Config | Horizon at 100 Hz | Target for input ending at sample k |
|---|---:|---|
| `configs/anticipation/tcn_50ms.yaml` | 5 samples | `moment[k + 5]` |
| `configs/anticipation/tcn_100ms.yaml` | 10 samples | `moment[k + 10]` |
| `configs/anticipation/tcn_150ms.yaml` | 15 samples | `moment[k + 15]` |

The configs inherit shared settings from `base.yaml`. `extends` paths are relative
to the YAML file; dataset/run paths are relative to `ML_model`. Every run saves a
fully resolved `config.yaml`, so it does not depend on later edits to the base.
Change `paths.dataset_dir` in the base to use a different local dataset.

## Training

From `ML_model`:

```sh
python3 scripts/train_anticipation.py --parallel
```

This trains exactly three models in separate processes, using five epochs, seed 0, the same
13/3/6 subject split and the existing augmentation settings. Omit `--parallel`
to run sequentially. Each model has its own `training.log` and `status.json`;
the prepared dataset is shared. Random latency,
time warping and stance jitter are disabled so each configured horizon has a
fixed meaning. The 50-sample training loss compares each output at input time `j`
with the label at `j+h`. The model architecture is causal within the input window.
At validation and test time the same label shift applies, without augmentation.

The processed cache is separate from existing experiments. Input IMU, goniometer
and optional force streams use one-pass anti-alias filtering, avoiding the
lookahead in the previous zero-phase input filter. Offline reference moments
retain their existing filtering. Scalers are fitted only on training subjects.
The source dataset may itself have preprocessing; this change guarantees no
lookahead in this repository's input decimation, not in undocumented upstream
processing. A causal filter also introduces its own delay, so these results are
not directly interchangeable with earlier zero-phase preprocessing results.

Windows remain within a single trial. Shifted labels must be valid, input windows
must be finite, and windows without enough future samples are dropped. For
partially labelled terrain data, `valid_tail` is checked at future target times.
Cache keys distinguish horizons. Milliseconds must map to an integer number of
samples at the declared sampling rate.

Results go under a new `runs/anticipation/<UTC timestamp>/` directory:

- `status.json`: preparing, training, evaluating, completed, or failed, with errors.
- `model_subject_embedding_future_action_<h>/config.yaml`: complete run settings
  (older experiments retain their original `tcn_<h>ms` directories).
- `best.pt`: checkpoint selected using validation loss, not test performance.
- `history.jsonl`: epoch losses, physical-unit errors and duration.
- `forecast_metrics.json`: held-out scores, including last-timestep streaming R².
- `summary.json`: completed horizon results.

The evaluator places an output made at `t` at its target timestamp `t+h` before
comparing it with reference torque. Streaming scores therefore measure future
prediction, not an artificial improvement from shifting curves after evaluation.
Exported model metadata and dashboard model information retain the horizon.
The generic trainer's `test_metrics.json` reports the configured loss-tail view;
use `forecast_metrics.json` for the endpoint/streaming comparison.

## Interpretation and deployment

These runs test whether forecasting helps. They do not establish that model delay
caused the wearer's late assistance: the November 2 recording also contained
constant IMU values throughout assistance. Fix and verify sensor freshness first.

The runner never copies a checkpoint to `runs/deploy`, arms a motor, or changes
online gating. Before using a forecast on hardware, reproduce its input filtering
and sampling rate and verify both torque directions. Using a future moment now is
a separate control-policy choice: present stance does not guarantee stance at
the forecast time, particularly near toe-off. Evaluate timing and torque at the
actuator before treating a 50/100/150 ms model horizon as the same amount of
physical assistance advance. The new timing-augmented suite includes a
zero-horizon model with the same causal preprocessing. The older three forecasts
used standard noise/gain augmentation; the new suite disables both, so differences
between suites cannot be attributed solely to adding timing augmentation.

## Timing-augmented suite (0, 50, 100, 150 ms)

`obs_aug_base.yaml` specifies timing-only observation corruption. Four leaf
configs named `model_obs_aug_subject_embedding_*.yaml` select the prediction
horizon. Run from `ML_model`:

```sh
python3 scripts/train_anticipation.py --parallel --device mps --configs configs/anticipation/model_obs_aug_*.yaml
```

| Model name | Target |
|---|---|
| `model_obs_aug_subject_embedding_current_action` | moment at t |
| `model_obs_aug_subject_embedding_future_action_50` | moment at t + 50 ms |
| `model_obs_aug_subject_embedding_future_action_100` | moment at t + 100 ms |
| `model_obs_aug_subject_embedding_future_action_150` | moment at t + 150 ms |

The word `action` is a naming convention: all models predict biological ankle
moment, not a learned actuator action. No controller delay is added by selecting
these names. Previous forecast packages are named
`model_subject_embedding_future_action_50`, `_100`, and `_150`.

For 75% of training windows, each IMU acceleration report, gyroscope report, and
ankle encoder receives an independent acquisition schedule. Axes within a report
share that schedule. Mean acquisition rates are sampled from 50--100 Hz, interval
jitter is up to 20%, and per-stream transport delay is 0--20 ms. Individual packet
loss is 2%; each stream has a 25% chance of one 20--80 ms arrival blackout. The
remaining 25% of windows are clean. These are initial experimental settings,
not ranges fitted to the hardware recording.

At each 100 Hz decision time, only the latest arrived observation is available.
Missed updates hold the previous measurement. The first input sample initialises
history; no samples wrap around from the end. Source times are rounded down to
the cached 100 Hz grid, so this does not simulate information above 100 Hz.
Targets and prediction horizons remain on the original clock. Stance is untouched.
The network still takes 14 channels; no age or missingness channels are added.

Rotation, offsets, stance-edge perturbation, clipping, quantisation, bias/drift,
gain errors, additive noise, generic time warping and `assist_perturb` are disabled
in this suite. Validation/test sets remain clean and use the original subject
split. The simulator models acquisition variability at a fixed decision rate;
it does not establish robustness to a control loop whose entire time base drifts.
An 80 ms blackout can yield older observations once acquisition and transport
delay are included. Live report ages are diagnostic only: late or missing reports
retain their last values without ending the run. Training success does not
establish hardware safety or benefit.

## Dashboard packages and GitHub

After an experiment completes, from the repository root:

```sh
python3 ML_model/scripts/package_anticipation.py --experiment ML_model/runs/anticipation/EXPERIMENT
```

This creates named folders directly under `ML_model/runs`, which the dashboard
Model selector lists for the selected subject-embedding family. Each contains weights, scalers, training-subject
metadata for embedding selection, ONNX and TorchScript, evaluation results,
resolved configs, and checksums. Packaging preserves the original training run.
`training_config.yaml` records original settings; `config.yaml` uses the current
`runs/deploy/config.yaml` hardware adapter and assistance settings. This matters
because transformed-data models require the identity frame mapping, rather than
the generic config's older fitted exo-to-GaTech transform.

The pre-exported preview uses 72 kg / 1.75 m / M. Live launch re-exports for the
wearer entered in the dashboard. Selecting a model does not start a process;
existing passive-start, assistance gating and torque caps remain in effect.
Model details show prediction horizon, timing augmentation and streaming test
RMSE. Input filtering/rate compatibility with live hardware still needs testing;
these packages do not add a new online filter or restore the removed asynchronous
inference-rate change.

### Jetson CPU/GPU selection

The dashboard's **Inference backend** menu offers **ONNX Runtime (CPU)** and
**TensorRT (GPU)**, plus TorchScript. TensorRT is selectable when the selected
package has ONNX and the dashboard's Python environment has TensorRT and PyCUDA.
Otherwise the option is disabled with the missing requirement shown. Dependencies
must be installed in the same Python environment used by the controller process.

For TensorRT, all launch modes first export for the entered wearer, then build a
local FP32 engine and compare its output with ONNX on three input probes. A
mismatch aborts before controller/motor setup. A successful engine is cached by
ONNX hash, TensorRT/CUDA versions and GPU/platform identity, with a separate
engine checksum. Model or wearer changes invalidate the cache. The first launch
can therefore take longer. Engines are not committed to GitHub; build them on the
target Jetson. This follows NVIDIA's named-I/O execution path:
https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html

The development Mac cannot execute TensorRT. Automated tests cover export order,
cache invalidation, failed-build behavior and dashboard argument routing; GPU
numerical validation runs on the Jetson at engine creation. This is not a claim
of measured Jetson latency or powered-trial validation.

Commit the portable model folders plus source/configuration/results, not the
processed dataset cache or unrelated local recordings. The package manifest
records the original run and checkpoint hash; the experiment source manifest
records training-source hashes and complete resolved configs.

## Paired timing stress check

`scripts/evaluate_timing_robustness.py --runs <package directories> --output <new JSON>`
compares clean inputs with timing-corrupted inputs at sampled held-out window
endpoints. It uses every tenth evaluation window by default, seed 17, and forces
the configured timing corruption on every sampled window. Each trial/endpoint
receives the same corruption across models. Results include both RMSE and
prediction R-squared, sample counts and the exact profile. Labels remain unchanged.
This is an offline window stress test, not continuous replay, and does not test
live sensor availability. These results are separate from the full clean
`forecast_metrics.json` and must not be substituted for its scores.

## Completed November 4 experiment

Four timing-augmented models completed five epochs, seed 0, on transformed
treadmill data. Full held-out endpoint scores are:

| Prediction horizon | Prediction R-squared | RMSE (N m/kg) |
|---|---:|---:|
| Current time | 0.9026 | 0.1609 |
| 50 ms | 0.8910 | 0.1702 |
| 100 ms | 0.8892 | 0.1717 |
| 150 ms | 0.8772 | 0.1808 |

In the paired sampled-window timing stress check, the augmented 50/100/150 ms
models scored corrupted-input RMSE 0.1824/0.1720/0.1847, compared with
0.2105/0.2019/0.2032 for the earlier forecasts. The current-time augmented model
scored 0.1649. These are single-seed offline comparisons; earlier models also used
noise/gain augmentation, unlike the timing-only suite. They do not isolate the
effect of one augmentation or validate physical assistance timing.

Full records: `runs/anticipation/20261104_200602_UTC_obs_aug/`, including
`comparison.csv`, `timing_stress_metrics.json` and `package_verification.json`.
All seven named packages passed checksum, dashboard discovery and CPU
ONNX/TorchScript replay checks. Actual TensorRT execution remains to be checked
on the Jetson; local tests verified its preparation and engine-cache logic.

## Dashboard catalog scope

The ML launch catalog is explicitly limited to eight subject-embedding TCNs:
`model_subject_embedding_current_action` (the existing `runs/deploy` baseline),
the three `model_subject_embedding_future_action_50/100/150` packages, and the
four `model_obs_aug_subject_embedding_current_action/future_action_50/100/150`
packages. TBE remains a separate controller choice.

FiLM, MoE, plain TCN, terrain models, older duplicate exports and sensor ablations
are excluded from the menu. Adding an unrelated export does not expose it for
hardware launch; the explicit list is in `dashboard/backend/model_catalog.py`.
The seven new variants are labelled "Awaiting hardware testing". Only the
baseline is labelled "Previously deployed baseline"; that does not claim its
hardware issues have been resolved. ONNX CPU and TensorRT GPU selection remain
available for this family.

After pulling the branch on the Jetson, rebuild `dashboard/frontend` with
`npm ci && npm run build` and restart the dashboard backend.
