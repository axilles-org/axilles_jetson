# Exo Dashboard

Trial tracking and live telemetry for both controllers in this repo — the
fixed torque-profile controller (`TBE_controller/main.py`) and the ML/TCN
controller (`ML_model/scripts/jetson_deploy.py`). Every run of either
controller is recorded as a **run**: full-rate sensor/command traces on disk,
a metadata row (model, config, status) in a local database, and — while it's
running — a live feed to a browser.

It also has a **Launch** page that starts controller processes for you
(model picker, drag-and-drop replay CSV, live console) instead of typing CLI
flags by hand. It only ever launches the non-torque modes — see
[Safety](#safety) below.

## Architecture

```
controller process (TBE_controller/main.py or ML_model/scripts/jetson_deploy.py)
  │  each control-loop iteration calls run_logger.log_sample(...) — a
  │  non-blocking in-memory append, never disk/network I/O in the hot loop
  ├─→ background thread  → Parquet files + SQLite row   (dashboard/data/)
  └─→ UDP, fire-and-forget → dashboard backend → WebSocket → browser (live view)

dashboard/backend  (FastAPI)
  /api/runs, /api/runs/{id}, /api/runs/{id}/data   history + full traces
  /api/runs/{id}/diff/{id}                          architecture/config diff between two runs
  /api/runs/{id}/export, /api/runs/import           zip a run for another machine, or pull one in
  /api/models                                        checkpoints under ML_model/runs/ + TBE
  /api/samples                                        bundled sample CSVs (dashboard/backend/sample_data/)
  /api/uploads/replay                                drag-and-drop a data_collection_*.csv
  /api/launch, /api/launch/status, /api/launch/stop  start/stop/poll a controller subprocess
  /ws/live                                            live sample stream, fanned out to all connected browsers

dashboard/frontend  (React + Vite + uPlot)
  Live    real-time charts for whichever run is currently streaming
  Launch  pick a model, optionally pick a bundled sample or drop your own replay CSV, start/stop, watch the console
  Runs    history table (filter/search), compare two runs' architecture, import/export
```

A run's `run_id` is the join key across all of it: the SQLite row, the
Parquet folder name, and every live UDP/WebSocket packet.

## Running it

```bash
# backend
pip install -r dashboard/backend/requirements.txt
python -m uvicorn dashboard.backend.server:app --host 0.0.0.0 --port 8000

# frontend (separate terminal)
cd dashboard/frontend
npm install
npm run dev            # http://localhost:5173, proxies /api and /ws to :8000
```

`vite.config.js` binds to `0.0.0.0`, so `http://<jetson-ip>:5173` works from
another machine on the same lab network — open it from a laptop while the
Jetson runs the controller.

To see it with fake data (no hardware, no checkpoint needed):

```bash
python dashboard/backend/simulate_run.py --duration 30
```

## What gets recorded

Every sample logged by a controller carries whatever fields that
controller's loop passes to `run_logger.log_sample(t=..., **fields)` — the
Live page groups fields into chart panels by name (`ankle_angle` /
`ankle_encoder_deg`, `torque_cmd` / `sent_torque_nm`, etc.) so it adapts to
either controller's schema instead of assuming one.

Every run's metadata also carries a `ModelInfo`:

| field | where it comes from |
|---|---|
| `architecture` | `ML_model/runs/<run>/model_meta.json` + `deploy_metadata.json` (TCN layer sizes, features, window length) — or, for TBE, its fixed torque-profile parameters, so both controllers are diffable |
| `architecture_hash` | SHA-256 of `architecture`, truncated — two runs with the same hash used the identical config |
| `wandb_run_url` | `ML_model/runs/<run>/wandb_run.json`, written automatically by `Trainer.fit()` right after `wandb.init()`. Checkpoints trained before this existed (e.g. `tcn_mid_stance_lastN_20260831_212705`) won't have one. |
| `checkpoint_ref` | path to the run directory the weights came from |

`GET /api/runs/{a}/diff/{b}` compares two runs' `architecture` and trial
`meta` key-by-key — use it to see exactly what changed between two trials
(e.g. `assistance_scale: 0.1 → 0.15`) without opening either checkpoint by
hand.

**Model display names**: `/api/models` shows a checkpoint's raw folder name
(a training timestamp, e.g. `tcn_mid_stance_lastN_20260831_212705`) unless
that checkpoint has a `display_name.txt`. Give it a real name with:
```bash
echo "Ankle Assist TCN v1" > ML_model/runs/<checkpoint>/display_name.txt
```

## Safety

**The dashboard can never send torque.** `dashboard/backend/process_manager.py`
raises before launching anything if `--arm` is in the argv, and the two
modes it's able to build are:

- `ml-mock` → `jetson_mock_deploy.py` — no `--arm` flag exists on this
  script at all; torque transmission isn't in the code path.
- `ml-dry-run` → `jetson_deploy.py --dry-run` (never `--arm`) — exercises
  the full command path (ramp, clamp, MIT packing) through `MotorInterface`
  with `dry_run=True`, so nothing reaches a CAN bus, but the run gets the
  full `ModelInfo`/telemetry wiring that only `jetson_deploy.py` has.

A **powered** (`--arm`) run stays CLI-only, same as before this dashboard
existed — see `ML_model/docs/DEPLOYMENT.md` for that progression and its
four safety layers. It can still be tracked here: run it from the terminal
as usual, and `RunLogger` records it the same way.

Only one job runs at a time — launching a second while one is active is
rejected outright, not queued or force-stopped, so a trial in progress can't
be silently interrupted by an unrelated click. `Stop` sends `SIGINT` first
(the controller scripts already zero torque and close resources on
`KeyboardInterrupt`), escalating to `SIGTERM`/`SIGKILL` only if that doesn't
land within a few seconds.

## Known gaps

- `ML_model/src/exo/data/` isn't committed on this branch (or on
  `ML_control`/`main`) — without it, `ExoController` can't import and
  neither deploy script will run. Worth fixing at the source; in the
  meantime, copy it in from wherever your team's local checkout has it.
- `jetson_mock_deploy.py` has no `RunLogger` wiring, so mock runs show up
  live in the Launch console but not in the Runs history or the Live page —
  use `ml-dry-run` mode instead if you want a run recorded.
