# Exo Dashboard

Live telemetry and trial history for both controllers in this repo —
`TBE_controller/main.py` (fixed torque profile) and
`ML_model/scripts/jetson_deploy.py` (TCN). Every run is recorded (full-rate
traces + config) and streamed to a browser while it's in progress. A
**Launch** page starts either controller without touching the CLI; a live ML
run starts passive and assistance is switched on from the **Live** page.

## Quickstart

```bash
# backend
pip install -r dashboard/backend/requirements.txt
python -m uvicorn dashboard.backend.server:app --host 0.0.0.0 --port 8000

# frontend, separate terminal
cd dashboard/frontend && npm install && npm run dev
```

Open **http://localhost:5173**. Binds to `0.0.0.0`, so it also works from
another machine on the lab network via `http://<jetson-ip>:5173`.

No hardware or checkpoint needed to try it — generate a fake trial with:
```bash
python dashboard/backend/simulate_run.py --duration 30
```

## Pages

| Page | What it does |
|---|---|
| **Live** | Real-time charts for whichever trial is currently streaming; Assist ON/OFF + scale for a live ML run |
| **Launch** | Pick a model, wearer (mass, height, sex), assistance scale, torque cap and mode; start/stop, watch the console |
| **Runs** | History table, filter/search, diff two runs' model config, import/export |
| **Experiments** | GaTech → exo transform verification (metrics + stride overlays) and transform-ablation test results |

## Files

```
backend/
  server.py            FastAPI app — history API, live WebSocket relay, launch/stop, model + sample listing
  run_logger.py         Recorder wired into each controller's loop (non-blocking; writes Parquet + SQLite, relays live over UDP)
  experiments.py        Read-only API over ML_model/runs/{transform_verification,ablation}
  process_manager.py    Launches/stops a controller subprocess for the Launch page; blocks any --arm (torque) invocation
  simulate_run.py        Fake-trial generator, for trying the dashboard with no hardware
  sample_data/            Bundled real exo CSVs, usable as a replay source (see below)

frontend/src/
  pages/LivePage.jsx       Live charts, auto-adapts to whichever fields the active controller logs
  pages/LaunchPage.jsx     Model picker, replay source (sample/upload), start/stop, live console
  pages/RunsPage.jsx       Run history, filters, compare, import
  pages/RunDetailPage.jsx  One run's full traces + model card
  pages/ExperimentsPage.jsx Transform verification + ablation results
  components/              Shared pieces: chart, drag-and-drop zone, diff view
  lib/api.js                Backend API client
```

`ML_model/src/exo/deploy/dashboard_info.py` (outside this folder) builds
each run's model metadata from its checkpoint directory; shared by both
`jetson_deploy.py` and `jetson_mock_deploy.py`.

## Sample data

`backend/sample_data/` — 5 real exoskeleton trials (`data_collection_*.csv`):
IMU (foot + shank), ankle encoder, and FSR signals at 100 Hz or 200 Hz, plain
and tight-shoe conditions. Selectable in the Launch page instead of live
sensors, so the ML controller runs end-to-end on any machine.

## Safety

Modes: **mock** (`jetson_mock_deploy.py`, no torque code path), **dry-run**
(`jetson_deploy.py --dry-run`, never transmitted) and **live**
(`jetson_deploy.py --arm --start-passive --ankle-zero auto`).

A live run engages the motor at 0 N·m. The first 10 s of walking zero the ankle
encoder; Assist ON is refused until then and until the 3 s window is full.
While assistance is on, the backend re-sends the on state every 0.2 s over UDP
47271; if it stops for 1 s the controller turns assistance off. Assist OFF,
Stop, a crash, or Ctrl-C all go to zero torque. The scale is capped at 0.3 and
torque at the cap set on launch (≤ 5 N·m). `process_manager.py` refuses
`--arm` without `--start-passive`; with a replay CSV, live runs as dry-run.
Only one job runs at a time.
