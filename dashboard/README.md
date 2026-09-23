# Exo Dashboard

Live telemetry and trial history for both controllers in this repo —
`TBE_controller/main.py` (fixed torque profile) and
`ML_model/scripts/jetson_deploy.py` (TCN). Every run is recorded (full-rate
traces + config) and streamed to a browser while it's in progress. A
**Launch** page starts either controller without touching the CLI, in
non-torque modes only.

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
| **Live** | Real-time charts for whichever trial is currently streaming |
| **Launch** | Pick a model + replay source, start/stop, watch the console — never launches an armed (torque-sending) run |
| **Runs** | History table, filter/search, diff two runs' model config, import/export |

## Files

```
backend/
  server.py            FastAPI app — history API, live WebSocket relay, launch/stop, model + sample listing
  run_logger.py         Recorder wired into each controller's loop (non-blocking; writes Parquet + SQLite, relays live over UDP)
  process_manager.py    Launches/stops a controller subprocess for the Launch page; blocks any --arm (torque) invocation
  simulate_run.py        Fake-trial generator, for trying the dashboard with no hardware
  sample_data/            Bundled real exo CSVs, usable as a replay source (see below)

frontend/src/
  pages/LivePage.jsx       Live charts, auto-adapts to whichever fields the active controller logs
  pages/LaunchPage.jsx     Model picker, replay source (sample/upload), start/stop, live console
  pages/RunsPage.jsx       Run history, filters, compare, import
  pages/RunDetailPage.jsx  One run's full traces + model card
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

The dashboard can only launch **mock** (`jetson_mock_deploy.py`, no torque
code path at all) or **dry-run** (`jetson_deploy.py --dry-run`, full command
path but never transmitted) — `process_manager.py` rejects any argv
containing `--arm`. A powered run stays CLI-only; see
`ML_model/docs/DEPLOYMENT.md`. Only one job runs at a time.
