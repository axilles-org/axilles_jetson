#!/bin/bash
# Waits for the GaTech terrain download, then converts, ingests and trains every
# terrain_experiment variant on the Mac GPU. Keeps the Mac awake while it runs.
#
#   nohup caffeinate -i scripts/run_terrain_pipeline.sh > ~/PycharmProjects/exo-data/logs/terrain_pipeline.log 2>&1 &
set -euo pipefail

ML="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$HOME/PycharmProjects/exo-data"
PY=/usr/local/bin/python3
FETCH_LOG="$DATA/logs/fetch_terrains.log"
EPOCHS="${EPOCHS:-5}"

echo "[$(date)] waiting for the download to finish"
while pgrep -f fetch_gatech_terrains.py > /dev/null; do sleep 60; done

# a failed member leaves no CSV and the fetcher skips finished files, so a re-run
# retries only what is missing; stop once a pass completes with no failures
for attempt in 1 2 3; do
    pass_log="$DATA/logs/fetch_verify_$attempt.log"
    "$PY" "$ML/scripts/fetch_gatech_terrains.py" --out "$DATA/csv_output" --workers 24 > "$pass_log" 2>&1 || true
    if tail -1 "$pass_log" | grep -q '^done' && ! grep -q FAILED "$pass_log"; then
        echo "[$(date)] download verified (pass $attempt)"
        break
    fi
    [ "$attempt" = 3 ] && { echo "[$(date)] download still failing, see $pass_log"; exit 1; }
done

echo "[$(date)] csv -> parquet"
"$PY" "$ML/scripts/csv_to_parquet.py" --src "$DATA/csv_output" --out "$DATA/hf_dataset_terrain" \
    --terrains treadmill levelground ramp | tail -1

echo "[$(date)] ingest + train ($EPOCHS epochs per model)"
cd "$ML"
"$PY" scripts/terrain_experiment.py --dataset "$DATA/hf_dataset_terrain" \
    --processed "$DATA/processed_terrain" --epochs "$EPOCHS"
echo "[$(date)] finished"
