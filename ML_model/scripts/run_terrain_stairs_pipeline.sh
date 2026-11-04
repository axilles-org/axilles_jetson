#!/bin/bash
# Same comparison as run_terrain_pipeline.sh with stairs added. Downloads the stair
# data now, then waits for the first pipeline to free the GPU before training.
#
#   nohup caffeinate -i scripts/run_terrain_stairs_pipeline.sh > ~/PycharmProjects/exo-data/logs/terrain_stairs_pipeline.log 2>&1 &
set -euo pipefail

ML="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$HOME/PycharmProjects/exo-data"
PY=/usr/local/bin/python3
EPOCHS="${EPOCHS:-5}"

echo "[$(date)] downloading stairs (8 workers, leaves CPU for the running training)"
for attempt in 1 2 3; do
    pass_log="$DATA/logs/fetch_stairs_$attempt.log"
    "$PY" "$ML/scripts/fetch_gatech_terrains.py" --out "$DATA/csv_output" --modes stair --workers 8 > "$pass_log" 2>&1 || true
    if tail -1 "$pass_log" | grep -q '^done' && ! grep -q FAILED "$pass_log"; then
        echo "[$(date)] stairs downloaded (pass $attempt)"
        break
    fi
    [ "$attempt" = 3 ] && { echo "[$(date)] stair download still failing, see $pass_log"; exit 1; }
done

echo "[$(date)] waiting for the first terrain run to finish"
while pgrep -f run_terrain_pipeline.sh > /dev/null; do sleep 60; done

echo "[$(date)] csv -> parquet (with stairs)"
"$PY" "$ML/scripts/csv_to_parquet.py" --src "$DATA/csv_output" --out "$DATA/hf_dataset_terrain_stairs" \
    --terrains treadmill levelground ramp stair | tail -1

echo "[$(date)] ingest + train ($EPOCHS epochs per model)"
cd "$ML"
"$PY" scripts/terrain_experiment.py --dataset "$DATA/hf_dataset_terrain_stairs" \
    --processed "$DATA/processed_terrain_stairs" --epochs "$EPOCHS" \
    --terrains treadmill,levelground,ramp,stair --name terrain_stairs
echo "[$(date)] finished"
