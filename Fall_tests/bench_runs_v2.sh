#!/bin/bash
# Test 1 steps 1-2 (and Test 2 step 1): hands-off bench runs, 5 min each.
cd "$(dirname "$0")"
for s in imu_foot imu_shank encoder fsr; do
  python3 fall_logger.py --test 1 --condition bench_${s}_alone_v2 --sensors $s --seconds 300 --notes "sensor alone, bench; GC off, pinned to CPU 5"
done
python3 fall_logger.py --test 1,2 --condition bench_all_still_v2 --seconds 300 --notes "full sensor set, stationary; GC off, pinned to CPU 5"
echo ALL_DONE_V2
