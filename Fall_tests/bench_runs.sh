#!/bin/bash
# Test 1 steps 1-2 (and Test 2 step 1): hands-off bench runs, 5 min each.
cd "$(dirname "$0")"
for s in imu_foot imu_shank encoder fsr; do
  python3 fall_logger.py --test 1 --condition bench_${s}_alone --sensors $s --seconds 300 --notes "sensor alone, bench"
done
python3 fall_logger.py --test 1,2 --condition bench_all_still --seconds 300 --notes "full sensor set, stationary"
echo ALL_DONE
