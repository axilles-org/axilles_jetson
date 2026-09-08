# calibration/

Output directory for exoskeleton calibration runs.

Each run of `python exo_frame.py calibrate` writes three files here, stamped with
the run time:

| File | Contents |
|---|---|
| `calibration_<stamp>.json` | Every measured quantity, every quality metric, and the pass/fail verdict for each check. This is the record of what was measured. |
| `raw_<stamp>.npz` | The raw per-phase sensor recordings, so a run can be re-analysed later without re-walking. |
| `exo_transform_<stamp>.py` | A self-contained, importable transform module generated from that calibration. Depends on numpy only. |

The newest `exo_transform_*.py` is also copied to `exo_transform_latest.py` so
downstream code can import a stable name.

## Using a generated transform

```python
from calibration.exo_transform_latest import features, ingest_gt_trial, FEATURES

# deployment, per sample, in the control loop
x = features(foot_accel, foot_gyro, shank_accel, shank_gyro,
             encoder_deg, toe_fsr, heel_fsr)

# training, a Georgia Tech trial rotated into this exo's frame
X, y = ingest_gt_trial("mrsd-exo-ankle", "AB06", "treadmill_01_01")
```

A generated module carries the calibration values as literals, so it never reads
the JSON at run time and has no dependency on `exo_frame.py`.
