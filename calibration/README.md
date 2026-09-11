# calibration/

Output directory for exoskeleton calibration runs.

```bash
python exo_frame.py calibrate
```

Runs on the Jetson, with the exo worn on the right leg and `mrsd-exo-ankle/`
present locally. Seven phases, 105 s of recording. **Every phase is gated at both
ends** — it shows the instructions and waits for ENTER, counts down from three,
records, then stops and waits for ENTER again before the next one. Nothing rolls on
automatically, so there is always room to change posture or rest.

## The protocol

| # | Phase | Time | What it measures |
|---|---|---|---|
| 1 | Quiet standing | 10 s | encoder zero, gravity reference for both IMUs |
| 2 | Neutral hold | 5 s | encoder reference pose |
| 3 | Dorsiflexion hold | 5 s | **encoder sign**, dorsiflexion ROM |
| 4 | Plantarflexion hold | 5 s | plantarflexion ROM |
| 5 | Slow ankle sweeps | 10 s | encoder-to-joint ratio, foot sagittal axis |
| 6 | Hip swings | 10 s | **shank sagittal axis**, `R_foot←shank` |
| 7 | Level walking | 60 s | both Georgia Tech rotations, FSR thresholds |

Posture changes only twice: standing to seated after phase 1, then up onto the left
leg for phase 6 and walking for phase 7.

Two instructions decide whether a run is usable. **Do not mix up phases 3 and 4** —
they set the encoder sign, and swapping them inverts every ankle angle downstream.
**Keep the ankle still during phase 6** — the hip swing only works because the foot
and shank move as one rigid body.

## Why the phases are shaped this way

An unlabelled ankle sweep cannot determine sagittal *direction*: PCA recovers the
axis but not its sign. The held poses in phases 3 and 4 make the sign a direct
measurement of the encoder alone, needing no IMU at all.

Phase 6 exists because without it the two segments are calibrated on very unequal
evidence. The foot gets a dedicated, labelled excitation; the shank would get
nothing, its rotation resting solely on the walking fit. With the knee and ankle
held, both IMUs observe the same physical angular velocity, so the swing axis is a
shared direction — and combined with standing gravity it determines `R_foot←shank`.
Gravity is not optional there: a planar swing is rank-1 in gyro, leaving rotation
*about* the swing axis unconstrained, and gravity supplies the missing direction.

## Output

| File | Contents |
|---|---|
| `calibration_<stamp>.json` | Every measured quantity, every quality metric, and the pass/fail verdict per check |
| `raw_<stamp>.npz` | Raw per-phase recordings, so a run can be re-analysed without re-walking |
| `exo_transform_<stamp>.py` | Self-contained importable transform, numpy-only |

The newest transform is also copied to `exo_transform_latest.py`. **A transform is
written only if every critical check passes** — a failed run leaves the raw
recordings and nothing else.

```python
from calibration.exo_transform_latest import features, ingest_gt_trial

# deployment, per sample, in the control loop
x = features(foot_accel, foot_gyro, shank_accel, shank_gyro,
             encoder_deg, toe_fsr, heel_fsr)

# training, a Georgia Tech trial rotated into this exo's frame
X, y = ingest_gt_trial("mrsd-exo-ankle", "AB06", "treadmill_01_01")
```

Both paths emit the same 15 columns in the same order: 12 IMU channels (foot and
shank, accel and gyro), the ankle angle from the encoder, and two contact flags.

## The segment-consistency constraint

Each calibration records a derived quantity, `r_foot_from_shank_gt`:

```
R_fs_GT = R_foot⁻¹ · R_fs_exo · R_shank
```

This is the Georgia Tech foot↔shank frame relationship — a fixed property of *their*
hardware, so it should come out the same on every run of every exo. Drift across
runs means an IMU moved or a phase was done badly, and is reported as a warning.

It is recorded rather than imposed, because it cannot be measured independently.
Georgia Tech walking never contains a rigid foot-shank epoch to measure it from:
fitting one of their segments' gyro onto the other's at mid-stance leaves a residual
of about 70% of signal. Your `R_fs_exo` alone does not constrain the two rotations —
it is exactly consumed determining this unknown.

Once you have several runs whose `R_fs_GT` agree, the constraint becomes usable:

```bash
python exo_frame.py calibrate --constrain-segments
```

This re-fits both rotations subject to `R_foot · R_fs_GT = R_fs_exo · R_shank`, using
the relationship from the most recent passing run, so the two segments cannot
disagree. It refuses to run if no prior relationship exists, and fails the run if the
constraint moves either rotation by more than 10°.

## Other options

| Flag | Effect |
|---|---|
| `--no-prompt` | Skip the ENTER gates (countdown still runs). For unattended replay only. |
| `--imu-hz` | IMU report rate to force during calibration. Default 200. |
| `--reference-subject` | Georgia Tech subject used as the rotation reference. Default AB06. |
| `--calib-dir` | Where runs are written. Default `calibration`. |

The run forces `IMU_REPORT_HZ` to 200 by patching the module global before
`SensorHub` is built, leaving `Data collection/data_collection.py` untouched. It
warns that your recording script still runs at 25, and fails the run if the measured
effective rate does not come back above 100 Hz.

## Validating without hardware

```bash
python test_calibration.py
```

Synthesises every phase with known values and requires them back. Recovers the
encoder zero to 0.002°, the ratio to 0.0006, `R_foot←shank` to 0.02°, and the
injected walking rotation exactly against its source trial. Includes negative
controls: swapping the dorsi/plantar holds must invert the sign, and replaying a
real 25 Hz recording must fail the bandwidth check and emit no module.
