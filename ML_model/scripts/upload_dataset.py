"""Create/update the Hugging Face dataset repo from the local Parquet tree.

    huggingface-cli login
    python scripts/csv_to_parquet.py --src <csv_output> --out <parquet_dir>
    python scripts/upload_dataset.py --src <parquet_dir> --repo aicognition/mrsd-exo-ankle --public
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_DATASET_CARD = """\
---
license: cc-by-4.0
language:
  - en
tags:
  - biomechanics
  - exoskeleton
  - wearable-sensors
  - ankle-torque
  - inverse-dynamics
  - gait
pretty_name: MRSD Ankle Exo — Torque Prediction
size_categories:
  - 100M<n<1B
source_datasets:
  - extended|other-camargo-2021-lower-limb-biomechanics
annotations_creators:
  - machine-generated
---

# MRSD Ankle Exoskeleton — Torque Prediction Dataset

Synchronised wearable-sensor and inverse-dynamics data for training models that
predict the **human ankle plantarflexion moment** from IMUs, an ankle angle and a
ground-contact flag. Used to drive assistive torque on the MRSD ankle
exoskeleton.

- **Subjects:** 22 (AB06–AB30), able-bodied
- **Task:** treadmill walking, 0.86–1.29 m/s
- **Trials:** 157 (~7 per subject, ~140 s each)
- **Rates:** IMU/ID 200 Hz, force plate/goniometer 1000 Hz, gait phase 200 Hz
- **Source:** derived from Camargo et al. 2021 (Georgia Tech EPIC lab); reshaped to
  Parquet, unused modalities (EMG, markers, IK, joint power) dropped

## Repository layout

```
metadata.parquet                              one row per (subject, trial)
subjects/<subject>/<trial>__imu.parquet        foot/shank/thigh/trunk accel+gyro
subjects/<subject>/<trial>__id.parquet         inverse-dynamics joint moments (N·m)
subjects/<subject>/<trial>__fp.parquet         treadmill force plate (GRF, CoP)
subjects/<subject>/<trial>__gon.parquet        goniometer joint angles (degrees)
subjects/<subject>/<trial>__gcRight.parquet    right-leg gait phase (HeelStrike, ToeOff)
```

Every sensor file has a `time_s` column (seconds) plus that sensor's channels, at
**native sample rate** — no resampling or filtering is applied at rest.

### `metadata.parquet` columns

`subject`, `trial`, `terrain`, `n_imu_samples`, `age`, `gender`, `height_m`,
`weight_kg`, `speed_mean_mps`, `trialstarts_s`, `trialends_s`.

### Sensor channels

| File | Rate (Hz) | Key channels | Units |
|------|-----------|--------------|-------|
| `imu` | 200 | `{foot,shank,thigh,trunk}_{Accel,Gyro}_{X,Y,Z}` | m/s², rad/s |
| `id` | 200 | `ankle_angle_r_moment` (**target**), plus hip/knee/pelvis/lumbar moments | N·m |
| `fp` | 1000 | `Treadmill_{R,L}_{vx,vy,vz,px,py,pz,moment_x,moment_y,moment_z}` | N, m, N·m |
| `gon` | 1000 | `{ankle,knee,hip}_{sagittal,frontal}` | degrees |
| `gcRight` | 200 | `HeelStrike`, `ToeOff` | percent of cycle (0–100), resets at the event |

## Prediction target

`id.ankle_angle_r_moment` — the right-ankle net plantarflexion/dorsiflexion
moment from OpenSim inverse dynamics, in **N·m**. **Plantarflexion (push-off) is
negative** in this convention. For subject-independent reporting, divide by
`metadata.weight_kg` to get N·m/kg.

`NaN` in the target marks samples outside the valid inverse-dynamics window.

## Download

```bash
pip install huggingface_hub pandas pyarrow
```

Whole dataset:

```python
from huggingface_hub import snapshot_download

root = snapshot_download(
    repo_id="aicognition/mrsd-exo-ankle",
    repo_type="dataset",
    revision="v1.0",
    local_dir="mrsd-exo-ankle",
)
```

One subject only:

```python
root = snapshot_download(
    repo_id="aicognition/mrsd-exo-ankle", repo_type="dataset", revision="v1.0",
    allow_patterns=["metadata.parquet", "subjects/AB06/*"],
    local_dir="mrsd-exo-ankle",
)
```

## Load a trial

```python
import pandas as pd
from pathlib import Path

root = Path("mrsd-exo-ankle")
meta = pd.read_parquet(root / "metadata.parquet")
subject, trial = "AB06", "treadmill_01_01"

imu = pd.read_parquet(root / f"subjects/{subject}/{trial}__imu.parquet")
idf = pd.read_parquet(root / f"subjects/{subject}/{trial}__id.parquet")
gon = pd.read_parquet(root / f"subjects/{subject}/{trial}__gon.parquet")
gc  = pd.read_parquet(root / f"subjects/{subject}/{trial}__gcRight.parquet")

mass = meta.loc[(meta.subject == subject) & (meta.trial == trial), "weight_kg"].item()
ankle_moment_nm_per_kg = idf["ankle_angle_r_moment"] / mass
```

## Visualise one gait cycle

```python
import numpy as np
import matplotlib.pyplot as plt

# heel-strike events = where the gait-phase signal wraps back to 0
hs = np.where(np.diff(gc["HeelStrike"].to_numpy()) < -50)[0] + 1
tau = (idf["ankle_angle_r_moment"] / mass).to_numpy()

cycles = []
for a, b in zip(hs[:-1], hs[1:]):
    seg = tau[a:b]
    if 60 < len(seg) < 400:
        cycles.append(np.interp(np.linspace(0, 1, 100),
                                np.linspace(0, 1, len(seg)), seg))
cycles = np.array(cycles)

phase = np.linspace(0, 100, 100)
plt.fill_between(phase, cycles.mean(0) - cycles.std(0),
                 cycles.mean(0) + cycles.std(0), alpha=0.2)
plt.plot(phase, cycles.mean(0), lw=2)
plt.gca().invert_yaxis()                       # plantarflexion is negative
plt.xlabel("gait cycle (%)"); plt.ylabel("ankle moment (N·m/kg)")
plt.title(f"{subject} — mean ± SD over {len(cycles)} strides")
plt.show()
```

The mean-cycle push-off peak is around −1.5 to −2 N·m/kg (single strides reach
−1.8 to −2.5), near zero through swing.

## Suggested split (subject-independent)

| Split | Subjects |
|-------|----------|
| train | AB06, AB07, AB08, AB11, AB12, AB13, AB14, AB15, AB16, AB17, AB18, AB19, AB20 |
| val   | AB09 |
| test  | AB10, AB21, AB23, AB24, AB25, AB27, AB28, AB30 |

Fit any normalisation on **train subjects only**.

## Full training pipeline

The `mrsd_exo` codebase ingests this dataset (100 Hz resample with anti-aliasing,
binary `stance` from the gait-phase signal, subject-wise scaling) and trains a
causal TCN. See the project repository for `scripts/prepare.py` and
`scripts/train.py`.

## Source, changes made, and licence

This is a **derived work** of the Georgia Tech / EPIC-Lab lower-limb biomechanics
dataset by Camargo et al. (2021), used and redistributed under the **Creative
Commons Attribution 4.0 International licence (CC BY 4.0)**.

- **Original dataset:** Camargo, Jonathan (2021), *"A comprehensive, open-source
  dataset of lower limb biomechanics ... Part 1 of 3"*, Mendeley Data, V2,
  DOI [10.17632/fcgm3chfff.2](https://doi.org/10.17632/fcgm3chfff.2).
  Download and mirror: <https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/>
- **Original publication:** Camargo, J., Ramanathan, A., Flanagan, W., Young, A.
  (2021). *A comprehensive, open-source dataset of lower limb biomechanics in
  multiple conditions of stairs, ramps, and level-ground ambulation and
  transitions.* Journal of Biomechanics, 119, 110320,
  DOI [10.1016/j.jbiomech.2021.110320](https://doi.org/10.1016/j.jbiomech.2021.110320).
- **Licence:** CC BY 4.0 —
  <https://creativecommons.org/licenses/by/4.0/legalcode>. This derived
  repository is redistributed under the same licence.

**Changes made in this derived version:**

1. Kept only the `treadmill` ambulation condition (stairs, ramps, level-ground
   overground, and walk-to-run transitions were excluded).
2. Kept only the `imu`, `id`, `fp`, `gon`, and `gcRight` sensor streams; excluded
   `emg`, `markers`, `ik`, `ik_offset`, `jp` (joint power), and `gcLeft`.
3. Reorganised from the original per-sensor CSV directory tree into one Parquet
   file per (subject, trial, sensor), with the original `Header` column renamed to
   `time_s`. Signal values, sample rates, and units are unchanged.
4. Consolidated per-subject demographics and per-trial condition files
   (`subject_info`, `speed`, `trialStarts`, `trialEnds`) into a single
   `metadata.parquet`.

No signal data was resampled, filtered, or otherwise altered in value; the
resampling and feature engineering used for model training happen downstream in
the `mrsd_exo` codebase, not in this dataset.

**No endorsement.** Nothing here implies that Georgia Tech, the EPIC Lab, or the
original authors endorse this derived dataset or its use.

**Third-party content.** Any material within the original dataset identified as
belonging to a third party may require separate permission; this redistribution
grants no additional rights over such material.

## Citation

> Camargo, J., Ramanathan, A., Flanagan, W., Young, A. (2021). A comprehensive,
> open-source dataset of lower limb biomechanics in multiple conditions of
> stairs, ramps, and level-ground ambulation and transitions.
> *Journal of Biomechanics*, 119, 110320.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="local Parquet dataset dir")
    ap.add_argument("--repo", required=True, help="e.g. aicognition/mrsd-exo-ankle")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--message", default="update dataset")
    ap.add_argument("--public", action="store_true", help="public repo (default: private)")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    src = Path(args.src)
    if not (src / "metadata.parquet").exists():
        raise SystemExit(f"{src} has no metadata.parquet")

    (src / "README.md").write_text(_DATASET_CARD)

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_folder(
        folder_path=str(src),
        repo_id=args.repo,
        repo_type="dataset",
        revision=args.revision,
        commit_message=args.message,
    )
    print(f"uploaded {src} -> https://huggingface.co/datasets/{args.repo} @ {args.revision}")


if __name__ == "__main__":
    main()
