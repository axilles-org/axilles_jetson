"""Original/transformed IMU mixing for the transform ablation."""
import numpy as np
import pandas as pd
import pytest

from exo.data.ablation import IMU_GROUPS, build_dataset, drop_features

TRIAL = "treadmill_01_01"


def _imu(value: float, n: int = 20) -> pd.DataFrame:
    cols = {"time_s": np.arange(n) / 200.0}
    for g in IMU_GROUPS + ("thigh_Accel",):
        for ax in "XYZ":
            cols[f"{g}_{ax}"] = np.full(n, value)
    return pd.DataFrame(cols)


@pytest.fixture
def roots(tmp_path):
    orig, tx = tmp_path / "orig", tmp_path / "tx"
    for root, value in ((orig, 1.0), (tx, 2.0)):
        subj = root / "subjects" / "AB06"
        subj.mkdir(parents=True)
        _imu(value).to_parquet(subj / f"{TRIAL}__imu.parquet")
        pd.DataFrame({"x": [0]}).to_parquet(subj / f"{TRIAL}__id.parquet")
        pd.DataFrame({"subject": ["AB06"]}).to_parquet(root / "metadata.parquet")
    return orig, tx, tmp_path / "out"


def test_mixes_only_requested_groups(roots):
    orig, tx, out = roots
    root = build_dataset(orig, tx, out, frozenset({"foot_Gyro"}))
    imu = pd.read_parquet(root / "subjects" / "AB06" / f"{TRIAL}__imu.parquet")
    assert (imu["foot_Gyro_X"] == 2.0).all()
    assert (imu["foot_Accel_X"] == 1.0).all()
    assert (imu["thigh_Accel_X"] == 1.0).all()
    assert (root / "subjects" / "AB06" / f"{TRIAL}__id.parquet").exists()
    assert (root / "metadata.parquet").exists()


def test_empty_and_full_reuse_existing_roots(roots):
    orig, tx, out = roots
    assert build_dataset(orig, tx, out, frozenset()) == orig
    assert build_dataset(orig, tx, out, frozenset(IMU_GROUPS)) == tx
    assert not out.exists()


def test_unknown_group_rejected(roots):
    orig, tx, out = roots
    with pytest.raises(ValueError):
        build_dataset(orig, tx, out, frozenset({"thigh_Gyro"}))


def test_drop_features():
    names = ["imu_foot_Gyro_X", "imu_foot_Gyro_Y", "imu_foot_Gyro_Z", "imu_foot_Accel_X", "stance"]
    assert drop_features(names, frozenset({"foot_Gyro"})) == ["imu_foot_Accel_X", "stance"]
    assert drop_features(names, frozenset({"stance"})) == names[:4]


def test_drop_features_rejects_unknown_and_empty():
    with pytest.raises(ValueError):
        drop_features(["stance"], frozenset({"gon_ankle_sagittal"}))
    with pytest.raises(ValueError):
        drop_features(["stance"], frozenset({"stance"}))
