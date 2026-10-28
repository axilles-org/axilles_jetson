"""The feature transform used in training and deployment must be identical."""
import numpy as np
import pandas as pd
import pytest

from exo.data.feature_pipeline import FeaturePipeline

FEATURES = ["imu_foot_Accel_X", "imu_shank_Gyro_Z", "stance", "gon_ankle_sagittal"]


def test_selects_and_orders_columns():
    pipeline = FeaturePipeline(FEATURES)
    df = pd.DataFrame({
        "gon_ankle_sagittal": [0.1, 0.2],
        "extra": [9.0, 9.0],
        "stance": [0.0, 1.0],
        "imu_shank_Gyro_Z": [-1.0, -2.0],
        "imu_foot_Accel_X": [3.0, 4.0],
    })
    out = pipeline.transform(df)
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out[0], [3.0, -1.0, 0.0, 0.1])


def test_missing_column_raises():
    pipeline = FeaturePipeline(FEATURES)
    with pytest.raises(ValueError):
        pipeline.transform(pd.DataFrame({"imu_foot_Accel_X": [1.0]}))


def test_non_binary_stance_raises():
    pipeline = FeaturePipeline(FEATURES)
    df = pd.DataFrame({c: [0.0] for c in FEATURES})
    df["stance"] = [0.5]
    with pytest.raises(ValueError):
        pipeline.transform(df)


def test_train_and_deploy_paths_match():
    """A raw frame fed through the training column order and through a dict (as the
    runtime builds it) yields the same vector."""
    pipeline = FeaturePipeline(FEATURES)
    raw = {"imu_foot_Accel_X": 1.5, "imu_shank_Gyro_Z": -0.3, "stance": 1.0,
           "gon_ankle_sagittal": 0.12, "unused": 0.0}

    train_df = pd.DataFrame([raw])
    deploy_df = pd.DataFrame([[raw[c] for c in FEATURES]], columns=FEATURES)

    np.testing.assert_array_equal(
        pipeline.transform(train_df), pipeline.transform(deploy_df)
    )


def test_transform_row_matches_dataframe_path():
    """The runtime's no-pandas path gives bit-identical float32 values."""
    pipeline = FeaturePipeline(FEATURES)
    rng = np.random.default_rng(0)
    for i in range(500):
        scale = 10.0 ** rng.integers(-8, 8)
        raw = {c: float(rng.normal() * scale) for c in FEATURES}
        raw["stance"] = [0.0, 1.0, float("nan")][i % 3]
        raw["unused"] = 123.0
        expected = pipeline.transform(pd.DataFrame([[raw[c] for c in FEATURES]],
                                                   columns=FEATURES))[0]
        row = pipeline.transform_row(raw)
        assert row.dtype == np.float32 and row.shape == (len(FEATURES),)
        np.testing.assert_array_equal(row, expected)


def test_transform_row_checks_like_transform():
    pipeline = FeaturePipeline(FEATURES)
    raw = {c: 0.0 for c in FEATURES}
    raw["stance"] = 0.5
    with pytest.raises(ValueError):
        pipeline.transform_row(raw)
    with pytest.raises(KeyError):
        pipeline.transform_row({"imu_foot_Accel_X": 1.0})
