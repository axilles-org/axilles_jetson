"""SensorAdapter: rotation, unit scaling, lever-arm correction, FSR debounce."""
import numpy as np
import pytest

from exo.config import SensorAdapterConfig
from exo.deploy.sensor_adapter import SensorAdapter, _skew


def _raw(foot_a=(0, 0, 1), foot_g=(0, 0, 0), shank_a=(0, 0, 1), shank_g=(0, 0, 0),
         ankle_deg=90.0, heel=0.0, toe=0.0):
    return {
        "foot_ax": foot_a[0], "foot_ay": foot_a[1], "foot_az": foot_a[2],
        "foot_gx": foot_g[0], "foot_gy": foot_g[1], "foot_gz": foot_g[2],
        "shank_ax": shank_a[0], "shank_ay": shank_a[1], "shank_az": shank_a[2],
        "shank_gx": shank_g[0], "shank_gy": shank_g[1], "shank_gz": shank_g[2],
        "ankle_encoder_deg": ankle_deg, "heel_fsr_raw": heel, "toe_fsr_raw": toe,
    }


def test_identity_adapter_passes_through():
    a = SensorAdapter(SensorAdapterConfig())
    out = a.transform_frame(_raw(foot_a=(1, 2, 3), foot_g=(4, 5, 6)), dt=0.01)
    assert out["imu_foot_Accel_X"] == pytest.approx(1.0)
    assert out["imu_foot_Gyro_Z"] == pytest.approx(6.0)


def test_rotation_and_scale_applied():
    # 90 deg about Z: x->y, y->-x
    Rz = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    cfg = SensorAdapterConfig(foot_rotation=Rz, foot_accel_scale=0.5)
    a = SensorAdapter(cfg)
    out = a.transform_frame(_raw(foot_a=(2, 0, 0)), dt=0.01)
    assert out["imu_foot_Accel_X"] == pytest.approx(0.0, abs=1e-9)
    assert out["imu_foot_Accel_Y"] == pytest.approx(1.0)   # 2 * 0.5, rotated


def test_encoder_to_radians_with_neutral_and_sign():
    cfg = SensorAdapterConfig(ankle_encoder_neutral_deg=90.0, ankle_encoder_sign=-1.0)
    a = SensorAdapter(cfg)
    out = a.transform_frame(_raw(ankle_deg=100.0), dt=0.01)
    assert out["gon_ankle_sagittal"] == pytest.approx(np.radians(-10.0))


def test_lever_arm_removes_centripetal_term():
    r = np.array([0.1, 0.0, 0.0])
    cfg = SensorAdapterConfig(foot_lever_arm_m=r.tolist())
    a = SensorAdapter(cfg)
    w = np.array([0.0, 0.0, 3.0])                      # spin about z
    # feed constant w so wdot -> 0 after the first step
    a.transform_frame(_raw(foot_g=w.tolist(), foot_a=(0, 0, 1)), dt=0.01)
    out = a.transform_frame(_raw(foot_g=w.tolist(), foot_a=(0, 0, 1)), dt=0.01)
    expected_centripetal = _skew(w) @ (_skew(w) @ r)   # = [-0.9, 0, 0]
    got = np.array([out["imu_foot_Accel_X"], out["imu_foot_Accel_Y"],
                    out["imu_foot_Accel_Z"]])
    assert got == pytest.approx(np.array([0, 0, 1.0]) - expected_centripetal, abs=1e-6)


def test_fsr_debounce_rejects_short_spike():
    cfg = SensorAdapterConfig(heel_fsr_threshold=1000.0, toe_fsr_threshold=1000.0,
                              fsr_debounce_s=0.05)
    a = SensorAdapter(cfg)
    dt = 0.01
    # 2 frames above threshold (0.02 s < 0.05 s debounce) then back to 0
    seq = [0, 5000, 5000, 0, 0, 0]
    states = [a.transform_frame(_raw(heel=v), dt)["stance"] for v in seq]
    assert states == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_fsr_debounce_accepts_sustained_contact():
    cfg = SensorAdapterConfig(heel_fsr_threshold=1000.0, toe_fsr_threshold=1000.0,
                              fsr_debounce_s=0.03)
    a = SensorAdapter(cfg)
    dt = 0.01
    seq = [5000] * 8
    states = [a.transform_frame(_raw(heel=v), dt)["stance"] for v in seq]
    assert states[-1] == 1.0
    assert states[0] == 0.0            # not latched instantly


def test_reset_clears_wdot_history():
    a = SensorAdapter(SensorAdapterConfig(foot_lever_arm_m=[0.1, 0, 0]))
    a.transform_frame(_raw(foot_g=(0, 0, 5)), dt=0.01)
    a.reset()
    assert a.foot._w_prev is None
