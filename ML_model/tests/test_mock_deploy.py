"""The mock-deploy inference path: raw exo frame -> ExoController.step -> sane output.

Exercises the same code the Jetson mock runner uses (SensorAdapter -> pipeline ->
buffer -> backend -> assistance), driven by a synthetic gait-like frame stream so
no trained checkpoint or hardware is needed.
"""
import json

import numpy as np
import pytest

from exo.config import Config
from exo.deploy.runtime import ExoController


@pytest.fixture
def tiny_run(tmp_path, cfg):
    """A minimal export dir: TorchScript of an untrained model + metadata."""
    torch = pytest.importorskip("torch")
    from exo.models import TCN

    feat = cfg.data.feature_sets["mid"]
    n_c, win = len(feat), cfg.data.window_length
    model = TCN.from_config(cfg.model, in_channels=n_c, out_channels=1, num_subjects=3).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.idx = torch.zeros(1, dtype=torch.long)

        def forward(self, x):                     # (1, C, T) raw -> scalar N.m/kg
            return self.m(x, self.idx)[0, :, -1]

    ts = torch.jit.trace(Wrap(model), torch.randn(1, n_c, win))
    run = tmp_path / "run"
    run.mkdir()
    ts.save(str(run / "best.ts"))
    (run / "config.yaml").write_text((tmp_path / "cfg.yaml").read_text()
                                     if (tmp_path / "cfg.yaml").exists() else "")
    cfg.save(run)
    (run / "deploy_metadata.json").write_text(json.dumps({
        "feature_names": feat, "num_input_channels": n_c, "window_length": win,
        "control_rate_hz": cfg.deploy.control_rate_hz,
    }))
    return run


def _gait_frame(t: float) -> dict[str, float]:
    """A crude but periodic exo raw frame: gravity + a swing oscillation, FSRs
    high for the first 60 % of each 1 s cycle."""
    ph = (t % 1.0)
    swing = np.sin(2 * np.pi * ph)
    contact = ph < 0.6
    return {
        "foot_ax": 9.0 + 2 * swing, "foot_ay": 0.5 * swing, "foot_az": 1.0,
        "foot_gx": 0.2 * swing, "foot_gy": 3.0 * swing, "foot_gz": 0.1 * swing,
        "shank_ax": 9.3, "shank_ay": 0.2 * swing, "shank_az": -0.5,
        "shank_gx": 0.1 * swing, "shank_gy": 2.5 * swing, "shank_gz": 0.05 * swing,
        "ankle_encoder_deg": 85.0 + 8 * swing,
        "heel_fsr_raw": 24000.0 if contact else 200.0,
        "toe_fsr_raw": 20000.0 if contact else 50.0,
    }


def test_step_fills_buffer_then_predicts(tiny_run):
    cfg = Config.load(tiny_run / "config.yaml")
    ctrl = ExoController(tiny_run, cfg.deploy, subject_mass_kg=72.0)
    ctrl.reset()
    dt = 1.0 / cfg.deploy.control_rate_hz

    win = cfg.data.window_length
    outs = []
    for i in range(win + 200):
        out = ctrl.step(_gait_frame(i * dt), dt)
        outs.append(out)

    assert outs[0]["buffer_ready"] == 0.0
    assert outs[-1]["buffer_ready"] == 1.0
    # a prediction is produced once the window is full
    assert outs[win]["predicted_nm_per_kg"] != 0.0 or outs[win + 1]["predicted_nm_per_kg"] != 0.0


def test_step_output_keys_and_no_nan(tiny_run):
    cfg = Config.load(tiny_run / "config.yaml")
    ctrl = ExoController(tiny_run, cfg.deploy, subject_mass_kg=72.0)
    ctrl.reset()
    dt = 1.0 / cfg.deploy.control_rate_hz
    for i in range(cfg.data.window_length + 50):
        out = ctrl.step(_gait_frame(i * dt), dt)
    assert set(out) == {"command_nm", "predicted_nm_per_kg", "stance", "ramp", "buffer_ready"}
    assert all(np.isfinite(v) for v in out.values())


def test_stance_flag_tracks_fsr(tiny_run):
    cfg = Config.load(tiny_run / "config.yaml")
    # thresholds below the synthetic contact level so stance actually toggles
    from exo.config import replace
    cfg = replace(cfg, **{
        "deploy.sensor_adapter.heel_fsr_threshold": 10000.0,
        "deploy.sensor_adapter.toe_fsr_threshold": 10000.0,
    })
    ctrl = ExoController(tiny_run, cfg.deploy, subject_mass_kg=72.0)
    ctrl.reset()
    dt = 1.0 / cfg.deploy.control_rate_hz
    stance = [ctrl.step(_gait_frame(i * dt), dt)["stance"]
              for i in range(cfg.data.window_length + 300)]
    # both states appear over several cycles
    assert 0.0 in stance and 1.0 in stance


# -- powered loop: the ramp / clamp / watchdog from scripts/jetson_deploy.py ---
# Driven by a constant known command so the checks don't depend on model weights.
def _powered_sent(*, command_nm, rate, arm_ramp_s, torque_cap, n_frames,
                  watchdog_ms=25.0, slow_frame=None):
    from exo.deploy.motor import MotorInterface

    motor = MotorInterface(torque_limit_nm=torque_cap, dry_run=True)
    motor.enter()
    dt = 1.0 / rate
    sent = []
    for i in range(n_frames):
        session_ramp = min(1.0, (i * dt) / arm_ramp_s)
        target = command_nm * session_ramp
        loop_ms = watchdog_ms + 5.0 if i == slow_frame else 1.0
        if loop_ms > watchdog_ms:
            target = 0.0
        sent.append(motor.send_torque(target))
    motor.stop()
    motor.shutdown()
    return sent


def test_powered_session_ramp_fades_in():
    rate = 100
    sent = _powered_sent(command_nm=4.0, rate=rate, arm_ramp_s=1.0,
                         torque_cap=5.0, n_frames=3 * rate)
    assert abs(sent[5]) < abs(sent[50]) < abs(sent[150])
    assert sent[-1] == pytest.approx(4.0)


def test_powered_torque_never_exceeds_cap():
    sent = _powered_sent(command_nm=9.0, rate=100, arm_ramp_s=0.01,
                         torque_cap=1.5, n_frames=200)
    assert max(abs(x) for x in sent) <= 1.5 + 1e-9


def test_powered_watchdog_zeros_torque_on_overrun():
    sent = _powered_sent(command_nm=3.0, rate=100, arm_ramp_s=0.01,
                         torque_cap=5.0, n_frames=50, slow_frame=20)
    assert sent[19] != 0.0
    assert sent[20] == 0.0
    assert sent[21] != 0.0
