"""S4: nothing sends the MIT enter/exit frames, and the sensor path leaves CAN alone."""
import re
import sys
import types

import pytest

from conftest import REPO
from fakes import FakeSMBus, fake_can, is_special

SOURCE_DIRS = ["TBE_controller", "Cubemars_AK80-9", "ML_model/src", "ML_model/scripts",
               "fall_test_plan", "Fall_tests", "Sensor_Tests", "dashboard/backend"]
SPECIAL = re.compile(r"(0x[fF]{2}\s*,\s*){7}0x[fF][cCdDeE]")


def test_no_source_file_contains_a_special_frame():
    hits = [str(p.relative_to(REPO)) for d in SOURCE_DIRS for p in (REPO / d).rglob("*.py")
            if SPECIAL.search(p.read_text(errors="ignore"))]
    assert hits == []


def _sensor_data(**kwargs):
    from data_obtainer import SensorData
    from utilities import Logger
    return SensorData(Logger(), **kwargs)


def test_sensordata_sends_nothing_at_start_or_shutdown():
    fake_can.opened.clear()
    s = _sensor_data()
    bus = fake_can.opened[0]
    s.shutdown()
    assert bus.frames() == []


def test_sensor_only_mode_never_opens_can():
    fake_can.opened.clear()
    s = _sensor_data(use_can=False)
    s.readSensors()
    s.shutdown()
    assert fake_can.opened == []


def test_jetson_sensors_never_opens_can(monkeypatch):
    # the path used by mock, dry-run and live runs with real sensors
    imu = types.ModuleType("bno085_lean")
    imu.LeanBNO085 = lambda *args, **kwargs: types.SimpleNamespace(
        poll=lambda: 0, counts={}, accel=None, gyro=None, close=lambda: None,
        needs_enable=False, reenable=lambda: None, reconnect=lambda: None)
    imu.ACCEL, imu.GYRO = 0x01, 0x02
    monkeypatch.setitem(sys.modules, "bno085_lean", imu)
    from exo.deploy.jetson_io import JetsonSensors

    fake_can.opened.clear()
    sensors = JetsonSensors(str(REPO))
    sensors.read()
    sensors.shutdown()
    assert fake_can.opened == []


def test_stuck_adc_raises_instead_of_hanging(monkeypatch):
    import data_obtainer
    s = _sensor_data(use_can=False)
    s.filtered_heel_fsr = s.filtered_toe_fsr = 123.0
    monkeypatch.setattr(FakeSMBus, "ready", False)
    # one channel read times out (after its retries) instead of polling forever
    with pytest.raises(TimeoutError):
        s.read_channel(s.bus, data_obtainer.CFG_AIN0)
    # readSensors skips failed samples and keeps the last FSR values...
    for _ in range(data_obtainer.FSR_MAX_CONSECUTIVE_FAILS - 1):
        s.readSensors()
    assert s.filtered_heel_fsr == s.filtered_toe_fsr == 123.0
    # ...then stops the controller once the FSR data is too stale to trust
    with pytest.raises(RuntimeError):
        s.readSensors()


def test_frame_detector_works():
    assert is_special(types.SimpleNamespace(data=[0xFF] * 7 + [0xFC]))
