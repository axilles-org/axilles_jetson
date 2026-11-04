"""Sensor thread, Pawan's lean IMU reader and the measured-timestep model input, with fake sensors."""
import csv
import importlib
import runpy
import struct
import sys
import time
import types

import numpy as np
import pytest

from conftest import REPO
from dashboard.backend import run_logger
from fakes import FakeSMBus, fake_can, wait_for

ACC, GYRO, QUAT = 0x01, 0x02, 0x08          # BNO085 report ids
RUN_DIR = REPO / "ML_model" / "runs" / "deploy"
SCRIPT = REPO / "ML_model" / "scripts" / "jetson_deploy.py"


class FakeLean:
    """One IMU; every poll delivers a new report unless frozen or failing."""

    made = {}

    def __init__(self, bus, address, report_hz=200.0, features=(ACC, GYRO), label=None):
        self.address, self.report_hz, self.features = address, report_hz, tuple(features)
        self.mode, self.count = "ok", 0
        self.accel = self.gyro = None
        self.counts = {ACC: 0, GYRO: 0}
        FakeLean.made[address] = self

    def poll(self):
        time.sleep(0.001)
        if self.mode == "fail":
            raise OSError("IMU unplugged")
        if self.mode == "frozen" or self.count % 2:     # every other poll finds the FIFO empty
            self.count += self.mode != "frozen"
            return 0
        self.count += 1
        self.accel, self.gyro = (float(self.count), 0.0, 9.8), (0.0, 0.1, 0.0)
        self.counts[ACC] += 1
        self.counts[GYRO] += 1
        return 1

    def close(self):
        pass


@pytest.fixture
def fake_lean_module(monkeypatch):
    mod = types.ModuleType("bno085_lean")
    mod.LeanBNO085, mod.ACCEL, mod.GYRO = FakeLean, ACC, GYRO
    monkeypatch.setitem(sys.modules, "bno085_lean", mod)
    FakeLean.made.clear()


@pytest.fixture
def sensors(fake_lean_module):
    from exo.deploy.jetson_io import JetsonSensors
    made = []

    def build(**kwargs):
        s = JetsonSensors(str(REPO), **kwargs)
        made.append(s)
        assert wait_for(lambda: s.age_ms() < 50 and FakeLean.made[0x4A].count > 5) is not None
        return s

    yield build
    for s in made:
        s.shutdown()


def test_both_imus_asked_for_100hz_accel_and_gyro_only(sensors):
    sensors()
    assert sorted(FakeLean.made) == [0x4A, 0x4B]
    for imu in FakeLean.made.values():
        assert imu.report_hz == pytest.approx(100.0)
        assert imu.features == (ACC, GYRO)


def test_read_returns_fresh_values_and_opens_no_can(sensors):
    fake_can.opened.clear()
    s = sensors()
    first = s.read()["foot_ax"]
    assert wait_for(lambda: s.read()["foot_ax"] > first + 3) is not None
    assert s.read()["shank_ax"] > 0
    assert s.age_ms() < 50
    assert fake_can.opened == []


@pytest.mark.parametrize("mode", ["frozen", "fail"])
@pytest.mark.parametrize("address", [0x4A, 0x4B])
def test_imu_dropout_keeps_last_values_and_run_alive(sensors, mode, address):
    s = sensors(stale_s=0.1)
    FakeLean.made[address].mode = mode
    time.sleep(0.25)
    prefix = 'foot' if address == 0x4A else 'shank'
    first = s.read()[f'{prefix}_ax']
    assert s.frame_at(time.monotonic())[f'{prefix}_ax'] == first
    assert s.ages_ms()[f'{prefix}_accel'] > 100
    time.sleep(.02)
    assert s.read()[f'{prefix}_ax'] == first


def test_missing_fsr_conversions_keep_run_alive(sensors, monkeypatch):
    s = sensors()
    monkeypatch.setattr(s._fsr, "step", lambda now: None)
    assert wait_for(lambda: s.ages_ms()["FSR"] > 150) is not None
    assert 'heel_fsr_raw' in s.read()


def test_fsrs_are_raw_counts_read_after_timed_conversions(sensors, monkeypatch):
    from exo.deploy.jetson_io import TimedADS1115
    log = []
    monkeypatch.setattr(FakeSMBus, "write_i2c_block_data",
                        lambda self, addr, reg, data: log.append(("start", data, time.perf_counter())))
    monkeypatch.setattr(FakeSMBus, "read_i2c_block_data",
                        lambda self, addr, reg, count: log.append(("read", reg, time.perf_counter()))
                        or [0x12, 0x34])
    s = sensors()
    assert wait_for(lambda: s.read()["toe_fsr_raw"] == 0x1234 == s.read()["heel_fsr_raw"]) is not None
    adc = [e for e in log if e[0] == "start" or e[1] == TimedADS1115.REG_CONV]
    assert TimedADS1115.REG_CFG not in [e[1] for e in log if e[0] == "read"]     # no ready-bit polls
    pairs = list(zip(adc, adc[1:]))
    assert all(b[0] == "read" for a, b in pairs if a[0] == "start")
    assert all(b[2] - a[2] >= TimedADS1115.CONV_S for a, b in pairs if a[0] == "start")
    assert [e[1] for e in adc if e[0] == "start"][:2] == list(TimedADS1115.CFG)


def test_no_teleplot_packets_from_the_sensor_thread(sensors, monkeypatch):
    import data_obtainer
    calls = []
    monkeypatch.setattr(data_obtainer.SensorData, "publishTelemetry", lambda self: calls.append(1))
    sensors()
    time.sleep(0.05)
    assert calls == []


def test_frame_at_interpolates_between_readings(sensors):
    s = sensors()
    with s._lock:
        history = list(s._history)
    (t0, f0), (t1, f1) = next((a, b) for a, b in zip(history, history[1:])
                              if b[1]["foot_ax"] != a[1]["foot_ax"])
    mid = s.frame_at((t0 + t1) / 2)
    assert mid["foot_ax"] == pytest.approx((f0["foot_ax"] + f1["foot_ax"]) / 2)
    assert s.frame_at(time.monotonic() + 1.0)["foot_ax"] == s.read()["foot_ax"]


@pytest.fixture
def lean(monkeypatch):
    """Pawan's real bno085_lean module: the installed Adafruit packet parser, with
    the setup driver and the /dev/i2c device file faked."""
    pytest.importorskip("adafruit_bno08x", reason="real parser dependency is installed on Jetson")
    log = {"enabled": [], "reports": [], "reads": []}

    class FakeBNO:
        def __init__(self, i2c, address):
            self._readings, self._sequence_number = {}, [0] * 6

        def enable_feature(self, *args):
            log["enabled"].append(args)

        def _process_report(self, report_id, report_bytes):
            log["reports"].append(report_id)
            self._readings[report_id] = (1.0, 2.0, 3.0)

        def _handle_packet(self, packet):
            pass

    for name, attrs in {"adafruit_extended_bus": {"ExtendedI2C": lambda bus: types.SimpleNamespace(deinit=lambda: None)},
                        "adafruit_bno08x.i2c": {"BNO08X_I2C": FakeBNO}}.items():
        fake = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(fake, key, value)
        monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.delitem(sys.modules, "bno085_lean", raising=False)
    monkeypatch.syspath_prepend(str(REPO / "pawan"))
    mod = importlib.import_module("bno085_lean")
    monkeypatch.setattr(mod, "BOOT_DELAY", 0)

    pending = []
    monkeypatch.setattr(mod, "os", types.SimpleNamespace(
        O_RDWR=2, open=lambda path, flags: log["reads"].append(path) or 7, close=lambda fd: None,
        read=lambda fd, n: pending.pop(0) if pending else bytes(n)))
    monkeypatch.setattr(mod, "fcntl", types.SimpleNamespace(ioctl=lambda fd, req, addr: None))
    yield mod, log, pending
    sys.modules.pop("bno085_lean", None)


def _accel_packet(seq=0):
    # 4-byte SHTP header + 5-byte base timestamp (0xFB) + 10-byte accel report (0x01)
    cargo = bytes([0xFB, 0, 0, 0, 0]) + bytes([ACC, seq, 0, 0]) + struct.pack("<hhh", 256, 512, 768)
    return struct.pack("<HBB", len(cargo) + 4, 3, seq) + cargo


def test_lean_reader_requests_10ms_accel_and_gyro(lean):
    mod, log, _ = lean
    imu = mod.LeanBNO085(bus=1, address=0x4A, report_hz=100.0, features=(mod.ACCEL, mod.GYRO))
    assert log["enabled"] == [(ACC, 10_000), (GYRO, 10_000)]
    assert log["reads"] == ["/dev/i2c-1"]
    imu.close()


def test_lean_reader_parses_one_report_in_one_read(lean):
    mod, log, pending = lean
    imu = mod.LeanBNO085(bus=1, address=0x4A, report_hz=100.0, features=(mod.ACCEL, mod.GYRO))
    assert imu.poll() == 0                          # empty FIFO
    pending.append(_accel_packet())
    assert imu.poll() == 1
    assert ACC in log["reports"] and imu.accel == (1.0, 2.0, 3.0)
    assert imu.packets == 1 and imu.continuations == 0


class _NoRunLogger:
    run_id = "test"

    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def stop(self, status=""):
        pass

    def log_sample(self, **kwargs):
        pass


@pytest.mark.skipif(not (RUN_DIR / "best.onnx").exists(), reason="no deploy model")
def test_live_loop_uses_one_observation_per_recorded_tick(fake_lean_module, monkeypatch, tmp_path):
    from exo.deploy.runtime import ExoController

    pushes = []
    real_step = ExoController.step
    def step(self, frame, dt):
        pushes.append(dt)
        return real_step(self, frame, dt)
    monkeypatch.setattr(ExoController, "step", step)
    monkeypatch.setattr(run_logger, "RunLogger", _NoRunLogger)
    out = tmp_path / "live.csv"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--run", str(RUN_DIR), "--mass", "72",
                                      "--dry-run", "--arm", "--duration", "3", "--no-teleplot",
                                      "--print-every", "0", "--out", str(out)])
    fake_can.opened.clear()
    runpy.run_path(str(SCRIPT), run_name="__main__")

    rows = list(csv.DictReader(out.open()))
    assert fake_can.opened == []
    assert len(pushes) == len(rows) > 100
    assert all(0 < dt <= .05 for dt in pushes)
    assert np.allclose(np.array(pushes)*1000,
                       [float(r["tick_ms"]) for r in rows], atol=.001)
    assert all(float(r["sensor_age_ms"]) < 100 for r in rows[5:])
