"""Stand-ins for the CAN bus and I2C bus, plus helpers for reading sent frames."""
import struct
import threading
import time
import types
from collections import deque

STATUS_ID = 0x2968


class FakeBus:
    """Records sent frames and produces motor status frames about every 20 ms."""

    def __init__(self):
        self.sent = []
        self.rx = deque()
        self.lock = threading.Lock()
        self.fail_sends = False
        self.status_on = True
        self.iq_a = 0.0
        self.error = 0
        self._last_status = 0.0

    def status_frame(self):
        data = struct.pack(">hhhBB", 875, 0, round(self.iq_a * 100), 33, self.error)
        return types.SimpleNamespace(arbitration_id=STATUS_ID, is_extended_id=True, data=data)

    def send(self, msg):
        if self.fail_sends:
            raise OSError("fake CAN failure")
        with self.lock:
            self.sent.append(msg)

    def recv(self, timeout=None):
        if self.rx:
            return self.rx.popleft()
        now = time.monotonic()
        if self.status_on and now - self._last_status >= 0.02:
            self._last_status = now
            return self.status_frame()
        return None

    def shutdown(self):
        pass

    def frames(self):
        with self.lock:
            return list(self.sent)


fake_can = types.ModuleType("can")
fake_can.opened = []


def _open_bus(**kwargs):
    bus = FakeBus()
    fake_can.opened.append(bus)
    return bus


fake_can.Message = lambda **kwargs: types.SimpleNamespace(**kwargs)
fake_can.interface = types.SimpleNamespace(Bus=_open_bus)
fake_can.Bus = _open_bus
fake_can.CanError = OSError


class FakeSMBus:
    ready = True                    # False: the ADC never finishes a conversion

    def __init__(self, bus=None):
        pass

    def write_i2c_block_data(self, addr, reg, data):
        pass

    def read_i2c_block_data(self, addr, reg, count):
        return [0x80 if FakeSMBus.ready else 0x00, 0x00][:count]

    def close(self):
        pass


fake_smbus2 = types.ModuleType("smbus2")
fake_smbus2.SMBus = FakeSMBus


def torque_of(frame):
    """Torque in a sent MIT frame, decoded with the ±22 N·m range."""
    d = frame.data
    return ((d[6] & 0xF) << 8 | d[7]) / 4095 * 44 - 22


def is_special(frame):
    return len(frame.data) == 8 and list(frame.data[:7]) == [0xFF] * 7


def wait_for(condition, timeout=1.0):
    """Seconds until ``condition()`` is true, or None if it never was."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if condition():
            return time.monotonic() - start
        time.sleep(0.002)
    return None
