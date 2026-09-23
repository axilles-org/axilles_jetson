#!/usr/bin/env python3
"""
bno085_lean.py

Lean BNO085 reader: one I2C read per packet, accel + gyro only.

Why the stock adafruit_bno08x reader is slow
--------------------------------------------
For every packet the stock _process_available_packets() does:
    4-byte header read   (_data_ready: "is anything there?")
    4-byte header read   (_read_packet: read the header again)
    full packet read     (header + payload, a third time)
and one more 4-byte read at the end to find the FIFO empty. On this bus a read
costs ~0.09 ms + ~0.031 ms per byte, so a 19-byte report packet costs ~1.2 ms.

What this reader does instead
-----------------------------
    * Reads READ_LEN bytes in ONE transaction. A single accel or gyro report
      arrives as a 19-byte packet (4 header + 5 timestamp + 10 report), so one
      19-byte read gets the whole packet: ~0.6 ms instead of ~1.2 ms.
    * If the header says the packet is longer (batched reports), the BNO085
      delivers the rest on the next read as a continuation (bit 15 of the
      length set). Exactly the remaining bytes are read and stitched back on.
    * An empty FIFO returns length 0, so "no data" costs the same one read.
    * Reads go straight to /dev/i2c-N with os.read(), bypassing Blinka.
    * The quaternion (game rotation vector) is not enabled by default, so each
      IMU sends 2 x report_hz packets instead of 3 x report_hz.

Setup (soft reset, enabling features) still uses the adafruit driver, and
report parsing reuses its parser, so values match the stock reader exactly.

The BNO085's INT pin would let us skip empty polls entirely, but it is not
wired to the Jetson on this board; polling with a single read is used instead.

Usage
-----
    from bno085_lean import LeanBNO085

    imu = LeanBNO085(bus=1, address=0x4A, report_hz=200)
    while True:
        if imu.poll():                # True when at least one report arrived
            ax, ay, az = imu.accel
            gx, gy, gz = imu.gyro
    imu.close()
"""

import fcntl
import os
import struct
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="I2C frequency is not settable")

from adafruit_extended_bus import ExtendedI2C as I2C
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_GAME_ROTATION_VECTOR,
    Packet,
    _separate_batch,
)

I2C_SLAVE = 0x0703          # ioctl: bind this fd to a device address
READ_LEN = 19               # one timestamped accel/gyro report, exactly
BOOT_DELAY = 0.8
CONNECT_RETRIES = 6
INPUT_REPORT_CHANNEL = 3    # SHTP channel that carries sensor reports

ACCEL = BNO_REPORT_ACCELEROMETER
GYRO = BNO_REPORT_GYROSCOPE
QUAT = BNO_REPORT_GAME_ROTATION_VECTOR


class LeanBNO085:
    """One BNO085, read with a single I2C transaction per packet."""

    def __init__(self, bus, address, report_hz=200.0,
                 features=(ACCEL, GYRO), label=None):
        self.bus = bus
        self.address = address
        self.report_hz = report_hz
        self.features = tuple(features)
        self.label = label or f"0x{address:02X}"

        self._i2c = None
        self._bno = None
        self._fd = None
        self._buf = bytearray(READ_LEN)
        self._slices = []

        # Stats, reset with reset_stats().
        self.counts = {}            # report_id -> reports received
        self.packets = 0            # packets that carried data
        self.empty_polls = 0        # polls that found the FIFO empty
        self.continuations = 0      # extra reads for packets > READ_LEN
        self.seq_gaps = 0           # missing packets on the report channel
        self.busy_s = 0.0           # time spent in reads that returned data
        self._last_seq = None

        self._connect()

    # ── setup ────────────────────────────────────────────────────────────────
    def _connect(self):
        interval_us = max(1000, int(1_000_000 / self.report_hz))
        for attempt in range(1, CONNECT_RETRIES + 1):
            try:
                self._close_handles()
                self._i2c = I2C(self.bus)
                self._bno = BNO08X_I2C(self._i2c, address=self.address)
                time.sleep(BOOT_DELAY)
                for feat in self.features:
                    self._bno.enable_feature(feat, interval_us)
                self._fd = os.open(f"/dev/i2c-{self.bus}", os.O_RDWR)
                fcntl.ioctl(self._fd, I2C_SLAVE, self.address)
                self._last_seq = None
                return
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[{self.label}] connect attempt {attempt} failed: {exc}")
                time.sleep(1.0)
        raise RuntimeError(f"[{self.label}] could not connect on bus {self.bus}")

    def reconnect(self):
        self._connect()

    def _close_handles(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        if self._i2c is not None:
            try:
                self._i2c.deinit()
            except Exception:
                pass
            self._i2c = None

    def close(self):
        self._close_handles()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── latest values ────────────────────────────────────────────────────────
    @property
    def accel(self):
        return self._bno._readings.get(ACCEL)

    @property
    def gyro(self):
        return self._bno._readings.get(GYRO)

    @property
    def quat(self):
        return self._bno._readings.get(QUAT)

    def reset_stats(self):
        self.counts.clear()
        self.packets = self.empty_polls = self.continuations = 0
        self.seq_gaps = 0
        self.busy_s = 0.0

    # ── reading ──────────────────────────────────────────────────────────────
    def poll(self):
        """
        One read. Returns the number of reports processed (0 if the FIFO was
        empty). Raises OSError on bus errors; call reconnect() to recover.
        """
        t0 = time.perf_counter()
        data = os.read(self._fd, READ_LEN)
        raw_len, channel, seq = struct.unpack_from("<HBB", data)
        length = raw_len & 0x7FFF

        if length == 0 or length == 0x7FFF:
            self.empty_polls += 1
            return 0

        if channel == INPUT_REPORT_CHANNEL:
            if self._last_seq is not None and seq != (self._last_seq + 1) & 0xFF:
                self.seq_gaps += (seq - self._last_seq - 1) & 0xFF
            self._last_seq = seq

        cargo = bytearray(data[4:min(length, READ_LEN)])
        remaining = length - READ_LEN
        while remaining > 0:
            # The BNO085 sends the rest with a fresh 4-byte header, which also
            # advances the sequence number.
            more = os.read(self._fd, remaining + 4)
            self.continuations += 1
            cont_len = struct.unpack_from("<H", more)[0] & 0x7FFF
            cargo += more[4:min(cont_len, remaining + 4)]
            remaining = cont_len - (remaining + 4)
            if channel == INPUT_REPORT_CHANNEL:
                self._last_seq = more[3]
        self.busy_s += time.perf_counter() - t0
        self.packets += 1

        return self._handle(channel, seq, cargo)

    def _handle(self, channel, seq, cargo):
        # Rebuild a whole packet so the adafruit parser can be reused.
        buf = bytearray(struct.pack("<HBB", len(cargo) + 4, channel, seq))
        buf += cargo
        packet = Packet(buf)
        self._bno._sequence_number[channel] = seq

        if channel != INPUT_REPORT_CHANNEL:
            self._bno._handle_packet(packet)      # control / command traffic
            return 0

        self._slices.clear()
        _separate_batch(packet, self._slices)
        n = 0
        for report_id, report_bytes in self._slices:
            self._bno._process_report(report_id, report_bytes)
            if report_id in self.features:
                self.counts[report_id] = self.counts.get(report_id, 0) + 1
                n += 1
        return n
