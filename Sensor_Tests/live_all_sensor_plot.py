#!/usr/bin/env python3
"""
Live Teleplot publisher for all sensors exposed by sensor_parse.SensorHub.

Published streams:
    - IMU-A Gyro: gx, gy, gz (same plot)
    - IMU-B Gyro: gx, gy, gz (same plot)
    - IMU-A Accel: ax, ay, az (same plot)
    - IMU-B Accel: ax, ay, az (same plot)
    - FSRs   (fixed mapping):
            foot <- fsr2 (AIN1)
            heel <- fsr1 (AIN0)
    - Encoder angle (own plot)

Run:
    python3 Sensor_Tests/live_all_sensor_plot.py
    Teleplot: UDP 127.0.0.1:47269

Open VS Code Teleplot extension and listen on UDP 127.0.0.1:47269.
"""

from __future__ import annotations

import time
import socket

from sensor_parse import SensorHub, quat_to_euler


PLOT_HZ = 25
TELEPLOT_HOST = "127.0.0.1"
TELEPLOT_PORT = 47269


class TeleplotClient:
    def __init__(self, host: str = TELEPLOT_HOST, port: int = TELEPLOT_PORT) -> None:
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_value(self, name: str, value: float, timestamp_ms: int) -> None:
        msg = f"{name}:{timestamp_ms}:{value}|g"
        self.sock.sendto(msg.encode("utf-8"), self.addr)

    def close(self) -> None:
        self.sock.close()


class LiveAllSensorsTeleplot:
    def __init__(self) -> None:
        self.interval_s = 1.0 / PLOT_HZ
        self.hub = SensorHub()
        self.teleplot = TeleplotClient()

        self.last = {
            "imu_a": None,
            "imu_b": None,
            "fsr_foot": 0.0,
            "fsr_heel": 0.0,
            "enc": 0.0,
        }

    def _append_imu(self, imu_key: str, imu_sample: dict | None) -> dict[str, float]:
        if imu_sample is not None:
            self.last[imu_key] = imu_sample

        sample = self.last[imu_key]
        if sample is None:
            r = p = y = 0.0
            ax = ay = az = 0.0
            gx = gy = gz = 0.0
        else:
            qi, qj, qk, qr = sample["quat"]
            r, p, y = quat_to_euler(qi, qj, qk, qr)
            ax, ay, az = sample["accel"]
            gx, gy, gz = sample["gyro"]

        return {
            "r": r,
            "p": p,
            "y": y,
            "ax": ax,
            "ay": ay,
            "az": az,
            "gx": gx,
            "gy": gy,
            "gz": gz,
        }

    def publish_once(self) -> None:
        frame = self.hub.read()
        now_ms = int(time.time() * 1000)

        imu_a = self._append_imu("imu_a", frame["imu_a"])
        imu_b = self._append_imu("imu_b", frame["imu_b"])

        # Fixed mapping: foot and heel were interchanged previously.
        self.last["fsr_foot"] = frame["fsr2"]
        self.last["fsr_heel"] = frame["fsr1"]
        self.last["enc"] = frame["angle_deg"]
        telemetry = [
            ("imu_a_gx,IMU A Gyro", imu_a["gx"]),
            ("imu_a_gy,IMU A Gyro", imu_a["gy"]),
            ("imu_a_gz,IMU A Gyro", imu_a["gz"]),
            ("imu_b_gx,IMU B Gyro", imu_b["gx"]),
            ("imu_b_gy,IMU B Gyro", imu_b["gy"]),
            ("imu_b_gz,IMU B Gyro", imu_b["gz"]),
            ("imu_a_ax,IMU A Accel", imu_a["ax"]),
            ("imu_a_ay,IMU A Accel", imu_a["ay"]),
            ("imu_a_az,IMU A Accel", imu_a["az"]),
            ("imu_b_ax,IMU B Accel", imu_b["ax"]),
            ("imu_b_ay,IMU B Accel", imu_b["ay"]),
            ("imu_b_az,IMU B Accel", imu_b["az"]),
            ("toe_fsr,FSR", self.last["fsr_foot"]),
            ("heel_fsr,FSR", self.last["fsr_heel"]),
            ("ankle_encoder_deg,Encoder", self.last["enc"]),
        ]
        for name, value in telemetry:
            self.teleplot.send_value(name, float(value), now_ms)

    def run(self) -> None:
        try:
            while True:
                start = time.time()
                self.publish_once()
                dt = time.time() - start
                sleep_s = self.interval_s - dt
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            self.hub.close()
            self.teleplot.close()


def main() -> None:
    print("Starting Teleplot stream from sensor_parse...")
    print(f"Teleplot UDP target: {TELEPLOT_HOST}:{TELEPLOT_PORT}")
    print("FSR mapping: foot <- fsr2/AIN1, heel <- fsr1/AIN0")
    app = LiveAllSensorsTeleplot()
    app.run()


if __name__ == "__main__":
    main()