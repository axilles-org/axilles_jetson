#!/usr/bin/env python3
"""
Live plotting dashboard for all sensors exposed by sensor_parse.SensorHub.

Plotted streams:
  - IMU-A: roll/pitch/yaw, accel xyz, gyro xyz
  - IMU-B: roll/pitch/yaw, accel xyz, gyro xyz
  - FSRs   (fixed mapping):
      foot <- fsr2 (AIN1)
      heel <- fsr1 (AIN0)
  - Encoder angle (deg)

Run:
  python3 Sensor_Tests/live_all_sensors_plot.py
"""

from __future__ import annotations

import time
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sensor_parse import SensorHub, quat_to_euler


PLOT_HZ = 25
WINDOW_SECONDS = 12


def _new_buf(maxlen: int):
    return {
        "t": deque(maxlen=maxlen),
        "imu_a_rpy": {"r": deque(maxlen=maxlen), "p": deque(maxlen=maxlen), "y": deque(maxlen=maxlen)},
        "imu_b_rpy": {"r": deque(maxlen=maxlen), "p": deque(maxlen=maxlen), "y": deque(maxlen=maxlen)},
        "imu_a_acc": {"x": deque(maxlen=maxlen), "y": deque(maxlen=maxlen), "z": deque(maxlen=maxlen)},
        "imu_b_acc": {"x": deque(maxlen=maxlen), "y": deque(maxlen=maxlen), "z": deque(maxlen=maxlen)},
        "imu_a_gyr": {"x": deque(maxlen=maxlen), "y": deque(maxlen=maxlen), "z": deque(maxlen=maxlen)},
        "imu_b_gyr": {"x": deque(maxlen=maxlen), "y": deque(maxlen=maxlen), "z": deque(maxlen=maxlen)},
        "fsr_foot": deque(maxlen=maxlen),
        "fsr_heel": deque(maxlen=maxlen),
        "enc": deque(maxlen=maxlen),
    }


class LiveAllSensorsPlot:
    def __init__(self) -> None:
        self.interval_s = 1.0 / PLOT_HZ
        self.maxlen = int(PLOT_HZ * WINDOW_SECONDS)
        self.buf = _new_buf(self.maxlen)
        self.t0 = time.time()

        self.hub = SensorHub()

        self.last = {
            "imu_a": None,
            "imu_b": None,
            "fsr_foot": 0.0,
            "fsr_heel": 0.0,
            "enc": 0.0,
        }

        self.fig, self.axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
        self.fig.suptitle("Live Sensor Dashboard (sensor_parse)")

        ax = self.axes
        self.lines = {}

        self.lines["a_r"] = ax[0, 0].plot([], [], label="roll", color="tab:blue")[0]
        self.lines["a_p"] = ax[0, 0].plot([], [], label="pitch", color="tab:orange")[0]
        self.lines["a_y"] = ax[0, 0].plot([], [], label="yaw", color="tab:green")[0]
        ax[0, 0].set_title("IMU-A Euler (deg)")
        ax[0, 0].grid(True, alpha=0.3)
        ax[0, 0].legend(loc="upper right", ncol=3, fontsize=8)

        self.lines["b_r"] = ax[0, 1].plot([], [], label="roll", color="tab:blue")[0]
        self.lines["b_p"] = ax[0, 1].plot([], [], label="pitch", color="tab:orange")[0]
        self.lines["b_y"] = ax[0, 1].plot([], [], label="yaw", color="tab:green")[0]
        ax[0, 1].set_title("IMU-B Euler (deg)")
        ax[0, 1].grid(True, alpha=0.3)
        ax[0, 1].legend(loc="upper right", ncol=3, fontsize=8)

        self.lines["a_ax"] = ax[1, 0].plot([], [], label="ax", color="tab:red")[0]
        self.lines["a_ay"] = ax[1, 0].plot([], [], label="ay", color="tab:purple")[0]
        self.lines["a_az"] = ax[1, 0].plot([], [], label="az", color="tab:brown")[0]
        ax[1, 0].set_title("IMU-A Accel (m/s²)")
        ax[1, 0].grid(True, alpha=0.3)
        ax[1, 0].legend(loc="upper right", ncol=3, fontsize=8)

        self.lines["b_ax"] = ax[1, 1].plot([], [], label="ax", color="tab:red")[0]
        self.lines["b_ay"] = ax[1, 1].plot([], [], label="ay", color="tab:purple")[0]
        self.lines["b_az"] = ax[1, 1].plot([], [], label="az", color="tab:brown")[0]
        ax[1, 1].set_title("IMU-B Accel (m/s²)")
        ax[1, 1].grid(True, alpha=0.3)
        ax[1, 1].legend(loc="upper right", ncol=3, fontsize=8)

        self.lines["a_gx"] = ax[2, 0].plot([], [], label="gx", color="tab:cyan")[0]
        self.lines["a_gy"] = ax[2, 0].plot([], [], label="gy", color="tab:pink")[0]
        self.lines["a_gz"] = ax[2, 0].plot([], [], label="gz", color="tab:gray")[0]
        ax[2, 0].set_title("IMU-A Gyro (rad/s)")
        ax[2, 0].grid(True, alpha=0.3)
        ax[2, 0].legend(loc="upper right", ncol=3, fontsize=8)

        self.lines["b_gx"] = ax[2, 1].plot([], [], label="gx", color="tab:cyan")[0]
        self.lines["b_gy"] = ax[2, 1].plot([], [], label="gy", color="tab:pink")[0]
        self.lines["b_gz"] = ax[2, 1].plot([], [], label="gz", color="tab:gray")[0]
        ax[2, 1].set_title("IMU-B Gyro (rad/s)")
        ax[2, 1].grid(True, alpha=0.3)
        ax[2, 1].legend(loc="upper right", ncol=3, fontsize=8)

        self.lines["fsr_foot"] = ax[3, 0].plot([], [], label="foot (AIN1/fsr2)", color="tab:red")[0]
        self.lines["fsr_heel"] = ax[3, 0].plot([], [], label="heel (AIN0/fsr1)", color="tab:green")[0]
        ax[3, 0].set_title("FSR Voltage (V)")
        ax[3, 0].set_xlabel("Time (s)")
        ax[3, 0].grid(True, alpha=0.3)
        ax[3, 0].legend(loc="upper right", fontsize=8)

        self.lines["enc"] = ax[3, 1].plot([], [], label="angle", color="tab:orange")[0]
        ax[3, 1].set_title("AS5600 Angle (deg)")
        ax[3, 1].set_xlabel("Time (s)")
        ax[3, 1].set_ylim(0, 360)
        ax[3, 1].grid(True, alpha=0.3)
        ax[3, 1].legend(loc="upper right", fontsize=8)

    def _append_imu(self, imu_key: str, imu_sample: dict | None) -> None:
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

        if imu_key == "imu_a":
            self.buf["imu_a_rpy"]["r"].append(r)
            self.buf["imu_a_rpy"]["p"].append(p)
            self.buf["imu_a_rpy"]["y"].append(y)
            self.buf["imu_a_acc"]["x"].append(ax)
            self.buf["imu_a_acc"]["y"].append(ay)
            self.buf["imu_a_acc"]["z"].append(az)
            self.buf["imu_a_gyr"]["x"].append(gx)
            self.buf["imu_a_gyr"]["y"].append(gy)
            self.buf["imu_a_gyr"]["z"].append(gz)
        else:
            self.buf["imu_b_rpy"]["r"].append(r)
            self.buf["imu_b_rpy"]["p"].append(p)
            self.buf["imu_b_rpy"]["y"].append(y)
            self.buf["imu_b_acc"]["x"].append(ax)
            self.buf["imu_b_acc"]["y"].append(ay)
            self.buf["imu_b_acc"]["z"].append(az)
            self.buf["imu_b_gyr"]["x"].append(gx)
            self.buf["imu_b_gyr"]["y"].append(gy)
            self.buf["imu_b_gyr"]["z"].append(gz)

    def _autoscale(self, axis, series_list):
        if not series_list or not series_list[0]:
            return
        y_min = min(min(series) for series in series_list)
        y_max = max(max(series) for series in series_list)
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        pad = 0.1 * (y_max - y_min)
        axis.set_ylim(y_min - pad, y_max + pad)

    def update(self, _frame_idx: int):
        frame = self.hub.read()
        t_now = time.time() - self.t0
        self.buf["t"].append(t_now)

        self._append_imu("imu_a", frame["imu_a"])
        self._append_imu("imu_b", frame["imu_b"])

        # Fixed mapping: foot and heel were interchanged previously.
        self.last["fsr_foot"] = frame["fsr2"]
        self.last["fsr_heel"] = frame["fsr1"]
        self.last["enc"] = frame["angle_deg"]

        self.buf["fsr_foot"].append(self.last["fsr_foot"])
        self.buf["fsr_heel"].append(self.last["fsr_heel"])
        self.buf["enc"].append(self.last["enc"])

        t = self.buf["t"]
        self.lines["a_r"].set_data(t, self.buf["imu_a_rpy"]["r"])
        self.lines["a_p"].set_data(t, self.buf["imu_a_rpy"]["p"])
        self.lines["a_y"].set_data(t, self.buf["imu_a_rpy"]["y"])
        self.lines["b_r"].set_data(t, self.buf["imu_b_rpy"]["r"])
        self.lines["b_p"].set_data(t, self.buf["imu_b_rpy"]["p"])
        self.lines["b_y"].set_data(t, self.buf["imu_b_rpy"]["y"])

        self.lines["a_ax"].set_data(t, self.buf["imu_a_acc"]["x"])
        self.lines["a_ay"].set_data(t, self.buf["imu_a_acc"]["y"])
        self.lines["a_az"].set_data(t, self.buf["imu_a_acc"]["z"])
        self.lines["b_ax"].set_data(t, self.buf["imu_b_acc"]["x"])
        self.lines["b_ay"].set_data(t, self.buf["imu_b_acc"]["y"])
        self.lines["b_az"].set_data(t, self.buf["imu_b_acc"]["z"])

        self.lines["a_gx"].set_data(t, self.buf["imu_a_gyr"]["x"])
        self.lines["a_gy"].set_data(t, self.buf["imu_a_gyr"]["y"])
        self.lines["a_gz"].set_data(t, self.buf["imu_a_gyr"]["z"])
        self.lines["b_gx"].set_data(t, self.buf["imu_b_gyr"]["x"])
        self.lines["b_gy"].set_data(t, self.buf["imu_b_gyr"]["y"])
        self.lines["b_gz"].set_data(t, self.buf["imu_b_gyr"]["z"])

        self.lines["fsr_foot"].set_data(t, self.buf["fsr_foot"])
        self.lines["fsr_heel"].set_data(t, self.buf["fsr_heel"])
        self.lines["enc"].set_data(t, self.buf["enc"])

        if t:
            x_min = max(0.0, t[-1] - WINDOW_SECONDS)
            x_max = max(WINDOW_SECONDS, t[-1])
            for row in self.axes:
                for axis in row:
                    axis.set_xlim(x_min, x_max)

            self._autoscale(self.axes[0, 0], [self.buf["imu_a_rpy"]["r"], self.buf["imu_a_rpy"]["p"], self.buf["imu_a_rpy"]["y"]])
            self._autoscale(self.axes[0, 1], [self.buf["imu_b_rpy"]["r"], self.buf["imu_b_rpy"]["p"], self.buf["imu_b_rpy"]["y"]])
            self._autoscale(self.axes[1, 0], [self.buf["imu_a_acc"]["x"], self.buf["imu_a_acc"]["y"], self.buf["imu_a_acc"]["z"]])
            self._autoscale(self.axes[1, 1], [self.buf["imu_b_acc"]["x"], self.buf["imu_b_acc"]["y"], self.buf["imu_b_acc"]["z"]])
            self._autoscale(self.axes[2, 0], [self.buf["imu_a_gyr"]["x"], self.buf["imu_a_gyr"]["y"], self.buf["imu_a_gyr"]["z"]])
            self._autoscale(self.axes[2, 1], [self.buf["imu_b_gyr"]["x"], self.buf["imu_b_gyr"]["y"], self.buf["imu_b_gyr"]["z"]])
            self._autoscale(self.axes[3, 0], [self.buf["fsr_foot"], self.buf["fsr_heel"]])

        return tuple(self.lines.values())

    def run(self):
        anim = FuncAnimation(
            self.fig,
            self.update,
            interval=int(self.interval_s * 1000),
            blit=False,
            cache_frame_data=False,
        )
        try:
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])
            plt.show()
        finally:
            self.hub.close()
            del anim


def main() -> None:
    print("Starting live all-sensor dashboard from sensor_parse...")
    print("FSR mapping fixed: foot <- fsr2/AIN1, heel <- fsr1/AIN0")
    app = LiveAllSensorsPlot()
    app.run()


if __name__ == "__main__":
    main()
