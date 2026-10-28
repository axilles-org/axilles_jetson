"""Simulated hardware for running Test 4 without the exo (--sim).

Only for checking the test pipeline itself (loop, logging, CAN capture,
analysis) on a machine with no sensors or motor. The TCN inference is still the
real model, so --sim does show what the controller step costs on the CPU. The
sensor read times below are placeholders, NOT measurements — use a hardware run
for any number that goes in the report.
"""

import time

# Cross-process virtual CAN bus (python-can udp_multicast, needs msgpack).
SIM_CAN_INTERFACE = "udp_multicast"
SIM_CAN_CHANNEL = "239.74.163.2"

IMU_READ_S = 0.0007       # per BNO085 _process_available_packets() burst
FSR_ENC_READ_S = 0.0030   # 2 ADS1115 single-shot conversions + AS5600 read


class SimSensors:
    """Same read() contract as test4_run.HardwareSensors."""

    def __init__(self, frame_keys, imu_read_s=IMU_READ_S, fsr_enc_read_s=FSR_ENC_READ_S):
        self._imu_s = imu_read_s
        self._fse_s = fsr_enc_read_s
        self._frame = {k: 0.0 for k in frame_keys}
        self._frame.update(foot_az=9.81, shank_az=9.81, ankle_encoder_deg=85.0,
                           heel_fsr_raw=12000.0, toe_fsr_raw=5000.0)

    def read(self):
        t0 = time.perf_counter()
        time.sleep(2 * self._imu_s)
        t1 = time.perf_counter()
        time.sleep(self._fse_s)
        t2 = time.perf_counter()
        return dict(self._frame), t1 - t0, t2 - t1

    def shutdown(self):
        pass


def make_sim_motor(MotorInterface, can_module, **kwargs):
    """A real MotorInterface whose CAN bus is the virtual one.

    MotorInterface opens socketcan itself, so it is built in dry-run mode and then
    given the virtual bus; every send then runs the same code as on hardware.
    """
    motor = MotorInterface(dry_run=True, **kwargs)
    motor._can = can_module
    motor._bus = can_module.Bus(interface=SIM_CAN_INTERFACE, channel=SIM_CAN_CHANNEL,
                                receive_own_messages=False)
    motor.dry_run = False
    return motor
