# file for all the parameters and utility functions of the TBE controller
import logging
import time
import numpy as np

# Creating a logger class for controller
class Logger():

    def __init__(self):
        self.logger = logging.getLogger("TBEController")
        self._init_time = time.perf_counter()
        self.logger.handlers.clear()    # prevent duplicate handlers on re-init
        self.setFormat()

        self.logger.info("Logger initialized.")

    def getCurrentTime(self):
        return time.perf_counter() - self._init_time

    def _get_timestamp(self):
        elapsed_time = self.getCurrentTime()
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        milliseconds = int((elapsed_time - int(elapsed_time)) * 1000)
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def setFormat(self):

        # Custom formatter that uses our elapsed time
        logger_ref = self  # capture reference for the inner class

        class ElapsedFormatter(logging.Formatter):
            def format(self, record):
                timestamp = logger_ref._get_timestamp()
                level = record.levelname.ljust(4)
                name = record.name
                msg = record.getMessage()
                return f"[{level}][{timestamp}][{name}]: {msg}"

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(ElapsedFormatter())
        self.logger.addHandler(console)
        self.logger.setLevel(logging.DEBUG)


# The number of strides over which TBE takes average
NUM_STRIDES = 5
ADAPTIVE_STRIDE_WINDOW = 0

# THresholds for heel strike and toe off detection
HEEL_STRIKE_THRESHOLD = 26100 
TOE_OFF_THRESHOLD = 17000

# ── Hardware ──────────────────────────────────────────────────────────────────
I2C_BUS      = 1
ADC_ADDR     = 0x48   # ADS1115 default address (ADDR pin → GND)

# ── ADS1115 Registers ─────────────────────────────────────────────────────────
REG_CONVERSION = 0x00
REG_CONFIG     = 0x01

# AS5600 Encoder
AS5600_ADDR      = 0x36
AS5600_REG_ANGLE = 0x0E   # filtered 12-bit angle (MSB + LSB)

# ── Config register templates (high byte, low byte) ──────────────────────────
# High byte:  OS=1 | MUX[2:0] | PGA=001 (±4.096 V) | MODE=1 (single‑shot)
# Low byte:   DR=100 (128 SPS) | COMP_MODE=0 | COMP_POL=0 | COMP_LAT=0 | COMP_QUE=11 (disabled)
_CFG_LO  = 0xE3   # 860 SPS, comparator disabled

CFG_AIN0 = [0xC3, _CFG_LO]   # OS=1, MUX=100 (AIN0 vs GND)
CFG_AIN1 = [0xD3, _CFG_LO]   # OS=1, MUX=101 (AIN1 vs GND)

# ── Scale ─────────────────────────────────────────────────────────────────────
VOLTS_PER_COUNT = 4.096 / 32767   # ~0.125 mV per count

# ── Sample interval ───────────────────────────────────────────────────────────
MOTOR_CONTROL_FREQ = 150

# Duty time of motor control loop
DT = 1.0 / MOTOR_CONTROL_FREQ

# cut off frequency for low-pass fsr filtering (Hz)
FSR_FILTER_CUTOFF = 20.0
ENC_VEL_CUTOFF = 20.0

# Encoder calibration and limit values
ENCODER_OFFSET = 165.0
PLANTARFLEXION_LIMIT = 30.0
DORSIFLEXION_LIMIT = 15.0 

# Impedance controller PD values
KP_IMPEDANCE = 0.0
KD_IMPEDANCE = 0.0

# Peak torque for torque profile during activation (Nm)

PEAK_TORQUE = 7.0 # DO NOT INCREASE ABOVE 10.0 WITHOUT PROPER TESTING

# Looks like the assistance is toe in and toe off
TAU_PHASE_ARRAY = np.array([0.0, 0.6, 0.7, 0.8, 0.9, 1.0])
TAU_VAL_ARRAY = np.array([0.0, 0.0, 0.2,  1.0,  0.2,  0.0])

# Assistance percentage for torque profile during activation
ASSISTANCE_LEVEL = 0.5

# Teleplot UDP configuration
TELEPLOT_HOST = "127.0.0.1"
TELEPLOT_PORT = 47269
