#!/usr/bin/env python3
"""
FSR ADC test — ADS1115 (foot + heel).

Reads Force‑Sensitive Resistor values from:
    HEEL FSR → AIN0 (single-ended vs GND)
    FOOT FSR → AIN1 (single-ended vs GND)

Shows raw ADC counts + voltage and runs a basic activity check:
if a sensor changes by at least DELTA_ACTIVE_COUNTS from startup baseline,
it is marked ACTIVE.

Press Ctrl+C to stop.
"""

import smbus2
import time
import struct

# ── Hardware ──────────────────────────────────────────────────────────────────
I2C_BUS      = 1
ADC_ADDR     = 0x48   # ADS1115 default address (ADDR pin → GND)

# ── ADS1115 Registers ─────────────────────────────────────────────────────────
REG_CONVERSION = 0x00
REG_CONFIG     = 0x01

# ── Config register templates (high byte, low byte) ──────────────────────────
# High byte:  OS=1 | MUX[2:0] | PGA=001 (±4.096 V) | MODE=1 (single‑shot)
# Low byte:   DR=100 (128 SPS) | COMP_MODE=0 | COMP_POL=0 | COMP_LAT=0 | COMP_QUE=11 (disabled)
_CFG_LO  = 0x83   # 128 SPS, comparator disabled

CFG_AIN0 = [0xC3, _CFG_LO]   # OS=1, MUX=100 (AIN0 vs GND)
CFG_AIN1 = [0xD3, _CFG_LO]   # OS=1, MUX=101 (AIN1 vs GND)

# ── Scale ─────────────────────────────────────────────────────────────────────
VOLTS_PER_COUNT = 4.096 / 32767   # ~0.125 mV per count

# ── Sample interval ───────────────────────────────────────────────────────────
SAMPLE_HZ  = 200          # target print rate
CONV_DELAY = 1 / 128 + 0.002   # >1 conversion period @ 128 SPS + margin
DELTA_ACTIVE_COUNTS = 300


def read_channel(bus: smbus2.SMBus, config: list[int]) -> int:
    """Trigger a single-shot conversion and return the signed 16-bit raw count."""
    # Write config register to start conversion
    bus.write_i2c_block_data(ADC_ADDR, REG_CONFIG, config)
    # Wait for conversion to complete
    time.sleep(CONV_DELAY)
    # Read 2 bytes from conversion register
    data = bus.read_i2c_block_data(ADC_ADDR, REG_CONVERSION, 2)
    # Big-endian signed 16-bit
    raw = struct.unpack(">h", bytes(data))[0]
    return raw


def state_from_delta(delta: int) -> str:
    return "ACTIVE" if abs(delta) >= DELTA_ACTIVE_COUNTS else "idle  "


def main():
    interval = 1.0 / SAMPLE_HZ

    print(f"ADS1115 FSR test — bus {I2C_BUS}, addr 0x{ADC_ADDR:02X}")
    print(f"PGA ±4.096 V | 128 SPS | print rate {SAMPLE_HZ} Hz")
    print(f"Activity threshold: |Δ| >= {DELTA_ACTIVE_COUNTS} counts")
    print(
        f"{'FOOT cnt':>9}  {'FOOT V':>8}  {'Δfoot':>7}  {'state':>6}"
        f"   {'HEEL cnt':>9}  {'HEEL V':>8}  {'Δheel':>7}  {'state':>6}"
    )
    print("-" * 84)

    with smbus2.SMBus(I2C_BUS) as bus:
        try:
            baseline_heel = read_channel(bus, CFG_AIN0)
            baseline_foot = read_channel(bus, CFG_AIN1)

            print(
                f"Baseline set  FOOT={baseline_foot}  HEEL={baseline_heel}"
            )

            while True:
                t0 = time.monotonic()

                raw_heel = read_channel(bus, CFG_AIN0)
                raw_foot = read_channel(bus, CFG_AIN1)

                v_foot = max(raw_foot, 0) * VOLTS_PER_COUNT
                v_heel = max(raw_heel, 0) * VOLTS_PER_COUNT

                delta_foot = raw_foot - baseline_foot
                delta_heel = raw_heel - baseline_heel

                foot_state = state_from_delta(delta_foot)
                heel_state = state_from_delta(delta_heel)

                print(
                    f"{raw_foot:>9d}  {v_foot:>8.4f}  {delta_foot:>+7d}  {foot_state:>6}"
                    f"   {raw_heel:>9d}  {v_heel:>8.4f}  {delta_heel:>+7d}  {heel_state:>6}"
                )

                elapsed = time.monotonic() - t0
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
