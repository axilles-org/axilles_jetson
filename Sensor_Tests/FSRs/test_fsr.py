#!/usr/bin/env python3
"""
FSR ADC test — ADS1115 on I2C bus 7, address 0x48.

Reads Force‑Sensitive Resistor (FSR) values from:
  A0  →  AIN0 (single‑ended vs GND)
  A1  →  AIN1 (single‑ended vs GND)

Prints raw ADC counts and the corresponding voltage at a fixed sample rate.
Press Ctrl+C to stop.

ADS1115 config used:
  PGA  = ±4.096 V  (1 LSB = 0.125 mV, covers 0–3.3 V signals)
  Mode = single‑shot
  DR   = 128 SPS
  Comparator disabled
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


def main():
    interval = 1.0 / SAMPLE_HZ

    print(f"ADS1115 FSR test — bus {I2C_BUS}, addr 0x{ADC_ADDR:02X}")
    print(f"PGA ±4.096 V | 128 SPS | printing at {SAMPLE_HZ} Hz")
    print(f"{'A0 counts':>12}  {'A0 volts':>10}  {'A1 counts':>12}  {'A1 volts':>10}")
    print("-" * 52)

    with smbus2.SMBus(I2C_BUS) as bus:
        try:
            while True:
                t0 = time.monotonic()

                raw0 = read_channel(bus, CFG_AIN0)
                raw1 = read_channel(bus, CFG_AIN1)

                v0 = max(raw0, 0) * VOLTS_PER_COUNT
                v1 = max(raw1, 0) * VOLTS_PER_COUNT

                print(f"{raw0:>12d}  {v0:>10.4f}  {raw1:>12d}  {v1:>10.4f}")

                elapsed = time.monotonic() - t0
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
