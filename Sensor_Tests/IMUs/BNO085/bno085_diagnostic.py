#!/usr/bin/env python3
"""
BNO085 Comprehensive I2C Diagnostic Script
Tests multiple initialization sequences, timings, and configurations
to find what actually works on Jetson Orin Nano.
"""

import time
import struct
import sys
import os
from smbus2 import SMBus, i2c_msg
from collections import defaultdict

# ─── CONFIG MATRIX ────────────────────────────────────────────────────────────
I2C_BUSES      = [7]            # Add 1, 8 if you want to test other buses
I2C_ADDR       = 0x4A
REPORT_IDS     = [0x01, 0x02, 0x05, 0x08, 0x14]  # accel, gyro, RV, game RV, raw accel
INIT_DELAYS    = [0.3, 0.5, 0.7, 1.0]  # seconds to wait after reset
INTER_CMD_DELAYS = [0.05, 0.1, 0.2, 0.3]  # seconds between commands
REPORT_INTERVALS_MS = [20, 50, 100]  # ms between sensor reports

# ─── SHTP CONSTANTS ────────────────────────────────────────────────────────────
CH_CMD       = 0   # Advertisement/error
CH_EXEC      = 1   # Executable: reset, sleep
CH_CONTROL   = 2   # Set features, commands
CH_REPORTS   = 3   # Sensor data
CH_WAKE      = 4   # Wake sensor data
CH_GYRO_RV   = 5   # Gyro rotation vector dedicated channel

# Report IDs
SET_FEATURE_CMD       = 0xFD
CMD_REQUEST           = 0xF2
CMD_INITIALIZE        = 0x04
PRODUCT_ID_REQUEST    = 0xF9
PRODUCT_ID_RESPONSE   = 0xF8
CMD_RESPONSE          = 0xF3
SHTP_EXECUTABLE_RESET = 0x01
BASE_TIMESTAMP        = 0xFB
EXEC_ON               = 0x02

REPORT_NAMES = {
    0x01: "Accelerometer",
    0x02: "Gyroscope",
    0x03: "Magnetometer",
    0x04: "Linear Accel",
    0x05: "Rotation Vector",
    0x08: "Game Rotation Vector",
    0x0A: "Gravity",
    0x14: "Raw Accelerometer",
    0x15: "Raw Gyro",
    0xFB: "Base Timestamp",
    0xF8: "Product ID Response",
    0xF3: "Command Response",
    0xF2: "Command Request",
    0xF9: "Product ID Request",
}

# ─── CORE I2C CLASS ────────────────────────────────────────────────────────────
class SHTP:
    def __init__(self, bus_num, addr):
        self.bus = SMBus(bus_num)
        self.addr = addr
        self.seq = [0] * 8
        self.stats = defaultdict(int)
        time.sleep(0.1)

    def write(self, data: bytes):
        msg = i2c_msg.write(self.addr, data)
        self.bus.i2c_rdwr(msg)

    def read(self, n: int) -> bytes:
        msg = i2c_msg.read(self.addr, n)
        self.bus.i2c_rdwr(msg)
        return bytes(msg)

    def send(self, channel: int, payload: bytes):
        length = len(payload) + 4
        header = struct.pack("<HBB", length, channel, self.seq[channel] & 0xFF)
        self.seq[channel] = (self.seq[channel] + 1) & 0xFF
        self.write(header + payload)
        self.stats[f"sent_ch{channel}"] += 1

    def recv(self, timeout=0.5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                hdr = self.read(4)
                length, channel, seq = struct.unpack("<HBB", hdr)
                length &= 0x7FFF
                if length == 0 or length < 4 or length > 1024:
                    time.sleep(0.002)
                    continue
                payload = b"" if length == 4 else self.read(length - 4)
                report_id = payload[0] if payload else 0
                self.stats[f"recv_ch{channel}_0x{report_id:02X}"] += 1
                return channel, seq, payload
            except:
                time.sleep(0.002)
        return None

    def drain(self, duration=0.5, verbose=False):
        t0 = time.time()
        packets = []
        while time.time() - t0 < duration:
            pkt = self.recv(timeout=0.05)
            if pkt:
                packets.append(pkt)
                if verbose:
                    ch, seq, payload = pkt
                    rid = payload[0] if payload else 0
                    name = REPORT_NAMES.get(rid, f"0x{rid:02X}")
                    print(f"    [drain] CH={ch} ID={name} len={len(payload)}")
        return packets

    def close(self):
        self.bus.close()


# ─── SEQUENCE STEPS ────────────────────────────────────────────────────────────
def step_soft_reset(imu):
    """Send soft reset via executable channel"""
    imu.send(CH_EXEC, bytes([0x01]))

def step_wake_on(imu):
    """Send 'ON' command via executable channel"""
    imu.send(CH_EXEC, bytes([EXEC_ON]))

def step_initialize(imu):
    """Send COMMAND_INITIALIZE (0x04) on control channel"""
    payload = bytearray(12)
    payload[0] = CMD_REQUEST
    payload[1] = 0  # sequence
    payload[2] = CMD_INITIALIZE
    imu.send(CH_CONTROL, bytes(payload))

def step_product_id(imu):
    """Request product ID"""
    imu.send(CH_CONTROL, bytes([PRODUCT_ID_REQUEST, 0x00]))

def step_enable_report(imu, report_id, interval_ms):
    """Enable a sensor report"""
    interval_us = interval_ms * 1000
    payload = bytearray(17)
    payload[0] = SET_FEATURE_CMD
    payload[1] = report_id
    payload[5:9] = struct.pack("<I", interval_us)
    imu.send(CH_CONTROL, bytes(payload))


# ─── PACKET ANALYSIS ───────────────────────────────────────────────────────────
def analyze_packet(ch, payload, results):
    if not payload:
        return
    rid = payload[0]
    
    if rid == PRODUCT_ID_RESPONSE and len(payload) >= 14:
        major = payload[2]
        minor = payload[3]
        patch = struct.unpack("<H", payload[12:14])[0]
        results["product_id"] = f"SW {major}.{minor}.{patch}"
        print(f"    ✓ Product ID: SW {major}.{minor}.{patch}")

    elif rid == CMD_RESPONSE and len(payload) >= 6:
        cmd = payload[2]
        status = payload[5]
        results["cmd_responses"].append((cmd, status))
        status_str = "OK" if status == 0 else f"ERR({status})"
        print(f"    CMD Response: cmd=0x{cmd:02X} status={status_str}")

    elif rid == SHTP_EXECUTABLE_RESET:
        results["got_exec_reset"] = True
        print(f"    ✓ Executable Reset received (sensor hub booted)")

    elif rid == BASE_TIMESTAMP:
        results["got_base_timestamp"] = True
        # Parse sensor data following timestamp
        parse_sensor_data(payload[5:], results)

    elif rid in REPORT_NAMES and rid not in (0xF8, 0xF3, 0xFB, 0xF9, 0xF2):
        results["got_sensor_data"] = True
        results["sensor_reports"].add(rid)
        print(f"    ✓ SENSOR DATA! Report=0x{rid:02X} ({REPORT_NAMES.get(rid, '?')})")
        parse_direct_report(rid, payload, results)

def parse_sensor_data(data, results):
    """Parse sensor reports embedded after a base timestamp"""
    offset = 0
    while offset < len(data) and offset + 1 < len(data):
        rid = data[offset]
        name = REPORT_NAMES.get(rid, f"0x{rid:02X}")
        results["got_sensor_data"] = True
        results["sensor_reports"].add(rid)
        print(f"    ✓ SENSOR DATA (after timestamp)! Report=0x{rid:02X} ({name})")
        parse_direct_report(rid, data[offset:], results)
        break  # parse one at a time for now

def parse_direct_report(rid, payload, results):
    """Print decoded sensor values"""
    try:
        if rid in (0x01, 0x14) and len(payload) >= 10:  # Accel / Raw Accel
            x, y, z = struct.unpack("<hhh", payload[4:10])
            scale = 1/100.0 if rid == 0x01 else 1.0
            print(f"       Accel: X={x*scale:.3f}  Y={y*scale:.3f}  Z={z*scale:.3f}")
            results["sample_data"][rid] = (x*scale, y*scale, z*scale)

        elif rid == 0x02 and len(payload) >= 10:  # Gyro
            x, y, z = struct.unpack("<hhh", payload[4:10])
            print(f"       Gyro:  X={x/512:.4f}  Y={y/512:.4f}  Z={z/512:.4f} rad/s")
            results["sample_data"][rid] = (x/512, y/512, z/512)

        elif rid in (0x05, 0x08) and len(payload) >= 14:  # Rotation/Game RV
            i, j, k, r = struct.unpack("<hhhh", payload[4:12])
            print(f"       Quat:  i={i/16384:.4f}  j={j/16384:.4f}  k={k/16384:.4f}  r={r/16384:.4f}")
            results["sample_data"][rid] = (i/16384, j/16384, k/16384, r/16384)
    except:
        pass


# ─── SINGLE TEST CASE ──────────────────────────────────────────────────────────
def run_test(bus_num, init_delay, inter_cmd_delay, report_interval_ms, report_ids, test_num, total):
    label = (f"Test {test_num}/{total} | "
             f"bus={bus_num} init_delay={init_delay}s "
             f"cmd_delay={inter_cmd_delay}s interval={report_interval_ms}ms "
             f"reports={[hex(r) for r in report_ids]}")
    
    print(f"\n{'='*70}")
    print(label)
    print('='*70)

    results = {
        "label": label,
        "product_id": None,
        "got_exec_reset": False,
        "got_base_timestamp": False,
        "got_sensor_data": False,
        "sensor_reports": set(),
        "cmd_responses": [],
        "sample_data": {},
        "success": False,
    }

    try:
        imu = SHTP(bus_num, I2C_ADDR)

        # ── Phase 1: Drain startup packets ──
        print("Phase 1: Draining startup...")
        pkts = imu.drain(0.5, verbose=False)
        print(f"  Drained {len(pkts)} packets")

        # Analyze for exec reset
        for ch, seq, payload in pkts:
            analyze_packet(ch, payload, results)

        # ── Phase 2: Soft Reset ──
        print("Phase 2: Soft reset...")
        step_soft_reset(imu)
        time.sleep(init_delay)

        pkts = imu.drain(0.3, verbose=True)
        for ch, seq, payload in pkts:
            analyze_packet(ch, payload, results)

        # ── Phase 3: Wake ON ──
        print("Phase 3: Wake ON...")
        step_wake_on(imu)
        time.sleep(inter_cmd_delay)

        # ── Phase 4: COMMAND_INITIALIZE ──
        print("Phase 4: COMMAND_INITIALIZE...")
        step_initialize(imu)
        time.sleep(inter_cmd_delay)

        pkts = imu.drain(0.3, verbose=True)
        for ch, seq, payload in pkts:
            analyze_packet(ch, payload, results)

        # ── Phase 5: Product ID ──
        print("Phase 5: Product ID request...")
        step_product_id(imu)
        time.sleep(inter_cmd_delay)

        pkts = imu.drain(0.3, verbose=True)
        for ch, seq, payload in pkts:
            analyze_packet(ch, payload, results)

        # ── Phase 6: Enable reports ──
        print(f"Phase 6: Enabling reports at {report_interval_ms}ms...")
        for rid in report_ids:
            name = REPORT_NAMES.get(rid, f"0x{rid:02X}")
            print(f"  Enabling {name} (0x{rid:02X})...")
            step_enable_report(imu, rid, report_interval_ms)
            time.sleep(inter_cmd_delay)

        # ── Phase 7: Listen for data ──
        print("Phase 7: Listening for sensor data (3 seconds)...")
        t0 = time.time()
        total_pkts = 0
        while time.time() - t0 < 3.0:
            pkt = imu.recv(timeout=0.05)
            if pkt:
                ch, seq, payload = pkt
                analyze_packet(ch, payload, results)
                total_pkts += 1
                if results["got_sensor_data"] and len(results["sensor_reports"]) >= len(report_ids):
                    print(f"  ✓ All requested reports received!")
                    break

        print(f"  Total packets received: {total_pkts}")

        results["success"] = results["got_sensor_data"]
        imu.close()

    except Exception as e:
        print(f"  ✗ Exception: {e}")
        results["error"] = str(e)
        try:
            imu.close()
        except:
            pass

    # ── Result Summary ──
    print("\n── Result ──")
    print(f"  Product ID:       {results['product_id'] or 'Not received'}")
    print(f"  Exec Reset:       {'✓' if results['got_exec_reset'] else '✗'}")
    print(f"  Base Timestamp:   {'✓' if results['got_base_timestamp'] else '✗'}")
    print(f"  Sensor Data:      {'✓ SUCCESS' if results['got_sensor_data'] else '✗ None'}")
    if results["sensor_reports"]:
        names = [REPORT_NAMES.get(r, f"0x{r:02X}") for r in results["sensor_reports"]]
        print(f"  Reports received: {names}")
    if results["sample_data"]:
        for rid, vals in results["sample_data"].items():
            print(f"  Sample [{REPORT_NAMES.get(rid,'?')}]: {vals}")

    return results


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     BNO085 Comprehensive I2C Diagnostic - Jetson Orin Nano       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"Testing bus(es): {I2C_BUSES}, address: 0x{I2C_ADDR:02X}")
    print()

    # Build test matrix - start targeted, expand if needed
    test_cases = []
    for bus in I2C_BUSES:
        for init_delay in INIT_DELAYS:
            for cmd_delay in INTER_CMD_DELAYS:
                for interval_ms in REPORT_INTERVALS_MS:
                    # Test both single reports and combined
                    test_cases.append((bus, init_delay, cmd_delay, interval_ms, [0x01]))         # just accel
                    test_cases.append((bus, init_delay, cmd_delay, interval_ms, [0x08]))         # just game RV
                    test_cases.append((bus, init_delay, cmd_delay, interval_ms, [0x01, 0x08]))   # both

    total = len(test_cases)
    print(f"Total test combinations: {total}")
    print("Will stop early if a working configuration is found.\n")

    successful = []
    failed = []

    for i, (bus, init_delay, cmd_delay, interval_ms, report_ids) in enumerate(test_cases, 1):
        result = run_test(bus, init_delay, cmd_delay, interval_ms, report_ids, i, total)

        if result["success"]:
            successful.append(result)
            print(f"\n🎉 SUCCESS FOUND at test {i}! Stopping early.")
            break
        else:
            failed.append(result)

        # Small pause between tests to let sensor settle
        time.sleep(0.5)

    # ── Final Report ──
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                      FINAL REPORT                                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"Tests run:    {len(successful) + len(failed)}")
    print(f"Successful:   {len(successful)}")
    print(f"Failed:       {len(failed)}")

    if successful:
        print("\n✓ WORKING CONFIGURATIONS:")
        for r in successful:
            print(f"  → {r['label']}")
            if r["sample_data"]:
                for rid, vals in r["sample_data"].items():
                    print(f"     [{REPORT_NAMES.get(rid,'?')}]: {vals}")
    else:
        print("\n✗ No working configuration found with current settings.")
        print("  Recommendations:")
        print("  1. Wire the BNO085 RST pin to a Jetson GPIO and add hardware reset")
        print("  2. Add 4.7kΩ external pull-up resistors to SDA and SCL")
        print("  3. Try UART mode (set PS1=1, PS0=0 on BNO085)")
        print("  4. Check VCC is stable 3.3V (not 5V)")

if __name__ == "__main__":
    main()
