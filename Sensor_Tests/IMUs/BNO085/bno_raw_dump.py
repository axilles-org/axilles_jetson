#!/usr/bin/env python3
"""Raw BNO085 packet dumper — shows exactly what bytes are coming from the sensor"""
import time, struct
from smbus2 import SMBus, i2c_msg

I2C_BUS  = 7
I2C_ADDR = 0x4A

CH_EXEC    = 1
CH_CONTROL = 2
SET_FEATURE_CMD = 0xFD
BASE_TIMESTAMP  = 0xFB

REPORT_NAMES = {
    0x01:"Accel", 0x02:"Gyro", 0x03:"Mag", 0x05:"RotVec",
    0x08:"GameRV", 0x0A:"Gravity", 0x14:"RawAccel", 0x15:"RawGyro",
    0xFB:"BaseTS", 0xF8:"ProdIDResp", 0xF3:"CmdResp",
    0xF9:"ProdIDReq", 0xF2:"CmdReq", 0xFC:"TSDelta",
}

class Raw:
    def __init__(self):
        self.bus = SMBus(I2C_BUS)
        self.seq = [0]*8
        time.sleep(0.1)
    def write(self, data):
        self.bus.i2c_rdwr(i2c_msg.write(I2C_ADDR, data))
    def read(self, n):
        m = i2c_msg.read(I2C_ADDR, n)
        self.bus.i2c_rdwr(m)
        return bytes(m)
    def send(self, ch, payload):
        ln = len(payload)+4
        hdr = struct.pack("<HBB", ln, ch, self.seq[ch] & 0xFF)
        self.seq[ch] = (self.seq[ch]+1) & 0xFF
        self.write(hdr+payload)
    def recv(self, timeout=0.1):
        dl = time.time()+timeout
        while time.time()<dl:
            try:
                hdr = self.read(4)
                ln, ch, seq = struct.unpack("<HBB", hdr)
                ln &= 0x7FFF
                if ln<4 or ln>1024:
                    time.sleep(0.001); continue
                payload = b"" if ln==4 else self.read(ln-4)
                return ch, seq, payload
            except:
                time.sleep(0.001)
        return None
    def enable(self, rid, ms):
        p = bytearray(17)
        p[0]=SET_FEATURE_CMD; p[1]=rid
        p[5:9]=struct.pack("<I", ms*1000)
        self.send(CH_CONTROL, bytes(p))

imu = Raw()

# Soft reset
print("Resetting...")
imu.send(CH_EXEC, bytes([0x01]))
time.sleep(0.5)

# Drain startup packets
t0=time.time()
while time.time()-t0<1.0:
    pkt=imu.recv(0.05)
    if pkt:
        ch,seq,p=pkt
        rid=p[0] if p else 0
        print(f"  [startup] CH={ch} ID=0x{rid:02X}({REPORT_NAMES.get(rid,'?')}) len={len(p)} raw={p[:16].hex()}")

print("\nEnabling reports...")
imu.send(CH_EXEC, bytes([0x02]))
time.sleep(0.05)

init=bytearray(12); init[0]=0xF2; init[2]=0x04
imu.send(CH_CONTROL, bytes(init))
time.sleep(0.05)

# Enable several reports at 20ms
for rid in [0x08, 0x01, 0x02, 0x14]:
    imu.enable(rid, 20)
    time.sleep(0.05)

print("\n--- Dumping 5 seconds of raw packets ---")
t0=time.time()
count=0
while time.time()-t0<5.0:
    pkt=imu.recv(0.05)
    if pkt:
        ch,seq,p=pkt
        rid=p[0] if p else 0
        name=REPORT_NAMES.get(rid,'?')
        print(f"CH={ch} ID=0x{rid:02X}({name:<10}) len={len(p):3d} data={p.hex()}")
        count+=1
        # If it's a sensor report, try to decode it
        if rid==BASE_TIMESTAMP and len(p)>5:
            sub_rid=p[5]
            sub_name=REPORT_NAMES.get(sub_rid,'?')
            print(f"  └── wrapped: 0x{sub_rid:02X}({sub_name}) data={p[5:].hex()}")
            if sub_rid in (0x01,0x14,0x02) and len(p)>=15:
                x,y,z=struct.unpack("<hhh",p[9:15])
                scale=1/100.0 if sub_rid==0x01 else (1/512.0 if sub_rid==0x02 else 1.0)
                print(f"       values: x={x*scale:.4f} y={y*scale:.4f} z={z*scale:.4f}")
            elif sub_rid==0x08 and len(p)>=17:
                i,j,k,r=struct.unpack("<hhhh",p[9:17])
                print(f"       quat: i={i/16384:.4f} j={j/16384:.4f} k={k/16384:.4f} r={r/16384:.4f}")
        elif rid in (0x01,0x14,0x02) and len(p)>=10:
            x,y,z=struct.unpack("<hhh",p[4:10])
            scale=1/100.0 if rid==0x01 else (1/512.0 if rid==0x02 else 1.0)
            print(f"  values: x={x*scale:.4f} y={y*scale:.4f} z={z*scale:.4f}")
        elif rid==0x08 and len(p)>=14:
            i,j,k,r=struct.unpack("<hhhh",p[4:12])
            print(f"  quat: i={i/16384:.4f} j={j/16384:.4f} k={k/16384:.4f} r={r/16384:.4f}")

print(f"\nTotal packets: {count}")
imu.bus.close()
