# LED strip (APA102 / SK9822 "DotStar")

The Jetson controls the strip through an **Arduino Micro** on USB. The Jetson
builds each frame and sends it over serial, and the Arduino clocks it out at 5 V.

The strip can't be driven straight from the Jetson's 40-pin header: those pins
are 3.3 V, and the 5 V strip doesn't read 3.3 V data.

## Wiring

| Strip pad | Connects to |
|---|---|
| DI (data in) | Arduino Micro **D14** |
| CI (clock in) | Arduino Micro **D15** |
| GND | Arduino GND **and** the strip's 5 V supply GND (common ground is required) |
| 5V | 5 V supply |

Arduino Micro → Jetson over USB. It appears as
`/dev/serial/by-id/usb-Arduino_LLC_Arduino_Micro-if00` (`/dev/ttyACM0`).

Use the input end of the strip, where the arrows point away from the wires.

## Setup

1. Upload [`arduino/dotstar_bridge/dotstar_bridge.ino`](arduino/dotstar_bridge/dotstar_bridge.ino)
   to the Micro (Arduino IDE, board **Arduino Micro**). No libraries needed.
2. `pip install -r led-strip/requirements.txt`
3. One time: `sudo usermod -aG dialout $USER`, then log out and back in. Until
   then, wrap commands in `sg dialout -c "..."`.

## Usage

```bash
python3 led-strip/led_strip.py --num 30 test              # R, G, B, W for 2 s each
python3 led-strip/led_strip.py --num 30 solid 0 255 0
python3 led-strip/led_strip.py --num 30 rainbow           # Ctrl+C to stop
python3 led-strip/led_strip.py --num 30 chase 255 80 0 --tail 6
python3 led-strip/led_strip.py --num 30 blink 255 0 0 --delay 0.25
python3 led-strip/led_strip.py --num 30 off
```

Options: `--brightness 0..1` (default 0.2), `--order` (default `BGR`), `--port`.

## From other code

```python
import sys; sys.path.insert(0, "led-strip")
from dotstar import DotStar

with DotStar(30, brightness=0.3) as strip:   # turns off on exit
    strip.fill((0, 0, 255))
    strip[0] = (255, 0, 0)
    strip.show()
```

## Serial protocol

Host → Arduino: `'D' 'S' <len hi> <len lo> <len bytes>` (raw APA102 frame).
Arduino → host: `'K'` once the bytes are clocked out. The Arduino streams bytes
out as they arrive, so there is no frame-size limit beyond 65535 bytes.
