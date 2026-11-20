# LED strip (APA102 / SK9822 "DotStar")

Drives a clocked LED strip from the Jetson Orin Nano's 40-pin header using
bit-banged GPIO on the SPI1 pins (default) or hardware SPI1 (`/dev/spidev0.0`).
The code never touches I2C or CAN.

## Wiring

| Jetson 40-pin header | Strip pad |
|---|---|
| Pin 19 (SPI1_MOSI) | DI / D (data in) |
| Pin 23 (SPI1_SCK) | CI / C (clock in) |
| Pin 6 (GND), or any GND pin | GND |
| External 5 V supply (+) | 5V / VCC |

- Connect to the **input** end of the strip (the arrows on the strip point away from it).
- **Power:** each LED can draw up to 60 mA at full white. Power the strip from its
  own 5 V supply and connect that supply's ground to the Jetson's GND. For a
  short test (about 10 LEDs or fewer, at the default 0.2 brightness) you can use
  header pin 2 or 4 (5 V) instead.
- **Logic level:** the header outputs 3.3 V, and a strip powered at 5 V
  officially wants 3.5 V or more. It usually works anyway with short wires. If
  colors glitch or the far end flickers, add a 74AHCT125 level shifter on DI
  and CI. With `--backend spi`, you can also lower the clock with `--speed 1000000`.
- Pins 21 (MISO) and 24 (CS0) are not used. Pins 19 and 23 aren't shared with
  the sensors' I2C buses or with CAN.

## Setup

`pip install -r led-strip/requirements.txt`. Your user must be in the `gpio`
group, which owns `/dev/gpiochip*` and `/dev/spidev*`.

### Two ways to drive the pins

**`--backend gpio` (default):** the program drives pins 19 and 23 as plain GPIO
lines (`PZ.05` data, `PZ.03` clock on `/dev/gpiochip0`) and generates the clock
itself. It needs no sudo, no reboot and no boot or device-tree change, and the
lines go back to inputs when the program exits. It is slower than hardware SPI:
roughly tens of ms per frame for 30 LEDs, which is fine for status lights and
simple animations.

**`--backend spi`:** hardware SPI1 through `/dev/spidev0.0`, which is fast, but
pins 19 and 23 only carry SPI after SPI1 is enabled in the pinmux. On this
Jetson, **don't use the `jetson-io.py` menu for that**: it adds a boot entry on
the stock device tree and makes it the default, which drops `/boot/custom.dtb`
(400 kHz I2C). If you ever need hardware SPI, generate only the overlay
(`sudo /opt/nvidia/jetson-io/config-by-function.py -o dtbo 1="spi1"`) and add an
`OVERLAYS` line under the existing `custom` boot entry.

## Usage

```bash
python led-strip/led_strip.py --num 30 test             # R, G, B, W for 2 s each
python led-strip/led_strip.py --num 30 solid 0 255 0
python led-strip/led_strip.py --num 30 rainbow
python led-strip/led_strip.py --num 30 chase 255 80 0 --tail 6
python led-strip/led_strip.py --num 30 blink 255 0 0 --delay 0.25
python led-strip/led_strip.py --num 30 off
```

Options: `--brightness 0..1` (default 0.2), `--order` (default `BGR`),
`--speed` in Hz (default 4 MHz), `--bus/--device` (default 0/0).

If `test` shows the wrong colors, try `--order BRG`, `GBR` or `RGB`.

## From other code

```python
import sys; sys.path.insert(0, "led-strip")
from dotstar import DotStar

with DotStar(30, brightness=0.3) as strip:   # turns off on exit
    strip.fill((0, 0, 255))
    strip[0] = (255, 0, 0)
    strip.show()
```
