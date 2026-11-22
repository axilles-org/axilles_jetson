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

Options: `--brightness 0..1` (default 0.2), `--order` (default `GBR`), `--port`.

## From other code

```python
import sys; sys.path.insert(0, "led-strip")
from dotstar import DotStar

with DotStar(30, brightness=0.3) as strip:   # turns off on exit
    strip.fill((0, 0, 255))
    strip[0] = (255, 0, 0)
    strip.show()
```

## Exo status lights

During a **live** ML run (`jetson_deploy.py --arm` without `--dry-run`, i.e. the
dashboard's "live" mode) the controller drives the strip through
`ML_model/src/exo/deploy/leds.py`. Mock and dry-run leave it alone; `--no-leds`
turns it off for a live run. The Arduino plays the timed patterns itself, so a
blink finishes even after the controller has exited.

| When | Strip (half brightness) |
|---|---|
| Arduino powers up | green blinking at 2 Hz (0.25 s on / 0.25 s off), twice |
| Ankle-zero calibration (`--ankle-zero auto`) | yellow blinking at 3 Hz (~0.17 s on / ~0.17 s off) until done |
| Calibrated, assistance off | green, steady (also after assistance is switched off again) |
| Assistance on, walking stance | 5-LED window: wired end at heel strike, last five LEDs by toe-off. Blue at 0 N·m to red at `--torque-cap`, from \|sent torque\| |
| Assistance on, swing | off |
| Assistance on, standing | off: no window until the next real step |
| Normal end (Stop, Ctrl-C, duration) | red, steady for 5 s, then off |
| Motor fault (E-stop cutting motor power shows as "no motor status") or crash | red blinking at 5 Hz (0.1 s on / 0.1 s off) for 10 s |

While a stop or E-stop red is playing, the strip ignores every other request
except another E-stop, so a run started within 5 s of a stop shows its yellow
only after the red ends.

**Walking vs standing.** The window follows the FSR stance flag, with a foot
gyro check so it only shows while walking:

- A heel strike starts a window only after a real swing: the stance flag off
  for at least 0.2 s with the foot gyro magnitude reaching 3 rad/s. Weight
  shifts while standing don't qualify.
- The stance flag drops out briefly at push-off (20-157 ms, gyro 1-2.4 rad/s).
  Gaps shorter than 0.2 s that stay under 3 rad/s don't end the window, so it
  sweeps once per stride.
- A stance longer than 1.2 s counts as standing, and the window turns off.
- If the foot gyro reads the same value for 0.2 s (a frozen IMU), swings are
  judged on duration alone.

These thresholds come from the 2026-11-21 powered run: standing foot gyro
peaked at 0.32 rad/s or less, real swings at 4.3-7.3 rad/s, and walking stances
lasted 0.75 s (median). The window speed comes from the median of the last 3
stance times (0.75 s until one is measured), so a long stance holds on the last
LEDs and a short one ends before reaching them. The stance flag the controller
uses for torque is not changed by any of this.

Settings, all constants at the top of `leds.py`:

| Constant | Now | Effect |
|---|---|---|
| `NUM_LEDS` | 12 | LEDs on the strip |
| `COLOR_ORDER` | `"GBR"` | color byte order this strip expects |
| `BRIGHTNESS` | 0.5 | 0-1, every pattern |
| `WINDOW_LEDS` | 5 | width of the moving window |
| `SWING_GYRO_RAD_S` | 3.0 | foot gyro a swing must reach to count as a step |
| `MIN_SWING_S` | 0.2 | shorter stance-flag gaps are push-off dropouts |
| `STANDING_STANCE_S` | 1.2 | longer stances count as standing |
| `FIRST_STANCE_S` | 0.75 | stance time assumed before one is measured |
| `STANCE_HISTORY` | 3 | stances in the running median |

`NUM_LEDS`, `COLOR_ORDER`, `BRIGHTNESS` and `WINDOW_LEDS` are sent to the
Arduino when a run connects, and it saves them in EEPROM for the power-on
blink. Blink rates and durations are constants at the top of the sketch
(`BOOT_BLINK_HZ`, `BOOT_MS`, `CALIB_BLINK_HZ`, `ESTOP_BLINK_HZ`, `ESTOP_MS`,
`STOP_MS`); changing them needs a re-upload:

```bash
arduino-cli compile --upload --fqbn arduino:avr:micro -p /dev/ttyACM0 led-strip/arduino/dotstar_bridge
```

**Strip power.** As shipped, the sketch assumes the strip powers up with the
Arduino and blinks green at boot. If the strip has its own supply, wire strip
5V → 10 kΩ → a spare pin, plus 100 kΩ from that pin to GND, and set
`STRIP_SENSE_PIN` in the sketch: it then waits for strip power before the green
blink and repeats it whenever strip power returns.

**Current.** At half brightness the worst pattern, full-strip yellow, peaks
near 240 mA (about 120 mA average while blinking). If the strip runs off the
Micro's 5V pin, it shares the Micro's 500 mA USB fuse: don't run
`led_strip.py test` above `--brightness 0.5` (its white step is about 720 mA at
full brightness).

## Serial protocol

All packets start with `D`:

| Host → Arduino | Meaning | Reply |
|---|---|---|
| `D S <len hi> <len lo> <len bytes>` | raw APA102 frame (`led_strip.py`, `dotstar.py`); patterns pause until the next `D P` | `K` |
| `D P <mode> <pos> <r> <g> <b>` | pattern: 0 off, 1 calibrating, 2 ready (steady green), 3 assist, 4 stop, 5 E-stop; `pos` is the window's first LED (255 none) | none |
| `D C <num> <o0> <o1> <o2> <brightness 0-31> <width>` | strip config, saved to EEPROM; `o0..o2` give the color of each wire byte (0 R, 1 G, 2 B) | none |

The Arduino streams raw-frame bytes out as they arrive, so there is no
frame-size limit beyond 65535 bytes. Calibrating, ready and assist go dark if no
`D P` packet arrives for 0.5 s; the controller sends one every 20 ms.
