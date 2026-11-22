"""APA102 / SK9822 ("DotStar") driver for the Jetson, through an Arduino Micro.

The Jetson's header pins are 3.3 V, and the 5 V strip ignores 3.3 V data, so the
strip hangs off an Arduino Micro (5 V pins) running arduino/dotstar_bridge. This
module builds each APA102 frame and sends it over USB serial; the Arduino clocks
the bytes out on D14 (data) and D15 (clock).

Frame format (MSB first):

    start frame   4 x 0x00
    per LED       0b111xxxxx (5-bit global brightness), then 3 color bytes
    end frame     zero bytes: 4 for SK9822 reset plus ceil(n/16) so the data
                  reaches the last LED of an APA102 chain

The end frame uses 0x00, not 0xFF: if the strip is longer than ``num_leds``,
the extra LEDs see another start frame and keep their state instead of going
full white.

Serial packet: b"DS" + 2-byte big-endian length + frame bytes. The Arduino
answers b"K" once the bytes are on the wire.

Colors are (r, g, b) tuples in 0-255. ``order`` is the byte order on the wire
after the brightness byte: "GBR" for the exo strip ("BGR" for most APA102 strips).
"""
from __future__ import annotations

import time
from typing import Iterable, Sequence

Color = tuple[int, int, int]

DEFAULT_PORT = "/dev/serial/by-id/usb-Arduino_LLC_Arduino_Micro-if00"
_VALID_ORDERS = {"RGB", "RBG", "GRB", "GBR", "BRG", "BGR"}


class DotStar:
    def __init__(
        self,
        num_leds: int,
        port: str = DEFAULT_PORT,
        order: str = "GBR",
        brightness: float = 0.2,
        timeout_s: float = 2.0,
    ) -> None:
        if num_leds < 1:
            raise ValueError("num_leds must be at least 1")
        order = order.upper()
        if order not in _VALID_ORDERS:
            raise ValueError(f"order must be one of {sorted(_VALID_ORDERS)}")

        import serial

        self.num_leds = num_leds
        self._index = tuple("RGB".index(c) for c in order)
        self._pixels: list[Color] = [(0, 0, 0)] * num_leds
        self.brightness = brightness
        self._end_frame = bytes(4 + (num_leds + 15) // 16)

        # 115200 is nominal (native USB ignores it). Never 1200: that baud rate
        # resets an Arduino Micro into its bootloader.
        self._ser = serial.Serial(port, 115200, timeout=timeout_s, write_timeout=timeout_s)
        time.sleep(0.2)                  # let the sketch see DTR and print its greeting
        self._ser.reset_input_buffer()

    # brightness -----------------------------------------------------------
    @property
    def brightness(self) -> float:
        return self._brightness

    @brightness.setter
    def brightness(self, value: float) -> None:
        self._brightness = min(max(float(value), 0.0), 1.0)

    # pixel access ---------------------------------------------------------
    def __len__(self) -> int:
        return self.num_leds

    def __getitem__(self, i: int) -> Color:
        return self._pixels[i]

    def __setitem__(self, i: int, color: Sequence[int]) -> None:
        self._pixels[i] = _clamp(color)

    def fill(self, color: Sequence[int]) -> None:
        self._pixels = [_clamp(color)] * self.num_leds

    def set_all(self, colors: Iterable[Sequence[int]]) -> None:
        colors = [_clamp(c) for c in colors]
        if len(colors) != self.num_leds:
            raise ValueError(f"expected {self.num_leds} colors, got {len(colors)}")
        self._pixels = colors

    # output ---------------------------------------------------------------
    def show(self) -> None:
        """Send the current buffer to the strip and wait for the Arduino's ack."""
        # Brightness uses the 5-bit global field, so dim colors keep full 8-bit
        # color resolution.
        header = 0xE0 | round(self._brightness * 31)
        a, b, c = self._index
        frame = bytearray(4)
        for px in self._pixels:
            frame += bytes((header, px[a], px[b], px[c]))
        frame += self._end_frame
        if len(frame) > 0xFFFF:
            raise ValueError("frame too long for one packet")

        self._ser.write(b"DS" + len(frame).to_bytes(2, "big") + frame)
        while True:                      # skip any greeting text until the ack
            r = self._ser.read(1)
            if not r:
                raise TimeoutError("no ack from the Arduino: is dotstar_bridge.ino uploaded?")
            if r == b"K":
                return

    def off(self) -> None:
        self._pixels = [(0, 0, 0)] * self.num_leds
        self.show()

    def close(self, turn_off: bool = True) -> None:
        if self._ser is None:
            return
        try:
            if turn_off:
                self.off()
        finally:
            self._ser.close()
            self._ser = None

    def __enter__(self) -> "DotStar":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _clamp(color: Sequence[int]) -> Color:
    r, g, b = color
    return (min(max(int(r), 0), 255), min(max(int(g), 0), 255), min(max(int(b), 0), 255))


def wheel(pos: int) -> Color:
    """Map 0-255 to a color around the hue wheel (R -> G -> B -> R)."""
    pos %= 256
    if pos < 85:
        return (255 - pos * 3, pos * 3, 0)
    if pos < 170:
        pos -= 85
        return (0, 255 - pos * 3, pos * 3)
    pos -= 170
    return (pos * 3, 0, 255 - pos * 3)
