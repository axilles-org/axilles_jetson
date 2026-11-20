"""Minimal APA102 / SK9822 ("DotStar") driver: bit-banged GPIO (default) or spidev.

The strip only needs data (DI) and clock (CI). The SPI chip-select and MISO pins
are not used. Frame format, sent MSB first in SPI mode 0:

    start frame   4 x 0x00
    per LED       0b111xxxxx (5-bit global brightness), then 3 color bytes
    end frame     zero bytes: 4 for SK9822 reset plus ceil(n/16) so the data
                  reaches the last LED of an APA102 chain

The end frame uses 0x00, not 0xFF: if the strip is longer than ``num_leds``,
the extra LEDs see the bytes as another start frame and stay as they were,
instead of being driven to full white.

Colors are given as (r, g, b) tuples in 0-255. ``order`` is the byte order the
strip expects on the wire after the brightness byte: "BGR" for most APA102
strips, though some batches use "BRG" or "GBR". Run ``led_strip.py test`` to
find the right one.
"""
from __future__ import annotations

from typing import Iterable, Sequence

Color = tuple[int, int, int]

_VALID_ORDERS = {"RGB", "RBG", "GRB", "GBR", "BRG", "BGR"}


class DotStar:
    def __init__(
        self,
        num_leds: int,
        bus: int = 0,
        device: int = 0,
        speed_hz: int = 4_000_000,
        order: str = "BGR",
        brightness: float = 0.2,
        auto_show: bool = False,
        backend: str = "gpio",
        gpio_chip: str = "/dev/gpiochip0",
        clock_line: str = "PZ.03",
        data_line: str = "PZ.05",
        half_period_s: float = 0.0,
    ) -> None:
        if num_leds < 1:
            raise ValueError("num_leds must be at least 1")
        order = order.upper()
        if order not in _VALID_ORDERS:
            raise ValueError(f"order must be one of {sorted(_VALID_ORDERS)}")

        if backend not in ("gpio", "spi"):
            raise ValueError("backend must be 'gpio' or 'spi'")

        self.num_leds = num_leds
        self.auto_show = auto_show
        self._index = tuple("RGB".index(c) for c in order)
        self._pixels: list[Color] = [(0, 0, 0)] * num_leds
        self._brightness = min(max(float(brightness), 0.0), 1.0)

        if backend == "spi":
            self._out = _SpiOut(bus, device, speed_hz)
        else:
            self._out = _GpioOut(gpio_chip, clock_line, data_line, half_period_s)

        self._end_frame = bytes(4 + (num_leds + 15) // 16)

    # brightness -----------------------------------------------------------
    @property
    def brightness(self) -> float:
        return self._brightness

    @brightness.setter
    def brightness(self, value: float) -> None:
        self._brightness = min(max(float(value), 0.0), 1.0)
        if self.auto_show:
            self.show()

    # pixel access ---------------------------------------------------------
    def __len__(self) -> int:
        return self.num_leds

    def __getitem__(self, i: int) -> Color:
        return self._pixels[i]

    def __setitem__(self, i: int, color: Sequence[int]) -> None:
        self._pixels[i] = _clamp(color)
        if self.auto_show:
            self.show()

    def fill(self, color: Sequence[int]) -> None:
        self._pixels = [_clamp(color)] * self.num_leds
        if self.auto_show:
            self.show()

    def set_all(self, colors: Iterable[Sequence[int]]) -> None:
        colors = [_clamp(c) for c in colors]
        if len(colors) != self.num_leds:
            raise ValueError(f"expected {self.num_leds} colors, got {len(colors)}")
        self._pixels = colors
        if self.auto_show:
            self.show()

    # output ---------------------------------------------------------------
    def show(self) -> None:
        """Send the current buffer to the strip."""
        # Brightness goes into the 5-bit global field rather than scaling the
        # 8-bit color values, so dim colors keep their full 8-bit resolution.
        level = round(self._brightness * 31)
        header = 0xE0 | level
        a, b, c = self._index
        buf = bytearray(4)
        for px in self._pixels:
            buf += bytes((header, px[a], px[b], px[c]))
        buf += self._end_frame
        self._out.write(buf)

    def off(self) -> None:
        self._pixels = [(0, 0, 0)] * self.num_leds
        self.show()

    def close(self, turn_off: bool = True) -> None:
        if self._out is None:
            return
        try:
            if turn_off:
                self.off()
        finally:
            self._out.close()
            self._out = None

    def __enter__(self) -> "DotStar":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _SpiOut:
    """Hardware SPI through spidev. Needs SPI1 enabled in the pinmux (jetson-io)."""

    def __init__(self, bus: int, device: int, speed_hz: int) -> None:
        import spidev

        self._dev = spidev.SpiDev()
        self._dev.open(bus, device)
        self._dev.mode = 0
        self._dev.max_speed_hz = speed_hz

    def write(self, buf: bytes) -> None:
        # writebytes2 splits buffers larger than the spidev bufsiz (4096 by default).
        self._dev.writebytes2(buf)

    def close(self) -> None:
        self._dev.close()


class _GpioOut:
    """Bit-banged clock + data on two GPIO lines through libgpiod.

    Needs no pinmux or boot change: it drives the same header pins (19 and 23
    by default) as plain GPIO outputs. APA102 latches data on the clock's rising
    edge and has no minimum clock rate, so jitter from Python is harmless; it is
    only slower, roughly tens of ms per frame for a 30-LED strip.

    On release the lines are set back to inputs, as they were before.
    """

    def __init__(self, chip: str, clock_line: str, data_line: str,
                 half_period_s: float = 0.0) -> None:
        import gpiod

        self._half = half_period_s
        from gpiod.line import Direction, Value

        self._lo, self._hi = Value.INACTIVE, Value.ACTIVE
        with gpiod.Chip(chip) as c:
            self._clk = c.line_offset_from_id(clock_line)
            self._dat = c.line_offset_from_id(data_line)
        self._req = gpiod.request_lines(
            chip,
            consumer="led-strip",
            config={(self._clk, self._dat): gpiod.LineSettings(
                direction=Direction.OUTPUT, output_value=Value.INACTIVE)},
        )
        self._input = gpiod.LineSettings(direction=Direction.INPUT)

    def write(self, buf: bytes) -> None:
        req, clk, dat, lo, hi = self._req, self._clk, self._dat, self._lo, self._hi
        # Precomputed "clock low + data bit" states for one call per half-period.
        low_with = ({clk: lo, dat: lo}, {clk: lo, dat: hi})
        set_values, set_value = req.set_values, req.set_value
        if self._half > 0:
            # Slow path: clock low, then data, then clock high, each held for
            # half_period_s, so the data is settled long before the rising edge.
            import time
            half = self._half
            for byte in buf:
                for shift in range(7, -1, -1):
                    set_value(clk, lo)
                    time.sleep(half / 2)
                    set_value(dat, hi if (byte >> shift) & 1 else lo)
                    time.sleep(half / 2)
                    set_value(clk, hi)
                    time.sleep(half)
            set_values(low_with[0])
            return
        for byte in buf:
            for shift in range(7, -1, -1):
                set_values(low_with[(byte >> shift) & 1])
                set_value(clk, hi)
        set_values(low_with[0])

    def close(self) -> None:
        try:
            self._req.reconfigure_lines({(self._clk, self._dat): self._input})
        finally:
            self._req.release()


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
