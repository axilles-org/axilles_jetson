from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class WaveformMetrics:
    rmse: float
    mae: float
    nrmse: float
    r2: float
    pearson_r: float

    xcorr_lag: int
    dtw_distance: float | None

    peak_value_error: float
    peak_value_pct_error: float
    peak_timing_error_pct: float
    rom_error: float

    cmc: float | None
    bland_altman_bias: float | None
    bland_altman_loa: tuple[float, float] | None
    icc: float | None

    def summary(self) -> str:
        lines = []
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if isinstance(value, float):
                value = round(value, 4)
            elif isinstance(value, tuple):
                value = tuple(round(v, 4) for v in value)
            lines.append(f"{f.name}: {value}")
        return "\n".join(lines)
