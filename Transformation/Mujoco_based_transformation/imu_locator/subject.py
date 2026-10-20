"""Subject anthropometrics -> segment dimensions for the MuJoCo walker.

Anything left as None is filled from Winter's anthropometric ratios (fraction of
body height) or from the OpenSim gait2392 generic model scaled by height.
Measured values always beat the defaults: the distal segment accelerations (and
therefore the estimated IMU positions) depend directly on these lengths.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

GAIT2392_HEIGHT = 1.80  # nominal height of the generic OpenSim model [m]


@dataclass
class Subject:
    height: float = 1.75            # [m]
    mass: float = 70.0              # [kg] (only used for inertia; kinematics don't need it)
    thigh_length: float | None = None   # hip joint centre -> knee joint centre [m]
    shank_length: float | None = None   # knee joint centre -> ankle joint centre [m]
    foot_length: float | None = None    # heel -> toe tip [m]
    ankle_height: float | None = None   # ankle joint centre above the sole [m]
    hip_width: float | None = None      # distance between the two hip joint centres [m]
    torso_length: float | None = None   # lumbar joint -> C7 [m]
    thigh_radius: float | None = None   # mid-thigh radius (circumference / 2pi) [m]
    shank_radius: float | None = None   # mid-shank radius [m]
    # Hip joint centre relative to the pelvis origin, OpenSim-style
    # (forward, down); None -> gait2392 values scaled by height.
    hip_offset_forward: float | None = None
    hip_offset_down: float | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "Subject":
        d = json.loads(Path(path).read_text())
        extra = d.pop("extra", {}) or {}
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__}
        return cls(**known, extra={**extra, **d})

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    def resolved(self) -> dict:
        """All dimensions with defaults filled in."""
        H = self.height
        s = H / GAIT2392_HEIGHT

        def pick(v, default):
            return float(default if v is None else v)

        return dict(
            height=H,
            mass=self.mass,
            thigh_length=pick(self.thigh_length, 0.245 * H),
            shank_length=pick(self.shank_length, 0.246 * H),
            foot_length=pick(self.foot_length, 0.152 * H),
            ankle_height=pick(self.ankle_height, 0.039 * H),
            hip_width=pick(self.hip_width, 2 * 0.0835 * s),
            torso_length=pick(self.torso_length, 0.288 * H),
            thigh_radius=pick(self.thigh_radius, 0.06 * s),
            shank_radius=pick(self.shank_radius, 0.045 * s),
            hip_offset_forward=pick(self.hip_offset_forward, -0.0707 * s),
            hip_offset_down=pick(self.hip_offset_down, 0.0661 * s),
            lumbar_offset_forward=-0.1007 * s,
            lumbar_offset_up=0.0815 * s,
        )
