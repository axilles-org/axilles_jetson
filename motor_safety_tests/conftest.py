"""Keeps these tests away from the motor.

They must never reach real hardware, so this file:
  1. refuses to start if any CAN interface is visible (use run.sh),
  2. replaces the ``can`` and ``smbus2`` modules with fakes before anything imports them,
  3. makes opening a CAN socket raise, in case some other code tries.
"""
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_visible = [name for _, name in socket.if_nameindex() if "can" in name]
if _visible:
    raise SystemExit(f"Refusing to run: CAN interface {_visible} is visible. "
                     "Use motor_safety_tests/run.sh, which hides it.")


class _NoCanSocket(socket.socket):
    def __init__(self, family=socket.AF_INET, *args, **kwargs):
        if family == getattr(socket, "AF_CAN", 29):
            raise RuntimeError("a test tried to open a CAN socket")
        super().__init__(family, *args, **kwargs)


socket.socket = _NoCanSocket

import fakes  # noqa: E402

sys.modules["can"] = fakes.fake_can
sys.modules["smbus2"] = fakes.fake_smbus2

for path in (REPO, REPO / "ML_model" / "src", REPO / "TBE_controller"):
    sys.path.insert(0, str(path))
