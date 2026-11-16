"""S5: every file packs torque the same way, and the packing matches the V3 manual."""
import re

import pytest

import data_obtainer
from conftest import REPO
from exo.deploy import motor

FILES = ["ML_model/src/exo/deploy/motor.py", "TBE_controller/data_obtainer.py",
         "Cubemars_AK80-9/ak_80_diag.py", "Cubemars_AK80-9/ak_80_test.py",
         "Cubemars_AK80-9/ak_80_torque_input.py", "Cubemars_AK80-9/ak_80_torque_sine.py",
         "Cubemars_AK80-9/ak_80_torque_test.py", "Fall_tests/fall_logger.py"]
RANGE = re.compile(r"MIT_T_MAX\s*=\s*(?:-[\d.]+,\s*)?([\d.]+)|MIT_T\s*=\s*\(\s*-([\d.]+)")


@pytest.mark.parametrize("name", FILES)
def test_torque_range_is_22_everywhere(name):
    match = RANGE.search((REPO / name).read_text())
    assert match, f"no torque range found in {name}"
    assert float(match.group(1) or match.group(2)) == 22.0


@pytest.mark.parametrize("name", FILES)
def test_no_file_uses_the_old_scaling(name):
    assert not re.search(r"\(1 << bits\) /", (REPO / name).read_text())


@pytest.mark.parametrize("to_uint", [motor._float_to_uint, data_obtainer._float_to_uint])
def test_matches_the_manuals_example_bytes(to_uint):
    # V3 manual p.45-46, AK10-9 ranges: "MIT Torque Ring" 7F FF 7F F8 4B / 97, "01 06 66"
    assert to_uint(2.0, -54.0, 54.0, 12) == 0x84B
    assert to_uint(4.0, -54.0, 54.0, 12) == 0x897
    assert to_uint(0.0, -28.0, 28.0, 12) == 0x7FF
    assert to_uint(0.0, -12.56, 12.56, 16) == 0x7FFF
    assert to_uint(2.0, 0.0, 500.0, 12) == 0x010
    assert to_uint(2.0, 0.0, 5.0, 12) == 0x666


@pytest.mark.parametrize("to_uint", [motor._float_to_uint, data_obtainer._float_to_uint])
def test_full_scale_fits_in_the_field(to_uint):
    assert to_uint(22.0, -22.0, 22.0, 12) == 0xFFF


def test_zero_torque_frame_is_exact():
    assert motor._pack_mit(0.0, 0.0, 0.0, 0.0, 0.0) == [0, 0, 0, 0x7F, 0xFF, 0x7F, 0xF7, 0xFF]


def test_pack_still_clamps_to_5_nm():
    assert motor._pack_mit(0, 0, 21.0, 0, 0) == motor._pack_mit(0, 0, 5.0, 0, 0)
