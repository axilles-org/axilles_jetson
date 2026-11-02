"""The real jetson_deploy.py in --dry-run on a recorded CSV: no CAN bus is ever opened."""
import csv
import runpy
import sys

import pytest

from conftest import REPO
from dashboard.backend import run_logger
from fakes import fake_can

SCRIPT = REPO / "ML_model" / "scripts" / "jetson_deploy.py"
RUN_DIR = REPO / "ML_model" / "runs" / "deploy"


class _NoRunLogger:
    """Keeps test runs out of the dashboard's trial history."""

    run_id = "test"

    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def stop(self, status=""):
        pass

    def log_sample(self, **kwargs):
        pass


def run_deploy(monkeypatch, *args):
    monkeypatch.setattr(run_logger, "RunLogger", _NoRunLogger)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *args])
    runpy.run_path(str(SCRIPT), run_name="__main__")


@pytest.mark.skipif(not (RUN_DIR / "best.onnx").exists(), reason="no deploy model")
def test_dry_run_stays_inside_the_cap_and_opens_no_can(monkeypatch, tmp_path):
    replay = sorted((REPO / "dashboard" / "backend" / "sample_data").glob("*100Hz*.csv"))[0]
    out = tmp_path / "run.csv"
    fake_can.opened.clear()
    run_deploy(monkeypatch, "--run", str(RUN_DIR), "--mass", "72", "--replay", str(replay),
               "--dry-run", "--arm", "--assist-scale", "0.1", "--torque-cap", "2.0",
               "--arm-ramp", "1", "--duration", "8", "--no-teleplot", "--print-every", "0",
               "--out", str(out))
    rows = list(csv.DictReader(out.open()))

    assert fake_can.opened == []
    assert len(rows) > 300
    sent = [float(r["sent_torque_nm"]) for r in rows]
    assert max(abs(s) for s in sent) <= 2.0
    assert max(sent) > 0.1 and min(sent) >= 0.0     # plantarflexion is positive torque
    assert all(r["model_fault"] == "0" for r in rows)


@pytest.mark.parametrize("cap", ["-2", "0", "5.5", "nan"])
def test_cli_refuses_a_bad_torque_cap(monkeypatch, cap):
    with pytest.raises(SystemExit) as exc:
        run_deploy(monkeypatch, "--run", str(RUN_DIR), "--mass", "72", "--dry-run",
                   f"--torque-cap={cap}")
    assert exc.value.code == 2
