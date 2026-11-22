"""Delay timing and final gating, without opening hardware."""
import dataclasses
import sys
from pathlib import Path

import pytest

from exo.deploy.delay import PredictionDelay
from exo.deploy.assistance import AssistanceController


def test_delay_uses_elapsed_time_at_irregular_rates():
    d = PredictionDelay(50)
    for t, value in [(0, -1), (.009, -2), (.031, +3), (.049, +4)]:
        assert d.update(value, t) == 0
        assert not d.ready
    assert d.update(+5, .050) == -1
    assert d.age_ms == pytest.approx(50)
    assert d.update(+6, .073) == -2
    assert d.age_ms == pytest.approx(64)
    assert d.update(+7, .081) == 3


def test_zero_delay_passthrough_and_bounded_history():
    d = PredictionDelay(0)
    for i in range(1000):
        assert d.update(i, i * .003) == i
        assert d.ready and d.age_ms == 0
    assert len(d._samples) == 1
    d = PredictionDelay(50)
    for i in range(10000):
        d.update(i, i * .01)
    assert len(d._samples) <= 7


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), 1001])
def test_invalid_delays_rejected(value):
    with pytest.raises(ValueError, match="assistance_delay_ms"):
        PredictionDelay(value)


def make_controller(cfg):
    return AssistanceController(dataclasses.replace(
        cfg.deploy, assistance_delay_ms=50, assistance_mode="bidirectional",
        assistance_scale=.02, ramp_in_s=.001, rate_limit_nm_per_s=10000,
        torque_limit_nm=2, dorsiflexion_limit_nm=.5))


def step(c, t, value=-1, **gates):
    args = dict(stance=True, ready=True, enabled=True, now=t)
    args.update(gates)
    return c.update(value, 70, 0, 0, .01, **args).torque_nm


@pytest.mark.parametrize("gate", ["stance", "enabled", "ready"])
def test_stop_is_immediate_and_restart_cannot_replay_old_torque(cfg, gate):
    c = make_controller(cfg)
    assert step(c, 0) == 0
    assert step(c, .05) == pytest.approx(1.4)
    assert step(c, .06, **{gate: False}) == 0
    assert not c.delay.ready
    assert step(c, .07, +1) == 0
    assert step(c, .11, +1) == 0
    assert step(c, .12, +1) == pytest.approx(-.5)


def test_fault_bypasses_delay_and_latches_until_reset(cfg):
    c = make_controller(cfg)
    step(c, 0)
    assert step(c, .05) > 0
    assert step(c, .06, float("nan")) == 0
    assert c.faulted and not c.delay.ready
    assert step(c, .12) == 0
    c.reset()
    assert step(c, .13) == 0
    assert step(c, .18) > 0


def test_scaling_and_direction_limits_apply_to_delayed_prediction(cfg):
    c = make_controller(cfg)
    step(c, 0, -10)
    assert step(c, .05, +10) == 2
    c.scale = .001
    assert step(c, .1, 0) == pytest.approx(-.5)  # dorsi cap, current scale
    c.scale = .0001
    assert step(c, .11, 0) == pytest.approx(-.07)


def test_runtime_delay_change_waits_for_end_of_stance(cfg):
    c = make_controller(cfg)
    assert step(c, 0) == 0
    assert step(c, .05) == pytest.approx(1.4)
    c.request_delay(20)
    assert c.target_delay_ms == 20
    # the stance in progress keeps the 50 ms delay and its filled buffer
    assert step(c, .06) == pytest.approx(1.4)
    assert c.delay.delay_ms == 50 and c.delay.ready
    # toe-off applies it; the next stance fills 20 ms, nothing from the old buffer
    assert step(c, .07, stance=False) == 0
    assert c.delay.delay_ms == 20 and c.pending_delay_ms is None
    assert step(c, .08) == 0
    assert step(c, .09) == 0
    assert step(c, .10) == pytest.approx(1.4)
    assert c.delay.age_ms == pytest.approx(20)


def test_runtime_delay_not_applied_in_stance_even_with_assistance_off(cfg):
    c = make_controller(cfg)
    c.request_delay(25)
    step(c, 0, enabled=False)
    step(c, .01, ready=False)
    assert c.delay.delay_ms == 50 and c.pending_delay_ms == 25
    step(c, .02, stance=False)
    assert c.delay.delay_ms == 25


def test_runtime_delay_request_matching_current_clears_pending(cfg):
    c = make_controller(cfg)
    c.request_delay(30)
    c.request_delay(50)
    assert c.pending_delay_ms is None and c.target_delay_ms == 50
    with pytest.raises(ValueError, match="assistance_delay_ms"):
        c.request_delay(float("nan"))


def test_switch_keeps_launch_delay_until_set_and_clamps_to_range():
    import json
    import socket
    import time
    from exo.deploy.assist_switch import AssistSwitch

    sw = AssistSwitch(.02, .3, port=0, delay_range_ms=(10, 40))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = sw._sock.getsockname()

    def deliver(msg):
        tx.sendto(json.dumps(msg).encode(), addr)
        deadline = time.monotonic() + 1.0
        before = sw._last_msg
        while sw._last_msg == before and time.monotonic() < deadline:
            sw.poll()

    try:
        deliver({"assist": False, "scale": .02, "delay_ms": None})
        assert sw.delay_ms is None
        deliver({"assist": False, "scale": .02, "delay_ms": 25})
        assert sw.delay_ms == 25
        deliver({"assist": False, "scale": .02, "delay_ms": 90})
        assert sw.delay_ms == 40
        deliver({"assist": False, "scale": .02, "delay_ms": 0})
        assert sw.delay_ms == 10
        deliver({"assist": False, "scale": .02})
        assert sw.delay_ms == 10
    finally:
        tx.close()
        sw.close()


@pytest.mark.parametrize("value", [9.9, 40.1, "abc", None, float("nan")])
def test_dashboard_rejects_live_delay_outside_10_to_40(value):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dashboard.backend.process_manager import checked_live_delay
    with pytest.raises(ValueError, match="assistance delay"):
        checked_live_delay(value)
    assert checked_live_delay(10) == 10 and checked_live_delay("35") == 35


def test_dashboard_live_run_gets_runtime_delay_range():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dashboard.backend.process_manager import build_argv
    argv, _ = build_argv("ml-live", "runs/deploy", 72, None, 10, "onnx", .02,
                         height=1.75, gender="M", torque_cap=2)
    assert argv[argv.index("--min-delay-ms") + 1] == "10.0"
    assert argv[argv.index("--max-delay-ms") + 1] == "40.0"


@pytest.mark.parametrize("mode", ["ml-live", "ml-mock", "ml-dry-run"])
def test_dashboard_passes_delay_to_each_ml_runner(mode):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dashboard.backend.process_manager import build_argv
    argv, _ = build_argv(mode, "runs/deploy", 72, None, 10, "onnx", .02,
                         height=1.75, gender="M", torque_cap=2,
                         assistance_delay_ms=75)
    assert argv[argv.index("--assistance-delay-ms") + 1] == "75.0"


@pytest.mark.parametrize("value", [-1, "abc", float("nan"), float("inf"), 1001])
def test_dashboard_rejects_invalid_delay(value):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dashboard.backend.process_manager import build_argv
    with pytest.raises(ValueError, match="assistance delay"):
        build_argv("ml-mock", "runs/deploy", 72, None, 10, "onnx", .02,
                   assistance_delay_ms=value)


def _live_manager(observed=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dashboard.backend.process_manager import JobState, ProcessManager
    m = ProcessManager()
    m._state = JobState(job_id="t", argv=[], cwd=".", status="running", live=True)
    m.observed_delay_ms = observed
    return m


def test_button_steps_from_launch_delay_then_from_last_change():
    m = _live_manager(observed=25.0)
    assert m.step_delay(+1)["delay_ms"] == 30
    m.update_assist(delay_ms=15.0)                    # dashboard Set in between
    assert m.step_delay(-1)["delay_ms"] == 10         # last change wins, no priority
    assert m.step_delay(-1)["delay_ms"] == 10         # floor
    m.update_assist(delay_ms=40.0)
    assert m.step_delay(+1)["delay_ms"] == 40         # ceiling


def test_button_clamps_out_of_range_launch_delay_and_ignores_without_live_run():
    assert _live_manager(observed=50.0).step_delay(-1)["delay_ms"] == 40
    assert _live_manager(observed=None).step_delay(+1) is None
    m = _live_manager(observed=25.0)
    m._state.live = False
    assert m.step_delay(+1) is None and m.assist_state["delay_ms"] is None


def test_button_press_and_dashboard_click_never_drop_each_other():
    import threading
    m = _live_manager(observed=25.0)
    m.update_assist(assist=True)
    off = threading.Thread(target=lambda: [m.update_assist(assist=False) for _ in range(2000)])
    off.start()
    for _ in range(2000):
        m.step_delay(+1)
    off.join()
    assert m.assist_state["assist"] is False and m.assist_state["delay_ms"] == 40


def test_button_debounce_reports_one_press_per_edge():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dashboard.backend.delay_buttons import DEBOUNCE_S, Button
    level = [False]
    b = Button(lambda: level[0])
    level[0] = True
    assert not b.pressed(0.0)                   # change seen, debouncing
    assert not b.pressed(DEBOUNCE_S / 2)
    assert b.pressed(DEBOUNCE_S + .001)         # press edge
    assert not b.pressed(DEBOUNCE_S * 3)        # held: no repeat
    level[0] = False                            # release: no press reported
    assert not b.pressed(1.0) and not b.pressed(1.0 + DEBOUNCE_S + .001)
    level[0] = True                             # bounce shorter than debounce
    assert not b.pressed(2.0)
    level[0] = False
    assert not b.pressed(2.01)
    level[0] = True
    assert not b.pressed(2.02) and not b.pressed(2.02 + DEBOUNCE_S / 2)
