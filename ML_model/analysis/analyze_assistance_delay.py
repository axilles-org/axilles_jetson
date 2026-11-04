"""Audit recorded command timing; replay commands only, never connect to hardware.

Run from ML_model: PYTHONPATH=src python3 analysis/analyze_assistance_delay.py
"""
from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from exo.config import Config, replace
from exo.deploy.assistance import AssistanceController
from exo.deploy.motor import MotorInterface


def main():
    source = ROOT / 'runs/imported/backup_ml_control_model_4745a55'
    output = ROOT / 'runs/hw_analysis/20261102_delay'
    output.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(source / 'ML_model/logs/powered_run.csv')
    t = d.t_s.to_numpy()
    stance = d.stance.to_numpy().astype(bool)
    enabled = d.assist_on.to_numpy().astype(bool)
    full = enabled & (d.session_ramp.to_numpy() >= .99)
    channels = [f'{part}_{kind}{axis}' for part in ('foot', 'shank')
                for kind in ('a', 'g') for axis in 'xyz']
    freshness = []
    for name in channels:
        changed = np.flatnonzero(np.diff(d[name].to_numpy()) != 0) + 1
        freshness.append(dict(channel=name, last_numeric_change_s=float(t[changed[-1]]) if len(changed) else None,
                              unique_values_while_enabled=int(d.loc[enabled, name].nunique())))
    pd.DataFrame(freshness).to_csv(output / 'imu_numeric_changes.csv', index=False)
    # Raw logged stance intervals may include FSR chatter. Do not call these
    # independent ground-truth strides or treat stance-off as measured toe-off.
    starts = np.flatnonzero(stance & ~np.r_[False, stance[:-1]])
    ends = np.flatnonzero(~stance & np.r_[False, stance[:-1]])
    intervals = []
    for a in starts:
        later = ends[ends > a]
        if not len(later):
            continue
        b = later[0]
        later = starts[starts > b]
        if not len(later):
            continue
        c = later[0]
        if not (.25 < t[b]-t[a] < 1.4 and .1 < t[c]-t[b] < 1 and full[a:c].all()):
            continue
        predicted = np.maximum(-d.predicted_nm_per_kg.iloc[a:c].to_numpy(), 0) * 72 * d.assist_scale.iloc[a:c].to_numpy()
        requested = d.assist_cmd_nm.iloc[a:c].to_numpy()
        queued = d.sent_torque_nm.iloc[a:c].to_numpy()
        if queued.max() < .3:
            continue
        def onset(x, threshold=.1):
            hits = np.flatnonzero(x > threshold*x.max())
            return t[a+hits[0]] if len(hits) else np.nan
        peak = t[a+np.argmax(queued)]
        active = np.flatnonzero(queued > .05)
        intervals.append(dict(start_s=t[a], stance_off_s=t[b], next_start_s=t[c],
                              queued_peak_before_stance_off_ms=(t[b]-peak)*1000,
                              queued_peak_stance_percent=100*(peak-t[a])/(t[b]-t[a]),
                              predicted_to_queued_onset_ms=(onset(queued)-onset(predicted))*1000,
                              requested_to_queued_onset_ms=(onset(queued)-onset(requested))*1000,
                              onset_lag_25pct_ms=(onset(queued,.25)-onset(predicted,.25))*1000,
                              swing_tail_ms=max(0, (t[a+active[-1]]-t[b])*1000)))
    intervals = pd.DataFrame(intervals)
    intervals.to_csv(output / 'stance_intervals.csv', index=False)

    # Counterfactual routing of the recorded model outputs. This deliberately
    # bypasses the new freshness stop to exercise BOTH signs offline. It is not
    # a replay of new model predictions or a validation of live sensor freshness.
    cfg = Config.load(ROOT / 'runs/deploy/config.yaml')
    controller = AssistanceController(replace(cfg, **{'deploy.assistance_mode': 'bidirectional'}).deploy)
    motor = MotorInterface(torque_limit_nm=5, slew_nm_per_s=40, command_sign=1, dry_run=True)
    rows = []
    for i, r in d.iterrows():
        dt = min(t[i]-t[i-1], .05) if i else .01
        controller.scale = r.assist_scale
        cmd = controller.update(r.predicted_nm_per_kg, 72, r.heel_fsr_raw, r.toe_fsr_raw,
                                dt, stance=bool(r.stance))
        allowed = bool(r.assist_on and r.stance and r.buffer_ready and r.calibrated and r.loop_ms <= 25)
        target = cmd.torque_nm*r.session_ramp if allowed else 0
        sent = motor.send_torque(target, dt=dt, immediate_zero=not allowed)
        rows.append(sent)
    simulated = np.asarray(rows)
    pd.DataFrame({'t_s':t, 'recorded_prediction':d.predicted_nm_per_kg,
                  'stance':stance, 'recorded_queued_nm':d.sent_torque_nm,
                  'counterfactual_queued_nm':simulated}).to_csv(output / 'command_counterfactual.csv', index=False)
    assert np.all(simulated[~stance] == 0)
    assert np.all(simulated[~enabled] == 0)
    assert np.max(np.abs(simulated)) <= 5
    assert np.any(simulated < 0), 'Dorsiflexion commands should occur in this record'
    summary = dict(source_commit='4745a55', csv='ML_model/logs/powered_run.csv',
                   associated_run_id='20261102_182941_1d037a',
                   association='inferred: latest source DB record uniquely matches CSV sample count; CSV has no run ID',
                   rows=len(d), duration_s=float(t[-1]-t[0]),
                   mean_update_hz=float((len(d)-1)/(t[-1]-t[0])),
                   loop_median_ms=float(d.loop_ms.median()),
                   enabled_start_s=float(t[enabled][0]), enabled_end_s=float(t[enabled][-1]),
                   enabled_positive_prediction_samples=int((enabled & (d.predicted_nm_per_kg > 0)).sum()),
                   enabled_positive_prediction_stance_samples=int((enabled & stance & (d.predicted_nm_per_kg > 0)).sum()),
                   recorded_negative_commands=int((d.sent_torque_nm < 0).sum()),
                   enabled_swing_nonzero_commands=int((enabled & ~stance & (d.sent_torque_nm.abs() > .01)).sum()),
                   all_imu_channels_constant_while_enabled=all(x['unique_values_while_enabled']==1 for x in freshness),
                   qualifying_intervals=len(intervals),
                   median_peak_before_stance_off_ms=float(intervals.queued_peak_before_stance_off_ms.median()),
                   median_prediction_to_command_onset_ms=float(intervals.predicted_to_queued_onset_ms.median()),
                   max_prediction_to_command_onset_ms=float(intervals.predicted_to_queued_onset_ms.max()),
                   median_swing_tail_ms=float(intervals.swing_tail_ms.median()),
                   median_requested_queued_error_nm=float(np.median(abs(d.loc[full,'sent_torque_nm']-d.loc[full,'assist_cmd_nm']))),
                   counterfactual_negative_commands=int((simulated < 0).sum()),
                   counterfactual_swing_commands=int((simulated[~stance] != 0).sum()),
                   counterfactual_peak_nm=float(np.max(abs(simulated))))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/axilles-delay-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4,1,figsize=(12,10),sharex=True)
    axes[0].plot(t,d.predicted_nm_per_kg,label='Predicted moment (Nm/kg)')
    axes[0].axhline(0,color='gray',lw=.5)
    axes[1].plot(t,d.assist_cmd_nm,label='Requested (Nm)')
    axes[1].plot(t,d.sent_torque_nm,'--',label='Queued command (Nm)')
    axes[1].plot(t,simulated,alpha=.65,label='Counterfactual bidirectional (Nm)')
    axes[2].plot(t,d.ankle_angle_deg,label='Logged ankle angle (deg)')
    zoom_angle = d.loc[(d.t_s >= 30) & (d.t_s <= 37), 'ankle_angle_deg']
    axes[2].set_ylim(zoom_angle.min()-3, zoom_angle.max()+3)
    axes[3].plot(t,d.foot_gy,label='Foot gyro Y (frozen)')
    axes[3].plot(t,d.shank_gy,label='Shank gyro Y (frozen)')
    for ax in axes:
        ax.fill_between(t,0,1,where=stance,transform=ax.get_xaxis_transform(),alpha=.1,color='green',label='Logged stance')
        ax.legend(loc='upper left',fontsize=8)
        ax.grid(alpha=.2)
    axes[-1].set(xlim=(30,37),xlabel='Time in recording (s)')
    fig.suptitle('November 2: recorded signals and offline command-routing counterfactual')
    fig.tight_layout()
    fig.savefig(output/'timing_and_counterfactual.png',dpi=160)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
