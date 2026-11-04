import sys,json,os
from pathlib import Path
import numpy as np
import pandas as pd
root=Path('/Users/narayanan/PycharmProjects/axilles_jetson/ML_model')
sys.path.insert(0,str(root/'src'))
from exo.config import Config
from exo.deploy.assistance import AssistanceController
from exo.deploy.motor import MotorInterface
p=root/'runs/imported/backup_ml_control_model_4745a55/ML_model/logs/powered_run.csv'
d=pd.read_csv(p)
cfg=Config.load(root/'runs/deploy/config.yaml').deploy
c=AssistanceController(cfg);motor=MotorInterface(torque_limit_nm=2,command_sign=1,slew_nm_per_s=20,dry_run=True)
rows=[]
for i,r in d.iterrows():
 dt=min(r.t_s-d.t_s.iloc[i-1],.05) if i else .01
 c.scale=r.assist_scale
 cmd=c.update(r.predicted_nm_per_kg,72,r.heel_fsr_raw,r.toe_fsr_raw,dt,stance=bool(r.stance))
 allowed=bool(r.assist_on and r.stance and r.buffer_ready and r.calibrated and not r.model_fault and r.loop_ms<=25)
 sent=motor.send_torque(cmd.torque_nm*r.session_ramp if allowed else 0,dt=dt,immediate_zero=not allowed)
 rows.append(sent)
d['simulated_current_command_nm']=rows
out=root/'analysis/gatech_moment_sign'
d[['t_s','predicted_nm_per_kg','predicted_nm','stance','assist_on','assist_scale','sent_torque_nm','simulated_current_command_nm']].to_csv(out/'nov2_command_evidence.csv',index=False)
summary=dict(source=str(p),mass_kg=72,simulation='Recorded predictions and scales; current bidirectional controller, motor cap 2 Nm, slew 20 Nm/s. No sensor freshness enforcement or new inference. NOT hardware measurements.',peaks={})
for col in ['sent_torque_nm','simulated_current_command_nm']:
 summary['peaks'][col]={key:d.loc[getattr(d[col],key)(),['t_s','predicted_nm_per_kg','stance','assist_on','assist_scale',col]].to_dict() for key in ['idxmin','idxmax']}
(out/'nov2_command_peaks.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
i=d.simulated_current_command_nm.idxmin();print(d.loc[i-2:i+3,['t_s','predicted_nm_per_kg','stance','assist_scale','sent_torque_nm','simulated_current_command_nm']].to_string(index=False))
os.environ['MPLCONFIGDIR']='/tmp/axilles-delay-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,1,figsize=(10,5),sharex=True,layout='constrained')
s=d[(d.t_s>=d.t_s.iloc[i]-.2)&(d.t_s<=d.t_s.iloc[i]+.25)]
ax[0].plot(s.t_s,s.predicted_nm_per_kg,marker='.',label='Recorded model prediction (Nm/kg)')
ax[1].plot(s.t_s,s.sent_torque_nm,marker='.',label='Recorded motor command (Nm)')
ax[1].plot(s.t_s,s.simulated_current_command_nm,marker='.',label='Offline updated logic: 2 Nm cap, 20 Nm/s slew')
for a in ax:
 a.axhline(0,color='black',lw=.8);a.legend(fontsize=8);a.grid(alpha=.2)
ax[0].set_ylabel('Nm/kg');ax[1].set(ylabel='Nm',xlabel='Time in November 2 recording (s)')
fig.suptitle('Recorded prediction sign changes; updated-command curve is simulation only')
fig.savefig(out/'nov2_sign_transition.png',dpi=150)
