from pathlib import Path
import os,json,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/axilles-delay-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'runs/imported/backup_ml_control_model_4745a55/ML_model/logs/powered_run.csv'
out=Path(__file__).resolve().parent
x=pd.read_csv(p);t=x.t_s.to_numpy();on=x.assist_on.astype(bool).to_numpy()
rows=[]
for part in ('foot','shank'):
 for kind in ('a','g'):
  for axis in 'xyz':
   col=f'{part}_{kind}{axis}';v=x[col].to_numpy();changes=np.flatnonzero(np.diff(v)!=0)+1
   rows.append(dict(channel=col,last_numeric_change_s=float(t[changes[-1]]) if len(changes) else None,unique_values_during_assistance=int(x.loc[on,col].nunique()),constant_value_during_assistance=float(x.loc[on,col].iloc[0])))
freeze=max(r['last_numeric_change_s'] for r in rows)
pd.DataFrame(rows).to_csv(out/'latest_channel_evidence.csv',index=False)
summary=dict(source_csv=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),associated_run_id='20261102_182941_1d037a',association='Inferred from latest source database record and unique sample-count match; CSV has no run ID.',rows=len(x),duration_s=float(t[-1]),all_imu_channels_constant_after_s=freeze,assistance_start_s=float(t[on][0]),assistance_end_s=float(t[on][-1]),enabled_samples=int(on.sum()),channels=rows,limitation='Proves logged raw values stop changing, not whether physical sensor packets continued arriving. No per-report receipt timestamps/counters are present.')
(out/'latest_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(7,1,figsize=(12,13),sharex=True,layout='constrained')
colors=['#1f77b4','#d95f02','#7570b3']
for ax,(part,kind,label,unit) in zip(axes[:4],[('foot','a','Foot acceleration','m/s²'),('shank','a','Shank acceleration','m/s²'),('foot','g','Foot gyroscope','rad/s'),('shank','g','Shank gyroscope','rad/s')]):
 for axis,color in zip('xyz',colors):ax.plot(t,x[f'{part}_{kind}{axis}'],label=axis.upper(),color=color,lw=1)
 ax.set_ylabel(f'{label}\n({unit})');ax.legend(loc='upper right',ncol=3,fontsize=8)
 ax.text(.42,.74,'All XYZ values constant throughout assistance',transform=ax.transAxes,fontsize=10,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
axes[4].plot(t,x.ankle_encoder_deg,color='#333333');axes[4].set_ylabel('Raw ankle encoder\n(deg)')
axes[5].plot(t,x.heel_fsr_raw,label='Heel',lw=.8);axes[5].plot(t,x.toe_fsr_raw,label='Toe',lw=.8);axes[5].set_ylabel('Contact sensors\n(raw counts)');axes[5].legend(loc='upper right',ncol=2,fontsize=8)
axes[6].plot(t,x.sent_torque_nm,color='#a63603',lw=.8);axes[6].set_ylabel('Logged motor command\n(Nm)');axes[6].set_xlabel('Time from recording start (s)')
for ax in axes:
 ax.axvline(freeze,color='#b2182b',ls='--',lw=1)
 ax.fill_between(t,0,1,where=on,transform=ax.get_xaxis_transform(),color='#55a868',alpha=.12)
 ax.grid(alpha=.2);ax.set_xlim(t[0],t[-1])
axes[0].set_title('Red dashed line: last IMU numeric change (6.1777 s)   |   Green: assistance enabled (19.2101–84.6397 s)',fontsize=10)
fig.suptitle('Latest available ML recording — November 2, 2026\n8,446 samples over 87.735 s; raw IMU values freeze while encoder and contact signals change',fontsize=14)
fig.savefig(out/'latest_imu_evidence.png',dpi=155)
fig.savefig(out/'latest_imu_evidence.pdf')
print(json.dumps({k:v for k,v in summary.items() if k!='channels'},indent=2))
