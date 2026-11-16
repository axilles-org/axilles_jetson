from pathlib import Path
import os,json,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/axilles-delay-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'logs/powered_run.csv';out=Path(__file__).resolve().parent
x=pd.read_csv(p);assert len(x)==24831,'Source file changed; re-identify recording before plotting'
t=x.t_s.to_numpy();on=x.assist_on.astype(bool).to_numpy()
rows=[];holds=[]
for part in ('foot','shank'):
 for kind in ('a','g'):
  cols=[f'{part}_{kind}{a}' for a in 'xyz']
  v=x[cols].to_numpy();starts=np.flatnonzero(np.r_[True,np.any(np.diff(v,axis=0)!=0,axis=1)])
  ends=np.r_[starts[1:]-1,len(x)-1]
  blocks=[(float(t[b]-t[a]),float(t[a]),float(t[b])) for a,b in zip(starts,ends) if on[a:b+1].any()]
  duration,start,end=max(blocks)
  holds.append(dict(group=f'{part}_{kind}',longest_constant_vector_s=duration,start_s=start,end_s=end))
  for col in cols:
   changes=np.flatnonzero(np.diff(x[col].to_numpy())!=0)+1
   rows.append(dict(channel=col,unique_values_during_assistance=int(x.loc[on,col].nunique()),last_numeric_change_s=float(t[changes[-1]])))
pd.DataFrame(rows).to_csv(out/'oct31_channel_evidence.csv',index=False)
(out/'oct31_summary.json').write_text(json.dumps(dict(source_csv=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),associated_run_id='20261031_211823_1ef069',association='Inferred from unique sample-count match to source database; CSV has no run ID.',duration_s=float(t[-1]),rows=len(x),channels=rows,longest_constant_vectors=holds,limitation='Changing numeric values rule out the sustained full-run freeze seen on Nov 2, but do not establish packet age or exclude shorter holds.'),indent=2)+'\n')
fig,axes=plt.subplots(7,1,figsize=(12,13),sharex=True,layout='constrained')
for ax,(part,kind,label,unit) in zip(axes[:4],[('foot','a','Foot acceleration','m/s²'),('shank','a','Shank acceleration','m/s²'),('foot','g','Foot gyroscope','rad/s'),('shank','g','Shank gyroscope','rad/s')]):
 for axis,color in zip('xyz',['#1f77b4','#d95f02','#7570b3']):ax.plot(t,x[f'{part}_{kind}{axis}'],label=axis.upper(),color=color,lw=.6)
 ax.set_ylabel(f'{label}\n({unit})');ax.legend(loc='upper right',ncol=3,fontsize=8)
axes[4].plot(t,x.ankle_encoder_deg,color='#333333',lw=.7);axes[4].set_ylabel('Raw ankle encoder\n(deg)')
axes[5].plot(t,x.heel_fsr_raw,label='Heel',lw=.6);axes[5].plot(t,x.toe_fsr_raw,label='Toe',lw=.6);axes[5].set_ylabel('Contact sensors\n(raw counts)');axes[5].legend(loc='upper right',ncol=2,fontsize=8)
axes[6].plot(t,x.sent_torque_nm,color='#a63603',lw=.6);axes[6].set_ylabel('Logged motor command\n(Nm)');axes[6].set_xlabel('Time from recording start (s)')
for ax in axes:
 ax.fill_between(t,0,1,where=on,transform=ax.get_xaxis_transform(),color='#55a868',alpha=.12)
 ax.grid(alpha=.2);ax.set_xlim(t[0],t[-1])
axes[0].set_title('Green shading: assistance enabled. Raw logged values; no smoothing applied.',fontsize=10)
fig.suptitle('October 31, 2026 — long ML recording\n24,831 samples over 350.120 s; acceleration and gyroscope values continue changing',fontsize=14)
fig.savefig(out/'oct31_imu_evidence.png',dpi=155);fig.savefig(out/'oct31_imu_evidence.pdf')
print('Saved October 31 full-recording plot.')
print(pd.DataFrame(rows).to_string(index=False))
