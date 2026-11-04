from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/Users/narayanan/PycharmProjects/exo-data/hf_dataset_tx_mujoco')
original=root.parent/'hf_dataset'
out=Path('/Users/narayanan/PycharmProjects/axilles_jetson/ML_model/analysis/gatech_moment_sign')
meta=pd.read_parquet(root/'metadata.parquet')
values=[]; curves=[]; peaks=[]; rows=[]; compared=0; unchanged=0; example=None
for r in meta.itertuples():
 rel=Path('subjects')/r.subject/f'{r.trial}__id.parquet'
 df=pd.read_parquet(root/rel,columns=['time_s','ankle_angle_r_moment'])
 y=df.ankle_angle_r_moment.to_numpy()/r.weight_kg; t=df.time_s.to_numpy()
 values.append(y[np.isfinite(y)])
 if (original/rel).exists():
  src=pd.read_parquet(original/rel,columns=['time_s','ankle_angle_r_moment'])
  compared+=1; unchanged+=int(np.array_equal(src.to_numpy(),df.to_numpy(),equal_nan=True))
 gc=pd.read_parquet(root/'subjects'/r.subject/f'{r.trial}__gcRight.parquet')
 assert np.allclose(gc.time_s.to_numpy(),t)
 hs=np.flatnonzero(np.diff(gc.HeelStrike.to_numpy()) < -50)+1
 n=0
 for a,b in zip(hs[:-1],hs[1:]):
  if not .5<=t[b]-t[a]<=2 or not np.isfinite(y[a:b+1]).all():continue
  curve=np.interp(np.linspace(t[a],t[b],101),t[a:b+1],y[a:b+1]);curves.append(curve)
  peaks.append((float(y[a:b].min()),float(y[a:b].max())))
  n+=1
 rows.append(dict(subject=r.subject,trial=r.trial,min_nm_per_kg=float(np.nanmin(y)),max_nm_per_kg=float(np.nanmax(y)),positive_fraction=float(np.mean(y>0)),complete_strides=n))
 if r.subject=='AB10' and r.trial=='treadmill_04_01':
  starts=hs[t[hs]>50]; a,b=starts[0],starts[3]
  example=(t[a:b+1]-t[a], y[a:b+1])
y=np.concatenate(values); c=np.array(curves); p=np.array(peaks)
s=dict(dataset=str(root),target='ankle_angle_r_moment',units='Nm/kg (raw Nm divided by each subject mass)',subjects=int(meta.subject.nunique()),trials=len(rows),finite_samples=len(y),positive_fraction=float(np.mean(y>0)),negative_fraction=float(np.mean(y<0)),positive_above_005_fraction=float(np.mean(y>.05)),negative_below_minus005_fraction=float(np.mean(y<-.05)),trials_with_both_signs=sum(r['min_nm_per_kg']<0<r['max_nm_per_kg'] for r in rows),complete_strides=len(c),median_stride_min_nm_per_kg=float(np.median(p[:,0])),median_stride_max_nm_per_kg=float(np.median(p[:,1])),original_label_files_compared=compared,original_label_files_identical=unchanged,notes='Raw native-rate labels; stride bounds from HeelStrike wraps, duration 0.5 to 2 seconds. Sample fractions include near-zero values and are not proof of physiological accuracy. 0.05 threshold is descriptive, not a control setting.')
(out/'summary.json').write_text(json.dumps(s,indent=2)+'\n');pd.DataFrame(rows).to_csv(out/'per_trial.csv',index=False)
fig,axes=plt.subplots(2,1,figsize=(10,6),layout='constrained')
x=np.arange(101);m=c.mean(axis=0)
axes[0].plot(x,m,color='#154f8c',label='Mean across complete strides')
axes[0].fill_between(x,np.percentile(c,10,axis=0),np.percentile(c,90,axis=0),alpha=.2,color='#154f8c',label='10–90% of strides')
axes[0].set(xlabel='Stride (%) — heel strike to heel strike',title=f'GaTech treadmill target: {len(c):,} strides, {meta.subject.nunique()} subjects')
axes[0].legend(fontsize=9)
t,v=example;axes[1].plot(t,v,color='#154f8c')
axes[1].fill_between(t,0,v,where=v>0,color='#d95f02',label='Positive: dorsiflexion moment')
axes[1].fill_between(t,0,v,where=v<0,color='#1b9e77',alpha=.5,label='Negative: plantarflexion moment')
axes[1].set(xlabel='Time from first selected heel strike (s)',title='Raw example: AB10, treadmill_04_01, first three full strides after 50 s')
axes[1].legend(fontsize=9)
for ax in axes:ax.axhline(0,color='black',lw=.8);ax.set_ylabel('Ankle moment (Nm/kg)');ax.grid(alpha=.2)
fig.savefig(out/'gatech_signed_moment.png',dpi=160)
print(json.dumps(s,indent=2))
