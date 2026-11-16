import sys,json
from pathlib import Path
import numpy as np
import torch
root=Path('/Users/narayanan/PycharmProjects/axilles_jetson/ML_model')
sys.path.insert(0,str(root/'src'))
from exo.config import Config,replace
from exo.training.evaluate import Evaluator,_metrics
from exo.data.dataset import WindowDataset
from exo.data.scalers import ScalerBundle
torch.set_num_threads(2)
run=root/'runs/deploy';cfg=replace(Config.load(run/'config.yaml'), **{'data.cache_window_index':False})
e=Evaluator(cfg,str(run/'deploy.pt'),device='cpu')
package=ScalerBundle.load(run)
assert np.array_equal(package.scaler_y.mean_,e.scalers.scaler_y.mean_)
assert np.array_equal(package.scaler_y.scale_,e.scalers.scaler_y.scale_)
assert np.array_equal(package.subset_x(cfg.data.feature_names())[0],e.scalers.subset_x(cfg.data.feature_names())[0])
d=WindowDataset(cfg.data,cfg.paths.processed(),e.scalers,split='test')
ids=list(range(0,len(d),10));pred=[];truth=[];stances=[]
with torch.inference_mode():
 for start in range(0,len(ids),128):
  items=[d[i] for i in ids[start:start+128]]
  x=torch.stack([a[0] for a in items]);sid=torch.tensor([e.subject_index[a[3]] for a in items]);mass=np.array([e.subject_mass[a[3]] for a in items])
  z=e.model(x,sid)[:,0,-1].numpy()
  pred.extend((z*e._y_scale+e._y_mean)/mass)
  truth.extend((np.array([a[1][0,-1].item() for a in items])*e._y_scale+e._y_mean)/mass)
  stance_index=d.feature_names.index('stance')
  stances.extend((x[:,stance_index,-1].numpy()*d._x_scale[stance_index]+d._x_mean[stance_index])>.5)
  if start%1280==0:print('Evaluated',start+len(items),'/',len(ids),flush=True)
p=np.array(pred);y=np.array(truth);s=np.array(stances)
result=dict(checkpoint=str(run/'deploy.pt'),split='test',subjects=cfg.data.split.test,windows=len(ids),sampling='Every tenth test window; last timestep only; clean offline inputs; no retraining',threshold_nm_per_kg=.05,groups={})
for name,mask in [('all',np.ones(len(y),bool)),('dorsi_y_gt_005',y>.05),('plantar_y_lt_minus005',y<-.05),('dorsi_in_stance',(y>.05)&s),('dorsi_in_swing',(y>.05)&~s)]:
 if not mask.any():continue
 result['groups'][name]=dict(n=int(mask.sum()),**_metrics(p[mask],y[mask]),positive_prediction_fraction=float(np.mean(p[mask]>0)),negative_prediction_fraction=float(np.mean(p[mask]<0)),mean_truth=float(y[mask].mean()),mean_prediction=float(p[mask].mean()))
out=root/'analysis/gatech_moment_sign/deployed_direction_metrics.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
