"""Forecast alignment, boundaries, causal filtering and saved configuration."""
from dataclasses import replace as dc_replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from exo.config import Config, replace
from exo.data.dataset import WindowDataset
from exo.data.raw_ingest import _decimate
from exo.data.window_index import WindowIndex, _key
from exo.training.evaluate import Evaluator


def toy(cfg,tmp_path,h=2):
    data=dc_replace(cfg.data,feature_sets={'toy':['sensor']},feature_set='toy',
                    window_length=4,min_segment_length=4,stride=1,
                    prediction_horizon_ms=h*10,cache_window_index=False)
    name=f'{data.split.train[0]}_trial'
    np.savez(tmp_path/f'{name}.npz',features=np.arange(50,dtype=np.float32)[:,None],
             target=np.arange(50,dtype=np.float32)[:,None],feature_columns=['sensor'])
    scalers=SimpleNamespace(subset_x=lambda _: (np.array([0.]),np.array([1.])),
                            scaler_y=SimpleNamespace(mean_=np.array([0.]),scale_=np.array([1.])))
    return WindowDataset(data,tmp_path,scalers),name


@pytest.mark.parametrize('h',[0,5,10,15])
def test_exact_horizon_and_input_unchanged(cfg,tmp_path,h):
    ds,_=toy(cfg,tmp_path,h)
    for i in [0,len(ds)-1]:
        x,y,_,_=ds[i]
        _,start,end=ds.index.entries[i]
        np.testing.assert_array_equal(x[0],np.arange(start,end+1))
        np.testing.assert_array_equal(y[0],np.arange(start+h,end+h+1))
        assert end+h<50


def test_future_observations_never_enter_input(cfg,tmp_path):
    ds,_=toy(cfg,tmp_path)
    x,y,_,_=ds[0]
    ds._feat_mm[0][4:]=9999
    np.testing.assert_array_equal(ds[0][0],x)
    assert y[0,-1]==5


def test_future_target_gaps_and_trial_boundaries(tmp_path):
    target=np.arange(12,dtype=float)[:,None]
    target[7]=np.nan
    for name in ['AB06_first','AB06_second']:
        np.savez(tmp_path/f'{name}.npz',features=np.ones((12,1)),target=target)
    idx=WindowIndex.build(tmp_path,['AB06_first','AB06_second'],4,1,4,
                          cache=False,valid_tail=2,horizon=2)
    assert set(idx.entries[:,0])=={0,1}
    for _,s,e in idx.entries:
        assert e+2<12
        assert np.isfinite(target[e+1:e+3]).all()
    assert not any(e in [5,6] for _,_,e in idx.entries)


def test_mode_filter_uses_future_label(tmp_path):
    mode=np.array(['walk']*12)
    mode[5]='idle'
    np.savez(tmp_path/'AB06_trial.npz',features=np.ones((12,1)),target=np.ones((12,1)),mode=mode)
    idx=WindowIndex.build(tmp_path,['AB06_trial'],4,1,4,cache=False,valid_tail=1,
                          drop_modes=('idle',),horizon=2)
    assert 3 not in idx.entries[:,2]


def test_cache_separates_horizons():
    assert len({_key(['AB06_trial'],300,10,300,horizon=h) for h in [0,5,10,15]})==4


@pytest.mark.parametrize('h',[-50,15,float('nan'),float('inf')])
def test_invalid_horizon_rejected(cfg,h):
    with pytest.raises(ValueError):
        replace(cfg,**{'data.prediction_horizon_ms':h})


def test_random_latency_not_mixed_with_fixed_forecast(cfg):
    with pytest.raises(ValueError,match='latency_samples'):
        replace(cfg,**{'data.prediction_horizon_ms':50,'train.augment.latency_samples':5})


def test_causal_filter_does_not_read_future():
    rng=np.random.default_rng(7)
    before=rng.normal(size=(1000,2))
    after=before.copy()
    after[500:]+=1000
    a=_decimate(before,10,True,zero_phase=False)
    b=_decimate(after,10,True,zero_phase=False)
    np.testing.assert_array_equal(a[:50],b[:50])


def test_saved_config_resolves_inheritance(tmp_path):
    root=Path(__file__).resolve().parents[1]
    for h in [50,100,150]:
        cfg=Config.load(root/f'configs/anticipation/tcn_{h}ms.yaml')
        assert cfg.data.prediction_horizon_samples==h//10
        assert cfg.data.ingest.causal_inputs
        cfg.save(tmp_path)
        assert Config.load(tmp_path/'config.yaml')==cfg


def test_inheritance_cycle_rejected(tmp_path):
    (tmp_path/'a.yaml').write_text('extends: b.yaml\n')
    (tmp_path/'b.yaml').write_text('extends: a.yaml\n')
    with pytest.raises(ValueError,match='cycle'):
        Config.load(tmp_path/'a.yaml')


def test_evaluation_places_prediction_at_target_time(cfg,tmp_path):
    ds,name=toy(cfg,tmp_path)
    e=Evaluator.__new__(Evaluator)
    e.cfg=SimpleNamespace(data=ds.cfg)
    e.device=torch.device('cpu')
    e.subject_index={name.split('_')[0]:0}
    e.subject_mass={name.split('_')[0]:1.0}
    e._y_mean,e._y_scale=0.0,1.0
    e.model=lambda x,s:x+2
    result=e._reconstruct(ds,0)
    valid=np.isfinite(result.pred_streaming)
    assert np.flatnonzero(valid)[0]==5
    np.testing.assert_array_equal(result.pred_streaming[valid],result.truth[valid])
    assert not result.valid[:2].any()
