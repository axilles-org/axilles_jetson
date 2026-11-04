"""Embedding ablation: score the held-out subjects with the nearest training subject's
embedding (as deployed), the average embedding, and each training subject's embedding in
turn. Saves embedding_ablation_<variant>.json in the run directory.

    python analysis/embedding_ablation.py runs/terrain/<stamp> tcn_emb treadmill,levelground,ramp
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML / "scripts")); sys.path.insert(0, str(ML / "src"))
import terrain_experiment as te
from exo.config import replace
from exo.data.dataset import WindowDataset
from exo.data.demographics import subject_mass
from exo.data.scalers import ScalerBundle
from exo.models import TCN
from exo.training.subject_embedding import SubjectIndex

out_dir, variant, terrains = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
D = Path.home() / "PycharmProjects/exo-data"
tag = "_stairs" if "stair" in terrains else ""
cfg = replace(te.base_config(D / f"hf_dataset_terrain{tag}", D / f"processed_terrain{tag}", 5, terrains),
              **{**te.VARIANTS[variant], "paths.runs_dir": str(out_dir), "data.stride": 10})
run = out_dir / variant
sc = ScalerBundle.load(run); meta = json.loads((run / "model_meta.json").read_text())
idx = SubjectIndex(mapping=dict(meta["subject_index"]), num_training_subjects=meta["num_training_subjects"])
dev = torch.device("mps")
m = TCN.from_config(cfg.model, cfg.data.num_features, 1, idx.num_training_subjects)
m.load_state_dict(torch.load(run / "best.pt", map_location="cpu")); m.to(dev).eval()
ds = WindowDataset(cfg.data, cfg.paths.processed(), sc, split="test")
mass = subject_mass(cfg.paths.demographics()); ym, ys = float(sc.scaler_y.mean_[0]), float(sc.scaler_y.scale_[0])
modes, bad = {}, set()
for n in ds.index.trial_names:
    with np.load(cfg.paths.processed() / f"{n}.npz") as d:
        modes[n] = d["mode"] if "mode" in d.files else None
        c = d["feature_columns"].tolist()
        if np.nanmax(np.abs(d["features"][:, c.index("gon_ankle_sagittal")])) > te.GON_LIMIT_DEG: bad.add(n)
W = m.subject_embedding.weight.detach().clone()
train_ids = sorted({v for v in idx.mapping.values() if v < W.shape[0]})
choices = {"nearest": None, "mean": W.mean(0)} | {f"train{j}": W[j] for j in range(W.shape[0])}
X, T, G, S, NI = [], [], [], [], []
loader = DataLoader(ds, batch_size=512, shuffle=False)
k = 0
for x, y, names, subs in loader:
    X.append(x); ends = ds.index.entries[k:k + len(names), 2]; k += len(names)
    msk = np.array([n not in bad for n in names])
    T.append(((y[:, 0, -1].numpy() * ys + ym) / np.array([mass[s] for s in subs]))); S += list(subs)
    G += [te.mode_group(modes[n][e]) if modes[n] is not None else "treadmill" for n, e in zip(names, ends)]
    NI.append(msk)
keep = np.concatenate(NI); t = np.concatenate(T)[keep]; g = np.array(G)[keep]; s = np.array(S)[keep]
mass_v = np.array([mass[z] for z in s])
res = {}
with torch.inference_mode():
    for name, row in choices.items():
        ps = []
        for xb, sb in zip(X, np.array_split(np.array(S), np.cumsum([len(b) for b in X])[:-1])):
            si = torch.tensor(idx.indices(list(sb)), device=dev)
            if row is not None:
                m.subject_embedding.weight.copy_(row.expand_as(W))
            ps.append(m(xb.to(dev), si)[:, 0, -1].float().cpu().numpy())
            m.subject_embedding.weight.copy_(W)
        p = (np.concatenate(ps)[keep] * ys + ym) / mass_v
        r2 = lambda a, b: 1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum()
        res[name] = {"overall": r2(p, t), **{gr: r2(p[g == gr], t[g == gr]) for gr in te.GROUPS if (g == gr).sum() > 100 and gr != "stand"},
                     "per_subject": {z: r2(p[s == z], t[s == z]) for z in sorted(set(s))}}
near = {z: idx.indices([z])[0] for z in sorted(set(s))}
tr = [v for k2, v in res.items() if k2.startswith("train")]
cols = [c for c in res["nearest"] if c != "per_subject"]
tofl = lambda v: {k: tofl(x) for k, x in v.items()} if isinstance(v, dict) else float(v)
(out_dir / f"embedding_ablation_{variant}.json").write_text(json.dumps(
    {"nearest_index": {str(z): int(i) for z, i in near.items()}, **{k: tofl(v) for k, v in res.items()}}, indent=1))
print(variant, out_dir.name, "nearest-train index per test subject:", near)
print(f"{'emb':10s}" + "".join(f"{c[:9]:>10s}" for c in cols))
for name in ["nearest", "mean"]:
    print(f"{name:10s}" + "".join(f"{res[name][c]:10.3f}" for c in cols))
for lab, f in [("train min", np.min), ("train med", np.median), ("train max", np.max)]:
    print(f"{lab:10s}" + "".join(f"{f([r[c] for r in tr]):10.3f}" for c in cols))
print("per subject (nearest | mean | best single train emb | worst):")
for z in sorted(set(s)):
    v = [r["per_subject"][z] for r in tr]
    print(f"  {z}: {res['nearest']['per_subject'][z]:.3f} | {res['mean']['per_subject'][z]:.3f} | {max(v):.3f} (train{int(np.argmax(v))}) | {min(v):.3f}")
