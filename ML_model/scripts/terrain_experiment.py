"""Multi-terrain architecture comparison: plain TCN vs subject embedding (concatenated
or FiLM) vs sensor-context FiLM vs mixture of experts.

Trains every variant on the same treadmill + level-ground + ramp data and split, then
scores each on the held-out subjects per locomotion mode (treadmill, walk, turn, ramp
ascent/descent, transitions, stand). Run after the multi-terrain dataset exists:

    python scripts/csv_to_parquet.py --src <csv_output> --out <hf_dataset_terrain> \\
        --terrains treadmill levelground ramp
    python scripts/terrain_experiment.py --dataset <hf_dataset_terrain> \\
        --processed <processed_terrain> --epochs 5
    python scripts/terrain_experiment.py ... --variants moe film --eval-only <out_dir>
    python scripts/terrain_experiment.py ... --terrains treadmill,levelground,ramp,stair --name terrain_stairs

Results land in ``runs/terrain/<stamp>/summary.{json,md}`` plus ``per_mode_r2.png``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transform_ablation import new_run_dir

from exo.config import Config, replace
from exo.data import ingest
from exo.data.dataset import WindowDataset
from exo.data.demographics import subject_mass
from exo.data.scalers import ScalerBundle
from exo.models import TCN
from exo.training.subject_embedding import SubjectIndex
from exo.training.trainer import Trainer

VARIANTS = {
    "tcn":     {"model.arch": "tcn", "model.use_subject_embedding": False},
    "tcn_emb": {"model.arch": "tcn", "model.use_subject_embedding": True},
    "film":    {"model.arch": "film", "model.film_source": "subject", "model.use_subject_embedding": True},
    "film_ctx": {"model.arch": "film", "model.film_source": "context", "model.use_subject_embedding": False},
    "moe":     {"model.arch": "moe", "model.use_subject_embedding": False},
}
# deployed tx_all split: AB21/AB23 moved from test to val
VAL = ["AB09", "AB21", "AB23"]
TEST = ["AB10", "AB24", "AB25", "AB27", "AB28", "AB30"]
GROUPS = ["treadmill", "walk", "turn", "rampascent", "rampdescent", "stairascent", "stairdescent",
          "transition", "stand"]
# ankle goniometer beyond this is a sensor fault (AB30 stair_3_l reads up to 290 deg)
GON_LIMIT_DEG = 60.0


def mode_group(label: str) -> str:
    if label == "treadmill":
        return label
    if "-" in label:
        return "transition"
    if label.startswith("turn"):
        return "turn"
    return label if label in GROUPS else "other"


def base_config(dataset: Path, processed: Path, epochs: int,
                terrains: str = "treadmill,levelground,ramp") -> Config:
    cfg = Config.load(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    split = cfg.data.split
    return replace(cfg, **{
        "paths.dataset_dir": str(dataset),
        "paths.processed_dir": str(processed),
        "data.feature_set": "mid",
        "data.ingest.terrain": terrains,
        "data.valid_tail": cfg.train.loss_last_n,
        "data.split.train": [s for s in split.train if s not in VAL + TEST],
        "data.split.val": VAL,
        "data.split.test": TEST,
        "train.epochs": epochs,
    })


@torch.inference_mode()
def evaluate(cfg: Config, run_dir: Path, device: torch.device, eval_stride: int) -> dict:
    """Last-timestep prediction of every test window (as deployed), scored per mode."""
    scalers = ScalerBundle.load(run_dir)
    meta = json.loads((run_dir / "model_meta.json").read_text())
    idx = SubjectIndex(mapping=dict(meta["subject_index"]),
                       num_training_subjects=meta["num_training_subjects"])
    model = TCN.from_config(cfg.model, cfg.data.num_features, 1, idx.num_training_subjects)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.to(device).eval()

    data_cfg = replace(cfg, **{"data.stride": eval_stride}).data
    ds = WindowDataset(data_cfg, cfg.paths.processed(), scalers, split="test")
    modes, faulty = {}, set()
    for name in ds.index.trial_names:
        with np.load(cfg.paths.processed() / f"{name}.npz") as d:
            modes[name] = d["mode"] if "mode" in d.files else None
            cols = d["feature_columns"].tolist()
            if "gon_ankle_sagittal" in cols and \
                    np.nanmax(np.abs(d["features"][:, cols.index("gon_ankle_sagittal")])) > GON_LIMIT_DEG:
                faulty.add(name)
    mass = subject_mass(cfg.paths.demographics())
    y_mean, y_scale = float(scalers.scaler_y.mean_[0]), float(scalers.scaler_y.scale_[0])

    embeddings = {"nearest": None}
    if cfg.model.use_subject_embedding:
        embeddings["mean"] = model.subject_embedding.weight.mean(0, keepdim=True)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    out = {}
    for emb_name, emb_row in embeddings.items():
        if emb_row is not None:
            saved = model.subject_embedding.weight.clone()
            model.subject_embedding.weight.copy_(emb_row.expand_as(saved))
        preds, truth, groups, gates, keep = [], [], [], [], []
        k = 0
        for x, y, names, subjects in loader:
            sidx = torch.tensor(idx.indices(list(subjects)), device=device)
            p = model(x.to(device), sidx)[:, 0, -1].float().cpu().numpy()
            m = np.array([mass[s] for s in subjects])
            preds.append((p * y_scale + y_mean) / m)
            truth.append((y[:, 0, -1].numpy() * y_scale + y_mean) / m)
            ends = ds.index.entries[k:k + len(names), 2]
            keep += [n not in faulty for n in names]
            groups += [mode_group(modes[n][e]) if modes[n] is not None else "treadmill"
                       for n, e in zip(names, ends)]
            if cfg.model.arch == "moe":
                gates.append(model.gate_weights(x.to(device), sidx)[:, :, -1].cpu().numpy())
            k += len(names)
        if emb_row is not None:
            model.subject_embedding.weight.copy_(saved)
        keep = np.array(keep)
        p, t, g = np.concatenate(preds)[keep], np.concatenate(truth)[keep], np.array(groups)[keep]
        res = {"overall": _score(p, t)}
        for grp in GROUPS:
            if (g == grp).sum() > 100:
                res[grp] = _score(p[g == grp], t[g == grp])
        overground = g != "treadmill"
        res["overground_all"] = _score(p[overground], t[overground])
        if gates:
            gw = np.concatenate(gates)[keep]
            res["gate_by_mode"] = {grp: np.round(gw[g == grp].mean(0), 3).tolist()
                                   for grp in GROUPS if (g == grp).sum() > 100}
        out[emb_name] = res
    out["excluded_trials"] = sorted(faulty)
    out["latency_ms"] = _latency(model, cfg.data.num_features, cfg.data.window_length)
    out["params"] = model.num_parameters
    return out


def _score(p: np.ndarray, t: np.ndarray) -> dict:
    err = p - t
    return {"n": int(len(t)), "r2": float(1 - (err ** 2).sum() / ((t - t.mean()) ** 2).sum()),
            "rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean())}


def _latency(model: TCN, channels: int, window: int, n: int = 200) -> float:
    """Single-window CPU latency (ms, median): the deployment call."""
    m = model.to("cpu").eval()
    x, s = torch.randn(1, channels, window), torch.zeros(1, dtype=torch.long)
    times = []
    with torch.inference_mode():
        for i in range(n + 20):
            t0 = time.perf_counter()
            m(x, s)
            if i >= 20:
                times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def write_summary(results: dict, out_dir: Path) -> None:
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    cols = ["overall", "overground_all", *GROUPS]
    lines = ["| model | " + " | ".join(cols) + " | params | CPU ms |",
             "|---" * (len(cols) + 3) + "|"]
    rows = []
    for name, r in results.items():
        for emb, res in r.items():
            if emb in ("latency_ms", "params", "excluded_trials"):
                continue
            label = name if emb == "nearest" else f"{name} ({emb} embedding)"
            rows.append((label, res, r))
    for label, res, r in rows:
        cells = [f"{res[c]['r2']:.3f}" if c in res else "—" for c in cols]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {r['params']:,} | {r['latency_ms']:.2f} |")
    lines += ["", "R² per mode on held-out subjects (last-timestep prediction, as deployed). "
              "Turns are scored only on steps that land on force plates; R² is not meaningful "
              "for stand (torque barely varies), use RMSE.", "",
              "| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for label, res, _ in rows:
        cells = [f"{res[c]['rmse']:.3f}" if c in res else "—" for c in cols]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "RMSE (N·m/kg) per mode on held-out subjects."]
    excluded = sorted({t for r in results.values() for t in r.get("excluded_trials", [])})
    if excluded:
        lines += ["", f"Excluded test trials (ankle goniometer beyond ±{GON_LIMIT_DEG:.0f}°, sensor fault): "
                  + ", ".join(excluded)]
    for name, r in results.items():
        if "gate_by_mode" in r["nearest"]:
            lines += ["", f"MoE gate weights by mode ({name}):", "",
                      "| mode | " + " | ".join(f"expert {i}" for i in range(len(next(iter(r['nearest']['gate_by_mode'].values()))))) + " |",
                      "|---" * (1 + len(next(iter(r['nearest']['gate_by_mode'].values())))) + "|"]
            lines += [f"| {m} | " + " | ".join(f"{v:.2f}" for v in w) + " |"
                      for m, w in r["nearest"]["gate_by_mode"].items()]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    groups = [g for g in ["overall", *GROUPS] if all(g in res for _, res, _ in rows)]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.8 / len(rows)
    for i, (label, res, _) in enumerate(rows):
        ax.bar(np.arange(len(groups)) + i * width, [res[g]["r2"] for g in groups], width * 0.92, label=label)
    ax.set_xticks(np.arange(len(groups)) + 0.4 - width / 2, groups)
    ax.set(ylabel="R² (held-out subjects)", ylim=(0, 1), title="Ankle torque prediction by locomotion mode")
    ax.legend(fontsize=9, ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "per_mode_r2.png", dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--processed", required=True, type=Path)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--eval-stride", type=int, default=5)
    ap.add_argument("--eval-only", type=Path, default=None, help="score runs already in this dir")
    ap.add_argument("--terrains", default="treadmill,levelground,ramp",
                    help="comma-separated terrains (add stair for the stairs run)")
    ap.add_argument("--name", default="terrain", help="results go to runs/<name>/<stamp>")
    a = ap.parse_args()

    base = base_config(a.dataset, a.processed, a.epochs, a.terrains)
    if not ingest.is_cached(base.paths.processed()):
        ingest.run(base, base.paths.dataset(), base.paths.processed(), "terrain-v1")
    out_dir = a.eval_only or new_run_dir(base.paths.runs() / a.name)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    results = json.loads((out_dir / "summary.json").read_text()) if (out_dir / "summary.json").exists() else {}
    for name in a.variants:
        cfg = replace(base, **{**VARIANTS[name], "paths.runs_dir": str(out_dir),
                               "train.wandb_run_name": f"terrain_{name}"})
        run_dir = out_dir / name
        if not a.eval_only:
            t0 = time.time()
            Trainer(cfg, run_dir, use_wandb=False).fit()
            print(f"[{name}] trained in {(time.time() - t0) / 60:.1f} min", flush=True)
        results[name] = evaluate(cfg, run_dir, device, a.eval_stride)
        write_summary(results, out_dir)


if __name__ == "__main__":
    main()
