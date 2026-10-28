"""Ablate the GaTech -> exo-frame IMU transform, channel group by channel group.

    python scripts/transform_ablation.py --tx-root /path/to/hf_dataset_tx
    python scripts/transform_ablation.py --tx-root ... --variants original tx_all --epochs 5

Each variant trains the same model with the same split; only the IMU inputs differ
(original GaTech frame, exo frame, or dropped). Results land in
``<runs_dir>/ablation/<stamp>/summary.{json,md}``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare import prepare

from exo.config import Config, replace
from exo.data.ablation import IMU_GROUPS, Variant, build_dataset, drop_features
from exo.training.evaluate import Evaluator
from exo.training.trainer import Trainer

ACCEL = frozenset({"foot_Accel", "shank_Accel"})
GYRO = frozenset({"foot_Gyro", "shank_Gyro"})
FOOT = frozenset({"foot_Accel", "foot_Gyro"})
SHANK = frozenset({"shank_Accel", "shank_Gyro"})
ALL = frozenset(IMU_GROUPS)

VARIANTS = {v.name: v for v in (
    Variant("original"),
    Variant("tx_all", tx_groups=ALL),
    Variant("tx_accel", tx_groups=ACCEL),
    Variant("tx_gyro", tx_groups=GYRO),
    Variant("tx_foot", tx_groups=FOOT),
    Variant("tx_shank", tx_groups=SHANK),
    Variant("tx_all_drop_foot_gyro", tx_groups=ALL, drop_groups=frozenset({"foot_Gyro"})),
    Variant("tx_all_drop_shank_gyro", tx_groups=ALL, drop_groups=frozenset({"shank_Gyro"})),
    Variant("tx_all_drop_foot_accel", tx_groups=ALL, drop_groups=frozenset({"foot_Accel"})),
    Variant("tx_all_drop_shank_accel", tx_groups=ALL, drop_groups=frozenset({"shank_Accel"})),
    Variant("tx_all_drop_gyro", tx_groups=ALL, drop_groups=GYRO),
    Variant("tx_all_drop_accel", tx_groups=ALL, drop_groups=ACCEL),
    Variant("tx_all_drop_stance", tx_groups=ALL, drop_groups=frozenset({"stance"})),
    Variant("tx_all_drop_ankle", tx_groups=ALL, drop_groups=frozenset({"gon_ankle_sagittal"})),
    Variant("no_imu", tx_groups=ALL, drop_groups=ALL),
    Variant("imu_only", tx_groups=ALL, drop_groups=frozenset({"stance", "gon_ankle_sagittal"})),
    Variant("original_drop_gyro", drop_groups=GYRO),
)}


def new_run_dir(parent: Path) -> Path:
    """Timestamped directory that is never shared, even by runs started in the same second."""
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(100):
        path = parent / (stamp if i == 0 else f"{stamp}_{i}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a run directory under {parent}")


def move_to_val(cfg: Config, subjects: list[str]) -> Config:
    split = cfg.data.split
    missing = set(subjects) - set(split.test)
    if missing:
        raise SystemExit(f"not in the test split: {sorted(missing)}")
    return replace(cfg, **{
        "data.split.val": list(split.val) + subjects,
        "data.split.test": [s for s in split.test if s not in subjects],
    })


def variant_config(base: Config, variant: Variant, dataset_root: Path,
                   work_root: Path, runs_root: Path, epochs: int | None) -> Config:
    features = drop_features(base.data.feature_names(), variant.drop_groups)
    overrides = {
        "paths.dataset_dir": str(dataset_root),
        "paths.processed_dir": str(work_root / variant.data_key / "processed"),
        "paths.runs_dir": str(runs_root),
        "data.feature_sets": {**base.data.feature_sets, variant.name: features},
        "data.feature_set": variant.name,
        "train.wandb_run_name": f"ablation_{variant.name}",
    }
    if epochs is not None:
        overrides["train.epochs"] = epochs
    return replace(base, **overrides)


def run_variant(cfg: Config, variant: Variant, runs_root: Path, use_wandb: bool) -> dict:
    prepare(cfg, do_fetch=False, verbose=True)
    run_dir = runs_root / variant.name
    best = Trainer(cfg, run_dir, use_wandb=use_wandb).fit()
    report, _ = Evaluator(cfg, str(best)).run(split="test")
    return {
        "variant": variant.name,
        "tx_groups": sorted(variant.tx_groups),
        "dropped_groups": sorted(variant.drop_groups),
        "features": cfg.data.feature_names(),
        "checkpoint": str(best),
        "test_windowed": report.overall_windowed,
        "test_streaming": report.overall_streaming,
        "test_per_subject": report.per_subject,
    }


def write_summary(results: list[dict], out_dir: Path) -> None:
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    baseline = next((r for r in results if r["variant"] == "original"), None)
    lines = [
        "| variant | transformed | dropped | RMSE (N·m/kg) | MAE | R² | ΔRMSE vs original |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: r["test_streaming"]["rmse_nm_per_kg"]):
        m = r["test_streaming"]
        delta = "—"
        if baseline and r is not baseline:
            base_rmse = baseline["test_streaming"]["rmse_nm_per_kg"]
            delta = f"{100 * (m['rmse_nm_per_kg'] - base_rmse) / base_rmse:+.1f}%"
        lines.append(
            f"| {r['variant']} | {', '.join(r['tx_groups']) or '—'} | "
            f"{', '.join(r['dropped_groups']) or '—'} | {m['rmse_nm_per_kg']:.4f} | "
            f"{m['mae_nm_per_kg']:.4f} | {m['r2']:.3f} | {delta} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tx-root", required=True, help="transformed dataset from run_alignment.py")
    ap.add_argument("--work-root", default=None,
                    help="where variant datasets and caches go (default: next to the dataset)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--move-to-val", nargs="+", default=[], metavar="SUBJECT",
                    help="move these subjects from the test split to the val split")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    base = move_to_val(Config.load(args.config), args.move_to_val)
    original_root = base.paths.dataset()
    tx_root = Path(args.tx_root).resolve()
    work_root = Path(args.work_root) if args.work_root else original_root.parent / "ablation"
    runs_root = new_run_dir(base.paths.runs() / "ablation")

    results = []
    for name in args.variants:
        variant = VARIANTS[name]
        print(f"\n=== {name} ===")
        dataset_root = build_dataset(original_root, tx_root,
                                     work_root / variant.data_key / "dataset", variant.tx_groups)
        cfg = variant_config(base, variant, dataset_root, work_root, runs_root, args.epochs)
        results.append(run_variant(cfg, variant, runs_root, args.wandb))
        write_summary(results, runs_root)

    print(f"\nsummary -> {runs_root / 'summary.md'}")


if __name__ == "__main__":
    main()
