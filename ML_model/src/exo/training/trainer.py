"""Training loop for the torque-prediction model."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import WindowDataset
from ..data.demographics import subject_mass
from ..data.scalers import ScalerBundle
from ..models import TCN
from .metrics import batch_metrics
from .runtime_utils import (
    apply_backend_flags,
    autocast_context,
    maybe_compile,
    resolve_amp_dtype,
    select_device,
)
from .subject_embedding import SubjectIndex

try:
    import wandb
except ImportError:  # wandb is optional
    wandb = None


@dataclass
class EpochResult:
    loss: float
    rmse_nm_per_kg: float
    mae_nm_per_kg: float
    normalized_mae: float


class Trainer:
    def __init__(self, cfg: Config, run_dir: str | Path, use_wandb: bool = True):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb and wandb is not None

        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)

        self.device = select_device(cfg.train.device)
        apply_backend_flags(cfg.train.perf, self.device)
        self.amp_dtype = resolve_amp_dtype(cfg.train.perf.amp, self.device)
        self.grad_scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.amp_dtype == torch.float16
        )

        self.scalers = ScalerBundle.load(cfg.paths.processed())
        self.subject_mass = subject_mass(cfg.paths.demographics())
        self.subject_index = SubjectIndex.build(
            cfg.data.split.train,
            str(cfg.paths.demographics()),
            self.scalers.demographics,
            unseen_subjects=cfg.data.split.val + cfg.data.split.test,
        )

        self.loaders = self._build_loaders()
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        print(f"[trainer] device={self.device} amp={self.amp_dtype} "
              f"params={self.model.num_parameters:,} "
              f"train_windows={len(self.loaders['train'].dataset)}")

    # -- setup ------------------------------------------------------
    def _build_loaders(self) -> dict[str, DataLoader]:
        d = self.cfg.data
        common = dict(
            batch_size=self.cfg.train.batch_size,
            num_workers=d.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=d.num_workers > 0,
        )
        if d.num_workers > 0:
            common["prefetch_factor"] = d.prefetch_factor

        datasets = {
            "train": WindowDataset(d, self.cfg.paths.processed(), self.scalers,
                                   split="train", augment_cfg=self.cfg.train.augment, verbose=True),
            "val": WindowDataset(d, self.cfg.paths.processed(), self.scalers,
                                 split="val", verbose=True),
            "test": WindowDataset(d, self.cfg.paths.processed(), self.scalers,
                                  split="test", verbose=True),
        }
        return {
            "train": DataLoader(datasets["train"], shuffle=True, drop_last=True, **common),
            "val": DataLoader(datasets["val"], shuffle=False, **common),
            "test": DataLoader(datasets["test"], shuffle=False, **common),
        }

    def _build_model(self) -> TCN:
        model = TCN.from_config(
            self.cfg.model,
            in_channels=self.cfg.data.num_features,
            out_channels=1,
            num_subjects=self.subject_index.num_training_subjects,
        ).to(self.device)
        return maybe_compile(model, self.cfg.train.perf, self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        fused = self.cfg.train.perf.fused_optimizer and self.device.type == "cuda"
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            fused=fused,
        )

    def _build_scheduler(self):
        if self.cfg.train.scheduler != "cosine":
            return None
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.train.epochs, eta_min=self.cfg.train.lr / 20
        )

    # -- training ------------------------------------------------
    def fit(self) -> Path:
        self.cfg.save(self.run_dir)
        self.scalers.save(self.run_dir)
        _write_model_meta(self.run_dir, self.cfg, self.subject_index)

        if self.use_wandb:
            wandb.init(project=self.cfg.train.wandb_project,
                       name=self.cfg.train.wandb_run_name,
                       dir=str(self.run_dir), config=self.cfg.to_dict())

        best_val = float("inf")
        best_path = self.run_dir / "best.pt"
        epochs_without_gain = 0

        for epoch in range(1, self.cfg.train.epochs + 1):
            t0 = time.time()
            train_res = self._run_epoch(self.loaders["train"], train=True, epoch=epoch)
            val_res = self._run_epoch(self.loaders["val"], train=False, epoch=epoch)
            if self.scheduler is not None:
                self.scheduler.step()

            self._log(epoch, train_res, val_res, time.time() - t0)

            if val_res.loss < best_val:
                best_val = val_res.loss
                torch.save(_state_dict(self.model), best_path)
                epochs_without_gain = 0
            else:
                epochs_without_gain += 1
                if (self.cfg.train.early_stop_patience
                        and epochs_without_gain >= self.cfg.train.early_stop_patience):
                    print(f"[trainer] early stop at epoch {epoch}")
                    break

        self._final_test(best_path)
        if self.use_wandb:
            wandb.finish()
        return best_path

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int) -> EpochResult:
        self.model.train(train)
        last_n = self.cfg.train.loss_last_n
        accum = self.cfg.train.perf.grad_accum_steps

        total_loss, seen = 0.0, 0
        rmse, mae, nmae = [], [], []
        torch.set_grad_enabled(train)
        max_steps = self.cfg.train.max_steps_per_epoch
        n_steps = len(loader) if not max_steps else min(max_steps, len(loader))
        t_epoch = time.time()

        for step, (x, y, names, subject_ids) in enumerate(loader):
            if max_steps and step >= max_steps:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            sidx = torch.tensor(self.subject_index.indices(list(subject_ids)),
                                dtype=torch.long, device=self.device)

            with autocast_context(self.amp_dtype, self.device):
                pred = self.model(x, sidx)
                loss = self._loss(pred, y, last_n)

            if train:
                self.grad_scaler.scale(loss / accum).backward()
                if (step + 1) % accum == 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.cfg.train.grad_clip)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

            bs = x.size(0)
            total_loss += loss.item() * bs
            seen += bs

            m = batch_metrics(pred.detach().float(), y.float(), list(names),
                              self.scalers.scaler_y, self.subject_mass, last_n=last_n)
            rmse.append(m["rmse_nm_per_kg"])
            mae.append(m["mae_nm_per_kg"])
            nmae.append(m["normalized_mae"])

            if train and self.cfg.train.log_every and (step + 1) % self.cfg.train.log_every == 0:
                rate = (step + 1) * x.size(0) / (time.time() - t_epoch)
                print(f"  epoch {epoch} [{step + 1}/{n_steps}] loss={loss.item():.4f} "
                      f"rmse={m['rmse_nm_per_kg']:.4f} Nm/kg  {rate:.0f} samp/s")

        torch.set_grad_enabled(True)
        return EpochResult(total_loss / max(seen, 1),
                           float(np.mean(rmse)), float(np.mean(mae)), float(np.mean(nmae)))

    def _loss(self, pred: torch.Tensor, target: torch.Tensor, last_n: int) -> torch.Tensor:
        if last_n > 0:
            pred = pred[..., -last_n:]
            target = target[..., -last_n:]
        base = F.mse_loss(pred, target)
        reg = self.model.regularization_loss() if hasattr(self.model, "regularization_loss") else 0.0
        return base + reg

    # -- evaluation / logging -------------------------------------
    def _final_test(self, best_path: Path) -> None:
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        res = self._run_epoch(self.loaders["test"], train=False, epoch=self.cfg.train.epochs)
        print(f"[test] loss={res.loss:.5f} rmse={res.rmse_nm_per_kg:.4f} "
              f"mae={res.mae_nm_per_kg:.4f} nmae={res.normalized_mae:.4f} (Nm/kg)")
        if self.use_wandb:
            wandb.log({"test/loss": res.loss, "test/rmse_nm_per_kg": res.rmse_nm_per_kg,
                       "test/mae_nm_per_kg": res.mae_nm_per_kg,
                       "test/normalized_mae": res.normalized_mae})
        with open(self.run_dir / "test_metrics.json", "w") as f:
            json.dump(res.__dict__, f, indent=2)

    def _log(self, epoch: int, train: EpochResult, val: EpochResult, secs: float) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}/{self.cfg.train.epochs}  "
              f"train_loss={train.loss:.5f}  val_loss={val.loss:.5f}  "
              f"val_rmse={val.rmse_nm_per_kg:.4f}  val_mae={val.mae_nm_per_kg:.4f} Nm/kg  "
              f"lr={lr:.2e}  {secs:.1f}s")
        if self.use_wandb:
            wandb.log({
                "epoch": epoch, "lr": lr, "epoch_time_s": secs,
                "train/loss": train.loss, "train/rmse_nm_per_kg": train.rmse_nm_per_kg,
                "val/loss": val.loss, "val/rmse_nm_per_kg": val.rmse_nm_per_kg,
                "val/mae_nm_per_kg": val.mae_nm_per_kg, "val/normalized_mae": val.normalized_mae,
            })


def _state_dict(model: torch.nn.Module) -> dict:
    return getattr(model, "_orig_mod", model).state_dict()


def _write_model_meta(run_dir: Path, cfg: Config, idx: SubjectIndex) -> None:
    meta = {
        "model": cfg.model.__dict__ if hasattr(cfg.model, "__dict__") else cfg.to_dict()["model"],
        "feature_names": cfg.data.feature_names(),
        "num_features": cfg.data.num_features,
        "window_length": cfg.data.window_length,
        "num_training_subjects": idx.num_training_subjects,
        "subject_index": idx.mapping,
    }
    with open(run_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
