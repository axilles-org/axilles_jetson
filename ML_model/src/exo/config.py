"""Typed configuration tree — the single source of truth.

One YAML file (``configs/default.yaml``) is loaded into a nested tree of frozen
dataclasses. The resolved config is saved next to every training run so that
evaluation and export can never drift from the architecture that was trained.

Usage::

    from exo.config import Config
    cfg = Config.load("configs/default.yaml")
    cfg.model.num_channels          # -> [80, 80, 80, 80, 80]
    cfg.save(run_dir)               # writes run_dir/config.yaml
    cfg = Config.load(run_dir / "config.yaml")
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import resolve


# Leaf configs
@dataclass(frozen=True)
class HubConfig:
    """Hugging Face dataset repo holding the canonical Parquet data."""
    repo_id: str = ""
    revision: str = "main"


@dataclass(frozen=True)
class PathsConfig:
    dataset_dir: str             # local Parquet dataset root (HF snapshot target)
    processed_dir: str           # where ingest writes per-trial .npz
    runs_dir: str = "runs"       # training run outputs
    hub: HubConfig = field(default_factory=HubConfig)

    def dataset(self) -> Path: return resolve(self.dataset_dir)
    def processed(self) -> Path: return resolve(self.processed_dir)
    def runs(self) -> Path: return resolve(self.runs_dir)

    def demographics(self) -> Path:
        """Demographics live in the dataset's metadata.parquet."""
        return self.dataset() / "metadata.parquet"


@dataclass(frozen=True)
class SubjectSplit:
    train: list[str]
    val: list[str]
    test: list[str]

    def all_subjects(self) -> list[str]:
        return [*self.train, *self.val, *self.test]

    def validate(self) -> None:
        seen: set[str] = set()
        for name, group in (("train", self.train), ("val", self.val), ("test", self.test)):
            for s in group:
                if s in seen:
                    raise ValueError(f"subject {s!r} appears in more than one split")
                seen.add(s)
        if not self.train:
            raise ValueError("train split is empty")


@dataclass(frozen=True)
class IngestConfig:
    """Raw-CSV -> aligned, decimated, z-scored per-trial arrays."""
    terrain: str = "treadmill"
    target_rate_hz: int = 100
    # native sensor rates in the GaTech dataset (verified from Header timestamps)
    native_rates_hz: dict[str, int] = field(default_factory=lambda: {
        "imu": 200, "id": 200, "fp": 1000, "gon": 1000,
    })
    target_column: str = "ankle_angle_r_moment"      # from the 'id' (inverse dynamics) sensor
    antialias: bool = True                            # scipy.signal.decimate on downsampling
    gon_to_radians: bool = True
    drop_imu_segments: tuple[str, ...] = ("trunk",)   # IMU mount points to drop
    stance_source: str = "gcRight"                    # gcRight | force_threshold
    force_threshold_n: float = 50.0                   # used when stance_source == force_threshold
    output_format: str = "npy"                        # npy | parquet


@dataclass(frozen=True)
class DataConfig:
    feature_set: str                     # key into feature_sets
    feature_sets: dict[str, list[str]]
    split: SubjectSplit
    window_length: int = 300             # samples fed to the model (== 3.0 s at 100 Hz)
    stride: int = 10                     # sliding-window hop
    min_segment_length: int = 300        # shortest contiguous valid region to window
    ingest: IngestConfig = field(default_factory=IngestConfig)
    num_workers: int = 4
    prefetch_factor: int = 4
    cache_window_index: bool = True

    def feature_names(self) -> list[str]:
        if self.feature_set not in self.feature_sets:
            raise ValueError(
                f"feature_set={self.feature_set!r} not in feature_sets "
                f"{list(self.feature_sets)}"
            )
        return list(self.feature_sets[self.feature_set])

    @property
    def num_features(self) -> int:
        return len(self.feature_names())


@dataclass(frozen=True)
class ModelConfig:
    num_channels: list[int] = field(default_factory=lambda: [80, 80, 80, 80, 80])
    kernel_size: int = 5
    activation: str = "ReLU"             # ELU | GELU | ReLU | Swish
    norm_type: str = "WeightNorm"        # BatchNorm | LayerNorm | WeightNorm
    dropout_type: str = "Spatial"        # Spatial | ElementWise
    dropout: float = 0.15
    l1_reg: float = 0.0
    l2_reg: float = 0.0
    use_subject_embedding: bool = True
    emb_dim: int = 4

    def receptive_field(self) -> int:
        rf = 1
        for i in range(len(self.num_channels)):
            rf += 2 * (self.kernel_size - 1) * (2 ** i)
        return rf


@dataclass(frozen=True)
class AugmentConfig:
    enabled: bool = True
    noise_std: float = 0.01              # additive Gaussian on z-scored inputs
    imu_gain_jitter: float = 0.05        # +/- fractional per-trial gain on IMU channels
    time_warp: float = 0.0              # max fractional time-warp (0 = off)
    channel_dropout: float = 0.0        # prob. of zeroing a whole input channel

    # Deployment domain randomisation (see docs/DOMAIN_RANDOMIZATION.md).
    latency_samples: int = 0           # predict up to N samples ahead      (try 5)
    imu_rotation_deg: float = 0.0      # random rotation per IMU triad       (try 8)
    encoder_offset_rad: float = 0.0    # random ankle-angle bias            (try 0.05)
    stance_jitter_samples: int = 0     # shift stance 0/1 edges +/- k        (try 3)
    assist_perturb: float = 0.0        # simulate kinematics under assist   (try 0.3)


@dataclass(frozen=True)
class PerfConfig:
    amp: str = "auto"                    # auto | bf16 | fp16 | off
    compile: bool = False               # torch.compile (CUDA only; auto-skips elsewhere)
    compile_mode: str = "max-autotune"
    tf32: bool = True                    # CUDA TF32 matmul/cudnn
    cudnn_benchmark: bool = True
    fused_optimizer: bool = True         # fused AdamW on CUDA
    grad_accum_steps: int = 1


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    epochs: int = 30
    lr: float = 5e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    loss_last_n: int = 50               # compute loss on the last N window timesteps
                                        # (matches deployment; 0 = whole window)
    scheduler: str = "cosine"           # cosine | none
    early_stop_patience: int = 8        # epochs without val improvement (0 = off)
    seed: int = 0
    max_steps_per_epoch: int = 0        # 0 = full epoch; >0 caps steps (smoke tests)
    log_every: int = 50
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)
    wandb_project: str = "Exoskeleton_MRSD"
    wandb_run_name: str = "tcn"
    device: str = "auto"                # auto | cuda | mps | cpu


@dataclass(frozen=True)
class SensorAdapterConfig:
    """Exo sensor frame -> GaTech convention. Identity defaults = no adaptation.

    Filled in by scripts/fit_adapter.py. See docs/SENSOR_ADAPTER.md.
    """
    foot_rotation: list[list[float]] = field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    shank_rotation: list[list[float]] = field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    foot_accel_scale: float = 1.0      # exo foot accel units -> g
    foot_gyro_scale: float = 1.0       # exo foot gyro units -> rad/s
    shank_accel_scale: float = 1.0
    shank_gyro_scale: float = 1.0
    foot_lever_arm_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    shank_lever_arm_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ankle_encoder_neutral_deg: float = 0.0   # encoder reading at the GaTech zero pose
    ankle_encoder_sign: float = 1.0
    heel_fsr_threshold: float = 2000.0  # raw ADC counts -> stance
    toe_fsr_threshold: float = 2000.0
    fsr_debounce_s: float = 0.10       # min stance/swing dwell time


@dataclass(frozen=True)
class DeployConfig:
    assistance_scale: float = 0.2       # fraction of predicted human moment
    torque_limit_nm: float = 30.0       # hard saturation on the exo command magnitude
    rate_limit_nm_per_s: float = 200.0  # slew-rate limit
    ramp_in_s: float = 0.05
    ramp_out_s: float = 0.05
    control_rate_hz: int = 100
    # In the GaTech convention the plantarflexion (push-off) moment is negative;
    # the exo assists that direction. This sign maps the biological moment to the
    # exo actuator's positive-torque convention.
    plantarflexion_sign: float = -1.0
    exo_command_sign: float = 1.0
    backend: str = "jit"               # jit | onnx | trt
    device: str = "cpu"
    sensor_adapter: SensorAdapterConfig = field(default_factory=SensorAdapterConfig)


# Root
@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)

    # -- (de)serialization --------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> Config:
        with open(resolve(path)) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        pd_paths = dict(d["paths"])
        if "hub" in pd_paths:
            pd_paths["hub"] = HubConfig(**pd_paths["hub"])
        paths = PathsConfig(**pd_paths)

        dd = dict(d["data"])
        dd["split"] = SubjectSplit(**dd["split"])
        if "ingest" in dd:
            dd["ingest"] = IngestConfig(**dd["ingest"])
        data = DataConfig(**dd)

        model = ModelConfig(**d.get("model", {}))

        td = dict(d.get("train", {}))
        if "augment" in td:
            td["augment"] = AugmentConfig(**td["augment"])
        if "perf" in td:
            td["perf"] = PerfConfig(**td["perf"])
        train = TrainConfig(**td)

        pd_ = dict(d.get("deploy", {}))
        if "sensor_adapter" in pd_:
            pd_["sensor_adapter"] = SensorAdapterConfig(**pd_["sensor_adapter"])
        deploy = DeployConfig(**pd_)

        cfg = cls(paths=paths, data=data, model=model, train=train, deploy=deploy)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, run_dir: str | Path, name: str = "config.yaml") -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        out = run_dir / name
        with open(out, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        return out

    # -- validation -------------------------------------------------------
    def validate(self) -> None:
        self.data.split.validate()
        _ = self.data.feature_names()  # raises if feature_set unknown
        if self.data.window_length < self.model.receptive_field():
            # not fatal, but the early part of every window will be under-contextualised
            import warnings
            warnings.warn(
                f"window_length={self.data.window_length} < model receptive field "
                f"{self.model.receptive_field()}; consider a longer window or a "
                f"shallower stack.",
                stacklevel=2,
            )
        if self.train.loss_last_n < 0:
            raise ValueError("train.loss_last_n must be >= 0")
        if self.train.loss_last_n > self.data.window_length:
            raise ValueError("train.loss_last_n cannot exceed window_length")
        if self.data.ingest.stance_source not in ("gcRight", "force_threshold"):
            raise ValueError(
                f"ingest.stance_source must be 'gcRight' or 'force_threshold', "
                f"got {self.data.ingest.stance_source!r}"
            )


def replace(cfg: Config, **overrides: Any) -> Config:
    """Shallow-override top-level or dotted fields, e.g.
    ``replace(cfg, **{"train.epochs": 1, "train.perf.compile": False})``.
    """
    d = cfg.to_dict()
    for key, val in overrides.items():
        node = d
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = val
    return Config.from_dict(d)
