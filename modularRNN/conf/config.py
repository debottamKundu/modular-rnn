from dataclasses import dataclass, field
from typing import List
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ModelConfig:
    n_visp: int = 64
    n_mos: int = 64
    n_stim: int = 2
    n_fb: int = 1  # 1 for reward only, 2 for reward+kernel
    rank_ff: int = 5
    dt: float = 10.0
    tau_visp_range: List[float] = field(default_factory=lambda: [20.0, 60.0])
    tau_mos_range: List[float] = field(default_factory=lambda: [100.0, 400.0])
    learnable_tau: bool = False
    rec_gain: float = 1.2
    ff_gain: float = 0.4
    fb_gain: float = 0.4
    in_gain: float = 1.0
    sigma_baseline: float = 0.05
    nonlinearity: str = "relu"  # "relu" | "tanh" | "softplus"


@dataclass
class DecisionConfig:
    t_min: int = 10
    theta: float = 0.9


@dataclass
class EnvConfig:
    batch_size: int = 32
    T_max: int = 100
    T_iti: int = 3
    sigma_stim: float = 1.0  # stimulus noise amplitude
    hold_fb_during_stim: bool = False
    n_blocks: int = 5  # total blocks/session
    n_unbiased_blocks: int = 1
    bias_high: float = 0.8
    block_len_mean: int = 60
    block_len_min: int = 20
    block_len_max: int = 100
    contrasts: List[float] = field(
        default_factory=lambda: [0.0, 0.0625, 0.125, 0.25, 1.0]
    )
    t_quiescence: int = 0
    eval_batch_size: int = 64
    kappa_kernel: float = 0.5
    xi_kernel: float = 0.2


@dataclass
class LossConfig:
    lambda_aux: float = 1.0
    lambda_rate: float = 1e-3
    lambda_weight: float = 1e-4
    lambda_switch: float = 0.3  # switching-cost weight


@dataclass
class TrainingConfig:
    n_steps: int = 20000
    chunk_dt_start: int = 40
    chunk_dt_growth_every: int = 500
    chunk_dt_max: int = 400
    grad_clip: float = 1.0
    lr: float = 1e-3
    teacher_forcing_start: float = 0.0
    teacher_forcing_end: float = 0.0
    teacher_forcing_anneal_frac: float = 0.2
    eval_every: int = 200
    ckpt_every: int = 1000
    use_wandb: bool = True
    store_hidden_during_training: bool = False
    seed: int = 42
    device: str = "cpu"  # "cpu" | "mps" | "cuda"
    compile_model: bool = False
    eval_n_trials: int = 20
    probe_every: int = 1000  # expensive decoding probe cadence
    probe_cv_folds: int = 3


@dataclass
class PathsConfig:
    checkpoint_dir: str = "???"
    data_dir: str = "???"
    wandb_dir: str = "???"


@dataclass
class LocalPaths(PathsConfig):
    checkpoint_dir: str = "./model/checkpoints"
    data_dir: str = "./data"
    wandb_dir: str = "./wandb"


@dataclass
class ServerPaths(PathsConfig):
    checkpoint_dir: str = "/usr/people/kundu/code/modular_rnn/model/checkpoints"
    data_dir: str = "/usr/people/kundu/code/modular_rnn/data"
    wandb_dir: str = "/usr/people/kundu/code/modular_rnn/wandb"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # paths: PathsConfig = field(default_factory=LocalPaths)
    paths: PathsConfig = MISSING


cs = ConfigStore.instance()
# cs.store(name="config", node=Config)
cs.store(name="base_config", node=Config)
cs.store(group="paths", name="local", node=LocalPaths)
cs.store(group="paths", name="server", node=ServerPaths)
