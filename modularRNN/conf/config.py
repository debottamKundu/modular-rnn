from dataclasses import dataclass, field
from typing import List, Optional
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ConnectionConfig:
    mode: str = "dense"  # "dense" | "sparse" | "zero" | "lowrank"
    gain: float = 0.5
    sparsity: float = 1.0  # used only if mode == "sparse"
    rank: int = 3  # u


@dataclass
class ModelConfig:
    n_v: int = 64
    n_c: int = 64
    n_p: int = 32
    n_stim: int = 2
    n_fb: int = 2
    dt: float = 10.0
    tau_v_range: List[float] = field(default_factory=lambda: [20.0, 60.0])
    tau_c_range: List[float] = field(default_factory=lambda: [100.0, 250.0])
    tau_p_range: List[float] = field(default_factory=lambda: [250.0, 600.0])
    learnable_tau: bool = False
    rec_gain: float = 1.2
    in_gain: float = 1.0
    sigma_baseline: float = 0.05
    nonlinearity: str = "relu"

    conn_c_from_v: ConnectionConfig = field(
        default_factory=lambda: ConnectionConfig(mode="lowrank", gain=0.4, rank=3)
    )
    conn_v_from_c: ConnectionConfig = field(
        default_factory=lambda: ConnectionConfig(mode="dense", gain=0.4)
    )
    conn_p_from_v: ConnectionConfig = field(
        default_factory=lambda: ConnectionConfig(mode="zero")
    )
    conn_v_from_p: ConnectionConfig = field(
        default_factory=lambda: ConnectionConfig(mode="dense", gain=0.4)
    )
    conn_p_from_c: ConnectionConfig = field(
        default_factory=lambda: ConnectionConfig(mode="dense", gain=0.4)
    )
    conn_c_from_p: ConnectionConfig = field(
        default_factory=lambda: ConnectionConfig(mode="dense", gain=0.4)
    )
    notes: List[str] = field(default_factory=lambda: ["notes for tracking"])


@dataclass
class DecisionConfig:
    t_min: int = 10
    theta: float = 0.9


@dataclass
class EnvConfig:
    batch_size: int = 32
    T_max: int = 100
    T_iti: int = 10
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
    reward_timing: str = (
        "pulse_only"  # "pulse_only" | "persist_iti" | "persist_iti_and_decision"
    )
    kernel_timing: str = "pulse_only"  # same three options, independently


@dataclass
class LossConfig:
    lambda_aux: float = 1.0
    lambda_rate: float = 1e-3
    lambda_weight: float = 1e-4
    lambda_switch: float = 0.4  # switching-cost weight


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
    # eval_n_trials: int = 20
    probe_every: int = 1000
    min_steps_before_stop: int = 10000
    plateau_patience: int = 10
    plateau_min_delta: float = 0.002
    plateau_smoothing_window: int = 10
    enable_early_stopping: bool = True
    resume_from: Optional[str] = None
    run_id: Optional[str] = None
    eval_every: int = 1000
    eval_n_dt_steps: int = 5000
    eval_batch_size: int = 128
    eval_plateau_patience: int = 2
    eval_plateau_min_delta: float = 0.01
    eval_plateau_smoothing_window: int = 5


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
class PosthocConfig:
    ckpt_path: str = MISSING
    mode: str = "single_session"  # "batched" | "single_session"
    n_dt_steps: int = 50000  # for batched mode
    batch_size: int = 256  # for batched mode
    max_trials: int = 600  # for single_session mode
    total_sessions: int = 2
    noise_scale: float = 1.0  # 1.0 = match training statistics; 0.0 = deterministic


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # paths: PathsConfig = field(default_factory=LocalPaths)
    paths: PathsConfig = MISSING
    posthoc: PosthocConfig = field(default_factory=PosthocConfig)


cs = ConfigStore.instance()
# cs.store(name="config", node=Config)
cs.store(name="base_config", node=Config)
cs.store(group="paths", name="local", node=LocalPaths)
cs.store(group="paths", name="server", node=ServerPaths)
