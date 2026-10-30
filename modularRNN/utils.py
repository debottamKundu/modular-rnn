import random
import numpy as np
import torch
import torch.nn.functional as F
import tempfile
import wandb
import csv
import os
from datetime import datetime

NONLINEARITIES = {"relu": F.relu, "tanh": torch.tanh, "softplus": F.softplus}


def get_nonlinearity(name: str):
    return NONLINEARITIES[name]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def spectral_radius(W: torch.Tensor) -> float:
    with torch.no_grad():
        return torch.linalg.eigvals(W).abs().max().item()


def get_run_id(use_wandb: bool) -> str:
    """Unique identifier for this run's artifacts. Uses the wandb run id
    when available; otherwise generates a disposable, clearly-labeled
    temporary id"""
    if use_wandb:
        assert wandb.run is not None
        return wandb.run.id
    else:
        names = tempfile._get_candidate_names()
    return "smoketest_" + next(names)


def validate_config_consistency(cfg):

    assert (cfg.env.kappa_kernel != 0.0) == (cfg.model.n_fb == 2), (
        f"env.kappa_kernel={cfg.env.kappa_kernel} implies action-kernel "
        f"{'enabled' if cfg.env.kappa_kernel != 0.0 else 'disabled'}, but "
        f"model.n_fb={cfg.model.n_fb} (must be 2 if enabled, 1 if disabled)."
    )

    if cfg.env.kappa_kernel == 0.0 and cfg.env.kernel_timing != "pulse_only":
        print(
            f"WARNING: env.kernel_timing={cfg.env.kernel_timing} has no effect "
            f"since env.kappa_kernel=0.0 (action kernel disabled)."
        )

    valid_timings = {"pulse_only", "persist_iti", "persist_iti_and_decision"}
    assert (
        cfg.env.reward_timing in valid_timings
    ), f"invalid env.reward_timing: {cfg.env.reward_timing}"
    assert (
        cfg.env.kernel_timing in valid_timings
    ), f"invalid env.kernel_timing: {cfg.env.kernel_timing}"

    assert (cfg.env.kappa_kernel != 0.0) == (
        cfg.model.n_fb == 2
    ), f"env.kappa_kernel={cfg.env.kappa_kernel} requires model.n_fb=2, got {cfg.model.n_fb}"

    valid_modes = {"dense", "sparse", "zero", "lowrank"}
    conn_fields = [
        "conn_c_from_v",
        "conn_v_from_c",
        "conn_p_from_v",
        "conn_v_from_p",
        "conn_p_from_c",
        "conn_c_from_p",
    ]
    for name in conn_fields:
        spec = getattr(cfg.model, name)
        assert spec.mode in valid_modes, f"model.{name}.mode invalid: {spec.mode}"
        if spec.mode == "sparse":
            assert 0.0 < spec.sparsity < 1.0, f"model.{name}.sparsity must be in (0,1)"
        if spec.mode == "lowrank":
            assert spec.rank > 0, f"model.{name}.rank must be > 0"

    valid_timings = {"pulse_only", "persist_iti", "persist_iti_and_decision"}
    assert cfg.env.reward_timing in valid_timings
    assert cfg.env.kernel_timing in valid_timings


def log_run_to_registry(cfg, run_id, registry_path="./runs/registry.csv"):

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "kappa_kernel": cfg.env.kappa_kernel,
        "xi_kernel": cfg.env.xi_kernel,
        "reward_timing": cfg.env.reward_timing,
        "kernel_timing": cfg.env.kernel_timing,
        "theta": cfg.decision.theta,
        "n_steps": cfg.training.n_steps,
        "iti": cfg.env.T_iti,
    }
    file_exists = os.path.exists(registry_path)
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


class PlateauMonitor:
    """Stops training if a smoothed metric hasn't improved by more than
    `min_delta` over `patience` evaluation checkpoints"""

    def __init__(self, patience=10, min_delta=0.002, smoothing_window=10):
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window
        self.values = []
        self.weights = []
        self.best = -float("inf")
        self.steps_since_best = 0

    def update(self, value: float, weight: float, curriculum_at_max: bool) -> bool:
        self.values.append(value)
        self.weights.append(max(weight, 1e-8))

        if not curriculum_at_max:
            self.steps_since_best = 0
            return False

        v = np.array(self.values[-self.smoothing_window :])
        w = np.array(self.weights[-self.smoothing_window :])
        smoothed = float(np.average(v, weights=w))

        if smoothed > self.best + self.min_delta:
            self.best = smoothed
            self.steps_since_best = 0
        else:
            self.steps_since_best += 1

        return self.steps_since_best >= self.patience
