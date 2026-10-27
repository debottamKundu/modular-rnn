import random
import numpy as np
import torch
import torch.nn.functional as F
import tempfile
import wandb

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

    if cfg.model.fb_mode == "sparse" and not (0.0 < cfg.model.fb_sparsity < 1.0):
        raise ValueError(
            f"model.fb_mode='sparse' requires 0 < fb_sparsity < 1, got {cfg.model.fb_sparsity}"
        )

    if cfg.model.fb_mode != "sparse" and cfg.model.fb_sparsity != 1.0:
        print(
            f"WARNING: model.fb_sparsity={cfg.model.fb_sparsity} has no effect "
            f"since model.fb_mode='{cfg.model.fb_mode}' (only used when fb_mode='sparse')."
        )

    valid_timings = {"pulse_only", "persist_iti", "persist_iti_and_decision"}
    assert (
        cfg.env.reward_timing in valid_timings
    ), f"invalid env.reward_timing: {cfg.env.reward_timing}"
    assert (
        cfg.env.kernel_timing in valid_timings
    ), f"invalid env.kernel_timing: {cfg.env.kernel_timing}"

    valid_fb_modes = {"dense", "zero", "sparse"}
    assert (
        cfg.model.fb_mode in valid_fb_modes
    ), f"invalid model.fb_mode: {cfg.model.fb_mode}"
