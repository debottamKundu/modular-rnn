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
    return "smoketest_" + next(tempfile._get_candidate_names())
