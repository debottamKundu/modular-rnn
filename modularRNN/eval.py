import torch
import numpy as np

from .env import AsyncClosedLoopEnv


def run_eval_rollout(
    model, plan_gen, device, env_cfg, decision_cfg, n_steps=5000, batch_size=128
):
    """Runs a dedicated, closed-loop (no teacher forcing) rollout purely for
    evaluation -- a SEPARATE environment instance from the training one, so
    this never perturbs the training env's ongoing session/hidden state."""
    eval_env = AsyncClosedLoopEnv(
        model=model,
        plan_gen=plan_gen,
        batch_size=batch_size,
        device=device,
        T_max=env_cfg.T_max,
        T_iti=env_cfg.T_iti,
        t_min=decision_cfg.t_min,
        theta=decision_cfg.theta,
        sigma_stim=env_cfg.sigma_stim,
        reward_timing=env_cfg.reward_timing,
        kernel_timing=env_cfg.kernel_timing,
        t_quiescence=env_cfg.t_quiescence,
        kappa_kernel=env_cfg.kappa_kernel,
        xi_kernel=env_cfg.xi_kernel,
    )
    eval_env.reset()

    model.eval()
    results = []
    with torch.no_grad():
        for _ in range(n_steps):
            res = eval_env.step_dt(teacher_forcing_prob=0.0)
            if res is not None:
                results.append(res)
    model.train()
    return results


def accuracy_on_low_contrast_incongruent(
    results, target_contrast=0.0625, min_trials=10
):
    """
    Returns (accuracy, n_trials) restricted to committed trials at the given
    contrast, incongruent only. Returns (nan, n) if fewer than `min_trials`
    are available -- caller should treat this as "not enough data to judge,"
    not as a genuine low score.
    """
    if len(results) == 0:
        return float("nan"), 0

    contrast = torch.cat([r.contrast for r in results])
    stim_side = torch.cat([r.stim_side for r in results])
    block_bias = torch.cat([r.block_bias for r in results])
    choice = torch.cat([r.choice for r in results])
    correct_label = torch.cat([r.correct_label for r in results])
    timed_out = torch.cat([r.timed_out for r in results])

    is_biased = block_bias != 0.5
    block_right = block_bias > 0.5
    stim_right = stim_side > 0
    incongruent = (block_right != stim_right) & is_biased

    mask = (
        torch.isclose(contrast, torch.tensor(target_contrast, device=contrast.device))
        & incongruent
        & (timed_out == 0)
    )

    n = int(mask.sum().item())
    if n < min_trials:
        return float("nan"), n

    acc = (choice[mask] == correct_label[mask]).float().mean().item()
    return acc, n
