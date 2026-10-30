import os
import copy
import tempfile
import hydra
from matplotlib import pyplot as plt
import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)
import wandb
from omegaconf import DictConfig, OmegaConf

from .conf.config import Config
from .model import ModularCTRNN3
from .env import AsyncClosedLoopEnv
from .generators import SessionPlanGenerator
from .losses import compute_losses
from .congruency import congruency_accuracy_scalars
from .storage import HiddenStateStore
from .utils import (
    PlateauMonitor,
    get_run_id,
    log_run_to_registry,
    set_seed,
    spectral_radius,
    validate_config_consistency,
)
from .eval import run_eval_rollout, accuracy_on_low_contrast_incongruent

# from eval import (
#     make_eval_env,
#     collect_eval_rollout,
#     fit_psychometric,
#     plot_rt_histogram,
#     run_probe_decoding,
#     plot_probe_decoding,
# )


def save_checkpoint(model, optimizer, step, cfg, run_ckpt_dir, run_id, tag="final"):
    path = os.path.join(run_ckpt_dir, f"ckpt_{tag}_{step}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "run_id": run_id,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )
    if cfg.training.use_wandb:
        art = wandb.Artifact(f"ckpt_{tag}_{run_id}_{step}", type="model")
        art.add_file(path)
        wandb.log_artifact(art)
    print(f"[step {step}] saved checkpoint ({tag}) to {path}")
    return path


def k_schedule(step, cfg):
    return min(
        cfg.training.k_start + step // cfg.training.k_growth_every, cfg.training.k_max
    )


def chunk_dt_schedule(step, cfg):
    return min(
        cfg.training.chunk_dt_start + step // cfg.training.chunk_dt_growth_every,
        cfg.training.chunk_dt_max,
    )


def anneal(step, n_steps, start, end, frac):
    anneal_steps = int(n_steps * frac)
    if step >= anneal_steps:
        return end
    t = step / max(anneal_steps, 1)
    return start + t * (end - start)


def build_conn_specs(cfg):
    def spec(c):
        return dict(mode=c.mode, gain=c.gain, sparsity=c.sparsity, rank=c.rank)

    return {
        ("c", "v"): spec(cfg.model.conn_c_from_v),
        ("v", "c"): spec(cfg.model.conn_v_from_c),
        ("p", "v"): spec(cfg.model.conn_p_from_v),
        ("v", "p"): spec(cfg.model.conn_v_from_p),
        ("p", "c"): spec(cfg.model.conn_p_from_c),
        ("c", "p"): spec(cfg.model.conn_c_from_p),
    }


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    validate_config_consistency(cfg)
    set_seed(cfg.training.seed)
    device = torch.device(cfg.training.device)

    model = ModularCTRNN3(
        n_v=cfg.model.n_v,
        n_c=cfg.model.n_c,
        n_p=cfg.model.n_p,
        n_stim=cfg.model.n_stim,
        n_fb=cfg.model.n_fb,
        dt=cfg.model.dt,
        tau_v_range=cfg.model.tau_v_range,
        tau_c_range=cfg.model.tau_c_range,
        tau_p_range=cfg.model.tau_p_range,
        learnable_tau=cfg.model.learnable_tau,
        rec_gain=cfg.model.rec_gain,
        in_gain=cfg.model.in_gain,
        sigma_baseline=cfg.model.sigma_baseline,
        nonlinearity=cfg.model.nonlinearity,
        conn_specs=build_conn_specs(cfg),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    plan_gen = SessionPlanGenerator(
        n_blocks=cfg.env.n_blocks,
        n_unbiased_blocks=cfg.env.n_unbiased_blocks,
        bias_high=cfg.env.bias_high,
        block_len_mean=cfg.env.block_len_mean,
        block_len_min=cfg.env.block_len_min,
        block_len_max=cfg.env.block_len_max,
        possible_contrasts=cfg.env.contrasts,
        device=device,
    )
    env = AsyncClosedLoopEnv(
        model=model,
        plan_gen=plan_gen,
        batch_size=cfg.env.batch_size,
        device=device,
        T_max=cfg.env.T_max,
        T_iti=cfg.env.T_iti,
        t_min=cfg.decision.t_min,
        theta=cfg.decision.theta,
        sigma_stim=cfg.env.sigma_stim,
        reward_timing=cfg.env.reward_timing,
        kernel_timing=cfg.env.kernel_timing,
        t_quiescence=cfg.env.t_quiescence,
        kappa_kernel=cfg.env.kappa_kernel,
        xi_kernel=cfg.env.xi_kernel,
    )

    if cfg.training.use_wandb:
        wandb.init(
            project="modular-rnn-ibl",
            dir=cfg.paths.wandb_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.watch(model, log="gradients", log_freq=200)

    run_id = get_run_id(cfg.training.use_wandb)
    run_ckpt_dir = os.path.join(cfg.paths.checkpoint_dir, run_id)
    os.makedirs(run_ckpt_dir, exist_ok=True)

    plateau_monitor = PlateauMonitor(
        patience=cfg.training.plateau_patience,
        min_delta=cfg.training.plateau_min_delta,
        smoothing_window=cfg.training.plateau_smoothing_window,
    )

    env.reset()
    last_good_state = copy.deepcopy(model.state_dict())
    last_good_opt_state = copy.deepcopy(optimizer.state_dict())
    stopped_early = False
    conn_edges = [
        ("c", "v"),
        ("v", "c"),
        ("p", "v"),
        ("v", "p"),
        ("p", "c"),
        ("c", "p"),
    ]

    def save_checkpoint(step, tag="periodic"):
        path = os.path.join(run_ckpt_dir, f"ckpt_{tag}_{step}.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "run_id": run_id,
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            path,
        )
        if cfg.training.use_wandb:
            art = wandb.Artifact(f"ckpt_{tag}_{run_id}_{step}", type="model")
            art.add_file(path)
            wandb.log_artifact(art)

    for step in range(cfg.training.n_steps):
        chunk_len = chunk_dt_schedule(step, cfg)
        tf_prob = anneal(
            step,
            cfg.training.n_steps,
            cfg.training.teacher_forcing_start,
            cfg.training.teacher_forcing_end,
            cfg.training.teacher_forcing_anneal_frac,
        )

        results, rate_terms = [], []
        for _ in range(chunk_len):
            res = env.step_dt(teacher_forcing_prob=tf_prob)
            if res is not None:
                results.append(res)
            rate_terms.append((model.phi(env.h) ** 2).mean())
        env.detach_state()

        if len(results) == 0:
            continue

        L_choice, L_aux, L_switch = compute_losses(results)
        L_rate = torch.stack(rate_terms).mean()
        L_weight = sum(
            model.connections[f"{d}_from_{s}"]().pow(2).sum() for d, s in conn_edges
        )
        L_total = (
            L_choice
            + cfg.loss.lambda_aux * L_aux
            + cfg.loss.lambda_rate * L_rate
            + cfg.loss.lambda_weight * L_weight
            + cfg.loss.lambda_switch * L_switch
        )

        optimizer.zero_grad()
        L_total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.training.grad_clip
        )

        if not torch.isfinite(L_total) or not torch.isfinite(grad_norm):
            model.load_state_dict(last_good_state)
            optimizer.load_state_dict(last_good_opt_state)
            env.reset()
            continue

        optimizer.step()
        if step % 50 == 0:
            last_good_state = copy.deepcopy(model.state_dict())
            last_good_opt_state = copy.deepcopy(optimizer.state_dict())

        all_choice = torch.cat([r.choice for r in results])
        all_correct = torch.cat([r.correct_label for r in results])
        all_timed_out = torch.cat([r.timed_out for r in results])
        acc = (all_choice == all_correct).float().mean().item()
        n_resolved = len(all_choice)

        if cfg.training.use_wandb:
            log_dict = {
                "train/L_total": L_total.item(),
                "train/L_choice": L_choice.item(),
                "train/L_aux": L_aux.item(),
                "train/L_rate": L_rate.item(),
                "train/L_switch": L_switch.item(),
                "train/grad_norm": grad_norm.item(),
                "train/accuracy": acc,
                "train/no_response_rate": all_timed_out.float().mean().item(),
                "train/n_resolved_trials": n_resolved,
                "train/chunk_len_dt": chunk_len,
                "step": step,
            }
            for d, s in conn_edges:
                log_dict[f"diag/rank_{d}_from_{s}"] = model.effective_rank(d, s)
                log_dict[f"diag/norm_{d}_from_{s}"] = model.conn_norm(d, s)
            wandb.log(log_dict)

        if cfg.training.use_wandb and step % cfg.training.ckpt_every == 0:
            save_checkpoint(step, tag="periodic")

        if cfg.training.enable_early_stopping:
            curriculum_at_max = chunk_len >= cfg.training.chunk_dt_max
            should_stop = plateau_monitor.update(
                acc, weight=n_resolved, curriculum_at_max=curriculum_at_max
            )
            if should_stop and step >= cfg.training.min_steps_before_stop:
                save_checkpoint(step, tag="early_stop_final")
                stopped_early = True
                break

    if not stopped_early:
        save_checkpoint(cfg.training.n_steps - 1, tag="final")
    log_run_to_registry(cfg, run_id)


if __name__ == "__main__":
    main()

"""
python modularRNN/train.py \
training.device=cpu training.use_wandb=true training.compile_model=false \
model.n_visp=32 model.n_mos=32 model.rank_ff=3 \
env.batch_size=32 \
env.n_blocks=5 env.n_unbiased_blocks=1 \
env.block_len_mean=60 env.block_len_min=20 env.block_len_max=100 \
env.contrasts="[0.0,0.0625,0.125,0.25,1.0]" \
env.T_max=100 env.T_iti=10 \
decision.t_min=10 decision.theta=0.9 \
env.kappa_kernel=0.5 env.xi_kernel=0.2 \
training.n_steps=20000 \
training.chunk_dt_start=150 training.chunk_dt_growth_every=500 training.chunk_dt_max=1000 \
training.ckpt_every=1000
"""
