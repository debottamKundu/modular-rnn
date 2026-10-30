import os
import hydra
import torch
from omegaconf import OmegaConf

from modularRNN.train import build_conn_specs
from .conf.config import Config
from .model import ModularCTRNN3
from .env import AsyncClosedLoopEnv
from .generators import SessionPlanGenerator
from .storage import HiddenStateStore
from .singlesession import run_and_save_single_session


def load_checkpoint(ckpt_path, device, fallback_config_path=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    if fallback_config_path is not None:
        cfg = OmegaConf.load(fallback_config_path)
    elif "config" in ckpt:
        cfg = OmegaConf.create(ckpt["config"])
    else:
        raise ValueError(
            f"{ckpt_path} has no embedded config and no fallback_config_path given -- "
            "fetch it from wandb first via fetch_and_save_config(...)."
        )
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
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from step {ckpt['step']}")
    return model, cfg


@torch.no_grad()
def generate_batched_metadata(
    model,
    cfg,
    device,
    n_steps,
    batch_size,
    out_path,
    teacher_forcing_prob=0.0,
    noise_scale=1.0,
):
    """Large-N, metadata-only rollout (no hidden states) -- for psychometrics,
    congruency breakdowns"""
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
        batch_size=batch_size,
        device=device,
        T_max=cfg.env.T_max,
        T_iti=cfg.env.T_iti,
        t_min=cfg.decision.t_min,
        theta=cfg.decision.theta,
        sigma_stim=cfg.env.sigma_stim,
        hold_fb_during_stim=cfg.env.hold_fb_during_stim,
        t_quiescence=cfg.env.t_quiescence,
        kappa_kernel=cfg.env.kappa_kernel,
        xi_kernel=cfg.env.xi_kernel,
    )
    env.reset()

    store = HiddenStateStore(
        out_path,
        cfg.model.n_visp,
        cfg.model.n_mos,
        cfg.model.n_stim,
        store_full_traces=False,
    )

    n_total_trials = 0
    for t in range(n_steps):
        res = env.step_dt(
            teacher_forcing_prob=teacher_forcing_prob,
        )
        if res is not None:
            store.append(res, training_step=-1)
            n_total_trials += len(res.choice)
        if t % 5000 == 0:
            print(f"  dt-step {t}/{n_steps} | trials resolved so far: {n_total_trials}")

    print(f"Done. {n_total_trials} trials written to {out_path}")
    return store


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(run_cfg):
    fallback = (
        "/Users/dkundu/Documents/phd/modular-rnn/model/checkpoints/scneods2/config.yaml"
    )
    run_cfg.posthoc.ckpt_path = "/Users/dkundu/Documents/phd/modular-rnn/model/checkpoints/scneods2/ckpt_19000.pt"
    model_name = run_cfg.posthoc.ckpt_path.rsplit("/")[-2]
    device = torch.device(run_cfg.training.device)
    model, train_cfg = load_checkpoint(
        run_cfg.posthoc.ckpt_path, device, fallback_config_path=fallback
    )

    os.makedirs(run_cfg.paths.data_dir, exist_ok=True)

    if run_cfg.posthoc.mode == "batched":
        generate_batched_metadata(
            model,
            train_cfg,
            device,
            n_steps=run_cfg.posthoc.n_dt_steps,
            batch_size=run_cfg.posthoc.batch_size,
            out_path=os.path.join(run_cfg.paths.data_dir, "posthoc_batched.zarr"),
            teacher_forcing_prob=0.0,
            noise_scale=run_cfg.posthoc.noise_scale,
        )
    elif run_cfg.posthoc.mode == "single_session":
        plan_gen = SessionPlanGenerator(
            n_blocks=train_cfg.env.n_blocks,
            n_unbiased_blocks=train_cfg.env.n_unbiased_blocks,
            bias_high=train_cfg.env.bias_high,
            block_len_mean=train_cfg.env.block_len_mean,
            block_len_min=train_cfg.env.block_len_min,
            block_len_max=train_cfg.env.block_len_max,
            possible_contrasts=train_cfg.env.contrasts,
            device=device,
        )
        os.makedirs(f"run_cfg.paths.data_dir/{model_name}/", exist_ok=True)
        for i in range(run_cfg.posthoc.total_sessions):
            run_and_save_single_session(
                model,
                plan_gen,
                device,
                train_cfg.env,
                train_cfg.decision,
                out_path=os.path.join(
                    run_cfg.paths.data_dir,
                    f"{model_name}/posthoc_{model_name}_{i}.zarr",
                ),
                max_trials=run_cfg.posthoc.max_trials,
                teacher_forcing_prob=0.0,
            )
    else:
        raise ValueError(f"unknown posthoc.mode: {run_cfg.posthoc.mode}")


def ablate_model(model):
    with torch.no_grad():
        model.W_fb.zero_()
    return model


def run_and_save_multi_session(
    model,
    plan_gen,
    device,
    env_cfg,
    decision_cfg,
    out_dir,
    n_sessions,
    teacher_forcing_prob=0.0,
):
    """Runs n_sessions independent full sessions, each with its own complete
    hidden-state traces, written to separate zarr stores. Use this (not
    single-session, not batched-metadata-only) when you want to POOL trials
    across sessions for decoding -- gives independent draws of the task
    rather than repeated samples from one session's particular realization."""
    paths = []
    for i in range(n_sessions):
        out_path = f"{out_dir}/session_{i:03d}.zarr"
        run_and_save_single_session(
            model,
            plan_gen,
            device,
            env_cfg,
            decision_cfg,
            out_path=out_path,
            teacher_forcing_prob=teacher_forcing_prob,
        )
        paths.append(out_path)
    return paths


if __name__ == "__main__":
    main()
