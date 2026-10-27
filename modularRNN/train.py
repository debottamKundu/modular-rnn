import os
import copy
import tempfile
import hydra
from matplotlib import pyplot as plt
import torch

torch.autograd.set_detect_anomaly(True)
import wandb
from omegaconf import DictConfig, OmegaConf

from .conf.config import Config
from .model import ModularCTRNN
from .env import AsyncClosedLoopEnv
from .generators import SessionPlanGenerator
from .losses import compute_losses
from .congruency import congruency_accuracy_scalars
from .storage import HiddenStateStore
from .utils import get_run_id, set_seed, spectral_radius, validate_config_consistency

# from eval import (
#     make_eval_env,
#     collect_eval_rollout,
#     fit_psychometric,
#     plot_rt_histogram,
#     run_probe_decoding,
#     plot_probe_decoding,
# )


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


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))
    validate_config_consistency(cfg)
    set_seed(cfg.training.seed)
    device = torch.device(cfg.training.device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Config requested training.device=cuda, but torch.cuda.is_available() "
                "is False. Check CUDA drivers / PyTorch CUDA build before proceeding."
            )
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.set_float32_matmul_precision("high")

    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.paths.data_dir, exist_ok=True)
    os.makedirs(cfg.paths.wandb_dir, exist_ok=True)

    model = ModularCTRNN(
        n_visp=cfg.model.n_visp,
        n_mos=cfg.model.n_mos,
        n_stim=cfg.model.n_stim,
        n_fb=cfg.model.n_fb,
        rank_ff=cfg.model.rank_ff,
        dt=cfg.model.dt,
        tau_visp_range=cfg.model.tau_visp_range,
        tau_mos_range=cfg.model.tau_mos_range,
        learnable_tau=cfg.model.learnable_tau,
        rec_gain=cfg.model.rec_gain,
        ff_gain=cfg.model.ff_gain,
        fb_gain=cfg.model.fb_gain,
        in_gain=cfg.model.in_gain,
        sigma_baseline=cfg.model.sigma_baseline,
        nonlinearity=cfg.model.nonlinearity,
        fb_mode=cfg.model.fb_mode,
        fb_sparsity=cfg.model.fb_sparsity,
    ).to(device)

    # if cfg.training.compile_model and device.type != "mps":
    #     model = torch.compile(model)

    if cfg.training.compile_model and device.type == "cuda":
        model.step = torch.compile(model.step)  # , mode="reduce-overhead")
        model.readout = torch.compile(model.readout)  # , mode="reduce-overhead")

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
    assert (cfg.env.kappa_kernel != 0.0) == (cfg.model.n_fb == 2), (
        f"Inconsistent config: env.kappa_kernel={cfg.env.kappa_kernel} implies "
        f"use_action_kernel={cfg.env.kappa_kernel != 0.0}, but model.n_fb={cfg.model.n_fb} "
        f"(must be 2 if action kernel is enabled, 1 if disabled)."
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

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    # TODO: change this every time for a new sweep, for my sanity
    if cfg.training.use_wandb:
        wandb.init(
            project="modular-rnn-ibl",
            dir=cfg.paths.wandb_dir,
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
            notes="Feedback matrix from Wmv zeroed out, action kernel and feedback are pulsed, action kernel is a moving average",
        )
        wandb.watch(model, log="gradients", log_freq=200)  # type: ignore
    run_id = get_run_id(cfg.training.use_wandb)

    if cfg.training.use_wandb:
        run_ckpt_dir = os.path.join(cfg.paths.checkpoint_dir, run_id)
        run_data_dir = os.path.join(cfg.paths.data_dir, run_id)
    else:
        # smoke-test
        run_ckpt_dir = tempfile.mkdtemp(
            prefix=f"{run_id}_ckpt_", dir=cfg.paths.checkpoint_dir
        )
        run_data_dir = tempfile.mkdtemp(
            prefix=f"{run_id}_data_", dir=cfg.paths.data_dir
        )
        print(
            f"[no-wandb run] checkpoints/data are temporary, written to:\n  {run_ckpt_dir}\n  {run_data_dir}"
        )
    os.makedirs(run_ckpt_dir, exist_ok=True)
    os.makedirs(run_data_dir, exist_ok=True)

    # hidden_store = None
    # if cfg.training.store_hidden_during_training:
    #     hidden_store = HiddenStateStore(
    #         os.path.join(run_data_dir, "hidden_states_train.zarr"),
    #         cfg.model.n_visp,
    #         cfg.model.n_mos,
    #         cfg.model.n_stim,
    #         store_full_traces=True,
    #         t_iti=cfg.env.T_iti,
    #     )

    env.reset()

    # eval_env = make_eval_env(
    #     model, plan_gen, cfg.env.eval_batch_size, device, cfg.env, cfg.decision
    # )
    last_good_state = copy.deepcopy(model.state_dict())
    last_good_opt_state = copy.deepcopy(optimizer.state_dict())

    for step in range(cfg.training.n_steps):  # to store the last checkpoint

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

        assert env.h is not None
        # env.h = env.h.detach()  # truncated BPTT boundary
        env.detach_state()

        if len(results) == 0:
            if cfg.training.use_wandb:
                wandb.log({"train/empty_chunk": 1, "step": step})
            else:
                print(
                    f"step {step:5d} | chunk_len={chunk_len} | no trials resolved this chunk"
                )
            continue

        L_choice, L_aux, L_switch = compute_losses(results)
        L_rate = torch.stack(rate_terms).mean()
        L_weight = (model.W_fb**2).sum()
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
            env.zero_hidden_state()
            env.reset()
            msg = (
                f"[step {step}] NaN/Inf detected -- rolled back to last good checkpoint"
            )
            (
                print(msg)
                if not cfg.training.use_wandb
                else wandb.log({"train/nan_rollback": 1, "step": step})
            )
            continue

        optimizer.step()

        if step % 50 == 0:
            last_good_state = copy.deepcopy(model.state_dict())
            last_good_opt_state = copy.deepcopy(optimizer.state_dict())

        all_choice = torch.cat([r.choice for r in results])
        all_correct = torch.cat([r.correct_label for r in results])
        all_timed_out = torch.cat([r.timed_out for r in results])
        has_prev = torch.cat([r.has_prev_choice for r in results])
        acc = (all_choice == all_correct).float().mean().item()
        no_resp = all_timed_out.float().mean().item()
        repeat_rate = None
        if has_prev.any():
            prev_cat = torch.cat([r.prev_model_choice for r in results]).long()
            repeat_rate = (
                (all_choice[has_prev] == prev_cat[has_prev]).float().mean().item()
            )
        congr_scalars = congruency_accuracy_scalars(results, cfg.env.contrasts)
        n_resolved = len(all_choice)

        if cfg.training.use_wandb:
            wandb.log(
                {
                    "train/L_total": L_total.item(),
                    "train/L_choice": L_choice.item(),
                    "train/L_aux": L_aux.item(),
                    "train/L_rate": L_rate.item(),
                    "train/L_switch": L_switch.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/accuracy": acc,
                    "train/no_response_rate": no_resp,
                    "train/repeat_rate": repeat_rate,
                    "train/n_resolved_trials": n_resolved,
                    "train/chunk_len_dt": chunk_len,
                    "train/teacher_forcing_prob": tf_prob,
                    "diag/W_ff_effective_rank": model.effective_rank_ff(),
                    "diag/W_fb_norm": model.W_fb.norm().item(),
                    "diag/W_vv_specrad": spectral_radius(model.W_vv),
                    "diag/W_mm_specrad": spectral_radius(model.W_mm),
                    "step": step,
                    "congruency": congr_scalars,
                }
            )
        else:
            print(
                f"step {step:5d} | chunk_dt={chunk_len:4d} | n_resolved={n_resolved:3d} | "
                f"L_total={L_total.item():.4f} L_choice={L_choice.item():.4f} "
                f"L_aux={L_aux.item():.4f} L_switch={L_switch.item():.4f} | acc={acc:.3f} "
                f"no_resp={no_resp:.3f} repeat={repeat_rate if repeat_rate is None else round(repeat_rate,3)} "
                f"| grad_norm={grad_norm.item():.3f}"
            )

        if cfg.training.use_wandb and step % cfg.training.ckpt_every == 0:
            path = os.path.join(run_ckpt_dir, f"ckpt_{step}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                path,
            )
            art = wandb.Artifact(f"ckpt_{step}", type="model")
            art.add_file(path)
            wandb.log_artifact(art)


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
