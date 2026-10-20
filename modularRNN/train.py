import os
import copy
import tempfile
import hydra
from matplotlib import pyplot as plt
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from conf.config import Config
from model import ModularCTRNN
from env import ClosedLoopEnv
from generators import SessionPlanGenerator
from losses import compute_losses
from storage import HiddenStateStore
from utils import get_run_id, set_seed, spectral_radius
from eval import (
    make_eval_env,
    collect_eval_rollout,
    fit_psychometric,
    plot_rt_histogram,
    run_probe_decoding,
    plot_probe_decoding,
)


def k_schedule(step, cfg):
    return min(
        cfg.training.k_start + step // cfg.training.k_growth_every, cfg.training.k_max
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
    set_seed(cfg.training.seed)
    device = torch.device(cfg.training.device)

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
    ).to(device)

    if cfg.training.compile_model and device.type != "mps":
        model = torch.compile(model)

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

    env = ClosedLoopEnv(
        model=model,
        plan_gen=plan_gen,
        batch_size=cfg.env.batch_size,
        device=device,
        T_stim=cfg.env.T_stim,
        T_resp=cfg.env.T_resp,
        T_iti=cfg.env.T_iti,
        t_min=cfg.decision.t_min,
        theta=cfg.decision.theta,
        sigma_stim=cfg.env.sigma_stim,
        hold_fb_during_stim=cfg.env.hold_fb_during_stim,
        t_quiescence=cfg.env.t_quiescence,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    if cfg.training.use_wandb:
        wandb.init(
            project="modular-rnn-ibl",
            dir=cfg.paths.wandb_dir,
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
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

    hidden_store = None
    if cfg.training.store_hidden_during_training:
        hidden_store = HiddenStateStore(
            os.path.join(run_data_dir, "hidden_states_train.zarr"),
            cfg.model.n_visp,
            cfg.model.n_mos,
            cfg.model.n_stim,
            store_full_traces=True,
        )

    env.reset()

    eval_env = make_eval_env(
        model, plan_gen, cfg.env.eval_batch_size, device, cfg.env, cfg.decision
    )
    last_good_state = copy.deepcopy(model.state_dict())
    last_good_opt_state = copy.deepcopy(optimizer.state_dict())

    for step in range(cfg.training.n_steps):
        K = k_schedule(step, cfg)
        tf_prob = anneal(
            step,
            cfg.training.n_steps,
            cfg.training.teacher_forcing_start,
            cfg.training.teacher_forcing_end,
            cfg.training.teacher_forcing_anneal_frac,
        )

        results, rate_terms = [], []
        for _ in range(K):
            store_this = (
                cfg.training.store_hidden_during_training and hidden_store is not None
            )
            res = env.run_trial(
                teacher_forcing_prob=tf_prob, store_full_trace=store_this
            )
            results.append(res)
            rate_terms.append((model.phi(env.h) ** 2).mean())
            if store_this:
                hidden_store.append(res, training_step=step)  # type: ignore

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
            env.h = torch.zeros_like(env.h)  # pyright: ignore[reportArgumentType]
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
        assert env.h is not None
        env.h = env.h.detach()  # truncated BPTT boundary

        if step % 50 == 0:
            last_good_state = copy.deepcopy(model.state_dict())
            last_good_opt_state = copy.deepcopy(optimizer.state_dict())

        # --- metrics (computed regardless of logging backend) ---
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
                    "train/chunk_len_K": K,
                    "train/teacher_forcing_prob": tf_prob,
                    "diag/W_ff_effective_rank": model.effective_rank_ff(),
                    "diag/W_fb_norm": model.W_fb.norm().item(),
                    "diag/W_vv_specrad": spectral_radius(model.W_vv),
                    "diag/W_mm_specrad": spectral_radius(model.W_mm),
                    "step": step,
                }
            )
        else:
            # console fallback
            print(
                f"step {step:5d} | K={K:3d} | L_total={L_total.item():.4f} "
                f"L_choice={L_choice.item():.4f} L_aux={L_aux.item():.4f} "
                f"L_switch={L_switch.item():.4f} | acc={acc:.3f} "
                f"no_resp={no_resp:.3f} repeat={repeat_rate if repeat_rate is None else round(repeat_rate,3)} "
                f"| grad_norm={grad_norm.item():.3f}"
            )



        # this ensures local training runs are never saved.
        if cfg.training.use_wandb and step % cfg.training.ckpt_every == 0:
            path = os.path.join(run_ckpt_dir, f"ckpt_{step}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                },
                path,
            )
            art = wandb.Artifact(f"ckpt_{step}", type="model")
            art.add_file(path)
            wandb.log_artifact(art)


if __name__ == "__main__":
    main()
