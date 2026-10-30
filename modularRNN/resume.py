import os
import copy
import hydra
import torch
import wandb
from omegaconf import OmegaConf, DictConfig

from modularRNN.train import build_conn_specs

from .conf.config import Config
from .model import ModularCTRNN3
from .env import AsyncClosedLoopEnv
from .generators import SessionPlanGenerator
from .losses import compute_losses
from .congruency import congruency_accuracy_scalars
from .utils import (
    set_seed,
    spectral_radius,
    get_run_id,
    log_run_to_registry,
    validate_config_consistency,
    PlateauMonitor,
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


def backfill_legacy_config(base_cfg):
    """
    Older checkpoints predate several config fields. Backfill them to the
    values that reproduce what those checkpoints were ACTUALLY trained
    under, not today's defaults -- these are not arbitrary choices.

    Known schema eras:
      1. Oldest: no fb_mode/fb_sparsity, no reward_timing/kernel_timing at all.
      2. Intermediate: has `hold_fb_during_stim` (bool), no fb_mode, no enums.
      3. Current: has fb_mode/fb_sparsity and reward_timing/kernel_timing directly.
    """
    if "fb_mode" not in base_cfg.model:
        print(
            "Backfilling model.fb_mode='dense', model.fb_sparsity=1.0 "
            "(checkpoint predates sparsity/ablation support -- was trained fully dense)"
        )
        base_cfg.model.fb_mode = "dense"
        base_cfg.model.fb_sparsity = 1.0

    if "reward_timing" not in base_cfg.env or "kernel_timing" not in base_cfg.env:
        if "hold_fb_during_stim" in base_cfg.env:
            legacy_hold = bool(base_cfg.env.hold_fb_during_stim)
            mapped = "persist_iti_and_decision" if legacy_hold else "persist_iti"
            print(
                f"Backfilling env.reward_timing=env.kernel_timing='{mapped}' "
                f"(mapped from legacy hold_fb_during_stim={legacy_hold})"
            )
            base_cfg.env.reward_timing = mapped
            base_cfg.env.kernel_timing = mapped
        else:
            print(
                "Backfilling env.reward_timing=env.kernel_timing='pulse_only' "
                "(oldest checkpoint schema, predates persistence options entirely)"
            )
            base_cfg.env.reward_timing = "pulse_only"
            base_cfg.env.kernel_timing = "pulse_only"

    return base_cfg


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cli_cfg: Config):
    resume_path = cli_cfg.training.resume_from
    if not resume_path:
        raise ValueError(
            "resume.py requires training.resume_from=<path to .pt checkpoint>"
        )

    ckpt = torch.load(resume_path, map_location="cpu")
    if "config" not in ckpt:
        raise ValueError(
            f"{resume_path} has no embedded config -- cannot safely resume."
        )

    base_cfg = OmegaConf.create(ckpt["config"])
    print("--- checkpoint's original config (before backfill) ---")
    print(OmegaConf.to_yaml(base_cfg))

    base_cfg = backfill_legacy_config(base_cfg)

    # only TRAINING-level fields may be overridden via CLI when resuming --
    # architecture fields are fixed by the checkpoint's own tensor shapes.
    cli_overrides = OmegaConf.create(
        {"training": OmegaConf.to_container(cli_cfg.training)}
    )
    cfg = OmegaConf.merge(base_cfg, cli_overrides)

    print(
        "--- final merged config (architecture from checkpoint, training from CLI) ---"
    )
    print(OmegaConf.to_yaml(cfg))

    validate_config_consistency(cfg)
    set_seed(cfg.training.seed)
    device = torch.device(cfg.training.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device=cuda but CUDA is not available.")

    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.paths.data_dir, exist_ok=True)
    os.makedirs(cfg.paths.wandb_dir, exist_ok=True)

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

    model.load_state_dict(ckpt["model"])
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except (ValueError, KeyError) as e:
        print(
            f"WARNING: could not restore optimizer state ({e}); starting optimizer fresh."
        )
    start_step = ckpt["step"] + 1
    run_id = cfg.training.run_id
    # run_id = ckpt.get("run_id", get_run_id(cfg.training.use_wandb))
    print(
        f"Resuming model+optimizer from step {ckpt['step']} -> continuing at step {start_step}"
    )

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

    run_ckpt_dir = os.path.join(cfg.paths.checkpoint_dir, run_id)
    run_data_dir = os.path.join(cfg.paths.data_dir, run_id)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    os.makedirs(run_data_dir, exist_ok=True)

    if cfg.training.use_wandb:
        wandb.init(
            project="modular-rnn-ibl",
            dir=cfg.paths.wandb_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            id=run_id,
            resume="allow",
        )
        wandb.watch(model, log="gradients", log_freq=200)

    plateau_monitor = PlateauMonitor(
        patience=cfg.training.plateau_patience,
        min_delta=cfg.training.plateau_min_delta,
        smoothing_window=cfg.training.plateau_smoothing_window,
    )
    plateau_state = ckpt.get("plateau_monitor_state")
    if plateau_state is not None:
        plateau_monitor.values = plateau_state["values"]
        plateau_monitor.weights = plateau_state["weights"]
        plateau_monitor.best = plateau_state["best"]
        plateau_monitor.steps_since_best = plateau_state["steps_since_best"]
        print("Restored plateau monitor state from checkpoint.")
    else:
        print(
            "No plateau monitor state in checkpoint -- starting it fresh from this point."
        )

    env.reset()
    last_good_state = copy.deepcopy(model.state_dict())
    last_good_opt_state = copy.deepcopy(optimizer.state_dict())
    stopped_early = False

    def save_checkpoint(step, tag="periodic"):
        path = os.path.join(run_ckpt_dir, f"ckpt_{tag}_{step}.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "run_id": run_id,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "plateau_monitor_state": {
                    "values": plateau_monitor.values,
                    "weights": plateau_monitor.weights,
                    "best": plateau_monitor.best,
                    "steps_since_best": plateau_monitor.steps_since_best,
                },
            },
            path,
        )
        if cfg.training.use_wandb:
            art = wandb.Artifact(f"ckpt_{tag}_{run_id}_{step}", type="model")
            art.add_file(path)
            wandb.log_artifact(art)
        print(f"[step {step}] saved checkpoint ({tag}) to {path}")

    for step in range(start_step, cfg.training.n_steps):
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
            if cfg.training.use_wandb:
                wandb.log({"train/empty_chunk": 1, "step": step})
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
            env.reset()
            if cfg.training.use_wandb:
                wandb.log({"train/nan_rollback": 1, "step": step})
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
        n_resolved = len(all_choice)
        repeat_rate = None
        if has_prev.any():
            prev_cat = torch.cat([r.prev_model_choice for r in results]).long()
            repeat_rate = (
                (all_choice[has_prev] == prev_cat[has_prev]).float().mean().item()
            )
        congr_scalars = congruency_accuracy_scalars(results, cfg.env.contrasts)

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
                f"L_total={L_total.item():.4f} acc={acc:.3f} no_resp={no_resp:.3f}"
            )

        if cfg.training.use_wandb and step % cfg.training.ckpt_every == 0:
            save_checkpoint(step, tag="periodic")

        if cfg.training.enable_early_stopping:
            curriculum_at_max = chunk_len >= cfg.training.chunk_dt_max
            should_stop = plateau_monitor.update(
                acc, weight=n_resolved, curriculum_at_max=curriculum_at_max
            )
            if should_stop and step >= cfg.training.min_steps_before_stop:
                print(f"[step {step}] plateau detected -- stopping early.")
                save_checkpoint(step, tag="early_stop_final")
                stopped_early = True
                break

    if not stopped_early:
        save_checkpoint(cfg.training.n_steps - 1, tag="final")

    log_run_to_registry(cfg, run_id)


if __name__ == "__main__":
    main()
