import torch
from env import AsyncClosedLoopEnv
from storage import HiddenStateStore


def run_and_save_single_session(
    model,
    plan_gen,
    device,
    env_cfg,
    decision_cfg,
    out_path,
    n_trials,
    teacher_forcing_prob=0.0,
):
    """Runs ONE session (batch_size=1) and writes a full per-trial hidden-state
    trace (padded to T_max + T_iti) plus all scalar metadata to a zarr store."""
    env = AsyncClosedLoopEnv(
        model=model,
        plan_gen=plan_gen,
        batch_size=1,
        device=device,
        T_max=env_cfg.T_max,
        T_iti=env_cfg.T_iti,
        t_min=decision_cfg.t_min,
        theta=decision_cfg.theta,
        sigma_stim=env_cfg.sigma_stim,
        hold_fb_during_stim=env_cfg.hold_fb_during_stim,
        t_quiescence=env_cfg.t_quiescence,
        kappa_kernel=env_cfg.kappa_kernel,
        xi_kernel=env_cfg.xi_kernel,
    )
    env.reset()

    max_len = env_cfg.T_max + env_cfg.T_iti
    store = HiddenStateStore(
        out_path, model.n_visp, model.n_mos, model.n_stim, store_full_traces=True
    )

    model.eval()
    buf_r_v, buf_r_m, buf_x_stim = [], [], []
    pending_result = None
    n_saved = 0

    with torch.no_grad():
        prev_phase, prev_phase_step = env.phase[0].item(), env.phase_step[0].item()

        while n_saved < n_trials:
            res, r_v, r_m, x_stim = env.step_dt(
                teacher_forcing_prob=teacher_forcing_prob, return_hidden=True
            )  # type: ignore
            buf_r_v.append(r_v[0])
            buf_r_m.append(r_m[0])
            buf_x_stim.append(x_stim[0])
            if res is not None:
                pending_result = (
                    res  # commit/timeout happened this step; ITI still to follow
                )

            new_phase, new_phase_step = env.phase[0].item(), env.phase_step[0].item()
            new_trial_started = (
                prev_phase == 1 and new_phase == 0 and new_phase_step == 0
            )

            if new_trial_started and pending_result is not None:
                trace_len = len(buf_r_v)
                pad = max_len - trace_len
                r_v_arr = torch.stack(buf_r_v, dim=0)
                r_m_arr = torch.stack(buf_r_m, dim=0)
                x_stim_arr = torch.stack(buf_x_stim, dim=0)
                if pad > 0:
                    r_v_arr = torch.cat(
                        [r_v_arr, torch.zeros(pad, model.n_visp, device=device)], dim=0
                    )
                    r_m_arr = torch.cat(
                        [r_m_arr, torch.zeros(pad, model.n_mos, device=device)], dim=0
                    )
                    x_stim_arr = torch.cat(
                        [x_stim_arr, torch.zeros(pad, model.n_stim, device=device)],
                        dim=0,
                    )

                pending_result.full_r_v = r_v_arr.unsqueeze(0)
                pending_result.full_r_m = r_m_arr.unsqueeze(0)
                pending_result.full_x_stim = x_stim_arr.unsqueeze(0)

                store.append(pending_result, training_step=-1)
                n_saved += 1
                buf_r_v, buf_r_m, buf_x_stim = [], [], []
                pending_result = None

            prev_phase, prev_phase_step = new_phase, new_phase_step

    model.train()
    return store


"""
command:
python modularRNN/train.py experiment=server_run \
  model.n_visp=64 model.n_mos=64 model.rank_ff=3 \
  env.n_blocks=5 env.n_unbiased_blocks=1 \
  env.block_len_mean=60 env.block_len_min=20 env.block_len_max=100 \
  env.contrasts="[0.0,0.0625,0.125,0.25,1.0]" \
  training.n_steps=50000 \
  training.k_start=4 training.k_max=60 training.k_growth_every=500 \
  training.eval_every=200 training.probe_every=1000 training.ckpt_every=1000
  """
