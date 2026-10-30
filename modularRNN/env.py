from dataclasses import dataclass, replace
from typing import Optional
import torch

from .inputs import make_stimulus_input_step, make_feedback_vector

DECISION, ITI = 0, 1


@dataclass
class TrialResult:
    z_c: torch.Tensor
    aux_pred: torch.Tensor
    target_stim: torch.Tensor
    correct_label: torch.Tensor
    choice: torch.Tensor
    reward: torch.Tensor
    rt: torch.Tensor
    timed_out: torch.Tensor
    block_id: torch.Tensor
    block_bias: torch.Tensor
    trial_in_block: torch.Tensor
    contrast: torch.Tensor
    stim_side: torch.Tensor
    prev_model_choice: torch.Tensor
    has_prev_choice: torch.Tensor
    session_id: torch.Tensor
    action_kernel: torch.Tensor
    batch_idx: torch.Tensor
    full_r_v: Optional[torch.Tensor] = None
    full_r_c: Optional[torch.Tensor] = None
    full_r_p: Optional[torch.Tensor] = None


@dataclass
class EnvState:
    h: torch.Tensor
    phase: torch.Tensor
    phase_step: torch.Tensor
    trial_idx: torch.Tensor
    session_count: torch.Tensor
    last_outcome_fb: torch.Tensor
    action_kernel: torch.Tensor
    prev_model_choice: torch.Tensor
    has_prev_choice: torch.Tensor
    aux_accum: torch.Tensor
    aux_count: torch.Tensor
    cur_contrast: torch.Tensor
    cur_stim_side: torch.Tensor
    cur_correct_label: torch.Tensor
    cur_block_id: torch.Tensor
    cur_block_bias: torch.Tensor
    cur_trial_in_block: torch.Tensor
    cur_session_id: torch.Tensor

    def detach(self) -> "EnvState":
        return EnvState(**{k: v.detach() for k, v in self.__dict__.items()})


def _fetch_new_trials(
    state: EnvState, plan, plan_gen, mask: torch.Tensor, device
) -> EnvState:
    """Pure: returns a NEW EnvState with rows in `mask` advanced to their next
    trial. `plan`'s own tensors (generated session metadata, no grad) may be
    refilled in place by plan_gen -- that's safe since plan never touches
    autograd. `state` itself is never mutated; every field is reconstructed."""
    if not mask.any():
        return state

    need_refill = mask & (state.trial_idx >= plan["total_trials"])
    if need_refill.any():
        plan_gen.refill(plan, need_refill)

    trial_idx = torch.where(
        need_refill, torch.zeros_like(state.trial_idx), state.trial_idx
    )
    B = trial_idx.shape[0]
    b_idx = torch.arange(B, device=device)
    safe_t_idx = torch.clamp(trial_idx, max=plan["contrast"].shape[1] - 1)

    new_contrast = plan["contrast"][b_idx, safe_t_idx]
    new_stim_side = plan["stim_side"][b_idx, safe_t_idx]
    new_correct_label = plan["correct_label"][b_idx, safe_t_idx]
    new_block_id = plan["block_id"][b_idx, safe_t_idx]
    new_block_bias = plan["block_bias"][b_idx, safe_t_idx]
    new_trial_in_block = plan["trial_in_block"][b_idx, safe_t_idx]

    session_count = state.session_count + need_refill.long()
    new_session_id = b_idx.to(torch.long) * 10_000 + session_count

    h = torch.where(need_refill.unsqueeze(-1), torch.zeros_like(state.h), state.h)
    last_outcome_fb = torch.where(
        need_refill.unsqueeze(-1),
        torch.zeros_like(state.last_outcome_fb),
        state.last_outcome_fb,
    )
    action_kernel = torch.where(
        need_refill, torch.zeros_like(state.action_kernel), state.action_kernel
    )
    prev_model_choice = torch.where(
        need_refill, torch.zeros_like(state.prev_model_choice), state.prev_model_choice
    )
    has_prev_choice = torch.where(
        need_refill, torch.zeros_like(state.has_prev_choice), state.has_prev_choice
    )

    return replace(
        state,
        h=h,
        phase=torch.where(mask, torch.full_like(state.phase, DECISION), state.phase),
        phase_step=torch.where(
            mask, torch.zeros_like(state.phase_step), state.phase_step
        ),
        trial_idx=torch.where(mask, trial_idx + 1, state.trial_idx),
        session_count=session_count,
        last_outcome_fb=last_outcome_fb,
        action_kernel=action_kernel,
        prev_model_choice=prev_model_choice,
        has_prev_choice=has_prev_choice,
        aux_accum=torch.where(mask, torch.zeros_like(state.aux_accum), state.aux_accum),
        aux_count=torch.where(mask, torch.zeros_like(state.aux_count), state.aux_count),
        cur_contrast=torch.where(mask, new_contrast, state.cur_contrast),
        cur_stim_side=torch.where(mask, new_stim_side, state.cur_stim_side),
        cur_correct_label=torch.where(mask, new_correct_label, state.cur_correct_label),
        cur_block_id=torch.where(mask, new_block_id, state.cur_block_id),
        cur_block_bias=torch.where(mask, new_block_bias, state.cur_block_bias),
        cur_trial_in_block=torch.where(
            mask, new_trial_in_block, state.cur_trial_in_block
        ),
        cur_session_id=torch.where(mask, new_session_id, state.cur_session_id),
    )


def init_state(model, plan, plan_gen, batch_size, device) -> EnvState:
    B = batch_size
    state = EnvState(
        h=torch.zeros(B, model.n_total, device=device),
        phase=torch.full((B,), DECISION, dtype=torch.long, device=device),
        phase_step=torch.zeros(B, dtype=torch.long, device=device),
        trial_idx=torch.zeros(B, dtype=torch.long, device=device),
        session_count=torch.zeros(B, dtype=torch.long, device=device),
        last_outcome_fb=torch.zeros(B, model.n_fb, device=device),
        action_kernel=torch.zeros(B, device=device),
        prev_model_choice=torch.zeros(B, device=device),
        has_prev_choice=torch.zeros(B, dtype=torch.bool, device=device),
        aux_accum=torch.zeros(B, device=device),
        aux_count=torch.zeros(B, device=device),
        cur_contrast=torch.zeros(B, device=device),
        cur_stim_side=torch.zeros(B, device=device),
        cur_correct_label=torch.zeros(B, dtype=torch.long, device=device),
        cur_block_id=torch.zeros(B, dtype=torch.long, device=device),
        cur_block_bias=torch.zeros(B, device=device),
        cur_trial_in_block=torch.zeros(B, dtype=torch.long, device=device),
        cur_session_id=torch.zeros(B, dtype=torch.long, device=device),
    )
    all_rows = torch.ones(B, dtype=torch.bool, device=device)
    return _fetch_new_trials(state, plan, plan_gen, all_rows, device)


def _timing_mask(
    preset: str,
    in_decision: torch.Tensor,
    in_iti: torch.Tensor,
    just_entered_iti: torch.Tensor,
) -> torch.Tensor:
    if preset == "pulse_only":
        return just_entered_iti
    elif preset == "persist_iti":
        return in_iti
    elif preset == "persist_iti_and_decision":
        return in_iti | in_decision
    else:
        raise ValueError(f"unknown timing preset: {preset}")


def step_dt(
    model,
    state: EnvState,
    plan,
    plan_gen,
    device,
    T_max,
    T_iti,
    t_min,
    theta,
    sigma_stim,
    reward_timing,
    kernel_timing,
    t_quiescence,
    use_action_kernel,
    kappa_kernel,
    xi_kernel,
    teacher_forcing_prob=0.0,
    return_hidden=False,
):
    B = state.h.shape[0]
    in_decision = state.phase == DECISION
    in_iti = ~in_decision

    x_stim_t = make_stimulus_input_step(
        state.cur_contrast,
        state.cur_stim_side,
        model.n_stim,
        sigma_stim,
        model.dt,
        device,
    )
    if t_quiescence > 0:
        in_quiescence = in_decision & (state.phase_step < t_quiescence)
        x_stim_t = x_stim_t * (~in_quiescence).float().unsqueeze(-1)
    x_stim_t = x_stim_t * in_decision.float().unsqueeze(-1)

    # just_entered_iti = in_iti & (state.phase_step == 0)
    # # ensure that the action kernel is injected during the entire ITI.

    # fb_active = just_entered_iti | (in_decision & hold_fb_during_stim)
    # x_fb_t = state.last_outcome_fb * fb_active.float().unsqueeze(-1)

    # new construction, to have more flexibility in feedback

    just_entered_iti = in_iti & (state.phase_step == 0)
    reward_active = _timing_mask(reward_timing, in_decision, in_iti, just_entered_iti)
    kernel_active = _timing_mask(kernel_timing, in_decision, in_iti, just_entered_iti)

    reward_component = state.last_outcome_fb[:, 0:1] * reward_active.float().unsqueeze(
        -1
    )
    if model.n_fb == 2:
        kernel_component = state.last_outcome_fb[
            :, 1:2
        ] * kernel_active.float().unsqueeze(-1)
        x_fb_t = torch.cat([reward_component, kernel_component], dim=-1)
    else:
        x_fb_t = reward_component

    h_new = model.step(state.h, x_stim_t, x_fb_t)
    # z, g, r_v, r_m = model.readout(h_new)
    z, g, r_v, r_c, r_p = model.readout(h_new)

    accumulate_aux = in_decision & (state.phase_step >= t_quiescence)
    aux_accum = state.aux_accum + g * accumulate_aux.float()
    aux_count = state.aux_count + accumulate_aux.float()

    y = z[:, 1] - z[:, 0]
    crossed = in_decision & (state.phase_step >= t_min) & (y.abs() >= theta)
    timed_out_now = in_decision & (~crossed) & (state.phase_step >= T_max - 1)
    resolved_now = crossed | timed_out_now

    model_choice_full = (z[:, 1] > z[:, 0]).long()
    model_reward_full = (model_choice_full == state.cur_correct_label).float()
    tf_mask = torch.rand(B, device=device) < teacher_forcing_prob
    fb_reward_full = torch.where(
        tf_mask, torch.ones(B, device=device), model_reward_full
    )

    if use_action_kernel:
        choice_signed_full = model_choice_full.float() * 2 - 1
        new_kernel_full = (
            1 - xi_kernel
        ) * state.action_kernel + xi_kernel * choice_signed_full
        kernel_term_full = kappa_kernel * new_kernel_full
    else:
        new_kernel_full, kernel_term_full = None, None

    new_fb_vec_full = make_feedback_vector(
        fb_reward_full, device, model.n_fb, kernel_term_full
    )

    result = None
    if resolved_now.any():
        ridx = resolved_now.nonzero(as_tuple=True)[0]
        aux_pred = aux_accum[ridx] / aux_count[ridx].clamp(min=1)
        result = TrialResult(
            z_c=z[ridx],
            aux_pred=aux_pred,
            target_stim=state.cur_stim_side[ridx] * state.cur_contrast[ridx],
            correct_label=state.cur_correct_label[ridx],
            choice=model_choice_full[ridx],
            reward=model_reward_full[ridx],
            rt=state.phase_step[ridx].clone(),
            timed_out=timed_out_now[ridx],
            block_id=state.cur_block_id[ridx],
            block_bias=state.cur_block_bias[ridx],
            trial_in_block=state.cur_trial_in_block[ridx],
            contrast=state.cur_contrast[ridx],
            stim_side=state.cur_stim_side[ridx],
            prev_model_choice=state.prev_model_choice[ridx].clone(),
            has_prev_choice=state.has_prev_choice[ridx].clone(),
            session_id=state.cur_session_id[ridx],
            action_kernel=(
                state.action_kernel[ridx].clone()
                if use_action_kernel
                else torch.full((len(ridx),), float("nan"), device=device)
            ),
            batch_idx=ridx,
        )

    last_outcome_fb = torch.where(
        resolved_now.unsqueeze(-1), new_fb_vec_full, state.last_outcome_fb
    )
    action_kernel = (
        torch.where(resolved_now, new_kernel_full, state.action_kernel)
        if use_action_kernel
        else state.action_kernel
    )
    prev_model_choice = torch.where(
        resolved_now, model_choice_full.float(), state.prev_model_choice
    )
    has_prev_choice = state.has_prev_choice | resolved_now
    phase = torch.where(resolved_now, torch.full_like(state.phase, ITI), state.phase)
    phase_step = torch.where(
        resolved_now, torch.zeros_like(state.phase_step), state.phase_step + 1
    )

    new_state = replace(
        state,
        h=h_new,
        phase=phase,
        phase_step=phase_step,
        last_outcome_fb=last_outcome_fb,
        action_kernel=action_kernel,
        prev_model_choice=prev_model_choice,
        has_prev_choice=has_prev_choice,
        aux_accum=aux_accum,
        aux_count=aux_count,
    )

    iti_finished = in_iti & (~just_entered_iti) & (new_state.phase_step >= T_iti)
    if iti_finished.any():
        new_state = _fetch_new_trials(new_state, plan, plan_gen, iti_finished, device)

    # if return_hidden:
    #     return new_state, result, r_v.detach(), r_m.detach(), x_stim_t.detach()

    if return_hidden:
        return (
            new_state,
            result,
            r_v.detach(),
            r_c.detach(),
            r_p.detach(),
            x_stim_t.detach(),
        )

    return new_state, result


class AsyncClosedLoopEnv:
    """Thin stateful convenience wrapper around the pure functions above --
    keeps train.py's call sites (env.reset(), env.step_dt(...), env.h,
    detach-at-chunk-boundary) almost unchanged, while every actual state
    transition underneath is a pure, out-of-place reconstruction."""

    def __init__(
        self,
        model,
        plan_gen,
        batch_size,
        device,
        T_max,
        T_iti,
        t_min,
        theta,
        sigma_stim,
        reward_timing="pulse_only",
        kernel_timing="persist_iti",
        t_quiescence=0,
        kappa_kernel=0.0,
        xi_kernel=0.2,
    ):
        self.model, self.plan_gen = model, plan_gen
        self.B, self.device = batch_size, device
        self.T_max, self.T_iti = T_max, T_iti
        self.t_min, self.theta, self.sigma_stim = t_min, theta, sigma_stim
        self.reward_timing = reward_timing
        self.kernel_timing = kernel_timing
        self.t_quiescence = t_quiescence
        self.use_action_kernel = kappa_kernel != 0.0
        self.kappa_kernel, self.xi_kernel = kappa_kernel, xi_kernel
        self.state: Optional[EnvState] = None
        self.plan = None

    def reset(self):
        self.plan = self.plan_gen.generate(self.B)
        self.state = init_state(
            self.model, self.plan, self.plan_gen, self.B, self.device
        )

    def zero_hidden_state(self):
        self.state = replace(self.state, h=torch.zeros_like(self.state.h))

    def step_dt(
        self,
        teacher_forcing_prob: float = 0.0,
        return_hidden=False,
    ):
        out = step_dt(
            self.model,
            self.state,
            self.plan,
            self.plan_gen,
            self.device,
            self.T_max,
            self.T_iti,
            self.t_min,
            self.theta,
            self.sigma_stim,
            self.reward_timing,
            self.kernel_timing,
            self.t_quiescence,
            self.use_action_kernel,
            self.kappa_kernel,
            self.xi_kernel,
            teacher_forcing_prob,
            return_hidden,
        )
        if return_hidden:
            self.state, result, r_v, r_c, r_p, x_stim = out
            return result, r_v, r_c, r_p, x_stim
        else:
            self.state, result = out
            return result

    def detach_state(self):
        self.state = self.state.detach()

    @property
    def h(self):
        return self.state.h
