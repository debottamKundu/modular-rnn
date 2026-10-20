from dataclasses import dataclass
from typing import Optional
import torch

from inputs import make_stimulus_input, make_feedback_vector


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
    session_id: (
        torch.Tensor
    )  # (batch_idx, refill_count) identity, for grouping trials into sessions post-hoc
    full_r_v: Optional[torch.Tensor] = None
    full_r_m: Optional[torch.Tensor] = None
    full_x_stim: Optional[torch.Tensor] = None


class ClosedLoopEnv:
    def __init__(
        self,
        model,
        plan_gen: "SessionPlanGenerator",
        batch_size: int,
        device,
        T_stim: int,
        T_resp: int,
        T_iti: int,
        t_min: int,
        theta: float,
        sigma_stim: float,
        hold_fb_during_stim: bool,
        t_quiescence: int = 0,
    ):
        self.model, self.plan_gen = model, plan_gen
        self.B, self.device = batch_size, device
        self.T_stim, self.T_resp, self.T_iti = T_stim, T_resp, T_iti
        self.t_min, self.theta, self.sigma_stim = t_min, theta, sigma_stim
        self.hold_fb_during_stim = hold_fb_during_stim
        self.t_quiescence = t_quiescence
        self.h = None

    def reset(self):
        self.h = torch.zeros(self.B, self.model.n_total, device=self.device)
        self._prev_reward = torch.zeros(self.B, device=self.device)
        self._prev_model_choice = torch.zeros(self.B, device=self.device)
        self._has_prev_choice = torch.zeros(
            self.B, dtype=torch.bool, device=self.device
        )
        self.trial_idx = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.session_count = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.plan = self.plan_gen.generate(self.B)

    def _reset_state(self, done_mask):
        with torch.no_grad():
            self.h[done_mask] = 0.0
            self._prev_reward[done_mask] = 0.0
            self._prev_model_choice[done_mask] = 0.0
            self._has_prev_choice[done_mask] = False
        self.session_count[done_mask] += 1

    def run_trial(
        self, teacher_forcing_prob: float = 0.0, store_full_trace: bool = False
    ):
        B, device, model = self.B, self.device, self.model

        # refill any sessions fully consumed, BEFORE reading this trial's data --
        # this is what makes session length variable/asynchronous across the batch
        done_mask = self.trial_idx >= self.plan["total_trials"]
        if done_mask.any():
            self.plan_gen.refill(self.plan, done_mask)
            self.trial_idx = torch.where(
                done_mask, torch.zeros_like(self.trial_idx), self.trial_idx
            )
            self._reset_state(done_mask)

        b_idx = torch.arange(B, device=device)
        t_idx = self.trial_idx
        contrast = self.plan["contrast"][b_idx, t_idx]
        stim_side = self.plan["stim_side"][b_idx, t_idx]
        correct_label = self.plan["correct_label"][b_idx, t_idx]
        block_id = self.plan["block_id"][b_idx, t_idx]
        block_bias = self.plan["block_bias"][b_idx, t_idx]
        trial_in_block = self.plan["trial_in_block"][b_idx, t_idx]
        session_id = (
            b_idx * 10_000 + self.session_count
        )  # unique id per (batch slot, refill)

        self.trial_idx = self.trial_idx + 1

        x_stim_seq = make_stimulus_input(
            contrast,
            stim_side,
            self.T_stim,
            model.dt,
            model.n_stim,
            self.sigma_stim,
            device,
        )
        if self.t_quiescence > 0:
            x_stim_seq[:, : self.t_quiescence, :] = (
                0.0  # optional pre-stimulus quiescence period
            )

        fb_const = make_feedback_vector(self._prev_reward, device, model.n_fb)
        zero_fb = torch.zeros(B, model.n_fb, device=device)
        zero_stim = torch.zeros(B, model.n_stim, device=device)

        z_hist, aux_hist, x_stim_hist = [], [], []
        r_v_hist, r_m_hist = ([], []) if store_full_trace else (None, None)

        for t in range(self.T_stim):
            x_fb_t = fb_const if self.hold_fb_during_stim else zero_fb
            self.h = model.step(self.h, x_stim_seq[:, t], x_fb_t)
            z, g, r_v, r_m = model.readout(self.h)
            z_hist.append(z)
            aux_hist.append(g)
            if store_full_trace:
                r_v_hist.append(r_v)
                r_m_hist.append(r_m)
                x_stim_hist.append(x_stim_seq[:, t])

        committed = torch.zeros(B, dtype=torch.bool, device=device)
        commit_idx = torch.full(
            (B,), self.T_stim + self.T_resp - 1, dtype=torch.long, device=device
        )

        for t in range(self.T_resp):
            x_fb_t = fb_const if self.hold_fb_during_stim else zero_fb
            self.h = model.step(self.h, zero_stim, x_fb_t)
            z, g, r_v, r_m = model.readout(self.h)
            z_hist.append(z)
            if store_full_trace:
                r_v_hist.append(r_v)
                r_m_hist.append(r_m)
                x_stim_hist.append(zero_stim)

            y = z[:, 1] - z[:, 0]
            cross = (~committed) & (t >= self.t_min) & (y.abs() >= self.theta)
            commit_idx = torch.where(
                cross, torch.full_like(commit_idx, self.T_stim + t), commit_idx
            )
            committed = committed | cross

        timed_out = ~committed

        Z = torch.stack(z_hist, dim=1)
        Aux = torch.stack(aux_hist, dim=1)
        z_c = Z[torch.arange(B, device=device), commit_idx]
        aux_pred = Aux.mean(dim=1)

        with torch.no_grad():
            model_choice = (z_c[:, 1] > z_c[:, 0]).long()
            model_reward = (model_choice == correct_label).float()
            tf_mask = torch.rand(B, device=device) < teacher_forcing_prob
            fb_reward = torch.where(tf_mask, torch.ones(B, device=device), model_reward)

        prev_model_choice = self._prev_model_choice.clone()
        has_prev_choice = self._has_prev_choice.clone()

        fb_vec = make_feedback_vector(fb_reward, device, model.n_fb)
        for t in range(self.T_iti):
            x_fb_t = fb_vec if t == 0 else zero_fb
            self.h = model.step(self.h, zero_stim, x_fb_t)
            if store_full_trace:
                _, _, r_v, r_m = model.readout(self.h)
                r_v_hist.append(r_v)
                r_m_hist.append(r_m)
                x_stim_hist.append(zero_stim)

        self._prev_reward = fb_reward
        self._prev_model_choice = model_choice.float()
        self._has_prev_choice = torch.ones(B, dtype=torch.bool, device=device)

        return TrialResult(
            z_c=z_c,
            aux_pred=aux_pred,
            target_stim=stim_side.float() * contrast,
            correct_label=correct_label,
            choice=model_choice,
            reward=model_reward,
            rt=(commit_idx - self.T_stim),
            timed_out=timed_out,
            block_id=block_id,
            block_bias=block_bias,
            trial_in_block=trial_in_block,
            contrast=contrast,
            stim_side=stim_side,
            prev_model_choice=prev_model_choice,
            has_prev_choice=has_prev_choice,
            session_id=session_id,
            full_r_v=torch.stack(r_v_hist, dim=1) if store_full_trace else None,
            full_r_m=torch.stack(r_m_hist, dim=1) if store_full_trace else None,
            full_x_stim=torch.stack(x_stim_hist, dim=1) if store_full_trace else None,
        )
