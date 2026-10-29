import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

from .utils import get_nonlinearity


class ModularCTRNN(nn.Module):
    """Continuous-time leaky RNN, Euler-discretized, two modules (VISp, MOs)"""

    def __init__(
        self,
        n_visp: int,
        n_mos: int,
        n_stim: int,
        n_fb: int,
        rank_ff: int,
        dt: float,
        tau_visp_range,
        tau_mos_range,
        learnable_tau: bool,
        rec_gain: float,
        ff_gain: float,
        fb_gain: float,
        in_gain: float,
        sigma_baseline: float,
        nonlinearity: str = "relu",
        fb_mode: str = "dense",
        fb_sparsity: float = 0.5,
        stim_fb_gain: float = 0.1,
    ):
        super().__init__()
        self.n_visp, self.n_mos = n_visp, n_mos
        self.n_total = n_visp + n_mos
        self.n_stim, self.n_fb = n_stim, n_fb
        self.dt = dt
        self.sigma_baseline = sigma_baseline  # QUE: What is sigma baseline doing?
        self.phi = get_nonlinearity(nonlinearity)
        self.fb_mode = fb_mode
        self.fb_sparsity = fb_sparsity

        tau_v = torch.empty(n_visp).uniform_(*tau_visp_range)
        tau_m = torch.empty(n_mos).uniform_(*tau_mos_range)
        log_tau_init = torch.log(torch.cat([tau_v, tau_m]))
        if learnable_tau:
            self.log_tau = nn.Parameter(log_tau_init)
        else:
            self.register_buffer("log_tau", log_tau_init)

        self.W_vv = nn.Parameter(
            rec_gain * torch.randn(n_visp, n_visp) / math.sqrt(n_visp)
        )
        self.W_mm = nn.Parameter(
            rec_gain * torch.randn(n_mos, n_mos) / math.sqrt(n_mos)
        )

        # feedforward VISp->MOs: hard low-rank factorization (never a free full matrix)
        self.U_ff = nn.Parameter(
            ff_gain * torch.randn(n_mos, rank_ff) / math.sqrt(rank_ff)
        )
        self.V_ff = nn.Parameter(
            ff_gain * torch.randn(rank_ff, n_visp) / math.sqrt(rank_ff)
        )

        # feedback MOs->VISp: free, dense
        # self.W_fb = nn.Parameter(
        #     fb_gain * torch.randn(n_visp, n_mos) / math.sqrt(n_mos)
        # )

        # set sparsity
        if self.fb_mode == "lowrank":
            self.U_fb = nn.Parameter(
                fb_gain * torch.randn(rank_ff, n_mos) / math.sqrt(rank_ff)
            )
            self.V_fb = nn.Parameter(
                fb_gain * torch.randn(n_visp, rank_ff) / math.sqrt(rank_ff)
            )
        else:
            self.W_fb = nn.Parameter(
                fb_gain * torch.randn(n_visp, n_mos) / math.sqrt(n_mos)
            )
            if self.fb_mode == "dense":
                self.register_buffer("W_fb_mask", torch.ones(n_visp, n_mos))
            elif self.fb_mode == "zero":
                self.register_buffer("W_fb_mask", torch.zeros(n_visp, n_mos))
            elif self.fb_mode == "sparse":
                mask = (
                    torch.rand(n_visp, n_mos) < self.fb_sparsity
                ).float()  # fb_sparsity = fraction KEPT nonzero
                self.register_buffer("W_fb_mask", mask)
            else:
                raise ValueError(f"unknown fb_mode: {fb_mode}")

        # input weights: hard block structure -- stim -> VISp only

        self.W_in_stim_visp = nn.Parameter(
            in_gain * torch.randn(n_visp, n_stim) / math.sqrt(n_stim)
        )
        # reduce this by stim_fb_gain

        self.W_in_fb_visp = nn.Parameter(
            stim_fb_gain * torch.randn(n_visp, n_fb) / math.sqrt(n_fb)
        )

        self.W_in_fb_mos = nn.Parameter(
            in_gain * torch.randn(n_mos, n_fb) / math.sqrt(n_fb)
        )

        self.bias = nn.Parameter(torch.zeros(self.n_total))
        self.W_choice = nn.Linear(n_mos, 2, bias=True)
        self.w_aux = nn.Linear(n_visp, 1, bias=True)

    @property
    def tau(self):
        return torch.exp(self.log_tau)

    def W_ff(self):
        return self.U_ff @ self.V_ff  # (n_mos, n_visp), rank <= rank_ff by construction

    def W_mv(self):
        if self.fb_mode == "lowrank":
            return self.V_fb @ self.U_fb
        else:
            return self.W_fb * self.W_fb_mask

    def effective_rank_ff(self) -> float:
        with torch.no_grad():
            s = torch.linalg.svdvals(self.W_ff())
            p = s / s.sum()
            p = p[p > 0]
            return torch.exp(-(p * p.log()).sum()).item()

    def step(self, h, x_stim, x_fb, noise_scale: float = 1.0):
        B = h.shape[0]
        r = self.phi(h)
        r_v, r_m = r[:, : self.n_visp], r[:, self.n_visp :]

        rec_v = r_v @ self.W_vv.T + r_m @ self.W_mv().T
        rec_m = r_m @ self.W_mm.T + r_v @ self.W_ff().T
        rec = torch.cat([rec_v, rec_m], dim=1)

        inp_v = x_stim @ self.W_in_stim_visp.T + x_fb @ self.W_in_fb_visp.T
        inp_m = x_fb @ self.W_in_fb_mos.T
        inp = torch.cat([inp_v, inp_m], dim=1)

        alpha = repeat(self.dt / self.tau, "n -> b n", b=B)
        dh = alpha * (-h + rec + inp + self.bias.unsqueeze(0))

        noise_std = self.sigma_baseline * torch.sqrt(2 * alpha)
        noise = noise_scale * noise_std * torch.randn(B, self.n_total, device=h.device)

        return h + dh + noise

    def readout(self, h):
        r = self.phi(h)
        r_v, r_m = r[:, : self.n_visp], r[:, self.n_visp :]
        z = self.W_choice(r_m)
        g = self.w_aux(r_v).squeeze(-1)
        return z, g, r_v, r_m
