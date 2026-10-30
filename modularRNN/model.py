import math
import torch
import torch.nn as nn
from einops import repeat

from .utils import get_nonlinearity


class ConnectionBlock(nn.Module):
    """A single directed connectivity block FROM `src` TO `dst`:
    - "dense":   free (n_dst, n_src) matrix
    - "sparse":  free matrix with a FIXED random binary mask (not trained)
    - "zero":    hard-wired to zero, no trainable parameters at all
    - "lowrank": hard factorization W = U @ V; rank is guaranteed by
                 construction, never merely encouraged
    """

    def __init__(self, n_dst, n_src, mode="dense", gain=1.0, sparsity=1.0, rank=3):
        super().__init__()
        self.mode = mode
        self.n_dst, self.n_src = n_dst, n_src

        if mode == "zero":
            self.register_buffer("_zero", torch.zeros(n_dst, n_src))
        elif mode == "lowrank":
            assert rank > 0, "lowrank mode requires rank > 0"
            self.U = nn.Parameter(gain * torch.randn(n_dst, rank) / math.sqrt(rank))
            self.V = nn.Parameter(gain * torch.randn(rank, n_src) / math.sqrt(rank))
        elif mode in ("dense", "sparse"):
            self.W = nn.Parameter(
                gain * torch.randn(n_dst, n_src) / math.sqrt(max(n_src, 1))
            )
            if mode == "sparse":
                assert 0.0 < sparsity < 1.0, "sparse mode requires 0 < sparsity < 1"
                mask = (torch.rand(n_dst, n_src) < sparsity).float()
            else:
                mask = torch.ones(n_dst, n_src)
            self.register_buffer("mask", mask)
        else:
            raise ValueError(f"unknown connection mode: {mode}")

    def forward(self):
        if self.mode == "zero":
            return self._zero
        if self.mode == "lowrank":
            return self.U @ self.V
        return self.W * self.mask

    def effective_rank(self):
        with torch.no_grad():
            W = self.forward()
            if torch.count_nonzero(W) == 0:
                return 0.0
            s = torch.linalg.svdvals(W)
            total = s.sum()
            if total <= 0:
                return 0.0
            p = s / total
            p = p[p > 0]
            return torch.exp(-(p * p.log()).sum()).item()

    def norm(self):
        with torch.no_grad():
            return self.forward().norm().item()


class ModularCTRNN3(nn.Module):
    """
    Three hard-structured modules:
      V (visual)  -- fast tau, receives stimulus input (hard-wired, exclusive)
      C (choice)  -- medium tau, produces the choice readout
      P (pfc)     -- slow tau, receives reward/action-kernel feedback
                     (hard-wired, exclusive), intended to be the locus of
                     any prior/history representation

    Only Wvv/Wcc/Wpp (within-module recurrence) are always free and dense.
    The six cross-module blocks are each independently configurable.
    """

    def __init__(
        self,
        n_v: int,
        n_c: int,
        n_p: int,
        n_stim: int,
        n_fb: int,
        dt: float,
        tau_v_range,
        tau_c_range,
        tau_p_range,
        learnable_tau: bool,
        rec_gain: float,
        in_gain: float,
        sigma_baseline: float,
        nonlinearity: str,
        conn_specs: dict,  # {(dst, src): dict(mode=, gain=, sparsity=, rank=)}
    ):
        super().__init__()
        self.n_v, self.n_c, self.n_p = n_v, n_c, n_p
        self.sizes = {"v": n_v, "c": n_c, "p": n_p}
        self.n_total = n_v + n_c + n_p
        self.n_stim, self.n_fb = n_stim, n_fb
        self.dt = dt
        self.sigma_baseline = sigma_baseline
        self.phi = get_nonlinearity(nonlinearity)

        tau_v = torch.empty(n_v).uniform_(*tau_v_range)
        tau_c = torch.empty(n_c).uniform_(*tau_c_range)
        tau_p = torch.empty(n_p).uniform_(*tau_p_range)
        log_tau_init = torch.log(torch.cat([tau_v, tau_c, tau_p]))
        if learnable_tau:
            self.log_tau = nn.Parameter(log_tau_init)
        else:
            self.register_buffer("log_tau", log_tau_init)

        self.W_vv = nn.Parameter(rec_gain * torch.randn(n_v, n_v) / math.sqrt(n_v))
        self.W_cc = nn.Parameter(rec_gain * torch.randn(n_c, n_c) / math.sqrt(n_c))
        self.W_pp = nn.Parameter(rec_gain * torch.randn(n_p, n_p) / math.sqrt(n_p))

        self.connections = nn.ModuleDict()
        for dst, src in [
            ("c", "v"),
            ("v", "c"),
            ("p", "v"),
            ("v", "p"),
            ("p", "c"),
            ("c", "p"),
        ]:
            spec = conn_specs[(dst, src)]
            self.connections[f"{dst}_from_{src}"] = ConnectionBlock(
                n_dst=self.sizes[dst],
                n_src=self.sizes[src],
                mode=spec.get("mode", "dense"),
                gain=spec.get("gain", 1.0),
                sparsity=spec.get("sparsity", 1.0),
                rank=spec.get("rank", 3),
            )

        # hard input constraints -- never relaxed
        self.W_in_stim_v = nn.Parameter(
            in_gain * torch.randn(n_v, n_stim) / math.sqrt(n_stim)
        )
        self.W_in_fb_p = nn.Parameter(
            in_gain * torch.randn(n_p, n_fb) / math.sqrt(n_fb)
        )

        self.bias = nn.Parameter(torch.zeros(self.n_total))
        self.W_choice = nn.Linear(n_c, 2, bias=True)
        self.w_aux = nn.Linear(n_v, 1, bias=True)

    @property
    def tau(self):
        return torch.exp(self.log_tau)

    def conn(self, dst, src):
        return self.connections[f"{dst}_from_{src}"]()

    def effective_rank(self, dst, src):
        return self.connections[f"{dst}_from_{src}"].effective_rank()

    def conn_norm(self, dst, src):
        return self.connections[f"{dst}_from_{src}"].norm()

    def _split(self, flat):
        v = flat[:, : self.n_v]
        c = flat[:, self.n_v : self.n_v + self.n_c]
        p = flat[:, self.n_v + self.n_c :]
        return v, c, p

    def step(self, h, x_stim, x_fb, noise_scale: float = 1.0):
        B = h.shape[0]
        r = self.phi(h)
        r_v, r_c, r_p = self._split(r)

        rec_v = (
            r_v @ self.W_vv.T
            + r_c @ self.conn("v", "c").T
            + r_p @ self.conn("v", "p").T
        )
        rec_c = (
            r_c @ self.W_cc.T
            + r_v @ self.conn("c", "v").T
            + r_p @ self.conn("c", "p").T
        )
        rec_p = (
            r_p @ self.W_pp.T
            + r_v @ self.conn("p", "v").T
            + r_c @ self.conn("p", "c").T
        )
        rec = torch.cat([rec_v, rec_c, rec_p], dim=1)

        inp_v = x_stim @ self.W_in_stim_v.T
        inp_c = torch.zeros(B, self.n_c, device=h.device)
        inp_p = x_fb @ self.W_in_fb_p.T
        inp = torch.cat([inp_v, inp_c, inp_p], dim=1)

        alpha = repeat(self.dt / self.tau, "n -> b n", b=B)
        dh = alpha * (-h + rec + inp + self.bias.unsqueeze(0))
        noise_std = self.sigma_baseline * torch.sqrt(2 * alpha)
        noise = noise_scale * noise_std * torch.randn(B, self.n_total, device=h.device)
        return h + dh + noise

    def readout(self, h):
        r = self.phi(h)
        r_v, r_c, r_p = self._split(r)
        z = self.W_choice(r_c)
        g = self.w_aux(r_v).squeeze(-1)
        return z, g, r_v, r_c, r_p
