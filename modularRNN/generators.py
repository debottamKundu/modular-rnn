import torch


class SessionPlanGenerator:
    """
    Batched, padded generator for the session
    (block lengths, bias schedule, per-trial contrast/side/label),
    vectorized across a batch with variable total trial counts per session
    (via a fixed worst-case ceiling + validity mask + per-sample in-place refill).
    """

    def __init__(
        self,
        n_blocks,
        n_unbiased_blocks,
        bias_high,
        block_len_mean,
        block_len_min,
        block_len_max,
        possible_contrasts,
        device,
    ):
        self.n_blocks = n_blocks
        self.n_unbiased = n_unbiased_blocks
        self.n_biased = n_blocks - n_unbiased_blocks
        self.bias_high = bias_high
        self.len_mean = block_len_mean
        self.len_min = block_len_min
        self.len_max = block_len_max
        self.contrasts = torch.tensor(possible_contrasts, device=device)
        self.device = device
        self.max_trials = n_blocks * block_len_max  # fixed worst-case ceiling

    def _sample_block_lengths(self, n):
        p = 1.0 / self.len_mean
        shape = (n, self.n_blocks)
        lengths = torch.full(shape, -1, dtype=torch.long, device=self.device)
        unresolved = torch.ones(shape, dtype=torch.bool, device=self.device)
        while unresolved.any():
            draws = (
                torch.distributions.Geometric(probs=p)
                .sample(shape)
                .to(self.device)
                .long()
                + 1
            )
            valid = (draws >= self.len_min) & (draws <= self.len_max)
            accept = unresolved & valid
            lengths[accept] = draws[accept]
            unresolved &= ~accept
        return lengths

    def _sample_bias_seq(self, n):
        bias_seq = torch.full((n, self.n_blocks), 0.5, device=self.device)
        first_right = torch.rand(n, device=self.device) < 0.5
        biased_right = torch.where(
            first_right,
            torch.full((n,), self.bias_high, device=self.device),
            torch.full((n,), 1 - self.bias_high, device=self.device),
        )
        for j in range(self.n_biased):
            col = self.n_unbiased + j
            bias_seq[:, col] = biased_right if j % 2 == 0 else 1 - biased_right
        bias_seq = torch.round(bias_seq * 1000) / 1000  # snap to 3 decimal places
        return bias_seq

    def generate(self, n):
        """Returns a dict of (n, max_trials) padded tensors + per-sample total_trials."""
        block_lengths = self._sample_block_lengths(n)
        bias_seq = self._sample_bias_seq(n)
        total_trials = block_lengths.sum(dim=1)

        block_starts = torch.cumsum(block_lengths, dim=1) - block_lengths
        block_ends = torch.cumsum(block_lengths, dim=1)

        trial_idx = (
            torch.arange(self.max_trials, device=self.device).unsqueeze(0).expand(n, -1)
        ).contiguous()
        valid_mask = trial_idx < total_trials.unsqueeze(1)

        block_id = torch.searchsorted(block_ends, trial_idx, right=True)
        block_id = torch.clamp(block_id, max=self.n_blocks - 1)
        trial_bias = torch.gather(bias_seq, 1, block_id)

        n_c = len(self.contrasts)
        contrast_idx = torch.randint(0, n_c, (n, self.max_trials), device=self.device)
        contrast = self.contrasts[contrast_idx]

        # side always drawn from block bias, independent of contrast level
        # (zero-contrast trials still have a true rewarded side, matching IBL design)
        is_right = torch.bernoulli(trial_bias).bool()
        stim_side = torch.where(
            is_right,
            torch.ones(n, self.max_trials, device=self.device),
            -torch.ones(n, self.max_trials, device=self.device),
        )
        correct_label = is_right.long()
        trial_in_block = trial_idx - torch.gather(block_starts, 1, block_id)

        return {
            "contrast": contrast,
            "stim_side": stim_side,
            "correct_label": correct_label,
            "block_id": block_id,
            "block_bias": trial_bias,
            "trial_in_block": trial_in_block,
            "valid_mask": valid_mask,
            "total_trials": total_trials,
        }

    def refill(self, plan, done_mask):
        idx = done_mask.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            return
        sub = self.generate(len(idx))
        for key in (
            "contrast",
            "stim_side",
            "correct_label",
            "block_id",
            "block_bias",
            "trial_in_block",
            "valid_mask",
        ):
            plan[key][idx] = sub[key]
        plan["total_trials"][idx] = sub["total_trials"]
