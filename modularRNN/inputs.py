import torch

# HACK: Should we not generate all of this beforehand?


def make_stimulus_input(contrast, stim_side, T, dt, n_stim, sigma_stim, device):
    """sigma_stim: config-controlled noise amplitude (env.sigma_stim)."""
    B = contrast.shape[0]
    mu_left = torch.clamp(-stim_side, min=0) * contrast
    mu_right = torch.clamp(stim_side, min=0) * contrast
    mu = torch.stack([mu_left, mu_right], dim=-1)
    if n_stim > 2:
        mu = torch.cat([mu, torch.zeros(B, n_stim - 2, device=device)], dim=-1)
    noise = torch.randn(B, T, n_stim, device=device) * (sigma_stim / (dt**0.5))
    return mu.unsqueeze(1).expand(B, T, n_stim) + noise


def make_feedback_vector(reward, device, n_fb=2):
    # choice_enc = choice.float() * 2 - 1
    reward_enc = reward.float() * 2 - 1
    x_fb = reward_enc.unsqueeze(-1)
    # x_fb = torch.stack([choice_enc, reward_enc], dim=-1)
    if n_fb > 1:
        # x_fb = torch.cat(
        #     [x_fb, torch.zeros(choice.shape[0], n_fb - 2, device=device)], dim=-1
        # )
        x_fb = torch.cat(
            [x_fb, torch.zeros(reward.shape[0], n_fb - 1, device=device)], dim=-1
        )
    return x_fb
