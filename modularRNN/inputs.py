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


# def make_feedback_vector(reward, device, n_fb=2, kernel_bias=None):
#     # choice_enc = choice.float() * 2 - 1
#     reward_enc = reward.float() * 2 - 1
#     x_fb = reward_enc.unsqueeze(-1)
#     # x_fb = torch.stack([choice_enc, reward_enc], dim=-1)
#     if n_fb > 1:
#         # x_fb = torch.cat(
#         #     [x_fb, torch.zeros(choice.shape[0], n_fb - 2, device=device)], dim=-1
#         # )
#         x_fb = torch.cat(
#             [x_fb, torch.zeros(reward.shape[0], n_fb - 1, device=device)], dim=-1
#         )
#     return x_fb


def make_feedback_vector(reward, device, n_fb, kernel_bias=None):
    """
    reward: (B,) tensor, values in {0,1}
    kernel_bias: optional (B,) tensor, already scaled by kappa_kernel.
        If None -> feedback is reward-only, n_fb must be 1.
        If given -> feedback is [reward, kernel_bias], n_fb must be 2.
    device:
    """
    reward_enc = (reward.float() * 2 - 1).to(device)

    if kernel_bias is None:
        assert n_fb == 1, f"n_fb={n_fb} but no kernel_bias provided"
        return reward_enc.unsqueeze(-1)

    assert n_fb == 2, f"n_fb={n_fb} but kernel_bias was provided"
    kernel_bias = kernel_bias.to(device)
    return torch.cat([reward_enc.unsqueeze(-1), kernel_bias.unsqueeze(-1)], dim=-1)


def make_stimulus_input_step(contrast, stim_side, n_stim, sigma_stim, dt, device):
    """Per-row, single-timestep version , stimulus is generated one dt at a time."""
    B = contrast.shape[0]
    mu_left = torch.clamp(-stim_side, min=0) * contrast
    mu_right = torch.clamp(stim_side, min=0) * contrast
    mu = torch.stack([mu_left, mu_right], dim=-1)
    if n_stim > 2:
        mu = torch.cat([mu, torch.zeros(B, n_stim - 2, device=device)], dim=-1)
    noise = torch.randn(B, n_stim, device=device) * (sigma_stim / dt**0.5)
    return mu + noise
