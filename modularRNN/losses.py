import torch
import torch.nn.functional as F


def gather_window_logits(Z, commit_idx, window=5):
    """Average logits over [commit_idx - window + 1, commit_idx], clipped at 0,
    rather than a single timestep. Reduces variance of the choice-loss
    gradient without reintroducing early-decision-phase ambiguity penalties."""
    B, T, _ = Z.shape
    idx_range = torch.arange(T, device=Z.device).unsqueeze(0)  # (1, T)
    commit_idx_exp = commit_idx.unsqueeze(1)  # (B, 1)
    window_mask = (idx_range <= commit_idx_exp) & (idx_range > commit_idx_exp - window)
    window_mask = window_mask.float().unsqueeze(-1)  # (B, T, 1)
    z_window = (Z * window_mask).sum(dim=1) / window_mask.sum(dim=1).clamp(min=1)
    return z_window


def compute_losses(results):
    z_c = torch.cat([r.z_c for r in results], dim=0)
    # z_c = gather_window_logits(Z, commit_idx, window=5)
    labels = torch.cat([r.correct_label for r in results], dim=0)
    aux_pred = torch.cat([r.aux_pred for r in results], dim=0)
    target_stim = torch.cat([r.target_stim for r in results], dim=0)

    L_choice = F.cross_entropy(z_c, labels)
    L_aux = F.mse_loss(aux_pred, target_stim)

    # switching cost: decision-stage only (uses z_c, the commit-time logits),
    # not accumulated over dt-timesteps. Target = model's own previous choice,
    # not the (possibly teacher-forced) feedback the network actually received.
    prev_choice = torch.cat([r.prev_model_choice for r in results], dim=0).long()
    has_prev = torch.cat([r.has_prev_choice for r in results], dim=0)
    if has_prev.any():
        L_switch = F.cross_entropy(z_c[has_prev], prev_choice[has_prev])
    else:
        L_switch = torch.zeros((), device=z_c.device)

    return L_choice, L_aux, L_switch
