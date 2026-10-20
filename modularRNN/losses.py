import torch
import torch.nn.functional as F


def compute_losses(results):
    z_c = torch.cat([r.z_c for r in results], dim=0)
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
