import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def congruency_masks(stim_side: torch.Tensor, block_bias: torch.Tensor):

    is_biased = block_bias != 0.5
    block_right = block_bias > 0.5
    stim_right = stim_side > 0
    congruent = (block_right == stim_right) & is_biased
    incongruent = (~congruent) & is_biased
    return congruent, incongruent, is_biased


def _outcome_masks(results_or_cat):
    """Returns (committed_correct, committed_incorrect, no_response) boolean masks.
    No-response trials are kept strictly separate from incorrect ones."""
    choice = results_or_cat["choice"]
    correct_label = results_or_cat["correct_label"]
    timed_out = results_or_cat["timed_out"].bool()

    committed = ~timed_out
    correct = committed & (choice == correct_label)
    incorrect = committed & (choice != correct_label)
    return correct, incorrect, timed_out


def congruency_accuracy_scalars(results, contrast_levels):
    contrast = torch.cat([r.contrast for r in results])
    stim_side = torch.cat([r.stim_side for r in results])
    block_bias = torch.cat([r.block_bias for r in results])
    choice = torch.cat([r.choice for r in results])
    correct_label = torch.cat([r.correct_label for r in results])
    timed_out = torch.cat([r.timed_out for r in results])

    cat = {"choice": choice, "correct_label": correct_label, "timed_out": timed_out}
    correct, incorrect, no_resp = _outcome_masks(cat)
    congruent, incongruent, is_biased = congruency_masks(stim_side, block_bias)

    def acc_given(mask):
        committed_mask = mask & (correct | incorrect)
        n = committed_mask.sum().item()
        if n == 0:
            return float("nan"), 0, 0
        acc = correct[committed_mask].float().mean().item()
        return acc, n, int(no_resp[mask].sum().item())

    out = {}
    acc_c, n_c, nr_c = acc_given(congruent)
    acc_i, n_i, nr_i = acc_given(incongruent)
    out["congr/acc_congruent"] = acc_c
    out["congr/acc_incongruent"] = acc_i
    out["congr/n_committed_congruent"] = n_c
    out["congr/n_committed_incongruent"] = n_i
    out["congr/no_resp_rate_congruent"] = nr_c / max(congruent.sum().item(), 1)
    out["congr/no_resp_rate_incongruent"] = nr_i / max(incongruent.sum().item(), 1)

    for c in contrast_levels:
        mask_c = torch.isclose(contrast, torch.tensor(c, device=contrast.device))
        acc_cc, n_cc, _ = acc_given(mask_c & congruent)
        acc_ci, n_ci, _ = acc_given(mask_c & incongruent)
        out[f"congr/acc_c{c}_congruent"] = acc_cc
        out[f"congr/acc_c{c}_incongruent"] = acc_ci
    return out
