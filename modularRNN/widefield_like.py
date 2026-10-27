import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def decode_trained_on_correct_project_errors(
    r_v,
    r_m,
    meta_df,
    trace_len,
    target: str = "choice",
    epoch_fracs=(0.2, 0.5, 0.8, 1.0),
    cv=5,
):
    """
    For each epoch and module (VISp, MOs):
      1. Fit a decoder for `target` using ONLY correct, committed trials
         (cross-validated fit, via cross_val_predict, so every correct
         trial still gets an out-of-fold prediction -- no train/test leakage
         even within the correct-trial training set).
      2. Using that SAME fitted decoder (refit on ALL correct trials, no
         held-out fold needed for this part since these trials were never
         used in fitting), project incorrect trials, split into:
         congruent-left, congruent-right, incongruent-left, incongruent-right.
      3. Return per-trial decoder outputs (predicted probability for
         classification, or predicted value for regression) for every
         correct (out-of-fold) and incorrect (projected) trial, tagged
         with its group label -- NOT a single collapsed accuracy number.

    target: 'choice' (classification, MOs-appropriate) or
            'stimulus' (regression, VISp-appropriate) or 'prior_block'
            (classification).
    """
    rt = meta_df["rt"].values
    signed_stim = (meta_df["stim_side"] * meta_df["contrast"]).values
    choice = meta_df["choice"].values
    block_label = (meta_df["block_bias"] > 0.5).astype(int).values
    correct = meta_df["correct"].values.astype(bool)
    incorrect = meta_df["incorrect"].values.astype(bool)
    congruent = meta_df["congruent"].values.astype(bool)
    incongruent = meta_df["incongruent"].values.astype(bool)
    stim_right = (meta_df["stim_side"] > 0).values

    target_map = {
        "choice": (choice, "classification"),
        "stimulus": (signed_stim, "regression"),
        "prior_block": (block_label, "classification"),
    }
    y_all, task = target_map[target]

    n_trials = r_v.shape[0]
    records = []

    for frac in epoch_fracs:
        epoch_name = f"decision_{int(frac*100)}pct"
        t_idx = np.clip((rt * frac).astype(int), 0, None)
        t_idx = np.minimum(t_idx, trace_len.astype(int) - 1)

        for module_name, r in [("VISp", r_v), ("MOs", r_m)]:
            X_all = r[np.arange(n_trials), t_idx, :]

            X_correct, y_correct = X_all[correct], y_all[correct]
            if len(np.unique(y_correct)) < 2 and task == "classification":
                continue

            if task == "classification":
                model = make_pipeline(
                    StandardScaler(), LogisticRegression(max_iter=1000)
                )
                splitter = StratifiedKFold(n_splits=cv)
                oof_pred = cross_val_predict(
                    model, X_correct, y_correct, cv=splitter, method="predict_proba"
                )[:, 1]
            else:
                model = make_pipeline(StandardScaler(), LinearRegression())
                oof_pred = cross_val_predict(model, X_correct, y_correct, cv=cv)

            # out-of-fold predictions for correct trials (honest, no leakage)
            correct_idx = np.where(correct)[0]
            for idx, pred in zip(correct_idx, oof_pred):
                records.append(
                    {
                        "epoch": epoch_name,
                        "module": module_name,
                        "target": target,
                        "group": "correct",
                        "trial_idx": idx,
                        "pred": pred,
                        "true": y_all[idx],
                    }
                )

            # refit on ALL correct trials (no CV needed here -- these trials
            # were never used to fit this final model, so projecting them is clean)
            model.fit(X_correct, y_correct)

            error_groups = {
                "congruent_left": incorrect & congruent & ~stim_right,
                "congruent_right": incorrect & congruent & stim_right,
                "incongruent_left": incorrect & incongruent & ~stim_right,
                "incongruent_right": incorrect & incongruent & stim_right,
            }
            for group_name, mask in error_groups.items():
                if mask.sum() == 0:
                    continue
                X_grp = X_all[mask]
                if task == "classification":
                    pred_grp = model.predict_proba(X_grp)[:, 1]
                else:
                    pred_grp = model.predict(X_grp)

                grp_idx = np.where(mask)[0]
                for idx, pred in zip(grp_idx, pred_grp):
                    records.append(
                        {
                            "epoch": epoch_name,
                            "module": module_name,
                            "target": target,
                            "group": group_name,
                            "trial_idx": idx,
                            "pred": pred,
                            "true": y_all[idx],
                        }
                    )

    return pd.DataFrame(records)


import matplotlib.pyplot as plt


def plot_decoder_projection_by_group(records_df, module="MOs", epoch="decision_100pct"):
    sub = records_df[(records_df["module"] == module) & (records_df["epoch"] == epoch)]
    groups = [
        "correct",
        "congruent_left",
        "congruent_right",
        "incongruent_left",
        "incongruent_right",
    ]
    colors = {
        "correct": "gray",
        "congruent_left": "tab:blue",
        "congruent_right": "tab:red",
        "incongruent_left": "tab:cyan",
        "incongruent_right": "tab:orange",
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for g in groups:
        vals = sub[sub["group"] == g]["pred"].dropna()
        if len(vals) == 0:
            continue
        ax.hist(
            vals,
            bins=20,
            alpha=0.5,
            density=True,
            color=colors[g],
            label=f"{g} (n={len(vals)})",
        )

    ax.axvline(0.5, color="k", lw=0.5, ls="--")
    ax.set_xlabel("decoder output (P(choice=right) or predicted value)")
    ax.set_ylabel("density")
    ax.set_title(f"{module}, {epoch}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
