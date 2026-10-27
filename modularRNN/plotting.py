import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import psychofit as psy
import seaborn as sns


def plot_psychometric(meta_df):
    """
    Per-session psychometric fits, split by TRUE block side (left-biased vs
    right-biased; unbiased blocks excluded), fit with psychofit, then
    averaged (mean fitted curve +/- SEM) across sessions. Empirical points
    (mean +/- SEM across sessions, per signed contrast) overlaid.
    """
    df_f = meta_df[meta_df["timed_out"] == 0].copy()  # committed trials only
    df_f = df_f[df_f["block_bias"] != 0.5].copy()  # drop unbiased blocks
    df_f["choice_right"] = df_f["choice"].astype(int)
    df_f["signed_contrast"] = (
        df_f["stim_side"] * df_f["contrast"]
    ) * 100  # psychofit % convention

    df_f["block_side"] = np.where(df_f["block_bias"] < 0.5, "Left Block", "Right Block")

    x_fit = np.linspace(-100, 100, 100)
    session_fits = {"Left Block": [], "Right Block": []}
    session_empirical = {"Left Block": [], "Right Block": []}
    saved_parameters = []

    sessions = df_f["session_id"].unique()
    for session in sessions:
        session_parameter = []
        df_sess = df_f[df_f["session_id"] == session]

        for block_name in ["Left Block", "Right Block"]:
            df_block = df_sess[df_sess["block_side"] == block_name]
            if len(df_block) < 10:
                continue

            agg = (
                df_block.groupby("signed_contrast")["choice_right"]
                .agg(n_trials="count", prob_right="mean")
                .reset_index()
            )
            session_empirical[block_name].append(agg)

            data_array = np.vstack(
                [
                    agg["signed_contrast"].values,
                    agg["n_trials"].values,
                    agg["prob_right"].values,
                ]
            )

            try:
                pars, _ = psy.mle_fit_psycho(
                    data_array,
                    P_model="erf_psycho_2gammas",
                    parstart=np.array([0, 20, 0.05, 0.05]),
                    parmin=np.array([-100, 5, 0, 0]),
                    parmax=np.array([100, 100, 1, 1]),
                )
                y_fit = psy.erf_psycho_2gammas(pars, x_fit)
                session_fits[block_name].append(y_fit)
                session_parameter.append(pars[0])
            except Exception:
                print(f"session {session}, {block_name}: failed to converge")
                continue
        saved_parameters.append(session_parameter)

    fig, ax = plt.subplots(figsize=(12, 6), ncols=2)
    colors = {"Left Block": "blue", "Right Block": "red"}

    for ax_idx, block_name in enumerate(["Left Block", "Right Block"]):
        if len(session_fits[block_name]) == 0:
            continue

        fits_array = np.array(session_fits[block_name])
        mean_fit = np.nanmean(fits_array, axis=0)
        sem_fit = stats.sem(fits_array, axis=0, nan_policy="omit")

        emp_df = pd.concat(session_empirical[block_name])
        emp_agg = (
            emp_df.groupby("signed_contrast")["prob_right"]
            .agg(["mean", "sem", "count"])
            .reset_index()
        )

        ax[ax_idx].plot(x_fit, mean_fit, color=colors[block_name], label=block_name)
        ax[ax_idx].fill_between(
            x_fit,
            mean_fit - sem_fit,
            mean_fit + sem_fit,
            color=colors[block_name],
            alpha=0.2,
        )
        ax[ax_idx].errorbar(
            emp_agg["signed_contrast"],
            emp_agg["mean"],
            yerr=emp_agg["sem"],
            fmt="o",
            color=colors[block_name],
            markersize=5,
            capsize=3,
        )

    ax[ax_idx].set_xlabel("Signed Contrast")
    ax[ax_idx].set_ylabel("Probability of Choosing Right")
    tick_pos = [-100, -25, -12.5, -6.25, 0, 6.25, 12.5, 25, 100]
    ax[ax_idx].set_xticks(tick_pos)
    ax[ax_idx].set_xticklabels([f"{t / 100:g}" for t in tick_pos], rotation=90)
    ax[ax_idx].legend()
    sns.despine()

    return saved_parameters, ax


def plot_population_psychometric_by_block_pooled(meta_df):
    """
    Pools ALL committed, biased-block trials across sessions first, then
    fits one psychometric curve per true block side (left vs right).
    No per-session fitting/averaging -- a single fit per condition on the
    full pooled dataset.
    """
    df_f = meta_df[meta_df["timed_out"] == 0].copy()
    df_f = df_f[df_f["block_bias"] != 0.5].copy()
    df_f["choice_right"] = df_f["choice"].astype(int)
    df_f["signed_contrast"] = (df_f["stim_side"] * df_f["contrast"]) * 100

    df_f["block_side"] = np.where(df_f["block_bias"] < 0.5, "Left Block", "Right Block")

    x_fit = np.linspace(-100, 100, 200)
    colors = {"Left Block": "blue", "Right Block": "red"}
    fig, ax = plt.subplots(figsize=(6, 5))
    fit_results = {}

    for block_name in ["Left Block", "Right Block"]:
        df_block = df_f[df_f["block_side"] == block_name]
        if len(df_block) < 10:
            print(f"{block_name}: too few trials ({len(df_block)}), skipping")
            continue

        agg = (
            df_block.groupby("signed_contrast")["choice_right"]
            .agg(n_trials="count", prob_right="mean")
            .reset_index()
        )

        data_array = np.vstack(
            [
                agg["signed_contrast"].values,
                agg["n_trials"].values,
                agg["prob_right"].values,
            ]
        )

        try:
            pars, logL = psy.mle_fit_psycho(
                data_array,
                P_model="erf_psycho_2gammas",
                parstart=np.array([0, 20, 0.05, 0.05]),
                parmin=np.array([-100, 5, 0, 0]),
                parmax=np.array([100, 100, 1, 1]),
            )
            y_fit = psy.erf_psycho_2gammas(pars, x_fit)
        except Exception as e:
            print(f"{block_name}: fit failed ({e})")
            continue

        fit_results[block_name] = {"pars": pars, "logL": logL, "agg": agg}

        ax.plot(
            x_fit,
            y_fit,
            color=colors[block_name],
            label=f"{block_name}",
        )
        ax.scatter(
            agg["signed_contrast"],
            agg["prob_right"],
            color=colors[block_name],
        )
        # for _, row in agg.iterrows():
        #     ax.annotate(
        #         f"n={int(row['n_trials'])}",
        #         (row["signed_contrast"], row["prob_right"]),
        #         textcoords="offset points",
        #         xytext=(0, 8),
        #         ha="center",
        #         fontsize=6,
        #         color=colors[block_name],
        #     )

    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Signed Contrast")
    ax.set_ylabel("Probability of Choosing Right")
    tick_pos = [-100, -25, -12.5, -6.25, 0, 6.25, 12.5, 25, 100]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{t / 100:g}" for t in tick_pos], rotation=90)
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()

    return fit_results, ax


def reaction_time_plots(meta_df, T_max, dt_ms=10):
    rt_df = meta_df.copy()
    rt_df = rt_df[rt_df["block_bias"] != 0.5]
    rt_df["rt_ms"] = np.where(
        rt_df["timed_out"] == 1, T_max * dt_ms, rt_df["rt"] * dt_ms
    )
    rt_df["signed_contrast"] = rt_df["contrast"] * rt_df["stim_side"]
    rt_df["is_congruent"] = np.where(
        (rt_df["congruent"] == 1), "Congruent", "Incongruent"
    )
    committed = rt_df[rt_df["committed"]].copy()

    result = (
        rt_df.groupby(["is_congruent", "contrast"])
        .agg(
            n_total=("timed_out", "size"),
            n_timed_out=("timed_out", "sum"),
        )
        .reset_index()
    )
    result["frac_uncommitted"] = result["n_timed_out"] / result["n_total"]

    fig, ax = plt.subplots(figsize=(10, 5), ncols=2)
    sns.barplot(
        data=committed,
        x="contrast",
        y="rt_ms",
        hue="is_congruent",
        legend=True,
        ax=ax[0],
    )
    ax[0].set_ylabel("Reaction time")
    ax[0].legend(frameon=False)
    sns.despine()

    sns.barplot(
        data=result,
        x="contrast",
        y="frac_uncommitted",
        hue="is_congruent",
        ax=ax[1],
        legend=False,
    )
    ax[1].set_ylabel("Fraction timed out")


# use decode results
# plot only stimulus, choice and prior
def pretty_plot(decode_results):
    decode_stimulus = decode_results[decode_results["target"] == "stimulus"]
    decode_choice = decode_results[decode_results["target"] == "choice"]
    decode_prior = decode_results[decode_results["target"] == "prior_block"]
    fig, ax = plt.subplots(figsize=(15, 5), ncols=3,sharex=True)

    visp_stimulus = decode_stimulus[decode_stimulus["module"] == "VISp"]
    mos_stimulus = decode_stimulus[decode_stimulus["module"] == "MOs"]
    ax[0].errorbar(
        np.arange(5),
        visp_stimulus.score,
        yerr=np.std(np.stack(visp_stimulus.fold_scores.values), axis=1),
        label="VISp",
    )
    ax[0].errorbar(
        np.arange(5),
        mos_stimulus.score,
        yerr=np.std(np.stack(mos_stimulus.fold_scores.values), axis=1),
        label="MOs",
    )

    ax[0].set_ylabel("$R^2$")
    ax[0].set_ylim(0, 1)
    ax[0].set_xticks(np.arange(5), labels=(0, 0.2, 0.5, 0.8, 1))
    ax[0].legend(frameon=False)
    ax[0].set_title("Stimulus Decoding")
    sns.despine()

    visp_choice = decode_choice[decode_choice["module"] == "VISp"]
    mos_choice = decode_choice[decode_choice["module"] == "MOs"]
    ax[1].errorbar(
        np.arange(5),
        visp_choice.score,
        yerr=np.std(np.stack(visp_choice.fold_scores.values), axis=1),
        label="VISp",
    )
    ax[1].errorbar(
        np.arange(5),
        mos_choice.score,
        yerr=np.std(np.stack(mos_stimulus.fold_scores.values), axis=1),
        label="MOs",
    )

    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(0, 1)
    ax[1].set_xticks(np.arange(5), labels=(0, 0.2, 0.5, 0.8, 1))
    ax[1].legend(frameon=False)
    ax[1].set_title("Choice Decoding")

    visp_prior = decode_prior[decode_prior["module"] == "VISp"]
    mos_prior = decode_prior[decode_prior["module"] == "MOs"]
    ax[2].errorbar(
        np.arange(5),
        visp_prior.score,
        yerr=np.std(np.stack(visp_prior.fold_scores.values), axis=1),
        label="VISp",
    )
    ax[2].errorbar(
        np.arange(5),
        mos_prior.score,
        yerr=np.std(np.stack(mos_stimulus.fold_scores.values), axis=1),
        label="MOs",
    )

    ax[2].set_ylabel("Accuracy")
    ax[2].set_ylim(0, 1)
    ax[2].set_xticks(np.arange(5), labels=(0, 0.2, 0.5, 0.8, 1))
    ax[2].legend(frameon=False)
    ax[2].set_title("True Block Decoding")
    for idx in range(3):
        ax[idx].set_xlabel('Time from Decision')
