import numpy as np
import pandas as pd
import matplotlib
import zarr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

# ---------------------------------------------------------------------------
# 1. Accuracy, split by contrast and congruence
# ---------------------------------------------------------------------------


def load_and_pool_sessions(session_paths, pad_to=None):
    """Loads multiple single-session zarr stores and concatenates them along
    the trial axis, keeping a session_id-like grouping column so pooled
    decoding can still respect session boundaries if needed (e.g. grouped CV)."""
    all_meta, all_r_v, all_r_c, all_r_p = [], [], [], []

    if pad_to is None:
        pad_to = 0
        for p in session_paths:
            store = zarr.open(p, mode="r")
            pad_to = max(pad_to, store["full_r_v"].shape[1])

    for i, p in enumerate(session_paths):
        store = zarr.open(p, mode="r")
        meta = pd.DataFrame(
            {
                f: store[f][:]
                for f in store.array_keys()
                if f not in ("full_r_v", "full_r_c", "full_r_p", "full_x_stim")
            }
        )
        meta["session_id"] = i
        meta["block_bias"] = meta["block_bias"].round(3)  # if needed
        all_meta.append(meta)

        r_v = store["full_r_v"][:]
        r_c = store["full_r_c"][:]
        r_p = store["full_r_p"][:]
        pad_v = pad_to - r_v.shape[1]
        pad_c = pad_to - r_c.shape[1]
        pad_p = pad_to - r_p.shape[1]
        if pad_v > 0:
            r_v = np.pad(r_v, ((0, 0), (0, pad_v), (0, 0)))
        if pad_c > 0:
            r_c = np.pad(r_c, ((0, 0), (0, pad_c), (0, 0)))
        if pad_p > 0:
            r_p = np.pad(r_p, ((0, 0), (0, pad_p), (0, 0)))
        all_r_v.append(r_v)
        all_r_c.append(r_c)
        all_r_p.append(r_p)

    meta_df = pd.concat(all_meta, ignore_index=True)
    r_v = np.concatenate(all_r_v, axis=0)
    r_c = np.concatenate(all_r_c, axis=0)
    r_p = np.concatenate(all_r_p, axis=0)
    return meta_df, r_v, r_c, r_p


def add_outcome_and_congruency_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds: committed, correct, incorrect, is_biased, congruent, incongruent.
    Mirrors the three-way outcome convention (correct/incorrect/no-response
    kept strictly separate) and the block-bias-based congruency definition
    used throughout this project."""
    df = df.copy()
    df["committed"] = df["timed_out"] == 0
    df["correct"] = df["committed"] & (df["choice"] == df["correct_label"])
    df["incorrect"] = df["committed"] & (df["choice"] != df["correct_label"])

    is_biased = df["block_bias"] != 0.5
    block_right = df["block_bias"] > 0.5
    stim_right = df["stim_side"] > 0
    df["is_biased"] = is_biased
    df["congruent"] = (block_right == stim_right) & is_biased
    df["incongruent"] = (~df["congruent"]) & is_biased
    return df


def plot_accuracy_by_contrast_congruency(df: pd.DataFrame, min_n: int = 5):
    """Three-way (correct/incorrect/no-response) stacked bars, as fractions
    of total trials per contrast x congruency cell, with trial counts
    annotated. Cells with fewer than min_n trials are shown greyed/flagged."""
    df = add_outcome_and_congruency_columns(df)
    contrasts = sorted(df["contrast"].unique())
    conds = ["congruent", "incongruent"]
    x = np.arange(len(contrasts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, cond in enumerate(conds):
        offset = (i - 0.5) * width
        fc, fi, fn, ns, low_n = [], [], [], [], []
        for c in contrasts:
            sub = df[(df["contrast"] == c) & (df[cond])]
            n = len(sub)
            ns.append(n)
            low_n.append(n < min_n)
            if n == 0:
                fc.append(0)
                fi.append(0)
                fn.append(0)
            else:
                fc.append(sub["correct"].sum() / n)
                fi.append(sub["incorrect"].sum() / n)
                fn.append((~sub["committed"]).sum() / n)

        b1 = ax.bar(
            x + offset,
            fc,
            width,
            color="tab:green",
            alpha=0.9,
            label="correct" if i == 0 else None,
        )
        b2 = ax.bar(
            x + offset,
            fi,
            width,
            bottom=fc,
            color="tab:red",
            alpha=0.9,
            label="incorrect" if i == 0 else None,
        )
        b3 = ax.bar(
            x + offset,
            fn,
            width,
            bottom=np.array(fc) + np.array(fi),
            color="tab:gray",
            alpha=0.9,
            label="no response" if i == 0 else None,
        )

        for xi, n, flag in zip(x + offset, ns, low_n):
            txt = f"n={n}" + (" (low N)" if flag else "")
            ax.text(
                xi,
                1.03,
                txt,
                ha="center",
                va="bottom",
                fontsize=6,
                color="red" if flag else "black",
            )
        ax.text(
            x[0] + offset,
            -0.12,
            cond,
            ha="center",
            fontsize=8,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in contrasts])
    ax.set_xlabel("contrast")
    ax.set_ylabel("fraction of trials")
    ax.set_ylim(-0.05, 1.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Reaction times
# ---------------------------------------------------------------------------


def plot_rt_distributions(df: pd.DataFrame, dt_ms: float, by: str = "contrast"):
    """RT histograms (committed trials only -- timed-out trials excluded,
    since their 'rt' reflects a forced cutoff, not genuine decision time),
    converted to ms, split by `by` ('contrast' or 'congruent'/'incongruent').
    Also returns median RT per group as a simple summary table."""
    df = add_outcome_and_congruency_columns(df)
    committed = df[df["committed"]].copy()
    committed["rt_ms"] = committed["rt"] * dt_ms

    if by == "contrast":
        groups = sorted(committed["contrast"].unique())
        get_group = lambda g: committed[committed["contrast"] == g]
    elif by == "congruency":
        groups = ["congruent", "incongruent"]
        get_group = lambda g: committed[committed[g]]
    else:
        raise ValueError(f"unknown `by`: {by}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    summary = []
    for g in groups:
        sub = get_group(g)
        if len(sub) == 0:
            continue
        ax.hist(
            sub["rt_ms"], bins=30, alpha=0.5, label=f"{g} (n={len(sub)})", density=True
        )
        summary.append(
            {
                "group": g,
                "n": len(sub),
                "median_rt_ms": sub["rt_ms"].median(),
                "mean_rt_ms": sub["rt_ms"].mean(),
            }
        )

    ax.set_xlabel("RT (ms)")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, pd.DataFrame(summary)


def plot_rt_vs_contrast(df: pd.DataFrame, dt_ms: float):
    """Classic chronometric-function style plot: median RT (+ IQR) as a
    function of contrast, committed trials only."""
    df = add_outcome_and_congruency_columns(df)
    committed = df[df["committed"]].copy()
    committed["rt_ms"] = committed["rt"] * dt_ms

    contrasts = sorted(committed["contrast"].unique())
    medians, q25, q75, ns = [], [], [], []
    for c in contrasts:
        sub = committed[committed["contrast"] == c]["rt_ms"]
        ns.append(len(sub))
        medians.append(sub.median() if len(sub) else np.nan)
        q25.append(sub.quantile(0.25) if len(sub) else np.nan)
        q75.append(sub.quantile(0.75) if len(sub) else np.nan)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar(
        contrasts,
        medians,
        yerr=[np.array(medians) - np.array(q25), np.array(q75) - np.array(medians)],
        marker="o",
        capsize=3,
    )
    for c, n in zip(contrasts, ns):
        ax.annotate(
            f"n={n}",
            (c, 0),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=6,
        )
    ax.set_xlabel("contrast")
    ax.set_ylabel("RT (ms), median ± IQR")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Psychometric curves
# ---------------------------------------------------------------------------


def _sigmoid(x, bias, slope, lapse_lo, lapse_hi):
    return lapse_lo + (1 - lapse_lo - lapse_hi) / (1 + np.exp(-slope * (x - bias)))


def fit_psychometric(df: pd.DataFrame, split_by_block: bool = True):
    """Fits P(choice=right) as a function of SIGNED contrast (stim_side *
    contrast), with a 4-parameter sigmoid (bias, slope, lower/upper lapse
    rates) -- standard psychometric convention. Committed trials only.
    Returns (figure, dict of fit params per group)."""
    df = add_outcome_and_congruency_columns(df)
    committed = df[df["committed"]].copy()
    committed["signed_stim"] = committed["stim_side"] * committed["contrast"]
    committed["choice_right"] = committed["choice"].astype(float)

    groups = (
        {
            "left-biased block": committed["block_bias"] < 0.5,
            "right-biased block": committed["block_bias"] > 0.5,
            "unbiased block": committed["block_bias"] == 0.5,
        }
        if split_by_block
        else {"all": pd.Series(True, index=committed.index)}
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fit_params = {}
    for name, mask in groups.items():
        sub = committed[mask]
        if len(sub) < 20:
            continue
        x, y = sub["signed_stim"].values, sub["choice_right"].values
        uniq = np.unique(x)
        emp = [y[x == u].mean() for u in uniq]
        emp_n = [int((x == u).sum()) for u in uniq]
        ax.scatter(
            uniq, emp, s=[max(10, n) for n in emp_n], label=f"{name} (n={len(sub)})"
        )

        try:
            popt, _ = curve_fit(
                _sigmoid,
                x,
                y,
                p0=[0.0, 1.0, 0.05, 0.05],
                bounds=([-np.inf, 0, 0, 0], [np.inf, np.inf, 0.3, 0.3]),
                maxfev=10000,
            )
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, _sigmoid(xs, *popt))
            fit_params[name] = dict(
                zip(["bias", "slope", "lapse_lo", "lapse_hi"], popt)
            )
        except RuntimeError:
            fit_params[name] = None

    ax.set_xlabel("signed contrast")
    ax.set_ylabel("P(choice = right)")
    ax.axhline(0.5, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, fit_params


# ---------------------------------------------------------------------------
# 4. Module specificity: decode stimulus / choice / prior from VISp vs MOs
# ---------------------------------------------------------------------------


def _cv_score(X, y, task, cv=5, n_shuffles=200, seed=0):
    """Returns (real_score, shuffle_null_mean, shuffle_null_std). Shuffle
    control is essential here -- an R^2/accuracy number alone doesn't tell
    you whether a module is 'specific', only relative to a null does."""
    rng = np.random.default_rng(seed)
    if task == "classification":
        if len(np.unique(y)) < 2:
            return np.nan, np.nan, np.nan
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        splitter = StratifiedKFold(n_splits=cv)
        scoring = "accuracy"
    else:
        model = make_pipeline(StandardScaler(), LinearRegression())
        splitter = KFold(n_splits=cv)
        scoring = "r2"

    try:
        fold_scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring)
        real_scores = fold_scores.mean()
    except ValueError:
        return np.nan, np.nan, np.nan, np.nan

    null_scores = []
    for _ in range(n_shuffles):
        y_shuf = rng.permutation(y)
        try:
            null_scores.append(
                cross_val_score(model, X, y_shuf, cv=splitter, scoring=scoring).mean()
            )
        except ValueError:
            continue
    null_scores = np.array(null_scores) if null_scores else np.array([np.nan])
    return real_scores, null_scores.mean(), null_scores.std(), fold_scores


import re
import numpy as np
import matplotlib.pyplot as plt


def _epoch_to_numeric_x(epoch_label: str):
    """Extracts the underlying fraction from 'decision_XXpct' labels;
    post_feedback is handled separately (not a decision fraction)."""
    m = re.match(r"decision_(\d+)pct", epoch_label)
    if m:
        return int(m.group(1))
    return None  # post_feedback or anything else handled separately


def decode_module_specificity(
    r_v,
    r_c,
    r_p,
    meta_df,
    trace_len,
    epoch_fracs=(0.2, 0.5, 0.8, 1.0),
    t_iti=None,
    cv=5,
    n_shuffles=20,
):
    n_trials = r_v.shape[0]
    rt = meta_df["rt"].values
    signed_stim = (meta_df["stim_side"] * meta_df["contrast"]).values
    choice = meta_df["choice"].values
    block_label = (meta_df["block_bias"] > 0.5).astype(int).values
    committed = (meta_df["timed_out"] == 0).values

    modules = {"V": r_v, "C": r_c, "P": r_p}
    records = []

    for frac in epoch_fracs:
        epoch_name = f"decision_{int(frac*100)}pct"
        t_idx = np.clip((rt * frac).astype(int), 0, None)
        t_idx = np.minimum(t_idx, trace_len.astype(int) - 1)

        for target_name, y, task in [
            ("stimulus", signed_stim, "regression"),
            ("choice", choice, "classification"),
            ("prior_block", block_label, "classification"),
        ]:
            for module_name, r in modules.items():
                X = r[np.arange(n_trials), t_idx, :]
                real, null_mean, null_std, fold_scores = _cv_score(
                    X[committed], y[committed], task, cv, n_shuffles
                )
                records.append(
                    {
                        "epoch": epoch_name,
                        "module": module_name,
                        "target": target_name,
                        "score": real,
                        "null_mean": null_mean,
                        "null_std": null_std,
                        "fold_scores":fold_scores
                    }
                )
    return pd.DataFrame(records)


def plot_module_specificity(records_df):
    targets = records_df["target"].unique()
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4))
    colors = {"V": "tab:blue", "C": "tab:orange", "P": "tab:green"}
    for ax, target in zip(axes, targets):
        sub = records_df[records_df["target"] == target]
        epochs = sorted(sub["epoch"].unique())
        for module, color in colors.items():
            msub = sub[sub["module"] == module].set_index("epoch").loc[epochs]
            ax.plot(epochs, msub["score"], marker="o", color=color, label=module)
        ax.set_title(target)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    store = zarr.open("posthoc_single_session.zarr", mode="r")
