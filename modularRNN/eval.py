import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from env import ClosedLoopEnv


def make_eval_env(model, plan_gen, batch_size, device, env_cfg, decision_cfg):
    return ClosedLoopEnv(
        model=model,
        plan_gen=plan_gen,
        batch_size=batch_size,
        device=device,
        T_stim=env_cfg.T_stim,
        T_resp=env_cfg.T_resp,
        T_iti=env_cfg.T_iti,
        t_min=decision_cfg.t_min,
        theta=decision_cfg.theta,
        sigma_stim=env_cfg.sigma_stim,
        hold_fb_during_stim=env_cfg.hold_fb_during_stim,
        t_quiescence=env_cfg.t_quiescence,
    )


@torch.no_grad()
def collect_eval_rollout(model, eval_env, n_trials, store_full_trace=False):
    model.eval()
    eval_env.reset()
    results = [
        eval_env.run_trial(teacher_forcing_prob=0.0, store_full_trace=store_full_trace)
        for _ in range(n_trials)
    ]
    model.train()
    return results


def _stack(results, key):
    return np.concatenate(
        [getattr(r, key).detach().cpu().numpy() for r in results], axis=0
    )


def _sigmoid(x, bias, slope):
    return 1.0 / (1.0 + np.exp(-slope * (x - bias)))


def fit_psychometric(results):
    signed_stim = _stack(results, "target_stim")
    choice = _stack(results, "choice")
    block_bias = _stack(results, "block_bias")

    fig, ax = plt.subplots(figsize=(5, 4))
    groups = {
        "left-biased block": block_bias < 0.5,
        "right-biased block": block_bias > 0.5,
    }
    fit_params = {}

    for name, mask in groups.items():
        if mask.sum() < 20:
            continue
        x, y = signed_stim[mask], choice[mask]
        uniq = np.unique(x)
        emp = [y[x == u].mean() for u in uniq]
        ax.scatter(uniq, emp, label=f"{name} (data)")
        try:
            popt, _ = curve_fit(_sigmoid, x, y, p0=[0.0, 1.0], maxfev=5000)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, _sigmoid(xs, *popt), label=f"{name} (fit)")
            key = name.split("-")[0]
            fit_params[f"{key}_bias"] = float(popt[0])
            fit_params[f"{key}_slope"] = float(popt[1])
        except RuntimeError:
            pass

    ax.set_xlabel("signed contrast")
    ax.set_ylabel("P(choice = right)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, fit_params


def plot_rt_histogram(results):
    rt = _stack(results, "rt")
    timed_out = _stack(results, "timed_out").astype(bool)
    correct = _stack(results, "choice") == _stack(results, "correct_label")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(
        rt[~timed_out & correct], bins=30, alpha=0.6, label="correct", color="tab:green"
    )
    ax.hist(
        rt[~timed_out & ~correct],
        bins=30,
        alpha=0.6,
        label="incorrect",
        color="tab:red",
    )
    ax.set_xlabel("RT (steps from stim onset)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, {"no_response_rate": float(timed_out.mean())}


def _epoch_indices(env_cfg):
    return {
        "early_stim": env_cfg.t_quiescence + 2,
        "late_stim": env_cfg.T_stim - 2,
        "late_resp": env_cfg.T_stim + env_cfg.T_resp - 2,
        "post_feedback": env_cfg.T_stim + env_cfg.T_resp + 2,
    }


def _cv_score(X, y, task, cv=3):
    try:
        if task == "classification":
            if len(np.unique(y)) < 2:
                return np.nan
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        else:
            model = make_pipeline(StandardScaler(), LinearRegression())
            scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        return float(scores.mean())
    except ValueError:
        return np.nan


def run_probe_decoding(results, env_cfg, cv=3):
    r_v = _stack(results, "full_r_v")  # (n_trials, T, N_visp)
    r_m = _stack(results, "full_r_m")  # (n_trials, T, N_mos)
    signed_stim = _stack(results, "target_stim")
    choice = _stack(results, "choice")
    block_label = (_stack(results, "block_bias") > 0.5).astype(int)

    records = []
    for epoch_name, t in _epoch_indices(env_cfg).items():
        t = min(t, r_v.shape[1] - 1)
        records.append(
            {
                "epoch": epoch_name,
                "stim_r2_visp": _cv_score(r_v[:, t, :], signed_stim, "regression", cv),
                "choice_acc_mos": _cv_score(r_m[:, t, :], choice, "classification", cv),
                "prior_acc_visp": _cv_score(
                    r_v[:, t, :], block_label, "classification", cv
                ),
                "prior_acc_mos": _cv_score(
                    r_m[:, t, :], block_label, "classification", cv
                ),
            }
        )
    return records


def plot_probe_decoding(records):
    epochs = [r["epoch"] for r in records]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    axes[0].plot(epochs, [r["stim_r2_visp"] for r in records], marker="o")
    axes[0].set_title("Stimulus decode (VISp, R²)")
    axes[0].axhline(0, color="gray", lw=0.5)

    axes[1].plot(epochs, [r["choice_acc_mos"] for r in records], marker="o")
    axes[1].set_title("Choice decode (MOs, acc)")
    axes[1].axhline(0.5, color="gray", lw=0.5)

    axes[2].plot(
        epochs, [r["prior_acc_visp"] for r in records], marker="o", label="VISp"
    )
    axes[2].plot(epochs, [r["prior_acc_mos"] for r in records], marker="o", label="MOs")
    axes[2].set_title("Block/prior decode (acc)")
    axes[2].axhline(0.5, color="gray", lw=0.5)
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


"""
This goes into eval loop later.
if step % cfg.training.eval_every == 0:
    need_probe = step % cfg.training.probe_every == 0
    eval_results = collect_eval_rollout(
        model, eval_env, cfg.training.eval_n_trials, store_full_trace=need_probe
    )

    fig_psych, psych_params = fit_psychometric(eval_results)
    fig_rt, rt_stats = plot_rt_histogram(eval_results)

    log_dict = {"step": step}
    log_dict.update({f"eval/psych_{k}": v for k, v in psych_params.items()})
    log_dict.update({f"eval/{k}": v for k, v in rt_stats.items()})  # type: ignore

    if cfg.training.use_wandb:
        log_dict["eval/psychometric"] = wandb.Image(fig_psych)  # type: ignore
        log_dict["eval/rt_hist"] = wandb.Image(fig_rt)  # type: ignore

    if need_probe:
        records = run_probe_decoding(
            eval_results, cfg.env, cv=cfg.training.probe_cv_folds
        )
        for r in records:
            log_dict.update(
                {
                    f"probe/{r['epoch']}_{k}": v
                    for k, v in r.items()
                    if k != "epoch"
                }
            )
        if cfg.training.use_wandb:
            fig_probe = plot_probe_decoding(records)
            log_dict["eval/probe_decoding"] = wandb.Image(fig_probe)  # type: ignore

    if cfg.training.use_wandb:
        wandb.log(log_dict)
        plt.close(fig_psych)
        plt.close(fig_rt)
        if need_probe:
            plt.close(fig_probe)
    else:
        print(f"[eval @ step {step}] {log_dict}")
"""
