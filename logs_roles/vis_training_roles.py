from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def read_json_lines(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(flatten_dict(obj))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def load_training_df(run_dir: Path) -> pd.DataFrame:
    progress_csv = run_dir / "progress.csv"
    result_json = run_dir / "result.json"
    results_json = run_dir / "results.json"

    if progress_csv.exists():
        return pd.read_csv(progress_csv)
    if result_json.exists():
        return read_json_lines(result_json)
    if results_json.exists():
        return read_json_lines(results_json)

    raise FileNotFoundError(f"No progress.csv, result.json, or results.json found in {run_dir}")


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def has(*cols: str) -> bool:
        return all(c in df.columns for c in cols)

    if has(
        "info/learner/shared_policy/learner_stats/r3dm/role_entropy",
        "info/learner/shared_policy/learner_stats/r3dm/marginal_role_entropy",
    ):
        role_ent = numeric_series(df, "info/learner/shared_policy/learner_stats/r3dm/role_entropy")
        marg_ent = numeric_series(df, "info/learner/shared_policy/learner_stats/r3dm/marginal_role_entropy")

        df["derived/r3dm_role_collapse_gap"] = marg_ent - role_ent
        df["derived/r3dm_role_entropy_to_marginal_ratio"] = role_ent / (marg_ent + 1e-8)

    if "info/learner/shared_policy/learner_stats/r3dm/mean_max_role_prob" in df.columns:
        mmrp = numeric_series(df, "info/learner/shared_policy/learner_stats/r3dm/mean_max_role_prob")
        df["derived/r3dm_role_confidence_excess"] = mmrp - 0.5

    if (
        "info/learner/shared_policy/learner_stats/r3dm/intrinsic_max" in df.columns
        and "info/learner/shared_policy/learner_stats/r3dm/intrinsic_min" in df.columns
    ):
        intr_max = numeric_series(df, "info/learner/shared_policy/learner_stats/r3dm/intrinsic_max")
        intr_min = numeric_series(df, "info/learner/shared_policy/learner_stats/r3dm/intrinsic_min")
        df["derived/r3dm_intrinsic_span"] = intr_max - intr_min

    if "info/learner/shared_policy/learner_stats/r3dm/role_ablation_action_same_fraction_zero_role" in df.columns:
        same = numeric_series(
            df,
            "info/learner/shared_policy/learner_stats/r3dm/role_ablation_action_same_fraction_zero_role",
        )
        df["derived/r3dm_ablation_action_change_fraction"] = 1.0 - same

    if "info/learner/shared_policy/learner_stats/r3dm/role_ablation_relative_logit_delta" in df.columns:
        rel = numeric_series(
            df,
            "info/learner/shared_policy/learner_stats/r3dm/role_ablation_relative_logit_delta",
        )
        df["derived/r3dm_ablation_role_effect_strength"] = rel

    if (
        "env_runners/episode_reward_mean" in df.columns
        and "env_runners/episode_len_mean" in df.columns
    ):
        rew = numeric_series(df, "env_runners/episode_reward_mean")
        ep_len = numeric_series(df, "env_runners/episode_len_mean")
        df["derived/reward_per_step_proxy"] = rew / (ep_len + 1e-8)

    return df


def plot_metric(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    smooth: int = 5,
):
    x = numeric_series(df, x_col)
    y = numeric_series(df, y_col)

    valid = ~(x.isna() | y.isna())
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return False

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="raw", alpha=0.35)

    if smooth > 1:
        y_smooth = y.rolling(window=smooth, min_periods=1).mean()
        plt.plot(x, y_smooth, label=f"rolling mean ({smooth})")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def plot_overview(
    df: pd.DataFrame,
    x_col: str,
    metric_map: Dict[str, str],
    out_path: Path,
    smooth: int = 5,
):
    available = []
    for metric_name, y_col in metric_map.items():
        if y_col in df.columns:
            x = numeric_series(df, x_col)
            y = numeric_series(df, y_col)
            valid = ~(x.isna() | y.isna())
            if valid.any():
                available.append((metric_name, y_col))

    if not available:
        return False

    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (metric_name, y_col) in zip(axes, available):
        x = numeric_series(df, x_col)
        y = numeric_series(df, y_col)

        valid = ~(x.isna() | y.isna())
        x = x[valid]
        y = y[valid]

        ax.plot(x, y, alpha=0.35, label="raw")
        if smooth > 1:
            y_smooth = y.rolling(window=smooth, min_periods=1).mean()
            ax.plot(x, y_smooth, label="smooth")

        ax.set_title(metric_name)
        ax.set_xlabel(x_col)
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(available):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def plot_role_diagnostics_overview(df: pd.DataFrame, x_col: str, out_path: Path, smooth: int = 5):
    interesting = [
        ("role_entropy", "info/learner/shared_policy/learner_stats/r3dm/role_entropy"),
        ("marginal_role_entropy", "info/learner/shared_policy/learner_stats/r3dm/marginal_role_entropy"),
        ("mean_max_role_prob", "info/learner/shared_policy/learner_stats/r3dm/mean_max_role_prob"),
        ("role_mi_proxy", "info/learner/shared_policy/learner_stats/r3dm/role_mi_proxy"),
        ("collapse_gap", "derived/r3dm_role_collapse_gap"),
        ("entropy_ratio", "derived/r3dm_role_entropy_to_marginal_ratio"),
        ("ablation_action_change_fraction", "derived/r3dm_ablation_action_change_fraction"),
        ("ablation_role_effect_strength", "derived/r3dm_ablation_role_effect_strength"),
    ]

    available = [(name, col) for name, col in interesting if col in df.columns]
    if not available:
        return False

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for ax, (name, col) in zip(axes, available):
        x = numeric_series(df, x_col)
        y = numeric_series(df, col)
        valid = ~(x.isna() | y.isna())
        x = x[valid]
        y = y[valid]

        if len(x) == 0:
            ax.axis("off")
            continue

        ax.plot(x, y, alpha=0.35, label="raw")
        if smooth > 1:
            ax.plot(x, y.rolling(window=smooth, min_periods=1).mean(), label="smooth")

        ax.set_title(name)
        ax.set_xlabel(x_col)
        ax.grid(True, alpha=0.3)

        if name == "mean_max_role_prob":
            ax.axhline(0.5, linestyle="--", alpha=0.5)
        if name in {"role_entropy", "marginal_role_entropy"}:
            ax.axhline(0.69314718056, linestyle="--", alpha=0.5)

    for ax in axes[len(available):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def build_metrics() -> Dict[str, list[str]]:
    return {
        "episode_reward_mean": [
            "env_runners/episode_reward_mean",
            "episode_reward_mean",
            "sampler_results/episode_reward_mean",
            "env_runners/episode_return_mean",
            "sampler_results/episode_return_mean",
        ],
        "episode_reward_max": [
            "env_runners/episode_reward_max",
            "episode_reward_max",
        ],
        "episode_reward_min": [
            "env_runners/episode_reward_min",
            "episode_reward_min",
        ],
        "episode_len_mean": [
            "env_runners/episode_len_mean",
            "episode_len_mean",
            "sampler_results/episode_len_mean",
        ],
        "episodes_this_iter": [
            "env_runners/episodes_this_iter",
            "episodes_this_iter",
        ],

        "total_loss": [
            "info/learner/shared_policy/learner_stats/total_loss",
            "info/learner/default_policy/learner_stats/total_loss",
            "learner_stats/total_loss",
        ],
        "policy_loss": [
            "info/learner/shared_policy/learner_stats/policy_loss",
            "info/learner/default_policy/learner_stats/policy_loss",
            "learner_stats/policy_loss",
        ],
        "vf_loss": [
            "info/learner/shared_policy/learner_stats/vf_loss",
            "info/learner/default_policy/learner_stats/vf_loss",
            "learner_stats/vf_loss",
        ],
        "vf_explained_var": [
            "info/learner/shared_policy/learner_stats/vf_explained_var",
            "info/learner/default_policy/learner_stats/vf_explained_var",
            "learner_stats/vf_explained_var",
        ],
        "kl": [
            "info/learner/shared_policy/learner_stats/kl",
            "info/learner/default_policy/learner_stats/kl",
            "info/learner/default_policy/learner_stats/mean_kl_loss",
            "learner_stats/kl",
        ],
        "entropy": [
            "info/learner/shared_policy/learner_stats/entropy",
            "info/learner/default_policy/learner_stats/entropy",
            "learner_stats/entropy",
        ],
        "entropy_coeff": [
            "info/learner/shared_policy/learner_stats/entropy_coeff",
            "info/learner/default_policy/learner_stats/entropy_coeff",
        ],
        "grad_gnorm": [
            "info/learner/shared_policy/learner_stats/grad_gnorm",
            "info/learner/default_policy/learner_stats/grad_gnorm",
        ],
        "cur_lr": [
            "info/learner/shared_policy/learner_stats/cur_lr",
            "info/learner/default_policy/learner_stats/cur_lr",
        ],
        "cur_kl_coeff": [
            "info/learner/shared_policy/learner_stats/cur_kl_coeff",
            "info/learner/default_policy/learner_stats/cur_kl_coeff",
        ],

        "policy_reward_mean_shared": [
            "env_runners/policy_reward_mean/shared_policy",
        ],
        "policy_reward_max_shared": [
            "env_runners/policy_reward_max/shared_policy",
        ],
        "policy_reward_min_shared": [
            "env_runners/policy_reward_min/shared_policy",
        ],

        "sample_throughput": [
            "num_env_steps_sampled_throughput_per_sec",
        ],
        "train_throughput": [
            "num_env_steps_trained_throughput_per_sec",
        ],
        "learn_throughput": [
            "timers/learn_throughput",
        ],
        "cpu_util_percent": [
            "perf/cpu_util_percent",
        ],
        "ram_util_percent": [
            "perf/ram_util_percent",
        ],

        "r3dm_aux_nce": [
            "info/learner/shared_policy/learner_stats/r3dm/aux_nce",
        ],
        "r3dm_role_mi_proxy": [
            "info/learner/shared_policy/learner_stats/r3dm/role_mi_proxy",
        ],
        "r3dm_role_entropy": [
            "info/learner/shared_policy/learner_stats/r3dm/role_entropy",
        ],
        "r3dm_marginal_role_entropy": [
            "info/learner/shared_policy/learner_stats/r3dm/marginal_role_entropy",
        ],
        "r3dm_mean_max_role_prob": [
            "info/learner/shared_policy/learner_stats/r3dm/mean_max_role_prob",
        ],
        "r3dm_intrinsic_abs_mean": [
            "info/learner/shared_policy/learner_stats/r3dm/intrinsic_abs_mean",
        ],
        "r3dm_intrinsic_max": [
            "info/learner/shared_policy/learner_stats/r3dm/intrinsic_max",
        ],
        "r3dm_intrinsic_min": [
            "info/learner/shared_policy/learner_stats/r3dm/intrinsic_min",
        ],
        "r3dm_gumbel_tau_learner": [
            "info/learner/shared_policy/learner_stats/r3dm/gumbel_tau",
        ],
        "r3dm_gumbel_tau_callback": [
            "r3dm_gumbel_tau_callback",
            "custom_metrics/gumbel_tau",
        ],

        "r3dm_role_ablation_mean_abs_logit_delta": [
            "info/learner/shared_policy/learner_stats/r3dm/role_ablation_mean_abs_logit_delta",
        ],
        "r3dm_role_ablation_max_abs_logit_delta": [
            "info/learner/shared_policy/learner_stats/r3dm/role_ablation_max_abs_logit_delta",
        ],
        "r3dm_role_ablation_mean_abs_logit_normal": [
            "info/learner/shared_policy/learner_stats/r3dm/role_ablation_mean_abs_logit_normal",
        ],
        "r3dm_role_ablation_mean_abs_logit_zero": [
            "info/learner/shared_policy/learner_stats/r3dm/role_ablation_mean_abs_logit_zero",
        ],
        "r3dm_role_ablation_relative_logit_delta": [
            "info/learner/shared_policy/learner_stats/r3dm/role_ablation_relative_logit_delta",
        ],
        "r3dm_role_ablation_action_same_fraction_zero_role": [
            "info/learner/shared_policy/learner_stats/r3dm/role_ablation_action_same_fraction_zero_role",
        ],

        "derived_r3dm_role_collapse_gap": [
            "derived/r3dm_role_collapse_gap",
        ],
        "derived_r3dm_role_entropy_to_marginal_ratio": [
            "derived/r3dm_role_entropy_to_marginal_ratio",
        ],
        "derived_r3dm_role_confidence_excess": [
            "derived/r3dm_role_confidence_excess",
        ],
        "derived_r3dm_intrinsic_span": [
            "derived/r3dm_intrinsic_span",
        ],
        "derived_r3dm_ablation_action_change_fraction": [
            "derived/r3dm_ablation_action_change_fraction",
        ],
        "derived_r3dm_ablation_role_effect_strength": [
            "derived/r3dm_ablation_role_effect_strength",
        ],
        "derived_reward_per_step_proxy": [
            "derived/reward_per_step_proxy",
        ],
    }


def diagnose_run(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def last(col: str):
        if col not in df.columns:
            return None
        s = numeric_series(df, col).dropna()
        if s.empty:
            return None
        return float(s.iloc[-1])

    role_ent = last("info/learner/shared_policy/learner_stats/r3dm/role_entropy")
    marg_ent = last("info/learner/shared_policy/learner_stats/r3dm/marginal_role_entropy")
    max_prob = last("info/learner/shared_policy/learner_stats/r3dm/mean_max_role_prob")
    action_same = last(
        "info/learner/shared_policy/learner_stats/r3dm/role_ablation_action_same_fraction_zero_role"
    )
    rel_delta = last(
        "info/learner/shared_policy/learner_stats/r3dm/role_ablation_relative_logit_delta"
    )
    reward_mean = last("env_runners/episode_reward_mean")
    ep_len = last("env_runners/episode_len_mean")

    out["last_role_entropy"] = role_ent
    out["last_marginal_role_entropy"] = marg_ent
    out["last_mean_max_role_prob"] = max_prob
    out["last_role_ablation_action_same_fraction_zero_role"] = action_same
    out["last_role_ablation_relative_logit_delta"] = rel_delta
    out["last_episode_reward_mean"] = reward_mean
    out["last_episode_len_mean"] = ep_len

    collapse_flag = (
        role_ent is not None and marg_ent is not None and max_prob is not None
        and role_ent < 0.1 and marg_ent < 0.1 and max_prob > 0.95
    )
    out["likely_role_collapse"] = bool(collapse_flag)

    role_used_flag = (
        action_same is not None and rel_delta is not None
        and action_same < 0.9 and rel_delta > 0.1
    )
    out["likely_role_affects_policy"] = bool(role_used_flag)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Path to one RLlib run folder")
    parser.add_argument("--run-name", type=str, required=True, help="Name for output folder")
    parser.add_argument("--logs-root", type=str, default="logs_roles", help="Root folder for pics/")
    parser.add_argument("--smooth", type=int, default=5, help="Rolling mean window")
    parser.add_argument("--print-columns", action="store_true", help="Print all columns and exit")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    logs_root = Path(args.logs_root).resolve()
    out_dir = logs_root / "pics" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_df(run_dir)
    df = add_derived_metrics(df)

    if args.print_columns:
        print("\nAvailable columns:")
        for c in df.columns:
            print(c)
        return

    x_col = find_col(df, [
        "timesteps_total",
        "num_env_steps_sampled_lifetime",
        "num_env_steps_sampled",
        "agent_timesteps_total",
        "training_iteration",
    ])
    if x_col is None:
        raise ValueError(f"Could not find x-axis column. Available columns: {list(df.columns)}")

    metrics = build_metrics()

    found_metrics: Dict[str, str] = {}
    summary: Dict[str, Any] = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "plots_dir": str(out_dir),
        "x_col": x_col,
    }

    print("\nMatched columns:")
    for metric_name, candidates in metrics.items():
        y_col = find_col(df, candidates)
        print(f"{metric_name:40s} -> {y_col}")
        if y_col is None:
            continue

        ok = plot_metric(
            df=df,
            x_col=x_col,
            y_col=y_col,
            title=f"{args.run_name} - {metric_name}",
            out_path=out_dir / f"{metric_name}.png",
            smooth=args.smooth,
        )
        if not ok:
            continue

        found_metrics[metric_name] = y_col
        vals = numeric_series(df, y_col).dropna()
        if not vals.empty:
            summary[metric_name] = {
                "column": y_col,
                "last": float(vals.iloc[-1]),
                "max": float(vals.max()),
                "min": float(vals.min()),
                "mean": float(vals.mean()),
            }

    plot_overview(
        df=df,
        x_col=x_col,
        metric_map=found_metrics,
        out_path=out_dir / "overview.png",
        smooth=args.smooth,
    )

    plot_role_diagnostics_overview(
        df=df,
        x_col=x_col,
        out_path=out_dir / "role_diagnostics_overview.png",
        smooth=args.smooth,
    )

    summary["diagnosis"] = diagnose_run(df)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not found_metrics:
        print("\nNo known metrics found.")
    else:
        print(f"\nWrote {len(found_metrics)} metric plots to: {out_dir}")
        print(f"Overview plot: {out_dir / 'overview.png'}")
        print(f"Role diagnostics: {out_dir / 'role_diagnostics_overview.png'}")
        print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
