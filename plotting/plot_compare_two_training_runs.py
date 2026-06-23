# plot_compare_two_training_runs.py
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

    raise FileNotFoundError(
        f"No progress.csv, result.json, or results.json found in {run_dir}"
    )


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def prepare_xy(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    smooth: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")

    valid = ~(x.isna() | y.isna())
    x = x[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    if smooth > 1:
        y_smooth = y.rolling(window=smooth, min_periods=1).mean()
    else:
        y_smooth = y.copy()

    return x, y, y_smooth


def plot_comparison_metric(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    x_col_a: str,
    x_col_b: str,
    y_col_a: str,
    y_col_b: str,
    label_a: str,
    label_b: str,
    metric_name: str,
    out_path: Path,
    smooth: int = 5,
):
    x_a, y_a, y_a_smooth = prepare_xy(df_a, x_col_a, y_col_a, smooth)
    x_b, y_b, y_b_smooth = prepare_xy(df_b, x_col_b, y_col_b, smooth)

    if len(x_a) == 0 or len(x_b) == 0:
        return

    plt.figure(figsize=(8, 5))

    plt.plot(x_a, y_a, alpha=0.20, linewidth=1.0, label=f"{label_a} raw")
    plt.plot(x_a, y_a_smooth, linewidth=2.0, label=f"{label_a} rolling mean")

    plt.plot(x_b, y_b, alpha=0.20, linewidth=1.0, label=f"{label_b} raw")
    plt.plot(x_b, y_b_smooth, linewidth=2.0, label=f"{label_b} rolling mean")

    plt.xlabel("Training timesteps")
    plt.ylabel(metric_name)
    plt.title(f"Training comparison: {metric_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def plot_overview(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    x_col_a: str,
    x_col_b: str,
    selected_metrics: dict[str, tuple[str, str]],
    label_a: str,
    label_b: str,
    out_path: Path,
    smooth: int = 5,
):
    n_metrics = len(selected_metrics)
    if n_metrics == 0:
        return

    ncols = 2
    nrows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 4.2 * nrows),
        squeeze=False,
    )

    axes_flat = axes.flatten()

    for ax, (metric_name, (y_col_a, y_col_b)) in zip(axes_flat, selected_metrics.items()):
        x_a, y_a, y_a_smooth = prepare_xy(df_a, x_col_a, y_col_a, smooth)
        x_b, y_b, y_b_smooth = prepare_xy(df_b, x_col_b, y_col_b, smooth)

        if len(x_a) == 0 or len(x_b) == 0:
            ax.set_visible(False)
            continue

        ax.plot(x_a, y_a, alpha=0.15, linewidth=0.8)
        ax.plot(x_a, y_a_smooth, linewidth=2.0, label=label_a)

        ax.plot(x_b, y_b, alpha=0.15, linewidth=0.8)
        ax.plot(x_b, y_b_smooth, linewidth=2.0, label=label_b)

        ax.set_title(metric_name)
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle("Training run comparison", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def summarize_metric(df: pd.DataFrame, y_col: str) -> dict[str, float | str]:
    vals = pd.to_numeric(df[y_col], errors="coerce").dropna()

    if vals.empty:
        return {
            "column": y_col,
            "last": None,
            "max": None,
            "min": None,
        }

    return {
        "column": y_col,
        "last": float(vals.iloc[-1]),
        "max": float(vals.max()),
        "min": float(vals.min()),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir-a",
        type=str,
        required=True,
        help="Path to first finished RLlib training run folder",
    )
    parser.add_argument(
        "--run-dir-b",
        type=str,
        required=True,
        help="Path to second finished RLlib training run folder",
    )
    parser.add_argument(
        "--label-a",
        type=str,
        required=True,
        help="Label for first run, e.g. CNN",
    )
    parser.add_argument(
        "--label-b",
        type=str,
        required=True,
        help="Label for second run, e.g. CNN+Roles",
    )
    parser.add_argument(
        "--comparison-name",
        type=str,
        required=True,
        help="Name for output subfolder, e.g. cnn_vs_roles_s2",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--logs-root",
        type=str,
        default="logs_train",
        help="Root folder where pics/ will be created",
    )

    args = parser.parse_args()

    run_dir_a = Path(args.run_dir_a).resolve()
    run_dir_b = Path(args.run_dir_b).resolve()

    logs_root = Path(args.logs_root).resolve()
    out_dir = logs_root / "pics" / args.comparison_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df_a = load_training_df(run_dir_a)
    df_b = load_training_df(run_dir_b)

    x_candidates = [
        "timesteps_total",
        "num_env_steps_sampled_lifetime",
        "agent_timesteps_total",
        "training_iteration",
    ]

    x_col_a = find_col(df_a, x_candidates)
    x_col_b = find_col(df_b, x_candidates)

    if x_col_a is None:
        raise ValueError(
            f"Could not find x-axis column for run A. Columns: {list(df_a.columns)}"
        )

    if x_col_b is None:
        raise ValueError(
            f"Could not find x-axis column for run B. Columns: {list(df_b.columns)}"
        )

    metrics = {
        "episode_reward_mean": [
            "episode_reward_mean",
            "env_runners/episode_reward_mean",
            "sampler_results/episode_reward_mean",
            "env_runners/episode_return_mean",
            "sampler_results/episode_return_mean",
        ],
        "episode_len_mean": [
            "episode_len_mean",
            "env_runners/episode_len_mean",
            "sampler_results/episode_len_mean",
        ],
        "entropy": [
            "info/learner/default_policy/learner_stats/entropy",
            "learner_stats/entropy",
        ],
        "policy_loss": [
            "info/learner/default_policy/learner_stats/policy_loss",
            "learner_stats/policy_loss",
        ],
        "vf_loss": [
            "info/learner/default_policy/learner_stats/vf_loss",
            "learner_stats/vf_loss",
        ],
        "total_loss": [
            "info/learner/default_policy/learner_stats/total_loss",
            "learner_stats/total_loss",
        ],
        "kl": [
            "info/learner/default_policy/learner_stats/kl",
            "info/learner/default_policy/learner_stats/mean_kl_loss",
            "learner_stats/kl",
            "learner_stats/mean_kl_loss",
        ],
    }

    found_metrics: dict[str, tuple[str, str]] = {}

    summary = {
        "comparison_name": args.comparison_name,
        "run_a": {
            "label": args.label_a,
            "run_dir": str(run_dir_a),
            "x_column": x_col_a,
        },
        "run_b": {
            "label": args.label_b,
            "run_dir": str(run_dir_b),
            "x_column": x_col_b,
        },
        "plots_dir": str(out_dir),
        "metrics": {},
    }

    for metric_name, candidates in metrics.items():
        y_col_a = find_col(df_a, candidates)
        y_col_b = find_col(df_b, candidates)

        if y_col_a is None or y_col_b is None:
            continue

        found_metrics[metric_name] = (y_col_a, y_col_b)

        plot_comparison_metric(
            df_a=df_a,
            df_b=df_b,
            x_col_a=x_col_a,
            x_col_b=x_col_b,
            y_col_a=y_col_a,
            y_col_b=y_col_b,
            label_a=args.label_a,
            label_b=args.label_b,
            metric_name=metric_name,
            out_path=out_dir / f"{metric_name}_comparison.png",
            smooth=args.smooth,
        )

        summary["metrics"][metric_name] = {
            args.label_a: summarize_metric(df_a, y_col_a),
            args.label_b: summarize_metric(df_b, y_col_b),
        }

    overview_metrics = {
        name: cols
        for name, cols in found_metrics.items()
        if name in [
            "episode_reward_mean",
            "episode_len_mean",
            "entropy",
            "policy_loss",
            "vf_loss",
            "total_loss",
        ]
    }

    plot_overview(
        df_a=df_a,
        df_b=df_b,
        x_col_a=x_col_a,
        x_col_b=x_col_b,
        selected_metrics=overview_metrics,
        label_a=args.label_a,
        label_b=args.label_b,
        out_path=out_dir / "overview_comparison.png",
        smooth=args.smooth,
    )

    summary_path = out_dir / "comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not found_metrics:
        print("No comparable known metric columns found.")
        print("\nAvailable columns in run A:")
        for c in df_a.columns:
            print(" -", c)

        print("\nAvailable columns in run B:")
        for c in df_b.columns:
            print(" -", c)
    else:
        print(f"Comparison plots written to: {out_dir}")
        print(f"Overview plot written to: {out_dir / 'overview_comparison.png'}")
        print(f"Summary written to: {summary_path}")
        print("\nCompared metrics:")
        for metric_name in found_metrics:
            print(f" - {metric_name}")


if __name__ == "__main__":
    main()
