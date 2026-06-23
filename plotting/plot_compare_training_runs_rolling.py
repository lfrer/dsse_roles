#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


RunData = Dict[str, Any]

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def read_json_lines(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(flatten_dict(obj))
    return pd.DataFrame(rows)


def load_training_df(run_dir: Path) -> pd.DataFrame:
    candidates = [
        run_dir / "progress.csv",
        run_dir / "result.json",
        run_dir / "results.json",
    ]

    for path in candidates:
        if path.exists():
            if path.suffix == ".csv":
                return pd.read_csv(path)
            return read_json_lines(path)

    raise FileNotFoundError(
        f"No progress.csv, result.json, or results.json found in {run_dir}"
    )


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def prepare_rolling_xy(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    smooth: int,
    max_x: Optional[float] = None,
) -> tuple[pd.Series, pd.Series]:

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")

    valid = ~(x.isna() | y.isna())
    if max_x is not None:
        valid &= x <= max_x

    x = x[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    if smooth > 1:
        y_roll = y.rolling(window=smooth, min_periods=1).mean()
    else:
        y_roll = y.copy()

    return x, y_roll


def summarize_rolling_metric(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    smooth: int,
    max_x: Optional[float] = None,
) -> dict[str, float | str | None]:
    x, y_roll = prepare_rolling_xy(df, x_col, y_col, smooth, max_x=max_x)

    if y_roll.empty:
        return {
            "column": y_col,
            "last_x": None,
            "last_rolling_mean": None,
            "max_rolling_mean": None,
            "min_rolling_mean": None,
        }

    return {
        "column": y_col,
        "last_x": float(x.iloc[-1]),
        "last_rolling_mean": float(y_roll.iloc[-1]),
        "max_rolling_mean": float(y_roll.max()),
        "min_rolling_mean": float(y_roll.min()),
    }


def plot_metric_rolling(
    runs: list[RunData],
    metric_name: str,
    metric_columns: dict[str, str],
    out_path: Path,
    smooth: int,
    max_x: Optional[float] = None,
    x_label: str = "Training timesteps",
    name: str = ""
) -> None:
    plt.figure(figsize=(8.5, 5.0))

    plotted = 0
    for run in runs:
        label = run["label"]
        y_col = metric_columns[label]
        x, y_roll = prepare_rolling_xy(
            run["df"], run["x_col"], y_col, smooth, max_x=max_x
        )

        if len(x) == 0:
            continue

        plt.plot(x, y_roll, linewidth=2.2, label=label)
        plotted += 1

    if plotted == 0:
        plt.close()
        return

    plt.xlabel(x_label)
    plt.ylabel(metric_name)
    plt.title(f"Training comparison:{name} {metric_name} rolling mean")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_overview_rolling(
    runs: list[RunData],
    selected_metrics: dict[str, dict[str, str]],
    out_path: Path,
    smooth: int,
    max_x: Optional[float] = None,
    x_label: str = "Training timesteps",
) -> None:
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

    for ax, (metric_name, metric_columns) in zip(axes_flat, selected_metrics.items()):
        plotted = 0
        for run in runs:
            label = run["label"]
            y_col = metric_columns[label]
            x, y_roll = prepare_rolling_xy(
                run["df"], run["x_col"], y_col, smooth, max_x=max_x
            )

            if len(x) == 0:
                continue

            ax.plot(x, y_roll, linewidth=2.0, label=label)
            plotted += 1

        if plotted == 0:
            ax.set_visible(False)
            continue

        ax.set_title(metric_name)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle(f"Training run comparison, rolling mean window = {smooth}", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare up to three RLlib training runs using rolling-mean plots only."
    )

    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One to three RLlib training run folders.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels for the runs, in the same order as --run-dirs.",
    )
    parser.add_argument(
        "--comparison-name",
        type=str,
        required=True,
        help="Name for output subfolder, e.g. cnn_roles_lstm_s2.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Rolling mean window size. Use 1 for no smoothing.",
    )
    parser.add_argument(
        "--max-x",
        type=float,
        default=None,
        help=(
            "Only use rows up to this x-axis value, e.g. first 5,000,000 "
            "training timesteps. The filter is applied before smoothing."
        ),
    )
    parser.add_argument(
        "--logs-root",
        type=str,
        default="logs_train",
        help="Root folder where pics/ will be created.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output plot format. Default is pdf for thesis-quality vector plots.",
    )

    args = parser.parse_args()

    if not (1 <= len(args.run_dirs) <= 3):
        raise ValueError("Please provide between one and three run directories.")

    if len(args.labels) != len(args.run_dirs):
        raise ValueError("--labels must contain exactly one label per --run-dirs entry.")

    if len(set(args.labels)) != len(args.labels):
        raise ValueError("Labels must be unique.")

    if args.smooth < 1:
        raise ValueError("--smooth must be at least 1.")

    if args.max_x is not None and args.max_x <= 0:
        raise ValueError("--max-x must be positive when provided.")

    logs_root = Path(args.logs_root).resolve()
    out_dir = logs_root / "pics" / args.comparison_name
    out_dir.mkdir(parents=True, exist_ok=True)

    name_run = args.comparison_name

    x_candidates = [
        "timesteps_total",
        "num_env_steps_sampled_lifetime",
        "num_env_steps_sampled",
        "agent_timesteps_total",
        "training_iteration",
    ]

    metric_candidates = {
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
            "info/learner/default_policy/learner_stats/entropy_mean",
            "learner_stats/entropy",
            "learner_stats/entropy_mean",
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
        "vf_explained_var": [
            "info/learner/default_policy/learner_stats/vf_explained_var",
            "learner_stats/vf_explained_var",
        ],
    }

    runs: list[RunData] = []
    for run_dir_str, label in zip(args.run_dirs, args.labels):
        run_dir = Path(run_dir_str).resolve()
        df = load_training_df(run_dir)
        x_col = find_col(df, x_candidates)

        if x_col is None:
            raise ValueError(
                f"Could not find x-axis column for run '{label}'.\n"
                f"Run dir: {run_dir}\n"
                f"Available columns: {list(df.columns)}"
            )

        runs.append(
            {
                "label": label,
                "run_dir": run_dir,
                "df": df,
                "x_col": x_col,
            }
        )

    found_metrics: dict[str, dict[str, str]] = {}

    for metric_name, candidates in metric_candidates.items():
        cols_for_metric: dict[str, str] = {}
        for run in runs:
            y_col = find_col(run["df"], candidates)
            if y_col is None:
                break
            cols_for_metric[run["label"]] = y_col

        if len(cols_for_metric) == len(runs):
            found_metrics[metric_name] = cols_for_metric

    summary: dict[str, Any] = {
        "comparison_name": args.comparison_name,
        "smooth": args.smooth,
        "max_x": args.max_x,
        "plots_dir": str(out_dir),
        "runs": {
            run["label"]: {
                "run_dir": str(run["run_dir"]),
                "x_column": run["x_col"],
                "n_rows": int(len(run["df"])),
            }
            for run in runs
        },
        "metrics": {},
    }

    for metric_name, metric_columns in found_metrics.items():
        out_path = out_dir / f"{metric_name}_rolling_comparison.{args.format}"
        plot_metric_rolling(
            runs=runs,
            metric_name=metric_name,
            metric_columns=metric_columns,
            out_path=out_path,
            smooth=args.smooth,
            max_x=args.max_x,
            name = name_run
        )

        summary["metrics"][metric_name] = {
            run["label"]: summarize_rolling_metric(
                run["df"],
                run["x_col"],
                metric_columns[run["label"]],
                args.smooth,
                max_x=args.max_x,
            )
            for run in runs
        }

    overview_names = [
        "episode_reward_mean",
        "episode_len_mean",
        "entropy",
        "policy_loss",
        "vf_loss",
        "total_loss",
    ]
    overview_metrics = {
        name: found_metrics[name]
        for name in overview_names
        if name in found_metrics
    }

    overview_path = out_dir / f"overview_rolling_comparison.{args.format}"
    plot_overview_rolling(
        runs=runs,
        selected_metrics=overview_metrics,
        out_path=overview_path,
        smooth=args.smooth,
        max_x=args.max_x,
    )

    summary_path = out_dir / "rolling_comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not found_metrics:
        print("No comparable known metric columns found in all runs.")
        for run in runs:
            print(f"\nAvailable columns in {run['label']}:")
            for c in run["df"].columns:
                print(" -", c)
        return

    print(f"Rolling-mean comparison plots written to: {out_dir}")
    if args.max_x is not None:
        print(f"Limited to x <= {args.max_x:g} before smoothing.")
    if overview_metrics:
        print(f"Overview plot written to: {overview_path}")
    print(f"Summary written to: {summary_path}")
    print("\nCompared metrics:")
    for metric_name in found_metrics:
        print(f" - {metric_name}")


if __name__ == "__main__":
    main()
