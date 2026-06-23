# plot_single_training_run.py
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


def plot_metric(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    smooth: int = 5,
):
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")

    valid = ~(x.isna() | y.isna())
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to one finished RLlib training run folder",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name for output subfolder, e.g. cnn_s1 or lstm_s3",
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

    run_dir = Path(args.run_dir).resolve()
    logs_root = Path(args.logs_root).resolve()
    out_dir = logs_root / "pics" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_df(run_dir)

    x_col = find_col(df, [
        "timesteps_total",
        "num_env_steps_sampled_lifetime",
        "agent_timesteps_total",
        "training_iteration",
    ])

    if x_col is None:
        raise ValueError(
            f"Could not find a training progress x-axis column. Columns: {list(df.columns)}"
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
        ],
    }

    found_any = False
    summary = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "plots_dir": str(out_dir),
    }

    for metric_name, candidates in metrics.items():
        y_col = find_col(df, candidates)
        if y_col is None:
            continue

        found_any = True
        plot_metric(
            df=df,
            x_col=x_col,
            y_col=y_col,
            title=f"{args.run_name} - {metric_name}",
            out_path=out_dir / f"{metric_name}.png",
            smooth=args.smooth,
        )

        vals = pd.to_numeric(df[y_col], errors="coerce").dropna()
        if not vals.empty:
            summary[metric_name] = {
                "column": y_col,
                "last": float(vals.iloc[-1]),
                "max": float(vals.max()),
                "min": float(vals.min()),
            }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not found_any:
        print("No known metric columns found.")
        print("Available columns:")
        for c in df.columns:
            print(" -", c)
    else:
        print(f"Plots written to: {out_dir}")
        print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
