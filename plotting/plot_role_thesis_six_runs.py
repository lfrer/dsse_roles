#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


X_CANDIDATES = [
    "timesteps_total",
    "num_env_steps_sampled_lifetime",
    "num_env_steps_sampled",
    "agent_timesteps_total",
    "training_iteration",
]

ROLE_METRICS: dict[str, list[str]] = {
    "ablation_action_change_fraction": [
        "derived/r3dm_ablation_action_change_fraction",
    ],
    "marginal_role_entropy": [
        "info/learner/shared_policy/learner_stats/r3dm/marginal_role_entropy",
        "learner_stats/r3dm/marginal_role_entropy",
        "r3dm/marginal_role_entropy",
        "custom_metrics/marginal_role_entropy",
    ],
    "mean_max_role_prob": [
        "info/learner/shared_policy/learner_stats/r3dm/mean_max_role_prob",
        "learner_stats/r3dm/mean_max_role_prob",
        "r3dm/mean_max_role_prob",
        "custom_metrics/mean_max_role_prob",
    ],
}

RAW_ABLATION_SAME_CANDIDATES = [
    "info/learner/shared_policy/learner_stats/r3dm/role_ablation_action_same_fraction_zero_role",
    "learner_stats/r3dm/role_ablation_action_same_fraction_zero_role",
    "r3dm/role_ablation_action_same_fraction_zero_role",
    "custom_metrics/role_ablation_action_same_fraction_zero_role",
]

PLOT_LABELS = {
    "ablation_action_change_fraction": "Ablation action change fraction",
    "marginal_role_entropy": "Marginal role entropy",
    "mean_max_role_prob": "Mean max role probability",
}


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


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    same_col = find_col(df, RAW_ABLATION_SAME_CANDIDATES)
    if same_col is not None:
        same = numeric_series(df, same_col)
        df["derived/r3dm_ablation_action_change_fraction"] = 1.0 - same
    return df


def prepare_xy(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    smooth: int,
    max_x: Optional[float] = None,
) -> tuple[pd.Series, pd.Series]:
    x = numeric_series(df, x_col)
    y = numeric_series(df, y_col)

    valid = ~(x.isna() | y.isna())
    if max_x is not None:
        valid &= x <= max_x

    x = x[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    if len(x) == 0:
        return x, y

    order = x.argsort(kind="mergesort")
    x = x.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)

    if smooth > 1:
        y = y.rolling(window=smooth, min_periods=1).mean()

    return x, y


def load_runs(run_dirs: list[str], labels: list[str]) -> list[dict[str, Any]]:
    runs = []
    for run_dir_str, label in zip(run_dirs, labels):
        run_dir = Path(run_dir_str).resolve()
        df = add_derived_metrics(load_training_df(run_dir))
        x_col = find_col(df, X_CANDIDATES)
        if x_col is None:
            raise ValueError(
                f"Could not find x-axis column for {label} in {run_dir}. "
                f"Available columns: {list(df.columns)}"
            )
        runs.append({"label": label, "run_dir": run_dir, "df": df, "x_col": x_col})
    return runs


def plot_metric_combined(
    runs: list[dict[str, Any]],
    metric_name: str,
    out_path: Path,
    smooth: int,
    max_x: Optional[float] = None,
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    summary: dict[str, Any] = {"metric": metric_name, "out_path": str(out_path), "runs": {}}
    plotted_any = False

    for run in runs:
        df = run["df"]
        y_col = find_col(df, ROLE_METRICS[metric_name])

        if y_col is None:
            summary["runs"][run["label"]] = {
                "run_dir": str(run["run_dir"]),
                "x_column": run["x_col"],
                "y_column": None,
                "n_points": 0,
                "status": "missing metric",
            }
            continue

        x, y = prepare_xy(df, run["x_col"], y_col, smooth=smooth, max_x=max_x)
        if len(x) == 0:
            summary["runs"][run["label"]] = {
                "run_dir": str(run["run_dir"]),
                "x_column": run["x_col"],
                "y_column": y_col,
                "n_points": 0,
                "status": "no valid values",
            }
            continue

        ax.plot(x, y, linewidth=1.8, label=run["label"])
        plotted_any = True

        vals = y.dropna()
        summary["runs"][run["label"]] = {
            "run_dir": str(run["run_dir"]),
            "x_column": run["x_col"],
            "y_column": y_col,
            "n_points": int(len(vals)),
            "last_smoothed": float(vals.iloc[-1]) if not vals.empty else None,
            "max_smoothed": float(vals.max()) if not vals.empty else None,
            "min_smoothed": float(vals.min()) if not vals.empty else None,
            "status": "plotted",
        }

    if metric_name == "mean_max_role_prob":
        ax.axhline(0.5, linestyle="--", linewidth=1.0, alpha=0.6, label="random baseline 0.5")
    elif metric_name == "marginal_role_entropy":
        ax.axhline(0.69314718056, linestyle="--", linewidth=1.0, alpha=0.6, label="ln(2)")

    smooth_text = f"rolling mean, window={smooth}" if smooth > 1 else "unsmoothed values"
    limit_text = f", x ≤ {max_x:g}" if max_x is not None else ""
    ax.set_title(f"{PLOT_LABELS[metric_name]} ({smooth_text}{limit_text})")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel(PLOT_LABELS[metric_name])
    ax.grid(True, alpha=0.3)

    if plotted_any:
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(out_path)

    plt.close(fig)
    summary["created"] = plotted_any
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dirs",
        nargs=6,
        required=True,
        help="Exactly six role training run folders.",
    )
    parser.add_argument(
        "--labels",
        nargs=6,
        required=True,
        help="Exactly six labels, one for each run folder.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="testresults/pics/thesis/role_training",
        help="Output folder for the thesis PDF figures.",
    )
    parser.add_argument("--smooth", type=int, default=5, help="Rolling mean window.")
    parser.add_argument(
        "--max-x",
        type=float,
        default=None,
        help="Only use rows up to this x-axis value before smoothing.",
    )
    parser.add_argument(
        "--print-columns",
        action="store_true",
        help="Print columns for all six runs and exit without plotting.",
    )
    args = parser.parse_args()

    if args.smooth < 1:
        raise ValueError("--smooth must be at least 1.")
    if args.max_x is not None and args.max_x <= 0:
        raise ValueError("--max-x must be positive when provided.")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.run_dirs, args.labels)

    if args.print_columns:
        for run in runs:
            print(f"\n=== {run['label']} ===")
            print(f"Run dir: {run['run_dir']}")
            print(f"Detected x column: {run['x_col']}")
            for c in run["df"].columns:
                print(c)
        return

    all_summaries: dict[str, Any] = {
        "out_dir": str(out_dir),
        "smooth": args.smooth,
        "max_x": args.max_x,
        "figures": {},
    }

    for metric_name in ROLE_METRICS:
        out_path = out_dir / f"{metric_name}.pdf"
        summary = plot_metric_combined(
            runs=runs,
            metric_name=metric_name,
            out_path=out_path,
            smooth=args.smooth,
            max_x=args.max_x,
        )
        all_summaries["figures"][metric_name] = summary

    summary_path = out_dir / "role_thesis_six_runs_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"Wrote thesis figures to: {out_dir}")
    for metric_name, fig_summary in all_summaries["figures"].items():
        status = "created" if fig_summary["created"] else "not created, metric missing in all runs"
        print(f" - {metric_name}.pdf: {status}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
