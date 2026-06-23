import json
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows[])


def make_output_dir(input_path_str: str) -> Path:
    return Path(input_path_str).resolve().parent


def set_scientific_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "savefig.dpi": 300,
        "figure.dpi": 120,
    })


def save_figure(fig, output_dir: Path, filename: str):
    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"- {png_path}")
    print(f"- {pdf_path}")


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "steps", "total_reward", "time_to_find", "unique_cells_total",
        "revisit_fraction_total", "repeated_search_fraction",
        "co_occupancy_fraction", "mean_pairwise_distance",
        "mean_prob_at_visit", "mean_prob_at_search",
        "entropy_start", "entropy_end", "entropy_drop",
        "stay_rate", "episode_idx"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "found" in df.columns:
        df["found"] = df["found"].astype(bool)

    if "time_to_find" in df.columns and "steps" in df.columns:
        df["time_to_find_plot"] = df["time_to_find"].fillna(df["steps"])

    return df


def print_summary(df: pd.DataFrame, output_dir: Path):
    success_rate = df["found"].mean() if "found" in df.columns else float("nan")
    mean_ttf_success = (
        df.loc[df["found"], "time_to_find"].mean()
        if "found" in df.columns and "time_to_find" in df.columns
        else float("nan")
    )
    mean_reward = df["total_reward"].mean() if "total_reward" in df.columns else float("nan")
    mean_revisit = (
        df["revisit_fraction_total"].mean()
        if "revisit_fraction_total" in df.columns
        else float("nan")
    )

    print("Episodes:", len(df))
    print("Success rate:", round(success_rate, 3))
    print("Mean time to find (successful episodes only):", round(mean_ttf_success, 2))
    print("Mean total reward:", round(mean_reward, 3))
    print("Mean revisit fraction:", round(mean_revisit, 3))
    print("Saving plots to:", output_dir)
    print("Saved files:")


def plot_time_to_find_histogram(df: pd.DataFrame, output_dir: Path):
    if "found" not in df.columns or "time_to_find" not in df.columns:
        return

    success_ttf = df.loc[df["found"], "time_to_find"].dropna()
    if len(success_ttf) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(success_ttf, bins=12, edgecolor="black")
    ax.axvline(
        success_ttf.mean(),
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {success_ttf.mean():.2f}"
    )
    ax.set_title("Distribution of Time to Find in Successful Episodes")
    ax.set_xlabel("Time to find (steps)")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)

    save_figure(fig, output_dir, "time_to_find_histogram")


def plot_reward_vs_episode_length(df: pd.DataFrame, output_dir: Path):
    required = {"found", "steps", "total_reward"}
    if not required.issubset(df.columns):
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    success_df = df[df["found"]]
    failure_df = df[~df["found"]]

    if len(success_df) > 0:
        ax.scatter(
            success_df["steps"],
            success_df["total_reward"],
            alpha=0.8,
            marker="o",
            label="Success"
        )

    if len(failure_df) > 0:
        ax.scatter(
            failure_df["steps"],
            failure_df["total_reward"],
            alpha=0.8,
            marker="x",
            label="Failure"
        )

    ax.set_title("Total Reward versus Episode Length")
    ax.set_xlabel("Episode length (steps)")
    ax.set_ylabel("Total reward")
    ax.legend(frameon=False)

    save_figure(fig, output_dir, "reward_vs_episode_length")


def plot_coordination_boxplot(df: pd.DataFrame, output_dir: Path):
    coord_cols = [
        "revisit_fraction_total",
        "repeated_search_fraction",
        "co_occupancy_fraction",
        "stay_rate"
    ]
    available = [c for c in coord_cols if c in df.columns]
    if not available:
        return

    label_map = {
        "revisit_fraction_total": "Revisit fraction",
        "repeated_search_fraction": "Repeated search fraction",
        "co_occupancy_fraction": "Co-occupancy fraction",
        "stay_rate": "Stay rate",
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    values = [df[c].dropna() for c in available]
    plot_labels = [label_map.get(c, c) for c in available]

    ax.boxplot(values, labels=plot_labels)
    ax.set_title("Distribution of Coordination and Inefficiency Metrics")
    ax.set_ylabel("Metric value")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    save_figure(fig, output_dir, "coordination_metrics_boxplot")


def plot_episode_outcomes(df: pd.DataFrame, output_dir: Path):
    required = {"found", "episode_idx", "time_to_find", "steps"}
    if not required.issubset(df.columns):
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    success_df = df[df["found"]]
    failure_df = df[~df["found"]]

    if len(success_df) > 0:
        ax.plot(
            success_df["episode_idx"],
            success_df["time_to_find"],
            marker="o",
            linestyle="-",
            label="Success"
        )

    if len(failure_df) > 0:
        ax.scatter(
            failure_df["episode_idx"],
            failure_df["steps"],
            marker="x",
            s=60,
            label="Failure (time limit reached)"
        )

    ax.set_title("Episode-wise Search Outcome")
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Steps to find / episode length")
    ax.legend(frameon=False)

    save_figure(fig, output_dir, "episode_outcomes")


def plot_repeated_search_vs_outcome(df: pd.DataFrame, output_dir: Path):
    required = {"found", "repeated_search_fraction", "time_to_find_plot"}
    if not required.issubset(df.columns):
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    success_df = df[df["found"]]
    failure_df = df[~df["found"]]

    if len(success_df) > 0:
        ax.scatter(
            success_df["repeated_search_fraction"],
            success_df["time_to_find_plot"],
            alpha=0.8,
            marker="o",
            label="Success"
        )

    if len(failure_df) > 0:
        ax.scatter(
            failure_df["repeated_search_fraction"],
            failure_df["time_to_find_plot"],
            alpha=0.8,
            marker="x",
            label="Failure"
        )

    ax.set_title("Repeated Search Fraction versus Search Outcome")
    ax.set_xlabel("Repeated search fraction")
    ax.set_ylabel("Time to find / episode length (steps)")
    ax.legend(frameon=False)

    save_figure(fig, output_dir, "repeated_search_vs_outcome")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DSSE evaluation results from a JSONL file."
    )
    parser.add_argument("path", help="Path to the JSONL evaluation file")
    args = parser.parse_args()

    set_scientific_style()

    df = load_jsonl(args.path)
    df = preprocess_dataframe(df)
    output_dir = make_output_dir(args.path)

    print_summary(df, output_dir)

    plot_time_to_find_histogram(df, output_dir)
    plot_reward_vs_episode_length(df, output_dir)
    plot_coordination_boxplot(df, output_dir)
    plot_episode_outcomes(df, output_dir)
    plot_repeated_search_vs_outcome(df, output_dir)


if __name__ == "__main__":
    main()
