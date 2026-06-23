import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def load_jsonl(path: str, limit: int = 500) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def make_output_dir(input_path_str: str) -> Path:
    return Path(input_path_str).resolve().parent


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
    "episode_idx",
    "steps",
    "total_reward",
    "time_to_find",
    "unique_cells_total",
    "revisit_steps_total",
    "revisit_fraction_total",
    "mean_revisit_gap_total",
    "median_revisit_gap_total",
    "min_revisit_gap_total",
    "total_search_actions",
    "unique_searched_cells",
    "repeated_search_actions",
    "repeated_search_fraction",
    "mean_search_gap",
    "median_search_gap",
    "min_search_gap",
    "co_occupancy_steps",
    "co_occupancy_fraction",
    "mean_pairwise_distance",
    "min_pairwise_distance",
    "mean_prob_at_visit",
    "mean_prob_at_search",
    "entropy_start",
    "entropy_end",
    "entropy_drop",
    "backtrack_rate",
    "stay_rate",

    # optional role-aware columns
    "role_0_fraction",
    "role_1_fraction",
    "role_entropy_mean_episode",
    "role_switch_rate_episode",
    "role_0_search_rate",
    "role_1_search_rate",
    "role_0_stay_rate",
    "role_1_stay_rate",
    "role_0_visit_prob_mean",
    "role_1_visit_prob_mean",
    "role_pair_dist_0_1_mean",
    "role_pair_close_rate_0_1",
    "role_pair_same_cell_rate_0_1",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "found" in df.columns:
        df["found"] = df["found"].astype(bool)
        df["outcome"] = np.where(df["found"], "Success", "Failure")

    if "has_roles" in df.columns:
        df["has_roles"] = df["has_roles"].astype(bool)
    else:
        df["has_roles"] = False

    if "time_to_find" in df.columns and "steps" in df.columns:
        df["time_to_find_plot"] = df["time_to_find"].fillna(df["steps"])

    if {"role_0_search_rate", "role_1_search_rate"}.issubset(df.columns):
        df["role_search_rate_gap"] = df["role_0_search_rate"] - df["role_1_search_rate"]

    if {"role_0_visit_prob_mean", "role_1_visit_prob_mean"}.issubset(df.columns):
        df["role_visit_prob_gap"] = df["role_0_visit_prob_mean"] - df["role_1_visit_prob_mean"]

    if {"role_0_stay_rate", "role_1_stay_rate"}.issubset(df.columns):
        df["role_stay_rate_gap"] = df["role_0_stay_rate"] - df["role_1_stay_rate"]

    return df

def has_role_data(df: pd.DataFrame) -> bool:
    role_cols = [
        "role_0_fraction",
        "role_1_fraction",
        "role_entropy_mean_episode",
        "role_switch_rate_episode",
        "role_0_search_rate",
        "role_1_search_rate",
        "role_0_stay_rate",
        "role_1_stay_rate",
        "role_0_visit_prob_mean",
        "role_1_visit_prob_mean",
        "role_pair_dist_0_1_mean",
        "role_pair_close_rate_0_1",
        "role_pair_same_cell_rate_0_1",
    ]

    available = [c for c in role_cols if c in df.columns]
    if not available:
        return False

    return df[available].notna().any().any()

COLORS = {
    "primary": "#355C7D",
    "secondary": "#6C8EBF",
    "success": "#2E8B57",
    "failure": "#C44E52",
    "accent": "#7A68A6",
    "neutral": "#5C5C5C",
    "light": "#D9E2F3",
}


def set_scientific_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "savefig.dpi": 300,
        "figure.dpi": 130,
        "axes.axisbelow": True,
    })


def save_figure(fig, output_dir: Path, filename: str):
    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"- {png_path}")
    print(f"- {pdf_path}")


def add_panel_label(ax, label: str):
    ax.text(
        -0.12, 1.04, label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left"
    )


def print_summary(df: pd.DataFrame, output_dir: Path):
    success_rate = df["found"].mean() if "found" in df.columns else float("nan")
    mean_ttf_success = (
        df.loc[df["found"], "time_to_find"].mean()
        if {"found", "time_to_find"}.issubset(df.columns)
        else float("nan")
    )
    median_ttf_success = (
        df.loc[df["found"], "time_to_find"].median()
        if {"found", "time_to_find"}.issubset(df.columns)
        else float("nan")
    )
    mean_reward = df["total_reward"].mean() if "total_reward" in df.columns else float("nan")

    print("Episodes:", len(df))
    print("Success rate:", round(success_rate, 3))
    print("Mean time to find (successful only):", round(mean_ttf_success, 2))
    print("Median time to find (successful only):", round(median_ttf_success, 2))
    print("Mean total reward:", round(mean_reward, 3))
    print("Role-aware file:", bool(has_role_data(df)))
    print("Saving plots to:", output_dir)
    print("Saved files:")


def plot_time_to_find_histogram(df: pd.DataFrame, output_dir: Path):
    if not {"found", "time_to_find"}.issubset(df.columns):
        return

    x = df.loc[df["found"], "time_to_find"].dropna()
    if len(x) == 0:
        return

    mean_x = x.mean()
    median_x = x.median()
    p90_x = np.percentile(x, 90)

    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    sns.histplot(
        x=x,
        bins=min(20, max(8, int(np.sqrt(len(x))))),
        kde=False,
        color=COLORS["secondary"],
        edgecolor="black",
        linewidth=0.7,
        ax=ax,
    )
    ax.axvline(mean_x, color=COLORS["failure"], linestyle="--", label=f"Mean = {mean_x:.2f}")
    ax.axvline(median_x, color=COLORS["primary"], linestyle="-", label=f"Median = {median_x:.2f}")
    ax.axvline(p90_x, color=COLORS["accent"], linestyle=":", label=f"P90 = {p90_x:.2f}")

    ax.set_title("Time to Find in Successful Episodes")
    ax.set_xlabel("Time to find (steps)")
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False)

    save_figure(fig, output_dir, "time_to_find_histogram")


def plot_time_to_find_ecdf(df: pd.DataFrame, output_dir: Path):
    if not {"found", "time_to_find"}.issubset(df.columns):
        return

    x = df.loc[df["found"], "time_to_find"].dropna()
    if len(x) == 0:
        return

    median_x = np.median(x)
    p90_x = np.percentile(x, 90)

    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    sns.ecdfplot(x=x, ax=ax, color=COLORS["primary"], linewidth=1.8)
    ax.axvline(median_x, color=COLORS["failure"], linestyle="--", label=f"Median = {median_x:.2f}")
    ax.axvline(p90_x, color=COLORS["accent"], linestyle=":", label=f"P90 = {p90_x:.2f}")

    ax.set_title("ECDF of Time to Find")
    ax.set_xlabel("Time to find (steps)")
    ax.set_ylabel("Cumulative proportion")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.legend(frameon=False)

    save_figure(fig, output_dir, "time_to_find_ecdf")


def plot_reward_vs_episode_length(df: pd.DataFrame, output_dir: Path):
    if not {"found", "steps", "total_reward"}.issubset(df.columns):
        return

    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    sns.scatterplot(
        data=df,
        x="steps",
        y="total_reward",
        hue="outcome",
        style="outcome",
        palette={"Success": COLORS["success"], "Failure": COLORS["failure"]},
        s=45,
        alpha=0.75,
        ax=ax,
    )

    ax.set_title("Total Reward vs Episode Length")
    ax.set_xlabel("Episode length (steps)")
    ax.set_ylabel("Total reward")
    ax.legend(frameon=False, title=None)

    save_figure(fig, output_dir, "reward_vs_episode_length")


def plot_coordination_boxplot(df: pd.DataFrame, output_dir: Path):
    coord_cols = [
        "revisit_fraction_total",
        "repeated_search_fraction",
        "co_occupancy_fraction",
        "stay_rate",
    ]
    available = [c for c in coord_cols if c in df.columns]
    if not available:
        return

    label_map = {
        "revisit_fraction_total": "Revisit",
        "repeated_search_fraction": "Repeated search",
        "co_occupancy_fraction": "Co-occupancy",
        "stay_rate": "Stay",
    }

    plot_df = df[available].melt(var_name="metric", value_name="value").dropna()
    plot_df["metric"] = plot_df["metric"].map(label_map)

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    sns.boxplot(
        data=plot_df,
        x="metric",
        y="value",
        palette=[COLORS["secondary"], COLORS["accent"], COLORS["primary"], COLORS["success"]],
        width=0.55,
        fliersize=2.5,
        linewidth=1.0,
        ax=ax,
    )

    ax.set_title("Coordination and Inefficiency Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    save_figure(fig, output_dir, "coordination_metrics_boxplot")


def plot_steps_by_outcome(df: pd.DataFrame, output_dir: Path):
    if not {"found", "steps", "outcome"}.issubset(df.columns):
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    sns.boxplot(
        data=df,
        x="outcome",
        y="steps",
        order=["Success", "Failure"],
        palette={"Success": COLORS["success"], "Failure": COLORS["failure"]},
        width=0.5,
        fliersize=2.5,
        linewidth=1.0,
        ax=ax,
    )

    ax.set_title("Episode Length by Outcome")
    ax.set_xlabel("")
    ax.set_ylabel("Steps")

    save_figure(fig, output_dir, "steps_by_outcome_boxplot")


def plot_repeated_search_vs_outcome(df: pd.DataFrame, output_dir: Path):
    if not {"found", "repeated_search_fraction", "time_to_find_plot", "outcome"}.issubset(df.columns):
        return

    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    sns.scatterplot(
        data=df,
        x="repeated_search_fraction",
        y="time_to_find_plot",
        hue="outcome",
        style="outcome",
        palette={"Success": COLORS["success"], "Failure": COLORS["failure"]},
        s=45,
        alpha=0.75,
        ax=ax,
    )

    ax.set_title("Repeated Search Fraction vs Search Outcome")
    ax.set_xlabel("Repeated search fraction")
    ax.set_ylabel("Time to find / episode length (steps)")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(frameon=False, title=None)

    save_figure(fig, output_dir, "repeated_search_vs_outcome")


def plot_probability_at_search_vs_ttf(df: pd.DataFrame, output_dir: Path):
    if not {"found", "mean_prob_at_search", "time_to_find"}.issubset(df.columns):
        return

    success_df = df.loc[df["found"], ["mean_prob_at_search", "time_to_find"]].dropna()
    if len(success_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    sns.scatterplot(
        data=success_df,
        x="mean_prob_at_search",
        y="time_to_find",
        color=COLORS["primary"],
        s=42,
        alpha=0.75,
        ax=ax,
    )

    ax.set_title("Mean Probability at Search vs Time to Find")
    ax.set_xlabel("Mean probability at search")
    ax.set_ylabel("Time to find (steps)")

    save_figure(fig, output_dir, "probability_at_search_vs_ttf")

def plot_metric_by_outcome_violin(df: pd.DataFrame, metric: str, y_label: str, filename: str, output_dir: Path):
    if not {"outcome", metric}.issubset(df.columns):
        return

    plot_df = df[["outcome", metric]].dropna()
    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    sns.violinplot(
        data=plot_df,
        x="outcome",
        y=metric,
        order=["Success", "Failure"],
        inner=None,
        cut=0,
        palette={"Success": COLORS["success"], "Failure": COLORS["failure"]},
        linewidth=1.0,
        ax=ax,
    )
    sns.boxplot(
        data=plot_df,
        x="outcome",
        y=metric,
        order=["Success", "Failure"],
        width=0.22,
        showcaps=True,
        boxprops={"facecolor": "white", "zorder": 3},
        whiskerprops={"linewidth": 1.0},
        medianprops={"color": "black", "linewidth": 1.2},
        fliersize=2,
        ax=ax,
    )

    ax.set_title(f"{y_label} by Outcome")
    ax.set_xlabel("")
    ax.set_ylabel(y_label)

    save_figure(fig, output_dir, filename)


def plot_metric_by_outcome_box(df: pd.DataFrame, metric: str, y_label: str, filename: str, output_dir: Path):
    if not {"outcome", metric}.issubset(df.columns):
        return

    plot_df = df[["outcome", metric]].dropna()
    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    sns.boxplot(
        data=plot_df,
        x="outcome",
        y=metric,
        order=["Success", "Failure"],
        palette={"Success": COLORS["success"], "Failure": COLORS["failure"]},
        width=0.5,
        fliersize=2.5,
        linewidth=1.0,
        ax=ax,
    )

    ax.set_title(f"{y_label} by Outcome")
    ax.set_xlabel("")
    ax.set_ylabel(y_label)

    save_figure(fig, output_dir, filename)


def plot_redundancy_vs_cooccupancy(df: pd.DataFrame, output_dir: Path):
    required = {"repeated_search_fraction", "co_occupancy_fraction", "time_to_find_plot"}
    if not required.issubset(df.columns):
        return

    plot_df = df[list(required) + (["outcome"] if "outcome" in df.columns else [])].dropna()
    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    sns.scatterplot(
        data=plot_df,
        x="repeated_search_fraction",
        y="co_occupancy_fraction",
        hue="time_to_find_plot",
        style="outcome" if "outcome" in plot_df.columns else None,
        palette="viridis",
        s=55,
        alpha=0.8,
        ax=ax,
    )

    ax.set_title("Repeated Search vs Co-occupancy")
    ax.set_xlabel("Repeated search fraction")
    ax.set_ylabel("Co-occupancy fraction")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    save_figure(fig, output_dir, "redundancy_vs_cooccupancy")


def plot_search_actions_vs_unique_searched(df: pd.DataFrame, output_dir: Path):
    required = {"total_search_actions", "unique_searched_cells"}
    if not required.issubset(df.columns):
        return

    extra = ["outcome"] if "outcome" in df.columns else []
    plot_df = df[list(required) + extra].dropna()
    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    sns.scatterplot(
        data=plot_df,
        x="total_search_actions",
        y="unique_searched_cells",
        hue="outcome" if "outcome" in plot_df.columns else None,
        style="outcome" if "outcome" in plot_df.columns else None,
        palette={"Success": COLORS["success"], "Failure": COLORS["failure"]} if "outcome" in plot_df.columns else None,
        s=48,
        alpha=0.75,
        ax=ax,
    )

    ax.set_title("Search Actions vs Unique Searched Cells")
    ax.set_xlabel("Total search actions")
    ax.set_ylabel("Unique searched cells")

    save_figure(fig, output_dir, "search_actions_vs_unique_searched")


def plot_pairwise_distance_vs_cooccupancy(df: pd.DataFrame, output_dir: Path):
    required = {"mean_pairwise_distance", "co_occupancy_fraction"}
    if not required.issubset(df.columns):
        return

    extra = ["outcome"] if "outcome" in df.columns else []
    plot_df = df[list(required) + extra].dropna()
    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    sns.scatterplot(
        data=plot_df,
        x="mean_pairwise_distance",
        y="co_occupancy_fraction",
        hue="outcome" if "outcome" in plot_df.columns else None,
        style="outcome" if "outcome" in plot_df.columns else None,
        palette={"Success": COLORS["success"], "Failure": COLORS["failure"]} if "outcome" in plot_df.columns else None,
        s=48,
        alpha=0.75,
        ax=ax,
    )

    ax.set_title("Pairwise Distance vs Co-occupancy")
    ax.set_xlabel("Mean pairwise distance")
    ax.set_ylabel("Co-occupancy fraction")

    save_figure(fig, output_dir, "pairwise_distance_vs_cooccupancy")


def plot_entropy_drop_vs_ttf(df: pd.DataFrame, output_dir: Path):
    required = {"found", "entropy_drop", "time_to_find"}
    if not required.issubset(df.columns):
        return

    plot_df = df.loc[df["found"], ["entropy_drop", "time_to_find"]].dropna()
    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    sns.scatterplot(
        data=plot_df,
        x="entropy_drop",
        y="time_to_find",
        color=COLORS["accent"],
        s=42,
        alpha=0.75,
        ax=ax,
    )

    ax.set_title("Entropy Drop vs Time to Find")
    ax.set_xlabel("Entropy drop")
    ax.set_ylabel("Time to find (steps)")

    save_figure(fig, output_dir, "entropy_drop_vs_ttf")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    corr_cols = [
        "steps",
        "total_reward",
        "time_to_find_plot",
        "revisit_fraction_total",
        "repeated_search_fraction",
        "co_occupancy_fraction",
        "mean_pairwise_distance",
        "mean_prob_at_visit",
        "mean_prob_at_search",
        "entropy_drop",
        "backtrack_rate",
        "stay_rate",
    ]
    available = [c for c in corr_cols if c in df.columns]
    if len(available) < 3:
        return

    corr = df[available].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8.0, 6.6), constrained_layout=True)
    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=False,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Correlation Matrix of Evaluation Metrics")

    save_figure(fig, output_dir, "correlation_heatmap")


def plot_pairplot_selected(df: pd.DataFrame, output_dir: Path):
    pair_cols = [
        "steps",
        "total_reward",
        "repeated_search_fraction",
        "co_occupancy_fraction",
        "mean_pairwise_distance",
        "mean_prob_at_search",
    ]
    available = [c for c in pair_cols if c in df.columns]
    if len(available) < 3 or "outcome" not in df.columns:
        return

    plot_df = df[available + ["outcome"]].dropna()
    if len(plot_df) < 10:
        return

    g = sns.pairplot(
        plot_df,
        vars=available,
        hue="outcome",
        palette={"Success": COLORS["success"], "Failure": COLORS["failure"]},
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.55, "s": 22},
        diag_kws={"alpha": 0.8, "bins": 15},
        height=2.0,
    )
    g.fig.suptitle("Pairwise Relationships of Selected Metrics", y=1.02)
    png_path = output_dir / "pairplot_selected_metrics.png"
    pdf_path = output_dir / "pairplot_selected_metrics.pdf"
    g.fig.savefig(png_path, bbox_inches="tight", dpi=300)
    g.fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.close(g.fig)
    print(f"- {png_path}")
    print(f"- {pdf_path}")


def plot_four_panel_overview(df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8), constrained_layout=True)

    # A
    ax = axes[0, 0]
    if {"found", "time_to_find"}.issubset(df.columns):
        x = df.loc[df["found"], "time_to_find"].dropna()
        if len(x) > 0:
            sns.histplot(
                x=x,
                bins=min(20, max(8, int(np.sqrt(len(x))))),
                color=COLORS["secondary"],
                edgecolor="black",
                linewidth=0.7,
                ax=ax,
            )
            ax.axvline(x.mean(), color=COLORS["failure"], linestyle="--", label=f"Mean = {x.mean():.2f}")
            ax.axvline(x.median(), color=COLORS["primary"], linestyle="-", label=f"Median = {x.median():.2f}")
            ax.set_title("Time to Find")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Frequency")
            ax.legend(frameon=False)
    add_panel_label(ax, "A")

    # B
    ax = axes[0, 1]
    if {"outcome", "steps"}.issubset(df.columns):
        sns.boxplot(
            data=df,
            x="outcome",
            y="steps",
            order=["Success", "Failure"],
            palette={"Success": COLORS["success"], "Failure": COLORS["failure"]},
            width=0.5,
            fliersize=2.5,
            linewidth=1.0,
            ax=ax,
        )
        ax.set_title("Episode Length by Outcome")
        ax.set_xlabel("")
        ax.set_ylabel("Steps")
    add_panel_label(ax, "B")

    # C
    ax = axes[1, 0]
    coord_cols = [
        "revisit_fraction_total",
        "repeated_search_fraction",
        "co_occupancy_fraction",
        "stay_rate",
    ]
    available = [c for c in coord_cols if c in df.columns]
    label_map = {
        "revisit_fraction_total": "Revisit",
        "repeated_search_fraction": "Repeated search",
        "co_occupancy_fraction": "Co-occupancy",
        "stay_rate": "Stay",
    }
    if available:
        plot_df = df[available].melt(var_name="metric", value_name="value").dropna()
        plot_df["metric"] = plot_df["metric"].map(label_map)
        sns.boxplot(
            data=plot_df,
            x="metric",
            y="value",
            palette=[COLORS["secondary"], COLORS["accent"], COLORS["primary"], COLORS["success"]],
            width=0.55,
            fliersize=2.5,
            linewidth=1.0,
            ax=ax,
        )
        ax.set_title("Coordination Metrics")
        ax.set_xlabel("")
        ax.set_ylabel("Metric value")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    add_panel_label(ax, "C")

    # D
    ax = axes[1, 1]
    if {"repeated_search_fraction", "time_to_find_plot", "outcome"}.issubset(df.columns):
        sns.scatterplot(
            data=df,
            x="repeated_search_fraction",
            y="time_to_find_plot",
            hue="outcome",
            style="outcome",
            palette={"Success": COLORS["success"], "Failure": COLORS["failure"]},
            s=42,
            alpha=0.75,
            ax=ax,
        )
        ax.set_title("Repeated Search vs Outcome")
        ax.set_xlabel("Repeated search fraction")
        ax.set_ylabel("Time to find / episode length")
        ax.legend(frameon=False, title=None)
    add_panel_label(ax, "D")

    save_figure(fig, output_dir, "overview_2x2")


def plot_role_fraction_distribution(df: pd.DataFrame, output_dir: Path):
    required = {"role_0_fraction", "role_1_fraction"}
    if not required.issubset(df.columns):
        return

    plot_df = df[list(required)].dropna()
    if len(plot_df) == 0:
        return

    long_df = plot_df.melt(var_name="role", value_name="fraction")
    long_df["role"] = long_df["role"].map({
        "role_0_fraction": "Role 0",
        "role_1_fraction": "Role 1",
    })

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    sns.violinplot(
        data=long_df,
        x="role",
        y="fraction",
        inner="box",
        cut=0,
        palette=[COLORS["primary"], COLORS["accent"]],
        linewidth=1.0,
        ax=ax,
    )

    ax.set_title("Role Usage Fraction per Episode")
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of steps")

    save_figure(fig, output_dir, "role_fraction_distribution")


def plot_role_entropy_and_switching(df: pd.DataFrame, output_dir: Path):
    required = {"role_entropy_mean_episode", "role_switch_rate_episode"}
    if not required.issubset(df.columns):
        return

    plot_df = df[list(required) + (["outcome"] if "outcome" in df.columns else [])].dropna()
    if len(plot_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)

    sns.histplot(
        data=plot_df,
        x="role_entropy_mean_episode",
        bins=min(20, max(8, int(np.sqrt(len(plot_df))))),
        color=COLORS["secondary"],
        edgecolor="black",
        linewidth=0.7,
        ax=axes[0],
    )
    axes[0].set_title("Role Entropy per Episode")
    axes[0].set_xlabel("Mean role entropy")
    axes[0].set_ylabel("Frequency")

    sns.histplot(
        data=plot_df,
        x="role_switch_rate_episode",
        bins=min(20, max(8, int(np.sqrt(len(plot_df))))),
        color=COLORS["accent"],
        edgecolor="black",
        linewidth=0.7,
        ax=axes[1],
    )
    axes[1].set_title("Role Switch Rate per Episode")
    axes[1].set_xlabel("Switch rate")
    axes[1].set_ylabel("Frequency")

    save_figure(fig, output_dir, "role_entropy_and_switching")


def plot_role_behavior_gaps(df: pd.DataFrame, output_dir: Path):
    gap_cols = {
        "role_search_rate_gap": "Search-rate gap",
        "role_visit_prob_gap": "Visit-probability gap",
        "role_stay_rate_gap": "Stay-rate gap",
    }

    available = [c for c in gap_cols if c in df.columns]
    if not available:
        return

    plot_df = df[available].dropna(how="all")
    if len(plot_df) == 0:
        return

    long_df = plot_df.melt(var_name="metric", value_name="value").dropna()
    long_df["metric"] = long_df["metric"].map(gap_cols)

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    sns.boxplot(
        data=long_df,
        x="metric",
        y="value",
        palette=[COLORS["primary"], COLORS["accent"], COLORS["secondary"]],
        width=0.55,
        fliersize=2.5,
        linewidth=1.0,
        ax=ax,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

    ax.set_title("Role-Specific Behavior Gaps")
    ax.set_xlabel("")
    ax.set_ylabel("Role 0 minus Role 1")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")

    save_figure(fig, output_dir, "role_behavior_gaps")


def plot_role_pair_coordination(df: pd.DataFrame, output_dir: Path):
    required = {
        "role_pair_dist_0_1_mean",
        "role_pair_close_rate_0_1",
        "role_pair_same_cell_rate_0_1",
    }
    if not required.issubset(df.columns):
        return

    plot_df = df[list(required) + (["outcome"] if "outcome" in df.columns else [])].dropna()
    if len(plot_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), constrained_layout=True)

    sns.scatterplot(
        data=plot_df,
        x="role_pair_dist_0_1_mean",
        y="role_pair_close_rate_0_1",
        hue="outcome" if "outcome" in plot_df.columns else None,
        style="outcome" if "outcome" in plot_df.columns else None,
        palette={"Success": COLORS["success"], "Failure": COLORS["failure"]} if "outcome" in plot_df.columns else None,
        s=48,
        alpha=0.75,
        ax=axes[0],
    )
    axes[0].set_title("Role 0–1 Distance vs Close-Contact Rate")
    axes[0].set_xlabel("Mean pair distance (role 0–1)")
    axes[0].set_ylabel("Close rate (dist ≤ 1)")

    sns.scatterplot(
        data=plot_df,
        x="role_pair_dist_0_1_mean",
        y="role_pair_same_cell_rate_0_1",
        hue="outcome" if "outcome" in plot_df.columns else None,
        style="outcome" if "outcome" in plot_df.columns else None,
        palette={"Success": COLORS["success"], "Failure": COLORS["failure"]} if "outcome" in plot_df.columns else None,
        s=48,
        alpha=0.75,
        ax=axes[1],
    )
    axes[1].set_title("Role 0–1 Distance vs Same-Cell Rate")
    axes[1].set_xlabel("Mean pair distance (role 0–1)")
    axes[1].set_ylabel("Same-cell rate")

    save_figure(fig, output_dir, "role_pair_coordination")

def plot_role_overview(df: pd.DataFrame, output_dir: Path):
    required = {
        "role_0_fraction",
        "role_1_fraction",
        "role_entropy_mean_episode",
        "role_switch_rate_episode",
        "role_pair_dist_0_1_mean",
    }
    if not required.intersection(df.columns):
        return

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)

    # A
    ax = axes[0, 0]
    if {"role_0_fraction", "role_1_fraction"}.issubset(df.columns):
        plot_df = df[["role_0_fraction", "role_1_fraction"]].dropna()
        if len(plot_df) > 0:
            long_df = plot_df.melt(var_name="role", value_name="fraction")
            long_df["role"] = long_df["role"].map({
                "role_0_fraction": "Role 0",
                "role_1_fraction": "Role 1",
            })
            sns.boxplot(
                data=long_df,
                x="role",
                y="fraction",
                palette=[COLORS["primary"], COLORS["accent"]],
                ax=ax,
            )
            ax.set_title("Role Usage")
            ax.set_xlabel("")
            ax.set_ylabel("Fraction")
    add_panel_label(ax, "A")

    # B
    ax = axes[0, 1]
    if "role_entropy_mean_episode" in df.columns:
        x = df["role_entropy_mean_episode"].dropna()
        if len(x) > 0:
            sns.histplot(
                x=x,
                bins=min(20, max(8, int(np.sqrt(len(x))))),
                color=COLORS["secondary"],
                edgecolor="black",
                linewidth=0.7,
                ax=ax,
            )
            ax.set_title("Role Entropy")
            ax.set_xlabel("Mean role entropy")
            ax.set_ylabel("Frequency")
    add_panel_label(ax, "B")

    # C
    ax = axes[1, 0]
    if "role_switch_rate_episode" in df.columns:
        x = df["role_switch_rate_episode"].dropna()
        if len(x) > 0:
            sns.histplot(
                x=x,
                bins=min(20, max(8, int(np.sqrt(len(x))))),
                color=COLORS["accent"],
                edgecolor="black",
                linewidth=0.7,
                ax=ax,
            )
            ax.set_title("Role Switching")
            ax.set_xlabel("Switch rate")
            ax.set_ylabel("Frequency")
    add_panel_label(ax, "C")

    # D
    ax = axes[1, 1]
    if {"role_pair_dist_0_1_mean", "outcome"}.issubset(df.columns):
        sns.boxplot(
            data=df.dropna(subset=["role_pair_dist_0_1_mean"]),
            x="outcome",
            y="role_pair_dist_0_1_mean",
            order=["Success", "Failure"] if "Failure" in df["outcome"].values else None,
            palette={"Success": COLORS["success"], "Failure": COLORS["failure"]},
            ax=ax,
        )
        ax.set_title("Role 0–1 Distance by Outcome")
        ax.set_xlabel("")
        ax.set_ylabel("Mean pair distance")
    add_panel_label(ax, "D")

    save_figure(fig, output_dir, "role_overview_2x2")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DSSE evaluation results from a JSONL file."
    )
    parser.add_argument("path", help="Path to the JSONL evaluation file")
    parser.add_argument("--limit", type=int, default=500, help="Number of episodes to use")
    args = parser.parse_args()

    set_scientific_style()

    df = load_jsonl(args.path, limit=args.limit)
    df = preprocess_dataframe(df)
    output_dir = make_output_dir(args.path)

    print_summary(df, output_dir)

    plot_time_to_find_histogram(df, output_dir)
    plot_time_to_find_ecdf(df, output_dir)
    plot_reward_vs_episode_length(df, output_dir)
    plot_coordination_boxplot(df, output_dir)
    plot_steps_by_outcome(df, output_dir)
    plot_repeated_search_vs_outcome(df, output_dir)
    plot_probability_at_search_vs_ttf(df, output_dir)

    plot_metric_by_outcome_violin(
        df, "repeated_search_fraction",
        "Repeated search fraction",
        "repeated_search_fraction_by_outcome",
        output_dir
    )
    plot_metric_by_outcome_violin(
        df, "co_occupancy_fraction",
        "Co-occupancy fraction",
        "cooccupancy_fraction_by_outcome",
        output_dir
    )
    plot_metric_by_outcome_violin(
        df, "mean_prob_at_search",
        "Mean probability at search",
        "mean_prob_at_search_by_outcome",
        output_dir
    )
    plot_metric_by_outcome_violin(
        df, "mean_pairwise_distance",
        "Mean pairwise distance",
        "mean_pairwise_distance_by_outcome",
        output_dir
    )
    plot_metric_by_outcome_box(
        df, "backtrack_rate",
        "Backtrack rate",
        "backtrack_rate_by_outcome",
        output_dir
    )
    plot_metric_by_outcome_box(
        df, "stay_rate",
        "Stay rate",
        "stay_rate_by_outcome",
        output_dir
    )
    plot_metric_by_outcome_box(
        df, "entropy_drop",
        "Entropy drop",
        "entropy_drop_by_outcome",
        output_dir
    )
    plot_metric_by_outcome_box(
        df, "revisit_fraction_total",
        "Revisit fraction",
        "revisit_fraction_by_outcome",
        output_dir
    )

    plot_redundancy_vs_cooccupancy(df, output_dir)
    plot_search_actions_vs_unique_searched(df, output_dir)
    plot_pairwise_distance_vs_cooccupancy(df, output_dir)
    plot_entropy_drop_vs_ttf(df, output_dir)

    plot_correlation_heatmap(df, output_dir)
    plot_pairplot_selected(df, output_dir)

    plot_four_panel_overview(df, output_dir)

    if has_role_data(df):
        plot_role_fraction_distribution(df, output_dir)
        plot_role_entropy_and_switching(df, output_dir)
        plot_role_behavior_gaps(df, output_dir)
        plot_role_pair_coordination(df, output_dir)
        plot_role_overview(df, output_dir)

if __name__ == "__main__":
    main()
