import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy import stats


def load_jsonl(path: str, limit: int | None = None) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def preprocess_dataframe(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
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
    ]

    df = df.copy()

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "found" in df.columns:
        df["found"] = df["found"].astype(bool)
        df["outcome"] = np.where(df["found"], "Success", "Failure")

    if "time_to_find" in df.columns and "steps" in df.columns:
        df["time_to_find_plot"] = df["time_to_find"].fillna(df["steps"])

    df["model"] = model_name
    return df


def make_output_dir(path_a: str, path_b: str, output_dir: str | None) -> Path:
    if output_dir is not None:
        out = Path(output_dir).resolve()
    else:
        parent = Path(path_a).resolve().parent
        out = parent / f"comparison_{Path(path_a).stem}_vs_{Path(path_b).stem}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_scientific_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.titlesize": 12,
        "grid.linestyle": "--",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "patch.linewidth": 0.8,
        "savefig.dpi": 300,
        "figure.dpi": 130,
        "axes.axisbelow": True,
        "legend.frameon": False,
    })


def get_model_palette(models):
    base = ["#2F4B7C", "#A05195", "#665191", "#D45087"]
    return {m: base[i % len(base)] for i, m in enumerate(models)}


def save_figure(fig, output_dir: Path, filename: str):
    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"- {png_path}")
    print(f"- {pdf_path}")
    return png_path


def add_panel_label(ax, label: str):
    ax.text(
        -0.14, 1.04, label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left"
    )


def annotate_test_p(ax, df: pd.DataFrame, metric: str, paired: bool = True):
    """
    Annotate the top of the plot with a Wilcoxon (paired) or Mann-Whitney p-value
    comparing the two models on `metric`. Falls back silently if not applicable.
    """
    if "model" not in df.columns or metric not in df.columns:
        return
    models = df["model"].dropna().unique()
    if len(models) != 2:
        return
    a = df.loc[df["model"] == models[0], metric].dropna().values
    b = df.loc[df["model"] == models[1], metric].dropna().values
    if len(a) == 0 or len(b) == 0:
        return

    try:
        if paired and len(a) == len(b):
            stat, p = stats.wilcoxon(a, b, zero_method="zsplit")
            test_name = "Wilcoxon"
        else:
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            test_name = "MWU"
    except Exception:
        return

    txt = f"{test_name} p = {p:.2e}" if p < 1e-3 else f"{test_name} p = {p:.3f}"
    ax.text(
        0.98, 0.98, txt,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8.5, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", lw=0.6, alpha=0.85),
    )


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model_name, g in df.groupby("model"):
        row = {
            "model": model_name,
            "episodes": len(g),
            "success_rate": g["found"].mean() if "found" in g.columns else np.nan,
            "mean_total_reward": g["total_reward"].mean() if "total_reward" in g.columns else np.nan,
            "median_total_reward": g["total_reward"].median() if "total_reward" in g.columns else np.nan,
            "mean_steps": g["steps"].mean() if "steps" in g.columns else np.nan,
            "median_steps": g["steps"].median() if "steps" in g.columns else np.nan,
            "mean_ttf_success_only": (
                g.loc[g["found"], "time_to_find"].mean()
                if {"found", "time_to_find"}.issubset(g.columns) else np.nan
            ),
            "median_ttf_success_only": (
                g.loc[g["found"], "time_to_find"].median()
                if {"found", "time_to_find"}.issubset(g.columns) else np.nan
            ),
            "mean_ttf_or_episode_len_all": g["time_to_find_plot"].mean() if "time_to_find_plot" in g.columns else np.nan,
            "mean_revisit_fraction": g["revisit_fraction_total"].mean() if "revisit_fraction_total" in g.columns else np.nan,
            "mean_repeated_search_fraction": g["repeated_search_fraction"].mean() if "repeated_search_fraction" in g.columns else np.nan,
            "mean_co_occupancy_fraction": g["co_occupancy_fraction"].mean() if "co_occupancy_fraction" in g.columns else np.nan,
            "mean_pairwise_distance": g["mean_pairwise_distance"].mean() if "mean_pairwise_distance" in g.columns else np.nan,
            "mean_prob_at_search": g["mean_prob_at_search"].mean() if "mean_prob_at_search" in g.columns else np.nan,
            "mean_prob_at_visit": g["mean_prob_at_visit"].mean() if "mean_prob_at_visit" in g.columns else np.nan,
            "mean_unique_cells_total": g["unique_cells_total"].mean() if "unique_cells_total" in g.columns else np.nan,
            "mean_entropy_drop": g["entropy_drop"].mean() if "entropy_drop" in g.columns else np.nan,
            "mean_backtrack_rate": g["backtrack_rate"].mean() if "backtrack_rate" in g.columns else np.nan,
            "mean_stay_rate": g["stay_rate"].mean() if "stay_rate" in g.columns else np.nan,
        }
        rows.append(row)

    summary = pd.DataFrame(rows)

    if len(summary) == 2:
        a = summary.iloc[0]
        b = summary.iloc[1]
        delta = {"model": "delta(B-A)"}
        for col in summary.columns:
            if col != "model":
                delta[col] = b[col] - a[col]
        summary = pd.concat([summary, pd.DataFrame([delta])], ignore_index=True)

    return summary


def save_summary_table(summary: pd.DataFrame, output_dir: Path):
    csv_path = output_dir / "comparison_summary.csv"
    txt_path = output_dir / "comparison_summary.txt"

    summary.to_csv(csv_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSummary table:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n- {csv_path}")
    print(f"- {txt_path}")


def plot_success_rate_on_ax(ax, df, palette):
    if "found" not in df.columns:
        ax.set_visible(False)
        return
    plot_df = (
        df.groupby("model", as_index=False)["found"]
        .mean()
        .rename(columns={"found": "success_rate"})
    )
    sns.barplot(data=plot_df, x="model", y="success_rate", palette=palette, ax=ax)
    ax.set_title("Success Rate")
    ax.set_xlabel("")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.0)
    for i, row in plot_df.iterrows():
        ax.text(i, row["success_rate"] + 0.02, f"{row['success_rate']:.3f}", ha="center", fontsize=9)


def plot_ecdf_ttf_on_ax(ax, df, palette):
    if not {"found", "time_to_find", "model"}.issubset(df.columns):
        ax.set_visible(False)
        return
    plot_df = df.loc[df["found"], ["time_to_find", "model"]].dropna()
    if len(plot_df) == 0:
        ax.set_visible(False)
        return
    sns.ecdfplot(data=plot_df, x="time_to_find", hue="model", palette=palette, linewidth=1.6, ax=ax)
    ax.set_title("ECDF of Time to Find")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cumulative proportion")
    ax.set_ylim(0, 1.02)
    leg = ax.get_legend()
    if leg:
        leg.set_title(None)


def plot_ttf_histogram_on_ax(ax, df, palette):
    if not {"found", "time_to_find", "model"}.issubset(df.columns):
        ax.set_visible(False)
        return
    plot_df = df.loc[df["found"], ["time_to_find", "model"]].dropna()
    if len(plot_df) == 0:
        ax.set_visible(False)
        return
    sns.histplot(
        data=plot_df, x="time_to_find", hue="model", palette=palette,
        bins=20, kde=False, common_norm=False, alpha=0.45,
        element="step", ax=ax,
    )
    ax.set_title("Time to Find (Successful Episodes)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def plot_reward_box_on_ax(ax, df, palette):
    if not {"model", "total_reward"}.issubset(df.columns):
        ax.set_visible(False)
        return
    plot_df = df[["model", "total_reward"]].dropna()
    if len(plot_df) == 0:
        ax.set_visible(False)
        return
    sns.boxplot(data=plot_df, x="model", y="total_reward", palette=palette, width=0.5, fliersize=2.0, linewidth=0.9, ax=ax)
    ax.set_title("Total Reward")
    ax.set_xlabel("")
    ax.set_ylabel("Reward")


def plot_steps_box_on_ax(ax, df, palette):
    if not {"model", "steps"}.issubset(df.columns):
        ax.set_visible(False)
        return
    plot_df = df[["model", "steps"]].dropna()
    if len(plot_df) == 0:
        ax.set_visible(False)
        return
    sns.boxplot(data=plot_df, x="model", y="steps", palette=palette, width=0.5, fliersize=2.0, linewidth=0.9, ax=ax)
    ax.set_title("Episode Length")
    ax.set_xlabel("")
    ax.set_ylabel("Steps")


def plot_outcome_split_ttf_on_ax(ax, df, palette=None):
    if not {"model", "outcome", "time_to_find_plot"}.issubset(df.columns):
        ax.set_visible(False)
        return
    plot_df = df[["model", "outcome", "time_to_find_plot"]].dropna()
    if len(plot_df) == 0:
        ax.set_visible(False)
        return
    sns.boxplot(
        data=plot_df, x="model", y="time_to_find_plot", hue="outcome",
        palette={"Success": "#2E8B57", "Failure": "#C44E52"},
        width=0.6, fliersize=2.0, linewidth=0.9, ax=ax,
    )
    ax.set_title("Episode Length by Outcome")
    ax.set_xlabel("")
    ax.set_ylabel("Steps (TTF or truncation)")
    leg = ax.get_legend()
    if leg:
        leg.set_title(None)


def plot_metric_box_on_ax(ax, df, metric, title, y_label, palette, annotate_p: bool = False):
    if not {"model", metric}.issubset(df.columns):
        ax.set_visible(False)
        return
    plot_df = df[["model", metric]].dropna()
    if len(plot_df) == 0:
        ax.set_visible(False)
        return
    sns.boxplot(data=plot_df, x="model", y=metric, palette=palette, width=0.5, fliersize=2.0, linewidth=0.9, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    if annotate_p:
        annotate_test_p(ax, df, metric)


def plot_metric_violin_on_ax(ax, df, metric, title, y_label, palette, annotate_p: bool = False):
    if not {"model", metric}.issubset(df.columns):
        ax.set_visible(False)
        return
    plot_df = df[["model", metric]].dropna()
    if len(plot_df) == 0:
        ax.set_visible(False)
        return
    sns.violinplot(
        data=plot_df, x="model", y=metric, palette=palette,
        inner=None, cut=0, linewidth=0.9, ax=ax
    )
    sns.boxplot(
        data=plot_df, x="model", y=metric,
        width=0.20, showcaps=True,
        boxprops={"facecolor": "white", "zorder": 3},
        whiskerprops={"linewidth": 0.9},
        medianprops={"color": "black", "linewidth": 1.1},
        fliersize=1.8, ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    if annotate_p:
        annotate_test_p(ax, df, metric)


def plot_mean_metric_bars_on_ax(ax, df, metrics, palette):
    available = [m for m in metrics if m in df.columns]
    if not available:
        ax.set_visible(False)
        return

    nice_names = {
        "revisit_fraction_total": "Revisit",
        "repeated_search_fraction": "Repeated search",
        "co_occupancy_fraction": "Co-occupancy",
        "stay_rate": "Stay",
        "backtrack_rate": "Backtrack",
        "entropy_drop": "Entropy drop",
        "mean_pairwise_distance": "Pairwise distance",
        "mean_prob_at_search": "Prob at search",
        "mean_prob_at_visit": "Prob at visit",
    }

    plot_df = (
        df.groupby("model")[available]
        .mean(numeric_only=True)
        .reset_index()
        .melt(id_vars="model", var_name="metric", value_name="value")
    )
    plot_df["metric"] = plot_df["metric"].map(lambda x: nice_names.get(x, x))

    sns.barplot(data=plot_df, x="metric", y="value", hue="model", palette=palette, ax=ax)
    ax.set_title("Mean Behaviour / Coordination Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Mean value")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    leg = ax.get_legend()
    if leg:
        leg.set_title(None)


def plot_effect_sizes_on_ax(ax, df, metrics, n_boot: int = 2000, seed: int = 0):
    """
    Forest plot of standardized mean differences (B - A)/pooled_SD with
    95% bootstrap CIs. Zero line = no difference.
    """
    available = [(m, lab) for m, lab in metrics if m in df.columns]
    if not available:
        ax.set_visible(False)
        return

    models = list(df["model"].dropna().unique())
    if len(models) != 2:
        ax.set_visible(False)
        return
    A, B = models
    rng = np.random.default_rng(seed)

    rows = []
    for metric, label in available:
        a = df.loc[df["model"] == A, metric].dropna().values
        b = df.loc[df["model"] == B, metric].dropna().values
        if len(a) == 0 or len(b) == 0:
            continue
        all_vals = np.concatenate([a, b])
        sd = all_vals.std()
        if not np.isfinite(sd) or sd == 0:
            continue

        diffs = np.empty(n_boot)
        for k in range(n_boot):
            sa = rng.choice(a, size=len(a), replace=True)
            sb = rng.choice(b, size=len(b), replace=True)
            diffs[k] = (sb.mean() - sa.mean()) / sd

        rows.append({
            "metric": label,
            "diff": float(np.mean(diffs)),
            "lo": float(np.percentile(diffs, 2.5)),
            "hi": float(np.percentile(diffs, 97.5)),
        })

    if not rows:
        ax.set_visible(False)
        return

    plot_df = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df))
    ax.errorbar(
        plot_df["diff"], y,
        xerr=[plot_df["diff"] - plot_df["lo"], plot_df["hi"] - plot_df["diff"]],
        fmt="o", color="#2F4B7C", ecolor="#2F4B7C",
        capsize=3, lw=1.2, markersize=5,
    )
    ax.axvline(0, ls="--", lw=1.0, color="#888888")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["metric"])
    ax.set_xlabel(f"Standardized mean diff  ({B} − {A}) / pooled SD")
    ax.set_title("Effect Sizes (95% bootstrap CI)")


def plot_paired_scatter_on_ax(ax, df, metric, label):
    """
    Per-episode A vs B scatter, paired by episode_idx.
    """
    if not {"model", metric, "episode_idx"}.issubset(df.columns):
        ax.set_visible(False)
        return
    models = list(df["model"].dropna().unique())
    if len(models) != 2:
        ax.set_visible(False)
        return
    a = df.loc[df["model"] == models[0], ["episode_idx", metric]].rename(columns={metric: "a"})
    b = df.loc[df["model"] == models[1], ["episode_idx", metric]].rename(columns={metric: "b"})
    merged = a.merge(b, on="episode_idx", how="inner").dropna()
    if len(merged) == 0:
        ax.set_visible(False)
        return

    ax.scatter(merged["a"], merged["b"], s=14, alpha=0.45, color="#2F4B7C", edgecolor="none")

    lo = float(min(merged["a"].min(), merged["b"].min()))
    hi = float(max(merged["a"].max(), merged["b"].max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            ls="--", lw=1.0, color="#888888", label="y = x")
    ax.axhline(merged["b"].mean(), color="#A05195", lw=0.8, ls=":", alpha=0.7)
    ax.axvline(merged["a"].mean(), color="#2F4B7C", lw=0.8, ls=":", alpha=0.7)

    ax.set_xlabel(f"{label}  ({models[0]})")
    ax.set_ylabel(f"{label}  ({models[1]})")
    ax.set_title(f"Paired: {label}")
    ax.legend(loc="upper left", frameon=True)

    n = len(merged)
    b_better = (merged["b"] > merged["a"]).sum()
    a_better = (merged["a"] > merged["b"]).sum()
    txt = f"{models[1]} > {models[0]}: {b_better}/{n}\n{models[0]} > {models[1]}: {a_better}/{n}"
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", lw=0.6, alpha=0.85),
    )


def plot_search_efficiency_scatter_on_ax(ax, df, palette):
    needed = {"model", "total_search_actions", "unique_searched_cells"}
    if not needed.issubset(df.columns):
        ax.set_visible(False)
        return
    plot_df = df[list(needed)].dropna()
    if len(plot_df) == 0:
        ax.set_visible(False)
        return
    sns.scatterplot(
        data=plot_df, x="total_search_actions", y="unique_searched_cells",
        hue="model", palette=palette,
        s=18, alpha=0.55, edgecolor="none", ax=ax,
    )
    lo = 0
    hi = max(plot_df["total_search_actions"].max(), plot_df["unique_searched_cells"].max())
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="#888888", label="no-repeat line")
    ax.set_xlabel("Total search actions")
    ax.set_ylabel("Unique cells searched")
    ax.set_title("Search Efficiency")
    leg = ax.get_legend()
    if leg:
        leg.set_title(None)


def plot_prob_vs_repeat_on_ax(ax, df, palette):
    needed = {"model", "mean_prob_at_search", "repeated_search_fraction"}
    if not needed.issubset(df.columns):
        ax.set_visible(False)
        return
    plot_df = df[list(needed)].dropna()
    if len(plot_df) == 0:
        ax.set_visible(False)
        return
    sns.scatterplot(
        data=plot_df, x="repeated_search_fraction", y="mean_prob_at_search",
        hue="model", palette=palette,
        s=18, alpha=0.55, edgecolor="none", ax=ax,
    )
    for m, sub in plot_df.groupby("model"):
        ax.scatter(
            sub["repeated_search_fraction"].mean(),
            sub["mean_prob_at_search"].mean(),
            marker="X", s=140, edgecolor="black", linewidth=0.8,
            label=f"{m} mean", zorder=5,
            color=palette.get(m, None) if isinstance(palette, dict) else None,
        )
    ax.set_xlabel("Repeated search fraction (lower = better)")
    ax.set_ylabel("Mean prob at search (higher = better)")
    ax.set_title("Search Quality vs Redundancy")
    leg = ax.get_legend()
    if leg:
        leg.set_title(None)


def plot_coverage_vs_revisit_on_ax(ax, df, palette):
    needed = {"model", "unique_cells_total", "revisit_fraction_total"}
    if not needed.issubset(df.columns):
        ax.set_visible(False)
        return
    plot_df = df[list(needed)].dropna()
    if len(plot_df) == 0:
        ax.set_visible(False)
        return
    sns.scatterplot(
        data=plot_df, x="unique_cells_total", y="revisit_fraction_total",
        hue="model", palette=palette,
        s=18, alpha=0.55, edgecolor="none", ax=ax,
    )
    for m, sub in plot_df.groupby("model"):
        ax.scatter(
            sub["unique_cells_total"].mean(),
            sub["revisit_fraction_total"].mean(),
            marker="X", s=140, edgecolor="black", linewidth=0.8,
            label=f"{m} mean", zorder=5,
            color=palette.get(m, None) if isinstance(palette, dict) else None,
        )
    ax.set_xlabel("Unique cells visited")
    ax.set_ylabel("Revisit fraction")
    ax.set_title("Coverage vs Revisits")
    leg = ax.get_legend()
    if leg:
        leg.set_title(None)


def plot_stay_vs_search_on_ax(ax, df, palette):
    needed = {"model", "stay_rate", "total_search_actions", "steps"}
    if not needed.issubset(df.columns):
        ax.set_visible(False)
        return
    plot_df = df[list(needed)].dropna().copy()
    plot_df = plot_df[plot_df["steps"] > 0]
    if len(plot_df) == 0:
        ax.set_visible(False)
        return
    plot_df["search_per_step"] = plot_df["total_search_actions"] / plot_df["steps"]
    sns.scatterplot(
        data=plot_df, x="stay_rate", y="search_per_step",
        hue="model", palette=palette,
        s=18, alpha=0.55, edgecolor="none", ax=ax,
    )
    ax.set_xlabel("Stay rate")
    ax.set_ylabel("Search actions / step")
    ax.set_title("Camping Signature")
    leg = ax.get_legend()
    if leg:
        leg.set_title(None)


def plot_correlation_heatmap_on_ax(ax, df, model_name):
    metric_set = [
        "steps", "time_to_find_plot", "total_reward",
        "unique_cells_total", "revisit_fraction_total",
        "total_search_actions", "repeated_search_fraction",
        "co_occupancy_fraction", "mean_pairwise_distance",
        "mean_prob_at_visit", "mean_prob_at_search",
        "entropy_drop", "stay_rate", "backtrack_rate",
    ]
    available = [m for m in metric_set if m in df.columns]
    if len(available) < 3:
        ax.set_visible(False)
        return
    sub = df.loc[df["model"] == model_name, available].copy()
    if sub.empty:
        ax.set_visible(False)
        return
    corr = sub.corr(numeric_only=True)
    sns.heatmap(
        corr, ax=ax, cmap="vlag", center=0, vmin=-1, vmax=1,
        square=True, cbar=True, linewidths=0.4, linecolor="white",
        xticklabels=corr.columns, yticklabels=corr.index,
    )
    ax.set_title(f"Correlation — {model_name}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def export_main_single_plots(df: pd.DataFrame, output_dir: Path):
    palette = get_model_palette(df["model"].unique())
    saved = []

    fig, ax = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)
    plot_success_rate_on_ax(ax, df, palette)
    saved.append(save_figure(fig, output_dir, "compare_success_rate"))

    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    plot_ecdf_ttf_on_ax(ax, df, palette)
    saved.append(save_figure(fig, output_dir, "compare_time_to_find_ecdf"))

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    plot_ttf_histogram_on_ax(ax, df, palette)
    saved.append(save_figure(fig, output_dir, "compare_time_to_find_histogram"))

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    plot_reward_box_on_ax(ax, df, palette)
    saved.append(save_figure(fig, output_dir, "compare_total_reward_boxplot"))

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    plot_steps_box_on_ax(ax, df, palette)
    saved.append(save_figure(fig, output_dir, "compare_steps_boxplot"))

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    plot_outcome_split_ttf_on_ax(ax, df, palette)
    saved.append(save_figure(fig, output_dir, "compare_outcome_split_ttf"))

    violin_specs = [
        ("mean_prob_at_search", "Mean Probability at Search", "Probability", "compare_mean_prob_at_search"),
        ("mean_prob_at_visit", "Mean Probability at Visit", "Probability", "compare_mean_prob_at_visit"),
        ("repeated_search_fraction", "Repeated Search Fraction", "Fraction", "compare_repeated_search_fraction"),
        ("co_occupancy_fraction", "Co-occupancy Fraction", "Fraction", "compare_cooccupancy_fraction"),
        ("mean_pairwise_distance", "Mean Pairwise Distance", "Distance", "compare_mean_pairwise_distance"),
        ("unique_cells_total", "Unique Cells Visited", "Cells", "compare_unique_cells_total"),
    ]
    for metric, title, ylabel, filename in violin_specs:
        fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
        plot_metric_violin_on_ax(ax, df, metric, title, ylabel, palette, annotate_p=True)
        saved.append(save_figure(fig, output_dir, filename))

    box_specs = [
        ("entropy_drop", "Entropy Drop", "Entropy drop", "compare_entropy_drop"),
        ("backtrack_rate", "Backtrack Rate", "Rate", "compare_backtrack_rate"),
        ("stay_rate", "Stay Rate", "Rate", "compare_stay_rate"),
        ("revisit_fraction_total", "Revisit Fraction", "Fraction", "compare_revisit_fraction"),
    ]
    for metric, title, ylabel, filename in box_specs:
        fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
        plot_metric_box_on_ax(ax, df, metric, title, ylabel, palette, annotate_p=True)
        saved.append(save_figure(fig, output_dir, filename))

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    plot_mean_metric_bars_on_ax(
        ax, df,
        metrics=[
            "revisit_fraction_total",
            "repeated_search_fraction",
            "co_occupancy_fraction",
            "stay_rate",
            "backtrack_rate",
            "entropy_drop",
            "mean_pairwise_distance",
            "mean_prob_at_search",
            "mean_prob_at_visit",
        ],
        palette=palette,
    )
    saved.append(save_figure(fig, output_dir, "compare_mean_metrics_bar"))

    fig, ax = plt.subplots(figsize=(7.4, 5.6), constrained_layout=True)
    plot_effect_sizes_on_ax(
        ax, df,
        metrics=[
            ("mean_prob_at_search", "Prob at search"),
            ("mean_prob_at_visit", "Prob at visit"),
            ("repeated_search_fraction", "Repeated search frac."),
            ("revisit_fraction_total", "Revisit frac."),
            ("unique_cells_total", "Unique cells visited"),
            ("co_occupancy_fraction", "Co-occupancy frac."),
            ("mean_pairwise_distance", "Pairwise distance"),
            ("stay_rate", "Stay rate"),
            ("backtrack_rate", "Backtrack rate"),
            ("entropy_drop", "Entropy drop"),
            ("total_reward", "Total reward"),
        ],
    )
    saved.append(save_figure(fig, output_dir, "compare_effect_sizes_forest"))

    fig, ax = plt.subplots(figsize=(6.6, 5.0), constrained_layout=True)
    plot_search_efficiency_scatter_on_ax(ax, df, palette)
    saved.append(save_figure(fig, output_dir, "compare_search_efficiency_scatter"))

    fig, ax = plt.subplots(figsize=(6.6, 5.0), constrained_layout=True)
    plot_prob_vs_repeat_on_ax(ax, df, palette)
    saved.append(save_figure(fig, output_dir, "compare_prob_vs_repeat_tradeoff"))

    fig, ax = plt.subplots(figsize=(6.6, 5.0), constrained_layout=True)
    plot_coverage_vs_revisit_on_ax(ax, df, palette)
    saved.append(save_figure(fig, output_dir, "compare_coverage_vs_revisit"))

    fig, ax = plt.subplots(figsize=(6.6, 5.0), constrained_layout=True)
    plot_stay_vs_search_on_ax(ax, df, palette)
    saved.append(save_figure(fig, output_dir, "compare_stay_vs_search_scatter"))

    paired_specs = [
        ("time_to_find_plot", "Time to find / truncation", "compare_paired_ttf"),
        ("mean_prob_at_search", "Prob at search", "compare_paired_prob_at_search"),
        ("unique_cells_total", "Unique cells", "compare_paired_unique_cells"),
    ]
    for metric, label, filename in paired_specs:
        fig, ax = plt.subplots(figsize=(5.6, 5.4), constrained_layout=True)
        plot_paired_scatter_on_ax(ax, df, metric, label)
        saved.append(save_figure(fig, output_dir, filename))

    models = list(df["model"].dropna().unique())
    if len(models) >= 1:
        fig, axes = plt.subplots(1, len(models),
                                 figsize=(5.5 * len(models), 5.0),
                                 constrained_layout=True)
        if len(models) == 1:
            axes = [axes]
        for ax, m in zip(axes, models):
            plot_correlation_heatmap_on_ax(ax, df, m)
        saved.append(save_figure(fig, output_dir, "compare_correlation_heatmaps"))

    return saved


def plot_overview_performance(df: pd.DataFrame, output_dir: Path):
    palette = get_model_palette(df["model"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=True)

    plot_success_rate_on_ax(axes[0, 0], df, palette)
    add_panel_label(axes[0, 0], "A")

    plot_ecdf_ttf_on_ax(axes[0, 1], df, palette)
    add_panel_label(axes[0, 1], "B")

    plot_reward_box_on_ax(axes[1, 0], df, palette)
    add_panel_label(axes[1, 0], "C")

    plot_steps_box_on_ax(axes[1, 1], df, palette)
    add_panel_label(axes[1, 1], "D")

    return save_figure(fig, output_dir, "overview_performance_2x2")


def plot_overview_coordination(df: pd.DataFrame, output_dir: Path):
    palette = get_model_palette(df["model"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=True)

    plot_metric_violin_on_ax(
        axes[0, 0], df, "repeated_search_fraction",
        "Repeated Search Fraction", "Fraction", palette
    )
    add_panel_label(axes[0, 0], "A")

    plot_metric_violin_on_ax(
        axes[0, 1], df, "co_occupancy_fraction",
        "Co-occupancy Fraction", "Fraction", palette
    )
    add_panel_label(axes[0, 1], "B")

    plot_metric_box_on_ax(
        axes[1, 0], df, "revisit_fraction_total",
        "Revisit Fraction", "Fraction", palette
    )
    add_panel_label(axes[1, 0], "C")

    plot_metric_box_on_ax(
        axes[1, 1], df, "mean_pairwise_distance",
        "Mean Pairwise Distance", "Distance", palette
    )
    add_panel_label(axes[1, 1], "D")

    return save_figure(fig, output_dir, "overview_coordination_2x2")


def plot_overview_efficiency(df: pd.DataFrame, output_dir: Path):
    palette = get_model_palette(df["model"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=True)

    plot_ecdf_ttf_on_ax(axes[0, 0], df, palette)
    add_panel_label(axes[0, 0], "A")

    plot_steps_box_on_ax(axes[0, 1], df, palette)
    add_panel_label(axes[0, 1], "B")

    plot_metric_box_on_ax(
        axes[1, 0], df, "entropy_drop",
        "Entropy Drop", "Entropy drop", palette
    )
    add_panel_label(axes[1, 0], "C")

    plot_metric_box_on_ax(
        axes[1, 1], df, "backtrack_rate",
        "Backtrack Rate", "Rate", palette
    )
    add_panel_label(axes[1, 1], "D")

    return save_figure(fig, output_dir, "overview_efficiency_2x2")


def plot_overview_mixed(df: pd.DataFrame, output_dir: Path):
    palette = get_model_palette(df["model"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=True)

    plot_success_rate_on_ax(axes[0, 0], df, palette)
    add_panel_label(axes[0, 0], "A")

    plot_reward_box_on_ax(axes[0, 1], df, palette)
    add_panel_label(axes[0, 1], "B")

    plot_metric_violin_on_ax(
        axes[1, 0], df, "repeated_search_fraction",
        "Repeated Search Fraction", "Fraction", palette
    )
    add_panel_label(axes[1, 0], "C")

    plot_metric_box_on_ax(
        axes[1, 1], df, "mean_pairwise_distance",
        "Mean Pairwise Distance", "Distance", palette
    )
    add_panel_label(axes[1, 1], "D")

    return save_figure(fig, output_dir, "overview_mixed_2x2")


def plot_overview_metric_bars(df: pd.DataFrame, output_dir: Path):
    palette = get_model_palette(df["model"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8), constrained_layout=True)

    plot_success_rate_on_ax(axes[0, 0], df, palette)
    add_panel_label(axes[0, 0], "A")

    plot_ecdf_ttf_on_ax(axes[0, 1], df, palette)
    add_panel_label(axes[0, 1], "B")

    plot_reward_box_on_ax(axes[1, 0], df, palette)
    add_panel_label(axes[1, 0], "C")

    plot_mean_metric_bars_on_ax(
        axes[1, 1], df,
        metrics=[
            "revisit_fraction_total",
            "repeated_search_fraction",
            "co_occupancy_fraction",
            "stay_rate",
            "backtrack_rate",
            "entropy_drop",
            "mean_pairwise_distance",
        ],
        palette=palette,
    )
    add_panel_label(axes[1, 1], "D")

    return save_figure(fig, output_dir, "overview_with_metric_bars_2x2")


def plot_overview_for_thesis(df: pd.DataFrame, output_dir: Path):

    palette = get_model_palette(df["model"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4), constrained_layout=True)

    plot_success_rate_on_ax(axes[0, 0], df, palette)
    add_panel_label(axes[0, 0], "A")

    plot_effect_sizes_on_ax(
        axes[0, 1], df,
        metrics=[
            ("mean_prob_at_search", "Prob at search"),
            ("mean_prob_at_visit", "Prob at visit"),
            ("repeated_search_fraction", "Repeated search frac."),
            ("revisit_fraction_total", "Revisit frac."),
            ("unique_cells_total", "Unique cells"),
            ("co_occupancy_fraction", "Co-occupancy frac."),
            ("mean_pairwise_distance", "Pairwise distance"),
            ("stay_rate", "Stay rate"),
            ("entropy_drop", "Entropy drop"),
            ("total_reward", "Total reward"),
        ],
    )
    add_panel_label(axes[0, 1], "B")

    plot_metric_violin_on_ax(
        axes[1, 0], df, "mean_prob_at_search",
        "Mean Probability at Search", "Probability", palette,
        annotate_p=True,
    )
    add_panel_label(axes[1, 0], "C")

    plot_coverage_vs_revisit_on_ax(axes[1, 1], df, palette)
    add_panel_label(axes[1, 1], "D")

    return save_figure(fig, output_dir, "overview_for_thesis_2x2")


def build_contact_sheet(image_paths: list, output_dir: Path,
                        filename: str = "all_plots_contact_sheet"):
    valid = [Path(p) for p in image_paths if p is not None and Path(p).exists()]
    if not valid:
        print("No images to combine into a contact sheet.")
        return None

    n = len(valid)
    if n <= 4:
        cols = 2
    elif n <= 12:
        cols = 3
    else:
        cols = 4
    rows = (n + cols - 1) // cols

    fig_w = 4.2 * cols
    fig_h = 3.2 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), constrained_layout=True)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    for idx, path in enumerate(valid):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        try:
            img = mpimg.imread(path)
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed to load:\n{path.name}\n{e}",
                    ha="center", va="center", fontsize=8, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(path.stem, fontsize=9)

    for k in range(len(valid), rows * cols):
        r, c = divmod(k, cols)
        axes[r, c].set_visible(False)

    fig.suptitle("DSSE Comparison — All Plots", fontsize=14)
    out_png = output_dir / f"{filename}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nContact sheet saved:")
    print(f"- {out_png}")
    return out_png


def main():
    parser = argparse.ArgumentParser(description="Compare two DSSE evaluation JSONL files.")
    parser.add_argument("path_a", help="Path to first JSONL evaluation file")
    parser.add_argument("path_b", help="Path to second JSONL evaluation file")
    parser.add_argument("--label-a", default="Model A", help="Display label for first model")
    parser.add_argument("--label-b", default="Model B", help="Display label for second model")
    parser.add_argument("--limit", type=int, default=None, help="Optional max episodes per file")
    parser.add_argument("--output-dir", default=None, help="Optional output directory")
    parser.add_argument("--no-contact-sheet", action="store_true",
                        help="Skip the combined PNG.")
    args = parser.parse_args()

    set_scientific_style()

    df_a = preprocess_dataframe(load_jsonl(args.path_a, args.limit), args.label_a)
    df_b = preprocess_dataframe(load_jsonl(args.path_b, args.limit), args.label_b)
    df = pd.concat([df_a, df_b], ignore_index=True)

    output_dir = make_output_dir(args.path_a, args.path_b, args.output_dir)

    print("Saving comparison outputs to:", output_dir)
    print("Saved files:")

    summary = build_summary_table(df)
    save_summary_table(summary, output_dir)

    saved_pngs = []
    saved_pngs.extend(export_main_single_plots(df, output_dir))

    saved_pngs.append(plot_overview_performance(df, output_dir))
    saved_pngs.append(plot_overview_coordination(df, output_dir))
    saved_pngs.append(plot_overview_efficiency(df, output_dir))
    saved_pngs.append(plot_overview_mixed(df, output_dir))
    saved_pngs.append(plot_overview_metric_bars(df, output_dir))
    saved_pngs.append(plot_overview_for_thesis(df, output_dir))

    if not args.no_contact_sheet:
        build_contact_sheet(saved_pngs, output_dir)


if __name__ == "__main__":
    main()
