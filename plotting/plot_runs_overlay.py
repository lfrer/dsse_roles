from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
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


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
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
            df, "info/learner/shared_policy/learner_stats/r3dm/role_ablation_action_same_fraction_zero_role"
        )
        df["derived/r3dm_ablation_action_change_fraction"] = 1.0 - same

    if "info/learner/shared_policy/learner_stats/r3dm/role_ablation_relative_logit_delta" in df.columns:
        rel = numeric_series(
            df, "info/learner/shared_policy/learner_stats/r3dm/role_ablation_relative_logit_delta"
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


class RunData:
    def __init__(
        self,
        label: str,
        df: pd.DataFrame,
        x_col: str,
        scenario: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.label = label
        self.df = df
        self.x_col = x_col
        self.scenario = scenario
        self.model = model


def parse_run_arg(arg: str) -> Tuple[str, str]:
    if "=" not in arg:
        raise ValueError(
            f"--run argument must be 'label=path', got: {arg}"
        )
    label, path = arg.split("=", 1)
    return label.strip(), path.strip()


def load_runs(
    specs: List[Dict[str, str]],
    x_col_candidates: List[str],
) -> List[RunData]:
    runs: List[RunData] = []
    for spec in specs:
        path = Path(spec["path"]).resolve()
        df = load_training_df(path)
        df = add_derived_metrics(df)

        x_col = find_col(df, x_col_candidates)
        if x_col is None:
            raise ValueError(
                f"Could not find x-axis column in {path}. "
                f"Tried: {x_col_candidates}"
            )

        runs.append(
            RunData(
                label=spec["label"],
                df=df,
                x_col=x_col,
                scenario=spec.get("scenario"),
                model=spec.get("model"),
            )
        )
    return runs


def build_style_map(runs: List[RunData]) -> Dict[str, Dict[str, Any]]:
    style: Dict[str, Dict[str, Any]] = {}

    scenarios = []
    for r in runs:
        if r.scenario and r.scenario not in scenarios:
            scenarios.append(r.scenario)

    if scenarios:
        cmap = cm.get_cmap("tab10")
        scenario_color = {s: cmap(i % 10) for i, s in enumerate(scenarios)}
    else:
        scenario_color = {}

    fallback_cmap = cm.get_cmap("tab10")
    fallback_idx = len(scenarios)

    def linestyle_for(model: Optional[str]) -> str:
        if model is None:
            return "-"
        m = model.lower()
        if "baseline" in m or m == "cnn":
            return "-"
        if "role" in m or "r3dm" in m:
            return "--"
        return ":"

    for r in runs:
        if r.scenario and r.scenario in scenario_color:
            color = scenario_color[r.scenario]
        else:
            color = fallback_cmap(fallback_idx % 10)
            fallback_idx += 1

        style[r.label] = {
            "color": color,
            "linestyle": linestyle_for(r.model),
        }

    return style

def plot_metric_overlay(
    runs: List[RunData],
    metric_name: str,
    candidates: List[str],
    out_path: Path,
    style_map: Dict[str, Dict[str, Any]],
    smooth: int = 5,
    title: Optional[str] = None,
    show_raw: bool = True,
) -> bool:
    plt.figure(figsize=(9, 5.5))
    any_plotted = False

    for r in runs:
        y_col = find_col(r.df, candidates)
        if y_col is None:
            continue

        x = numeric_series(r.df, r.x_col)
        y = numeric_series(r.df, y_col)
        valid = ~(x.isna() | y.isna())
        x = x[valid]
        y = y[valid]

        if len(x) == 0:
            continue

        style = style_map.get(r.label, {})
        color = style.get("color")
        linestyle = style.get("linestyle", "-")

        if show_raw and smooth > 1:
            plt.plot(x, y, color=color, linestyle=linestyle, alpha=0.18, linewidth=0.9)
            y_smooth = y.rolling(window=smooth, min_periods=1).mean()
            plt.plot(
                x, y_smooth,
                color=color, linestyle=linestyle,
                label=r.label, linewidth=1.8,
            )
        else:
            plt.plot(
                x, y,
                color=color, linestyle=linestyle,
                label=r.label, linewidth=1.6,
            )
        any_plotted = True

    if not any_plotted:
        plt.close()
        return False

    plt.xlabel("training timesteps")
    plt.ylabel(metric_name)
    plt.title(title or metric_name)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def plot_overview_grid(
    runs: List[RunData],
    metric_specs: List[Tuple[str, List[str]]],
    out_path: Path,
    style_map: Dict[str, Dict[str, Any]],
    smooth: int = 5,
    ncols: int = 3,
) -> bool:
    available: List[Tuple[str, List[str]]] = []
    for name, cands in metric_specs:
        for r in runs:
            if find_col(r.df, cands) is not None:
                available.append((name, cands))
                break

    if not available:
        return False

    n = len(available)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    legend_handles = {}

    for ax, (name, cands) in zip(axes, available):
        for r in runs:
            y_col = find_col(r.df, cands)
            if y_col is None:
                continue
            x = numeric_series(r.df, r.x_col)
            y = numeric_series(r.df, y_col)
            valid = ~(x.isna() | y.isna())
            x = x[valid]
            y = y[valid]
            if len(x) == 0:
                continue

            style = style_map.get(r.label, {})
            color = style.get("color")
            linestyle = style.get("linestyle", "-")

            if smooth > 1:
                y = y.rolling(window=smooth, min_periods=1).mean()
            line, = ax.plot(
                x, y,
                color=color, linestyle=linestyle, linewidth=1.4,
            )
            if r.label not in legend_handles:
                legend_handles[r.label] = line

        if name == "mean_max_role_prob":
            ax.axhline(0.5, linestyle="--", color="grey", alpha=0.5)
        if name in {"role_entropy", "marginal_role_entropy"}:
            ax.axhline(0.69314718056, linestyle="--", color="grey", alpha=0.5)

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("timesteps", fontsize=9)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(available):]:
        ax.axis("off")

    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="upper center",
            ncol=min(len(legend_handles), 4),
            bbox_to_anchor=(0.5, 1.0),
            fontsize=9,
        )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return True


def build_metric_specs() -> Dict[str, List[str]]:
    return {
        "episode_reward_mean": [
            "env_runners/episode_reward_mean",
            "episode_reward_mean",
            "sampler_results/episode_reward_mean",
            "env_runners/episode_return_mean",
            "sampler_results/episode_return_mean",
        ],
        "episode_len_mean": [
            "env_runners/episode_len_mean",
            "episode_len_mean",
            "sampler_results/episode_len_mean",
        ],
        "policy_loss": [
            "info/learner/shared_policy/learner_stats/policy_loss",
            "info/learner/default_policy/learner_stats/policy_loss",
        ],
        "vf_loss": [
            "info/learner/shared_policy/learner_stats/vf_loss",
            "info/learner/default_policy/learner_stats/vf_loss",
        ],
        "kl": [
            "info/learner/shared_policy/learner_stats/kl",
            "info/learner/default_policy/learner_stats/kl",
        ],
        "entropy": [
            "info/learner/shared_policy/learner_stats/entropy",
            "info/learner/default_policy/learner_stats/entropy",
        ],
        "role_entropy": [
            "info/learner/shared_policy/learner_stats/r3dm/role_entropy",
        ],
        "marginal_role_entropy": [
            "info/learner/shared_policy/learner_stats/r3dm/marginal_role_entropy",
        ],
        "mean_max_role_prob": [
            "info/learner/shared_policy/learner_stats/r3dm/mean_max_role_prob",
        ],
        "role_mi_proxy": [
            "info/learner/shared_policy/learner_stats/r3dm/role_mi_proxy",
        ],
        "ablation_action_change_fraction": [
            "derived/r3dm_ablation_action_change_fraction",
        ],
        "ablation_role_effect_strength": [
            "derived/r3dm_ablation_role_effect_strength",
        ],
        "role_collapse_gap": [
            "derived/r3dm_role_collapse_gap",
        ],
        "role_entropy_ratio": [
            "derived/r3dm_role_entropy_to_marginal_ratio",
        ],
        "r3dm_aux_nce": [
            "info/learner/shared_policy/learner_stats/r3dm/aux_nce",
        ],
        "gumbel_tau": [
            "info/learner/shared_policy/learner_stats/r3dm/gumbel_tau",
            "r3dm_gumbel_tau_callback",
            "custom_metrics/gumbel_tau",
        ],
    }

ROLE_DIAGNOSTIC_METRICS = [
    "role_entropy",
    "marginal_role_entropy",
    "mean_max_role_prob",
    "role_mi_proxy",
    "ablation_action_change_fraction",
    "ablation_role_effect_strength",
    "role_collapse_gap",
    "role_entropy_ratio",
]

TRAINING_OVERVIEW_METRICS = [
    "episode_reward_mean",
    "episode_len_mean",
    "policy_loss",
    "vf_loss",
    "kl",
    "entropy",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", default=[],
        help="A run as 'label=path'. Repeatable.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON file with a list of {label, path, scenario, model} entries.",
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--smooth", type=int, default=5,
        help="Rolling-mean window for smoothing.",
    )
    parser.add_argument(
        "--no-raw", action="store_true",
        help="In per-metric plots, hide the raw (unsmoothed) trace.",
    )
    parser.add_argument(
        "--x-col", type=str, default=None,
        help="Force a specific x-axis column. Default: auto-detect.",
    )
    args = parser.parse_args()

    specs: List[Dict[str, str]] = []
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            specs.extend(json.load(f))
    for run_arg in args.run:
        label, path = parse_run_arg(run_arg)
        specs.append({"label": label, "path": path})

    if not specs:
        parser.error("Provide at least one --run or --config")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    x_col_candidates = (
        [args.x_col] if args.x_col else [
            "timesteps_total",
            "num_env_steps_sampled_lifetime",
            "num_env_steps_sampled",
            "agent_timesteps_total",
            "training_iteration",
        ]
    )

    runs = load_runs(specs, x_col_candidates)
    style_map = build_style_map(runs)

    print(f"\nLoaded {len(runs)} runs:")
    for r in runs:
        print(f"  {r.label:40s}  x_col={r.x_col}  rows={len(r.df)}  "
              f"scenario={r.scenario}  model={r.model}")

    metric_specs = build_metric_specs()

    per_metric_dir = out_dir / "per_metric"
    per_metric_dir.mkdir(exist_ok=True)
    written: List[str] = []
    for name, cands in metric_specs.items():
        ok = plot_metric_overlay(
            runs=runs,
            metric_name=name,
            candidates=cands,
            out_path=per_metric_dir / f"{name}.png",
            style_map=style_map,
            smooth=args.smooth,
            show_raw=not args.no_raw,
        )
        if ok:
            written.append(name)

    plot_overview_grid(
        runs=runs,
        metric_specs=[(n, metric_specs[n]) for n in TRAINING_OVERVIEW_METRICS if n in metric_specs],
        out_path=out_dir / "training_overview.png",
        style_map=style_map,
        smooth=args.smooth,
        ncols=3,
    )

    plot_overview_grid(
        runs=runs,
        metric_specs=[(n, metric_specs[n]) for n in ROLE_DIAGNOSTIC_METRICS if n in metric_specs],
        out_path=out_dir / "role_diagnostics_overview.png",
        style_map=style_map,
        smooth=args.smooth,
        ncols=4,
    )

    print(f"\nWrote {len(written)} per-metric plots to {per_metric_dir}")
    print(f"Overview: {out_dir / 'training_overview.png'}")
    print(f"Role diagnostics: {out_dir / 'role_diagnostics_overview.png'}")


if __name__ == "__main__":
    main()
