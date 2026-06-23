

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
C_CLUST = "#c0392b"   # deep red  -> clusterer / searcher
C_EXPL  = "#2c5f7c"   # deep blue -> explorer
C_GREY  = "#8a8a8a"
C_LIGHT = "#e8e8e8"
C_DARK  = "#222222"


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "DejaVu Serif", "Times New Roman"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })


def _safe_float(x: Any) -> float:
    """Return a finite float or NaN."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(v) or math.isinf(v):
        return float("nan")
    return v


def _mean_finite(vals: List[float]) -> float:
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def _ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return float("nan")
    return float(num / den)


def _pair_key(a: int, b: int) -> str:
    a, b = sorted([int(a), int(b)])
    return f"{a}_{b}"


def _ceil_to(x: float, step: float) -> float:
    return step * math.ceil(max(x, step) / step)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return float("nan")
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])



def _role_props_all(summary: Dict[str, Any]) -> Dict[int, float]:
    raw = summary.get("role_props_all", {})
    out: Dict[int, float] = {}
    for k, v in raw.items():
        out[int(k)] = _safe_float(v)
    return out


def parse_summary(summary: Dict[str, Any]) -> Dict[str, Any]:

    role_ids = [0, 1]

    behaviour_keys = [
        ("search_rate",     "role_{r}_search_rate_mean"),
        ("visit_prob_mean", "role_{r}_visit_prob_mean_mean"),
        ("search_prob_mean","role_{r}_search_prob_mean_mean"),
    ]

    parsed: Dict[str, Any] = {
        "role_props":       _role_props_all(summary),
        "search_rate":      {},
        "visit_prob_mean":  {},
        "search_prob_mean": {},
        "pair_dist":        {},
        "pair_same":        {},
        "pair_close":       {},
        "switch_rate":      _safe_float(summary.get("role_switch_rate_episode_mean")),
        "switches_total":   _safe_float(summary.get("role_switches_total_mean")),
    }

    for r in role_ids:
        for out_field, key_tpl in behaviour_keys:
            parsed[out_field][r] = _safe_float(summary.get(key_tpl.format(r=r)))

    for a in role_ids:
        for b in role_ids:
            if a > b:
                continue
            k = _pair_key(a, b)
            parsed["pair_dist"][k] = _safe_float(
                summary.get(f"role_pair_dist_{a}_{b}_mean_mean")
            )
            parsed["pair_same"][k] = _safe_float(
                summary.get(f"role_pair_same_cell_rate_{a}_{b}_mean")
            )
            parsed["pair_close"][k] = _safe_float(
                summary.get(f"role_pair_close_rate_{a}_{b}_mean")
            )

    # Fill in missing role_props from role_*_fraction_mean if needed.
    if not parsed["role_props"]:
        for r in role_ids:
            parsed["role_props"][r] = _safe_float(
                summary.get(f"role_{r}_fraction_mean")
            )

    return parsed


def assign_clusterer(parsed: Dict[str, Any]) -> Tuple[int, int]:
    def score(r: int) -> float:
        same = parsed["pair_same"].get(_pair_key(r, r), float("nan"))
        parts = [
            parsed["search_rate"].get(r, float("nan")),
            same,
        ]
        return float(np.nansum(parts))

    score0, score1 = score(0), score(1)

    if abs(score0 - score1) < 1e-12:
        prop0 = parsed["role_props"].get(0, 0.0)
        prop1 = parsed["role_props"].get(1, 0.0)
        clusterer = 0 if prop0 >= prop1 else 1
    else:
        clusterer = 0 if score0 > score1 else 1

    explorer = 1 - clusterer
    return clusterer, explorer


def _detect_episode_file(summary_path: Path, summary_data: Dict[str, Any]) -> Optional[Path]:
    ep = summary_data.get("episodes_jsonl")
    if ep:
        p = Path(ep)
        if p.exists():
            return p

        q = summary_path.parent / p.name
        if q.exists():
            return q

    name = summary_path.name.replace("summary_", "")
    candidates = [summary_path.with_name(Path(name).with_suffix(".jsonl").name)]

    m = re.search(r"checkpoint_\d+", summary_path.name)
    if m:
        candidates.extend(summary_path.parent.glob(f"*{m.group(0)}*.jsonl"))
        candidates.extend(Path.cwd().glob(f"*{m.group(0)}*.jsonl"))

    for c in candidates:
        if c.exists():
            return c

    return None



def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get_first_finite(row: Dict[str, Any], keys: List[str]) -> float:
    for k in keys:
        if k in row:
            v = _safe_float(row.get(k))
            if not math.isnan(v):
                return v
    return float("nan")


def _get_success(row: Dict[str, Any]) -> float:
    for k in [
        "success",
        "successful",
        "found",
        "target_found",
        "person_found",
        "is_success",
        "done_success",
    ]:
        if k in row:
            v = row[k]
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            vf = _safe_float(v)
            if not math.isnan(vf):
                return 1.0 if vf > 0 else 0.0

    ttf = _get_first_finite(row, ["time_to_find", "ttf", "steps_to_find"])
    if not math.isnan(ttf):
        return 1.0

    return float("nan")


def _get_reward(row: Dict[str, Any]) -> float:
    return _get_first_finite(
        row,
        [
            "total_reward",
            "episode_reward",
            "reward",
            "return",
            "episode_return",
            "mean_total_reward",
        ],
    )


def _get_steps(row: Dict[str, Any]) -> float:
    return _get_first_finite(
        row,
        [
            "steps",
            "episode_steps",
            "episode_len",
            "episode_length",
            "length",
            "timestep",
            "timesteps",
        ],
    )


def _get_unique_cells(row: Dict[str, Any]) -> float:
    return _get_first_finite(
        row,
        [
            "unique_cells_total",
            "unique_cells",
            "n_unique_cells",
            "visited_unique_cells",
        ],
    )


def _get_explorer_fraction(row: Dict[str, Any], explorer_role: int) -> float:
    direct_key = f"role_{explorer_role}_fraction"
    if direct_key in row:
        return _safe_float(row.get(direct_key))

    props = row.get("role_props") or row.get("role_props_all")
    if isinstance(props, dict):
        return _safe_float(props.get(str(explorer_role), props.get(explorer_role)))

    return float("nan")


def collect_episode_rows(
    episodes: List[Dict[str, Any]],
    explorer_role: int,
) -> List[Dict[str, float]]:

    rows: List[Dict[str, float]] = []

    for ep in episodes:
        explorer_frac = _get_explorer_fraction(ep, explorer_role)
        if math.isnan(explorer_frac):
            continue

        success = _get_success(ep)
        reward = _get_reward(ep)

        steps = _get_steps(ep)
        unique_cells = _get_unique_cells(ep)
        unique_per_step = (
            unique_cells / steps
            if np.isfinite(unique_cells) and np.isfinite(steps) and steps > 0
            else float("nan")
        )

        revisit_fraction = _get_first_finite(
            ep,
            ["revisit_fraction", "mean_revisit_fraction", "revisit_fraction_mean"],
        )
        repeated_search = _get_first_finite(
            ep,
            [
                "repeated_search_fraction",
                "mean_repeated_search_fraction",
                "repeated_search_fraction_mean",
            ],
        )
        co_occupancy = _get_first_finite(
            ep,
            [
                "co_occupancy_fraction",
                "mean_co_occupancy_fraction",
                "co_occupancy_fraction_mean",
            ],
        )

        rows.append({
            "explorer_frac": explorer_frac,
            "success": success,
            "reward": reward,
            "unique_per_step": unique_per_step,
            "revisit_fraction": revisit_fraction,
            "repeated_search": repeated_search,
            "co_occupancy": co_occupancy,
        })

    return rows


def split_low_high_explorer(
    rows: List[Dict[str, float]],
    q: float = 0.25,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    if not rows:
        return [], []

    explorer_vals = np.array([r["explorer_frac"] for r in rows], dtype=float)
    explorer_vals = explorer_vals[np.isfinite(explorer_vals)]

    if len(explorer_vals) == 0:
        return [], []

    lo_thr = np.nanquantile(explorer_vals, q)
    hi_thr = np.nanquantile(explorer_vals, 1.0 - q)

    low = [r for r in rows if np.isfinite(r["explorer_frac"]) and r["explorer_frac"] <= lo_thr]
    high = [r for r in rows if np.isfinite(r["explorer_frac"]) and r["explorer_frac"] >= hi_thr]

    return low, high


def summarize_low_high_explorer(
    rows: List[Dict[str, float]],
    q: float = 0.25,
) -> Dict[str, float]:
    low, high = split_low_high_explorer(rows, q=q)

    def mean_valid(group: List[Dict[str, float]], key: str) -> float:
        vals = np.array([r[key] for r in group], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if len(vals) else float("nan")

    def mean_reward_success_only(group: List[Dict[str, float]]) -> float:
        vals = np.array(
            [
                r["reward"]
                for r in group
                if np.isfinite(r["reward"])
                and np.isfinite(r["success"])
                and r["success"] > 0
            ],
            dtype=float,
        )
        return float(np.mean(vals)) if len(vals) else float("nan")

    x = np.array([r["explorer_frac"] for r in rows], dtype=float)
    s = np.array([r["success"] for r in rows], dtype=float)
    rew = np.array([r["reward"] for r in rows], dtype=float)

    return {
        "low_success": mean_valid(low, "success"),
        "high_success": mean_valid(high, "success"),
        "low_reward": mean_valid(low, "reward"),
        "high_reward": mean_valid(high, "reward"),
        "low_reward_success": mean_reward_success_only(low),
        "high_reward_success": mean_reward_success_only(high),

        "low_unique_per_step": mean_valid(low, "unique_per_step"),
        "high_unique_per_step": mean_valid(high, "unique_per_step"),
        "low_revisit_fraction": mean_valid(low, "revisit_fraction"),
        "high_revisit_fraction": mean_valid(high, "revisit_fraction"),
        "low_repeated_search": mean_valid(low, "repeated_search"),
        "high_repeated_search": mean_valid(high, "repeated_search"),
        "low_co_occupancy": mean_valid(low, "co_occupancy"),
        "high_co_occupancy": mean_valid(high, "co_occupancy"),

        "corr_success": pearson_corr(x, s),
        "corr_reward": pearson_corr(x, rew),

        "low_n": len(low),
        "high_n": len(high),
    }


def binned_success_by_explorer(
    rows: List[Dict[str, float]],
    bins: int = 5,
) -> List[Dict[str, float]]:

    clean = [
        r for r in rows
        if np.isfinite(r["explorer_frac"]) and np.isfinite(r["success"])
    ]
    if not clean:
        return []

    x = np.array([r["explorer_frac"] for r in clean], dtype=float)
    edges = np.nanquantile(x, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)

    if len(edges) < 2:
        return []

    out: List[Dict[str, float]] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]

        if i == len(edges) - 2:
            group = [r for r in clean if lo <= r["explorer_frac"] <= hi]
        else:
            group = [r for r in clean if lo <= r["explorer_frac"] < hi]

        if not group:
            continue

        success_vals = np.array([r["success"] for r in group], dtype=float)
        exp_vals = np.array([r["explorer_frac"] for r in group], dtype=float)

        out.append({
            "x_mean": float(np.mean(exp_vals)),
            "x_lo": float(lo),
            "x_hi": float(hi),
            "success": float(np.mean(success_vals)),
            "n": float(len(group)),
        })

    return out



BEHAVIOUR_METRICS: List[Tuple[str, str]] = [
    ("Search rate",          "search_rate"),
    ("Mean belief at visit", "visit_prob_mean"),
]


def compute_global_scales(parsed_list: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    xmax_b = 0.0
    dmax = 0.0
    smax = 0.0

    for p in parsed_list:
        for _lbl, key in BEHAVIOUR_METRICS:
            for r in (0, 1):
                v = p[key].get(r, float("nan"))
                if not math.isnan(v):
                    xmax_b = max(xmax_b, v)

        for d in p["pair_dist"].values():
            if not math.isnan(d):
                dmax = max(dmax, d)

        for s in p["pair_same"].values():
            if not math.isnan(s):
                smax = max(smax, s)

    xmax_b = _ceil_to(xmax_b * 1.15, 0.1)
    dmax = _ceil_to(dmax * 1.10, 5.0)
    smax = _ceil_to(smax * 1.15, 0.1)

    return xmax_b, dmax, smax


def _draw_behaviour(
    ax: plt.Axes,
    vals: List[float],
    labels: List[str],
    color: str,
    role_name: str,
    role_id: int,
    share: float,
    show_xlabel: bool,
    show_ylabels: bool,
    xmax: float,
) -> None:
    y = np.arange(len(labels))[::-1]
    safe = [0.0 if math.isnan(v) else v for v in vals]

    ax.barh(y, safe, color=color, height=0.62, edgecolor="none")

    for yi, v_raw, v in zip(y, vals, safe):
        txt = "n/a" if math.isnan(v_raw) else f"{v_raw:.3f}"
        ax.text(
            v + 0.008,
            yi,
            txt,
            va="center",
            ha="left",
            fontsize=7,
            color="#333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels if show_ylabels else [])
    ax.set_xlim(0, xmax)

    ticks = np.linspace(0, xmax, 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.1f}" for t in ticks])

    if show_xlabel:
        ax.set_xlabel("rate / probability", fontsize=8)

    ax.set_title(
        role_name,
        color=color,
        fontweight="bold",
        loc="left",
        pad=14,
        fontsize=9.5,
    )

    share_txt = "n/a" if math.isnan(share) else f"{share * 100:.0f}% of steps"
    ax.text(
        0.0,
        1.02,
        f"role {role_id}  ·  {share_txt}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=color,
        alpha=0.85,
    )


def _draw_pair_geometry(
    ax: plt.Axes,
    parsed: Dict[str, Any],
    clust: int,
    expl: int,
    show_xlabel: bool,
    show_title: bool,
    dist_max: float,
    same_max: float,
) -> None:
    cells = [
        ("C–C", parsed["pair_dist"][_pair_key(clust, clust)],
                parsed["pair_same"][_pair_key(clust, clust)], C_CLUST),
        ("C–E", parsed["pair_dist"][_pair_key(clust, expl)],
                parsed["pair_same"][_pair_key(clust, expl)], C_GREY),
        ("E–E", parsed["pair_dist"][_pair_key(expl, expl)],
                parsed["pair_same"][_pair_key(expl, expl)], C_EXPL),
    ]

    x = np.arange(3)
    dists_raw = [c[1] for c in cells]
    sames_raw = [c[2] for c in cells]
    cols = [c[3] for c in cells]
    labs = [c[0] for c in cells]

    dists = [0.0 if math.isnan(d) else d for d in dists_raw]
    sames = [float("nan") if math.isnan(s) else s for s in sames_raw]

    ax.bar(x, dists, color=cols, width=0.62, edgecolor="none", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylim(0, dist_max)
    ax.set_ylabel("mean dist.", fontsize=8)

    if show_xlabel:
        ax.set_xlabel("pair type", fontsize=8)

    if show_title:
        ax.set_title("Pair geometry", loc="left", pad=4, fontsize=9, fontweight="bold")

    ax2 = ax.twinx()
    finite = [(xi, s) for xi, s in zip(x, sames) if not math.isnan(s)]

    if finite:
        fx, fs = zip(*finite)
        ax2.plot(
            fx,
            fs,
            "o-",
            color=C_DARK,
            markersize=4.5,
            linewidth=1.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )

    ax2.set_ylim(0, same_max)
    ax2.set_ylabel("same-cell rate", fontsize=8, color=C_DARK)
    ax2.tick_params(axis="y", labelsize=7, colors=C_DARK)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.6)

    for xi, d, raw in zip(x, dists, dists_raw):
        txt = "n/a" if math.isnan(raw) else f"{raw:.1f}"
        ax.text(
            xi,
            d + 0.02 * dist_max,
            txt,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333",
        )


def build_main_profile_figure(
    parsed_list: List[Dict[str, Any]],
    labels: List[str],
    title: str,
    subtitle: str,
) -> plt.Figure:
    n = len(parsed_list)

    title_in = 1.05
    legend_in = 0.85
    row_in = 2.4
    interrow_in = 0.75

    fig_w = 7.6
    fig_h = title_in + legend_in + n * row_in + max(0, n - 1) * interrow_in

    fig = plt.figure(figsize=(fig_w, fig_h))

    top = 1.0 - title_in / fig_h
    bottom = legend_in / fig_h
    hspace = interrow_in / row_in

    gs = fig.add_gridspec(
        nrows=n,
        ncols=4,
        width_ratios=[1.0, 0.7, 0.10, 0.78],
        hspace=hspace,
        wspace=0.15,
        left=0.22,
        right=0.96,
        top=top,
        bottom=bottom,
    )

    xmax_b, dmax, smax = compute_global_scales(parsed_list)

    behaviour_labels = [m[0] for m in BEHAVIOUR_METRICS]
    behaviour_keys = [m[1] for m in BEHAVIOUR_METRICS]

    for row, (parsed, label) in enumerate(zip(parsed_list, labels)):
        is_first = row == 0
        is_last = row == n - 1

        clust, expl = assign_clusterer(parsed)

        ax_c = fig.add_subplot(gs[row, 0])
        ax_e = fig.add_subplot(gs[row, 1])
        ax_g = fig.add_subplot(gs[row, 3])

        vals_c = [parsed[k].get(clust, float("nan")) for k in behaviour_keys]
        vals_e = [parsed[k].get(expl, float("nan")) for k in behaviour_keys]

        _draw_behaviour(
            ax_c,
            vals_c,
            behaviour_labels,
            C_CLUST,
            "Clusterer / searcher",
            clust,
            parsed["role_props"].get(clust, float("nan")),
            show_xlabel=is_last,
            show_ylabels=True,
            xmax=xmax_b,
        )

        _draw_behaviour(
            ax_e,
            vals_e,
            behaviour_labels,
            C_EXPL,
            "Explorer",
            expl,
            parsed["role_props"].get(expl, float("nan")),
            show_xlabel=is_last,
            show_ylabels=False,
            xmax=xmax_b,
        )

        _draw_pair_geometry(
            ax_g,
            parsed,
            clust,
            expl,
            show_xlabel=is_last,
            show_title=is_first,
            dist_max=dmax,
            same_max=smax,
        )

        ax_pos = ax_c.get_position()
        y_mid = (ax_pos.y0 + ax_pos.y1) / 2
        fig.text(
            0.025,
            y_mid,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#222",
        )

    fig.text(
        0.02,
        1.0 - 0.30 / fig_h,
        title,
        fontsize=12,
        fontweight="bold",
        color="#111",
        va="top",
    )

    fig.text(
        0.02,
        1.0 - 0.55 / fig_h,
        subtitle,
        fontsize=8.5,
        color="#555",
        va="top",
    )

    leg_y = 0.25 / fig_h

    fig.add_artist(plt.Line2D(
        [0.22, 0.25],
        [leg_y, leg_y],
        color=C_CLUST,
        lw=4,
        solid_capstyle="butt",
    ))
    fig.text(
        0.26,
        leg_y,
        "clusterer / searcher",
        va="center",
        fontsize=8,
        color=C_CLUST,
    )

    fig.add_artist(plt.Line2D(
        [0.47, 0.50],
        [leg_y, leg_y],
        color=C_EXPL,
        lw=4,
        solid_capstyle="butt",
    ))
    fig.text(
        0.51,
        leg_y,
        "explorer",
        va="center",
        fontsize=8,
        color=C_EXPL,
    )

    fig.text(
        0.72,
        leg_y,
        "C–C, C–E, E–E = pair types",
        va="center",
        fontsize=7.5,
        color="#555",
        style="italic",
    )

    return fig


def build_explorer_fingerprint_figure(
    parsed_list: List[Dict[str, Any]],
    labels: List[str],
    title: str = "Explorer behavioural fingerprint",
    subtitle: str = (
        "Values show explorer/searcher ratios. Values below 1 indicate that the "
        "explorer performs that behaviour less often than the searcher."
    ),
) -> plt.Figure:
    metrics = [
        ("Search", "search_rate"),
        ("Belief at visit", "visit_prob_mean"),
        ("Belief at search", "search_prob_mean"),
    ]

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(7.6, 3.35), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, parsed, label in zip(axes, parsed_list, labels):
        clust, expl = assign_clusterer(parsed)

        vals = []
        for _name, key in metrics:
            vals.append(_ratio(parsed[key].get(expl, float("nan")),
                               parsed[key].get(clust, float("nan"))))

        x = np.arange(len(metrics))
        plot_vals = [0.0 if math.isnan(v) else v for v in vals]

        ax.bar(x, plot_vals, color=C_EXPL, edgecolor="none", width=0.62)
        ax.axhline(1.0, color=C_DARK, linewidth=0.8, linestyle="--", alpha=0.65)

        for xi, raw, val in zip(x, vals, plot_vals):
            txt = "n/a" if math.isnan(raw) else f"{raw:.2f}×"
            ax.text(
                xi,
                val + 0.05,
                txt,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333",
            )

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in metrics], rotation=35, ha="right")
        ax.grid(axis="y", color=C_LIGHT, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("explorer / searcher ratio")

    fig.text(0.02, 0.98, title, fontsize=12, fontweight="bold", va="top")
    fig.text(0.02, 0.90, subtitle, fontsize=8.5, color="#555", va="top")
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.78])

    return fig


def build_explorer_spatial_detail_figure(
    parsed_list: List[Dict[str, Any]],
    labels: List[str],
    title: str = "Explorer spatial role geometry",
    subtitle: str = (
        "Distances and overlap rates by pair type. C = clusterer/searcher, "
        "E = explorer."
    ),
) -> plt.Figure:
    n = len(labels)
    fig, axes = plt.subplots(2, n, figsize=(7.6, 4.5), sharex=False)

    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    pair_labels = ["C–C", "C–E", "E–E"]
    pair_colors = [C_CLUST, C_GREY, C_EXPL]

    all_d = []
    all_same = []

    for parsed in parsed_list:
        clust, expl = assign_clusterer(parsed)
        keys = [_pair_key(clust, clust), _pair_key(clust, expl), _pair_key(expl, expl)]
        all_d.extend([parsed["pair_dist"].get(k, float("nan")) for k in keys])
        all_same.extend([parsed["pair_same"].get(k, float("nan")) for k in keys])

    dmax = _ceil_to(np.nanmax(all_d) * 1.15 if np.any(np.isfinite(all_d)) else 1.0, 5.0)
    smax = _ceil_to(np.nanmax(all_same) * 1.20 if np.any(np.isfinite(all_same)) else 1.0, 0.1)

    for col, (parsed, label) in enumerate(zip(parsed_list, labels)):
        clust, expl = assign_clusterer(parsed)
        keys = [_pair_key(clust, clust), _pair_key(clust, expl), _pair_key(expl, expl)]

        d_vals = [parsed["pair_dist"].get(k, float("nan")) for k in keys]
        s_vals = [parsed["pair_same"].get(k, float("nan")) for k in keys]

        x = np.arange(3)

        ax_d = axes[0, col]
        d_plot = [0.0 if math.isnan(v) else v for v in d_vals]
        ax_d.bar(x, d_plot, color=pair_colors, edgecolor="none", width=0.62)
        ax_d.set_ylim(0, dmax)
        ax_d.set_title(label, fontsize=10, fontweight="bold")
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(pair_labels)
        ax_d.grid(axis="y", color=C_LIGHT, linewidth=0.6)
        ax_d.set_axisbelow(True)

        for xi, raw, val in zip(x, d_vals, d_plot):
            txt = "n/a" if math.isnan(raw) else f"{raw:.1f}"
            ax_d.text(xi, val + 0.02 * dmax, txt, ha="center", fontsize=7)

        ax_s = axes[1, col]
        s_plot = [0.0 if math.isnan(v) else v for v in s_vals]
        ax_s.bar(x, s_plot, color=pair_colors, edgecolor="none", width=0.62)
        ax_s.set_ylim(0, smax)
        ax_s.set_xticks(x)
        ax_s.set_xticklabels(pair_labels)
        ax_s.grid(axis="y", color=C_LIGHT, linewidth=0.6)
        ax_s.set_axisbelow(True)

        for xi, raw, val in zip(x, s_vals, s_plot):
            txt = "n/a" if math.isnan(raw) else f"{raw:.3f}"
            ax_s.text(xi, val + 0.02 * smax, txt, ha="center", fontsize=7)

    axes[0, 0].set_ylabel("mean pairwise distance")
    axes[1, 0].set_ylabel("same-cell rate")

    fig.text(0.02, 0.98, title, fontsize=12, fontweight="bold", va="top")
    fig.text(0.02, 0.91, subtitle, fontsize=8.5, color="#555", va="top")
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.82])

    return fig


def build_explorer_episode_coverage_figure(
    parsed_list: List[Dict[str, Any]],
    episode_paths: List[Path],
    labels: List[str],
    title: str = "Episode-level coverage under low vs high explorer usage",
    subtitle: str = (
        "Episodes are split into bottom and top quartiles by explorer usage. "
        "Coverage is measured at team level."
    ),
) -> plt.Figure:
    stats = []

    for parsed, path in zip(parsed_list, episode_paths):
        _clust, explorer = assign_clusterer(parsed)
        episodes = load_jsonl(path)
        rows = collect_episode_rows(episodes, explorer_role=explorer)
        stats.append(summarize_low_high_explorer(rows))

    metrics = [
        ("Unique cells / step", "low_unique_per_step", "high_unique_per_step"),
        ("Revisit fraction", "low_revisit_fraction", "high_revisit_fraction"),
        ("Repeated search", "low_repeated_search", "high_repeated_search"),
        ("Co-occupancy", "low_co_occupancy", "high_co_occupancy"),
    ]

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(7.6, 3.5), sharey=False)
    if n == 1:
        axes = [axes]

    x = np.arange(len(metrics))
    width = 0.36

    for ax, label, st in zip(axes, labels, stats):
        low_vals = [st.get(lo, float("nan")) for _name, lo, _hi in metrics]
        high_vals = [st.get(hi, float("nan")) for _name, _lo, hi in metrics]

        low_plot = [0.0 if math.isnan(v) else v for v in low_vals]
        high_plot = [0.0 if math.isnan(v) else v for v in high_vals]

        ax.bar(x - width / 2, low_plot, width, color=C_GREY, edgecolor="none", label="low explorer")
        ax.bar(x + width / 2, high_plot, width, color=C_EXPL, edgecolor="none", label="high explorer")

        for xi, lv, hv in zip(x, low_vals, high_vals):
            ymax = np.nanmax([lv, hv])
            if math.isnan(ymax):
                continue
            ratio = _ratio(hv, lv)
            txt = "n/a" if math.isnan(ratio) else f"{ratio:.2f}×"
            ax.text(xi, ymax + 0.04 * max(1.0, ymax), txt, ha="center", fontsize=7, color="#555")

        success_txt = ""
        if np.isfinite(st.get("low_success", float("nan"))) and np.isfinite(st.get("high_success", float("nan"))):
            success_txt = f"success {st['low_success']:.2f} → {st['high_success']:.2f}"

        ax.set_title(f"{label}\n{success_txt}", fontsize=9.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in metrics], rotation=35, ha="right")
        ax.grid(axis="y", color=C_LIGHT, linewidth=0.6)
        ax.set_axisbelow(True)

        local_max = np.nanmax(low_vals + high_vals)
        if math.isnan(local_max):
            local_max = 1.0
        ax.set_ylim(0, local_max * 1.25)

    axes[0].set_ylabel("episode-level mean")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(0.16, 0.82), ncol=2, frameon=False)

    fig.text(0.02, 0.98, title, fontsize=12, fontweight="bold", va="top")
    fig.text(0.02, 0.90, subtitle, fontsize=8.5, color="#555", va="top")
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.76])

    return fig


def build_explorer_success_reward_figure(
    parsed_list: List[Dict[str, Any]],
    episode_paths: List[Path],
    labels: List[str],
    title: str = "Outcome effect of explorer usage",
    subtitle: str = (
        "Episodes are split into low- and high-explorer groups using the "
        "bottom and top quartiles of explorer-role usage."
    ),
) -> plt.Figure:
    stats = []

    for parsed, path in zip(parsed_list, episode_paths):
        _clust, explorer = assign_clusterer(parsed)
        episodes = load_jsonl(path)
        rows = collect_episode_rows(episodes, explorer_role=explorer)
        stats.append(summarize_low_high_explorer(rows))

    metrics = [
        ("Success rate", "low_success", "high_success"),
        ("Mean reward", "low_reward", "high_reward"),
        ("Reward if found", "low_reward_success", "high_reward_success"),
    ]

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(7.6, 3.35), sharey=False)
    if n == 1:
        axes = [axes]

    x = np.arange(len(metrics))
    width = 0.36

    for ax, label, st in zip(axes, labels, stats):
        low_vals = [st.get(lo_key, float("nan")) for _name, lo_key, _hi_key in metrics]
        high_vals = [st.get(hi_key, float("nan")) for _name, _lo_key, hi_key in metrics]

        low_plot = [0.0 if math.isnan(v) else v for v in low_vals]
        high_plot = [0.0 if math.isnan(v) else v for v in high_vals]

        ax.bar(x - width / 2, low_plot, width, color=C_GREY, edgecolor="none", label="low explorer")
        ax.bar(x + width / 2, high_plot, width, color=C_EXPL, edgecolor="none", label="high explorer")

        for xi, lv, hv in zip(x, low_vals, high_vals):
            ymax = np.nanmax([lv, hv])
            if math.isnan(ymax):
                continue
            ax.text(
                xi,
                ymax + 0.04,
                f"{lv:.2f} → {hv:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="#555",
            )

        ax.set_title(label, fontweight="bold", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in metrics], rotation=32, ha="right")
        ax.grid(axis="y", color=C_LIGHT, linewidth=0.6)
        ax.set_axisbelow(True)

        local_max = np.nanmax(low_vals + high_vals)
        if math.isnan(local_max):
            local_max = 1.0
        ax.set_ylim(0, local_max * 1.24)

    axes[0].set_ylabel("episode-level mean")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(0.16, 0.82), ncol=2, frameon=False)

    fig.text(0.02, 0.98, title, fontsize=12, fontweight="bold", va="top")
    fig.text(0.02, 0.90, subtitle, fontsize=8.5, color="#555", va="top")
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.76])

    return fig


def build_explorer_success_correlation_figure(
    parsed_list: List[Dict[str, Any]],
    episode_paths: List[Path],
    labels: List[str],
    title: str = "Correlation between explorer usage and success",
    subtitle: str = (
        "Episodes are grouped into quantile bins by explorer-role usage; "
        "points show mean success rate per bin."
    ),
) -> plt.Figure:
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(7.6, 3.35), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, parsed, path, label in zip(axes, parsed_list, episode_paths, labels):
        _clust, explorer = assign_clusterer(parsed)
        episodes = load_jsonl(path)
        rows = collect_episode_rows(episodes, explorer_role=explorer)

        bins = binned_success_by_explorer(rows, bins=5)

        x_raw = np.array([r["explorer_frac"] for r in rows], dtype=float)
        y_raw = np.array([r["success"] for r in rows], dtype=float)
        r_success = pearson_corr(x_raw, y_raw)

        # Raw jittered points in the background.
        mask = np.isfinite(x_raw) & np.isfinite(y_raw)
        if np.any(mask):
            rng = np.random.default_rng(7)
            y_jitter = y_raw[mask] + rng.normal(0.0, 0.018, size=np.sum(mask))
            ax.scatter(
                x_raw[mask],
                y_jitter,
                s=7,
                color=C_GREY,
                alpha=0.18,
                edgecolors="none",
            )

        # Quantile-bin means.
        if bins:
            x = np.array([b["x_mean"] for b in bins], dtype=float)
            y = np.array([b["success"] for b in bins], dtype=float)
            n_bin = np.array([b["n"] for b in bins], dtype=float)

            ax.plot(
                x,
                y,
                "o-",
                color=C_EXPL,
                linewidth=1.4,
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.8,
            )

            for xi, yi, ni in zip(x, y, n_bin):
                ax.text(
                    xi,
                    yi + 0.035,
                    f"n={int(ni)}",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    color="#555",
                )

        ax.set_title(label, fontweight="bold", fontsize=10)
        ax.set_xlabel("explorer fraction per episode")
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(axis="y", color=C_LIGHT, linewidth=0.6)
        ax.set_axisbelow(True)

        r_txt = "n/a" if math.isnan(r_success) else f"{r_success:+.2f}"
        ax.text(
            0.03,
            0.06,
            f"corr. with success: {r_txt}",
            transform=ax.transAxes,
            fontsize=7.5,
            color="#555",
            ha="left",
            va="bottom",
        )

    axes[0].set_ylabel("success rate")

    fig.text(0.02, 0.98, title, fontsize=12, fontweight="bold", va="top")
    fig.text(0.02, 0.90, subtitle, fontsize=8.5, color="#555", va="top")
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.78])

    return fig


def default_label_for(path: Path) -> str:
    """Derive a scenario label from a filename."""
    stem = path.stem

    for suffix in ("_summary", ".summary"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    m = re.search(r"\bS\d+\b", stem.upper())
    if m:
        return m.group(0)

    head = stem.split("_")[0]
    if head and head[0].upper() == "S" and head[1:].isdigit():
        return head.upper()

    return stem


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    basename: str,
    formats: List[str],
) -> None:
    for fmt in formats:
        out_path = out_dir / f"{basename}.{fmt}"
        kwargs = {"bbox_inches": "tight"}
        if fmt.lower() == "png":
            kwargs["dpi"] = 300
        fig.savefig(out_path, **kwargs)
        print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "summaries",
        nargs="+",
        help="One or more paths to evaluation summary JSON files.",
    )

    ap.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Row labels, one per summary. Default: derived from filename.",
    )

    ap.add_argument(
        "--out-dir",
        default=".",
        help="Directory to write figures into. Default: current directory.",
    )

    ap.add_argument(
        "--basename",
        default="role_profile",
        help="Basename for output files without extension.",
    )

    ap.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "svg", "png"],
        help="Output formats to write. Default: pdf svg png.",
    )

    ap.add_argument(
        "--title",
        default="Emergent role specialization across scenarios",
        help="Figure super-title.",
    )

    ap.add_argument(
        "--subtitle",
        default=(
            "Role IDs are not aligned across runs, but the same functional "
            "split—clusterer/searcher vs explorer—reappears."
        ),
        help="Figure subtitle.",
    )

    args = ap.parse_args()

    summary_paths = [Path(p) for p in args.summaries]

    if args.labels is not None and len(args.labels) != len(summary_paths):
        ap.error(
            f"--labels expects exactly {len(summary_paths)} entries "
            f"(one per summary), got {len(args.labels)}."
        )

    labels: List[str] = (
        list(args.labels)
        if args.labels is not None
        else [default_label_for(p) for p in summary_paths]
    )

    parsed_list: List[Dict[str, Any]] = []
    episode_paths: List[Optional[Path]] = []

    for path in summary_paths:
        with path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        parsed_list.append(parse_summary(summary))
        episode_paths.append(_detect_episode_file(path, summary))

    configure_style()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = build_main_profile_figure(
        parsed_list=parsed_list,
        labels=labels,
        title=args.title,
        subtitle=args.subtitle,
    )
    save_figure(fig, out_dir, args.basename, args.formats)
    plt.close(fig)

    fig_fp = build_explorer_fingerprint_figure(
        parsed_list=parsed_list,
        labels=labels,
    )
    save_figure(fig_fp, out_dir, f"{args.basename}_explorer_fingerprint", args.formats)
    plt.close(fig_fp)

    fig_spatial = build_explorer_spatial_detail_figure(
        parsed_list=parsed_list,
        labels=labels,
    )
    save_figure(fig_spatial, out_dir, f"{args.basename}_explorer_spatial_detail", args.formats)
    plt.close(fig_spatial)

    if all(p is not None for p in episode_paths):
        ep_paths = [p for p in episode_paths if p is not None]

        fig_cov = build_explorer_episode_coverage_figure(
            parsed_list=parsed_list,
            episode_paths=ep_paths,
            labels=labels,
        )
        save_figure(fig_cov, out_dir, f"{args.basename}_explorer_episode_coverage", args.formats)
        plt.close(fig_cov)

        fig_outcome = build_explorer_success_reward_figure(
            parsed_list=parsed_list,
            episode_paths=ep_paths,
            labels=labels,
        )
        save_figure(fig_outcome, out_dir, f"{args.basename}_explorer_success_reward", args.formats)
        plt.close(fig_outcome)

        fig_corr = build_explorer_success_correlation_figure(
            parsed_list=parsed_list,
            episode_paths=ep_paths,
            labels=labels,
        )
        save_figure(fig_corr, out_dir, f"{args.basename}_explorer_success_correlation", args.formats)
        plt.close(fig_corr)

    else:
        missing = [
            str(summary_paths[i])
            for i, p in enumerate(episode_paths)
            if p is None
        ]
        print(
            "[warning] Could not find episode-level JSONL files for:\n  "
            + "\n  ".join(missing)
        )
        print(
            "[warning] Skipping episode-level explorer coverage, "
            "success/reward, and success-correlation figures."
        )


if __name__ == "__main__":
    main()
