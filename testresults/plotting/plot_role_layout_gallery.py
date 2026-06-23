#!/usr/bin/env python3


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


C_CLUST = "#c0392b"   # clusterer / searcher
C_EXPL = "#2c5f7c"    # explorer
C_GREY = "#8a8a8a"
C_LIGHT = "#e8e8e8"
C_DARK = "#222222"
C_TEXT = "#333333"


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "TeX Gyre Termes", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "grid.color": C_LIGHT,
        "grid.linewidth": 0.55,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(v) or math.isinf(v):
        return float("nan")
    return v


def _mean_finite(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def _ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return float("nan")
    return float(num / den)


def _ceil_to(x: float, step: float) -> float:
    return step * math.ceil(max(float(x), step) / step)


def _pair_key(a: int, b: int) -> str:
    a, b = sorted([int(a), int(b)])
    return f"{a}_{b}"


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _format_float(v: float, digits: int = 2, suffix: str = "") -> str:
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{digits}f}{suffix}"


def _add_figure_title(fig: plt.Figure, title: str, subtitle: str, top: float = 0.98) -> None:
    fig.text(0.02, top, title, fontsize=12, fontweight="bold", va="top", color="#111")
    if subtitle:
        fig.text(0.02, top - 0.075, subtitle, fontsize=8.5, va="top", color="#555")


def save_figure(fig: plt.Figure, out_dir: Path, basename: str, formats: List[str]) -> None:
    for fmt in formats:
        out_path = out_dir / f"{basename}.{fmt}"
        kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
        if fmt.lower() == "png":
            kwargs["dpi"] = 300
        fig.savefig(out_path, **kwargs)
        print(f"wrote {out_path}")


def _role_props_all(summary: Dict[str, Any]) -> Dict[int, float]:
    raw = summary.get("role_props_all", {})
    out: Dict[int, float] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[int(k)] = _safe_float(v)
    return out


def parse_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    role_ids = [0, 1]
    behaviour_keys = [
        ("search_rate", "role_{r}_search_rate_mean"),
        ("visit_prob_mean", "role_{r}_visit_prob_mean_mean"),
        ("search_prob_mean", "role_{r}_search_prob_mean_mean"),
    ]
    parsed: Dict[str, Any] = {
        "role_props": _role_props_all(summary),
        "search_rate": {},
        "visit_prob_mean": {},
        "search_prob_mean": {},
        "pair_dist": {},
        "pair_same": {},
        "pair_close": {},
        "switch_rate": _safe_float(summary.get("role_switch_rate_episode_mean")),
        "switches_total": _safe_float(summary.get("role_switches_total_mean")),
        "summary": summary,
    }
    for r in role_ids:
        for out_field, key_tpl in behaviour_keys:
            parsed[out_field][r] = _safe_float(summary.get(key_tpl.format(r=r)))
    for a in role_ids:
        for b in role_ids:
            if a > b:
                continue
            k = _pair_key(a, b)
            parsed["pair_dist"][k] = _safe_float(summary.get(f"role_pair_dist_{a}_{b}_mean_mean"))
            parsed["pair_same"][k] = _safe_float(summary.get(f"role_pair_same_cell_rate_{a}_{b}_mean"))
            parsed["pair_close"][k] = _safe_float(summary.get(f"role_pair_close_rate_{a}_{b}_mean"))
    if not parsed["role_props"]:
        for r in role_ids:
            parsed["role_props"][r] = _safe_float(summary.get(f"role_{r}_fraction_mean"))
    return parsed


def assign_clusterer(parsed: Dict[str, Any]) -> Tuple[int, int]:

    def score(r: int) -> float:
        same = parsed["pair_same"].get(_pair_key(r, r), float("nan"))
        return float(np.nansum([
            parsed["search_rate"].get(r, float("nan")),
            parsed["search_prob_mean"].get(r, float("nan")),
            same,
        ]))

    s0, s1 = score(0), score(1)
    if abs(s0 - s1) < 1e-12:
        prop0 = parsed["role_props"].get(0, 0.0)
        prop1 = parsed["role_props"].get(1, 0.0)
        clusterer = 0 if prop0 >= prop1 else 1
    else:
        clusterer = 0 if s0 > s1 else 1
    explorer = 1 - clusterer
    return clusterer, explorer


def functional_values(parsed: Dict[str, Any]) -> Dict[str, float]:
    c, e = assign_clusterer(parsed)
    return {
        "c_role": float(c), "e_role": float(e),
        "c_share": parsed["role_props"].get(c, float("nan")),
        "e_share": parsed["role_props"].get(e, float("nan")),
        "c_search": parsed["search_rate"].get(c, float("nan")),
        "e_search": parsed["search_rate"].get(e, float("nan")),
        "c_visit_prob": parsed["visit_prob_mean"].get(c, float("nan")),
        "e_visit_prob": parsed["visit_prob_mean"].get(e, float("nan")),
        "c_search_prob": parsed["search_prob_mean"].get(c, float("nan")),
        "e_search_prob": parsed["search_prob_mean"].get(e, float("nan")),
        "cc_dist": parsed["pair_dist"].get(_pair_key(c, c), float("nan")),
        "ce_dist": parsed["pair_dist"].get(_pair_key(c, e), float("nan")),
        "ee_dist": parsed["pair_dist"].get(_pair_key(e, e), float("nan")),
        "cc_same": parsed["pair_same"].get(_pair_key(c, c), float("nan")),
        "ce_same": parsed["pair_same"].get(_pair_key(c, e), float("nan")),
        "ee_same": parsed["pair_same"].get(_pair_key(e, e), float("nan")),
    }


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
    candidates: List[Path] = [summary_path.with_name(Path(name).with_suffix(".jsonl").name)]
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
    for k in ["success", "successful", "found", "target_found", "person_found", "is_success", "done_success"]:
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
    return _get_first_finite(row, ["total_reward", "episode_reward", "reward", "return", "episode_return", "mean_total_reward"])


def _get_steps(row: Dict[str, Any]) -> float:
    return _get_first_finite(row, ["steps", "episode_steps", "episode_len", "episode_length", "length", "timestep", "timesteps"])


def _get_unique_cells(row: Dict[str, Any]) -> float:
    return _get_first_finite(row, ["unique_cells_total", "unique_cells", "n_unique_cells", "visited_unique_cells"])


def _get_explorer_fraction(row: Dict[str, Any], explorer_role: int) -> float:
    key = f"role_{explorer_role}_fraction"
    if key in row:
        return _safe_float(row.get(key))
    props = row.get("role_props") or row.get("role_props_all")
    if isinstance(props, dict):
        return _safe_float(props.get(str(explorer_role), props.get(explorer_role)))
    return float("nan")


def collect_episode_rows(episodes: List[Dict[str, Any]], explorer_role: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for ep in episodes:
        explorer_frac = _get_explorer_fraction(ep, explorer_role)
        if math.isnan(explorer_frac):
            continue
        steps = _get_steps(ep)
        unique_cells = _get_unique_cells(ep)
        unique_per_step = unique_cells / steps if np.isfinite(unique_cells) and np.isfinite(steps) and steps > 0 else float("nan")
        rows.append({
            "explorer_frac": explorer_frac,
            "success": _get_success(ep),
            "reward": _get_reward(ep),
            "unique_per_step": unique_per_step,
            "revisit_fraction": _get_first_finite(ep, ["revisit_fraction", "mean_revisit_fraction", "revisit_fraction_mean"]),
            "repeated_search": _get_first_finite(ep, ["repeated_search_fraction", "mean_repeated_search_fraction", "repeated_search_fraction_mean"]),
            "co_occupancy": _get_first_finite(ep, ["co_occupancy_fraction", "mean_co_occupancy_fraction", "co_occupancy_fraction_mean"]),
        })
    return rows


def split_low_high_explorer(rows: List[Dict[str, float]], q: float = 0.25) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    if not rows:
        return [], []
    vals = np.array([r["explorer_frac"] for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return [], []
    lo_thr = np.nanquantile(vals, q)
    hi_thr = np.nanquantile(vals, 1.0 - q)
    low = [r for r in rows if np.isfinite(r["explorer_frac"]) and r["explorer_frac"] <= lo_thr]
    high = [r for r in rows if np.isfinite(r["explorer_frac"]) and r["explorer_frac"] >= hi_thr]
    return low, high


def summarize_low_high_explorer(rows: List[Dict[str, float]], q: float = 0.25) -> Dict[str, float]:
    low, high = split_low_high_explorer(rows, q=q)
    def mean_valid(group: List[Dict[str, float]], key: str) -> float:
        return _mean_finite([r.get(key, float("nan")) for r in group])
    def mean_reward_success_only(group: List[Dict[str, float]]) -> float:
        vals = [r["reward"] for r in group if np.isfinite(r.get("reward", float("nan"))) and np.isfinite(r.get("success", float("nan"))) and r["success"] > 0]
        return _mean_finite(vals)
    x = np.array([r["explorer_frac"] for r in rows], dtype=float)
    s = np.array([r["success"] for r in rows], dtype=float)
    rew = np.array([r["reward"] for r in rows], dtype=float)
    return {
        "low_success": mean_valid(low, "success"), "high_success": mean_valid(high, "success"),
        "low_reward": mean_valid(low, "reward"), "high_reward": mean_valid(high, "reward"),
        "low_reward_success": mean_reward_success_only(low), "high_reward_success": mean_reward_success_only(high),
        "low_unique_per_step": mean_valid(low, "unique_per_step"), "high_unique_per_step": mean_valid(high, "unique_per_step"),
        "low_revisit_fraction": mean_valid(low, "revisit_fraction"), "high_revisit_fraction": mean_valid(high, "revisit_fraction"),
        "low_repeated_search": mean_valid(low, "repeated_search"), "high_repeated_search": mean_valid(high, "repeated_search"),
        "low_co_occupancy": mean_valid(low, "co_occupancy"), "high_co_occupancy": mean_valid(high, "co_occupancy"),
        "corr_success": pearson_corr(x, s), "corr_reward": pearson_corr(x, rew),
        "low_n": float(len(low)), "high_n": float(len(high)),
    }


def binned_success_by_explorer(rows: List[Dict[str, float]], bins: int = 5) -> List[Dict[str, float]]:
    clean = [r for r in rows if np.isfinite(r["explorer_frac"]) and np.isfinite(r["success"])]
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
        group = [r for r in clean if (lo <= r["explorer_frac"] <= hi if i == len(edges) - 2 else lo <= r["explorer_frac"] < hi)]
        if not group:
            continue
        out.append({
            "x_mean": float(np.mean([r["explorer_frac"] for r in group])),
            "x_lo": float(lo), "x_hi": float(hi),
            "success": float(np.mean([r["success"] for r in group])),
            "n": float(len(group)),
        })
    return out


BEHAVIOUR_METRICS: List[Tuple[str, str]] = [
    ("Search rate", "search_rate"),
    ("Mean belief at visit", "visit_prob_mean"),
    ("Mean belief when searching", "search_prob_mean"),
]


def global_behaviour_xmax(parsed_list: List[Dict[str, Any]]) -> float:
    xmax = 0.0
    for p in parsed_list:
        for _lbl, key in BEHAVIOUR_METRICS:
            for r in (0, 1):
                v = p[key].get(r, float("nan"))
                if np.isfinite(v):
                    xmax = max(xmax, v)
    return _ceil_to(xmax * 1.15, 0.1)


def global_pair_limits(parsed_list: List[Dict[str, Any]]) -> Tuple[float, float]:
    ds, ss = [], []
    for p in parsed_list:
        ds.extend(list(p["pair_dist"].values()))
        ss.extend(list(p["pair_same"].values()))
    dmax = _ceil_to(np.nanmax(ds) * 1.10 if np.any(np.isfinite(ds)) else 1.0, 5.0)
    smax = _ceil_to(np.nanmax(ss) * 1.15 if np.any(np.isfinite(ss)) else 1.0, 0.1)
    return dmax, smax


def draw_pair_geometry_panel(ax: plt.Axes, parsed: Dict[str, Any], dist_max: float, same_max: float, title: str = "Pair geometry", show_xlabel: bool = True, show_ylabel: bool = True) -> None:
    fv = functional_values(parsed)
    labels = ["C–C", "C–E", "E–E"]
    dists_raw = [fv["cc_dist"], fv["ce_dist"], fv["ee_dist"]]
    same_raw = [fv["cc_same"], fv["ce_same"], fv["ee_same"]]
    colors = [C_CLUST, C_GREY, C_EXPL]
    x = np.arange(3)
    dists = [0.0 if not np.isfinite(v) else v for v in dists_raw]
    ax.bar(x, dists, color=colors, width=0.62, edgecolor="none", alpha=0.9)
    ax.set_ylim(0, dist_max)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    if show_ylabel:
        ax.set_ylabel("mean dist.")
    if show_xlabel:
        ax.set_xlabel("pair type")
    ax.set_title(title, loc="left", pad=4)
    ax2 = ax.twinx()
    finite = [(xi, s) for xi, s in zip(x, same_raw) if np.isfinite(s)]
    if finite:
        fx, fs = zip(*finite)
        ax2.plot(fx, fs, "o-", color=C_DARK, markersize=4.5, linewidth=1.0, markeredgecolor="white", markeredgewidth=0.7)
    ax2.set_ylim(0, same_max)
    ax2.set_ylabel("same-cell rate", color=C_DARK, fontsize=8)
    ax2.tick_params(axis="y", labelsize=7, colors=C_DARK)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.spines["top"].set_visible(False)
    for xi, raw, val in zip(x, dists_raw, dists):
        ax.text(xi, val + 0.02 * dist_max, "n/a" if not np.isfinite(raw) else f"{raw:.1f}", ha="center", va="bottom", fontsize=7, color=C_TEXT)


def draw_grouped_role_bars(ax: plt.Axes, parsed: Dict[str, Any], xmax: float, show_ylabel: bool = True, show_xlabel: bool = True, title: str = "") -> None:
    fv = functional_values(parsed)
    metrics = [
        ("Search rate", fv["c_search"], fv["e_search"]),
        ("Mean belief at visit", fv["c_visit_prob"], fv["e_visit_prob"]),
        ("Mean belief when searching", fv["c_search_prob"], fv["e_search_prob"]),
    ]
    y = np.arange(len(metrics))[::-1]
    h = 0.32
    c_vals = [0.0 if not np.isfinite(m[1]) else m[1] for m in metrics]
    e_vals = [0.0 if not np.isfinite(m[2]) else m[2] for m in metrics]
    ax.barh(y + h / 2, c_vals, height=h, color=C_CLUST, edgecolor="none", label="clusterer / searcher")
    ax.barh(y - h / 2, e_vals, height=h, color=C_EXPL, edgecolor="none", label="explorer")
    for yi, (_name, cv, ev), cp, ep in zip(y, metrics, c_vals, e_vals):
        ax.text(cp + 0.008, yi + h / 2, _format_float(cv, 3), va="center", ha="left", fontsize=6.8, color=C_TEXT)
        ax.text(ep + 0.008, yi - h / 2, _format_float(ev, 3), va="center", ha="left", fontsize=6.8, color=C_TEXT)
    ax.set_xlim(0, xmax)
    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in metrics] if show_ylabel else [])
    ax.set_xlabel("rate / probability" if show_xlabel else "")
    ax.set_title(title, loc="left")
    ax.grid(axis="x")
    ax.set_axisbelow(True)


def build_enhanced_rows_figure(parsed_list: List[Dict[str, Any]], labels: List[str], title: str, subtitle: str) -> plt.Figure:
    n = len(parsed_list)
    title_in, legend_in, row_in, interrow_in = 1.05, 0.85, 2.2, 0.7
    fig_w = 7.6
    fig_h = title_in + legend_in + n * row_in + max(0, n - 1) * interrow_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    top = 1.0 - title_in / fig_h
    bottom = legend_in / fig_h
    gs = fig.add_gridspec(nrows=n, ncols=4, width_ratios=[1.0, 0.7, 0.10, 0.78], hspace=interrow_in / row_in, wspace=0.15, left=0.22, right=0.96, top=top, bottom=bottom)
    xmax = global_behaviour_xmax(parsed_list)
    dmax, smax = global_pair_limits(parsed_list)
    behaviour_labels = [m[0] for m in BEHAVIOUR_METRICS]
    behaviour_keys = [m[1] for m in BEHAVIOUR_METRICS]
    for row, (parsed, label) in enumerate(zip(parsed_list, labels)):
        c, e = assign_clusterer(parsed)
        ax_c = fig.add_subplot(gs[row, 0])
        ax_e = fig.add_subplot(gs[row, 1])
        ax_g = fig.add_subplot(gs[row, 3])
        def _draw(ax: plt.Axes, vals: List[float], color: str, role_name: str, role_id: int, share: float, show_y: bool) -> None:
            y = np.arange(len(behaviour_labels))[::-1]
            safe = [0.0 if not np.isfinite(v) else v for v in vals]
            ax.barh(y, safe, color=color, height=0.62, edgecolor="none")
            for yi, raw, val in zip(y, vals, safe):
                ax.text(val + 0.008, yi, _format_float(raw, 3), va="center", ha="left", fontsize=7, color=C_TEXT)
            ax.set_yticks(y)
            ax.set_yticklabels(behaviour_labels if show_y else [])
            ax.set_xlim(0, xmax)
            ax.set_xticks(np.linspace(0, xmax, 5))
            ax.set_xticklabels([f"{t:.1f}" for t in np.linspace(0, xmax, 5)])
            if row == n - 1:
                ax.set_xlabel("rate / probability")
            ax.set_title(role_name, color=color, loc="left", pad=14)
            share_txt = "n/a" if not np.isfinite(share) else f"{share * 100:.0f}% of steps"
            ax.text(0.0, 1.02, f"role {role_id}  ·  {share_txt}", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=color, alpha=0.85)
        _draw(ax_c, [parsed[k].get(c, float("nan")) for k in behaviour_keys], C_CLUST, "Clusterer / searcher", c, parsed["role_props"].get(c, float("nan")), True)
        _draw(ax_e, [parsed[k].get(e, float("nan")) for k in behaviour_keys], C_EXPL, "Explorer", e, parsed["role_props"].get(e, float("nan")), False)
        draw_pair_geometry_panel(ax_g, parsed, dmax, smax, title="Pair geometry" if row == 0 else "", show_xlabel=(row == n - 1), show_ylabel=True)
        pos = ax_c.get_position()
        fig.text(0.025, (pos.y0 + pos.y1) / 2, label, rotation=90, ha="center", va="center", fontsize=10, fontweight="bold", color=C_DARK)
    fig.text(0.02, 1.0 - 0.30 / fig_h, title, fontsize=12, fontweight="bold", color="#111", va="top")
    fig.text(0.02, 1.0 - 0.55 / fig_h, subtitle, fontsize=8.5, color="#555", va="top")
    leg_y = 0.25 / fig_h
    fig.add_artist(plt.Line2D([0.22, 0.25], [leg_y, leg_y], color=C_CLUST, lw=4, solid_capstyle="butt"))
    fig.text(0.26, leg_y, "clusterer / searcher", va="center", fontsize=8, color=C_CLUST)
    fig.add_artist(plt.Line2D([0.47, 0.50], [leg_y, leg_y], color=C_EXPL, lw=4, solid_capstyle="butt"))
    fig.text(0.51, leg_y, "explorer", va="center", fontsize=8, color=C_EXPL)
    fig.text(0.72, leg_y, "C–C, C–E, E–E = pair types", va="center", fontsize=7.5, color="#555", style="italic")
    return fig


def build_compact_rows_figure(parsed_list: List[Dict[str, Any]], labels: List[str]) -> plt.Figure:
    n = len(parsed_list)
    fig_h = 1.2 + 1.95 * n
    fig = plt.figure(figsize=(7.6, fig_h))
    gs = fig.add_gridspec(nrows=n, ncols=2, width_ratios=[1.45, 0.8], hspace=0.58, wspace=0.30, left=0.19, right=0.97, top=0.86, bottom=0.10)
    xmax = global_behaviour_xmax(parsed_list)
    dmax, smax = global_pair_limits(parsed_list)
    for i, (p, lab) in enumerate(zip(parsed_list, labels)):
        ax_b = fig.add_subplot(gs[i, 0])
        ax_g = fig.add_subplot(gs[i, 1])
        draw_grouped_role_bars(ax_b, p, xmax, show_ylabel=True, show_xlabel=(i == n - 1), title="")
        draw_pair_geometry_panel(ax_g, p, dmax, smax, title="" if i > 0 else "Pair geometry", show_xlabel=(i == n - 1), show_ylabel=True)
        fv = functional_values(p)
        ax_b.text(0.0, 1.08, f"searcher role {int(fv['c_role'])} ({fv['c_share'] * 100:.0f}%) · explorer role {int(fv['e_role'])} ({fv['e_share'] * 100:.0f}%)", transform=ax_b.transAxes, ha="left", va="bottom", fontsize=9.2, fontweight="bold", color=C_DARK)
        pos = ax_b.get_position()
        fig.text(0.035, (pos.y0 + pos.y1) / 2, lab, rotation=90, ha="center", va="center", fontsize=10, fontweight="bold")
    handles = [plt.Line2D([0], [0], color=C_CLUST, lw=5), plt.Line2D([0], [0], color=C_EXPL, lw=5), plt.Line2D([0], [0], color=C_GREY, lw=5)]
    fig.legend(handles, ["clusterer / searcher", "explorer", "mixed pair"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.015), frameon=False)
    return fig


def build_scenario_page_figure(parsed: Dict[str, Any], label: str) -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.0, 0.36], width_ratios=[1.35, 0.85], hspace=0.35, wspace=0.32, left=0.12, right=0.96, bottom=0.12, top=0.78)
    ax_b = fig.add_subplot(gs[0, 0])
    ax_g = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[1, :])
    ax_txt.axis("off")
    xmax = global_behaviour_xmax([parsed])
    dmax, smax = global_pair_limits([parsed])
    draw_grouped_role_bars(ax_b, parsed, xmax, show_ylabel=True, show_xlabel=True, title="Behaviour profile")
    draw_pair_geometry_panel(ax_g, parsed, dmax, smax, title="Pair geometry", show_xlabel=True, show_ylabel=True)
    fv = functional_values(parsed)
    s_ratio = _ratio(fv["e_search"], fv["c_search"])
    search_prob_ratio = _ratio(fv["e_search_prob"], fv["c_search_prob"])
    ce_over_cc = _ratio(fv["ce_dist"], fv["cc_dist"])
    interpretation = f"Searcher role {int(fv['c_role'])}: higher direct search behaviour ({fv['c_search']:.3f} search rate, {fv['c_search_prob']:.3f} mean belief when searching).   Explorer role {int(fv['e_role'])}: different search/belief profile ({s_ratio:.2f}× searcher search rate, {search_prob_ratio:.2f}× searcher belief-when-searching) and spatial separation (C–E distance {ce_over_cc:.2f}× C–C distance)."
    ax_txt.text(0.0, 0.65, "Interpretation", fontsize=9.5, fontweight="bold", color=C_DARK, transform=ax_txt.transAxes)
    ax_txt.text(0.0, 0.20, interpretation, fontsize=8.2, color="#555", wrap=True, transform=ax_txt.transAxes)
    fig.text(0.02, 0.96, f"Functional role interpretation: {label}", fontsize=12, fontweight="bold", va="top")
    fig.text(0.02, 0.88, "One-scenario view for detailed Results discussion.", fontsize=8.5, color="#555", va="top")
    return fig


def build_metric_first_figure(parsed_list: List[Dict[str, Any]], labels: List[str], episode_stats: Optional[List[Dict[str, float]]] = None) -> plt.Figure:
    fig = plt.figure(figsize=(7.6, 6.4))
    gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0.52, wspace=0.35, left=0.11, right=0.96, top=0.82, bottom=0.10)
    x = np.arange(len(labels)); width = 0.36
    ax = fig.add_subplot(gs[0, 0])
    c_vals = [functional_values(p)["c_search"] for p in parsed_list]
    e_vals = [functional_values(p)["e_search"] for p in parsed_list]
    ax.bar(x - width/2, c_vals, width, color=C_CLUST, edgecolor="none", label="searcher")
    ax.bar(x + width/2, e_vals, width, color=C_EXPL, edgecolor="none", label="explorer")
    ax.set_title("A. Search activity", loc="left"); ax.set_ylabel("search rate"); ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[0, 1])
    c_prob = [functional_values(p)["c_search_prob"] for p in parsed_list]
    e_prob = [functional_values(p)["e_search_prob"] for p in parsed_list]
    ax.bar(x - width/2, c_prob, width, color=C_CLUST, edgecolor="none")
    ax.bar(x + width/2, e_prob, width, color=C_EXPL, edgecolor="none")
    ax.set_title("B. Belief when searching", loc="left"); ax.set_ylabel("mean belief at search"); ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(axis="y"); ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[1, 0])
    cc = [functional_values(p)["cc_dist"] for p in parsed_list]
    ce = [functional_values(p)["ce_dist"] for p in parsed_list]
    ee = [functional_values(p)["ee_dist"] for p in parsed_list]
    w = 0.25
    ax.bar(x - w, cc, w, color=C_CLUST, edgecolor="none", label="C–C")
    ax.bar(x, ce, w, color=C_GREY, edgecolor="none", label="C–E")
    ax.bar(x + w, ee, w, color=C_EXPL, edgecolor="none", label="E–E")
    ax.set_title("C. Spatial separation", loc="left"); ax.set_ylabel("mean pairwise distance"); ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(loc="upper right", ncol=3, fontsize=7)

    ax = fig.add_subplot(gs[1, 1])
    if episode_stats and len(episode_stats) == len(labels):
        low = [st.get("low_success", float("nan")) for st in episode_stats]
        high = [st.get("high_success", float("nan")) for st in episode_stats]
        ax.bar(x - width/2, low, width, color=C_GREY, edgecolor="none", label="low explorer")
        ax.bar(x + width/2, high, width, color=C_EXPL, edgecolor="none", label="high explorer")
        ax.set_ylabel("success rate"); ax.legend(loc="upper right")
        for xi, lv, hv in zip(x, low, high):
            if np.isfinite(lv) and np.isfinite(hv):
                ax.text(xi, max(lv, hv) + 0.035, f"{lv:.2f}→{hv:.2f}", ha="center", fontsize=7, color="#555")
    else:
        vals = [p["summary"].get("summary", {}).get("success_rate", float("nan")) for p in parsed_list]
        ax.bar(x, vals, width=0.48, color=C_GREY, edgecolor="none")
        ax.set_ylabel("scenario success rate")
        ax.text(0.5, 0.95, "Episode JSONL not found", transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#777")
    ax.set_title("D. Outcome association", loc="left"); ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.08); ax.grid(axis="y"); ax.set_axisbelow(True)
    _add_figure_title(fig, "Role semantics by diagnostic question", "Panels are grouped by interpretation: who searches, where searching is concentrated, who separates, and whether explorer usage is associated with outcomes.", top=0.98)
    return fig


def build_small_multiples_figure(parsed_list: List[Dict[str, Any]], labels: List[str]) -> plt.Figure:
    metrics = [
        ("Search rate", lambda fv: (fv["c_search"], fv["e_search"])),
        ("Belief at visit", lambda fv: (fv["c_visit_prob"], fv["e_visit_prob"])),
        ("Belief at search", lambda fv: (fv["c_search_prob"], fv["e_search_prob"])),
        ("Pair distance", lambda fv: (fv["cc_dist"], fv["ee_dist"])),
        ("Same-cell rate", lambda fv: (fv["cc_same"], fv["ee_same"])),
    ]
    nrows, ncols = len(metrics), len(labels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(7.6, 2.3 * ncols), 1.2 + 1.05 * nrows), squeeze=False)
    ylims = []
    for _name, getter in metrics:
        vals = []
        for p in parsed_list:
            a, b = getter(functional_values(p)); vals.extend([a, b])
        mx = np.nanmax(vals) if np.any(np.isfinite(vals)) else 1.0
        ylims.append(_ceil_to(mx * 1.20, 0.1 if mx < 1 else 5.0))
    for r, (metric_name, getter) in enumerate(metrics):
        for cidx, (p, lab) in enumerate(zip(parsed_list, labels)):
            ax = axes[r, cidx]
            v_c, v_e = getter(functional_values(p))
            vals = [0.0 if not np.isfinite(v_c) else v_c, 0.0 if not np.isfinite(v_e) else v_e]
            ax.bar([0, 1], vals, color=[C_CLUST, C_EXPL], width=0.58, edgecolor="none")
            ax.set_ylim(0, ylims[r]); ax.set_xticks([0, 1]); ax.set_xticklabels(["C", "E"]); ax.grid(axis="y"); ax.set_axisbelow(True)
            if cidx == 0: ax.set_ylabel(metric_name)
            if r == 0: ax.set_title(lab, fontsize=10, fontweight="bold")
            for xi, raw, val in zip([0, 1], [v_c, v_e], vals):
                if np.isfinite(raw): ax.text(xi, val + 0.03 * ylims[r], f"{raw:.2f}" if raw >= 1 else f"{raw:.3f}", ha="center", va="bottom", fontsize=6.8, color=C_TEXT)
    handles = [plt.Line2D([0], [0], color=C_CLUST, lw=5), plt.Line2D([0], [0], color=C_EXPL, lw=5)]
    fig.legend(handles, ["C = clusterer/searcher", "E = explorer"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.015), frameon=False)
    _add_figure_title(fig, "Role diagnostics as small multiples", "Each cell compares clusterer/searcher with explorer for one metric and one scenario.", top=0.98)
    fig.tight_layout(rect=[0.03, 0.06, 0.99, 0.86])
    return fig


def build_explorer_ratio_figure(parsed_list: List[Dict[str, Any]], labels: List[str]) -> plt.Figure:
    metrics = [
        ("Search rate", lambda fv: _ratio(fv["e_search"], fv["c_search"])),
        ("Belief at visit", lambda fv: _ratio(fv["e_visit_prob"], fv["c_visit_prob"])),
        ("Belief at search", lambda fv: _ratio(fv["e_search_prob"], fv["c_search_prob"])),
        ("E–E / C–C distance", lambda fv: _ratio(fv["ee_dist"], fv["cc_dist"])),
        ("C–E / C–C distance", lambda fv: _ratio(fv["ce_dist"], fv["cc_dist"])),
    ]
    n = len(labels); fig, axes = plt.subplots(1, n, figsize=(7.6, 3.65), sharey=True)
    if n == 1: axes = [axes]
    all_vals = []
    for p in parsed_list: all_vals.extend([fn(functional_values(p)) for _name, fn in metrics])
    xmax = _ceil_to(np.nanmax(all_vals) * 1.15 if np.any(np.isfinite(all_vals)) else 1.0, 0.5)
    for ax, p, lab in zip(axes, parsed_list, labels):
        vals = [fn(functional_values(p)) for _name, fn in metrics]
        y = np.arange(len(metrics))[::-1]
        plot_vals = [0.0 if not np.isfinite(v) else v for v in vals]
        colors = [C_EXPL if i < 3 else C_GREY for i in range(len(metrics))]
        ax.barh(y, plot_vals, color=colors, edgecolor="none", height=0.62)
        ax.axvline(1.0, color=C_DARK, linestyle="--", linewidth=0.8, alpha=0.7)
        for yi, raw, val in zip(y, vals, plot_vals): ax.text(val + 0.03, yi, "n/a" if not np.isfinite(raw) else f"{raw:.2f}×", va="center", ha="left", fontsize=7, color=C_TEXT)
        ax.set_xlim(0, xmax); ax.set_yticks(y); ax.set_yticklabels([m[0] for m in metrics] if ax is axes[0] else []); ax.set_title(lab, fontweight="bold"); ax.set_xlabel("explorer / searcher ratio"); ax.grid(axis="x"); ax.set_axisbelow(True)
    _add_figure_title(fig, "Explorer as relative behavioural deviation from the searcher", "Ratios are explorer/searcher. Values below 1 mean the explorer has a lower value for that diagnostic; distance ratios above 1 indicate separation.", top=0.98)
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.80])
    return fig


def build_summary_figure(parsed_list: List[Dict[str, Any]], labels: List[str], episode_stats: Optional[List[Dict[str, float]]] = None) -> plt.Figure:
    fig = plt.figure(figsize=(7.6, 4.3))
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1.0, 1.0], hspace=0.48, wspace=0.38, left=0.10, right=0.96, top=0.80, bottom=0.12)
    x = np.arange(len(labels)); width = 0.36
    ax = fig.add_subplot(gs[0, 0])
    c_search = [functional_values(p)["c_search"] for p in parsed_list]
    e_search = [functional_values(p)["e_search"] for p in parsed_list]
    ax.bar(x - width/2, c_search, width, color=C_CLUST, edgecolor="none"); ax.bar(x + width/2, e_search, width, color=C_EXPL, edgecolor="none")
    ax.set_title("1. Search activity", loc="left"); ax.set_ylabel("search rate"); ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(axis="y"); ax.set_axisbelow(True)
    ax = fig.add_subplot(gs[0, 1])
    search_prob_ratio = [_ratio(functional_values(p)["e_search_prob"], functional_values(p)["c_search_prob"]) for p in parsed_list]
    ax.bar(x, [0 if not np.isfinite(v) else v for v in search_prob_ratio], width=0.55, color=C_EXPL, edgecolor="none"); ax.axhline(1.0, color=C_DARK, linestyle="--", linewidth=0.8)
    ax.set_title("2. Explorer search-belief ratio", loc="left"); ax.set_ylabel("explorer / searcher"); ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(axis="y"); ax.set_axisbelow(True)
    ax = fig.add_subplot(gs[0, 2])
    sep = [_ratio(functional_values(p)["ce_dist"], functional_values(p)["cc_dist"]) for p in parsed_list]
    ax.bar(x, [0 if not np.isfinite(v) else v for v in sep], width=0.55, color=C_GREY, edgecolor="none"); ax.axhline(1.0, color=C_DARK, linestyle="--", linewidth=0.8)
    ax.set_title("3. Mixed-role separation", loc="left"); ax.set_ylabel("C–E / C–C distance"); ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(axis="y"); ax.set_axisbelow(True)
    ax = fig.add_subplot(gs[1, :2])
    if episode_stats and len(episode_stats) == len(labels):
        low = [st.get("low_success", float("nan")) for st in episode_stats]; high = [st.get("high_success", float("nan")) for st in episode_stats]
        ax.bar(x - width/2, low, width, color=C_GREY, edgecolor="none", label="low explorer"); ax.bar(x + width/2, high, width, color=C_EXPL, edgecolor="none", label="high explorer")
        ax.set_ylim(0, 1.08); ax.legend(loc="upper left", ncol=2)
        for xi, lv, hv in zip(x, low, high):
            if np.isfinite(lv) and np.isfinite(hv): ax.text(xi, max(lv, hv) + 0.035, f"{lv:.2f}→{hv:.2f}", ha="center", fontsize=7, color="#555")
        ax.set_ylabel("success rate"); ax.set_title("4. Success under low vs high explorer usage", loc="left")
    else:
        vals = [p["summary"].get("summary", {}).get("success_rate", float("nan")) for p in parsed_list]
        ax.bar(x, vals, width=0.55, color=C_GREY, edgecolor="none"); ax.set_ylim(0, 1.08); ax.set_ylabel("success rate"); ax.set_title("4. Scenario-level success", loc="left")
        ax.text(0.5, 0.90, "Episode JSONL not found", transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#777")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(axis="y"); ax.set_axisbelow(True)
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    return fig


def build_explorer_episode_coverage_figure(parsed_list: List[Dict[str, Any]], episode_paths: List[Path], labels: List[str]) -> plt.Figure:
    stats = []
    for parsed, path in zip(parsed_list, episode_paths):
        _c, e = assign_clusterer(parsed)
        stats.append(summarize_low_high_explorer(collect_episode_rows(load_jsonl(path), explorer_role=e)))
    metrics = [("Unique cells / step", "low_unique_per_step", "high_unique_per_step"), ("Revisit fraction", "low_revisit_fraction", "high_revisit_fraction"), ("Repeated search", "low_repeated_search", "high_repeated_search"), ("Co-occupancy", "low_co_occupancy", "high_co_occupancy")]
    n = len(labels); fig, axes = plt.subplots(1, n, figsize=(7.6, 3.5), sharey=False)
    if n == 1: axes = [axes]
    x = np.arange(len(metrics)); width = 0.36
    for ax, lab, st in zip(axes, labels, stats):
        low = [st.get(lo, float("nan")) for _name, lo, _hi in metrics]; high = [st.get(hi, float("nan")) for _name, _lo, hi in metrics]
        ax.bar(x - width/2, [0 if not np.isfinite(v) else v for v in low], width, color=C_GREY, edgecolor="none", label="low explorer")
        ax.bar(x + width/2, [0 if not np.isfinite(v) else v for v in high], width, color=C_EXPL, edgecolor="none", label="high explorer")
        for xi, lv, hv in zip(x, low, high):
            ymax = np.nanmax([lv, hv])
            if np.isfinite(ymax): ax.text(xi, ymax + 0.04 * max(1.0, ymax), _format_float(_ratio(hv, lv), 2, "×"), ha="center", fontsize=7, color="#555")
        success_txt = f"success {st['low_success']:.2f} → {st['high_success']:.2f}" if np.isfinite(st.get("low_success", float("nan"))) and np.isfinite(st.get("high_success", float("nan"))) else ""
        ax.set_title(f"{lab}\n{success_txt}", fontsize=9.5, fontweight="bold"); ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metrics], rotation=35, ha="right"); ax.grid(axis="y"); ax.set_axisbelow(True)
        ymax = np.nanmax(low + high); ax.set_ylim(0, (ymax if np.isfinite(ymax) else 1.0) * 1.25)
    axes[0].set_ylabel("episode-level mean")
    handles, legend_labels = axes[0].get_legend_handles_labels(); fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(0.16, 0.82), ncol=2, frameon=False)
    _add_figure_title(fig, "Episode-level coverage under low vs high explorer usage", "Episodes are split into bottom and top quartiles by explorer usage. Coverage is measured at team level.", top=0.98)
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.76]); return fig


def build_explorer_success_reward_figure(parsed_list: List[Dict[str, Any]], episode_paths: List[Path], labels: List[str]) -> plt.Figure:
    stats = []
    for parsed, path in zip(parsed_list, episode_paths):
        _c, e = assign_clusterer(parsed)
        stats.append(summarize_low_high_explorer(collect_episode_rows(load_jsonl(path), explorer_role=e)))
    metrics = [("Success rate", "low_success", "high_success"), ("Mean reward", "low_reward", "high_reward"), ("Reward if found", "low_reward_success", "high_reward_success")]
    n = len(labels); fig, axes = plt.subplots(1, n, figsize=(7.6, 3.35), sharey=False)
    if n == 1: axes = [axes]
    x = np.arange(len(metrics)); width = 0.36
    for ax, lab, st in zip(axes, labels, stats):
        low = [st.get(lo, float("nan")) for _name, lo, _hi in metrics]; high = [st.get(hi, float("nan")) for _name, _lo, hi in metrics]
        ax.bar(x - width/2, [0 if not np.isfinite(v) else v for v in low], width, color=C_GREY, edgecolor="none", label="low explorer")
        ax.bar(x + width/2, [0 if not np.isfinite(v) else v for v in high], width, color=C_EXPL, edgecolor="none", label="high explorer")
        for xi, lv, hv in zip(x, low, high):
            ymax = np.nanmax([lv, hv])
            if np.isfinite(ymax): ax.text(xi, ymax + 0.04, f"{lv:.2f} → {hv:.2f}", ha="center", va="bottom", fontsize=7.2, color="#555")
        ax.set_title(lab, fontweight="bold", fontsize=10); ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metrics], rotation=32, ha="right"); ax.grid(axis="y"); ax.set_axisbelow(True)
        ymax = np.nanmax(low + high); ax.set_ylim(0, (ymax if np.isfinite(ymax) else 1.0) * 1.24)
    axes[0].set_ylabel("episode-level mean")
    handles, legend_labels = axes[0].get_legend_handles_labels(); fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(0.16, 0.82), ncol=2, frameon=False)
    _add_figure_title(fig, "Outcome effect of explorer usage", "Episodes are split into low- and high-explorer groups using the bottom and top quartiles of explorer-role usage.", top=0.98)
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.76]); return fig


def build_explorer_success_correlation_figure(parsed_list: List[Dict[str, Any]], episode_paths: List[Path], labels: List[str]) -> plt.Figure:
    n = len(labels); fig, axes = plt.subplots(1, n, figsize=(7.6, 3.35), sharey=True)
    if n == 1: axes = [axes]
    for ax, parsed, path, lab in zip(axes, parsed_list, episode_paths, labels):
        _c, e = assign_clusterer(parsed)
        rows = collect_episode_rows(load_jsonl(path), explorer_role=e)
        bins = binned_success_by_explorer(rows, bins=5)
        x_raw = np.array([r["explorer_frac"] for r in rows], dtype=float); y_raw = np.array([r["success"] for r in rows], dtype=float)
        r_success = pearson_corr(x_raw, y_raw)
        mask = np.isfinite(x_raw) & np.isfinite(y_raw)
        if np.any(mask):
            rng = np.random.default_rng(7); y_jitter = y_raw[mask] + rng.normal(0.0, 0.018, size=np.sum(mask))
            ax.scatter(x_raw[mask], y_jitter, s=7, color=C_GREY, alpha=0.18, edgecolors="none")
        if bins:
            x = np.array([b["x_mean"] for b in bins], dtype=float); y = np.array([b["success"] for b in bins], dtype=float); n_bin = np.array([b["n"] for b in bins], dtype=float)
            ax.plot(x, y, "o-", color=C_EXPL, linewidth=1.4, markersize=5.5, markeredgecolor="white", markeredgewidth=0.8)
            for xi, yi, ni in zip(x, y, n_bin): ax.text(xi, yi + 0.035, f"n={int(ni)}", ha="center", va="bottom", fontsize=6.8, color="#555")
        ax.set_title(lab, fontweight="bold", fontsize=10); ax.set_xlabel("explorer fraction per episode"); ax.set_ylim(-0.05, 1.08); ax.set_xlim(-0.02, 1.02); ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.text(0.03, 0.06, f"corr. with success: {'n/a' if not np.isfinite(r_success) else f'{r_success:+.2f}'}", transform=ax.transAxes, fontsize=7.5, color="#555", ha="left", va="bottom")
    axes[0].set_ylabel("success rate")
    _add_figure_title(fig, "Correlation between explorer usage and success", "Episodes are grouped into quantile bins by explorer-role usage; points show mean success rate per bin.", top=0.98)
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.78]); return fig


LAYOUTS = ["enhanced_rows", "compact_rows", "scenario_pages", "metric_first", "small_multiples", "explorer_ratio", "summary", "episode_coverage", "success_reward", "success_correlation"]


def default_label_for(path: Path) -> str:
    stem = path.stem
    for suffix in ("_summary", ".summary"):
        if stem.endswith(suffix): stem = stem[:-len(suffix)]
    m = re.search(r"\bS\d+\b", stem.upper())
    if m: return m.group(0)
    head = stem.split("_")[0]
    if head and head[0].upper() == "S" and head[1:].isdigit(): return head.upper()
    return stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summaries", nargs="+", help="Evaluation summary JSON files, one per scenario.")
    ap.add_argument("--labels", nargs="+", default=None, help="Labels, one per summary. Default: derived from filenames.")
    ap.add_argument("--out-dir", default="role_layout_gallery", help="Directory to write figures into.")
    ap.add_argument("--basename", default="role_layout", help="Basename prefix for output files.")
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"], help="Output formats. Default: pdf svg png.")
    ap.add_argument("--layout", nargs="+", default=["all"], choices=["all"] + LAYOUTS, help="Layouts to build. Default: all.")
    ap.add_argument("--title", default="Emergent role specialization across scenarios", help="Title for row-wise figures.")
    ap.add_argument("--subtitle", default="Role IDs are not aligned across runs, but the same functional split—clusterer/searcher vs explorer—reappears.", help="Subtitle for row-wise figures.")
    args = ap.parse_args()

    summary_paths = [Path(p) for p in args.summaries]
    if args.labels is not None and len(args.labels) != len(summary_paths):
        ap.error(f"--labels expects exactly {len(summary_paths)} entries, got {len(args.labels)}.")
    labels = list(args.labels) if args.labels is not None else [default_label_for(p) for p in summary_paths]

    parsed_list: List[Dict[str, Any]] = []
    episode_paths: List[Optional[Path]] = []
    for path in summary_paths:
        with path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        parsed_list.append(parse_summary(summary))
        episode_paths.append(_detect_episode_file(path, summary))

    configure_style()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    selected = LAYOUTS if "all" in args.layout else args.layout
    has_episode_files = all(p is not None for p in episode_paths)
    ep_paths = [p for p in episode_paths if p is not None]

    episode_stats: Optional[List[Dict[str, float]]] = None
    if has_episode_files:
        episode_stats = []
        for p, ep in zip(parsed_list, ep_paths):
            _c, e = assign_clusterer(p)
            episode_stats.append(summarize_low_high_explorer(collect_episode_rows(load_jsonl(ep), explorer_role=e)))
    else:
        missing = [str(summary_paths[i]) for i, p in enumerate(episode_paths) if p is None]
        print("[warning] Some episode JSONL files were not detected:")
        for m in missing: print(f"  {m}")
        print("[warning] Episode-level layouts will be skipped.")

    if "enhanced_rows" in selected:
        fig = build_enhanced_rows_figure(parsed_list, labels, args.title, args.subtitle); save_figure(fig, out_dir, f"{args.basename}_01_enhanced_rows", args.formats); plt.close(fig)
    if "compact_rows" in selected:
        fig = build_compact_rows_figure(parsed_list, labels); save_figure(fig, out_dir, f"{args.basename}_02_compact_rows", args.formats); plt.close(fig)
    if "scenario_pages" in selected:
        for p, lab in zip(parsed_list, labels):
            fig = build_scenario_page_figure(p, lab); safe_lab = re.sub(r"[^A-Za-z0-9_-]+", "_", lab); save_figure(fig, out_dir, f"{args.basename}_03_scenario_page_{safe_lab}", args.formats); plt.close(fig)
    if "metric_first" in selected:
        fig = build_metric_first_figure(parsed_list, labels, episode_stats=episode_stats); save_figure(fig, out_dir, f"{args.basename}_04_metric_first", args.formats); plt.close(fig)
    if "small_multiples" in selected:
        fig = build_small_multiples_figure(parsed_list, labels); save_figure(fig, out_dir, f"{args.basename}_05_small_multiples", args.formats); plt.close(fig)
    if "explorer_ratio" in selected:
        fig = build_explorer_ratio_figure(parsed_list, labels); save_figure(fig, out_dir, f"{args.basename}_06_explorer_ratio", args.formats); plt.close(fig)
    if "summary" in selected:
        fig = build_summary_figure(parsed_list, labels, episode_stats=episode_stats); save_figure(fig, out_dir, f"{args.basename}_07_summary", args.formats); plt.close(fig)
    if has_episode_files and "episode_coverage" in selected:
        fig = build_explorer_episode_coverage_figure(parsed_list, ep_paths, labels); save_figure(fig, out_dir, f"{args.basename}_08_explorer_episode_coverage", args.formats); plt.close(fig)
    if has_episode_files and "success_reward" in selected:
        fig = build_explorer_success_reward_figure(parsed_list, ep_paths, labels); save_figure(fig, out_dir, f"{args.basename}_09_explorer_success_reward", args.formats); plt.close(fig)
    if has_episode_files and "success_correlation" in selected:
        fig = build_explorer_success_correlation_figure(parsed_list, ep_paths, labels); save_figure(fig, out_dir, f"{args.basename}_10_explorer_success_correlation", args.formats); plt.close(fig)

    print(f"\nDone. Wrote selected layouts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
