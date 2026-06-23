from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_GRID = "#dddddd"

def _scenario_name(summary_obj: dict, fallback_path: str) -> str:
    scen = summary_obj.get("run_info", {}).get("scenario", "")
    base = os.path.basename(scen) if scen else os.path.basename(fallback_path)
    m = _SCENARIO_RE.search(base)
    if not m:
        parent = os.path.basename(os.path.dirname(fallback_path))
        m = _SCENARIO_RE.search(parent)
    if not m:
        raise ValueError(f"Could not infer scenario name from {fallback_path!r} "
                         f"(scenario field: {scen!r})")
    return m.group(1).upper()


def load_summary(path: str) -> Tuple[str, dict]:
    with open(path) as f:
        obj = json.load(f)
    return _scenario_name(obj, path), obj


def expand_paths(paths_or_globs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for p in paths_or_globs:
        matches = sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
        if not matches:
            print(f"warning: no files matched {p!r}", file=sys.stderr)
            continue
        for m in matches:
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def collect_summaries(paths: List[str]) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for p in paths:
        scen, obj = load_summary(p)
        if scen in result:
            print(f"warning: duplicate scenario {scen}; "
                  f"keeping {p!r}, overwriting earlier file",
                  file=sys.stderr)
        result[scen] = obj
    return result


ROLE_SUMMARY_GLOB = "eval_metrics_cnn_r3dm_summary_*.json"
BASE_SUMMARY_GLOB = "eval_metrics_cnn_summary_*.json"


def collect_from_dir(root: str, glob_pat: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    out: List[str] = []
    for entry in sorted(os.listdir(root)):
        sub = os.path.join(root, entry)
        if not os.path.isdir(sub):
            continue
        if not _SCENARIO_RE.match(entry):
            continue
        matches = sorted(glob.glob(os.path.join(sub, glob_pat)))
        if not matches:
            print(f"warning: no summary in {sub} matching {glob_pat}",
                  file=sys.stderr)
            continue
        out.append(matches[-1])
    return out


def role_metrics(summary: dict) -> dict:
    sm = summary["summary"]
    return {
        "success":         sm["success_rate"],
        "steps_mean":      sm["steps_mean"],
        "ent_mean_all":    summary["role_entropy"]["role_entropy_mean_all"],
        "ent_std_all":     summary["role_entropy"]["role_entropy_std_all"],
        "ent_mean_ep":     summary["role_entropy_mean_episode_mean"],
        "sw_mean":         summary["role_switches_total_mean"],
        "sw_std":          summary["role_switches_total_std"],
    }


def baseline_metrics(summary: dict) -> dict:
    sm = summary["summary"]
    return {"success": sm["success_rate"], "steps_mean": sm["steps_mean"]}


def assemble_table(role_runs: Dict[str, dict],
                   base_runs: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    missing_base = []
    for scen, robj in sorted(role_runs.items(), key=_scenario_sort_key):
        bobj = base_runs.get(scen)
        if bobj is None:
            missing_base.append(scen)
            continue
        rm = role_metrics(robj)
        bm = baseline_metrics(bobj)
        rows.append({
            "scenario": scen,
            "ent": rm["ent_mean_ep"],
            "sw_mean": rm["sw_mean"],
            "sw_std": rm["sw_std"],
            "sw_cv": rm["sw_std"] / max(rm["sw_mean"], 1e-9),
            "delta_success_pp": (rm["success"] - bm["success"]) * 100,
            "role_success": rm["success"],
            "base_success": bm["success"],
        })
    if missing_base:
        print(f"warning: no baseline found for {missing_base}; "
              f"these scenarios will be omitted",
              file=sys.stderr)
    return rows


def _scenario_sort_key(item: Tuple[str, object]) -> Tuple[int, str]:
    name = item[0]
    m = re.match(r"S(\d+)", name)
    return (int(m.group(1)) if m else 999, name)


def plot_combined(rows: List[dict], outdir: str, basename: str,
                  highlight_scenarios: Optional[List[str]] = None) -> None:
    if not rows:
        raise RuntimeError("nothing to plot - no scenarios after merging")

    highlight_scenarios = set(highlight_scenarios or [])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ent = np.array([r["ent"] for r in rows])
    cv = np.array([r["sw_cv"] for r in rows])
    sw_mean = np.array([r["sw_mean"] for r in rows])
    delta = np.array([r["delta_success_pp"] for r in rows])
    names = [r["scenario"] for r in rows]

    vmax = max(abs(delta).max(), 5.0) 
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = mpl.colormaps["RdYlGn"]

    if sw_mean.max() > 0:
        sizes = 80 + (sw_mean / sw_mean.max()) * 320
    else:
        sizes = np.full_like(sw_mean, 100.0)

    sc = ax.scatter(ent, cv, c=delta, s=sizes, cmap=cmap, norm=norm,
                    edgecolor="black", linewidth=0.8, zorder=3)

    default_offsets = {
        "S1": (-12, -2),
        "S2": (10,  -2),
        "S3": (-12,  2),
        "S4": (10,   2),
        "S5": (-12, -2),
        "S6": (10,  -2),
    }
    for x, y, name in zip(ent, cv, names):
        dx, dy = default_offsets.get(name, (10, -2))
        ha = "left" if dx >= 0 else "right"
        weight = "bold" if name in highlight_scenarios else "normal"
        ax.annotate(name, xy=(x, y), xytext=(dx, dy),
                    textcoords="offset points",
                    ha=ha, va="center", fontsize=10, fontweight=weight)

    ax.set_xscale("log")
    ax.set_xlabel(r"Role entropy  $\overline{H}(\mathrm{role})$   (log scale; "
                  r"lower $\Rightarrow$ more committed)")
    ax.set_ylabel(r"Switch-count dispersion  $\sigma_{\mathrm{switches}} / "
                  r"\mu_{\mathrm{switches}}$" + "\n(lower " + r"$\Rightarrow$"
                  + " more consistent switching across episodes)")
    ax.set_title("Role-noise vs task-performance, all scenarios")

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.6)
    ax.xaxis.grid(True, which="both", color=C_GRID, linewidth=0.6)

    x_lo, x_hi = ent.min() * 0.55, ent.max() * 1.80
    ax.set_xlim(x_lo, x_hi)
    y_range = cv.max() - cv.min()
    ax.set_ylim(max(0, cv.min() - y_range * 0.10), cv.max() + y_range * 0.18)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label(r"$\Delta$ success rate vs baseline (pp)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)

    _size_legend(ax, sw_mean.min(), sw_mean.max())

    fig.tight_layout()
    _save(fig, outdir, basename)


def _size_legend(ax, sw_min: float, sw_max: float) -> None:

    if sw_max <= 0:
        return

    def _nice(v: float) -> float:
        if v < 1:
            return round(v, 1)
        if v < 10:
            return round(v)
        return round(v / 5) * 5

    refs = np.array([_nice(sw_min),
                     _nice((sw_min + sw_max) / 2),
                     _nice(sw_max)])
    refs = np.unique(refs)
    ref_sizes = 80 + (refs / sw_max) * 320
    handles = []
    for v, s in zip(refs, ref_sizes):
        label = f"{v:.0f}" if v >= 1 else f"{v:.1f}"
        h = ax.scatter([], [], s=s, facecolor="white",
                       edgecolor="black", linewidth=0.8,
                       label=label)
        handles.append(h)
    leg = ax.legend(handles=handles, title="mean switches\nper episode",
                    loc="upper left", labelspacing=1.1, borderpad=0.7,
                    title_fontsize=8.5, fontsize=8.5,
                    handletextpad=1.0, frameon=True,
                    facecolor="white", edgecolor=C_GRID)
    leg.get_frame().set_linewidth(0.6)


def _save(fig, outdir: str, basename: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    pdf_path = os.path.join(outdir, f"{basename}.pdf")
    png_path = os.path.join(outdir, f"{basename}.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--role", action="append", default=[],
                   help="Path or glob to a role-augmented summary JSON. "
                        "May be passed multiple times.")
    p.add_argument("--baseline", action="append", default=[],
                   help="Path or glob to a baseline summary JSON. "
                        "May be passed multiple times.")
    p.add_argument("--role-dir",
                   help=f"Directory containing S<n>/ subdirs with role "
                        f"summaries (glob {ROLE_SUMMARY_GLOB!r}).")
    p.add_argument("--baseline-dir",
                   help=f"Directory containing S<n>/ subdirs with baseline "
                        f"summaries (glob {BASE_SUMMARY_GLOB!r}).")
    p.add_argument("--outdir", default="thesis_figures",
                   help="Where to write the PDF + PNG (default: %(default)s).")
    p.add_argument("--basename", default="fig_role_noise_combined",
                   help="Output filename stem (default: %(default)s).")
    p.add_argument("--highlight", action="append", default=[],
                   help="Scenario(s) whose label should be bold "
                        "(e.g. --highlight S5). May be repeated.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    role_paths = expand_paths(args.role)
    base_paths = expand_paths(args.baseline)

    if args.role_dir:
        role_paths += collect_from_dir(args.role_dir, ROLE_SUMMARY_GLOB)
    if args.baseline_dir:
        base_paths += collect_from_dir(args.baseline_dir, BASE_SUMMARY_GLOB)

    if not role_paths:
        print("error: provide at least one --role file or --role-dir",
              file=sys.stderr)
        return 2
    if not base_paths:
        print("error: provide at least one --baseline file or --baseline-dir",
              file=sys.stderr)
        return 2

    role_runs = collect_summaries(role_paths)
    base_runs = collect_summaries(base_paths)
    rows = assemble_table(role_runs, base_runs)

    if not rows:
        print("error: no scenarios with both role + baseline data",
              file=sys.stderr)
        return 2

    print(f"plotting {len(rows)} scenarios: "
          f"{', '.join(r['scenario'] for r in rows)}")
    plot_combined(rows, args.outdir, args.basename,
                  highlight_scenarios=args.highlight)
    return 0


if __name__ == "__main__":
    sys.exit(main())
