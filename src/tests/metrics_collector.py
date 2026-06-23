# src/eval/metrics_collector.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

def _safe_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= eps:
        return float("nan")
    p = p / s
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def _as_prob_matrix(prob_obs: Any) -> np.ndarray:

    pm = np.asarray(prob_obs)
    if pm.ndim == 2:
        return pm.astype(np.float64, copy=False)
    if pm.ndim == 3:
        return pm[-1].astype(np.float64, copy=False)
    raise ValueError(f"Unexpected prob_matrix shape: {pm.shape}")


def _as_positions_array(pos_obs: Any) -> np.ndarray:

    p = np.asarray(pos_obs)
    if p.ndim == 1:
        return p
    if p.ndim == 2:

        if p.shape[1] == 2 and p.shape[0] > 2:
            return p[-1] if p.shape[0] != 2 else p
        return p
    if p.ndim == 3:
        return p[-1]
    raise ValueError(f"Unexpected positions shape: {p.shape}")


def _default_agent_sort_key(agent_id: str) -> Tuple[int, str]:
    digits = ""
    for ch in reversed(agent_id):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    if digits:
        return (int(digits), agent_id)
    return (10**9, agent_id)


def _pos_to_cell(pos_xy: np.ndarray, grid_size: Optional[int] = None) -> Tuple[int, int]:

    x = float(pos_xy[0])
    y = float(pos_xy[1])
    cx = int(round(x))
    cy = int(round(y))
    if grid_size is not None:
        cx = max(0, min(grid_size - 1, cx))
        cy = max(0, min(grid_size - 1, cy))
    return (cx, cy)


def _quantiles(xs: List[float], qs=(0.1, 0.5, 0.9)) -> Dict[str, float]:
    if not xs:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    arr = np.asarray(xs, dtype=np.float64)
    out = {}
    for q in qs:
        out[f"p{int(q*100)}"] = float(np.quantile(arr, q))
    return out


@dataclass
class EpisodeMetrics:
    episode_idx: int
    steps: int
    total_reward: float
    found: bool
    time_to_find: Optional[int]

    unique_cells_total: int
    revisit_steps_total: int
    revisit_fraction_total: float
    mean_revisit_gap_total: float
    median_revisit_gap_total: float
    min_revisit_gap_total: float

    total_search_actions: int
    unique_searched_cells: int
    repeated_search_actions: int
    repeated_search_fraction: float
    mean_search_gap: float
    median_search_gap: float
    min_search_gap: float

    co_occupancy_steps: int
    co_occupancy_fraction: float
    mean_pairwise_distance: float
    min_pairwise_distance: float

    mean_prob_at_visit: float
    mean_prob_at_search: float
    entropy_start: float
    entropy_end: float
    entropy_drop: float

    backtrack_rate: float       
    stay_rate: float           


class DSSEMetricsCollector:

    def __init__(
        self,
        grid_size: Optional[int] = None,
        search_action_ids: Optional[List[int]] = None,
        agent_order: Optional[List[str]] = None,
        save_jsonl: Optional[Union[str, Path]] = None,
    ):
        self.grid_size = grid_size
        self.search_action_ids = set(search_action_ids) if search_action_ids else None
        self.agent_order = agent_order 
        self.save_jsonl = Path(save_jsonl) if save_jsonl else None

        self.episodes: List[EpisodeMetrics] = []
        self._jsonl_fh = None

        self._ep_idx = 0
        self._t = 0
        self._agents: List[str] = []
        self._found = False
        self._time_to_find: Optional[int] = None
        self._reward_sum = 0.0

        self._cells_by_agent: Dict[str, List[Tuple[int, int]]] = {}
        self._actions_by_agent: Dict[str, List[int]] = {}
        self._prob_at_visit: List[float] = []
        self._prob_at_search: List[float] = []
        self._entropy_series: List[float] = []

        self._last_visit_time_global: Dict[Tuple[int, int], int] = {}
        self._revisit_gaps_global: List[int] = []

        self._search_times_by_cell: Dict[Tuple[int, int], List[int]] = {}
        self._search_gaps: List[int] = []

        self._co_occupancy_steps = 0
        self._pairwise_distances: List[float] = []
        self._min_pairwise_distances: List[float] = []

        self._backtracks = 0
        self._stays = 0

    def _ensure_jsonl(self):
        if self.save_jsonl and self._jsonl_fh is None:
            self.save_jsonl.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_fh = self.save_jsonl.open("a", encoding="utf-8")

    def close(self):
        if self._jsonl_fh is not None:
            self._jsonl_fh.close()
            self._jsonl_fh = None

    def start_episode(self, obs: Dict[str, Any]):
        self._t = 0
        self._reward_sum = 0.0
        self._found = False
        self._time_to_find = None

        agents = list(obs.keys())
        if self.agent_order:
            agents = [a for a in self.agent_order if a in obs]
        else:
            agents = sorted(agents, key=_default_agent_sort_key)

        self._agents = agents
        self._cells_by_agent = {a: [] for a in agents}
        self._actions_by_agent = {a: [] for a in agents}

        self._prob_at_visit = []
        self._prob_at_search = []
        self._entropy_series = []

        self._last_visit_time_global = {}
        self._revisit_gaps_global = []

        self._search_times_by_cell = {}
        self._search_gaps = []

        self._co_occupancy_steps = 0
        self._pairwise_distances = []
        self._min_pairwise_distances = []

        self._backtracks = 0
        self._stays = 0

        self._record_from_obs(obs, actions=None)

    def step(
        self,
        actions: Dict[str, Any],
        next_obs: Dict[str, Any],
        rewards: Dict[str, float],
        infos: Dict[str, Any],
    ):
        self._t += 1
        self._reward_sum += float(sum(rewards.values()))

        if not self._found:
            for _, v in infos.items():
                if isinstance(v, dict) and v.get("Found", False):
                    self._found = True
                    self._time_to_find = self._t
                    break

        self._record_from_obs(next_obs, actions=actions)

    def end_episode(
    self,
    final_infos: Dict[str, Any],
    extra_episode_fields: Optional[Dict[str, Any]] = None,
) -> EpisodeMetrics:
        if not self._found:
            for _, v in final_infos.items():
                if isinstance(v, dict) and v.get("Found", False):
                    self._found = True
                    self._time_to_find = self._t
                    break

        steps = self._t
        total_reward = float(self._reward_sum)

        all_cells = []
        revisit_steps = 0

        visited_global = set()
        for tt in range(len(self._cells_by_agent[self._agents[0]])):
            step_cells = [self._cells_by_agent[a][tt] for a in self._agents]
            for c in step_cells:
                if c in visited_global:
                    revisit_steps += 1
                visited_global.add(c)
            all_cells.extend(step_cells)

        unique_cells_total = len(set(all_cells))
        revisit_fraction = (revisit_steps / (steps * max(1, len(self._agents)))) if steps > 0 else 0.0

        gaps = self._revisit_gaps_global
        mean_gap = float(np.mean(gaps)) if gaps else float("nan")
        med_gap = float(np.median(gaps)) if gaps else float("nan")
        min_gap = float(np.min(gaps)) if gaps else float("nan")

        total_search_actions = 0
        unique_searched_cells = 0
        repeated_search_actions = 0
        repeated_search_fraction = float("nan")
        mean_search_gap = float("nan")
        median_search_gap = float("nan")
        min_search_gap = float("nan")

        if self.search_action_ids is not None:
            searched_cells = []
            for a in self._agents:
                for tt, act in enumerate(self._actions_by_agent[a]):
                    if act in self.search_action_ids:
                        cell = self._cells_by_agent[a][tt + 1]  
                        searched_cells.append(cell)
                        total_search_actions += 1

            unique_searched_cells = len(set(searched_cells))
            seen = set()
            for c in searched_cells:
                if c in seen:
                    repeated_search_actions += 1
                seen.add(c)

            repeated_search_fraction = (
                repeated_search_actions / total_search_actions if total_search_actions > 0 else 0.0
            )

            if self._search_gaps:
                mean_search_gap = float(np.mean(self._search_gaps))
                median_search_gap = float(np.median(self._search_gaps))
                min_search_gap = float(np.min(self._search_gaps))
            else:
                mean_search_gap = float("nan")
                median_search_gap = float("nan")
                min_search_gap = float("nan")

        co_occ_frac = (self._co_occupancy_steps / steps) if steps > 0 else 0.0
        mean_pwd = float(np.mean(self._pairwise_distances)) if self._pairwise_distances else float("nan")
        min_pwd = float(np.min(self._min_pairwise_distances)) if self._min_pairwise_distances else float("nan")

        mean_prob_visit = float(np.mean(self._prob_at_visit)) if self._prob_at_visit else float("nan")
        mean_prob_search = float(np.mean(self._prob_at_search)) if self._prob_at_search else float("nan")

        ent0 = float(self._entropy_series[0]) if self._entropy_series else float("nan")
        ent1 = float(self._entropy_series[-1]) if self._entropy_series else float("nan")
        ent_drop = float(ent0 - ent1) if (not math.isnan(ent0) and not math.isnan(ent1)) else float("nan")

        denom = steps * max(1, len(self._agents))
        backtrack_rate = (self._backtracks / denom) if denom > 0 else 0.0
        stay_rate = (self._stays / denom) if denom > 0 else 0.0

        ep = EpisodeMetrics(
            episode_idx=self._ep_idx,
            steps=steps,
            total_reward=total_reward,
            found=bool(self._found),
            time_to_find=self._time_to_find,

            unique_cells_total=unique_cells_total,
            revisit_steps_total=revisit_steps,
            revisit_fraction_total=float(revisit_fraction),
            mean_revisit_gap_total=mean_gap,
            median_revisit_gap_total=med_gap,
            min_revisit_gap_total=min_gap,

            total_search_actions=int(total_search_actions),
            unique_searched_cells=int(unique_searched_cells),
            repeated_search_actions=int(repeated_search_actions),
            repeated_search_fraction=float(repeated_search_fraction) if not math.isnan(repeated_search_fraction) else float("nan"),
            mean_search_gap=float(mean_search_gap),
            median_search_gap=float(median_search_gap),
            min_search_gap=float(min_search_gap),

            co_occupancy_steps=int(self._co_occupancy_steps),
            co_occupancy_fraction=float(co_occ_frac),
            mean_pairwise_distance=float(mean_pwd),
            min_pairwise_distance=float(min_pwd),

            mean_prob_at_visit=float(mean_prob_visit),
            mean_prob_at_search=float(mean_prob_search),
            entropy_start=float(ent0),
            entropy_end=float(ent1),
            entropy_drop=float(ent_drop),

            backtrack_rate=float(backtrack_rate),
            stay_rate=float(stay_rate),
        )

        self.episodes.append(ep)
        self._ep_idx += 1
       
        row = asdict(ep)
        if extra_episode_fields:
            row.update(extra_episode_fields)

        if self.save_jsonl:
            self._ensure_jsonl()
            self._jsonl_fh.write(json.dumps(row) + "\n")
            self._jsonl_fh.flush()

        return ep

    def _record_from_obs(self, obs: Dict[str, Any], actions: Optional[Dict[str, Any]]):
        any_agent = self._agents[0]
        prob_obs = obs[any_agent][1]
        prob = _as_prob_matrix(prob_obs)
        self._entropy_series.append(_safe_entropy(prob))
        pos_any = _as_positions_array(obs[any_agent][0])

        cells: Dict[str, Tuple[int, int]] = {}

        if pos_any.ndim == 2 and pos_any.shape[1] == 2:
            for idx, a in enumerate(self._agents):
                if idx < pos_any.shape[0]:
                    cells[a] = _pos_to_cell(pos_any[idx], self.grid_size)
                else:
                    pa = _as_positions_array(obs[a][0])
                    if pa.ndim == 1:
                        cells[a] = _pos_to_cell(pa, self.grid_size)
                    else:
                        cells[a] = _pos_to_cell(pa[-1], self.grid_size)
        else:
            for a in self._agents:
                pa = _as_positions_array(obs[a][0])
                if pa.ndim == 1:
                    cells[a] = _pos_to_cell(pa, self.grid_size)
                else:
                    cells[a] = _pos_to_cell(pa[-1], self.grid_size)

        step_cells = [cells[a] for a in self._agents]

        if len(step_cells) >= 2:
            if len(set(step_cells)) < len(step_cells):
                self._co_occupancy_steps += 1

            dists = []
            for (x1, y1), (x2, y2) in combinations(step_cells, 2):
                d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                dists.append(d)
            if dists:
                self._pairwise_distances.append(float(np.mean(dists)))
                self._min_pairwise_distances.append(float(np.min(dists)))

        for a in self._agents:
            c = cells[a]
            self._cells_by_agent[a].append(c)

            px = c[0]
            py = c[1]
            if 0 <= py < prob.shape[0] and 0 <= px < prob.shape[1]:
                self._prob_at_visit.append(float(prob[py, px]))  
            else:
                self._prob_at_visit.append(float("nan"))

            if c in self._last_visit_time_global:
                gap = self._t - self._last_visit_time_global[c]
                if gap >= 0:
                    self._revisit_gaps_global.append(gap)
            self._last_visit_time_global[c] = self._t

            traj = self._cells_by_agent[a]
            if len(traj) >= 2 and traj[-1] == traj[-2]:
                self._stays += 1
            if len(traj) >= 3 and traj[-1] == traj[-3] and traj[-2] != traj[-1]:
                self._backtracks += 1

            if actions is not None:
                act = actions.get(a, None)
                if act is None:
                    act_i = -1
                else:
                    try:
                        act_i = int(act)
                    except Exception:
                        act_i = -1
                self._actions_by_agent[a].append(act_i)

                if self.search_action_ids is not None and act_i in self.search_action_ids:
                    if 0 <= py < prob.shape[0] and 0 <= px < prob.shape[1]:
                        self._prob_at_search.append(float(prob[py, px]))
                    else:
                        self._prob_at_search.append(float("nan"))

                    times = self._search_times_by_cell.setdefault(c, [])
                    if times:
                        gap = self._t - times[-1]
                        if gap >= 0:
                            self._search_gaps.append(gap)
                    times.append(self._t)

    def summary(self) -> Dict[str, Any]:
        if not self.episodes:
            return {}

        founds = [1.0 if e.found else 0.0 for e in self.episodes]
        steps = [e.steps for e in self.episodes]
        rewards = [e.total_reward for e in self.episodes]
        ttf = [e.time_to_find for e in self.episodes if e.time_to_find is not None]

        def mean(xs):
            return float(np.mean(xs)) if xs else float("nan")

        def median(xs):
            return float(np.median(xs)) if xs else float("nan")

        out = {
            "n_episodes": len(self.episodes),
            "success_rate": mean(founds),

            "reward_mean": mean(rewards),
            "reward_median": median(rewards),
            **{f"reward_{k}": v for k, v in _quantiles(rewards).items()},

            "steps_mean": mean(steps),
            "steps_median": median(steps),
            **{f"steps_{k}": v for k, v in _quantiles(steps).items()},

            "time_to_find_mean": mean(ttf),
            "time_to_find_median": median(ttf),
            **{f"time_to_find_{k}": v for k, v in _quantiles(ttf).items()},

            "unique_cells_total_mean": mean([e.unique_cells_total for e in self.episodes]),
            "revisit_fraction_mean": mean([e.revisit_fraction_total for e in self.episodes]),
            "mean_revisit_gap_mean": mean([e.mean_revisit_gap_total for e in self.episodes]),

            "search_actions_mean": mean([e.total_search_actions for e in self.episodes]),
            "repeated_search_fraction_mean": mean([e.repeated_search_fraction for e in self.episodes if not math.isnan(e.repeated_search_fraction)]),

            "co_occupancy_fraction_mean": mean([e.co_occupancy_fraction for e in self.episodes]),
            "mean_pairwise_distance_mean": mean([e.mean_pairwise_distance for e in self.episodes if not math.isnan(e.mean_pairwise_distance)]),

            "mean_prob_at_visit_mean": mean([e.mean_prob_at_visit for e in self.episodes if not math.isnan(e.mean_prob_at_visit)]),
            "mean_prob_at_search_mean": mean([e.mean_prob_at_search for e in self.episodes if not math.isnan(e.mean_prob_at_search)]),
            "entropy_drop_mean": mean([e.entropy_drop for e in self.episodes if not math.isnan(e.entropy_drop)]),

            "backtrack_rate_mean": mean([e.backtrack_rate for e in self.episodes]),
            "stay_rate_mean": mean([e.stay_rate for e in self.episodes]),
        }
        return out


    def write_summary_json(self, path: str | Path, extra: Optional[Dict[str, Any]] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {"summary": self.summary()}
        if extra:
            payload.update(extra)

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
