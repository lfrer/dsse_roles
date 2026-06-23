from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, Any

import numpy as np


def _safe_mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def _safe_std(xs):
    return float(np.std(xs)) if xs else float("nan")


def _safe_rate(num, den):
    return float(num / den) if den > 0 else float("nan")


def _pair_key(r1: int, r2: int) -> str:
    a, b = sorted((int(r1), int(r2)))
    return f"{a}-{b}"


def _extract_pos(obs_item):
    try:
        pos = np.asarray(obs_item[0])
        if pos.ndim == 1 and len(pos) >= 2:
            return tuple(int(x) for x in pos[:2])
    except Exception:
        pass
    return None


def _extract_prob_matrix(obs_item):
    try:
        return np.asarray(obs_item[1], dtype=np.float64)
    except Exception:
        return None


@dataclass
class RoleEpisodeMetrics:
    role_0_fraction: float = float("nan")
    role_1_fraction: float = float("nan")
    role_entropy_mean_episode: float = float("nan")
    role_entropy_std_episode: float = float("nan")
    role_switches_total: int = 0
    role_switch_rate_episode: float = float("nan")

    role_0_search_rate: float = float("nan")
    role_1_search_rate: float = float("nan")

    role_0_stay_rate: float = float("nan")
    role_1_stay_rate: float = float("nan")

    role_0_backtrack_rate: float = float("nan")
    role_1_backtrack_rate: float = float("nan")

    role_0_visit_prob_mean: float = float("nan")
    role_1_visit_prob_mean: float = float("nan")

    role_0_search_prob_mean: float = float("nan")
    role_1_search_prob_mean: float = float("nan")

    role_pair_dist_0_0_mean: float = float("nan")
    role_pair_dist_0_1_mean: float = float("nan")
    role_pair_dist_1_1_mean: float = float("nan")

    role_pair_close_rate_0_0: float = float("nan")
    role_pair_close_rate_0_1: float = float("nan")
    role_pair_close_rate_1_1: float = float("nan")

    role_pair_same_cell_rate_0_0: float = float("nan")
    role_pair_same_cell_rate_0_1: float = float("nan")
    role_pair_same_cell_rate_1_1: float = float("nan")


class DSSEMetricsRoleCollector:
    def __init__(self, search_action_ids):
        self.search_action_ids = set(search_action_ids)

        self.episode_rows = []

        self._counts_all = Counter()
        self._counts_by_agent = defaultdict(Counter)

        self._entropy_all = []
        self._entropy_by_agent = defaultdict(list)

        self._reset_episode()

    def _reset_episode(self):
        self.role_counts = Counter()
        self.role_entropy = []

        self.role_action_counts = defaultdict(Counter)
        self.role_search_counts = Counter()
        self.role_step_counts = Counter()
        self.role_stay_counts = Counter()
        self.role_backtrack_counts = Counter()

        self.role_visit_probs = defaultdict(list)
        self.role_search_probs = defaultdict(list)

        self.role_pair_distances = defaultdict(list)
        self.role_pair_close = Counter()
        self.role_pair_same_cell = Counter()
        self.role_pair_total = Counter()

        self.role_seq_by_agent = defaultdict(list)
        self.role_switches_by_agent = Counter()
        self.role_steps_by_agent = Counter()

        self.prev_role_by_agent = {}
        self.prev_pos_by_agent = {}
        self.prev_prev_pos_by_agent = {}

    def start_episode(self):
        self._reset_episode()

    def step(
        self,
        obs_before: Dict[str, Any],
        actions: Dict[str, int],
        role_ids_this_step: Dict[str, int],
        role_entropy_this_step: Dict[str, float],
    ):
        for aid, action in actions.items():
            if aid not in obs_before or aid not in role_ids_this_step:
                continue

            role = int(role_ids_this_step[aid])
            ent = role_entropy_this_step.get(aid, float("nan"))

            self.role_counts[role] += 1
            self._counts_all[role] += 1
            self._counts_by_agent[aid][role] += 1

            if not np.isnan(ent):
                self.role_entropy.append(float(ent))
                self._entropy_all.append(float(ent))
                self._entropy_by_agent[aid].append(float(ent))

            self.role_action_counts[role][int(action)] += 1
            self.role_step_counts[role] += 1

            self.role_seq_by_agent[aid].append(role)
            self.role_steps_by_agent[aid] += 1

            if int(action) in self.search_action_ids:
                self.role_search_counts[role] += 1

            obs_item = obs_before[aid]
            pos = _extract_pos(obs_item)
            prob_matrix = _extract_prob_matrix(obs_item)

            if pos is not None and prob_matrix is not None:
                x, y = pos
                if 0 <= y < prob_matrix.shape[0] and 0 <= x < prob_matrix.shape[1]:
                    p = float(prob_matrix[y, x])
                    self.role_visit_probs[role].append(p)
                    if int(action) in self.search_action_ids:
                        self.role_search_probs[role].append(p)

            prev = self.prev_pos_by_agent.get(aid)
            prev_prev = self.prev_prev_pos_by_agent.get(aid)

            if pos is not None:
                if prev is not None and pos == prev:
                    self.role_stay_counts[role] += 1
                if prev_prev is not None and prev is not None and pos == prev_prev and pos != prev:
                    self.role_backtrack_counts[role] += 1

                self.prev_prev_pos_by_agent[aid] = prev
                self.prev_pos_by_agent[aid] = pos

            prev_role = self.prev_role_by_agent.get(aid)
            if prev_role is not None and prev_role != role:
                self.role_switches_by_agent[aid] += 1
            self.prev_role_by_agent[aid] = role

        aids = list(actions.keys())
        for i in range(len(aids)):
            for j in range(i + 1, len(aids)):
                a1, a2 = aids[i], aids[j]
                if a1 not in role_ids_this_step or a2 not in role_ids_this_step:
                    continue
                if a1 not in obs_before or a2 not in obs_before:
                    continue

                r1, r2 = int(role_ids_this_step[a1]), int(role_ids_this_step[a2])
                key = _pair_key(r1, r2)

                p1 = _extract_pos(obs_before[a1])
                p2 = _extract_pos(obs_before[a2])
                if p1 is None or p2 is None:
                    continue

                dist = float(np.linalg.norm(np.array(p1) - np.array(p2)))
                self.role_pair_distances[key].append(dist)
                self.role_pair_total[key] += 1

                if dist <= 1.0:
                    self.role_pair_close[key] += 1
                if p1 == p2:
                    self.role_pair_same_cell[key] += 1

    def end_episode(self, episode_idx: int) -> Dict[str, Any]:
        total_role_steps = sum(self.role_counts.values())
        role_switches_total = sum(self.role_switches_by_agent.values())
        role_steps_total = sum(self.role_steps_by_agent.values())

        row = RoleEpisodeMetrics(
            role_0_fraction=_safe_rate(self.role_counts[0], total_role_steps),
            role_1_fraction=_safe_rate(self.role_counts[1], total_role_steps),
            role_entropy_mean_episode=_safe_mean(self.role_entropy),
            role_entropy_std_episode=_safe_std(self.role_entropy),
            role_switches_total=int(role_switches_total),
            role_switch_rate_episode=_safe_rate(role_switches_total, role_steps_total),

            role_0_search_rate=_safe_rate(self.role_search_counts[0], self.role_step_counts[0]),
            role_1_search_rate=_safe_rate(self.role_search_counts[1], self.role_step_counts[1]),

            role_0_stay_rate=_safe_rate(self.role_stay_counts[0], self.role_step_counts[0]),
            role_1_stay_rate=_safe_rate(self.role_stay_counts[1], self.role_step_counts[1]),

            role_0_backtrack_rate=_safe_rate(self.role_backtrack_counts[0], self.role_step_counts[0]),
            role_1_backtrack_rate=_safe_rate(self.role_backtrack_counts[1], self.role_step_counts[1]),

            role_0_visit_prob_mean=_safe_mean(self.role_visit_probs[0]),
            role_1_visit_prob_mean=_safe_mean(self.role_visit_probs[1]),

            role_0_search_prob_mean=_safe_mean(self.role_search_probs[0]),
            role_1_search_prob_mean=_safe_mean(self.role_search_probs[1]),

            role_pair_dist_0_0_mean=_safe_mean(self.role_pair_distances["0-0"]),
            role_pair_dist_0_1_mean=_safe_mean(self.role_pair_distances["0-1"]),
            role_pair_dist_1_1_mean=_safe_mean(self.role_pair_distances["1-1"]),

            role_pair_close_rate_0_0=_safe_rate(self.role_pair_close["0-0"], self.role_pair_total["0-0"]),
            role_pair_close_rate_0_1=_safe_rate(self.role_pair_close["0-1"], self.role_pair_total["0-1"]),
            role_pair_close_rate_1_1=_safe_rate(self.role_pair_close["1-1"], self.role_pair_total["1-1"]),

            role_pair_same_cell_rate_0_0=_safe_rate(self.role_pair_same_cell["0-0"], self.role_pair_total["0-0"]),
            role_pair_same_cell_rate_0_1=_safe_rate(self.role_pair_same_cell["0-1"], self.role_pair_total["0-1"]),
            role_pair_same_cell_rate_1_1=_safe_rate(self.role_pair_same_cell["1-1"], self.role_pair_total["1-1"]),
        )

        row_dict = asdict(row)
        row_dict["episode_idx"] = int(episode_idx)
        self.episode_rows.append(row_dict)
        return row_dict

    @staticmethod
    def _proportions(counter: Counter):
        total = sum(counter.values())
        if total <= 0:
            return {}
        return {int(k): float(v / total) for k, v in counter.items()}

    def summary(self) -> Dict[str, Any]:
        summary = {}

        numeric_keys = [
            "role_0_fraction",
            "role_1_fraction",
            "role_entropy_mean_episode",
            "role_entropy_std_episode",
            "role_switches_total",
            "role_switch_rate_episode",
            "role_0_search_rate",
            "role_1_search_rate",
            "role_0_stay_rate",
            "role_1_stay_rate",
            "role_0_backtrack_rate",
            "role_1_backtrack_rate",
            "role_0_visit_prob_mean",
            "role_1_visit_prob_mean",
            "role_0_search_prob_mean",
            "role_1_search_prob_mean",
            "role_pair_dist_0_0_mean",
            "role_pair_dist_0_1_mean",
            "role_pair_dist_1_1_mean",
            "role_pair_close_rate_0_0",
            "role_pair_close_rate_0_1",
            "role_pair_close_rate_1_1",
            "role_pair_same_cell_rate_0_0",
            "role_pair_same_cell_rate_0_1",
            "role_pair_same_cell_rate_1_1",
        ]

        for key in numeric_keys:
            vals = [float(r[key]) for r in self.episode_rows if key in r and not np.isnan(r[key])]
            summary[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
            summary[f"{key}_std"] = float(np.std(vals)) if vals else float("nan")

        summary["role_counts_all"] = dict(self._counts_all)
        summary["role_props_all"] = self._proportions(self._counts_all)
        summary["role_counts_by_agent"] = {aid: dict(c) for aid, c in self._counts_by_agent.items()}
        summary["role_props_by_agent"] = {
            aid: self._proportions(c) for aid, c in self._counts_by_agent.items()
        }
        summary["role_entropy"] = {
            "role_entropy_mean_all": float(np.mean(self._entropy_all)) if self._entropy_all else float("nan"),
            "role_entropy_std_all": float(np.std(self._entropy_all)) if self._entropy_all else float("nan"),
            "role_entropy_mean_by_agent": {
                aid: float(np.mean(v)) if v else float("nan")
                for aid, v in self._entropy_by_agent.items()
            },
        }
        summary["role_episode_count"] = len(self.episode_rows)
        summary["role_episode_preview"] = self.episode_rows[:3]
        return summary
