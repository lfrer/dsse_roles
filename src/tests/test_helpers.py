from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Callable, Optional, Tuple, List

import numpy as np
import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPO

from src.utils.config import load_json, get_project_root
from src.envs.make_dsse_from_scenario import make_dsse_env_from_scenario
from src.tests.metrics_collector import DSSEMetricsCollector

@dataclass
class EvalArgs:
    checkpoint: str
    scenario: str
    see: bool = False


def load_scenario_cfg(scenario_arg: str) -> Dict[str, Any]:
    root = get_project_root()
    scenario_path = Path(scenario_arg) if scenario_arg.startswith("/") else (root / scenario_arg)
    cfg = load_json(scenario_path)
    return cfg


def resolve_checkpoint_path(checkpoint_arg: str) -> Path:
    root = get_project_root()
    return Path(checkpoint_arg) if checkpoint_arg.startswith("/") else (root / checkpoint_arg)


def build_env(cfg: Dict[str, Any]):
    return make_dsse_env_from_scenario(cfg)


def register_dsse_env(env_name: str) -> None:
    def env_creator(env_ctx):
        env = build_env(env_ctx)
        return ParallelPettingZooEnv(env)

    register_env(env_name, env_creator)


def init_ray_once():
    if not ray.is_initialized():
        ray.init(
    		ignore_reinit_error=True,
   		 include_dashboard=False,
   		 logging_level="ERROR",
		)


def load_agent(checkpoint_path: Path) -> PPO:
    return PPO.from_checkpoint(checkpoint_path)


def eval_stateless(
    agent,
    env,
    cfg,
    num_episodes: int,
    record_gif: bool,
    recorder_factory=None,
    metrics_jsonl: str | None = None,
    metrics_summary_json: str | None = None,
    run_info: dict | None = None,
):
    grid_size = cfg.get("grid_size", 40)
    search_ids = cfg.get("search_action_ids", [8])

    collector = DSSEMetricsCollector(
        grid_size=grid_size,
        search_action_ids=search_ids,
        save_jsonl=metrics_jsonl,
    )

    rewards, steps = [], []
    founds = 0

    rec = None
    if record_gif:
        if recorder_factory is None:
            raise ValueError("recorder_factory must be provided when record_gif=True")
        rec = recorder_factory()
        rec.__enter__()

    try:
        for _ in range(num_episodes):
            obs, info = env.reset()
            collector.start_episode(obs)

            ep_rew = 0.0
            t = 0

            while env.agents:
                actions = {k: agent.compute_single_action(v, explore=False) for k, v in obs.items()}
                obs, rw, term, trunc, info = env.step(actions)

                ep_rew += float(sum(rw.values()))
                t += 1
                collector.step(actions=actions, next_obs=obs, rewards=rw, infos=info)

                if rec is not None:
                    rec.add_frame()

            ep = collector.end_episode(final_infos=info)
            rewards.append(ep_rew)
            steps.append(t)
            if ep.found:
                founds += 1

        if metrics_summary_json:
            collector.write_summary_json(
                metrics_summary_json,
                extra={
                    "run_info": run_info or {},
                    "episodes_jsonl": metrics_jsonl,
                },
            )

    finally:
        collector.close()
        if rec is not None:
            rec.__exit__(None, None, None)

    return {
        "avg_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "avg_steps": float(np.mean(steps)) if steps else float("nan"),
        "median_steps": float(np.median(steps)) if steps else float("nan"),
        "found_rate": float(founds / num_episodes) if num_episodes else float("nan"),
        "metrics_summary": collector.summary(),
    }


def eval_recurrent(
    agent,
    env,
    cfg,
    num_episodes: int,
    record_gif: bool,
    recorder_factory=None,
    metrics_jsonl: str | None = None,
    metrics_summary_json: str | None = None,
    run_info: dict | None = None,
):
    grid_size = cfg.get("grid_size", 40)
    search_ids = cfg.get("search_action_ids", [8])

    collector = DSSEMetricsCollector(
        grid_size=grid_size,
        search_action_ids=search_ids,
        save_jsonl=metrics_jsonl,
    )

    policy = agent.get_policy("default_policy")
    init_state = policy.get_initial_state()

    rewards, steps = [], []
    founds = 0

    rec = None
    if record_gif:
        if recorder_factory is None:
            raise ValueError("recorder_factory must be provided when record_gif=True")
        rec = recorder_factory()
        rec.__enter__()

    try:
        for _ in range(num_episodes):
            obs, info = env.reset()
            collector.start_episode(obs)

            states = {aid: [np.copy(s) for s in init_state] for aid in obs.keys()}
            ep_rew = 0.0
            t = 0

            while env.agents:
                actions, new_states = {}, {}
                for aid, o in obs.items():
                    if aid not in states:
                        states[aid] = [np.copy(s) for s in init_state]
                    a, s_out, _ = agent.compute_single_action(
                                                               o,
                                                               state=states[aid],
                                                               policy_id="default_policy",
                                                               explore=False,
                                                               )
                    actions[aid] = a
                    new_states[aid] = [np.copy(s) for s in s_out]

                obs, rw, term, trunc, info = env.step(actions)

                ep_rew += float(sum(rw.values()))
                t += 1
                collector.step(actions=actions, next_obs=obs, rewards=rw, infos=info)

                states = new_states
                if rec is not None:
                    rec.add_frame()

            ep = collector.end_episode(final_infos=info)
            rewards.append(ep_rew)
            steps.append(t)
            if ep.found:
                founds += 1

        if metrics_summary_json:
            collector.write_summary_json(
                metrics_summary_json,
                extra={
                    "run_info": run_info or {},
                    "episodes_jsonl": metrics_jsonl,
                },
            )

    finally:
        collector.close()
        if rec is not None:
            rec.__exit__(None, None, None)

    return {
        "avg_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "avg_steps": float(np.mean(steps)) if steps else float("nan"),
        "median_steps": float(np.median(steps)) if steps else float("nan"),
        "found_rate": float(founds / num_episodes) if num_episodes else float("nan"),
        "metrics_summary": collector.summary(),
    }
