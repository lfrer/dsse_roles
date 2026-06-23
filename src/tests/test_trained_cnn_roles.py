import argparse
import shutil
from pathlib import Path
from pprint import pp


import numpy as np
from ray.rllib.models import ModelCatalog

from src.models.CNNModelR3DM import CNNModelR3DM
from src.policies.r3dm_cnn_policy import R3DMRolePPOTorchPolicyCNN
from src.recorder import PygameRecord
from src.tests.test_helpers import (
    load_scenario_cfg,
    resolve_checkpoint_path,
    build_env,
    init_ray_once,
    load_agent,
    register_dsse_env,
)
from src.tests.metrics_collector import DSSEMetricsCollector
from src.tests.metrics_roles_collector import DSSEMetricsRoleCollector

def _role_stats_from_probs(rp: np.ndarray):
    rp = np.asarray(rp, dtype=np.float64)
    rp = np.clip(rp, 1e-12, 1.0)
    rp = rp / rp.sum()
    role_id = int(np.argmax(rp))
    entropy = float(-(rp * np.log(rp)).sum())
    return role_id, entropy


def find_checkpoints(path_str: str) -> list[Path]:
    p = Path(path_str)

    try:
        resolved = Path(resolve_checkpoint_path(str(p)))
        if resolved.exists():
            if resolved.is_file() or resolved.name.startswith("checkpoint_"):
                return [resolved]
    except Exception:
        pass

    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")

    checkpoints = []
    for candidate in sorted(p.rglob("checkpoint_*")):
        if candidate.name.startswith("checkpoint_"):
            checkpoints.append(candidate)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found below: {p}")

    seen = set()
    unique = []
    for ckpt in checkpoints:
        key = str(ckpt.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(ckpt)

    return unique


def better(new_val, best_val, mode: str) -> bool:
    if best_val is None:
        return True
    return new_val > best_val if mode == "max" else new_val < best_val


def eval_stateless_roles(
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

    role_collector = DSSEMetricsRoleCollector(search_action_ids=search_ids)

    rewards, steps = [], []
    founds = 0

    policy = agent.get_policy("shared_policy")
    if policy is None:
        raise RuntimeError("Could not load policy 'shared_policy' from checkpoint.")

    rec = None
    if record_gif:
        if recorder_factory is None:
            raise ValueError("recorder_factory must be provided when record_gif=True")
        rec = recorder_factory()
        rec.__enter__()

    try:
        for ep_idx in range(num_episodes):
            obs, info = env.reset()
            collector.start_episode(obs)
            role_collector.start_episode()

            ep_rew = 0.0
            t = 0

            while env.agents:
                actions = {}

                obs_before = {aid: o for aid, o in obs.items()}
                role_ids_this_step = {}
                role_entropy_this_step = {}

                for aid, o in obs.items():
                    a = agent.compute_single_action(
                        o,
                        explore=False,
                        policy_id="shared_policy",
                    )
                    if isinstance(a, tuple):
                        a = a[0]
                    actions[aid] = a

                    m = getattr(policy, "model", None)
                    role_probs = getattr(m, "last_role_probs", None)

                    if role_probs is not None:
                        try:
                            rp = role_probs.detach().cpu().numpy()
                            rp1 = rp if rp.ndim == 1 else rp[0]
                            role_id, ent = _role_stats_from_probs(rp1)

                            role_ids_this_step[aid] = role_id
                            role_entropy_this_step[aid] = ent
                        except Exception:
                            pass

                role_collector.step(
                    obs_before=obs_before,
                    actions=actions,
                    role_ids_this_step=role_ids_this_step,
                    role_entropy_this_step=role_entropy_this_step,
                )

                obs, rw, term, trunc, info = env.step(actions)

                ep_rew += float(sum(rw.values()))
                t += 1
                collector.step(actions=actions, next_obs=obs, rewards=rw, infos=info)

                if rec is not None:
                    rec.add_frame()

            role_episode_fields = role_collector.end_episode(ep_idx)
            ep = collector.end_episode(
                final_infos=info,
                extra_episode_fields=role_episode_fields,
            )

            rewards.append(ep_rew)
            steps.append(t)
            if ep.found:
                founds += 1

        role_summary = role_collector.summary()

        if metrics_summary_json:
            collector.write_summary_json(
                metrics_summary_json,
                extra={
                    "run_info": run_info or {},
                    "episodes_jsonl": metrics_jsonl,
                    **role_summary,
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
        **role_summary,
    }


argparser = argparse.ArgumentParser()
argparser.add_argument("--checkpoint", type=str, required=True)
argparser.add_argument("--scenario", type=str, required=True)
argparser.add_argument("--see", action="store_true", default=False)
argparser.add_argument(
    "--select_metric",
    type=str,
    default="success_rate",
    help="Metric from metrics_summary used to select the best checkpoint",
)
argparser.add_argument(
    "--mode",
    type=str,
    choices=["max", "min"],
    default="max",
    help="Whether larger or smaller metric is better",
)
args = argparser.parse_args()

scenario_name = Path(args.scenario).stem
scenario_id = scenario_name.split("_")[0]

output_dir = Path("testresults") / "CNN_R3DM" / scenario_id
output_dir.mkdir(parents=True, exist_ok=True)

tmp_root = Path("testresults") / "CNN_R3DM" / "_tmp" / scenario_id
tmp_root.mkdir(parents=True, exist_ok=True)

ModelCatalog.register_custom_model("CNNModelR3DM", CNNModelR3DM)

cfg = load_scenario_cfg(args.scenario)
pp(cfg)

env = build_env(cfg)
register_dsse_env("DSSE_PPO_CNN_R3DM")
init_ray_once()

checkpoint_paths = find_checkpoints(args.checkpoint)
print(f"Found {len(checkpoint_paths)} checkpoints")

best_result = None
best_metric_value = None
best_checkpoint = None

for ckpt_path in checkpoint_paths:
    ckpt_path = ckpt_path.resolve()
    print(f"\nEvaluating checkpoint: {ckpt_path}")

    trial_dir = ckpt_path.parent.parent.name if ckpt_path.parent.parent else "unknown_trial"
    ckpt_dir = ckpt_path.name
    tag = f"{trial_dir}_{ckpt_dir}"

    tmp_jsonl = tmp_root / f"eval_metrics_cnn_r3dm_{tag}.jsonl"
    tmp_summary = tmp_root / f"eval_metrics_cnn_r3dm_summary_{tag}.json"

    try:
        agent = load_agent(str(ckpt_path))
        policy = agent.get_policy("shared_policy")
        print("  is_recurrent:", policy.is_recurrent() if policy is not None else "policy not found")

        res = eval_stateless_roles(
            agent=agent,
            env=env,
            cfg=cfg,
            num_episodes=cfg.get("num_eval_episodes", 32),
            record_gif=False,
            recorder_factory=lambda: PygameRecord("test_trained_cnn_r3dm.gif", 5),
            metrics_jsonl=str(tmp_jsonl),
            metrics_summary_json=str(tmp_summary),
            run_info={
                "checkpoint": str(ckpt_path),
                "scenario": str(args.scenario),
                "model": "CNNModelR3DM",
            },
        )

        summary = res["metrics_summary"]

        if args.select_metric not in summary:
            raise KeyError(
                f"Metric '{args.select_metric}' not found in metrics_summary. "
                f"Available keys: {list(summary.keys())}"
            )

        metric_value = summary[args.select_metric]
        print(f"  {args.select_metric} = {metric_value}")
        print(f"  role_props_all = {res['role_props_all']}")
        print(f"  role_entropy = {res['role_entropy']}")

        if better(metric_value, best_metric_value, args.mode):
            best_metric_value = metric_value
            best_result = {
                "res": res,
                "summary": summary,
                "jsonl": tmp_jsonl,
                "summary_json": tmp_summary,
                "tag": tag,
            }
            best_checkpoint = ckpt_path

    except Exception as e:
        print(f"  Failed for {ckpt_path}: {e}")

if best_result is None:
    env.close()
    raise RuntimeError("No checkpoint could be evaluated successfully.")

print("\nBest checkpoint:")
print(best_checkpoint)
print(f"Best {args.select_metric}: {best_metric_value}")

best_tag = best_result["tag"]
final_jsonl = output_dir / f"eval_metrics_cnn_r3dm_{best_tag}.jsonl"
final_summary = output_dir / f"eval_metrics_cnn_r3dm_summary_{best_tag}.json"

shutil.copy2(best_result["jsonl"], final_jsonl)
shutil.copy2(best_result["summary_json"], final_summary)

if args.see:
    print("\nRe-running best checkpoint with GIF recording...")
    best_agent = load_agent(str(best_checkpoint))
    gif_path = output_dir / f"best_cnn_r3dm_{best_tag}.gif"

    eval_stateless_roles(
        agent=best_agent,
        env=env,
        cfg=cfg,
        num_episodes=cfg.get("num_eval_episodes", 32),
        record_gif=True,
        recorder_factory=lambda: PygameRecord(str(gif_path), 5),
        metrics_jsonl=str(final_jsonl),
        metrics_summary_json=str(final_summary),
        run_info={
            "checkpoint": str(best_checkpoint),
            "scenario": str(args.scenario),
            "model": "CNNModelR3DM",
        },
    )

    print("Saved GIF to:", gif_path)

print("\nSaved best evaluation only:")
print("  Summary:", final_summary)
print("  Episode metrics:", final_jsonl)

shutil.rmtree(tmp_root, ignore_errors=True)
env.close()
