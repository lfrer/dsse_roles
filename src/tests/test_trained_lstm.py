import argparse
import shutil
from pathlib import Path
from pprint import pp
import traceback
import numpy as np
from ray.rllib.models import ModelCatalog

from src.models.CNNLSTMModel import CNNLSTMModel
from src.recorder import PygameRecord
from src.tests.test_helpers import (
    load_scenario_cfg,
    resolve_checkpoint_path,
    build_env,
    init_ray_once,
    load_agent,
    eval_recurrent,
    register_dsse_env
)

print("USING TEST_TRAINED_LSTM FROM:", __file__)
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Single checkpoint path OR directory containing multiple checkpoints",
)
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


def find_checkpoints(path_str: str) -> list[Path]:
    p = Path(path_str)

    try:
        resolved = Path(resolve_checkpoint_path(str(p))).resolve()
        if resolved.exists() and resolved.name.startswith("checkpoint_"):
            return [resolved]
    except Exception:
        pass

    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")

    checkpoints = []
    for candidate in sorted(p.rglob("checkpoint_*")):
        if candidate.name.startswith("checkpoint_"):
            checkpoints.append(candidate.resolve())

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found below: {p}")

    seen = set()
    unique = []
    for ckpt in checkpoints:
        key = str(ckpt)
        if key not in seen:
            seen.add(key)
            unique.append(ckpt)

    return unique


def better(new_val, best_val, mode: str) -> bool:
    if best_val is None:
        return True
    return new_val > best_val if mode == "max" else new_val < best_val


scenario_id = Path(args.scenario).stem.split("_")[0]

output_dir = Path("testresults") / "LSTM" / scenario_id
output_dir.mkdir(parents=True, exist_ok=True)

tmp_root = Path("testresults") / "LSTM" / "_tmp" / scenario_id
tmp_root.mkdir(parents=True, exist_ok=True)

ModelCatalog.register_custom_model("CNNLSTMModel", CNNLSTMModel)

cfg = load_scenario_cfg(args.scenario)
pp(cfg)

env = build_env(cfg)
register_dsse_env("DSSE_PPO_LSTM")
init_ray_once()

checkpoint_paths = find_checkpoints(args.checkpoint)
print(f"Found {len(checkpoint_paths)} checkpoints")

best_result = None
best_metric_value = None
best_checkpoint = None

for ckpt_path in checkpoint_paths:
    ckpt_path = ckpt_path.resolve()
    print(f"\nEvaluating checkpoint: {ckpt_path}")

    trial_dir = ckpt_path.parent.parent.name
    ckpt_dir = ckpt_path.name
    tag = f"{trial_dir}_{ckpt_dir}"

    tmp_jsonl = tmp_root / f"eval_metrics_lstm_{tag}.jsonl"
    tmp_summary = tmp_root / f"eval_metrics_lstm_summary_{tag}.json"

    try:
        agent = load_agent(str(ckpt_path))

        policy = agent.get_policy()
        print("is_recurrent:", policy.is_recurrent())
        print("init_state shapes:", [np.shape(s) for s in policy.get_initial_state()])

        res = eval_recurrent(
            agent=agent,
            env=env,
            cfg=cfg,
            num_episodes=cfg.get("num_eval_episodes", 32),
            record_gif=False,
            recorder_factory=lambda: PygameRecord("test_trained_lstm.gif", 5),
            metrics_jsonl=str(tmp_jsonl),
            metrics_summary_json=str(tmp_summary),
            run_info={
                "checkpoint": str(ckpt_path),
                "scenario": str(args.scenario),
                "model": "CNNLSTMModel",
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
        print("TRACEBACK MARKER 123")
        traceback.print_exc()

if best_result is None:
    env.close()
    raise RuntimeError("No checkpoint could be evaluated successfully.")

print("\nBest checkpoint:")
print(best_checkpoint)
print(f"Best {args.select_metric}: {best_metric_value}")

best_tag = best_result["tag"]
final_jsonl = output_dir / f"eval_metrics_lstm_{best_tag}.jsonl"
final_summary = output_dir / f"eval_metrics_lstm_summary_{best_tag}.json"

shutil.copy2(best_result["jsonl"], final_jsonl)
shutil.copy2(best_result["summary_json"], final_summary)

if args.see:
    print("\nRe-running best checkpoint with GIF recording...")
    best_agent = load_agent(str(best_checkpoint))
    gif_path = output_dir / f"best_lstm_{best_tag}.gif"

    res = eval_recurrent(
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
            "model": "CNNLSTMModel",
        },
    )

    print("Saved GIF to:", gif_path)

print("\nSaved best evaluation only:")
print("  Summary:", final_summary)
print("  Episode metrics:", final_jsonl)

shutil.rmtree(tmp_root, ignore_errors=True)
env.close()
