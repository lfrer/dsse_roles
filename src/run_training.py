import argparse
from pathlib import Path

from src.utils.config import load_json, get_project_root
from src.baselines.random_baseline import run_random_baseline
from src.baselines.train_ppo_cnn import train_ppo_cnn
from src.baselines.train_ppo_lstm import train_ppo_lstm
from src.role_extensions.train_ppo_cnn_r3dm import train_ppo_cnn_roles
from src.role_extensions.train_ppo_lstm_r3dm import train_ppo_lstm_roles
ALGORITHMS = {
    "random": run_random_baseline,
    "ppo_centralized": train_ppo_cnn,
    "ppo_lstm": train_ppo_lstm,
    "ppo_cnn_roles":  train_ppo_cnn_roles,
    "ppo_lstm_roles": train_ppo_lstm_roles
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=ALGORITHMS.keys(),
        help="Which algorithm to run (e.g. random_baseline, ppo_centralized).",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Path to scenario JSON (relative to project root or absolute).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (relative to project root). "
             "If omitted, defaults to logs/<algorithm>.",
    )
    parser.add_argument(
        "--train_config",
        default=None,
        help="Path to training JSON.",
    )    
    args = parser.parse_args()

    root = get_project_root()

    if args.scenario.startswith("/"):
        scenario_path = Path(args.scenario)
    else:
        scenario_path = root / args.scenario

    train_cfg = None
    if args.train_config:
      if args.train_config.startswith("/"):
          train_cfg = Path(args.train_config)
      else:
          train_cfg = root / args.train_config

      train_cfg = load_json(train_cfg)
      

    cfg = load_json(scenario_path)

    if args.outdir is None:
        outdir = root / "logs" / args.algorithm
    else:
        outdir = root / args.outdir

    algo_fn = ALGORITHMS[args.algorithm]
    algo_fn(cfg, train_cfg, outdir)


if __name__ == "__main__":
    main()
