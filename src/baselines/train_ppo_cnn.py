import os
from pathlib import Path
from typing import Dict, Any

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.tune import CheckpointConfig

from src.stopcriteria.plateaustopper import PlateauStopper
from DSSE import DroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from torch import nn
import torch

from src.envs.randomized_vec_env import RandomVectorAndStartDroneSwarmSearch
from src.envs.RewardScheme import SparseProbSearchWrapper
from src.models.CNNModel import CNNModel
from src.envs.make_dsse_from_scenario import make_dsse_env_from_scenario


def make_dsse_env_from_scenario_cnn(scenario_config: Dict[str, Any]):

    env = make_dsse_env_from_scenario(scenario_config)

    return env


def train_ppo_cnn(
    scenario_config: Dict[str, Any],
    train_config: Dict[str, Any],
    outdir: str | Path,
) -> None:
    train_config = train_config or {}

    # ----- Hyperparameters from JSON -----
    debug_mode = bool(train_config.get("debug", False))

    num_envs = int(train_config.get("num_envs", 1))
    num_gpus = int(train_config.get("num_gpus", 1))

    batch_size = int(train_config.get("batch_size", 8192))
    total_steps = int(train_config.get("total_timesteps", 20_000_000))
    lr = float(train_config.get("learning_rate", 1e-5))
    gamma = float(train_config.get("gamma", 0.9999999))
    lam = float(train_config.get("lambda", 0.95))
    use_gae = bool(train_config.get("use_gae", True))

    minibatch = int(train_config.get("minibatch_size", 300))
    num_epochs = int(train_config.get("num_epochs", 10))
    entropy_coef = float(train_config.get("entropy_coef", 0.0))
    clip_param = float(train_config.get("clip_range", 0.3))
    vf_loss_coef = float(train_config.get("vf_coef", 1.0))

    rollout_fragment_length = train_config.get("rollout_fragment_length", "auto")

    seed = train_config.get("seed", None)

    evaluation_interval = None
    evaluation_num_episodes = 0
    evaluation_explore = False

    checkpoint_freq = int(train_config.get("checkpoint_freq", 10))
    keep_checkpoints_num = int(train_config.get("keep_checkpoints_num", 1))

    checkpoint_score_attr = str(train_config.get("checkpoint_score_attr", "env_runners/episode_reward_mean"))

    checkpoint_score_attr_eval = str(
        train_config.get(
            "checkpoint_score_attr_eval",
            f"evaluation/{checkpoint_score_attr}" if not checkpoint_score_attr.startswith("evaluation/") else checkpoint_score_attr,
        )
    )

    checkpoint_score_on_eval = False 

    if debug_mode:
        num_envs = min(num_envs, 3)
        num_gpus = 0
        print(
            "[DEBUG] Running in debug mode on CPU: "
            f"num_envs={num_envs}, batch_size={batch_size}, total_steps={total_steps}"
        )

    # ----- Start Ray -----
    ray.init(
       ignore_reinit_error=True,
       include_dashboard=False,
       logging_level="ERROR",
    )

    env_name = "DSSE_PPO_Baseline"

    def rllib_env_creator(env_config):
        env = make_dsse_env_from_scenario_cnn(env_config)
        return ParallelPettingZooEnv(env)

    register_env(env_name, rllib_env_creator)
    ModelCatalog.register_custom_model("CNNModel", CNNModel)

    # ----- PPOConfig -----
    config = (
        PPOConfig()
        .environment(env=env_name, env_config=scenario_config)
        .env_runners(
            num_env_runners=num_envs,
            num_envs_per_env_runner=1,
            rollout_fragment_length=rollout_fragment_length,
            batch_mode="complete_episodes",
        )
        .resources(
            num_gpus=num_gpus,
            num_gpus_per_worker=0,
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .training(
            train_batch_size=batch_size,
            lr=lr,
            gamma=gamma,
            lambda_=lam,
            use_gae=use_gae,
            entropy_coeff=entropy_coef,
            clip_param=clip_param,
            vf_loss_coeff=vf_loss_coef,
            minibatch_size=minibatch,
            num_epochs=num_epochs,
            model={
                "custom_model": "CNNModel",
                "_disable_preprocessor_api": True,
            },
        )
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
    )

    if seed is not None:
        config.seed = int(seed)

    config = config.multi_agent(
        policies={"default_policy": (None, None, None, {})},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "default_policy",
    )

    if evaluation_interval is not None and evaluation_num_episodes > 0:
        if checkpoint_score_on_eval:
            checkpoint_score_attr = checkpoint_score_attr_eval

        config = config.evaluation(
            evaluation_interval=int(evaluation_interval),
            evaluation_duration=evaluation_num_episodes,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": evaluation_explore},
        )

    # ----- Tune run -----
    outdir = Path(outdir)
    storage_path = str(outdir / "ray_res" / env_name)

    tune.run(
        "PPO",
        name="PPO_CNN",
        #stop={"timesteps_total": total_steps}
        stop={"timesteps_total": total_steps},
        checkpoint_config=CheckpointConfig(
        checkpoint_frequency=10,
        num_to_keep=10,
        checkpoint_score_attribute="env_runners/episode_return_mean",
        checkpoint_score_order="max",
        checkpoint_at_end=False,
    ),
        storage_path=storage_path,
        config=config.to_dict(),
    )

    ray.shutdown()
