from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.air.config import CheckpointConfig

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from src.envs.make_dsse_from_scenario import make_dsse_env_from_scenario
from src.models.CNNModelR3DM import CNNModelR3DM
from src.policies.r3dm_cnn_policy import R3DMRolePPOTorchPolicyCNN

class R3DMCallbacks(DefaultCallbacks):

    def on_algorithm_init(self, *, algorithm, **kwargs):
        if getattr(algorithm, "_printed_debug_setup", False):
            return
        algorithm._printed_debug_setup = True

        print("=== DEBUG: RLlib setup ===")

        try:
            pol = algorithm.get_policy("shared_policy")
            print("Policy object:", type(pol))

            if pol is not None:
                try:
                    print("Policy is_recurrent():", pol.is_recurrent())
                except Exception as e:
                    print("Could not query is_recurrent():", e)

                try:
                    init_state = pol.get_initial_state()
                    print("Initial state len:", len(init_state))
                    print("Initial state shapes:", [np.shape(s) for s in init_state])
                except Exception as e:
                    print("Could not get initial state:", e)
            else:
                print("Policy not available yet during on_algorithm_init.")

        except Exception as e:
            print("Could not inspect policy during init:", e)

        try:
            print("model:", algorithm.config.get("model"))
            print("multi_agent:", algorithm.config.get("multi_agent"))
        except Exception as e:
            print("Could not print config info:", e)

        print("=== END DEBUG ===")

    def on_train_result(self, *, algorithm, result, **kwargs):
        try:
            cfg = algorithm.config["model"].get("custom_model_config", {})

            tau_start = float(cfg.get("gumbel_tau_start", cfg.get("gumbel_tau", 1.0)))
            tau_end = float(cfg.get("gumbel_tau_end", 0.1))
            anneal_steps = int(cfg.get("gumbel_anneal_steps", 2_000_000))

            timesteps = result.get("timesteps_total", 0)
            frac = min(timesteps / max(anneal_steps, 1), 1.0)
            tau = tau_start + frac * (tau_end - tau_start)

            
            pol = algorithm.get_policy("shared_policy")
            if pol is not None and hasattr(pol, "model"):
                if hasattr(pol.model, "set_gumbel_tau"):
                    pol.model.set_gumbel_tau(tau)
                elif hasattr(pol.model, "gumbel_tau"):
                    pol.model.gumbel_tau = float(max(tau, 1e-6))

                result["r3dm_gumbel_tau_callback"] = tau
                print(f"[tau] timesteps={timesteps}, tau={tau}")


        except Exception as e:
            print(f"[WARN] Could not update gumbel tau in callback: {e}")


def _extract_single_agent_spaces(par_env) -> tuple:
    aid = None
    if hasattr(par_env, "possible_agents") and par_env.possible_agents:
        aid = par_env.possible_agents[0]
    elif hasattr(par_env, "agents") and par_env.agents:
        aid = par_env.agents[0]

    if aid is not None and callable(getattr(par_env, "observation_space", None)):
        obs_space = par_env.observation_space(aid)
        act_space = par_env.action_space(aid)
        return obs_space, act_space

    obs_space = getattr(par_env, "observation_space", None)
    act_space = getattr(par_env, "action_space", None)
    return obs_space, act_space


def train_ppo_cnn_roles(
    scenario_config: Dict[str, Any],
    train_config: Dict[str, Any],
    outdir: str | Path,
) -> None:
    train_config = train_config or {}

    debug_mode = bool(train_config.get("debug", False))

    num_envs = int(train_config.get("num_envs", 6))
    num_gpus = int(train_config.get("num_gpus", 1))

    batch_size = int(train_config.get("batch_size", 12_000))
    total_steps = int(train_config.get("total_timesteps", 20_000_000))
    lr = float(train_config.get("learning_rate", 1e-5))
    gamma = float(train_config.get("gamma", 0.995))
    lam = float(train_config.get("lambda", 0.95))
    use_gae = bool(train_config.get("use_gae", True))

    minibatch = int(train_config.get("minibatch_size", 512))
    num_epochs = int(train_config.get("num_epochs", 15))
    entropy_coef = float(train_config.get("entropy_coef", 0.05))
    clip_param = float(train_config.get("clip_range", 0.1))
    vf_loss_coef = float(train_config.get("vf_coef", 1.0))

    rollout_fragment_length = train_config.get("rollout_fragment_length", "auto")
    seed = train_config.get("seed", 42)
    evaluation_interval = None
    evaluation_num_episodes = 0
    evaluation_explore = False
    
    checkpoint_freq = int(train_config.get("checkpoint_freq", 10))
    keep_checkpoints_num = int(train_config.get("keep_checkpoints_num", 1))
    checkpoint_score_attr = str(
        train_config.get("checkpoint_score_attr", "env_runners/episode_reward_mean")
    )
    checkpoint_score_attr_eval = str(
        train_config.get(
            "checkpoint_score_attr_eval",
            f"evaluation/{checkpoint_score_attr}"
            if not checkpoint_score_attr.startswith("evaluation/")
            else checkpoint_score_attr,
        )
    )
    checkpoint_score_on_eval = False 

    num_roles = int(train_config.get("num_roles", 3))
    role_emb_dim = int(train_config.get("role_emb_dim", 16))
    dyn_horizon = int(train_config.get("dyn_horizon", 5))
    nce_temperature = float(train_config.get("nce_temperature", 0.2))

    aux_loss_coef = float(train_config.get("aux_loss_coef", 0.1))
    intrinsic_reward_coef = float(train_config.get("intrinsic_reward_coef", 0.01))
    role_mi_coef = float(train_config.get("role_mi_coef", 0.01))

    max_nce_pairs_loss = int(train_config.get("max_nce_pairs_loss", 1024))
    max_nce_pairs_intrinsic = int(train_config.get("max_nce_pairs_intrinsic", 512))

    use_gumbel_roles = bool(train_config.get("use_gumbel_roles", True))
    gumbel_tau = float(train_config.get("gumbel_tau", 1.0))
    gumbel_tau_start = float(train_config.get("gumbel_tau_start", gumbel_tau))
    gumbel_tau_end = float(train_config.get("gumbel_tau_end", 0.1))
    gumbel_anneal_steps = int(train_config.get("gumbel_anneal_steps", 2_000_000))
    gumbel_hard = bool(train_config.get("gumbel_hard", False))
    role_temp = float(train_config.get("role_temp", 1.0))

    if debug_mode:
        num_envs = min(num_envs, 3)
        num_gpus = 0
        print(
            "[DEBUG] Running in debug mode on CPU: "
            f"num_envs={num_envs}, batch_size={batch_size}, total_steps={total_steps}"
        )

    ray.init(
       ignore_reinit_error=True,
       include_dashboard=False,
       logging_level="ERROR",
    )

    env_name = "DSSE_PPO_CNN_R3DM"

    def rllib_env_creator(env_config):
        env = make_dsse_env_from_scenario(scenario_config)
        return ParallelPettingZooEnv(env)

    register_env(env_name, rllib_env_creator)
    ModelCatalog.register_custom_model("CNNModelR3DM", CNNModelR3DM)

    tmp_par_env = rllib_env_creator({})
    par_env = getattr(tmp_par_env, "par_env", tmp_par_env)
    obs_space, act_space = _extract_single_agent_spaces(par_env)
    tmp_par_env.close()

    shared_policy_id = "shared_policy"
    policies = {
        shared_policy_id: (R3DMRolePPOTorchPolicyCNN, obs_space, act_space, {})
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return shared_policy_id

    config = (
        PPOConfig()
        .environment(env=env_name, env_config=scenario_config)
        .env_runners(
            num_env_runners=num_envs,
            num_envs_per_env_runner=1,
            rollout_fragment_length=rollout_fragment_length,
            batch_mode="complete_episodes",
        )
        .resources(num_gpus=num_gpus, num_gpus_per_worker=0)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .framework(framework="torch")
        .callbacks(callbacks_class=R3DMCallbacks)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
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
                "custom_model": "CNNModelR3DM",
                "_disable_preprocessor_api": True,
                "custom_model_config": {
                    "num_roles": num_roles,
                    "role_emb_dim": role_emb_dim,
                    "role_temp": role_temp,
                    "use_gumbel_roles": use_gumbel_roles,
                    "gumbel_tau": gumbel_tau_start,
                    "gumbel_tau_start": gumbel_tau_start,
                    "gumbel_tau_end": gumbel_tau_end,
                    "gumbel_anneal_steps": gumbel_anneal_steps,
                    "gumbel_hard": gumbel_hard,
                    "dyn_horizon": dyn_horizon,
                    "nce_temperature": nce_temperature,
                    "aux_loss_coef": aux_loss_coef,
                    "role_mi_coef": role_mi_coef,
                    "intrinsic_reward_coef": intrinsic_reward_coef,
                    "max_nce_pairs_loss": max_nce_pairs_loss,
                    "max_nce_pairs_intrinsic": max_nce_pairs_intrinsic,
                },
            },
        )
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
    )

    if seed is not None:
        config.seed = int(seed)

    if evaluation_interval is not None and evaluation_num_episodes > 0:
        score_attr = checkpoint_score_attr_eval if checkpoint_score_on_eval else checkpoint_score_attr
        config = config.evaluation(
            evaluation_interval=int(evaluation_interval),
            evaluation_duration=evaluation_num_episodes,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": evaluation_explore},
        )
        checkpoint_score_attr_used = score_attr
    else:
        checkpoint_score_attr_used = checkpoint_score_attr

    outdir = Path(outdir)
    storage_path = str(outdir / "ray_res" / env_name)

    tune.run(
        "PPO",
        name="PPO_CNN_R3DM",
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
