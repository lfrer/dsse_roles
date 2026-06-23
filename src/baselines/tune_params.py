from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import ray
from ray import tune
from ray.tune import Tuner, TuneConfig, RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CheckpointConfig

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from src.envs.make_dsse_from_scenario import make_dsse_env_from_scenario
from src.models.CNNModel import CNNModel
from src.models.CNNLSTMModel import CNNLSTMModel

from src.models.CNNLSTMModelR3DM import CNNLSTMModelR3DM
from src.policies.R3DMRolePPOTorchPolicy import R3DMRolePPOTorchPolicy  # noqa: F401


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _extract_single_agent_spaces(par_env) -> Tuple[Any, Any]:
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


def _get_cfg_value(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in cfg:
        return cfg[key]
    tr = cfg.get("training", {}) or {}
    if key in tr:
        return tr[key]
    return default


def _get_model_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    m = cfg.get("model", None)
    if isinstance(m, dict):
        return m
    tr = cfg.get("training", {}) or {}
    m = tr.get("model", None)
    return m if isinstance(m, dict) else {}

def build_base_rllib_config(
    *,
    scenario_config: Dict[str, Any],
    train_cfg_json: Dict[str, Any],
    algo_kind: str,  # "cnn" or "lstm" or "lstm_roles"
    env_name: str,
    obs_space=None,
    act_space=None,
) -> PPOConfig:
    debug_mode = bool(train_cfg_json.get("debug", False))

    num_envs = int(train_cfg_json.get("num_envs", 3 if debug_mode else 6))
    num_gpus = int(train_cfg_json.get("num_gpus", 0 if debug_mode else 1))

    train_batch_size = int(train_cfg_json.get("batch_size", 12000))
    lr = float(train_cfg_json.get("learning_rate", 1e-5))
    gamma = float(train_cfg_json.get("gamma", 0.995))
    lam = float(train_cfg_json.get("lambda", 0.95))
    use_gae = bool(train_cfg_json.get("use_gae", True))

    minibatch_size = int(train_cfg_json.get("minibatch_size", 512))
    num_epochs = int(train_cfg_json.get("num_epochs", 15))
    entropy_coeff = float(train_cfg_json.get("entropy_coef", 0.05))
    clip_param = float(train_cfg_json.get("clip_range", 0.1))
    vf_loss_coeff = float(train_cfg_json.get("vf_coef", 1.0))

    rollout_fragment_length = train_cfg_json.get("rollout_fragment_length", "auto")
    seed = train_cfg_json.get("seed", None)

    evaluation_interval = train_cfg_json.get("evaluation_interval", None)
    evaluation_num_episodes = int(train_cfg_json.get("evaluation_num_episodes", 0))
    evaluation_explore = bool(train_cfg_json.get("evaluation_explore", False))

    max_seq_len = int(train_cfg_json.get("max_seq_len", 32))
    lstm_state_size = int(train_cfg_json.get("lstm_state_size", 256))

    fixed_num_roles = int(train_cfg_json.get("num_roles", 3))  
    role_emb_dim = int(train_cfg_json.get("role_emb_dim", 16))
    role_temp = float(train_cfg_json.get("role_temp", 1.0))
    use_gumbel_roles = bool(train_cfg_json.get("use_gumbel_roles", True))
    gumbel_tau = float(train_cfg_json.get("gumbel_tau", 1.0))
    gumbel_hard = bool(train_cfg_json.get("gumbel_hard", False))

    dyn_horizon = int(train_cfg_json.get("dyn_horizon", 5))
    nce_temperature = float(train_cfg_json.get("nce_temperature", 0.2))
    aux_loss_coef = float(train_cfg_json.get("aux_loss_coef", 0.1))
    intrinsic_reward_coef = float(train_cfg_json.get("intrinsic_reward_coef", 0.01))

    role_mi_coef = float(train_cfg_json.get("role_mi_coef", 0.01))

    max_nce_pairs_loss = int(train_cfg_json.get("max_nce_pairs_loss", 1024))
    max_nce_pairs_intrinsic = int(train_cfg_json.get("max_nce_pairs_intrinsic", 512))

    if debug_mode:
        num_gpus = 0
        num_envs = min(num_envs, 3)

    if algo_kind == "cnn":
        model = {"custom_model": "CNNModel", "_disable_preprocessor_api": True}

    elif algo_kind == "lstm":
        model = {
            "custom_model": "CNNLSTMModel",
            "_disable_preprocessor_api": True,
            "max_seq_len": max_seq_len,
            "custom_model_config": {"lstm_state_size": lstm_state_size},
        }

    elif algo_kind == "lstm_roles":
        if obs_space is None or act_space is None:
            raise ValueError("For algo_kind='lstm_roles', obs_space and act_space must be provided.")

        model = {
            "custom_model": "CNNLSTMModelR3DM",
            "_disable_preprocessor_api": True,
            "max_seq_len": max_seq_len,
            "custom_model_config": {
                "lstm_state_size": lstm_state_size,
                "num_roles": fixed_num_roles,
                "role_emb_dim": role_emb_dim,
                "role_temp": role_temp,
                "use_gumbel_roles": use_gumbel_roles,
                "gumbel_tau": gumbel_tau,
                "gumbel_hard": gumbel_hard,
                "dyn_horizon": dyn_horizon,
                "nce_temperature": nce_temperature,
                "aux_loss_coef": aux_loss_coef,
                "intrinsic_reward_coef": intrinsic_reward_coef,
                "role_mi_coef": role_mi_coef,
                "max_nce_pairs_loss": max_nce_pairs_loss,
                "max_nce_pairs_intrinsic": max_nce_pairs_intrinsic,
            },
        }
    else:
        raise ValueError(algo_kind)

    cfg = (
        PPOConfig()
        .environment(env=env_name, env_config=scenario_config)
        .env_runners(
            num_env_runners=num_envs,
            num_envs_per_env_runner=1,
            rollout_fragment_length=rollout_fragment_length,
            batch_mode="complete_episodes",
        )
        .resources(num_gpus=num_gpus, num_gpus_per_worker=0)
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .training(
            train_batch_size=train_batch_size,
            lr=lr,
            gamma=gamma,
            lambda_=lam,
            use_gae=use_gae,
            entropy_coeff=entropy_coeff,
            clip_param=clip_param,
            vf_loss_coeff=vf_loss_coeff,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            model=model,
        )
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework("torch")
    )

    if seed is not None:
        cfg.seed = int(seed)

    if algo_kind == "lstm_roles":
        cfg = cfg.multi_agent(
            policies={
                "default_policy": (R3DMRolePPOTorchPolicy, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "default_policy",
        )
    else:
        cfg = cfg.multi_agent(
            policies={"default_policy": (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "default_policy",
        )

    if evaluation_interval is not None and evaluation_num_episodes > 0:
        cfg = cfg.evaluation(
            evaluation_interval=int(evaluation_interval),
            evaluation_duration=evaluation_num_episodes,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": evaluation_explore},
        )

    return cfg


def rllib_best_to_train_json(
    base_train_cfg: Dict[str, Any],
    best_rllib_cfg: Dict[str, Any],
    algo_kind: str,
) -> Dict[str, Any]:

    out = deepcopy(base_train_cfg)

    out["learning_rate"] = float(_get_cfg_value(best_rllib_cfg, "lr", out.get("learning_rate", 1e-5)))
    out["gamma"] = float(_get_cfg_value(best_rllib_cfg, "gamma", out.get("gamma", 0.995)))

    lam = _get_cfg_value(best_rllib_cfg, "lambda_", None)
    if lam is None:
        lam = _get_cfg_value(best_rllib_cfg, "lambda", out.get("lambda", 0.95))
    out["lambda"] = float(lam)

    out["clip_range"] = float(_get_cfg_value(best_rllib_cfg, "clip_param", out.get("clip_range", 0.1)))
    out["entropy_coef"] = float(_get_cfg_value(best_rllib_cfg, "entropy_coeff", out.get("entropy_coef", 0.0)))
    out["vf_coef"] = float(_get_cfg_value(best_rllib_cfg, "vf_loss_coeff", out.get("vf_coef", 1.0)))

    out["batch_size"] = int(_get_cfg_value(best_rllib_cfg, "train_batch_size", out.get("batch_size", 12000)))
    out["minibatch_size"] = int(_get_cfg_value(best_rllib_cfg, "minibatch_size", out.get("minibatch_size", 512)))
    out["num_epochs"] = int(_get_cfg_value(best_rllib_cfg, "num_epochs", out.get("num_epochs", 10)))

    m = _get_model_dict(best_rllib_cfg)
    cmc = (m.get("custom_model_config", {}) or {})

    if algo_kind in ("lstm", "lstm_roles"):
        out["max_seq_len"] = int(m.get("max_seq_len", out.get("max_seq_len", 32)))
        out["lstm_state_size"] = int(cmc.get("lstm_state_size", out.get("lstm_state_size", 256)))

    if algo_kind == "lstm_roles":
        out["num_roles"] = int(base_train_cfg.get("num_roles", out.get("num_roles", 3)))

        out["role_emb_dim"] = int(cmc.get("role_emb_dim", out.get("role_emb_dim", 16)))
        out["role_temp"] = float(cmc.get("role_temp", out.get("role_temp", 1.0)))
        out["use_gumbel_roles"] = bool(cmc.get("use_gumbel_roles", out.get("use_gumbel_roles", True)))
        out["gumbel_tau"] = float(cmc.get("gumbel_tau", out.get("gumbel_tau", 1.0)))
        out["gumbel_hard"] = bool(cmc.get("gumbel_hard", out.get("gumbel_hard", False)))

        out["dyn_horizon"] = int(cmc.get("dyn_horizon", out.get("dyn_horizon", 5)))
        out["nce_temperature"] = float(cmc.get("nce_temperature", out.get("nce_temperature", 0.2)))
        out["aux_loss_coef"] = float(cmc.get("aux_loss_coef", out.get("aux_loss_coef", 0.1)))
        out["intrinsic_reward_coef"] = float(cmc.get("intrinsic_reward_coef", out.get("intrinsic_reward_coef", 0.01)))
        out["role_mi_coef"] = float(cmc.get("role_mi_coef", out.get("role_mi_coef", 0.01)))

        out["max_nce_pairs_loss"] = int(cmc.get("max_nce_pairs_loss", out.get("max_nce_pairs_loss", 1024)))
        out["max_nce_pairs_intrinsic"] = int(
            cmc.get("max_nce_pairs_intrinsic", out.get("max_nce_pairs_intrinsic", 512))
        )

    out["checkpoint_score_attr"] = "env_runners/episode_return_mean"
    out["checkpoint_score_attr_eval"] = "evaluation/env_runners/episode_return_mean"

    out.setdefault("system_defaults", {})
    out["system_defaults"]["laptop_debug"] = {
        "debug": True,
        "num_envs": 3,
        "num_gpus": 0,
        "max_concurrent_trials": 1,
    }
    out["system_defaults"]["bigger_machine"] = {
        "debug": False,
        "num_envs": 8,
        "num_gpus": 1,
        "max_concurrent_trials": 2,
    }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["cnn", "lstm", "lstm_roles"], required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--train_cfg", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--num_samples", type=int, default=30)
    ap.add_argument("--max_concurrent", type=int, default=1) 
    ap.add_argument("--max_env_steps", type=int, default=20_000_000)
    args = ap.parse_args()

    scenario_config = load_json(Path(args.scenario))
    base_train_cfg = load_json(Path(args.train_cfg))

    outdir = Path(args.outdir).expanduser().resolve()
    storage_path = (outdir / "ray_res" / f"tuning_{args.algo}").resolve()
    storage_path.mkdir(parents=True, exist_ok=True)

    ray.init(
    ignore_reinit_error=True,
    include_dashboard=False,
    )

    env_name = f"DSSE_TUNE_{args.algo.upper()}"

    def rllib_env_creator(env_config):
        env = make_dsse_env_from_scenario(env_config)
        return ParallelPettingZooEnv(env)

    register_env(env_name, rllib_env_creator)

    ModelCatalog.register_custom_model("CNNModel", CNNModel)
    ModelCatalog.register_custom_model("CNNLSTMModel", CNNLSTMModel)
    ModelCatalog.register_custom_model("CNNLSTMModelR3DM", CNNLSTMModelR3DM)

    metric = "evaluation/env_runners/episode_return_mean"
    mode = "max"

    if int(base_train_cfg.get("evaluation_num_episodes", 0)) <= 0 or base_train_cfg.get(
        "evaluation_interval", None
    ) is None:
        raise ValueError(
            "Episode return mean does not exist"
        )

    obs_space = act_space = None
    if args.algo == "lstm_roles":
        tmp = rllib_env_creator(scenario_config)
        par_env = getattr(tmp, "par_env", tmp)
        obs_space, act_space = _extract_single_agent_spaces(par_env)
        tmp.close()

    base_cfg = build_base_rllib_config(
        scenario_config=scenario_config,
        train_cfg_json=base_train_cfg,
        algo_kind=args.algo,
        env_name=env_name,
        obs_space=obs_space,
        act_space=act_space,
    )

    train_batch_choices = []
    mb_div_choices = []  # minibatch_size = train_batch_size / div

    if args.algo == "cnn":
        train_batch_choices = [8192, 16384, 32768, 65536]
        mb_div_choices = [4, 8, 16]

        def sample_minibatch(spec):
            tb = int(spec.config["train_batch_size"])
            div = mb_div_choices[hash(str(spec.config)) % len(mb_div_choices)]
            mb = int(tb // div)
            return max(256, mb)

        base_cfg = base_cfg.training(
            lr=tune.loguniform(5e-6, 1e-4),
            gamma=tune.choice([0.995, 0.999, 0.9999]),
            lambda_=tune.choice([0.90, 0.95, 0.97]),
            clip_param=tune.choice([0.1, 0.2, 0.3]),
            entropy_coeff=tune.choice([0.0, 0.01, 0.05]),
            vf_loss_coeff=tune.choice([0.5, 1.0]),
            train_batch_size=tune.choice(train_batch_choices),
            minibatch_size=tune.sample_from(sample_minibatch),
            num_epochs=tune.choice([10, 15]),
            )

    else:
        lstm_safe_configs = [
            {"train_batch_size": 4096,  "max_seq_len": 16, "lstm_state_size": 64,  "minibatch_size": 256},
            {"train_batch_size": 4096,  "max_seq_len": 32, "lstm_state_size": 128, "minibatch_size": 256},
            {"train_batch_size": 8192,  "max_seq_len": 16, "lstm_state_size": 128, "minibatch_size": 512},
            {"train_batch_size": 8192,  "max_seq_len": 32, "lstm_state_size": 128, "minibatch_size": 512},
            {"train_batch_size": 16384, "max_seq_len": 16, "lstm_state_size": 64,  "minibatch_size": 512},
            {"train_batch_size": 16384, "max_seq_len": 32, "lstm_state_size": 64,  "minibatch_size": 512},
            ]

        def sample_lstm_preset(_spec):
            return lstm_safe_configs[hash(str(_spec.config)) % len(lstm_safe_configs)]

        base_cfg = base_cfg.training(
            lr=tune.loguniform(5e-6, 1e-4),
            gamma=tune.choice([0.995, 0.999, 0.9999]),
            lambda_=tune.choice([0.90, 0.95, 0.97]),
            clip_param=tune.choice([0.1, 0.2, 0.3]),
            entropy_coeff=tune.choice([0.0, 0.01, 0.05]),
            vf_loss_coeff=tune.choice([0.5, 1.0]),
            train_batch_size=tune.sample_from(lambda spec: sample_lstm_preset(spec)["train_batch_size"]),
            minibatch_size=tune.sample_from(lambda spec: sample_lstm_preset(spec)["minibatch_size"]),
            num_epochs=tune.choice([10, 15]),
        )

    if args.algo in ("lstm", "lstm_roles"):
        cur_model = _get_model_dict(base_cfg.to_dict())
        cur_cmc = (cur_model.get("custom_model_config", {}) or {})

        base_cfg = base_cfg.training(
            model={
                **cur_model,
                "max_seq_len": tune.sample_from(lambda spec: sample_lstm_preset(spec)["max_seq_len"]),
                "custom_model_config": {
                    **cur_cmc,
                    "lstm_state_size": tune.sample_from(
                        lambda spec: sample_lstm_preset(spec)["lstm_state_size"]
                    ),
                },
            }
        )


    if args.algo == "lstm_roles":
        fixed_num_roles = int(base_train_cfg.get("num_roles", 3))  

        cur_model = _get_model_dict(base_cfg.to_dict())
        cur_cmc = (cur_model.get("custom_model_config", {}) or {})

        base_cfg = base_cfg.training(
            model={
                **cur_model,
                "custom_model_config": {
                    **cur_cmc,
                    "num_roles": fixed_num_roles,
                    "role_emb_dim": tune.choice([8, 16, 32]),
                    "role_temp": tune.choice([0.7, 1.0, 1.5]),
                    "use_gumbel_roles": tune.choice([True, False]),
                    "gumbel_tau": tune.choice([0.5, 1.0, 2.0]),
                    "gumbel_hard": tune.choice([False]), 
                    "dyn_horizon": tune.choice([3, 5, 8]),
                    "nce_temperature": tune.choice([0.1, 0.2, 0.5]),
                    "aux_loss_coef": tune.choice([0.0, 0.05, 0.1, 0.2]),
                    "intrinsic_reward_coef": tune.choice([0.0, 0.005, 0.01, 0.02]),
                    "role_mi_coef": tune.choice([0.0, 0.005, 0.01, 0.02]),
                    "max_nce_pairs_loss": tune.choice([512, 1024]),
                    "max_nce_pairs_intrinsic": tune.choice([256, 512]),
                },
            }
        )

    param_space = base_cfg.to_dict()

    scheduler = ASHAScheduler(
        time_attr="num_env_steps_sampled",
        grace_period=300_000,
        reduction_factor=2,
        max_t=args.max_env_steps,
    )

    tuner = Tuner(
        "PPO",
        param_space=param_space,
        tune_config=TuneConfig(
            metric=metric,
            mode=mode,
            num_samples=args.num_samples,
            scheduler=scheduler,
            max_concurrent_trials=args.max_concurrent,
        ),
        run_config=RunConfig(
            name=f"PPO_TUNE_{args.algo.upper()}",
            storage_path=str(storage_path),
            stop={"num_env_steps_sampled": args.max_env_steps},
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=int(base_train_cfg.get("checkpoint_freq", 20)),
                num_to_keep=int(base_train_cfg.get("keep_checkpoints_num", 5)),
                checkpoint_score_attribute=metric,
                checkpoint_score_order="max",
                checkpoint_at_end=True,
            ),
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric=metric, mode=mode)

    best_train_cfg = rllib_best_to_train_json(
        base_train_cfg=base_train_cfg,
        best_rllib_cfg=best.config,
        algo_kind=args.algo,
    )

    best_path = outdir / "best_configs" / f"best_train_config_{args.algo}.json"
    save_json(best_path, best_train_cfg)

    print("\n=== TUNING DONE ===")
    print("Best trial:", best.path)

    best_metric = best.metrics.get(metric)

    if best_metric is None:
        best_metric = best.metrics.get("evaluation", {}) \
                                  .get("env_runners", {}) \
                                  .get("episode_return_mean")

    if best_metric is None:
        best_metric = best.metrics.get("env_runners", {}) \
                                  .get("episode_return_mean")

    print("Best evaluation return:", best_metric)
    print("Saved best train config:", str(best_path))

    ray.shutdown()

if __name__ == "__main__":
    main()
