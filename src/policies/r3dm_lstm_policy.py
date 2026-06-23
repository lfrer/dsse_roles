from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Dict as DictSpace, Tuple as TupleSpace

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.annotations import override
from ray.rllib.policy.policy import Policy

from src.policies.r3dm_base_policy import BaseR3DMRolePPOTorchPolicy
from src.policies.r3dm_utils import to_torch_obs

def _tuple_to_dict_space(space: gym.Space) -> gym.Space:
    if isinstance(space, TupleSpace) and len(space.spaces) == 2:
        return DictSpace({
            "positions": space.spaces[0],
            "prob_matrix": space.spaces[1],
        })
    return space


def _convert_tuple_obs_in_batch(batch: SampleBatch) -> None:
    for key in (SampleBatch.OBS, SampleBatch.NEXT_OBS):
        if key not in batch:
            continue
        v = batch[key]

        if isinstance(v, (tuple, list)) and len(v) == 2:
            batch[key] = {
                "positions": np.asarray(v[0]),
                "prob_matrix": np.asarray(v[1]),
            }
        elif (
            isinstance(v, np.ndarray)
            and v.dtype == object
            and len(v) > 0
            and isinstance(v[0], dict)
        ):
            keys = v[0].keys()
            batch[key] = {k: np.stack([d[k] for d in v]) for k in keys}


def _trajectory_length(obs) -> int:
    if isinstance(obs, dict):
        return next(iter(obs.values())).shape[0]
    if isinstance(obs, (tuple, list)):
        return obs[0].shape[0]
    return obs.shape[0]


def _fresh_initial_state(model, device) -> list:
    states = []
    for s in model.get_initial_state():
        if not torch.is_tensor(s):
            s = torch.as_tensor(s, dtype=torch.float32, device=device)
        else:
            s = s.to(device=device, dtype=torch.float32)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        states.append(s)
    return states


class R3DMRolePPOTorchPolicyLSTM(BaseR3DMRolePPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        self._original_obs_space = observation_space
        dict_obs_space = _tuple_to_dict_space(observation_space)
        super().__init__(dict_obs_space, action_space, config)

        self._aux_forward_cache_token = None

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        _convert_tuple_obs_in_batch(sample_batch)
        if other_agent_batches:
            for v in other_agent_batches.values():
                ob = v[-1] if isinstance(v, tuple) else v
                _convert_tuple_obs_in_batch(ob)
                
        need_vf = SampleBatch.VF_PREDS not in sample_batch
        need_logits = SampleBatch.ACTION_DIST_INPUTS not in sample_batch
        need_logp = SampleBatch.ACTION_LOGP not in sample_batch

        if need_vf or need_logits or need_logp:
            T = len(sample_batch[SampleBatch.REWARDS])
            if not self.loss_initialized():
                if need_vf:
                    sample_batch[SampleBatch.VF_PREDS] = np.zeros(T, dtype=np.float32)
                if need_logits:
                    sample_batch[SampleBatch.ACTION_DIST_INPUTS] = np.zeros(
                        (T, self._action_dist_inputs_size()), dtype=np.float32
                    )
                if need_logp:
                    sample_batch[SampleBatch.ACTION_LOGP] = np.zeros(T, dtype=np.float32)
            else:
                logits, vf = self._run_traj_forward_and_cache(sample_batch)
                if need_vf:
                    sample_batch[SampleBatch.VF_PREDS] = vf
                if need_logits:
                    sample_batch[SampleBatch.ACTION_DIST_INPUTS] = logits
                if need_logp:
                    sample_batch[SampleBatch.ACTION_LOGP] = self._compute_action_logp(
                        logits, sample_batch[SampleBatch.ACTIONS]
                    )

        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)


    @torch.no_grad()
    def _run_traj_forward_and_cache(self, sample_batch: SampleBatch):
        model = self.model
        device = next(model.parameters()).device
        obs = to_torch_obs(sample_batch[SampleBatch.OBS], device=device)
        T = _trajectory_length(obs)

        seq_lens = torch.tensor([T], dtype=torch.int32, device=device)
        states = _fresh_initial_state(model, device)

        logits, _ = model({"obs": obs}, states, seq_lens)
        vf = model.value_function()

        self._aux_forward_cache_token = id(sample_batch[SampleBatch.OBS])

        return (
            logits.detach().cpu().numpy().astype(np.float32),
            vf.detach().cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def _forward_model_for_aux(self, model, batch: SampleBatch) -> torch.device:
        device = next(model.parameters()).device

        token = id(batch[SampleBatch.OBS])
        if (
            self._aux_forward_cache_token == token
            and getattr(model, "last_phi", None) is not None
            and getattr(model, "last_role_emb", None) is not None
        ):
            return device

        obs = to_torch_obs(batch[SampleBatch.OBS], device=device)
        T = _trajectory_length(obs)
        seq_lens = torch.tensor([T], dtype=torch.int32, device=device)
        states = _fresh_initial_state(model, device)
        _ = model({"obs": obs}, states, seq_lens)
        return device


    def _action_dist_inputs_size(self) -> int:
        space = self.action_space
        if hasattr(space, "n"):
            return int(space.n)
        return int(self.model.num_outputs)

    @staticmethod
    def _compute_action_logp(logits: np.ndarray, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.int64)
        m = logits.max(axis=-1, keepdims=True)
        log_sum_exp = np.log(np.exp(logits - m).sum(axis=-1, keepdims=True)) + m
        log_probs = logits - log_sum_exp
        rows = np.arange(len(actions))
        return log_probs[rows, actions].astype(np.float32)

    @with_lock
    @override(Policy)
    def compute_gradients(self, postprocessed_batch):
        if not postprocessed_batch.zero_padded:
            self._rebuild_seq_lens_from_eps_id(postprocessed_batch)
        return super().compute_gradients(postprocessed_batch)

    @staticmethod
    def _rebuild_seq_lens_from_eps_id(batch):
        eps_id = np.asarray(batch[SampleBatch.EPS_ID])
        T = len(eps_id)
        unroll_id = (
            np.asarray(batch[SampleBatch.UNROLL_ID])
            if SampleBatch.UNROLL_ID in batch
            else np.zeros(T, dtype=np.int64)
        )
        agent_idx = (
            np.asarray(batch[SampleBatch.AGENT_INDEX])
            if SampleBatch.AGENT_INDEX in batch
            else np.zeros(T, dtype=np.int64)
        )
        max_seq_len = batch.max_seq_len or 32

        same_key = (
            (eps_id[1:] == eps_id[:-1])
            & (unroll_id[1:] == unroll_id[:-1])
            & (agent_idx[1:] == agent_idx[:-1])
        )
        key_change_starts = np.concatenate(([0], np.where(~same_key)[0] + 1))

        seq_start_indices = []
        seq_lens = []
        run_starts = np.concatenate((key_change_starts, [T]))
        for a, b in zip(run_starts[:-1], run_starts[1:]):
            run_len = int(b - a)
            full = run_len // max_seq_len
            for c in range(full):
                seq_start_indices.append(int(a + c * max_seq_len))
                seq_lens.append(max_seq_len)
            tail = run_len - full * max_seq_len
            if tail > 0:
                seq_start_indices.append(int(a + full * max_seq_len))
                seq_lens.append(tail)

        seq_lens = np.asarray(seq_lens, dtype=np.int32)
        assert int(seq_lens.sum()) == T, (
            f"Rebuilt seq_lens sum {int(seq_lens.sum())} != T {T}"
        )

        seq_start_idx = np.asarray(seq_start_indices, dtype=np.int64)
        for state_idx in range(10):
            key = f"state_in_{state_idx}"
            if key not in batch:
                break
            s = np.asarray(batch[key])
            if s.shape[0] == T:
                batch[key] = s[seq_start_idx]
            elif s.shape[0] == len(seq_lens):
                pass  
            else:
                hidden_shape = s.shape[1:]
                batch[key] = np.zeros(
                    (len(seq_lens),) + hidden_shape, dtype=s.dtype
                )

        batch[SampleBatch.SEQ_LENS] = seq_lens
