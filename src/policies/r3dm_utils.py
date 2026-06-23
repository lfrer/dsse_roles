from __future__ import annotations

from typing import List

import numpy as np
import torch

from ray.rllib.policy.sample_batch import SampleBatch


def to_torch_obs(obs, device: torch.device):

    if isinstance(obs, dict):
        return {
            k: (v.to(device) if torch.is_tensor(v) else torch.as_tensor(v, device=device))
            for k, v in obs.items()
        }

    if isinstance(obs, (tuple, list)) and len(obs) == 2:
        return (
            torch.as_tensor(obs[0], device=device),
            torch.as_tensor(obs[1], device=device),
        )

    if isinstance(obs, np.ndarray) and obs.dtype == object:
        pos_list, prob_list = zip(*obs.tolist())
        pos = torch.as_tensor(np.stack(pos_list, axis=0), device=device)
        prob = torch.as_tensor(np.stack(prob_list, axis=0), device=device)
        return (pos, prob)

    raise TypeError(f"Unsupported obs structure for tuple obs: {type(obs)}")


def valid_future_indices(train_batch: SampleBatch, horizon: int) -> np.ndarray:
    actions = train_batch[SampleBatch.ACTIONS]
    T = int(actions.shape[0])
    if T <= horizon:
        return np.zeros((0,), dtype=np.int64)

    eps_id = train_batch.get(SampleBatch.EPS_ID, None)
    terminateds = train_batch.get(SampleBatch.TERMINATEDS, None)
    truncateds = train_batch.get(SampleBatch.TRUNCATEDS, None)

    def _scalar(x):
        if torch.is_tensor(x):
            return x.item()
        return x

    def _any_true(x):
        if x is None:
            return False
        if torch.is_tensor(x):
            return bool(torch.any(x).item())
        return bool(np.any(np.asarray(x, dtype=np.bool_)))

    valid = []
    for i in range(T - horizon):
        j = i + horizon

        if eps_id is not None:
            if _scalar(eps_id[i]) != _scalar(eps_id[j]):
                continue

        crossed_boundary = False
        if terminateds is not None and _any_true(terminateds[i:j]):
            crossed_boundary = True
        if truncateds is not None and _any_true(truncateds[i:j]):
            crossed_boundary = True

        if not crossed_boundary:
            valid.append(i)

    return np.asarray(valid, dtype=np.int64)


def get_recurrent_state_inputs(train_batch: SampleBatch, device: torch.device) -> List[torch.Tensor]:

    states = []
    i = 0
    while f"state_in_{i}" in train_batch:
        s = train_batch[f"state_in_{i}"]
        if not torch.is_tensor(s):
            s = torch.as_tensor(s, device=device)
        else:
            s = s.to(device)
        states.append(s)
        i += 1
    return states


def get_seq_lens(train_batch: SampleBatch, device: torch.device):
    seq_lens = train_batch.get(SampleBatch.SEQ_LENS, None)
    if seq_lens is None:
        return None

    if not torch.is_tensor(seq_lens):
        seq_lens = torch.as_tensor(seq_lens, device=device, dtype=torch.long)
    else:
        seq_lens = seq_lens.to(device=device, dtype=torch.long)

    return seq_lens
