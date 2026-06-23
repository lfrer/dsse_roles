from __future__ import annotations

import torch

from ray.rllib.policy.sample_batch import SampleBatch

from src.policies.r3dm_base_policy import BaseR3DMRolePPOTorchPolicy
from src.policies.r3dm_utils import to_torch_obs


class R3DMRolePPOTorchPolicyCNN(BaseR3DMRolePPOTorchPolicy):
    @torch.no_grad()
    def _forward_model_for_aux(self, model, batch: SampleBatch) -> torch.device:
        device = next(model.parameters()).device
        obs = to_torch_obs(batch[SampleBatch.OBS], device=device)
        _ = model({"obs": obs}, [], None)
        return device
