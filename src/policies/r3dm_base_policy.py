from __future__ import annotations

import math
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch

from src.policies.r3dm_utils import valid_future_indices


class BaseR3DMRolePPOTorchPolicy(PPOTorchPolicy):


    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self._last_aux_stats: Dict[str, float] = {}
        self._last_intrinsic_mean: float = 0.0
        self._last_intrinsic_array: np.ndarray | None = None


    def _forward_model_for_aux(self, model, batch: SampleBatch) -> torch.device:
        raise NotImplementedError

    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches=None,
        episode=None,
    ) -> SampleBatch:
        sample_batch = self._add_intrinsic_reward(sample_batch)
        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)

    @torch.no_grad()
    def _add_intrinsic_reward(self, sample_batch: SampleBatch) -> SampleBatch:

        cmc = self.config.get("model", {}).get("custom_model_config", {})
        beta = float(cmc.get("intrinsic_reward_coef", 0.01))
        tau = float(cmc.get("nce_temperature", 0.2))
        horizon = int(cmc.get("dyn_horizon", 5))
        max_pairs = int(cmc.get("max_nce_pairs_intrinsic", 512))

        self._last_intrinsic_mean = 0.0

        rewards = sample_batch[SampleBatch.REWARDS].astype(np.float32, copy=True)
        intrinsic_rewards = np.zeros_like(rewards, dtype=np.float32)

        if beta == 0.0:
            return sample_batch

        valid_idx = valid_future_indices(sample_batch, horizon)
        if len(valid_idx) < 2:
            return sample_batch

        model = self.model
        device = self._forward_model_for_aux(model, sample_batch)

        actions = sample_batch[SampleBatch.ACTIONS]
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, device=device)
        else:
            actions = actions.to(device)

        phi = model.last_phi
        role_emb = model.last_role_emb

        if phi is None or role_emb is None:
            return sample_batch

        idx_t = torch.as_tensor(valid_idx, device=device, dtype=torch.long)
        idx_f = idx_t + horizon

        phi_t = phi[idx_t]
        phi_tpH = phi[idx_f]
        act_t = actions[idx_t]
        remb_t = role_emb[idx_t]

        N = phi_t.shape[0]
        M = min(max_pairs, N)
        if M < 2:
            return sample_batch

        perm = torch.randperm(N, device=device)[:M]
        phi_t = phi_t[perm]
        phi_tpH = phi_tpH[perm]
        act_t = act_t[perm]
        remb_t = remb_t[perm]
        idx_t = idx_t[perm]

        phi_pred = model.predict_future_phi(phi_t, act_t, remb_t)
        sim = (phi_pred @ phi_tpH.T) / max(tau, 1e-6)
        logp_pos = F.log_softmax(sim, dim=1).diag()

        r_int = beta * (logp_pos + math.log(float(M)))
        self._last_intrinsic_mean = float(r_int.mean().detach().cpu().item())

        idx_np = idx_t.detach().cpu().numpy()
        r_np = r_int.detach().cpu().numpy().astype(np.float32)

        for k, r in zip(idx_np, r_np):
            k = int(k)
            if 0 <= k < len(rewards):
                rewards[k] += r
                intrinsic_rewards[k] += r

        sample_batch[SampleBatch.REWARDS] = rewards
        self._last_intrinsic_array = intrinsic_rewards
        return sample_batch

    def loss(self, model, dist_class, train_batch: SampleBatch) -> torch.Tensor:
        base_loss = super().loss(model, dist_class, train_batch)

        cmc = self.config.get("model", {}).get("custom_model_config", {})
        aux_coef = float(cmc.get("aux_loss_coef", 0.1))
        tau = float(cmc.get("nce_temperature", 0.2))
        horizon = int(cmc.get("dyn_horizon", 5))
        max_pairs = int(cmc.get("max_nce_pairs_loss", 1024))
        role_mi_coef = float(cmc.get("role_mi_coef", 0.01))

        if aux_coef == 0.0 and role_mi_coef == 0.0:
            self._last_aux_stats = {}
            return base_loss

        device = self._forward_model_for_aux(model, train_batch)

        actions = train_batch[SampleBatch.ACTIONS]
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, device=device)
        else:
            actions = actions.to(device)

        phi = model.last_phi
        role_probs = model.last_role_probs
        role_emb = model.last_role_emb

        aux_nce = torch.tensor(0.0, device=device)
        mi_proxy = torch.tensor(0.0, device=device)
        role_entropy = torch.tensor(0.0, device=device)
        marginal_role_entropy = torch.tensor(0.0, device=device)
        mean_max_role_prob = torch.tensor(0.0, device=device)

        if role_probs is not None:
            p = role_probs.clamp_min(1e-8)
            p_bar = p.mean(dim=0)
            h_bar = -(p_bar * torch.log(p_bar)).sum()
            h_each = -(p * torch.log(p)).sum(dim=1).mean()

            role_entropy = h_each
            marginal_role_entropy = h_bar
            mean_max_role_prob = p.max(dim=1).values.mean()

            if role_mi_coef != 0.0:
                mi_proxy = h_bar - h_each

        if aux_coef != 0.0 and phi is not None and role_emb is not None:
            valid_idx = valid_future_indices(train_batch, horizon)
            if len(valid_idx) >= 2:
                idx_t = torch.as_tensor(valid_idx, device=device, dtype=torch.long)
                idx_f = idx_t + horizon

                phi_t = phi[idx_t]
                phi_tpH = phi[idx_f]
                act_t = actions[idx_t]
                remb_t = role_emb[idx_t]

                N = phi_t.shape[0]
                M = min(max_pairs, N)
                if M >= 2:
                    perm = torch.randperm(N, device=device)[:M]
                    phi_t = phi_t[perm]
                    phi_tpH = phi_tpH[perm]
                    act_t = act_t[perm]
                    remb_t = remb_t[perm]

                    phi_pred = model.predict_future_phi(phi_t, act_t, remb_t)
                    sim = (phi_pred @ phi_tpH.T) / max(tau, 1e-6)
                    labels = torch.arange(M, device=device)
                    aux_nce = F.cross_entropy(sim, labels)

        total = base_loss
        if aux_coef != 0.0:
            total = total + aux_coef * aux_nce
        if role_mi_coef != 0.0:
            total = total - role_mi_coef * mi_proxy

        self._last_aux_stats = {
            "aux_nce": float(aux_nce.detach().cpu().item()),
            "role_mi_proxy": float(mi_proxy.detach().cpu().item()),
            "role_entropy": float(role_entropy.detach().cpu().item()),
            "marginal_role_entropy": float(marginal_role_entropy.detach().cpu().item()),
            "mean_max_role_prob": float(mean_max_role_prob.detach().cpu().item()),
        }

        return total

    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, Any]:
        out = super().stats_fn(train_batch)

        out.update({f"r3dm/{k}": v for k, v in (self._last_aux_stats or {}).items()})


        intr = getattr(self, "_last_intrinsic_array", None)

        if intr is None and "r3dm_intrinsic_reward" in train_batch:
            intr = train_batch["r3dm_intrinsic_reward"]

        if intr is not None and len(intr) > 0:
            intr_np = np.asarray(intr, dtype=np.float32)
            out["r3dm/intrinsic_mean"] = float(intr_np.mean())
            out["r3dm/intrinsic_abs_mean"] = float(np.abs(intr_np).mean())
            out["r3dm/intrinsic_max"] = float(intr_np.max())
            out["r3dm/intrinsic_min"] = float(intr_np.min())
        else:
            out["r3dm/intrinsic_mean"] = 0.0
            out["r3dm/intrinsic_abs_mean"] = 0.0
            out["r3dm/intrinsic_max"] = 0.0
            out["r3dm/intrinsic_min"] = 0.0

        if hasattr(self.model, "gumbel_tau"):
            out["r3dm/gumbel_tau"] = float(self.model.gumbel_tau)

        if hasattr(self.model, "role_logit_ablation_stats"):
            try:
                abl = self.model.role_logit_ablation_stats()
                for k, v in abl.items():
                    out[f"r3dm/role_ablation_{k}"] = v
            except Exception:
                pass

        return out
