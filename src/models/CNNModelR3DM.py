from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class CNNModelR3DM(TorchModelV2, nn.Module):

    def __init__(self, obs_space, act_space, num_outputs, model_config, name, **kw):
        num_outputs = act_space.n

        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, model_config, name, **kw)
        nn.Module.__init__(self)

        H, W = obs_space[1].shape
        flatten_size = 32 * (H - 10) * (W - 10)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(8, 8), stride=(1, 1)),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(1, 1)),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.Tanh(),
        )

        self.pos_mlp = nn.Sequential(
            nn.Linear(obs_space[0].shape[0], 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
        )

        self.join = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.Tanh(),
        )

        cmc = model_config.get("custom_model_config", {})

        # ---- Roles ----
        self.num_roles = int(cmc.get("num_roles", 3))
        self.role_emb_dim = int(cmc.get("role_emb_dim", 16))
        self.role_temp = float(cmc.get("role_temp", 1.0))

        self.use_gumbel = bool(cmc.get("use_gumbel_roles", True))
        self.gumbel_tau = float(cmc.get("gumbel_tau", 1.0))
        self.gumbel_tau_start = float(cmc.get("gumbel_tau_start", self.gumbel_tau))
        self.gumbel_tau_end = float(cmc.get("gumbel_tau_end", 0.1))
        self.gumbel_anneal_steps = int(cmc.get("gumbel_anneal_steps", 2_000_000))
        self.gumbel_hard = bool(cmc.get("gumbel_hard", False))

        self.role_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.num_roles),
        )
        self.role_emb = nn.Embedding(self.num_roles, self.role_emb_dim)

        self.policy_fn = nn.Linear(256 + self.role_emb_dim, num_outputs)
        self.value_fn = nn.Linear(256 + self.role_emb_dim, 1)
        self._value_out = None

        self.num_actions = int(num_outputs)
        self.dyn_hidden = int(cmc.get("dyn_hidden", 256))
        self.dyn_latent_dim = 256

        dyn_in_dim = 256 + self.num_actions + self.role_emb_dim
        self.dyn_mlp = nn.Sequential(
            nn.Linear(dyn_in_dim, self.dyn_hidden),
            nn.Tanh(),
            nn.Linear(self.dyn_hidden, self.dyn_latent_dim),
        )

        self.last_phi = None
        self.last_role_logits = None
        self.last_role_probs = None
        self.last_role_sample = None
        self.last_role_emb = None
        self.last_phi_role = None

    def set_gumbel_tau(self, tau: float) -> None:
        self.gumbel_tau = float(max(tau, 1e-6))

    def forward(self, input_dict, state, seq_lens):
        pos_vec = input_dict["obs"][0].float()
        prob_mat = input_dict["obs"][1].float()

        prob_mat = prob_mat.unsqueeze(1)      
        cnn_out = self.cnn(prob_mat)         
        pos_out = self.pos_mlp(pos_vec)     

        phi = self.join(torch.cat([cnn_out, pos_out], dim=1))  
        role_logits = self.role_head(phi)  
        role_probs = torch.softmax(role_logits / max(self.role_temp, 1e-6), dim=-1)

        if self.training and self.use_gumbel:
            z = F.gumbel_softmax(
                role_logits,
                tau=max(self.gumbel_tau, 1e-6),
                hard=self.gumbel_hard,
                dim=-1,
            )
        else:
            #z = role_probs
            z = F.one_hot(role_probs.argmax(dim=-1), num_classes=self.num_roles).float()
        role_emb = z @ self.role_emb.weight
        phi_role = torch.cat([phi, role_emb], dim=1)

        logits = self.policy_fn(phi_role)
        self._value_out = self.value_fn(phi_role)

        self.last_phi = phi
        self.last_role_logits = role_logits
        self.last_role_probs = role_probs
        self.last_role_sample = z
        self.last_role_emb = role_emb
        self.last_phi_role = phi_role

        return logits, state

    def value_function(self):
        return self._value_out.flatten()


    def role_logit_ablation_stats(self):

        if self.last_phi is None or self.last_role_emb is None:
            return {}

        with torch.no_grad():
            phi = self.last_phi
            role_emb = self.last_role_emb

            phi_role_normal = torch.cat([phi, role_emb], dim=1)
            phi_role_zero = torch.cat([phi, torch.zeros_like(role_emb)], dim=1)

            logits_normal = self.policy_fn(phi_role_normal)
            logits_zero = self.policy_fn(phi_role_zero)

            diff = logits_normal - logits_zero

            mean_abs_logit_delta = diff.abs().mean()
            max_abs_logit_delta = diff.abs().max()

            mean_abs_logit_normal = logits_normal.abs().mean()
            mean_abs_logit_zero = logits_zero.abs().mean()

            rel = mean_abs_logit_delta / (mean_abs_logit_normal + 1e-8)

            act_normal = logits_normal.argmax(dim=1)
            act_zero = logits_zero.argmax(dim=1)
            action_same_fraction = (act_normal == act_zero).float().mean()

            return {
                "mean_abs_logit_delta": float(mean_abs_logit_delta.cpu().item()),
                "max_abs_logit_delta": float(max_abs_logit_delta.cpu().item()),
                "mean_abs_logit_normal": float(mean_abs_logit_normal.cpu().item()),
                "mean_abs_logit_zero": float(mean_abs_logit_zero.cpu().item()),
                "relative_logit_delta": float(rel.cpu().item()),
                "action_same_fraction_zero_role": float(action_same_fraction.cpu().item()),
            }

    def predict_future_phi(
        self,
        phi_t: torch.Tensor,
        actions_t: torch.Tensor,
        role_emb_t: torch.Tensor,
    ) -> torch.Tensor:
        a_oh = F.one_hot(actions_t.long(), num_classes=self.num_actions).float()
        dyn_in = torch.cat([phi_t, a_oh, role_emb_t], dim=1)
        return self.dyn_mlp(dyn_in)
