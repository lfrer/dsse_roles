from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from ray.rllib.models import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.policy.rnn_sequencing import add_time_dimension


class CNNLSTMModelR3DM(TorchRNN, nn.Module):

    def __init__(self, obs_space, act_space, num_outputs, model_config, name, **kw):
        num_outputs = act_space.n

        nn.Module.__init__(self)
        super().__init__(obs_space, act_space, num_outputs, model_config, name, **kw)

        space = getattr(obs_space, "original_space", obs_space)

        if hasattr(space, "spaces") and isinstance(space.spaces, dict):
            pos_space = space["positions"]
            prob_space = space["prob_matrix"]
        elif hasattr(space, "spaces") and isinstance(space.spaces, (list, tuple)):
            pos_space = space.spaces[0]
            prob_space = space.spaces[1]
        else:
            try:
                pos_space = space["positions"]
                prob_space = space["prob_matrix"]
            except (TypeError, KeyError):
                pos_space = space[0]
                prob_space = space[1]

     
        N = pos_space.shape[0]
        H, W = prob_space.shape

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

        self.linear = nn.Linear(N, 512)

        cmc = model_config.get("custom_model_config", {})

        self.lstm_state_size = int(cmc.get("lstm_state_size", 256))
        self.lstm = nn.LSTM(512, self.lstm_state_size, batch_first=True)

        self.join = nn.Sequential(
            nn.Linear(256 + self.lstm_state_size, 256),
            nn.Tanh(),
        )

        # ---- Roles ----
        self.num_roles = int(cmc.get("num_roles", 3))
        self.role_emb_dim = int(cmc.get("role_emb_dim", 16))
        self.role_temp = float(cmc.get("role_temp", 1.0))

        self.use_gumbel = bool(cmc.get("use_gumbel_roles", True))
        self.gumbel_tau = float(cmc.get("gumbel_tau", 1.0))
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
        self.last_role_emb = None
        self.last_phi_role = None


    @override(ModelV2)
    def get_initial_state(self) -> List[torch.Tensor]:
        h = self.linear.weight.new_zeros(self.lstm_state_size)
        c = self.linear.weight.new_zeros(self.lstm_state_size)
        return [h, c]

    @override(ModelV2)
    def value_function(self) -> torch.Tensor:
        return self._value_out.flatten()

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        if isinstance(obs, dict) and "positions" in obs:
            pos_vec = obs["positions"]
            prob_mat = obs["prob_matrix"]

        elif isinstance(obs, (tuple, list)) and len(obs) == 2:
            pos_vec, prob_mat = obs[0], obs[1]

        elif isinstance(obs, np.ndarray) and obs.dtype == object:
            pos_vec = np.stack([d["positions"] for d in obs])
            prob_mat = np.stack([d["prob_matrix"] for d in obs])

        else:
            raise TypeError(
                f"Unexpected obs type/structure in CNNLSTMModelR3DM.forward: "
                f"type={type(obs).__name__} dtype={getattr(obs, 'dtype', None)}"
            )

        device = next(self.parameters()).device
        if not torch.is_tensor(pos_vec):
            pos_vec = torch.as_tensor(pos_vec, device=device)
        if not torch.is_tensor(prob_mat):
            prob_mat = torch.as_tensor(prob_mat, device=device)
        pos_vec = pos_vec.float()
        prob_mat = prob_mat.float() 


        prob_mat = prob_mat.unsqueeze(1)  
        cnn_out = self.cnn(prob_mat)     

        flat_inputs = pos_vec.flatten(start_dim=1)
        time_major = self.model_config.get("_time_major", False)
        inputs_time = add_time_dimension(
            flat_inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=time_major,
        )

        lstm_out, new_state = self.forward_rnn(inputs_time, state, seq_lens)
        lstm_out = torch.reshape(lstm_out, [-1, self.lstm_state_size])  

        phi = torch.cat([cnn_out, lstm_out], dim=1)
        phi = self.join(phi) 

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
            z = role_probs

        role_emb = z @ self.role_emb.weight 
        phi_role = torch.cat([phi, role_emb], dim=1)

        logits = self.policy_fn(phi_role)
        self._value_out = self.value_fn(phi_role)

        self.last_phi = phi
        self.last_role_logits = role_logits
        self.last_role_probs = role_probs
        self.last_role_emb = role_emb
        self.last_phi_role = phi_role

        return logits, new_state

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        x = torch.tanh(self.linear(inputs)) 
        h0 = state[0].unsqueeze(0)           
        c0 = state[1].unsqueeze(0)
        out, (h, c) = self.lstm(x, (h0, c0))
        return out, [h.squeeze(0), c.squeeze(0)]


    def predict_future_phi(self, phi_t: torch.Tensor, actions_t: torch.Tensor, role_emb_t: torch.Tensor) -> torch.Tensor:
        a_oh = F.one_hot(actions_t.long(), num_classes=self.num_actions).float()
        dyn_in = torch.cat([phi_t, a_oh, role_emb_t], dim=1)
        return self.dyn_mlp(dyn_in)  
