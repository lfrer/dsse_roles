from ray.rllib.models import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from torch import nn
import torch
from ray.rllib.utils.annotations import override
from ray.rllib.policy.rnn_sequencing import add_time_dimension



class CNNLSTMModel(TorchRNN, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, model_config, name, **kw):
        num_outputs = act_space.n

        nn.Module.__init__(self)
        super().__init__(obs_space, act_space, num_outputs, model_config, name, **kw)

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

        self.linear = nn.Linear(obs_space[0].shape[0], 512)

        cmc = model_config.get("custom_model_config", {})
        self.lstm_state_size = int(cmc.get("lstm_state_size", 256))
        self.lstm = nn.LSTM(512, self.lstm_state_size, batch_first=True)

        self.join = nn.Sequential(
            nn.Linear(256 + self.lstm_state_size, 256),
            nn.Tanh(),
        )

        self.policy_fn = nn.Linear(256, num_outputs)
        self.value_fn = nn.Linear(256, 1)
        self._value_out = None

    @override(ModelV2)
    def get_initial_state(self):
        h = self.linear.weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        c = self.linear.weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        return [h, c]

    def value_function(self):
        return self._value_out.flatten()

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        pos_vec = input_dict["obs"][0].float()
        prob_mat = input_dict["obs"][1].float()

        prob_mat = prob_mat.unsqueeze(1)  # [B,1,H,W]
        cnn_out = self.cnn(prob_mat)       # [B,256]

        flat_inputs = pos_vec.flatten(start_dim=1)
        time_major = self.model_config.get("_time_major", False)
        inputs_time = add_time_dimension(
            flat_inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=time_major,
        )

        lstm_out, new_state = self.forward_rnn(inputs_time, state, seq_lens)
        lstm_out = torch.reshape(lstm_out, [-1, self.lstm_state_size])  # [B, lstm_state]

        joined = torch.cat([cnn_out, lstm_out], dim=1)
        joined = self.join(joined)

        self._value_out = self.value_fn(joined)
        return self.policy_fn(joined), new_state

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        x = torch.tanh(self.linear(inputs))  # [B,T,512]
        h0 = torch.unsqueeze(state[0], 0)
        c0 = torch.unsqueeze(state[1], 0)
        out, (h, c) = self.lstm(x, (h0, c0))
        return out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

