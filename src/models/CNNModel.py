from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
import torch



class CNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, model_config, name, **kw):
        print("OBSSPACE: ", obs_space)
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, model_config, name, **kw)
        nn.Module.__init__(self)

        flatten_size = 32 * (obs_space[1].shape[0] - 7 - 3) * (obs_space[1].shape[1] - 7 - 3)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(8, 8), stride=(1, 1)),
            nn.Tanh(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(1, 1)),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.Tanh(),
        )

        self.linear = nn.Sequential(
            nn.Linear(obs_space[0].shape[0], 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
        )

        self.join = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.Tanh(),
        )

        self.policy_fn = nn.Linear(256, num_outputs)
        self.value_fn = nn.Linear(256, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        input_positions = input_dict["obs"][0].float()
        input_matrix = input_dict["obs"][1].float()

        input_matrix = input_matrix.unsqueeze(1)  
        cnn_out = self.cnn(input_matrix)
        linear_out = self.linear(input_positions)

        joined = torch.cat((cnn_out, linear_out), dim=1)
        joined = self.join(joined)

        self._value_out = self.value_fn(joined)
        return self.policy_fn(joined), state

    def value_function(self):
        return self._value_out.flatten()
