import numpy as np
from ray.rllib.models import ModelCatalog

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer, same_padding, \
    SlimConv2d, SlimFC
from ray.rllib.models.utils import get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()



class VisionNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        # MUST be named "cnn_base" to load weights from VAE pretraining
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(obs_space.shape[0], 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)

        output_nodes = 4
        self._logits = nn.Sequential(
            nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, output_nodes)

        )
        self._value_branch = nn.Sequential(
            nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1)
        )

        print("LOADING PRETRAINED WEIGHTS")

        self.load_cnn_base_state_dict("/home/twiggers/xtma_racecar_2/pytorch_car_caring/ppo_net_params_zigzag.pkl")
        print()

    def load_cnn_base_state_dict(self, path):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        state_dict = torch.load(path, map_location=device)
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                # print(name, "NOT IN", own_state.keys())
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            print("loaded weigths for", name)
            own_state[name].copy_(param)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):

        x = input_dict["obs"]

        self._features = self.cnn_base(x).view(-1, 256)

        logits = self._logits(self._features)


        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self._value_branch(self._features).squeeze(1)


