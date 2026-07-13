import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class RND(nn.Module):
    def __init__(self, obs_shape, action_shape, device, lr=1e-4):
        super(RND, self).__init__()

        self.input_size = obs_shape
        self.output_size = action_shape
        obs_channels = obs_shape[0]

        # 2026-05-18: compute feature_output dynamically based on obs spatial
        # size (was hardcoded to 7*7*64=3136 which only works at 64x64 input).
        # Conv chain: (k=6,s=3) -> (k=4,s=2) -> (k=3,s=1), all stride no pad.
        def _co(in_size, k, s):
            return (in_size - k) // s + 1
        _H = obs_shape[1]
        _h1 = _co(_H, 6, 3); _h2 = _co(_h1, 4, 2); _h3 = _co(_h2, 3, 1)
        feature_output = _h3 * _h3 * 64
        self.predictor = nn.Sequential(
            nn.Conv2d(
                in_channels=obs_channels,
                out_channels=32,
                kernel_size=6,
                stride=3),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.target = nn.Sequential(
            nn.Conv2d(
                in_channels=obs_channels,
                out_channels=32,
                kernel_size=6,
                stride=3),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device

    def forward(self, next_obs):
        next_obs = next_obs / 255.0 - 0.5

        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)
        return predict_feature, target_feature

    @torch.no_grad()
    def calc_reward(self, obs, next_obs):
        predict_next_feature, target_next_feature = self(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
        return intrinsic_reward.unsqueeze(1)

    def update(self, batch):
        metrics = dict()
        
        _, _, _, _, next_obs = batch
        next_obs = torch.as_tensor(next_obs, device=self.device)

        predict_next_feature, target_next_feature = self(next_obs)
        loss = F.mse_loss(predict_next_feature, target_next_feature)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()

        metrics['rnd_loss'] = loss.item()
        return metrics
