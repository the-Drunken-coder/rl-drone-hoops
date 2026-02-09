"""Neural network models for recurrent actor-critic policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_drone_hoops.rl.distributions import SquashedDiagGaussian


class SmallCNN(nn.Module):
    """Lightweight CNN for FPV image encoding.

    Architecture:
        Conv2d(1->32, k=8, s=4) -> ReLU
        Conv2d(32->64, k=4, s=2) -> ReLU
        Conv2d(64->64, k=3, s=1) -> ReLU
        AdaptiveAvgPool2d -> Linear -> ReLU

    Makes the encoder resolution-agnostic by pooling to a fixed spatial size.
    """

    def __init__(self, in_ch: int = 1, feat_dim: int = 256, pool_hw: int = 7) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Make the encoder resolution-agnostic by pooling to a fixed spatial size.
        self.pool = nn.AdaptiveAvgPool2d((int(pool_hw), int(pool_hw)))
        self.fc = nn.Linear(64 * int(pool_hw) * int(pool_hw), feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode FPV image to feature vector.

        Args:
            x: Image tensor (B, 1, H, W) with values in [0, 1]

        Returns:
            Feature vector (B, feat_dim)
        """
        z = self.conv(x)
        z = self.pool(z)
        z = z.flatten(1)
        return F.relu(self.fc(z))


class IMUEncoder(nn.Module):
    """MLP encoder for IMU sensor history.

    Flattens and encodes a sequence of IMU readings (gyro + accel).
    """

    def __init__(self, in_dim: int, feat_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feat_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ActorCriticOutput:
    action: torch.Tensor  # (N,A) in [-1,1]
    logp: torch.Tensor  # (N,)
    value: torch.Tensor  # (N,)
    h: torch.Tensor  # (1,N,H)
    entropy: torch.Tensor  # (N,)


class RecurrentActorCritic(nn.Module):
    """Recurrent actor-critic policy for drone control.

    Architecture:
        - Image encoder: CNN
        - IMU encoder: MLP
        - Fusion: Concatenate encodings with last action, then MLP
        - Memory: GRU for temporal dependency
        - Heads: Policy (tanh-squashed Gaussian) and value function

    Attributes:
        log_std: Learned action standard deviation
    """

    def __init__(
        self,
        *,
        image_size: int,
        imu_window_n: int,
        action_dim: int = 4,
        cnn_dim: int = 256,
        imu_dim: int = 128,
        fused_dim: int = 256,
        rnn_hidden: int = 256,
    ) -> None:
        """Initialize the recurrent actor-critic model.

        Args:
            image_size: FPV image resolution (square)
            imu_window_n: Number of IMU samples in history window
            action_dim: Dimension of action space
            cnn_dim: CNN feature dimension
            imu_dim: IMU encoder feature dimension
            fused_dim: Fused feature dimension before RNN
            rnn_hidden: GRU hidden state dimension
        """
        super().__init__()
        self.image_size = int(image_size)
        self.imu_window_n = int(imu_window_n)
        self.action_dim = int(action_dim)

        self.cnn = SmallCNN(in_ch=1, feat_dim=cnn_dim)
        self.imu_enc = IMUEncoder(in_dim=imu_window_n * 6, feat_dim=imu_dim)

        self.fuse = nn.Sequential(
            nn.Linear(cnn_dim + imu_dim + action_dim, fused_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=fused_dim, hidden_size=rnn_hidden, num_layers=1)

        self.pi = nn.Linear(rnn_hidden, action_dim)
        self.v = nn.Linear(rnn_hidden, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5, dtype=torch.float32))

    def initial_hidden(self, n_envs: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((1, n_envs, self.gru.hidden_size), device=device, dtype=torch.float32)

    def _encode_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # obs tensors are (B, ...)
        img = obs["image"]  # uint8 (B,H,W,1) or float
        if img.dtype != torch.float32:
            img = img.float()
        img = img / 255.0
        img = img.permute(0, 3, 1, 2).contiguous()  # (B,1,H,W)

        imu = obs["imu"].float()  # (B,T,6)
        imu = imu.flatten(1)  # (B,T*6)

        last_a = obs["last_action"].float()  # (B,4)

        z_img = self.cnn(img)
        z_imu = self.imu_enc(imu)
        z = torch.cat([z_img, z_imu, last_a], dim=-1)
        return self.fuse(z)

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor], h: torch.Tensor, deterministic: bool = False) -> ActorCriticOutput:
        x = self._encode_obs(obs)  # (N,F)
        x_seq = x.unsqueeze(0)  # (1,N,F)
        y, h2 = self.gru(x_seq, h)
        y0 = y.squeeze(0)

        mean = self.pi(y0)
        log_std = self.log_std.expand_as(mean).clamp(-5.0, 2.0)
        dist = SquashedDiagGaussian(mean=mean, log_std=log_std)
        if deterministic:
            a, logp, _u = dist.mode()
        else:
            a, logp, _u = dist.sample()
        v = self.v(y0).squeeze(-1)
        ent = dist.entropy_approx()
        return ActorCriticOutput(action=a, logp=logp, value=v, h=h2, entropy=ent)

    def forward_sequence(
        self, obs_seq: Dict[str, torch.Tensor], h0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward a full rollout sequence.

        Inputs:
        - obs_seq: tensors shaped (T,N,...) for each key
        - h0: (1,N,H)

        Returns:
        - mean: (T,N,A)
        - value: (T,N)
        - hT: (1,N,H)
        """
        T, N = obs_seq["last_action"].shape[0], obs_seq["last_action"].shape[1]

        # Flatten T*N for encoders.
        flat = {
            "image": obs_seq["image"].reshape(T * N, *obs_seq["image"].shape[2:]),
            "imu": obs_seq["imu"].reshape(T * N, *obs_seq["imu"].shape[2:]),
            "last_action": obs_seq["last_action"].reshape(T * N, *obs_seq["last_action"].shape[2:]),
        }
        x = self._encode_obs(flat).reshape(T, N, -1)  # (T,N,F)
        y, hT = self.gru(x, h0)
        mean = self.pi(y)
        value = self.v(y).squeeze(-1)
        return mean, value, hT

    def forward_sequence_masked(
        self, obs_seq: Dict[str, torch.Tensor], done_seq: torch.Tensor, h0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward a rollout sequence while resetting hidden state after done steps.

        done_seq: (T,N) where True means the env finished after that step (i.e. before next obs).
        """
        T, N = done_seq.shape

        flat = {
            "image": obs_seq["image"].reshape(T * N, *obs_seq["image"].shape[2:]),
            "imu": obs_seq["imu"].reshape(T * N, *obs_seq["imu"].shape[2:]),
            "last_action": obs_seq["last_action"].reshape(T * N, *obs_seq["last_action"].shape[2:]),
        }
        x = self._encode_obs(flat).reshape(T, N, -1)  # (T,N,F)

        means = []
        values = []
        h = h0
        for t in range(T):
            y, h = self.gru(x[t : t + 1], h)  # y: (1,N,H)
            yt = y.squeeze(0)
            means.append(self.pi(yt))
            values.append(self.v(yt).squeeze(-1))

            # Reset hidden for envs that terminated/truncated after this step.
            mask = (1.0 - done_seq[t].float()).view(1, N, 1)
            h = h * mask

        mean = torch.stack(means, dim=0)
        value = torch.stack(values, dim=0)
        return mean, value, h
