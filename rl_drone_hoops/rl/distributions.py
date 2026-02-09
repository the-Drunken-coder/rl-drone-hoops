from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class SquashedDiagGaussian:
    """
    Tanh-squashed diagonal Gaussian distribution.

    We represent the pre-squash action u ~ N(mean, std) and a = tanh(u).
    Log-prob includes the change-of-variables correction.
    """

    mean: torch.Tensor  # (..., A)
    log_std: torch.Tensor  # (..., A)
    eps: float = 1e-6

    @property
    def std(self) -> torch.Tensor:
        return torch.exp(self.log_std)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.mean + self.std * torch.randn_like(self.mean)
        a = torch.tanh(u)
        logp = self.log_prob(u, a)
        return a, logp, u

    def mode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.mean
        a = torch.tanh(u)
        logp = self.log_prob(u, a)
        return a, logp, u

    def log_prob(self, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # Base log prob of u under Normal(mean, std).
        var = self.std * self.std
        logp_u = -0.5 * (((u - self.mean) ** 2) / (var + self.eps) + 2.0 * self.log_std + math.log(2.0 * math.pi))
        logp_u = logp_u.sum(dim=-1)

        # Tanh correction: log |det da/du| = sum log(1 - tanh(u)^2)
        # Use a (already tanh(u)) for numerical stability.
        corr = torch.log(1.0 - a * a + self.eps).sum(dim=-1)
        return logp_u - corr

    def entropy_approx(self) -> torch.Tensor:
        # Approximate with Gaussian entropy (common and stable for PPO logging).
        # Exact tanh-squashed entropy isn't needed for the PPO objective.
        ent = 0.5 + 0.5 * math.log(2.0 * math.pi) + self.log_std
        return ent.sum(dim=-1)

