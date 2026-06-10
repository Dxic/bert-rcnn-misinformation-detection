from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss:
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    """

    def __init__(
        self,
        alpha: Optional[Sequence[float]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction harus 'mean', 'sum', atau 'none'.")

        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer(
                "alpha",
                torch.tensor(alpha, dtype=torch.float32),
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        target_probs = probs.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        focal_factor = (1.0 - target_probs).pow(self.gamma)
        loss = -focal_factor * target_log_probs

        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)
            loss = loss * alpha[targets]

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss
