"""Logit-level Knowledge Distillation loss.

L_logit = T² * KL( softmax(t_logits / T) || softmax(s_logits / T) )

Minimizing this KL divergence is equivalent to MLE under the teacher's
distribution — a direct connection to CS229 PS3 Q2c.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitKDLoss(nn.Module):
    """Temperature-scaled KL divergence on classification logits.

    Args:
        temperature: Softmax temperature T. Higher values produce softer
                     distributions. Typical values: {2, 4, 8}.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.T = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean", log_target=False)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: [B, C] raw student logits.
            teacher_logits: [B, C] raw teacher logits (detached).

        Returns:
            Scalar KD loss.
        """
        s_log_prob = F.log_softmax(student_logits / self.T, dim=-1)
        t_prob     = F.softmax(teacher_logits / self.T, dim=-1)
        return self.kl_div(s_log_prob, t_prob) * (self.T ** 2)

    def extra_repr(self) -> str:
        return f"temperature={self.T}"