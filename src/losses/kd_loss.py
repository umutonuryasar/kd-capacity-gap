"""Combined Knowledge Distillation loss.

L_total = α · L_kd + (1 - α) · L_ce

where:
  - L_ce:  standard cross-entropy against ground truth labels
  - L_kd:  distillation loss (LogitKD or FeatureKD)
  - α:     distillation weight in [0, 1]

This formulation follows Hinton et al. (2015): as α increases,
the student relies more on the teacher's soft targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distillation.logit_kd import LogitKDLoss
from ..distillation.feature_kd import FeatureKDLoss


class KDLoss(nn.Module):
    """Unified KD loss: α · L_kd + (1 - α) · L_ce.

    Args:
        kd_type:          'logit', 'feature', or 'none'.
        alpha:            Distillation weight in [0, 1].
                          α=0 → pure cross-entropy (baseline)
                          α=1 → pure distillation
        temperature:      Softmax temperature T (logit KD only).
        feat_beta:        Cosine similarity weight inside FeatureKD.
        student_channels: Per-layer channel dict for the student (feature KD only).
                          None defaults to RESNET_BASIC_CHANNELS (R18 / R34).
        teacher_channels: Per-layer channel dict for the teacher (feature KD only).
                          None defaults to RESNET_BOTTLENECK_CHANNELS (R50).
    """

    def __init__(
        self,
        kd_type: str = "logit",
        alpha: float = 0.5,
        temperature: float = 4.0,
        feat_beta: float = 0.5,
        feat_norm: str = "none",
        student_channels: dict[str, int] | None = None,
        teacher_channels: dict[str, int] | None = None,
    ):
        super().__init__()
        assert kd_type in ("logit", "feature", "none"), \
            f"kd_type must be 'logit', 'feature', or 'none', got '{kd_type}'"
        assert 0.0 <= alpha <= 1.0, f"alpha must be in [0, 1], got {alpha}"

        self.kd_type = kd_type
        self.alpha   = alpha

        if kd_type == "logit":
            self.kd_loss_fn = LogitKDLoss(temperature=temperature)
        elif kd_type == "feature":
            self.kd_loss_fn = FeatureKDLoss(
                student_channels=student_channels,  # None → RESNET_BASIC_CHANNELS default
                teacher_channels=teacher_channels,  # None → RESNET_BOTTLENECK_CHANNELS default
                beta=feat_beta,
                feat_norm=feat_norm,
            )
        else:
            self.kd_loss_fn = None

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_features: dict[str, torch.Tensor] | None = None,
        teacher_features: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute total loss.

        Args:
            student_logits:   [B, C] student class logits.
            teacher_logits:   [B, C] teacher class logits.
            labels:           [B] ground truth class indices.
            student_features: Intermediate feature maps (feature KD only).
            teacher_features: Intermediate feature maps (feature KD only).

        Returns:
            Dict with scalar losses:
              'loss_ce':    cross-entropy loss
              'loss_kd':    distillation loss (0 for baseline)
              'loss_total': α · loss_kd + (1 - α) · loss_ce
        """
        loss_ce = F.cross_entropy(student_logits, labels)

        if self.kd_type == "none" or self.kd_loss_fn is None:
            return {
                "loss_ce":    loss_ce,
                "loss_kd":    torch.tensor(0.0, device=loss_ce.device),
                "loss_total": loss_ce,
            }

        if self.kd_type == "logit":
            loss_kd = self.kd_loss_fn(student_logits, teacher_logits.detach())
            extra   = {}
        else:
            feat_losses = self.kd_loss_fn(student_features, teacher_features)
            loss_kd     = feat_losses["loss_kd"]
            extra       = {
                "loss_mse": feat_losses["loss_mse"],
                "loss_cos": feat_losses["loss_cos"],
            }

        loss_total = self.alpha * loss_kd + (1.0 - self.alpha) * loss_ce

        return {
            "loss_ce":    loss_ce,
            "loss_kd":    loss_kd,
            "loss_total": loss_total,
            **extra,
        }

    def extra_repr(self) -> str:
        return f"kd_type={self.kd_type}, alpha={self.alpha}"
