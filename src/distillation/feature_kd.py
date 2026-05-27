"""Feature-level Knowledge Distillation loss.

Aligns intermediate feature maps at four ResNet stages:
  - layer1, layer2: early features (edges, textures)
  - layer3:         mid-level features (parts)
  - layer4:         high-level semantic features

Two loss components per layer pair:
  1. MSE loss on projected student features (L_mse)
  2. Cosine similarity loss on globally pooled features (L_cos)

Combined:
  L_feat = mean(L_mse across layers) + beta * mean(L_cos across layers)
  L_kd   = L_feat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ResNet-18 and ResNet-50 output channels per stage
STUDENT_CHANNELS = {
    "layer1": 64,
    "layer2": 128,
    "layer3": 256,
    "layer4": 512,
}
TEACHER_CHANNELS = {
    "layer1": 256,
    "layer2": 512,
    "layer3": 1024,
    "layer4": 2048,
}


class FeatureKDLoss(nn.Module):
    """Multi-layer feature alignment KD loss.

    Args:
        student_channels: Dict of layer -> channel count for student.
        teacher_channels: Dict of layer -> channel count for teacher.
        beta:             Weight for cosine similarity term.
        layers:           Which ResNet stages to align.
    """

    def __init__(
        self,
        student_channels: int = 512,   # kept for API compat, unused
        teacher_channels: int = 2048,  # kept for API compat, unused
        beta: float = 0.5,
        layers: list[str] | None = None,
    ):
        super().__init__()
        self.beta   = beta
        self.layers = layers or ["layer1", "layer2", "layer3", "layer4"]

        # 1x1 Conv projections: student_dim -> teacher_dim per layer
        self.projections = nn.ModuleDict()
        for layer in self.layers:
            s_ch = STUDENT_CHANNELS[layer]
            t_ch = TEACHER_CHANNELS[layer]
            proj = nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False)
            nn.init.xavier_uniform_(proj.weight)
            self.projections[layer] = proj

    def forward(
        self,
        student_features: dict[str, torch.Tensor],
        teacher_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute multi-layer feature KD loss.

        Args:
            student_features: Dict layer -> [B, C_s, H, W].
            teacher_features: Dict layer -> [B, C_t, H, W].

        Returns:
            Dict with scalar losses:
              'loss_mse': averaged MSE across layers.
              'loss_cos': averaged cosine loss across layers.
              'loss_kd':  loss_mse + beta * loss_cos.
        """
        device = next(iter(student_features.values())).device
        dtype  = next(iter(student_features.values())).dtype

        mse_losses = []
        cos_losses = []

        for layer in self.layers:
            s = student_features[layer]  # [B, C_s, H, W]
            t = teacher_features[layer]  # [B, C_t, H, W]

            # Project student to teacher channel dim
            proj_weight = self.projections[layer].weight.to(device=device, dtype=dtype)
            s_proj = F.conv2d(s, proj_weight)  # [B, C_t, H, W]

            # Align spatial dims if needed
            if s_proj.shape[-2:] != t.shape[-2:]:
                s_proj = F.adaptive_avg_pool2d(s_proj, t.shape[-2:])

            # MSE loss
            mse_losses.append(F.mse_loss(s_proj, t.detach()))

            # Cosine similarity on globally pooled features
            s_gap = s_proj.mean(dim=[2, 3])       # [B, C_t]
            t_gap = t.detach().mean(dim=[2, 3])   # [B, C_t]
            cos_sim = F.cosine_similarity(s_gap, t_gap, dim=-1)
            cos_losses.append((1.0 - cos_sim).mean())

        loss_mse = torch.stack(mse_losses).mean()
        loss_cos = torch.stack(cos_losses).mean()
        loss_kd  = loss_mse + self.beta * loss_cos

        return {
            "loss_mse": loss_mse,
            "loss_cos": loss_cos,
            "loss_kd":  loss_kd,
        }

    def extra_repr(self) -> str:
        return f"layers={self.layers}, beta={self.beta}"
