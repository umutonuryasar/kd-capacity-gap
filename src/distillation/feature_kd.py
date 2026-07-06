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


# Channel widths per ResNet family.
# R18 and R34 use BasicBlock (no expansion), so they share the same widths.
# R50+ use Bottleneck (expansion=4), yielding 4× wider feature maps.
RESNET_BASIC_CHANNELS: dict[str, int] = {
    "layer1": 64,
    "layer2": 128,
    "layer3": 256,
    "layer4": 512,
}
RESNET_BOTTLENECK_CHANNELS: dict[str, int] = {
    "layer1": 256,
    "layer2": 512,
    "layer3": 1024,
    "layer4": 2048,
}


class FeatureKDLoss(nn.Module):
    """Multi-layer feature alignment KD loss.

    Args:
        student_channels: Dict of layer -> channel count for the student.
                          Defaults to RESNET_BASIC_CHANNELS (R18 / R34).
        teacher_channels: Dict of layer -> channel count for the teacher.
                          Defaults to RESNET_BOTTLENECK_CHANNELS (R50).
        beta:             Weight for the cosine similarity term.
        layers:           Which ResNet stages to align.
    """

    def __init__(
        self,
        student_channels: dict[str, int] | None = None,
        teacher_channels: dict[str, int] | None = None,
        beta: float = 0.5,
        layers: list[str] | None = None,
        feat_norm: str = "none",
    ):
        super().__init__()
        assert feat_norm in ("none", "teacher_std"), \
            f"feat_norm must be 'none' or 'teacher_std', got '{feat_norm}'"
        self.beta      = beta
        self.feat_norm = feat_norm
        self.layers    = layers or ["layer1", "layer2", "layer3", "layer4"]

        s_ch_map = student_channels or RESNET_BASIC_CHANNELS
        t_ch_map = teacher_channels or RESNET_BOTTLENECK_CHANNELS

        # 1×1 Conv projections: student_dim → teacher_dim per layer
        self.projections = nn.ModuleDict()
        for layer in self.layers:
            s_ch = s_ch_map[layer]
            t_ch = t_ch_map[layer]
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
        mse_losses = []
        cos_losses = []

        for layer in self.layers:
            s = student_features[layer]  # [B, C_s, H, W]
            t = teacher_features[layer]  # [B, C_t, H, W]

            # Project student to teacher channel dim. The module lives on the
            # correct device already (Trainer moves loss_fn to device).
            s_proj = self.projections[layer](s)  # [B, C_t, H, W]

            # Align spatial dims if needed
            if s_proj.shape[-2:] != t.shape[-2:]:
                s_proj = F.adaptive_avg_pool2d(s_proj, t.shape[-2:])

            t = t.detach()

            # Optional scale normalization: raw MSE depends on the magnitude
            # of teacher features, which grows with teacher depth and differs
            # per layer. 'teacher_std' divides both sides by the teacher's
            # per-layer std (detached), making the loss scale-invariant and
            # the alpha grid comparable across teacher architectures.
            if self.feat_norm == "teacher_std":
                scale  = t.std().clamp_min(1e-6)
                s_proj = s_proj / scale
                t      = t / scale

            # MSE loss
            mse_losses.append(F.mse_loss(s_proj, t))

            # Cosine similarity on globally pooled features
            s_gap = s_proj.mean(dim=[2, 3])   # [B, C_t]
            t_gap = t.mean(dim=[2, 3])        # [B, C_t]
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
        return f"layers={self.layers}, beta={self.beta}, feat_norm={self.feat_norm}"
