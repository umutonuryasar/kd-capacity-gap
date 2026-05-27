"""CIFAR-10 ResNet with forward hooks for Knowledge Distillation.

Critical modification: replaces the ImageNet-oriented 7×7 conv (stride=2)
and MaxPool with a 3×3 conv (stride=1) and Identity. This prevents aggressive
spatial downsampling of 32×32 CIFAR images.

Feature hooks on layer1–layer4 expose intermediate activations for Feature-KD.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNet(nn.Module):
    def __init__(
        self,
        variant: str = "resnet18",
        num_classes: int = 10,
        pretrained: bool = False,
    ):
        super().__init__()

        if variant == "resnet18":
            base = tv_models.resnet18(
                weights=tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        elif variant == "resnet34":
            base = tv_models.resnet34(
                weights=tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            )
        elif variant == "resnet50":
            base = tv_models.resnet50(
                weights=tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        # CIFAR-10 modification: prevent aggressive downsampling
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        # Replace final FC for CIFAR-10
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, num_classes)

        self.model = base
        self.features: dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(self.model, name)
            layer.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.features.clear()
        return self.model(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_resnet(variant: str, num_classes: int = 10, pretrained: bool = False) -> "ResNet":
    """Build a CIFAR-adapted ResNet.  variant ∈ {resnet18, resnet34, resnet50}."""
    return ResNet(variant=variant, num_classes=num_classes, pretrained=pretrained)