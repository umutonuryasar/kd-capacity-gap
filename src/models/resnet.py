"""CIFAR-10 ResNet with forward hooks for Knowledge Distillation.

Critical modification (stem="cifar", default): replaces the ImageNet-oriented
7x7 conv (stride=2) and MaxPool with a 3x3 conv (stride=1) and Identity.
This prevents aggressive spatial downsampling of 32x32 CIFAR images.

stem="imagenet" keeps the original stem and is provided ONLY for the
stem-ablation experiment (Table: "Architecture Dominates KD").

Feature hooks on layer1-layer4 expose intermediate activations for Feature-KD.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

_VARIANTS = {
    "resnet18":  (tv_models.resnet18,  "ResNet18_Weights"),
    "resnet34":  (tv_models.resnet34,  "ResNet34_Weights"),
    "resnet50":  (tv_models.resnet50,  "ResNet50_Weights"),
    "resnet101": (tv_models.resnet101, "ResNet101_Weights"),
}


class ResNet(nn.Module):
    def __init__(
        self,
        variant: str = "resnet18",
        num_classes: int = 10,
        pretrained: bool = False,
        stem: str = "cifar",
    ):
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError(f"Unsupported variant: {variant}")
        if stem not in ("cifar", "imagenet"):
            raise ValueError(f"stem must be 'cifar' or 'imagenet', got '{stem}'")

        builder, weights_name = _VARIANTS[variant]
        weights = getattr(tv_models, weights_name).IMAGENET1K_V1 if pretrained else None
        base = builder(weights=weights)

        if stem == "cifar":
            # CIFAR-10 modification: prevent aggressive downsampling
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()

        # Replace final FC for CIFAR-10
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, num_classes)

        self.model = base
        self.stem = stem
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


def build_resnet(
    variant: str,
    num_classes: int = 10,
    pretrained: bool = False,
    stem: str = "cifar",
) -> "ResNet":
    """Build a CIFAR-adapted ResNet.

    variant in {resnet18, resnet34, resnet50, resnet101}.
    stem in {cifar, imagenet} — 'imagenet' only for the stem ablation.
    """
    return ResNet(variant=variant, num_classes=num_classes, pretrained=pretrained, stem=stem)
