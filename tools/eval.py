#!/usr/bin/env python3
"""Fidelity and agreement metrics for a trained student against its teacher.

Motivation: paper v2 claims that student capacity moderates how much "dark
knowledge" a student can absorb. Accuracy deltas of 0.1-0.3 pp cannot carry
that claim alone. This script measures distillation *fidelity* directly
(cf. Stanton et al., 2021, "Does Knowledge Distillation Really Work?"):

  - top1_agreement:  fraction of test samples where argmax(student) == argmax(teacher)
  - kl_t1:           mean KL( p_T || p_S ) at temperature 1
  - kl_t4:           mean KL( p_T || p_S ) at temperature 4 (soft-target regime)
  - student/teacher test accuracy and per-class student accuracy

Usage:
  python tools/eval.py \
      --student-arch resnet34 --student-weights runs/.../checkpoint_best.pth \
      --teacher-arch resnet50 --teacher-weights checkpoints/teacher_r50.pth \
      --output runs/.../fidelity.json
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.resnet import build_resnet

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KD fidelity evaluation")
    p.add_argument("--student-arch", required=True,
                   choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    p.add_argument("--student-weights", required=True)
    p.add_argument("--teacher-arch", required=True,
                   choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    p.add_argument("--teacher-weights", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None,
                   help="Where to write fidelity.json (default: alongside student weights).")
    return p.parse_args()


def load_model(arch: str, weights: str, device: torch.device):
    model = build_resnet(arch, num_classes=10, pretrained=False)
    ckpt = torch.load(weights, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    test_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_ds = torchvision.datasets.CIFAR10(args.data_root, train=False, download=True, transform=test_tf)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    student = load_model(args.student_arch, args.student_weights, device)
    teacher = load_model(args.teacher_arch, args.teacher_weights, device)

    n = 0
    s_correct = t_correct = agree = 0
    kl_t1_sum = kl_t4_sum = 0.0
    per_class_correct = torch.zeros(10, dtype=torch.long)
    per_class_total   = torch.zeros(10, dtype=torch.long)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        s_logits = student(images)
        t_logits = teacher(images)

        s_pred = s_logits.argmax(dim=1)
        t_pred = t_logits.argmax(dim=1)

        n         += labels.size(0)
        s_correct += (s_pred == labels).sum().item()
        t_correct += (t_pred == labels).sum().item()
        agree     += (s_pred == t_pred).sum().item()

        for T_ in (1.0, 4.0):
            s_logp = F.log_softmax(s_logits / T_, dim=-1)
            t_p    = F.softmax(t_logits / T_, dim=-1)
            kl     = F.kl_div(s_logp, t_p, reduction="none", log_target=False).sum(dim=-1)
            if T_ == 1.0:
                kl_t1_sum += kl.sum().item()
            else:
                kl_t4_sum += kl.sum().item()

        for c in range(10):
            mask = labels == c
            per_class_total[c]   += mask.sum().item()
            per_class_correct[c] += (s_pred[mask] == c).sum().item()

    results = {
        "student_arch":     args.student_arch,
        "student_weights":  args.student_weights,
        "teacher_arch":     args.teacher_arch,
        "student_test_acc": s_correct / n,
        "teacher_test_acc": t_correct / n,
        "top1_agreement":   agree / n,
        "kl_t1":            kl_t1_sum / n,
        "kl_t4":            kl_t4_sum / n,
        "per_class_acc": {
            CIFAR10_CLASSES[c]: per_class_correct[c].item() / max(per_class_total[c].item(), 1)
            for c in range(10)
        },
    }

    out = Path(args.output) if args.output else Path(args.student_weights).parent / "fidelity.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nStudent acc:     {results['student_test_acc']:.4f}")
    print(f"Teacher acc:     {results['teacher_test_acc']:.4f}")
    print(f"Top-1 agreement: {results['top1_agreement']:.4f}")
    print(f"KL(T||S) @T=1:   {results['kl_t1']:.4f}")
    print(f"KL(T||S) @T=4:   {results['kl_t4']:.4f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
