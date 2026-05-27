#!/usr/bin/env python3
"""Main training entry point for KD-CIFAR10.

Loss formulation (Hinton et al., 2015):
  L_total = α · L_kd + (1 - α) · L_ce

Usage examples:

# Baseline (no KD)
python tools/train.py --kd-type none --output-dir runs/baseline

# Logit-KD
python tools/train.py --kd-type logit --alpha 0.5 --temperature 4 --output-dir runs/logit_a0.5_t4

# Feature-KD
python tools/train.py --kd-type feature --alpha 0.5 --output-dir runs/feature_a0.5
"""

import sys
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.resnet import build_resnet
from src.models.kd_model import KDModel
from src.losses.kd_loss import KDLoss
from src.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KD-CIFAR10 Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"],
                   help="Student model architecture. Use resnet50 for teacher training.")

    # KD settings
    p.add_argument("--kd-type", default="logit", choices=["logit", "feature", "none"])
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Distillation weight in [0,1]. L_total = α·L_kd + (1-α)·L_ce.")
    p.add_argument("--temperature", type=float, default=4.0,
                   help="Softmax temperature T for logit KD.")
    p.add_argument("--feat-beta", type=float, default=0.5,
                   help="Cosine similarity weight inside FeatureKD.")

    # Teacher weights
    p.add_argument("--teacher-weights", default=None,
                   help="Path to pretrained teacher checkpoint.")

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum",     type=float, default=0.9)

    # Output
    p.add_argument("--output-dir",  default="runs/experiment")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def build_dataloaders(batch_size: int, num_workers: int):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.ImageFolder(root="data/train", transform=train_tf)
    val_ds   = torchvision.datasets.ImageFolder(root="data/test",  transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Data ----
    train_loader, val_loader = build_dataloaders(args.batch_size, args.num_workers)
    logger.info(f"Train: {len(train_loader.dataset):,}  Val: {len(val_loader.dataset):,}")

    # ---- Models ----
    student = build_resnet(args.model, num_classes=10, pretrained=False)
    logger.info(f"Student params: {student.num_parameters:,}")

    if args.kd_type != "none":
        teacher = build_resnet("resnet50", num_classes=10, pretrained=False)
        logger.info(f"Teacher params: {teacher.num_parameters:,}")

        if args.teacher_weights:
            logger.info(f"Loading teacher weights: {args.teacher_weights}")
            ckpt  = torch.load(args.teacher_weights, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            teacher.load_state_dict(state)
        else:
            logger.warning("No teacher weights — teacher uses random init.")

        model = KDModel(student=student, teacher=teacher)
    else:
        model = student

    # ---- Loss ----
    loss_fn = KDLoss(
        kd_type=args.kd_type,
        alpha=args.alpha,
        temperature=args.temperature,
        feat_beta=args.feat_beta,
        student_channels=512,
        teacher_channels=2048,
    )

    # ---- Optimizer ----
    base_model = getattr(model, "student", model)
    optimizer  = torch.optim.SGD(
        base_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # Also optimize FeatureKD projection layers
    kd_params = [p for p in loss_fn.parameters() if p.requires_grad]
    if kd_params:
        optimizer.add_param_group({"params": kd_params, "lr": args.lr})

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-4
    )

    # ---- Train ----
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=vars(args),
        device=device,
    )
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
