#!/usr/bin/env python3
"""Main training entry point for kd-capacity-gap.

Loss formulation (Hinton et al., 2015):
  L_total = alpha * L_kd + (1 - alpha) * L_ce

Data protocol (v2):
  CIFAR-10 train (50k) is split ONCE, deterministically, into 45k train / 5k val
  using a fixed SPLIT_SEED that is independent of the run seed. All model and
  hyperparameter selection happens on the val split. The 10k test set is touched
  exactly twice per run: once with the best-val checkpoint, once with the final
  checkpoint. Both numbers are written to results.json.

Usage examples:

# Baseline (no KD)
python tools/train.py --kd-type none --output-dir runs/baseline

# Logit-KD (R50 teacher -> R18 student)
python tools/train.py --kd-type logit --alpha 0.5 --temperature 4 \
    --teacher-weights checkpoints/teacher_r50.pth --output-dir runs/logit_a0.5_t4

# Feature-KD via config file
python tools/train.py --config configs/r34_to_r18_feature.yaml

# Stem ablation (ImageNet stem, for the "Architecture Dominates KD" table)
python tools/train.py --kd-type none --stem imagenet --output-dir runs/stem_ablation/r18
"""

import sys
import random
import argparse
import logging
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.seed import set_seed
from src.models.resnet import build_resnet
from src.models.kd_model import KDModel
from src.losses.kd_loss import KDLoss
from src.distillation.feature_kd import RESNET_BASIC_CHANNELS, RESNET_BOTTLENECK_CHANNELS
from src.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

# Fixed split seed — deliberately independent of the run seed so every run
# (all seeds, all configs) shares the exact same 45k/5k partition.
SPLIT_SEED = 1234
VAL_SIZE = 5000

# Per-channel statistics of the CIFAR-10 training set.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Channel map by architecture family
_ARCH_CHANNELS = {
    "resnet18":  RESNET_BASIC_CHANNELS,
    "resnet34":  RESNET_BASIC_CHANNELS,
    "resnet50":  RESNET_BOTTLENECK_CHANNELS,
    "resnet101": RESNET_BOTTLENECK_CHANNELS,
}

_ARCHS = list(_ARCH_CHANNELS.keys())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="kd-capacity-gap Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config", default=None,
                   help="Path to YAML config file. CLI flags override config values.")

    # Model
    p.add_argument("--model", default="resnet18", choices=_ARCHS,
                   help="Student architecture.")
    p.add_argument("--teacher", default="resnet50", choices=_ARCHS,
                   help="Teacher architecture (ignored when --kd-type none).")
    p.add_argument("--stem", default="cifar", choices=["cifar", "imagenet"],
                   help="Input stem. 'imagenet' only for the stem ablation.")

    # KD settings
    p.add_argument("--kd-type", default="logit", choices=["logit", "feature", "none"])
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Distillation weight in [0,1]. L_total = a*L_kd + (1-a)*L_ce.")
    p.add_argument("--temperature", type=float, default=4.0,
                   help="Softmax temperature T for logit KD.")
    p.add_argument("--feat-beta", type=float, default=0.5,
                   help="Cosine similarity weight inside FeatureKD.")
    p.add_argument("--no-proj-clip", action="store_true",
                   help="BUG-REPRODUCTION MODE: exclude Feature-KD projection "
                        "layers from gradient clipping, replicating the v1 bug. "
                        "Only for the bugged-vs-corrected ablation.")
    p.add_argument("--feat-norm", default="none", choices=["none", "teacher_std"],
                   help="Feature scale normalization inside FeatureKD. "
                        "'teacher_std' makes MSE scale-invariant across teachers.")

    # Teacher weights
    p.add_argument("--teacher-weights", default=None,
                   help="Path to pretrained teacher checkpoint. REQUIRED for KD runs.")

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum",     type=float, default=0.9)

    # Data
    p.add_argument("--data-root",   default="data",
                   help="CIFAR-10 download/cache directory.")
    p.add_argument("--val-size",    type=int, default=VAL_SIZE,
                   help="Held-out validation size taken from the 50k train set.")

    # Output / misc
    p.add_argument("--output-dir",  default="runs/experiment")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    return p


def parse_args() -> argparse.Namespace:
    p = build_parser()

    # Two-pass: first extract --config, then re-parse with YAML as defaults.
    pre, _ = p.parse_known_args()
    if pre.config:
        with open(pre.config) as f:
            cfg = yaml.safe_load(f) or {}
        # Replace hyphens -> underscores so YAML keys match argparse dests
        cfg = {k.replace("-", "_"): v for k, v in cfg.items()}
        cfg.pop("config", None)  # don't overwrite the config path itself
        p.set_defaults(**cfg)

    return p.parse_args()


def _worker_init_fn(worker_id: int) -> None:
    """Seed numpy/random inside each DataLoader worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    run_seed: int,
    val_size: int = VAL_SIZE,
):
    """Return (train_loader, val_loader, test_loader).

    - train/val come from a fixed, seed-independent split of the 50k train set.
    - val and test use eval transforms (no augmentation).
    - Shuffling order depends on the run seed via a dedicated generator.
    """
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Two dataset views over the same files: one with train transforms,
    # one with eval transforms. Split indices are shared.
    full_train_aug  = torchvision.datasets.CIFAR10(data_root, train=True,  download=True, transform=train_tf)
    full_train_eval = torchvision.datasets.CIFAR10(data_root, train=True,  download=True, transform=eval_tf)
    test_ds         = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=eval_tf)

    n_total = len(full_train_aug)  # 50000
    split_gen = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(n_total, generator=split_gen).tolist()
    val_indices   = perm[:val_size]
    train_indices = perm[val_size:]

    train_ds = Subset(full_train_aug,  train_indices)
    val_ds   = Subset(full_train_eval, val_indices)

    loader_gen = torch.Generator().manual_seed(run_seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        generator=loader_gen, worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    logger.info(f"Seed: {args.seed}  |  Split seed (fixed): {SPLIT_SEED}")

    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Fail fast: teacher weights are mandatory for KD runs ----
    # (Checked BEFORE data download so a misconfigured run dies immediately.
    #  Paper §4.2: silent fallback to a random teacher is not permitted.)
    if args.kd_type != "none":
        if not args.teacher_weights:
            raise ValueError(
                "KD run requested but --teacher-weights not provided. "
                "Silent fallback to a random teacher is not permitted."
            )
        if not Path(args.teacher_weights).is_file():
            raise ValueError(f"Teacher checkpoint not found: {args.teacher_weights}")

    # ---- Data ----
    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_root, args.batch_size, args.num_workers,
        run_seed=args.seed, val_size=args.val_size,
    )
    logger.info(
        f"Train: {len(train_loader.dataset):,}  "
        f"Val: {len(val_loader.dataset):,}  "
        f"Test: {len(test_loader.dataset):,}"
    )

    # ---- Models ----
    student = build_resnet(args.model, num_classes=10, pretrained=False, stem=args.stem)
    logger.info(f"Student ({args.model}, stem={args.stem}): {student.num_parameters:,} params")

    if args.kd_type != "none":
        teacher = build_resnet(args.teacher, num_classes=10, pretrained=False, stem=args.stem)
        logger.info(f"Teacher ({args.teacher}): {teacher.num_parameters:,} params")

        ckpt_path = Path(args.teacher_weights)
        logger.info(f"Loading teacher weights: {ckpt_path}")
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        teacher.load_state_dict(state)

        model = KDModel(student=student, teacher=teacher)
    else:
        model = student

    # ---- Loss ----
    loss_fn = KDLoss(
        kd_type=args.kd_type,
        alpha=args.alpha,
        temperature=args.temperature,
        feat_beta=args.feat_beta,
        feat_norm=args.feat_norm,
        student_channels=_ARCH_CHANNELS[args.model],
        teacher_channels=_ARCH_CHANNELS[args.teacher],
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
        test_loader=test_loader,
        cfg=vars(args),
        device=device,
    )
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
