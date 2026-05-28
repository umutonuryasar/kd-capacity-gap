"""Training loop for Knowledge Distillation on CIFAR-10."""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Trainer:
    """KD trainer for CIFAR-10.

    Args:
        model:        KDModel or plain ResNet (baseline).
        loss_fn:      KDLoss instance.
        optimizer:    Optimizer.
        scheduler:    LR scheduler (step per epoch).
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        cfg:          Config dict.
        device:       torch.device.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
    ):
        self.model        = model.to(device)
        self.loss_fn      = loss_fn.to(device)
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device

        self.output_dir = Path(cfg.get("output_dir", "runs/experiment"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.writer    = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))
        self.best_acc  = 0.0
        self.global_step = 0

    def train(self, epochs: int) -> None:
        logger.info(f"Starting training for {epochs} epochs.")
        logger.info(f"Output dir: {self.output_dir}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch()
            elapsed       = time.time() - t0

            acc = val_metrics["acc"]
            logger.info(
                f"Epoch {epoch}/{epochs} [{elapsed:.1f}s] "
                f"loss={train_metrics['loss_total']:.4f}  "
                f"ce={train_metrics['loss_ce']:.4f}  "
                f"kd={train_metrics.get('loss_kd', 0.0):.4f}  "
                f"val_acc={acc:.4f}"
            )

            # TensorBoard
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            self.writer.add_scalar("val/acc", acc, epoch)
            self.writer.add_scalar(
                "train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )

            if self.scheduler is not None:
                self.scheduler.step()

            # Save best
            if acc > self.best_acc:
                self.best_acc = acc
                self._save_checkpoint(epoch, tag="best")
                logger.info(f"  New best acc: {self.best_acc:.4f}")

        logger.info(f"Training complete. Best acc: {self.best_acc:.4f}")
        self.writer.close()

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.loss_fn.train()

        running: dict[str, float] = {}
        correct, total = 0, 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            # Baseline: model returns logits directly
            # KDModel: model returns dict
            if isinstance(outputs, dict):
                student_logits  = outputs["student_logits"]
                teacher_logits  = outputs["teacher_logits"]
                student_features = outputs["student_features"]
                teacher_features = outputs["teacher_features"]
            else:
                student_logits  = outputs
                teacher_logits  = outputs.detach()
                student_features = None
                teacher_features = None

            losses = self.loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                student_features=student_features,
                teacher_features=teacher_features,
            )

            losses["loss_total"].backward()
            all_params = list(self.model.parameters()) + list(self.loss_fn.parameters())
            nn.utils.clip_grad_norm_(all_params, 1.0)
            self.optimizer.step()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()

            preds = student_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            self.global_step += 1

        n = len(self.train_loader)
        avg = {k: v / n for k, v in running.items()}
        avg["train_acc"] = correct / total
        return avg

    @torch.no_grad()
    def _val_epoch(self) -> dict[str, float]:
        self.model.eval()
        correct, total = 0, 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            logits  = outputs["student_logits"] if isinstance(outputs, dict) else outputs

            preds    = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        return {"acc": correct / total}

    def _save_checkpoint(self, epoch: int, tag: str = "") -> None:
        model_to_save = getattr(self.model, "student", self.model)
        ckpt = {
            "epoch":            epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_acc":         self.best_acc,
            "cfg":              self.cfg,
        }
        fname = f"checkpoint_{tag}.pth" if tag else f"checkpoint_epoch{epoch:04d}.pth"
        path  = self.output_dir / fname
        torch.save(ckpt, path)
        logger.info(f"Saved checkpoint: {path}")