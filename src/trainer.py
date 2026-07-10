"""Training loop for Knowledge Distillation on CIFAR-10.

Evaluation protocol (v2):
  - Per-epoch evaluation happens ONLY on the held-out val split (5k).
  - The best checkpoint is selected by val accuracy.
  - The 10k test set is evaluated exactly twice, after training:
      (a) with the best-val checkpoint  -> test_acc_best
      (b) with the final-epoch weights  -> test_acc_final
  - Everything is written to <output_dir>/results.json for aggregation
    by tools/collect_results.py.
"""

import json
import time
import logging
from pathlib import Path

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
        train_loader: Training DataLoader (45k).
        val_loader:   Validation DataLoader (5k, used for model selection).
        test_loader:  Test DataLoader (10k, used only for final reporting).
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
        test_loader: DataLoader,
        cfg: dict,
        device: torch.device,
    ):
        self.model        = model.to(device)
        self.loss_fn      = loss_fn.to(device)
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.cfg          = cfg
        self.device       = device

        self.output_dir = Path(cfg.get("output_dir", "runs/experiment"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.writer         = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))
        self.best_val_acc   = 0.0
        self.best_val_epoch = 0
        self.global_step    = 0
        self.history: list[dict] = []

    def train(self, epochs: int) -> dict:
        logger.info(f"Starting training for {epochs} epochs.")
        logger.info(f"Output dir: {self.output_dir}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_acc       = self._evaluate(self.val_loader)
            elapsed       = time.time() - t0

            logger.info(
                f"Epoch {epoch}/{epochs} [{elapsed:.1f}s] "
                f"loss={train_metrics['loss_total']:.4f}  "
                f"ce={train_metrics['loss_ce']:.4f}  "
                f"kd={train_metrics.get('loss_kd', 0.0):.4f}  "
                f"val_acc={val_acc:.4f}"
            )

            # TensorBoard
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            self.writer.add_scalar("val/acc", val_acc, epoch)
            self.writer.add_scalar(
                "train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )
            self.history.append(
                {"epoch": epoch, "val_acc": val_acc, **train_metrics}
            )

            if self.scheduler is not None:
                self.scheduler.step()

            # Save best (selected on VAL, never on test)
            if val_acc > self.best_val_acc:
                self.best_val_acc   = val_acc
                self.best_val_epoch = epoch
                self._save_checkpoint(epoch, tag="best")
                logger.info(f"  New best val acc: {self.best_val_acc:.4f}")

        # ---- Final reporting: the ONLY two test-set evaluations ----
        self._save_checkpoint(epochs, tag="final")
        test_acc_final = self._evaluate(self.test_loader)

        self._load_student_state(self.output_dir / "checkpoint_best.pth")
        test_acc_best = self._evaluate(self.test_loader)

        logger.info(
            f"Training complete. "
            f"best_val={self.best_val_acc:.4f} (epoch {self.best_val_epoch})  "
            f"test@best_val={test_acc_best:.4f}  test@final={test_acc_final:.4f}"
        )

        results = {
            "model":          self.cfg.get("model"),
            "teacher":        self.cfg.get("teacher") if self.cfg.get("kd_type") != "none" else None,
            "kd_type":        self.cfg.get("kd_type"),
            "alpha":          self.cfg.get("alpha"),
            "temperature":    self.cfg.get("temperature"),
            "feat_beta":      self.cfg.get("feat_beta"),
            "stem":           self.cfg.get("stem", "cifar"),
            "seed":           self.cfg.get("seed"),
            "no_proj_clip":   bool(self.cfg.get("no_proj_clip", False)),
            "grad_norm_max_overall":
                max((h.get("grad_norm_max", 0.0) for h in self.history), default=0.0),
            "proj_grad_norm_max_overall":
                max((h.get("proj_grad_norm_max", 0.0) for h in self.history), default=0.0),
            "epochs":         self.cfg.get("epochs"),
            "best_val_acc":   self.best_val_acc,
            "best_val_epoch": self.best_val_epoch,
            "test_acc_best":  test_acc_best,
            "test_acc_final": test_acc_final,
        }
        with open(self.output_dir / "results.json", "w") as f:
            json.dump({**results, "history": self.history}, f, indent=2)
        logger.info(f"Wrote {self.output_dir / 'results.json'}")

        self.writer.close()
        return results

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.loss_fn.train()

        running: dict[str, float] = {}
        correct, total = 0, 0
        epoch_grad_norm_max = 0.0
        epoch_proj_norm_max = 0.0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            # Baseline: model returns logits directly
            # KDModel: model returns dict
            if isinstance(outputs, dict):
                student_logits   = outputs["student_logits"]
                teacher_logits   = outputs["teacher_logits"]
                student_features = outputs["student_features"]
                teacher_features = outputs["teacher_features"]
            else:
                student_logits   = outputs
                teacher_logits   = outputs.detach()
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
            # Clip the union of student and projection-layer parameters.
            # Excluding projections was the v1 bug (paper §4.2); the
            # no_proj_clip flag reproduces that bug deliberately for the
            # bugged-vs-corrected ablation. In both modes we record the
            # pre-clip gradient norms as evidence.
            if self.cfg.get("no_proj_clip"):
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                proj_grads = [p.grad for p in self.loss_fn.parameters()
                              if p.grad is not None]
                if proj_grads:
                    proj_norm = torch.norm(
                        torch.stack([g.norm() for g in proj_grads])
                    ).item()
                    epoch_proj_norm_max = max(epoch_proj_norm_max, proj_norm)
            else:
                all_params = list(self.model.parameters()) + list(self.loss_fn.parameters())
                total_norm = nn.utils.clip_grad_norm_(all_params, 1.0)
            epoch_grad_norm_max = max(epoch_grad_norm_max, float(total_norm))
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
        avg["grad_norm_max"] = epoch_grad_norm_max
        if epoch_proj_norm_max > 0:
            avg["proj_grad_norm_max"] = epoch_proj_norm_max
        return avg

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            logits  = outputs["student_logits"] if isinstance(outputs, dict) else outputs

            preds    = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        return correct / total

    def _load_student_state(self, ckpt_path: Path) -> None:
        ckpt  = torch.load(ckpt_path, map_location=self.device)
        state = ckpt["model_state_dict"]
        target = getattr(self.model, "student", self.model)
        target.load_state_dict(state)

    def _save_checkpoint(self, epoch: int, tag: str = "") -> None:
        model_to_save = getattr(self.model, "student", self.model)
        ckpt = {
            "epoch":                epoch,
            "model_state_dict":     model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc":         self.best_val_acc,
            "cfg":                  self.cfg,
        }
        fname = f"checkpoint_{tag}.pth" if tag else f"checkpoint_epoch{epoch:04d}.pth"
        path  = self.output_dir / fname
        torch.save(ckpt, path)
        logger.info(f"Saved checkpoint: {path}")
