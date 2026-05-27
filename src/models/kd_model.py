"""Teacher-Student wrapper for Knowledge Distillation.

Teacher is frozen and kept in eval mode throughout training.
Student is trained normally.
"""

import torch
import torch.nn as nn

from .resnet import ResNet


class KDModel(nn.Module):
    """Wraps student and teacher for KD training.

    Args:
        student: Student ResNet (will be trained).
        teacher: Teacher ResNet (frozen, eval mode).
    """

    def __init__(self, student: ResNet, teacher: ResNet):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "KDModel":
        super().train(mode)
        self.teacher.eval()  # always keep teacher in eval
        return self

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Student forward (gradients flow)
        student_logits = self.student(x)
        student_features = dict(self.student.features)

        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_logits = self.teacher(x)
            teacher_features = dict(self.teacher.features)

        return {
            "student_logits":   student_logits,
            "teacher_logits":   teacher_logits,
            "student_features": student_features,
            "teacher_features": teacher_features,
        }

    @property
    def num_student_parameters(self) -> int:
        return self.student.num_parameters

    @property
    def num_teacher_parameters(self) -> int:
        return self.teacher.num_parameters