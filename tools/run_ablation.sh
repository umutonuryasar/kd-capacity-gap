#!/bin/bash
# KD-CIFAR10 Ablation Study — Experiment 2 (CIFAR-specific architecture)
# Loss: L_total = α · L_kd + (1 - α) · L_ce  (Hinton et al., 2015)
#
# Grid:
#   Logit-KD α sweep: α ∈ {0.3, 0.5, 0.7}, T=2 (best from Exp 1)
#   Logit-KD T sweep: T ∈ {2, 3, 4}, α=0.5 (best from Exp 1)
#   Feature-KD α sweep: α ∈ {0.3, 0.5, 0.7}, β=0.5
#   Total: 8 unique runs (logit_a0.5_t2 shared between α and T sweeps)

TEACHER="runs/teacher_r50_v2/checkpoint_best.pth"
EPOCHS=100
BS=128
LR=0.1
DEVICE="cuda"

echo "============================================"
echo "KD-CIFAR10 Ablation Study — Experiment 2"
echo "Loss: L_total = α·L_kd + (1-α)·L_ce"
echo "Teacher: $TEACHER"
echo "============================================"

# ── Logit-KD: alpha sweep (T=2 fixed) ────────────────────────────────────
echo "[1/8] Logit-KD α=0.3 T=2"
python tools/train.py --model resnet18 \
    --kd-type logit --alpha 0.3 --temperature 2 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/logit_a0.3_t2 --device $DEVICE

echo "[2/8] Logit-KD α=0.5 T=2  (shared with T-sweep run 4)"
python tools/train.py --model resnet18 \
    --kd-type logit --alpha 0.5 --temperature 2 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/logit_a0.5_t2 --device $DEVICE

echo "[3/8] Logit-KD α=0.7 T=2"
python tools/train.py --model resnet18 \
    --kd-type logit --alpha 0.7 --temperature 2 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/logit_a0.7_t2 --device $DEVICE

# ── Logit-KD: temperature sweep (α=0.5 fixed) ────────────────────────────
# T=2 already done above (runs/logit_a0.5_t2)

echo "[5/8] Logit-KD α=0.5 T=3"
python tools/train.py --model resnet18 \
    --kd-type logit --alpha 0.5 --temperature 3 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/logit_a0.5_t3 --device $DEVICE

echo "[6/8] Logit-KD α=0.5 T=4"
python tools/train.py --model resnet18 \
    --kd-type logit --alpha 0.5 --temperature 4 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/logit_a0.5_t4 --device $DEVICE

# ── Feature-KD: alpha sweep ───────────────────────────────────────────────
echo "[7/8] Feature-KD α=0.3"
python tools/train.py --model resnet18 \
    --kd-type feature --alpha 0.3 --feat-beta 0.5 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/feature_a0.3 --device $DEVICE

echo "[8/8] Feature-KD α=0.5"
python tools/train.py --model resnet18 \
    --kd-type feature --alpha 0.5 --feat-beta 0.5 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/feature_a0.5 --device $DEVICE

echo "[9/8] Feature-KD α=0.7"
python tools/train.py --model resnet18 \
    --kd-type feature --alpha 0.7 --feat-beta 0.5 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/feature_a0.7 --device $DEVICE

echo "============================================"
echo "Ablation complete!"
echo "============================================"
