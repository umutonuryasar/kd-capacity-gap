#!/bin/bash
# Train R50 and R34 teacher models on CIFAR-10 from scratch.
#
# Both models use the CIFAR-specific stem: 3×3 conv (stride=1) + Identity
# maxpool instead of the ImageNet 7×7 conv (stride=2) + MaxPool. This is
# handled automatically by build_resnet() in src/models/resnet.py.
#
# Checkpoints are saved to:
#   checkpoints/teacher_r50.pth
#   checkpoints/teacher_r34.pth
#
# Override defaults via environment:
#   EPOCHS=200 DEVICE=cuda:1 bash tools/train_teachers.sh
#
# Skip a model if its checkpoint already exists. Delete the file to retrain.

set -euo pipefail

EPOCHS="${EPOCHS:-200}"
BS="${BS:-128}"
LR="${LR:-0.1}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"

CKPT_DIR="checkpoints"
mkdir -p "${CKPT_DIR}"

# ── Helper ────────────────────────────────────────────────────────────────────
train_teacher() {
    local arch=$1                               # e.g. resnet50
    local short=$2                              # e.g. r50
    local ckpt="${CKPT_DIR}/teacher_${short}.pth"
    local outdir="runs/teachers/${short}"

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Teacher: ${arch}  →  ${ckpt}"
    echo "  Epochs: ${EPOCHS}  |  LR: ${LR}  |  Seed: ${SEED}"
    echo "══════════════════════════════════════════════════════"

    if [[ -f "${ckpt}" ]]; then
        echo "  Already exists — skipping. Delete ${ckpt} to retrain."
        return
    fi

    python tools/train.py \
        --model        "${arch}"    \
        --kd-type      none         \
        --seed         "${SEED}"    \
        --epochs       "${EPOCHS}"  \
        --batch-size   "${BS}"      \
        --lr           "${LR}"      \
        --device       "${DEVICE}"  \
        --output-dir   "${outdir}"

    # Lift the best checkpoint to the canonical flat path
    cp "${outdir}/checkpoint_best.pth" "${ckpt}"
    echo "  Saved: ${ckpt}"
}

# ── Train teachers ────────────────────────────────────────────────────────────
train_teacher resnet50 r50
train_teacher resnet34 r34

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Teacher training complete."
for ckpt in "${CKPT_DIR}/teacher_r50.pth" "${CKPT_DIR}/teacher_r34.pth"; do
    if [[ -f "${ckpt}" ]]; then
        echo "  ✓ ${ckpt}"
    else
        echo "  ✗ MISSING: ${ckpt}"
    fi
done
echo ""
echo "  Run distillation experiments with:"
echo "    bash tools/run_ablation.sh"
echo "══════════════════════════════════════════════════════"
