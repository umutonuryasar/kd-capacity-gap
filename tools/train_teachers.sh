#!/bin/bash
# Train R50, R34, and R101 teacher models on CIFAR-10 from scratch.
#
# All models use the CIFAR-specific stem (3x3/s1 + Identity); handled by
# build_resnet() in src/models/resnet.py. Teachers are trained with 3 seeds
# so that teacher variance can be reported (paper v1 limitation).
# The seed-0 checkpoint is lifted to the canonical flat path used by KD runs.
#
# Override defaults via environment:
#   EPOCHS=200 DEVICE=cuda:1 bash tools/train_teachers.sh

set -euo pipefail

EPOCHS="${EPOCHS:-200}"
BS="${BS:-128}"
LR="${LR:-0.1}"
SEEDS=(0 1 2)
DEVICE="${DEVICE:-cuda}"

CKPT_DIR="checkpoints"
mkdir -p "${CKPT_DIR}"

train_teacher() {
    local arch=$1 short=$2
    local ckpt="${CKPT_DIR}/teacher_${short}.pth"

    for SEED in "${SEEDS[@]}"; do
        local outdir="runs/teachers/${short}/seed${SEED}"
        if [[ -f "${outdir}/results.json" ]]; then
            echo "  SKIP ${arch} seed${SEED} — already trained."
            continue
        fi
        echo ""
        echo "══════════════════════════════════════════════════════"
        echo "  Teacher: ${arch} seed${SEED}  |  Epochs: ${EPOCHS}"
        echo "══════════════════════════════════════════════════════"
        python tools/train.py \
            --model "${arch}" --kd-type none \
            --seed "${SEED}" --epochs "${EPOCHS}" --batch-size "${BS}" \
            --lr "${LR}" --device "${DEVICE}" --output-dir "${outdir}"
    done

    # Canonical checkpoint used by all KD runs = seed 0, best-val weights
    cp "runs/teachers/${short}/seed0/checkpoint_best.pth" "${ckpt}"
    echo "  Saved canonical teacher: ${ckpt}"
}

train_teacher resnet50  r50
train_teacher resnet34  r34
train_teacher resnet101 r101

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Teacher training complete. Teacher stats:"
echo "    python tools/collect_results.py runs/teachers"
echo "  Next: bash tools/run_ablation.sh"
echo "══════════════════════════════════════════════════════"
