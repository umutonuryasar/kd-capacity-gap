#!/bin/bash
# KD-CIFAR10 Full Ablation
#
# Pairs:  R50→R18  |  R34→R18  |  R50→R34
# Types:  logit-KD  |  feature-KD
# Seeds:  0  1  2
# Total:  3 pairs × 2 KD types × 3 seeds = 18 runs
#
# Output layout:
#   runs/{teacher}_to_{student}/{kd_type}/seed{seed}/
#
# Aggregate mean ± std across seeds with:
#   python tools/collect_results.py runs/

set -uo pipefail

# ── Hyper-parameters ──────────────────────────────────────────────────────────
SEEDS=(0 1 2)
EPOCHS="${EPOCHS:-100}"
BS="${BS:-128}"
LR="${LR:-0.1}"
ALPHA="${ALPHA:-0.5}"
TEMPERATURE="${TEMPERATURE:-4.0}"
FEAT_BETA="${FEAT_BETA:-0.5}"
DEVICE="${DEVICE:-cuda}"

# Pre-trained teacher checkpoints (must exist before running KD experiments)
TEACHER_R50="${TEACHER_R50:-checkpoints/teacher_r50.pth}"
TEACHER_R34="${TEACHER_R34:-checkpoints/teacher_r34.pth}"

# ── Helpers ───────────────────────────────────────────────────────────────────
TOTAL=$((${#SEEDS[@]} * 3 * 2))   # 3 pairs × 2 KD types × 3 seeds
COUNT=0
FAILED=()

run_exp() {
    local student=$1 teacher=$2 kd_type=$3 seed=$4 teacher_ckpt=$5

    COUNT=$((COUNT + 1))
    local tag="${teacher}_to_${student}/${kd_type}/seed${seed}"
    local out="runs/${tag}"

    echo ""
    echo "── [${COUNT}/${TOTAL}] ${tag} ──────────────────────────────────"

    if [[ -f "${out}/checkpoint_best.pth" ]]; then
        echo "  SKIP: checkpoint already exists — ${out}/checkpoint_best.pth"
        return
    fi

    if [[ ! -f "${teacher_ckpt}" ]]; then
        echo "  SKIP: teacher checkpoint not found: ${teacher_ckpt}"
        FAILED+=("${tag} (missing teacher weights)")
        return
    fi

    python tools/train.py \
        --model        "${student}"     \
        --teacher      "${teacher}"     \
        --kd-type      "${kd_type}"     \
        --alpha        "${ALPHA}"       \
        --temperature  "${TEMPERATURE}" \
        --feat-beta    "${FEAT_BETA}"   \
        --teacher-weights "${teacher_ckpt}" \
        --seed         "${seed}"        \
        --epochs       "${EPOCHS}"      \
        --batch-size   "${BS}"          \
        --lr           "${LR}"          \
        --device       "${DEVICE}"      \
        --output-dir   "${out}"         \
    || FAILED+=("${tag}")
}

# ── Main loop ─────────────────────────────────────────────────────────────────
echo "======================================================"
echo "  KD-CIFAR10 Full Ablation"
echo "  Pairs:  R50→R18 | R34→R18 | R50→R34"
echo "  Types:  logit | feature"
echo "  Seeds:  ${SEEDS[*]}"
echo "  Total:  ${TOTAL} runs"
echo "======================================================"

for SEED in "${SEEDS[@]}"; do

    # ── R50 → R18 ─────────────────────────────────────────────────────────────
    run_exp resnet18 resnet50 logit   "${SEED}" "${TEACHER_R50}"
    run_exp resnet18 resnet50 feature "${SEED}" "${TEACHER_R50}"

    # ── R34 → R18 ─────────────────────────────────────────────────────────────
    run_exp resnet18 resnet34 logit   "${SEED}" "${TEACHER_R34}"
    run_exp resnet18 resnet34 feature "${SEED}" "${TEACHER_R34}"

    # ── R50 → R34 ─────────────────────────────────────────────────────────────
    run_exp resnet34 resnet50 logit   "${SEED}" "${TEACHER_R50}"
    run_exp resnet34 resnet50 feature "${SEED}" "${TEACHER_R50}"

done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Ablation complete  (${COUNT}/${TOTAL} attempted)"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  FAILED runs (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do echo "    - ${f}"; done
    exit 1
else
    echo "  All runs succeeded."
fi
echo "======================================================"
