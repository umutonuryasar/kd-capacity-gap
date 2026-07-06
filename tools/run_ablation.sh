#!/bin/bash
# kd-capacity-gap — Stage 1: SELECTION grid
#
# Hyperparameter selection runs. All selection happens on the 5k VAL split
# (see tools/train.py); the test set plays no role in this stage.
#
# Grid (per pair):
#   Logit-KD:   alpha in {0.3, 0.5, 0.7} x T in {2, 3, 4}   -> 9 runs
#   Feature-KD: alpha in {0.3, 0.5, 0.7}                    -> 3 runs
# Pairs: R50->R18 | R34->R18 | R50->R34 | R101->R34
# Seeds: 0 (selection only; final numbers come from tools/run_final.sh, 5 seeds)
# Total: 4 pairs x 12 configs = 48 runs
#
# After completion:
#   python tools/collect_results.py runs/select --write-best best_configs.json
#   bash tools/run_final.sh

set -uo pipefail

SELECT_SEED="${SELECT_SEED:-0}"
EPOCHS="${EPOCHS:-100}"
BS="${BS:-128}"
LR="${LR:-0.1}"
FEAT_BETA="${FEAT_BETA:-0.5}"
FEAT_NORM="${FEAT_NORM:-none}"
DEVICE="${DEVICE:-cuda}"

TEACHER_R50="${TEACHER_R50:-checkpoints/teacher_r50.pth}"
TEACHER_R34="${TEACHER_R34:-checkpoints/teacher_r34.pth}"
TEACHER_R101="${TEACHER_R101:-checkpoints/teacher_r101.pth}"

ALPHAS=(0.3 0.5 0.7)
TEMPS=(2 3 4)

COUNT=0
FAILED=()

run_exp() {
    local student=$1 teacher=$2 kd_type=$3 alpha=$4 temp=$5 teacher_ckpt=$6

    COUNT=$((COUNT + 1))
    local tag
    if [[ "${kd_type}" == "logit" ]]; then
        tag="${teacher}_to_${student}/logit/a${alpha}_t${temp}/seed${SELECT_SEED}"
    else
        tag="${teacher}_to_${student}/feature/a${alpha}/seed${SELECT_SEED}"
    fi
    local out="runs/select/${tag}"

    echo ""
    echo "── [${COUNT}] ${tag} ──────────────────────────────────"

    if [[ -f "${out}/results.json" ]]; then
        echo "  SKIP: results already exist — ${out}/results.json"
        return
    fi
    if [[ ! -f "${teacher_ckpt}" ]]; then
        echo "  SKIP: teacher checkpoint not found: ${teacher_ckpt}"
        FAILED+=("${tag} (missing teacher weights)")
        return
    fi

    python tools/train.py \
        --model "${student}" --teacher "${teacher}" \
        --kd-type "${kd_type}" --alpha "${alpha}" --temperature "${temp}" \
        --feat-beta "${FEAT_BETA}" --feat-norm "${FEAT_NORM}" \
        --teacher-weights "${teacher_ckpt}" \
        --seed "${SELECT_SEED}" --epochs "${EPOCHS}" --batch-size "${BS}" \
        --lr "${LR}" --device "${DEVICE}" --output-dir "${out}" \
    || FAILED+=("${tag}")
}

run_pair_grid() {
    local student=$1 teacher=$2 teacher_ckpt=$3
    for A in "${ALPHAS[@]}"; do
        for T in "${TEMPS[@]}"; do
            run_exp "${student}" "${teacher}" logit "${A}" "${T}" "${teacher_ckpt}"
        done
        run_exp "${student}" "${teacher}" feature "${A}" 4 "${teacher_ckpt}"
    done
}

echo "======================================================"
echo "  Stage 1: SELECTION grid (val-based, seed ${SELECT_SEED})"
echo "  Pairs: R50->R18 | R34->R18 | R50->R34 | R101->R34"
echo "======================================================"

run_pair_grid resnet18 resnet50  "${TEACHER_R50}"
run_pair_grid resnet18 resnet34  "${TEACHER_R34}"
run_pair_grid resnet34 resnet50  "${TEACHER_R50}"
run_pair_grid resnet34 resnet101 "${TEACHER_R101}"

echo ""
echo "======================================================"
echo "  Selection grid complete (${COUNT} attempted)"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  FAILED runs (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do echo "    - ${f}"; done
    exit 1
fi
echo "  Next:"
echo "    python tools/collect_results.py runs/select --write-best best_configs.json"
echo "    bash tools/run_final.sh"
echo "======================================================"
