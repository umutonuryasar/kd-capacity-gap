#!/bin/bash
# kd-capacity-gap — Stage 2: FINAL runs
#
# Reads best_configs.json (produced by collect_results.py --write-best from the
# val-based selection grid) and re-runs each best config with 5 seeds.
# Also runs 5-seed baselines for R18 and R34.
#
# These runs produce the numbers reported in the paper:
#   test_acc_best (test accuracy of the best-val checkpoint), mean ± std over 5 seeds.

set -uo pipefail

BEST_JSON="${BEST_JSON:-best_configs.json}"
SEEDS=(0 1 2 3 4)
EPOCHS="${EPOCHS:-100}"
BS="${BS:-128}"
LR="${LR:-0.1}"
FEAT_BETA="${FEAT_BETA:-0.5}"
FEAT_NORM="${FEAT_NORM:-none}"
DEVICE="${DEVICE:-cuda}"

TEACHER_R50="${TEACHER_R50:-checkpoints/teacher_r50.pth}"
TEACHER_R34="${TEACHER_R34:-checkpoints/teacher_r34.pth}"
TEACHER_R101="${TEACHER_R101:-checkpoints/teacher_r101.pth}"

[[ -f "${BEST_JSON}" ]] || { echo "Missing ${BEST_JSON}. Run stage 1 first."; exit 1; }

teacher_ckpt_for() {
    case "$1" in
        resnet50)  echo "${TEACHER_R50}" ;;
        resnet34)  echo "${TEACHER_R34}" ;;
        resnet101) echo "${TEACHER_R101}" ;;
        *) echo ""; ;;
    esac
}

FAILED=()

# ── Baselines: 5 seeds each ───────────────────────────────────────────────────
for STUDENT in resnet18 resnet34; do
    for SEED in "${SEEDS[@]}"; do
        out="runs/final/baseline/${STUDENT}/seed${SEED}"
        if [[ -f "${out}/results.json" ]]; then
            echo "SKIP baseline ${STUDENT} seed${SEED}"
            continue
        fi
        echo "── baseline ${STUDENT} seed${SEED} ──"
        python tools/train.py \
            --model "${STUDENT}" --kd-type none \
            --seed "${SEED}" --epochs "${EPOCHS}" --batch-size "${BS}" \
            --lr "${LR}" --device "${DEVICE}" --output-dir "${out}" \
        || FAILED+=("baseline/${STUDENT}/seed${SEED}")
    done
done

# ── Best KD configs: 5 seeds each ─────────────────────────────────────────────
# Parse best_configs.json into "name teacher student kd alpha temp" lines.
while read -r NAME TEACHER STUDENT KD ALPHA TEMP; do
    CKPT="$(teacher_ckpt_for "${TEACHER}")"
    if [[ ! -f "${CKPT}" ]]; then
        echo "SKIP ${NAME}: teacher checkpoint missing (${CKPT})"
        FAILED+=("${NAME} (missing teacher)")
        continue
    fi
    for SEED in "${SEEDS[@]}"; do
        out="runs/final/${NAME}/seed${SEED}"
        if [[ -f "${out}/results.json" ]]; then
            echo "SKIP ${NAME} seed${SEED}"
            continue
        fi
        echo "── ${NAME} seed${SEED} (alpha=${ALPHA}, T=${TEMP}) ──"
        python tools/train.py \
            --model "${STUDENT}" --teacher "${TEACHER}" \
            --kd-type "${KD}" --alpha "${ALPHA}" --temperature "${TEMP}" \
            --feat-beta "${FEAT_BETA}" --feat-norm "${FEAT_NORM}" \
            --teacher-weights "${CKPT}" \
            --seed "${SEED}" --epochs "${EPOCHS}" --batch-size "${BS}" \
            --lr "${LR}" --device "${DEVICE}" --output-dir "${out}" \
        || FAILED+=("${NAME}/seed${SEED}")

        # Fidelity metrics on the best-val checkpoint
        if [[ -f "${out}/checkpoint_best.pth" && ! -f "${out}/fidelity.json" ]]; then
            python tools/eval.py \
                --student-arch "${STUDENT}" --student-weights "${out}/checkpoint_best.pth" \
                --teacher-arch "${TEACHER}" --teacher-weights "${CKPT}" \
                --device "${DEVICE}" --output "${out}/fidelity.json" \
            || FAILED+=("${NAME}/seed${SEED} (fidelity)")
        fi
    done
done < <(python -c "
import json
best = json.load(open('${BEST_JSON}'))
for name, b in best.items():
    temp = b['temperature'] if b['temperature'] is not None else 4.0
    print(name.replace('/', '_'), b['teacher'], b['student'], b['kd_type'], b['alpha'], temp)
")

echo ""
echo "======================================================"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  FAILED runs (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do echo "    - ${f}"; done
    exit 1
fi
echo "  Final runs complete. Aggregate with:"
echo "    python tools/collect_results.py runs/final --csv final_results.csv"
echo "======================================================"
