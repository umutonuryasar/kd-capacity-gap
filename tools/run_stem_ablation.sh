#!/bin/bash
# Stem ablation: ImageNet stem (7x7/s2 + MaxPool) vs CIFAR stem (3x3/s1 + Identity)
#
# Produces the data behind the paper's ">5 pp" claim (Architecture Dominates KD).
# 3 seeds per (arch, stem) for the students; single seed for R50 to bound cost.

set -uo pipefail

EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-cuda}"
FAILED=()

run_stem() {
    local arch=$1 stem=$2 seed=$3 epochs=$4
    local out="runs/stem_ablation/${arch}_${stem}/seed${seed}"
    [[ -f "${out}/results.json" ]] && { echo "SKIP ${arch}/${stem}/seed${seed}"; return; }
    echo "── ${arch} stem=${stem} seed${seed} ──"
    python tools/train.py \
        --model "${arch}" --kd-type none --stem "${stem}" \
        --seed "${seed}" --epochs "${epochs}" --device "${DEVICE}" \
        --output-dir "${out}" \
    || FAILED+=("${arch}/${stem}/seed${seed}")
}

for STEM in cifar imagenet; do
    for SEED in 0 1 2; do
        run_stem resnet18 "${STEM}" "${SEED}" "${EPOCHS}"
        run_stem resnet34 "${STEM}" "${SEED}" "${EPOCHS}"
    done
    run_stem resnet50 "${STEM}" 0 200
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED: ${FAILED[*]}"; exit 1
fi
echo "Stem ablation complete. Aggregate with:"
echo "  python tools/collect_results.py runs/stem_ablation"
