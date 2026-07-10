#!/bin/bash
# Bugged-vs-corrected Feature-KD ablation (paper tab:bug, v2 protocol).
#
# Reproduces the v1 bug (projection layers excluded from gradient clipping)
# under IDENTICAL conditions to the corrected Stage-2 final runs:
# R50->R18 Feature-KD, alpha matching best_configs.json, seeds {0..4},
# same 45k/5k split, same baseline. The corrected counterpart already exists
# in runs/final/, so only the bugged runs are needed here.
#
# Pre-clip gradient norms (total and projection-only) are recorded per epoch
# in results.json -> history, substantiating the instability claim directly.

set -uo pipefail

SEEDS=(0 1 2 3 4)
EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-cuda}"
TEACHER_R50="${TEACHER_R50:-checkpoints/teacher_r50.pth}"
BEST_JSON="${BEST_JSON:-best_configs.json}"

[[ -f "${BEST_JSON}" ]] || { echo "Missing ${BEST_JSON}."; exit 1; }
ALPHA=$(python -c "import json; print(json.load(open('${BEST_JSON}'))['resnet50_to_resnet18/feature']['alpha'])")
echo "Bugged Feature-KD ablation: R50->R18, alpha=${ALPHA}, seeds ${SEEDS[*]}"

FAILED=()
for SEED in "${SEEDS[@]}"; do
    out="runs/bug_ablation/resnet50_to_resnet18/feature_bugged/a${ALPHA}/seed${SEED}"
    [[ -f "${out}/results.json" ]] && { echo "SKIP seed${SEED}"; continue; }
    echo "── bugged seed${SEED} ──"
    python tools/train.py \
        --model resnet18 --teacher resnet50 \
        --kd-type feature --alpha "${ALPHA}" --no-proj-clip \
        --teacher-weights "${TEACHER_R50}" \
        --seed "${SEED}" --epochs "${EPOCHS}" --device "${DEVICE}" \
        --output-dir "${out}" \
    || FAILED+=("seed${SEED}")
done

if [[ ${#FAILED[@]} -gt 0 ]]; then echo "FAILED: ${FAILED[*]}"; exit 1; fi
echo "Done. Compare with:"
echo "  python tools/collect_results.py runs  # bugged rows show kd=feature(bugged)"
python - << 'PYEOF'
import json, glob
norms = []
for rj in sorted(glob.glob("runs/bug_ablation/**/results.json", recursive=True)):
    r = json.load(open(rj))
    norms.append((r["seed"], r.get("proj_grad_norm_max_overall", 0.0)))
if norms:
    print("Max unclipped projection grad norm per seed:")
    for s, n in norms:
        print(f"  seed{s}: {n:.2f}")
PYEOF
