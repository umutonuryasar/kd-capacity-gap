# Student Capacity Moderates Knowledge Distillation Effectiveness
### A Systematic Study Across ResNet Teacher-Student Pairs on CIFAR-10

[![arXiv](https://img.shields.io/badge/arXiv-2605.31191-b31b1b.svg)](https://arxiv.org/abs/2605.31191)
[![HF Spaces](https://img.shields.io/badge/HF%20Spaces-Demo-orange.svg)](https://huggingface.co/spaces/umutonuryasar/kd-capacity-gap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)]()

---

## Overview

This repository contains the full code and results for a systematic study of knowledge distillation (KD) effectiveness across **four** ResNet teacher-student capacity pairs on CIFAR-10. Logit-KD and Feature-KD are compared under a leakage-free protocol: hyperparameters and checkpoints selected on a held-out validation split, selected configurations re-run with **5 seeds**, and the test set used exclusively for final reporting — alongside distillation *fidelity* metrics (teacher-student agreement, KL divergence).

Key findings (v2):

1. **Student capacity moderates KD** — the only statistically significant gains belong to R34 students under Feature-KD; doubling teacher size at fixed teacher accuracy (R50 → R101) leaves the gain unchanged.
2. **Feature-KD ≥ Logit-KD in all four pairs**, and Feature-KD students land closer to the teacher's output distribution despite never observing teacher logits.
3. **v1 correction** — the gradient-clipping bug blamed in v1 for Feature-KD's underperformance had *no measurable effect* on controlled re-run; v1's larger gains are explained by test-set hyperparameter selection.

## Demo

Interactive results explorer available on HF Spaces: [huggingface.co/spaces/umutonuryasar/kd-capacity-gap](https://huggingface.co/spaces/umutonuryasar/kd-capacity-gap)

---

## Key Results (v2 protocol)

KD gain over baseline (test acc, mean over 5 seeds; * = p<0.05, Welch vs. baseline):

| Pair | Logit-KD Δ | Feature-KD Δ | Best | Teacher-student agreement (best) |
|------|-----------|--------------|------|----------------------------------|
| R34→R18  | −0.08 pp | +0.13 pp  | Feature | 95.53% |
| R50→R18  | +0.06 pp | +0.10 pp  | Feature | 95.49% |
| R50→R34  | +0.08 pp | **+0.19 pp*** | Feature | 95.41% |
| R101→R34 | +0.06 pp | **+0.21 pp*** | Feature | 95.53% |

Baselines: R18 94.86 ± 0.14, R34 95.04 ± 0.13. Teachers (3 seeds): R34 95.30,
R50 95.36, R101 95.37.

Headline findings:
- **Student capacity moderates KD**: the only significant gains belong to R34
  students under Feature-KD. Doubling teacher size at fixed teacher accuracy
  (R50 → R101) leaves the gain unchanged — the moderating variable sits on
  the student side.
- **Feature-KD ≥ Logit-KD in all four pairs**, and Feature-KD students land
  closer to the teacher's T=1 output distribution (KL 0.14–0.19 vs 0.21–0.28)
  despite never observing teacher logits.
- **Fidelity decouples from accuracy**: top-1 teacher-student agreement is
  flat (95.3–95.5%) across all cells.
- **Architecture dominates KD**: the CIFAR stem correction is worth
  +5.50 to +7.15 pp — more than 25× the largest KD gain.
- **v1 correction**: the gradient-clipping bug blamed in v1 had no measurable
  effect on controlled re-run (bugged 95.00 ± 0.18 vs corrected 94.95 ± 0.20,
  p=0.69; unclipped projection norms ≤ 0.21 against a threshold of 1.0). v1's
  larger gains are explained by test-set hyperparameter selection.

Full per-pair tables, fidelity metrics, and statistics: see the paper (v2)
and [`results/`](results/). Regenerate figures with
`python tools/make_figures.py`.

## Evaluation Protocol (v2)

- CIFAR-10 train (50k) is split once, deterministically (fixed `SPLIT_SEED=1234`,
  independent of run seeds), into **45k train / 5k val**.
- **All** model and hyperparameter selection happens on the val split.
- The 10k test set is evaluated exactly twice per run: with the best-val
  checkpoint and with the final-epoch checkpoint. Both are reported.
- Two-stage experiment protocol:
  1. **Selection** (`tools/run_ablation.sh`): full hyperparameter grid, seed 0,
     compared by val accuracy. `tools/collect_results.py --write-best` records
     the winning config per (pair, KD method).
  2. **Final** (`tools/run_final.sh`): best configs and baselines re-run with
     seeds {0,1,2,3,4}; the paper reports `test_acc_best` as mean ± std.
- Fidelity metrics (`tools/eval.py`): teacher-student top-1 agreement,
  KL(p_T‖p_S) at T∈{1,4}, per-class accuracy — computed on every final run.
- Stem ablation (`tools/run_stem_ablation.sh`): ImageNet stem vs CIFAR stem,
  quantifying the "Architecture Dominates KD" claim.

## Reproducing

```bash
pip install -r requirements.txt
bash tools/train_teachers.sh                                        # R50, R34, R101 (3 seeds each)
bash tools/run_ablation.sh                                          # Stage 1: selection grid
python tools/collect_results.py runs/select --write-best best_configs.json
bash tools/run_final.sh                                             # Stage 2: 5-seed finals + fidelity
bash tools/run_stem_ablation.sh                                     # Stem ablation
python tools/collect_results.py runs/final --csv final_results.csv
```

CIFAR-10 downloads automatically to `data/` on first run.

---

## Repository Structure

```
kd-capacity-gap/
├── configs/                  # YAML configs for each teacher-student pair × KD method
│   ├── r34_to_r18_logit.yaml
│   ├── r34_to_r18_feature.yaml
│   ├── r50_to_r18_logit.yaml
│   ├── r50_to_r18_feature.yaml
│   ├── r50_to_r34_logit.yaml
│   ├── r50_to_r34_feature.yaml
│   ├── r101_to_r34_logit.yaml
│   └── r101_to_r34_feature.yaml
├── notebooks/
│   └── run_experiments.ipynb # Colab harness for the experiment scripts
├── results/                  # Final CSVs, selected configs, raw per-run JSON
├── src/
│   ├── distillation/         # Logit-KD and Feature-KD loss implementations
│   ├── losses/
│   ├── models/               # ResNet definitions with CIFAR-specific stem
│   ├── utils/
│   ├── __init__.py
│   └── trainer.py
├── tools/
│   ├── train.py              # Single-run training entry point
│   ├── eval.py               # Fidelity metrics (agreement, KL, per-class acc)
│   ├── train_teachers.sh     # Train all teacher models (3 seeds each)
│   ├── run_ablation.sh       # Stage 1: selection grid, seed 0, compared on val
│   ├── run_final.sh          # Stage 2: selected configs + baselines, 5 seeds
│   ├── run_stem_ablation.sh  # ImageNet stem vs CIFAR stem
│   ├── run_bug_ablation.sh   # Controlled re-run of the v1 clipping bug
│   ├── collect_results.py    # Aggregate runs into CSV / best_configs.json
│   ├── make_figures.py       # Regenerate the paper figures
│   └── __init__.py
├── requirements.txt
└── LICENSE
```

---

## Reproducibility

All experiments are fully deterministic given a fixed seed. The following are set at the start of every run:

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
numpy.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Seeds: selection grid on seed **0**; final runs (selected configs and
baselines) on seeds **{0, 1, 2, 3, 4}**, reported as mean ± std; teachers on 3
seeds. The train/val split uses its own fixed `SPLIT_SEED=1234`, independent
of the run seed.

---

## Experimental Setup

| Hyperparameter | Value |
|----------------|-------|
| Dataset | CIFAR-10 (45k train / 5k val / 10k test) |
| Input resolution | 32×32 |
| Optimizer | SGD (momentum=0.9, weight_decay=5e-4, Nesterov=True) |
| Learning rate | 0.1 with CosineAnnealingLR (T_max=100, η_min=1e-4) |
| Batch size | 128 |
| Student epochs | 100 |
| Teacher epochs | 200 |
| Hardware | NVIDIA A100-SXM4-40GB |

### KD Hyperparameter Grid

**Logit-KD:** α ∈ {0.3, 0.5, 0.7}, T ∈ {2, 3, 4} — selected per pair on val accuracy (seed 0).

**Feature-KD:** α ∈ {0.3, 0.5, 0.7}, β = 0.5 — selected per pair on val accuracy (seed 0).

Selected configurations are recorded in [`results/best_configs.json`](results/best_configs.json).

### Architecture

All models use a **CIFAR-specific stem**: the standard ResNet `conv1` (kernel=7, stride=2) and MaxPool are replaced with `conv1` (kernel=3, stride=1) and Identity. This preserves the full 32×32 spatial resolution through the first residual block and is critical for effective distillation on small inputs.

| Model | Params | Block type |
|-------|--------|------------|
| ResNet-101 (teacher) | 42.5M | Bottleneck |
| ResNet-50 (teacher) | 23.5M | Bottleneck |
| ResNet-34 (teacher / student) | 21.3M | BasicBlock |
| ResNet-18 (student) | 11.2M | BasicBlock |

Parameter counts are for the CIFAR-stem, 10-class models as built by
`src/models/resnet.py`.

---

## Implementation Notes

### Gradient Clipping Bug

Clipping is applied to the **union** of student model parameters and Feature-KD projection layer parameters:

```python
params = list(student.parameters()) + list(projection_layers.parameters())
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
```

v1 of this work excluded the projection layers from clipping and blamed that
for Feature-KD's underperformance. The v2 controlled re-run
(`tools/run_bug_ablation.sh`, [`results/bug_ablation.csv`](results/bug_ablation.csv))
does not support that: bugged 95.00 ± 0.18 vs corrected 94.95 ± 0.20 (p=0.69),
with unclipped projection norms ≤ 0.21 against a threshold of 1.0. The union
clipping is kept because it is the correct behaviour, not because it changes
the result.

### Teacher Loading

Missing teacher weights raise a hard `ValueError`. Silent fallback to a random teacher is not permitted and will corrupt all downstream results.

---

## Quick Start

```bash
git clone https://github.com/umutonuryasar/kd-capacity-gap.git
cd kd-capacity-gap
pip install -r requirements.txt
```

**Train all teachers:**
```bash
bash tools/train_teachers.sh
```

**Run a single experiment (R50→R34, Feature-KD, seed 0):**
```bash
python tools/train.py \
  --config configs/r50_to_r34_feature.yaml \
  --seed 0
```

**Run the selection grid (Stage 1, seed 0) across all pairs and methods:**
```bash
bash tools/run_ablation.sh
```

All eight configs follow the same naming convention:

| Config file | Pair | Method |
|-------------|------|--------|
| `r34_to_r18_logit.yaml` | R34→R18 | Logit-KD |
| `r34_to_r18_feature.yaml` | R34→R18 | Feature-KD |
| `r50_to_r18_logit.yaml` | R50→R18 | Logit-KD |
| `r50_to_r18_feature.yaml` | R50→R18 | Feature-KD |
| `r50_to_r34_logit.yaml` | R50→R34 | Logit-KD |
| `r50_to_r34_feature.yaml` | R50→R34 | Feature-KD |
| `r101_to_r34_logit.yaml` | R101→R34 | Logit-KD |
| `r101_to_r34_feature.yaml` | R101→R34 | Feature-KD |

For the full two-stage protocol behind the reported numbers, see
[Reproducing](#reproducing).

---

## Citation

If you use this code or build on these findings, please cite:

```bibtex
@misc{yasar2026kd,
  title   = {Student Capacity Moderates Knowledge Distillation Effectiveness:
             A Systematic Study Across ResNet Teacher-Student Pairs on CIFAR-10},
  author  = {Ya{\c{s}}ar, Umut Onur},
  year    = {2026},
  eprint  = {2605.31191},
  archivePrefix = {arXiv}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
