# Student Capacity Moderates Knowledge Distillation Effectiveness
### A Systematic Study Across ResNet Teacher-Student Pairs on CIFAR-10

[![arXiv](https://img.shields.io/badge/arXiv-2605.31191-b31b1b.svg)](https://arxiv.org/abs/2605.31191)
[![HF Spaces](https://img.shields.io/badge/HF%20Spaces-Demo-orange.svg)](https://huggingface.co/spaces/umutonuryasar/kd-capacity-gap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)]()

---

## Overview

This repository contains the full code and results for a systematic study of knowledge distillation (KD) effectiveness across three ResNet teacher-student capacity pairs on CIFAR-10. We compare Logit-KD and Feature-KD under controlled, reproducible conditions (3 seeds, mean ± std reported throughout) and identify two key findings:

1. **Student capacity**, not raw teacher-student accuracy gap, is the key moderating factor in KD effectiveness — R34 students consistently benefit more from distillation than R18 students, even when gap magnitudes are comparable.
2. **Implementation correctness** critically affects Feature-KD: an unclipped projection-layer gradient suppresses Feature-KD performance and produces misleading comparisons with Logit-KD. After correction, Feature-KD matches or outperforms Logit-KD in two of three pairs.

## Demo

Interactive results explorer available on HF Spaces: [huggingface.co/spaces/umutonuryasar/kd-capacity-gap](https://huggingface.co/spaces/umutonuryasar/kd-capacity-gap)

---

## Key Results

| Teacher | Student | T-S Gap | Logit-KD Δ | Feature-KD Δ | Best |
|---------|---------|---------|------------|--------------|------|
| R34 (95.70%) | R18 (95.13%) | 0.57 pp | +0.00 pp | +0.18 pp | Feature |
| R50 (95.81%) | R18 (95.13%) | 0.68 pp | +0.21 pp | +0.08 pp | Logit |
| R50 (95.81%) | R34 (95.25%) | 0.56 pp | +0.21 pp | **+0.30 pp** | Feature |

All gains relative to the corresponding student baseline. Results reported as mean ± std across seeds {0, 1, 2}.

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
│   └── r50_to_r34_feature.yaml
├── notebooks/
│   └── run_experiments.ipynb # Result aggregation and plotting
├── src/
│   ├── distillation/         # Logit-KD and Feature-KD loss implementations
│   ├── losses/
│   ├── models/               # ResNet definitions with CIFAR-specific stem
│   ├── utils/
│   ├── __init__.py
│   └── trainer.py
├── tools/
│   ├── train.py              # Single-run training entry point
│   ├── train_teachers.sh     # Train all teacher models
│   ├── run_ablation.sh       # Run full ablation across seeds and configs
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

Seeds used: **{0, 1, 2}**. Results in the paper are mean ± std across all three seeds.

---

## Experimental Setup

| Hyperparameter | Value |
|----------------|-------|
| Dataset | CIFAR-10 (50k train / 10k test) |
| Input resolution | 32×32 |
| Optimizer | SGD (momentum=0.9, weight_decay=5e-4, Nesterov=True) |
| Learning rate | 0.1 with CosineAnnealingLR (T_max=100, η_min=1e-4) |
| Batch size | 128 |
| Student epochs | 100 |
| Teacher epochs | 200 |
| Hardware | NVIDIA A100-SXM4-40GB |

### KD Hyperparameter Grid

**Logit-KD:** α ∈ {0.3, 0.5, 0.7}, T ∈ {2, 3, 4} — best per pair reported.

**Feature-KD:** α ∈ {0.3, 0.5, 0.7}, β = 0.5 — best per pair reported.

### Architecture

All models use a **CIFAR-specific stem**: the standard ResNet `conv1` (kernel=7, stride=2) and MaxPool are replaced with `conv1` (kernel=3, stride=1) and Identity. This preserves the full 32×32 spatial resolution through the first residual block and is critical for effective distillation on small inputs.

| Model | Params | Block type |
|-------|--------|------------|
| ResNet-50 (teacher) | 23.5M | Bottleneck |
| ResNet-34 (teacher / student) | 21.8M | BasicBlock |
| ResNet-18 (student) | 11.2M | BasicBlock |

---

## Implementation Notes

### Gradient Clipping Bug

A common implementation oversight in Feature-KD is excluding projection layer parameters from gradient clipping. In this repo, clipping is applied to the **union** of student model parameters and Feature-KD projection layer parameters:

```python
params = list(student.parameters()) + list(projection_layers.parameters())
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
```

Excluding projection layers leads to unclipped gradient norms of up to **4.65** in early training, suppressing Feature-KD performance and producing misleading Logit-KD vs. Feature-KD comparisons. See Table 3 in the paper for a quantitative comparison.

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

**Run full ablation across all pairs, methods, and seeds:**
```bash
bash tools/run_ablation.sh
```

All six configs follow the same naming convention:

| Config file | Pair | Method |
|-------------|------|--------|
| `r34_to_r18_logit.yaml` | R34→R18 | Logit-KD |
| `r34_to_r18_feature.yaml` | R34→R18 | Feature-KD |
| `r50_to_r18_logit.yaml` | R50→R18 | Logit-KD |
| `r50_to_r18_feature.yaml` | R50→R18 | Feature-KD |
| `r50_to_r34_logit.yaml` | R50→R34 | Logit-KD |
| `r50_to_r34_feature.yaml` | R50→R34 | Feature-KD |

To reproduce full results across all seeds and pairs:
```bash
bash tools/run_all.sh
```

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
