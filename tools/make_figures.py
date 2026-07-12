#!/usr/bin/env python3
"""Regenerate paper figures from runs/final/.

Usage:
  python tools/make_figures.py [--runs runs] [--out figures]

Figure 1: KD gain over baseline per pair, 5 seeds, error bars = SE of the
          difference vs. baseline.
Figure 2: validation-accuracy convergence curves per pair (mean ± std over
          seeds, EMA-smoothed, span=9), with teacher/baseline reference lines.
"""

import json, glob, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIRS = ["R34\u2192R18", "R50\u2192R18", "R50\u2192R34", "R101\u2192R34"]
DIRS = {
    "R34\u2192R18":  ("resnet34_to_resnet18",  "resnet18", "resnet34"),
    "R50\u2192R18":  ("resnet50_to_resnet18",  "resnet18", "resnet50"),
    "R50\u2192R34":  ("resnet50_to_resnet34",  "resnet34", "resnet50"),
    "R101\u2192R34": ("resnet101_to_resnet34", "resnet34", "resnet101"),
}
COLORS = {"logit": "#4878A8", "feature": "#2E8B57"}


def load(pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(pattern, recursive=True))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    R = args.runs
    Path(args.out).mkdir(exist_ok=True)

    def accs(d):
        return np.array([r["test_acc_best"] * 100
                         for r in load(f"{R}/final/{d}/seed*/results.json")])

    base = {s: accs(f"baseline/{s}") for s in ("resnet18", "resnet34")}
    teacher_acc = {}
    for arch, short in (("resnet34", "r34"), ("resnet50", "r50"), ("resnet101", "r101")):
        rs = load(f"{R}/teachers/{short}/seed*/results.json")
        if rs:
            teacher_acc[arch] = float(np.mean([r["test_acc_best"] * 100 for r in rs]))
    # fall back to paper values if teacher runs absent from this tree
    teacher_acc = {**{"resnet34": 95.30, "resnet50": 95.36, "resnet101": 95.37},
                   **teacher_acc}

    # ---------- Figure 1 ----------
    fig, ax = plt.subplots(figsize=(8.2, 5))
    width, x = 0.36, np.arange(len(PAIRS))
    for j, kd in enumerate(("logit", "feature")):
        means, errs = [], []
        for p in PAIRS:
            d, s, _ = DIRS[p]
            a, b = accs(f"{d}_{kd}"), base[s]
            means.append(a.mean() - b.mean())
            errs.append(np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b)))
        ax.bar(x + (j - 0.5) * width, means, width, yerr=errs, capsize=4,
               label=f"{'Logit' if kd == 'logit' else 'Feature'}-KD",
               color=COLORS[kd], error_kw=dict(lw=1.2))
        for xi, m, e in zip(x + (j - 0.5) * width, means, errs):
            ax.text(xi, m + e + 0.012 if m >= 0 else m - e - 0.03,
                    f"{m:+.2f}", ha="center", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(PAIRS)
    ax.set_ylabel("KD Gain over Baseline (pp)")
    ax.set_title("KD Gain by Teacher-Student Capacity Pair (5 seeds, ±SE of difference)")
    ax.legend(); ax.set_ylim(-0.22, 0.38)
    plt.tight_layout()
    f1 = Path(args.out) / "figure1_capacity_gap.png"
    plt.savefig(f1, dpi=200); plt.close()
    print(f"wrote {f1}")

    # ---------- Figure 2 ----------
    def curves(d):
        hs = [r["history"] for r in load(f"{R}/final/{d}/seed*/results.json")]
        va = np.array([[h["val_acc"] for h in hist] for hist in hs])
        a_ = 2 / (9 + 1)
        sm = np.copy(va)
        for i in range(1, sm.shape[1]):
            sm[:, i] = a_ * va[:, i] + (1 - a_) * sm[:, i - 1]
        return sm.mean(0), sm.std(0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, p in zip(axes, PAIRS):
        d, s, t = DIRS[p]
        for kd in ("logit", "feature"):
            m, sd = curves(f"{d}_{kd}")
            ep = np.arange(1, len(m) + 1)
            ax.plot(ep, m, color=COLORS[kd], lw=1.4,
                    label=f"{'Logit' if kd == 'logit' else 'Feature'}-KD")
            ax.fill_between(ep, m - sd, m + sd, color=COLORS[kd], alpha=0.18)
        ax.axhline(teacher_acc[t] / 100, ls=":", color="black", lw=1.1,
                   label=f"Teacher ({teacher_acc[t]:.2f}%)")
        ax.axhline(base[s].mean() / 100, ls="-.", color="gray", lw=1.1,
                   label=f"Baseline ({base[s].mean():.2f}%)")
        ax.set_title(p); ax.set_xlabel("Epoch"); ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, loc="lower right")
    axes[0].set_ylabel("Validation Accuracy")
    fig.suptitle("Convergence Curves by Teacher-Student Capacity Pair "
                 "(mean ± std across 5 seeds, smoothed)", y=1.02)
    plt.tight_layout()
    f2 = Path(args.out) / "figure2_convergence.png"
    plt.savefig(f2, dpi=200, bbox_inches="tight"); plt.close()
    print(f"wrote {f2}")


if __name__ == "__main__":
    main()
