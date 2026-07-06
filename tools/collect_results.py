#!/usr/bin/env python3
"""Aggregate results.json files under runs/ into mean +- std tables.

Two modes:
  1. Default: aggregate across seeds, grouped by (pair, kd_type, alpha, T).
     Reports val acc (selection metric) and test acc (reporting metric)
     side by side, so selection and reporting stay auditable.
  2. --write-best: additionally select the best config per (pair, kd_type)
     BY VAL ACCURACY and write it to best_configs.json, which
     tools/run_final.sh consumes for the 5-seed final runs.

Usage:
  python tools/collect_results.py runs/
  python tools/collect_results.py runs/ --write-best best_configs.json
"""

import sys
import json
import argparse
import statistics
from pathlib import Path
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate KD results")
    p.add_argument("root", nargs="?", default="runs")
    p.add_argument("--write-best", default=None, metavar="PATH",
                   help="Write best-by-val config per (pair, kd_type) to this JSON.")
    p.add_argument("--csv", default=None, metavar="PATH",
                   help="Also write the full aggregated table as CSV.")
    return p.parse_args()


def fmt(mean: float, std: float | None) -> str:
    if std is None:
        return f"{100*mean:.2f}"
    return f"{100*mean:.2f} ± {100*std:.2f}"


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    runs = []
    for rj in sorted(root.rglob("results.json")):
        with open(rj) as f:
            r = json.load(f)
        r.pop("history", None)
        r["_path"] = str(rj.parent)
        runs.append(r)

    if not runs:
        sys.exit(f"No results.json found under {root}")

    # Group by everything except seed
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        kd = r.get("kd_type", "none")
        key = (
            r.get("teacher") or "-",
            r.get("model"),
            kd,
            r.get("alpha") if kd != "none" else None,
            r.get("temperature") if kd == "logit" else None,
            r.get("stem", "cifar"),
        )
        groups[key].append(r)

    header = ["Teacher", "Student", "KD", "alpha", "T", "Stem",
              "Seeds", "Val acc (%)", "Test@best (%)", "Test@final (%)"]
    rows = []
    for key, rs in sorted(groups.items()):
        teacher, student, kd, alpha, temp, stem = key
        seeds = sorted(r["seed"] for r in rs)

        def agg(field):
            vals = [r[field] for r in rs]
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else None
            return m, s

        val_m,  val_s  = agg("best_val_acc")
        tb_m,   tb_s   = agg("test_acc_best")
        tf_m,   tf_s   = agg("test_acc_final")

        rows.append({
            "key": key, "n_seeds": len(seeds), "seeds": seeds,
            "val_mean": val_m,
            "cells": [
                teacher, student, kd,
                "-" if alpha is None else f"{alpha:g}",
                "-" if temp  is None else f"{temp:g}",
                stem, f"{len(seeds)} {seeds}",
                fmt(val_m, val_s), fmt(tb_m, tb_s), fmt(tf_m, tf_s),
            ],
        })

    widths = [max(len(header[i]), max(len(r["cells"][i]) for r in rows)) for i in range(len(header))]
    line = " | ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-|-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(r["cells"], widths)))

    if args.csv:
        with open(args.csv, "w") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(c) for c in r["cells"]) + "\n")
        print(f"\nWrote {args.csv}")

    if args.write_best:
        # Best config per (teacher, student, kd_type), selected by VAL accuracy.
        best: dict[str, dict] = {}
        for r in rows:
            teacher, student, kd, alpha, temp, stem = r["key"]
            if kd == "none" or stem != "cifar":
                continue
            name = f"{teacher}_to_{student}/{kd}"
            if name not in best or r["val_mean"] > best[name]["val_mean"]:
                best[name] = {
                    "teacher": teacher, "student": student, "kd_type": kd,
                    "alpha": float(alpha),
                    "temperature": None if temp in (None, "-") else float(temp),
                    "val_mean": r["val_mean"],
                    "selected_from_seeds": r["seeds"],
                }
        with open(args.write_best, "w") as f:
            json.dump(best, f, indent=2)
        print(f"\nBest-by-val configs written to {args.write_best}:")
        for name, b in best.items():
            t = f", T={b['temperature']:g}" if b["temperature"] else ""
            print(f"  {name}: alpha={b['alpha']:g}{t}  (val {100*b['val_mean']:.2f}%)")


if __name__ == "__main__":
    main()
