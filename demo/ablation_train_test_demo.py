#!/usr/bin/env python3
"""Train vs test accuracy for SHD checkpoint under several forward ablations.

Uses ``no_history/analyze_correctness_ablation.py`` (same forward logic as that script)
and plots a grouped bar chart: each ablation mode has train and test bars.

Example (dendroprop env, from repo root)::

    conda run -n dendroprop python demo/ablation_train_test_demo.py
    conda run -n dendroprop python demo/ablation_train_test_demo.py \\
        --checkpoint no_history/outputs/shd_run1 --samples_per_class 10

Outputs (in ``demo/`` by default):
  ablation_train_test.png, ablation_train_test.svg
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
_NO_HISTORY = os.path.join(_ROOT, "no_history")
for _p in (_NO_HISTORY, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax

jax.config.update("jax_enable_x64", False)
from jax import random

from data.shd_binned import load_shd_binned
from checkpoint import load_checkpoint
from analyze_correctness_ablation import (
    DEMO_ABLATION_LABELS,
    DEMO_ABLATION_ORDER,
    Ablation,
    evaluate_split_accuracy,
)


def resolve_checkpoint_base(path: str) -> str:
    path = os.path.expanduser(path)
    rel = path[:-4] if path.endswith(".npz") else path
    candidates = [rel]
    if not os.path.isabs(rel):
        candidates.extend([os.path.join(_ROOT, rel), os.path.join(_NO_HISTORY, rel)])
    for base in candidates:
        if os.path.isfile(base + ".npz") and os.path.isfile(base + ".meta.json"):
            return base
    raise FileNotFoundError(f"Checkpoint not found for prefix: {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train/test ablation accuracy bar chart.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("no_history", "outputs", "shd_run1"),
        help="Checkpoint prefix (no .npz suffix).",
    )
    p.add_argument("--samples_per_class", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "ablation_train_test"),
        help="Output path without extension (.png and .svg written).",
    )
    p.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Comma-separated ablation modes (default: demo subset).",
    )
    return p.parse_args()


def plot_results(
    modes: Tuple[Ablation, ...],
    train_acc: np.ndarray,
    test_acc: np.ndarray,
    out_base: str,
    checkpoint_label: str,
    n_per_class: int,
) -> None:
    labels = [DEMO_ABLATION_LABELS.get(m, m) for m in modes]
    x = np.arange(len(modes))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(8, len(modes) * 1.5), 5))
    bars_train = ax.bar(x - width / 2, train_acc, width, label="Train", color="#4C72B0")
    bars_test = ax.bar(x + width / 2, test_acc, width, label="Test", color="#DD8452")

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Forward mode")
    ax.set_title(
        f"SHD accuracy by ablation (train vs test)\n"
        f"{checkpoint_label}  |  {n_per_class} samples/class  |  dropout off at eval"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.axhline(100.0 / 20.0, color="gray", linestyle=":", linewidth=1, label="Chance (5%)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    for bars in (bars_train, bars_test):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    png_path = out_base + ".png"
    svg_path = out_base + ".svg"
    fig.savefig(png_path, dpi=150)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Wrote {png_path}\nWrote {svg_path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    ckpt_base = resolve_checkpoint_base(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_base}", flush=True)
    net, meta = load_checkpoint(ckpt_base, key=random.PRNGKey(args.seed), dropout_rate=0.0)

    extra = meta.get("extra", {})
    data_cfg = extra.get("data", {})
    bin_size_ms = float(data_cfg.get("bin_size_ms", meta["config"]["dt"]))
    collapse_factor = int(data_cfg.get("collapse_factor", 5))
    max_duration_ms = float(data_cfg.get("max_duration_ms", 1400.0))
    binarize = bool(data_cfg.get("binarize", False))
    input_scale = float(data_cfg.get("input_scale", 1.0))
    n_per = args.samples_per_class

    if args.modes:
        modes = tuple(m.strip() for m in args.modes.split(","))
    else:
        modes = DEMO_ABLATION_ORDER

    print(
        f"Data: bin={bin_size_ms}ms collapse={collapse_factor} "
        f"samples_per_class={n_per}",
        flush=True,
    )
    print("Loading SHD train + test...", flush=True)
    X_tr, y_tr, _, X_te, y_te, _ = load_shd_binned(
        bin_size_ms=bin_size_ms,
        collapse_factor=collapse_factor,
        max_duration_ms=max_duration_ms,
        train_samples_per_class=n_per,
        test_samples_per_class=n_per,
        binarize=binarize,
        dtype=np.float32,
    )
    if input_scale != 1.0:
        X_tr = X_tr * input_scale
        X_te = X_te * input_scale

    train_acc = np.zeros(len(modes))
    test_acc = np.zeros(len(modes))

    print(f"\n{'mode':<14}  {'train':>8}  {'test':>8}")
    print("-" * 34)
    for i, mode in enumerate(modes):
        print(f"Evaluating '{mode}' on train ({X_tr.shape[0]} trials)...", flush=True)
        train_acc[i] = evaluate_split_accuracy(net, X_tr, y_tr, ablation=mode)
        print(f"Evaluating '{mode}' on test ({X_te.shape[0]} trials)...", flush=True)
        test_acc[i] = evaluate_split_accuracy(net, X_te, y_te, ablation=mode)
        label = DEMO_ABLATION_LABELS.get(mode, mode)
        print(f"{label:<14}  {train_acc[i]:7.2f}%  {test_acc[i]:7.2f}%")

    out_base = args.out
    if not out_base.endswith((".png", ".svg")):
        plot_base = out_base
    else:
        plot_base = os.path.splitext(out_base)[0]

    plot_results(
        modes,
        train_acc,
        test_acc,
        plot_base,
        os.path.basename(ckpt_base),
        n_per,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
