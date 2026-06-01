#!/usr/bin/env python3
"""Analyze gain modulation (plateau vs somatic spikes) for a trained no_history SHD model.

Loads a checkpoint, runs inference (no dropout) on train and test with N samples per class,
and prints a terminal summary of hidden-layer spike rates, plateau occupancy, and
time-normalized spike rates inside vs outside plateau (same-timestep h).

Example (use the dendroprop conda env)::

    conda run -n dendroprop python analyze_gain_modulation.py
    conda run -n dendroprop python analyze_gain_modulation.py --checkpoint outputs/shd_run1 --samples_per_class 10
"""
from __future__ import annotations

import argparse
import os
import sys

import jax

jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import random
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data.shd_binned import load_shd_binned
from checkpoint import load_checkpoint
from two_comp_neuron import TwoCompNeuron
from lif_neuron import LINeuron


def parse_args():
    p = argparse.ArgumentParser(
        description="Gain-modulation / plateau activity analysis for a trained SHD checkpoint.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "outputs", "shd_run1"),
        help="Checkpoint path prefix (without .npz); default: outputs/shd_run1",
    )
    p.add_argument("--samples_per_class", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_forward_collect_hidden(net, x_input):
    """Forward pass; return hidden spikes (T, N) and plateau flags (T, N)."""
    config = net.config
    h_layer = net.hidden
    r = net.readout
    carry = h_layer.init_carry()
    r_carry = r.init_carry()
    T = int(x_input.shape[0])
    ho_list, h_list = [], []
    for t in range(T):
        x_t = x_input[t]
        dend_in = x_t @ h_layer.w_dend.T
        soma_in = x_t @ h_layer.w_soma.T
        carry, h_o, _h_v_pre, h_h, _h_h_prev, _mu_at_tp = TwoCompNeuron.forward_step(
            carry,
            dend_in,
            soma_in,
            t,
            h_layer.alpha_s,
            h_layer.alpha_d,
            h_layer.T_p,
            config,
        )
        hidden_o_float = h_o.astype(jnp.float32)
        r_carry, _r_v, _r_E = LINeuron.forward_step(
            r_carry,
            hidden_o_float,
            r.w,
            r.alpha_m,
        )
        ho_list.append(np.asarray(h_o, dtype=np.float32))
        h_list.append(np.asarray(h_h, dtype=np.float32))
    return np.stack(ho_list), np.stack(h_list)


def analyze_split(net, X, y, n_classes: int):
    """Aggregate metrics for one dataset split."""
    n_samples, T, n_hidden = X.shape[0], X.shape[1], net.n_hidden
    labels = np.asarray(y, dtype=np.int32)

    ho_all = np.zeros((n_samples, T, n_hidden), dtype=np.float32)
    h_all = np.zeros((n_samples, T, n_hidden), dtype=np.float32)
    for i in range(n_samples):
        ho, h = run_forward_collect_hidden(net, jnp.asarray(X[i]))
        ho_all[i] = ho
        h_all[i] = h

    neuron_steps = float(n_samples * T)

    # (N,) mean hidden spike rate over all samples and time
    spike_rate_per_neuron = ho_all.sum(axis=(0, 1)) / neuron_steps

    # (C, N) mean spike rate per class per neuron
    spike_rate_per_class_neuron = np.zeros((n_classes, n_hidden), dtype=np.float64)
    for c in range(n_classes):
        mask = labels == c
        if not np.any(mask):
            continue
        spike_rate_per_class_neuron[c] = ho_all[mask].mean(axis=(0, 1))

    # (N,) plateau occupancy (fraction of neuron-steps with h=1)
    plateau_occupancy_per_neuron = h_all.mean(axis=(0, 1))

    spikes_in_plat = float((ho_all * h_all).sum())
    spikes_out_plat = float((ho_all * (1.0 - h_all)).sum())
    steps_in_plat = float(h_all.sum())
    steps_out_plat = float((1.0 - h_all).sum())

    ho_in = ho_all * h_all
    ho_out = ho_all * (1.0 - h_all)
    h_sum = h_all.sum(axis=(0, 1))
    h_out_sum = (1.0 - h_all).sum(axis=(0, 1))

    rate_in_per_neuron = np.divide(
        ho_in.sum(axis=(0, 1)),
        h_sum,
        out=np.zeros(n_hidden, dtype=np.float64),
        where=h_sum > 0,
    )
    rate_out_per_neuron = np.divide(
        ho_out.sum(axis=(0, 1)),
        h_out_sum,
        out=np.zeros(n_hidden, dtype=np.float64),
        where=h_out_sum > 0,
    )
    plateau_spike_ratio_per_neuron = np.divide(
        rate_in_per_neuron,
        rate_out_per_neuron,
        out=np.full(n_hidden, np.nan, dtype=np.float64),
        where=rate_out_per_neuron > 0,
    )

    global_rate_in = spikes_in_plat / max(steps_in_plat, 1.0)
    global_rate_out = spikes_out_plat / max(steps_out_plat, 1.0)
    global_plateau_ratio = global_rate_in / max(global_rate_out, 1e-12)

    pred_correct = sum(int(net.predict(jnp.asarray(X[i])) == int(labels[i])) for i in range(n_samples))
    acc = 100.0 * pred_correct / max(n_samples, 1)

    return {
        "n_samples": n_samples,
        "T": T,
        "n_hidden": n_hidden,
        "accuracy_pct": acc,
        "spike_rate_per_neuron": spike_rate_per_neuron,
        "spike_rate_per_class_neuron": spike_rate_per_class_neuron,
        "plateau_occupancy_per_neuron": plateau_occupancy_per_neuron,
        "spikes_in_plat": spikes_in_plat,
        "spikes_out_plat": spikes_out_plat,
        "steps_in_plat": steps_in_plat,
        "steps_out_plat": steps_out_plat,
        "global_rate_in_plat": global_rate_in,
        "global_rate_out_plat": global_rate_out,
        "global_plateau_spike_ratio": global_plateau_ratio,
        "plateau_spike_ratio_per_neuron": plateau_spike_ratio_per_neuron,
    }


def _dist_summary(x: np.ndarray, name: str, indent: str = "  ") -> None:
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        print(f"{indent}{name}: (no finite values)")
        return
    print(
        f"{indent}{name}: mean={finite.mean():.6f}  median={np.median(finite):.6f}  "
        f"min={finite.min():.6f}  max={finite.max():.6f}  "
        f"(n_finite={finite.size}/{x.size})",
    )


def _print_class_curve(curve: np.ndarray, split_name: str, indent: str = "    ") -> None:
    """Per-class mean spike rate averaged over neurons (20-point curve summary)."""
    mean_over_neurons = curve.mean(axis=1)
    print(f"{indent}{split_name} mean spike rate per class (avg over neurons):")
    parts = [f"{v:.5f}" for v in mean_over_neurons]
    print(f"{indent}  " + " ".join(f"c{i}={parts[i]}" for i in range(len(parts))))


def print_split_summary(stats: dict, split_name: str) -> None:
    print(f"\n=== {split_name} ({stats['n_samples']} samples, T={stats['T']}, "
          f"N_hidden={stats['n_hidden']}) ===")
    print(f"  Accuracy: {stats['accuracy_pct']:.2f}%")
    print("  Hidden spike rate (all neuron-steps):")
    _dist_summary(stats["spike_rate_per_neuron"], "per neuron")
    print("  Plateau occupancy (fraction of time h=1):")
    _dist_summary(stats["plateau_occupancy_per_neuron"], "per neuron")
    print("  Spike counts (same-timestep h):")
    print(f"    in plateau:  {stats['spikes_in_plat']:.0f} spikes over "
          f"{stats['steps_in_plat']:.0f} neuron-steps")
    print(f"    out plateau: {stats['spikes_out_plat']:.0f} spikes over "
          f"{stats['steps_out_plat']:.0f} neuron-steps")
    print("  Time-normalized spike rates:")
    print(f"    rate in plateau:  {stats['global_rate_in_plat']:.6f}")
    print(f"    rate out plateau: {stats['global_rate_out_plat']:.6f}")
    print(f"    ratio (in/out):   {stats['global_plateau_spike_ratio']:.4f}x")
    print("  Per-neuron plateau spike ratio (rate_in / rate_out):")
    _dist_summary(stats["plateau_spike_ratio_per_neuron"], "per neuron")
    _print_class_curve(stats["spike_rate_per_class_neuron"], split_name)


def print_train_test_comparison(train: dict, test: dict) -> None:
    print("\n=== Train vs test (delta: test - train) ===")
    for key, label in (
        ("spike_rate_per_neuron", "mean hidden spike rate (per neuron)"),
        ("plateau_occupancy_per_neuron", "plateau occupancy (per neuron)"),
        ("plateau_spike_ratio_per_neuron", "plateau spike ratio (per neuron)"),
    ):
        a = train[key]
        b = test[key]
        delta = b - a
        _dist_summary(delta, f"{label} delta")
    print(f"  Accuracy: train={train['accuracy_pct']:.2f}%  test={test['accuracy_pct']:.2f}%  "
          f"delta={test['accuracy_pct'] - train['accuracy_pct']:+.2f} pp")
    print(f"  Global plateau spike ratio: train={train['global_plateau_spike_ratio']:.4f}x  "
          f"test={test['global_plateau_spike_ratio']:.4f}x  "
          f"delta={test['global_plateau_spike_ratio'] - train['global_plateau_spike_ratio']:+.4f}x")
    tr_curve = train["spike_rate_per_class_neuron"].mean(axis=1)
    te_curve = test["spike_rate_per_class_neuron"].mean(axis=1)
    print("  Per-class mean spike rate (avg over neurons), test - train:")
    parts = [f"c{i}={te_curve[i] - tr_curve[i]:+.5f}" for i in range(len(tr_curve))]
    print("    " + " ".join(parts))


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    net, meta = load_checkpoint(args.checkpoint, key=random.PRNGKey(args.seed), dropout_rate=0.0)
    extra = meta.get("extra", {})
    data_cfg = extra.get("data", {})
    bin_size_ms = float(data_cfg.get("bin_size_ms", meta["config"]["dt"]))
    collapse_factor = int(data_cfg.get("collapse_factor", 5))
    max_duration_ms = float(data_cfg.get("max_duration_ms", 1400.0))
    binarize = bool(data_cfg.get("binarize", False))
    input_scale = float(data_cfg.get("input_scale", 1.0))
    n_classes = int(meta["n_outputs"])
    n_per = args.samples_per_class

    print(
        f"Data: bin={bin_size_ms}ms collapse={collapse_factor} "
        f"max_duration={max_duration_ms}ms binarize={binarize} "
        f"samples_per_class={n_per} (train+test)  precision=float32",
        flush=True,
    )
    print(
        f"Model: {meta['n_inputs']} -> {meta['n_hidden']} hidden, "
        f"gamma={meta['config']['gamma']} v_th={meta['config']['v_th']} "
        f"mu_th={meta['config']['mu_th']}",
        flush=True,
    )

    print("Loading SHD...", flush=True)
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

    print("Analyzing train split...", flush=True)
    train_stats = analyze_split(net, X_tr, y_tr, n_classes)
    print("Analyzing test split...", flush=True)
    test_stats = analyze_split(net, X_te, y_te, n_classes)

    print_split_summary(train_stats, "Train")
    print_split_summary(test_stats, "Test")
    print_train_test_comparison(train_stats, test_stats)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
