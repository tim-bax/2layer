#!/usr/bin/env python3
"""Test-set analysis: correct vs incorrect trials, and plateau ablations.

1. Correct vs incorrect (test only): hidden spike rate, plateau occupancy,
   time-normalized in/out plateau spike ratio, readout voltage totals and margin.

2. Plateau necessity ablations (accuracy on test):
   - none: normal dynamics
   - gamma0: natural h, gamma=0 on soma threshold (no gain modulation, natural plateaus)
   - h0_soma / h1_soma: natural dendrite/carry; force h=0/1 for soma threshold only
   - h0 / h1: force h=0/1 in carry and soma (full clamp)
   - no_dend_in: zero w_dend @ x

Example (dendroprop env)::

    conda run -n dendroprop python analyze_correctness_ablation.py
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Literal, Tuple

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

Ablation = Literal[
    "none",
    "gamma0",
    "h0_soma",
    "h1_soma",
    "h0",
    "h1",
    "no_dend_in",
    "no_soma_in",
]

ABLATION_ORDER: Tuple[Ablation, ...] = (
    "none",
    "gamma0",
    "h0_soma",
    "h1_soma",
    "h0",
    "h1",
    "no_dend_in",
    "no_soma_in",
)

# Subset for train/test accuracy demos (grouped bar charts).
DEMO_ABLATION_ORDER: Tuple[Ablation, ...] = (
    "none",
    "h0",
    "h1",
    "no_dend_in",
    "no_soma_in",
)

DEMO_ABLATION_LABELS: Dict[Ablation, str] = {
    "none": "Regular",
    "h0": "Force h=0",
    "h1": "Force h=1",
    "no_dend_in": "No dend in",
    "no_soma_in": "No soma in",
    "gamma0": "gamma=0",
    "h0_soma": "h0 soma only",
    "h1_soma": "h1 soma only",
}


def parse_args():
    p = argparse.ArgumentParser(description="Correct vs incorrect trials and plateau ablations (test).")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "outputs", "shd_run1"),
    )
    p.add_argument("--samples_per_class", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _soma_spike(
    v_pre: jnp.ndarray,
    h_for_threshold: jnp.ndarray,
    gamma: float,
    v_th: float,
) -> jnp.ndarray:
    return jnp.where(
        v_pre >= v_th - gamma * h_for_threshold.astype(v_pre.dtype),
        1,
        0,
    ).astype(jnp.int32)


def forward_trial(
    net,
    x_input: jnp.ndarray,
    ablation: Ablation = "none",
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    """Run one trial; return pred, ho, h, readout_v_avg, margin, hidden_rate, plat_occ."""
    config = net.config
    h_layer = net.hidden
    r = net.readout
    carry = h_layer.init_carry()
    r_carry = r.init_carry()
    T = int(x_input.shape[0])
    n_hidden = net.n_hidden
    ho_list: List[np.ndarray] = []
    h_list: List[np.ndarray] = []

    for t in range(T):
        x_t = x_input[t]
        dend_in = x_t @ h_layer.w_dend.T
        soma_in = x_t @ h_layer.w_soma.T
        if ablation == "no_dend_in":
            dend_in = jnp.zeros_like(dend_in)
        if ablation == "no_soma_in":
            soma_in = jnp.zeros_like(soma_in)

        carry, h_o, h_v_pre, h_h, _h_h_prev, _mu_at_tp = TwoCompNeuron.forward_step(
            carry,
            dend_in,
            soma_in,
            t,
            h_layer.alpha_s,
            h_layer.alpha_d,
            h_layer.T_p,
            config,
        )

        h_record = h_h
        gamma_eff = float(config.gamma)

        if ablation in ("none", "no_dend_in", "no_soma_in"):
            pass
        elif ablation == "gamma0":
            gamma_eff = 0.0
            h_o = _soma_spike(h_v_pre, h_h, gamma_eff, config.v_th)
            mu, v, h, t_prime, mu_at_tp, E_soma, dmu_dw, dmu_atp = carry
            carry = (mu, h_v_pre * (1 - h_o), h, t_prime, mu_at_tp, E_soma, dmu_dw, dmu_atp)
        elif ablation == "h0_soma":
            h_o = _soma_spike(h_v_pre, jnp.zeros(n_hidden, dtype=jnp.int32), gamma_eff, config.v_th)
            mu, v, h, t_prime, mu_at_tp, E_soma, dmu_dw, dmu_atp = carry
            carry = (mu, h_v_pre * (1 - h_o), h, t_prime, mu_at_tp, E_soma, dmu_dw, dmu_atp)
        elif ablation == "h1_soma":
            h_o = _soma_spike(h_v_pre, jnp.ones(n_hidden, dtype=jnp.int32), gamma_eff, config.v_th)
            mu, v, h, t_prime, mu_at_tp, E_soma, dmu_dw, dmu_atp = carry
            carry = (mu, h_v_pre * (1 - h_o), h, t_prime, mu_at_tp, E_soma, dmu_dw, dmu_atp)
        elif ablation == "h0":
            h_eff = jnp.zeros(n_hidden, dtype=jnp.int32)
            h_o = _soma_spike(h_v_pre, h_eff, gamma_eff, config.v_th)
            mu, v, h, t_prime, mu_at_tp, E_soma, dmu_dw, dmu_atp = carry
            carry = (mu, h_v_pre * (1 - h_o), h_eff, t_prime, mu_at_tp, E_soma, dmu_dw, dmu_atp)
            h_record = h_eff
        elif ablation == "h1":
            h_eff = jnp.ones(n_hidden, dtype=jnp.int32)
            h_o = _soma_spike(h_v_pre, h_eff, gamma_eff, config.v_th)
            mu, v, h, t_prime, mu_at_tp, E_soma, dmu_dw, dmu_atp = carry
            carry = (mu, h_v_pre * (1 - h_o), h_eff, t_prime, mu_at_tp, E_soma, dmu_dw, dmu_atp)
            h_record = h_eff
        else:
            raise ValueError(f"Unknown ablation: {ablation}")

        hidden_o_float = h_o.astype(jnp.float32)
        r_carry, _r_v, _r_E = LINeuron.forward_step(
            r_carry,
            hidden_o_float,
            r.w,
            r.alpha_m,
        )
        ho_list.append(np.asarray(h_o, dtype=np.float32))
        h_list.append(np.asarray(h_record, dtype=np.float32))

    readout_v_sum = np.asarray(r_carry[1], dtype=np.float64)
    readout_v_avg = readout_v_sum / T
    pred = int(np.argmax(readout_v_avg))
    ho = np.stack(ho_list)
    h = np.stack(h_list)

    sorted_avg = np.sort(readout_v_avg)
    margin = float(sorted_avg[-1] - sorted_avg[-2]) if readout_v_avg.size >= 2 else 0.0
    hidden_rate = float(ho.mean())
    plat_occ = float(h.mean())
    readout_v_total = float(readout_v_sum.sum())
    return pred, ho, h, readout_v_avg, margin, hidden_rate, plat_occ, readout_v_total


def plateau_in_out_ratio(ho: np.ndarray, h: np.ndarray) -> float:
    steps_in = float(h.sum())
    steps_out = float((1.0 - h).sum())
    rate_in = float((ho * h).sum()) / max(steps_in, 1.0)
    rate_out = float((ho * (1.0 - h)).sum()) / max(steps_out, 1.0)
    return rate_in / max(rate_out, 1e-12)


def collect_trial_records(
    net,
    X: np.ndarray,
    y: np.ndarray,
    ablation: Ablation = "none",
) -> List[Dict]:
    labels = np.asarray(y, dtype=np.int32)
    records: List[Dict] = []
    for i in range(X.shape[0]):
        pred, ho, h, v_avg, margin, hidden_rate, plat_occ, ro_v_total = forward_trial(
            net, jnp.asarray(X[i]), ablation=ablation,
        )
        label = int(labels[i])
        records.append({
            "label": label,
            "pred": pred,
            "correct": pred == label,
            "hidden_spike_rate": hidden_rate,
            "plateau_occupancy": plat_occ,
            "plateau_in_out_ratio": plateau_in_out_ratio(ho, h),
            "readout_v_total": ro_v_total,
            "readout_margin": margin,
            "readout_v_avg": v_avg,
        })
    return records


def summarize_group(records: List[Dict], name: str) -> None:
    n = len(records)
    if n == 0:
        print(f"  {name}: (no trials)")
        return
    hidden = np.array([r["hidden_spike_rate"] for r in records])
    plat = np.array([r["plateau_occupancy"] for r in records])
    ratio = np.array([r["plateau_in_out_ratio"] for r in records])
    ro_v = np.array([r["readout_v_total"] for r in records], dtype=np.float64)
    margin = np.array([r["readout_margin"] for r in records], dtype=np.float64)

    print(f"  {name} (n={n}):")
    print(f"    hidden spike rate:     mean={hidden.mean():.6f}  std={hidden.std():.6f}")
    print(f"    plateau occupancy:     mean={plat.mean():.6f}  std={plat.std():.6f}")
    print(f"    plateau in/out ratio:  mean={ratio.mean():.4f}x  median={np.median(ratio):.4f}x")
    print(f"    readout v sum/trial:   mean={ro_v.mean():.3f}  std={ro_v.std():.3f}")
    print(f"    readout margin:        mean={margin.mean():.4f}  std={margin.std():.4f}  "
          f"median={np.median(margin):.4f}")


def print_correct_vs_incorrect(records: List[Dict]) -> None:
    correct = [r for r in records if r["correct"]]
    incorrect = [r for r in records if not r["correct"]]
    n = len(records)
    acc = 100.0 * len(correct) / max(n, 1)

    print(f"\n=== Correct vs incorrect (test, baseline forward, n={n}) ===")
    print(f"  Accuracy: {acc:.2f}%  ({len(correct)} correct, {len(incorrect)} incorrect)")

    summarize_group(correct, "Correct")
    summarize_group(incorrect, "Incorrect")

    if incorrect:
        print("  Top confusion pairs (true -> pred, count):")
        from collections import Counter
        pairs = Counter((r["label"], r["pred"]) for r in incorrect)
        for (true_l, pred_l), cnt in pairs.most_common(8):
            print(f"    {true_l:2d} -> {pred_l:2d}: {cnt}")

    if correct and incorrect:
        c_h = np.mean([r["hidden_spike_rate"] for r in correct])
        i_h = np.mean([r["hidden_spike_rate"] for r in incorrect])
        c_m = np.mean([r["readout_margin"] for r in correct])
        i_m = np.mean([r["readout_margin"] for r in incorrect])
        print("  Incorrect - correct (mean delta):")
        print(f"    hidden spike rate: {i_h - c_h:+.6f}")
        print(f"    plateau occupancy: "
              f"{np.mean([r['plateau_occupancy'] for r in incorrect]) - np.mean([r['plateau_occupancy'] for r in correct]):+.6f}")
        print(f"    plateau in/out:    "
              f"{np.mean([r['plateau_in_out_ratio'] for r in incorrect]) - np.mean([r['plateau_in_out_ratio'] for r in correct]):+.4f}x")
        print(f"    readout margin:    {i_m - c_m:+.4f}")


def accuracy_from_records(records: List[Dict]) -> float:
    if not records:
        return 0.0
    return 100.0 * sum(r["correct"] for r in records) / len(records)


def evaluate_split_accuracy(
    net,
    X: np.ndarray,
    y: np.ndarray,
    ablation: Ablation = "none",
) -> float:
    """Classification accuracy (%) for one split and ablation mode."""
    labels = np.asarray(y, dtype=np.int32)
    n = X.shape[0]
    if n == 0:
        return 0.0
    correct = 0
    for i in range(n):
        pred, *_ = forward_trial(net, jnp.asarray(X[i]), ablation=ablation)
        if pred == int(labels[i]):
            correct += 1
    return 100.0 * correct / n


def _ratio_str(recs: List[Dict]) -> str:
    p_mean = np.mean([r["plateau_occupancy"] for r in recs])
    if p_mean < 1e-6 or p_mean > 1.0 - 1e-6:
        return "n/a"
    return f"{np.mean([r['plateau_in_out_ratio'] for r in recs]):.2f}x"


def print_ablation_results(
    baseline: List[Dict],
    ablations: Dict[Ablation, List[Dict]],
) -> None:
    base_acc = accuracy_from_records(baseline)
    print(f"\n=== Plateau necessity ablations (test, n={len(baseline)}) ===")
    print(f"  {'mode':<14}  {'accuracy':>8}  {'delta vs none':>14}  "
          f"{'hidden rate':>12}  {'plat occ':>10}  {'in/out':>8}")
    print("  " + "-" * 72)

    labels = {
        "none": "none (baseline)",
        "gamma0": "gamma0",
        "h0_soma": "h0_soma",
        "h1_soma": "h1_soma",
        "h0": "h0",
        "h1": "h1",
        "no_dend_in": "no_dend_in",
        "no_soma_in": "no_soma_in",
    }
    for mode in ABLATION_ORDER:
        recs = ablations[mode]
        acc = accuracy_from_records(recs)
        delta = acc - base_acc
        h_mean = np.mean([r["hidden_spike_rate"] for r in recs])
        p_mean = np.mean([r["plateau_occupancy"] for r in recs])
        print(
            f"  {labels[mode]:<14}  {acc:7.2f}%  {delta:+13.2f} pp  "
            f"{h_mean:12.6f}  {p_mean:10.4f}  {_ratio_str(recs):>8}"
        )

    print()
    print("  Ablation notes:")
    print("    none        — trained dynamics")
    print("    gamma0      — natural plateaus; gamma=0 on soma (no threshold lowering)")
    print("    h0_soma     — natural plateaus; soma threshold as if h=0 (gamma still applied)")
    print("    h1_soma     — natural plateaus; soma threshold as if h=1")
    print("    h0          — force h=0 in carry + soma (dendrite always integrates input)")
    print("    h1          — force h=1 in carry + soma (dendrite input gated off)")
    print("    no_dend_in  — zero w_dend @ x; soma path w_soma @ x unchanged")
    print("    no_soma_in  — zero w_soma @ x; dendritic path w_dend @ x unchanged")

    h0_recs = ablations["h0"]
    nd_recs = ablations["no_dend_in"]
    g0_recs = ablations["gamma0"]
    print()
    print("  Why no_dend_in is not the same as h0:")
    print("    h0: dendritic input ON, plateaus can build in mu, but carry h clamped to 0")
    print("        (dendrite runs; soma never gets gain-modulated threshold).")
    print("    no_dend_in: dendritic input OFF; only soma drive w_soma @ x remains.")
    print(f"    h0 hidden rate={np.mean([r['hidden_spike_rate'] for r in h0_recs]):.4f}  "
          f"plat occ={np.mean([r['plateau_occupancy'] for r in h0_recs]):.4f}  "
          f"acc={accuracy_from_records(h0_recs):.1f}%")
    print(f"    no_dend_in     rate={np.mean([r['hidden_spike_rate'] for r in nd_recs]):.4f}  "
          f"plat occ={np.mean([r['plateau_occupancy'] for r in nd_recs]):.4f}  "
          f"acc={accuracy_from_records(nd_recs):.1f}%")
    print(f"    gamma0         rate={np.mean([r['hidden_spike_rate'] for r in g0_recs]):.4f}  "
          f"plat occ={np.mean([r['plateau_occupancy'] for r in g0_recs]):.4f}  "
          f"acc={accuracy_from_records(g0_recs):.1f}%  "
          f"(isolates gamma vs natural h timing)")


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
    n_per = args.samples_per_class

    print(
        f"Test data: bin={bin_size_ms}ms collapse={collapse_factor} "
        f"samples_per_class={n_per}  gamma={meta['config']['gamma']}",
        flush=True,
    )

    print("Loading SHD test split...", flush=True)
    _, _, _, X_te, y_te, _ = load_shd_binned(
        bin_size_ms=bin_size_ms,
        collapse_factor=collapse_factor,
        max_duration_ms=max_duration_ms,
        train_samples_per_class=1,
        test_samples_per_class=n_per,
        binarize=binarize,
        dtype=np.float32,
    )
    if input_scale != 1.0:
        X_te = X_te * input_scale

    n_test = X_te.shape[0]
    print(f"Running baseline forward on {n_test} test trials...", flush=True)
    baseline = collect_trial_records(net, X_te, y_te, ablation="none")
    print_correct_vs_incorrect(baseline)

    ablation_records: Dict[Ablation, List[Dict]] = {"none": baseline}
    for mode in ABLATION_ORDER:
        if mode == "none":
            continue
        print(f"Running ablation '{mode}' on {n_test} test trials...", flush=True)
        ablation_records[mode] = collect_trial_records(net, X_te, y_te, ablation=mode)

    print_ablation_results(baseline, ablation_records)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
