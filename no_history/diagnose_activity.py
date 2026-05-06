#!/usr/bin/env python3
"""Diagnose per-layer activity statistics to figure out why the network isn't learning.

Reports for the chosen hyperparameters:
  - Dendritic membrane mu: max/mean (target > mu_th to fire plateaus)
  - Somatic membrane v:    max/mean (target > v_th to fire spikes)
  - Plateau activation rate (h=1 fraction)
  - Hidden spike rate (per (t, neuron))
  - Readout spike rate (per (t, output))
  - "Dead" neuron counts (never cross threshold across the inspected samples)

Run from the no_history/ folder:

    python diagnose_activity.py --bin_size_ms 4 --collapse_factor 5
    python diagnose_activity.py --bin_size_ms 4 --collapse_factor 5 --weight_scale 1.0 --input_scale 5
    python diagnose_activity.py --bin_size_ms 4 --collapse_factor 5 --mu_th 0.3 --v_th 0.5
"""
import argparse
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
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
from config import NeuronConfig
from network import Network
from two_comp_neuron import TwoCompNeuron
from lif_neuron import LIFNeuron


def parse_args():
    p = argparse.ArgumentParser(description="Activity diagnostic for no_history net on SHD.")
    p.add_argument("--bin_size_ms", type=float, default=4.0)
    p.add_argument("--collapse_factor", type=int, default=5)
    p.add_argument("--max_duration_ms", type=float, default=1400.0)
    p.add_argument("--input_scale", type=float, default=1.0)
    p.add_argument("--weight_scale", type=float, default=0.25)
    p.add_argument("--mu_th", type=float, default=1.0)
    p.add_argument("--v_th", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--n_hidden", type=int, default=64)
    p.add_argument("--n_outputs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--tau_soma", type=float, default=15.0)
    p.add_argument("--tau_dend", type=float, default=15.0)
    p.add_argument("--tau_m", type=float, default=20.0)
    p.add_argument("--tau_plat_min", type=float, default=100.0)
    p.add_argument("--tau_plat_max", type=float, default=350.0)
    p.add_argument("--beta_s", type=float, default=1.0)
    p.add_argument("--beta_d", type=float, default=1.5)
    return p.parse_args()


def run_forward_collect(net, x_input):
    """Run forward (no learning) and collect mu/v/h/o trajectories. Pure Python loop."""
    config = net.config
    h = net.hidden
    r = net.readout
    h_carry = h.init_carry()
    r_carry = r.init_carry()

    mus, vs, hs, h_os, r_os, r_vs = [], [], [], [], [], []
    T = x_input.shape[0]
    for t in range(T):
        x_t = x_input[t]
        dend_in = x_t @ h.w_dend.T
        soma_in = x_t @ h.w_soma.T
        h_carry, h_o, h_v_pre, h_h, h_h_prev, h_mu_at_tp = TwoCompNeuron.forward_step(
            h_carry, dend_in, soma_in, t, h.alpha_s, h.alpha_d, h.T_p, net.gamma_h, config,
        )
        hidden_o_float = h_o.astype(jnp.float64)
        r_carry, r_o, r_v_pre, r_E = LIFNeuron.forward_step(
            r_carry, hidden_o_float, r.w, r.alpha_m, config.v_th,
        )
        mu, v = h_carry[0], h_carry[1]
        mus.append(np.asarray(mu))
        vs.append(np.asarray(v))
        hs.append(np.asarray(h_h))
        h_os.append(np.asarray(h_o))
        r_os.append(np.asarray(r_o))
        r_vs.append(np.asarray(r_v_pre))
    return (np.stack(mus), np.stack(vs), np.stack(hs),
            np.stack(h_os), np.stack(r_os), np.stack(r_vs))


def main():
    args = parse_args()
    np.random.seed(args.seed)
    key = random.PRNGKey(args.seed)

    samples_per_class = max(1, args.n_samples // 20)
    print(f"Loading SHD ({samples_per_class} train samples per class)...", flush=True)
    X_tr, y_tr, _, _, _, _ = load_shd_binned(
        bin_size_ms=args.bin_size_ms,
        collapse_factor=args.collapse_factor,
        max_duration_ms=args.max_duration_ms,
        train_samples_per_class=samples_per_class,
        test_samples_per_class=1,
        dtype=np.float64,
    )
    if args.input_scale != 1.0:
        X_tr = X_tr * args.input_scale

    n_inputs = X_tr.shape[2]
    T = X_tr.shape[1]
    n_use = min(args.n_samples, len(X_tr))

    print()
    print(f"=== Input statistics ({n_use} samples) ===")
    print(f"  X shape per sample: ({T}, {n_inputs})")
    print(f"  bin counts: mean={X_tr.mean():.4f}, max={X_tr.max():.0f}, "
          f"std={X_tr.std():.4f}")
    print(f"  fraction of bins with at least 1 spike: "
          f"{(X_tr > 0).mean()*100:.2f}%")
    print(f"  fraction of timesteps with at least 1 active channel: "
          f"{(X_tr.any(axis=2)).mean()*100:.2f}%")
    print(f"  mean total spikes per sample: "
          f"{X_tr.sum(axis=(1,2)).mean():.0f}")

    config = NeuronConfig(
        dt=args.bin_size_ms,
        tau_soma=args.tau_soma, tau_dend=args.tau_dend, tau_m=args.tau_m,
        tau_plat_min=args.tau_plat_min, tau_plat_max=args.tau_plat_max,
        weight_scale=args.weight_scale,
        mu_th=args.mu_th, v_th=args.v_th, gamma=args.gamma,
        beta_s=args.beta_s, beta_d=args.beta_d,
    )

    print()
    print(f"=== Network configuration ===")
    print(f"  dt={config.dt}ms  tau_soma={config.tau_soma}  tau_dend={config.tau_dend}  tau_m={config.tau_m}")
    print(f"  alpha_s={float(np.exp(-config.dt/config.tau_soma)):.4f}  "
          f"alpha_d={float(np.exp(-config.dt/config.tau_dend)):.4f}  "
          f"alpha_m={float(np.exp(-config.dt/config.tau_m)):.4f}")
    print(f"  mu_th={config.mu_th}  v_th={config.v_th}  gamma={config.gamma}")
    print(f"  weight_scale={config.weight_scale}  input_scale={args.input_scale}")

    net = Network(key, n_inputs, args.n_hidden, args.n_outputs, config)
    print(f"  w_dend std={float(net.hidden.w_dend.std()):.5f}, "
          f"w_soma std={float(net.hidden.w_soma.std()):.5f}, "
          f"w_readout std={float(net.readout.w.std()):.5f}")

    all_mu_max, all_mu_mean = [], []
    all_v_max, all_v_mean = [], []
    all_rv_max, all_rv_mean, all_rv_std = [], [], []
    all_h_rate, all_ho_rate, all_ro_rate = [], [], []
    dead_dend = np.zeros(args.n_hidden)
    dead_soma = np.zeros(args.n_hidden)
    dead_readout = np.zeros(args.n_outputs)

    print()
    print(f"=== Running forward pass on {n_use} samples ===", flush=True)
    for i in range(n_use):
        x = jnp.array(X_tr[i])
        mu_seq, v_seq, h_seq, ho_seq, ro_seq, rv_seq = run_forward_collect(net, x)
        all_mu_max.append(mu_seq.max())
        all_mu_mean.append(mu_seq.mean())
        all_v_max.append(v_seq.max())
        all_v_mean.append(v_seq.mean())
        all_rv_max.append(rv_seq.max())
        all_rv_mean.append(rv_seq.mean())
        all_rv_std.append(rv_seq.std())
        all_h_rate.append(h_seq.mean())
        all_ho_rate.append(ho_seq.mean())
        all_ro_rate.append(ro_seq.mean())
        dead_dend += (mu_seq.max(axis=0) < config.mu_th).astype(int)
        dead_soma += (ho_seq.sum(axis=0) == 0).astype(int)
        dead_readout += (ro_seq.sum(axis=0) == 0).astype(int)

    mu_th, v_th = config.mu_th, config.v_th
    print()
    print(f"=== Activity report ===")
    print(f"  Dendritic mu:   max={np.mean(all_mu_max):.3f}  "
          f"mean={np.mean(all_mu_mean):.3f}   (mu_th={mu_th})")
    print(f"  Hidden v:       max={np.mean(all_v_max):.3f}  "
          f"mean={np.mean(all_v_mean):.3f}   (v_th={v_th})")
    print(f"  Readout v:      max={np.mean(all_rv_max):.3f}  "
          f"mean={np.mean(all_rv_mean):.3f}  std={np.mean(all_rv_std):.3f}   (v_th={v_th})")
    print(f"  Plateau h rate: {np.mean(all_h_rate)*100:.3f}%  "
          f"(fraction of timesteps × neurons where h=1)")
    print(f"  Hidden spikes:  {np.mean(all_ho_rate)*100:.3f}%  per (t, neuron)")
    print(f"  Readout spikes: {np.mean(all_ro_rate)*100:.3f}%  per (t, output)")
    print()
    print(f"  Hidden neurons whose mu NEVER crosses mu_th: "
          f"{int((dead_dend == n_use).sum())}/{args.n_hidden}")
    print(f"  Hidden neurons that NEVER spike:              "
          f"{int((dead_soma == n_use).sum())}/{args.n_hidden}")
    print(f"  Readout neurons that NEVER spike:             "
          f"{int((dead_readout == n_use).sum())}/{args.n_outputs}")
    print()

    h_rate = np.mean(all_h_rate)
    ho_rate = np.mean(all_ho_rate)
    ro_rate = np.mean(all_ro_rate)
    n_dead_readout = int((dead_readout == n_use).sum())
    rv_max_mean = np.mean(all_rv_max)
    print("=== Suggested action ===")
    if h_rate < 1e-3:
        print("  Plateau rate < 0.1% -> dendritic gradient path is essentially DEAD.")
        print("  Try (any of):  --weight_scale 1.0   --input_scale 5   --mu_th 0.3")
    elif h_rate > 0.5:
        print("  Plateau rate > 50% -> dendrite saturated; surrogate gradient ~ 0.")
        print("  Try:  --weight_scale 0.1  or  --mu_th 1.5")
    if ho_rate < 1e-3:
        print("  Hidden spike rate < 0.1% -> readout sees no signal.")
        print("  Try:  --weight_scale 1.0  --input_scale 5  --v_th 0.5  --gamma 0.7")
    elif ho_rate > 0.5:
        print("  Hidden spike rate > 50% -> soma saturated; surrogate gradient ~ 0.")
        print("  Try:  --weight_scale 0.1  or  --v_th 1.5")
    if n_dead_readout > args.n_outputs // 4:
        print(f"  {n_dead_readout}/{args.n_outputs} readout neurons NEVER spike -> "
              f"those classes can never be predicted.")
        print(f"  Readout v_max (mean over samples)={rv_max_mean:.3f}, v_th={v_th}.")
        if rv_max_mean < v_th * 0.5:
            print("  Readout drive is way below threshold. Try:  --weight_scale 1.0  "
                  "or  --input_scale 5  (or both).")
        elif rv_max_mean < v_th:
            print("  Readout drive grazes threshold. Try:  --weight_scale 0.5  "
                  "or  --v_th 0.5.")
    elif 1e-3 <= h_rate <= 0.5 and 1e-3 <= ho_rate <= 0.5 and ro_rate >= 1e-3 and n_dead_readout < args.n_outputs // 4:
        print("  Activity looks healthy. If still not learning, check learning rate / loss / optimizer.")


if __name__ == "__main__":
    main()
