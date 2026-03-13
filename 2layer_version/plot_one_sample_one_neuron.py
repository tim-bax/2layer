#!/usr/bin/env python3
"""
For one sample and one neuron per layer: plot forward (dendritic + somatic) and backward
(dendritic + somatic) for the extra layer, the hidden layer, and the readout (somatic only).
Saves three figures: extra_neuron_*.png, hidden_neuron_*.png, readout_neuron_*.png.

Eligibility traces (E): per synapse (one series per input). Formula E_t = α*E_{t-1} + x_t.
For SHD, create_shd_input_jax uses an alpha kernel (default): each spike is convolved so
x_t is not 0/1 but can be ~5 per bin (peak) and smeared over ~33 ms. So E rises fast
because each 1 ms bin can add up to ~5 (not 1); with α≈0.94, a short burst gives E≫1.
We plot only the first 3 (extra/hidden) or 5 (readout) E curves to avoid clutter.

dμ/dw (dendritic eligibility): per synapse (shape T, n_neurons, n_inputs). We plot the first 3
synapses for the chosen neuron (like E) plus the L2 norm. For the extra layer, the full dendritic
gradient uses the same trace dμ/dw_extra but evaluated at different times: (1) path via hidden
soma uses dμ/dw_extra(t) at each t; (2) path via hidden dendrite uses dμ/dw_extra(t′_hidden(t))
(i.e. indexed by the hidden layer’s t′ at each t). So the values in the sum differ—the code
uses dmu_dw_extra[t] for the first term and dmu_dw_extra[t_prime_h_int] for the second (see 2layer.py).
We plot the full trace dμ/dw_extra(t); it is that trace sampled at t and at t′_hidden(t) in the two terms.
"""
import os
import sys
import argparse
import importlib.util

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

from data import load_shd_data, create_shd_input_jax


def _load_twolayer_module(lowmemory: bool):
    """Return (JAXEPropNetworkTwoLayer, JAXTwoCompartmentalLayer) from 2layer or 2layer_lowmemory."""
    basename = "2layer_lowmemory.py" if lowmemory else "2layer.py"
    path = os.path.join(_SCRIPT_DIR, basename)
    spec = importlib.util.spec_from_file_location("twolayer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.JAXEPropNetworkTwoLayer, mod.JAXTwoCompartmentalLayer


def _compute_backward_trajectories(network, params, x_input, target, Layer):
    """Run forward then compute full backward time series (same as _compute_gradients but return arrays)."""
    (mu_e, v_e, h_e, extra_o, mu_history_e, t_prime_e, v_history_e,
     mu_h, v_h, h_h, hidden_o, mu_history_h, t_prime_h, v_history_h,
     readout_v, readout_o) = network._forward_with_params(params, x_input)

    T = x_input.shape[0]
    n_hidden = network.n_hidden
    n_extra = network.n_extra
    n_inputs = network.n_inputs
    cfg = network.config

    global_errors = network.compute_global_errors(readout_o, target)
    effective_error_hidden = jnp.einsum('tj,j,ji->ti', Layer.surrogate_sigma(readout_v - cfg.v_th, cfg.beta_s), global_errors, params.w_readout)

    E_readout = Layer.compute_eligibility_traces(hidden_o, network.readout_layer.alpha)
    sigma_prime_readout = Layer.surrogate_sigma(readout_v - cfg.v_th, cfg.beta_s)

    soma_input_h = v_history_h + cfg.gamma * h_h - cfg.v_th
    sigma_prime_hidden = Layer.surrogate_sigma(soma_input_h, cfg.beta_s)
    t_prime_h_int = t_prime_h.astype(jnp.int32)
    neuron_idx_h = jnp.arange(n_hidden)[None, :]
    mu_at_tprime_h = mu_history_h[t_prime_h_int, neuron_idx_h]
    h_prime_hidden = Layer.surrogate_sigma(mu_at_tprime_h - cfg.mu_th, cfg.beta_d)
    E_soma_hidden = Layer.compute_eligibility_traces(extra_o, network.hidden_layer.alpha_s)
    dmu_dw_hidden = Layer.compute_dmu_tprime_dw(extra_o, h_h, t_prime_h, network.hidden_layer.alpha)

    soma_input_e = v_history_e + cfg.gamma * h_e - cfg.v_th
    sigma_prime_extra = Layer.surrogate_sigma(soma_input_e, cfg.beta_s)
    t_prime_e_int = t_prime_e.astype(jnp.int32)
    neuron_idx_e = jnp.arange(n_extra)[None, :]
    mu_at_tprime_e = mu_history_e[t_prime_e_int, neuron_idx_e]
    h_prime_extra = Layer.surrogate_sigma(mu_at_tprime_e - cfg.mu_th, cfg.beta_d)
    E_soma_extra = Layer.compute_eligibility_traces(x_input, network.extra_layer.alpha_s)
    dmu_dw_extra = Layer.compute_dmu_tprime_dw(x_input, h_e, t_prime_e, network.extra_layer.alpha)

    return {
        "mu_e": mu_history_e, "h_e": h_e, "v_e": v_history_e, "extra_o": extra_o,
        "mu_h": mu_history_h, "h_h": h_h, "v_h": v_history_h, "hidden_o": hidden_o,
        "readout_v": readout_v, "readout_o": readout_o,
        "sigma_prime_readout": sigma_prime_readout, "E_readout": E_readout,
        "sigma_prime_hidden": sigma_prime_hidden, "h_prime_hidden": h_prime_hidden,
        "E_soma_hidden": E_soma_hidden, "dmu_dw_hidden": dmu_dw_hidden,
        "sigma_prime_extra": sigma_prime_extra, "h_prime_extra": h_prime_extra,
        "E_soma_extra": E_soma_extra, "dmu_dw_extra": dmu_dw_extra,
        "t_prime_e": t_prime_e, "t_prime_h": t_prime_h,
    }


def main():
    p = argparse.ArgumentParser(description="Plot forward/backward for one sample, one neuron per layer")
    p.add_argument("--pkl", type=str, default=os.path.join(_SCRIPT_DIR, "model", "shd_two_layer_65_10.pkl"), help="Path to .pkl model")
    p.add_argument("--sample_idx", type=int, default=0, help="Index of sample in test set")
    p.add_argument("--neuron_extra", type=int, default=0, help="Extra layer neuron index")
    p.add_argument("--neuron_hidden", type=int, default=0, help="Hidden layer neuron index")
    p.add_argument("--neuron_readout", type=int, default=0, help="Readout neuron index")
    p.add_argument("--split", type=str, default="test", choices=("train", "test"))
    p.add_argument("--lowmemory", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default=None, help="Directory to save figures (default: script dir)")
    p.add_argument("--no_kernel", action="store_true", help="Use use_kernel=False for input (no alpha kernel; instantaneous spike_amplitude per bin)")
    p.add_argument("--spike_amplitude", type=float, default=None, help="Spike amplitude when --no_kernel (default: same as create_shd_input_jax, 5.0)")
    args = p.parse_args()

    if not os.path.isfile(args.pkl):
        print(f"Error: model file not found: {args.pkl}")
        sys.exit(1)

    JAXEPropNetworkTwoLayer, JAXTwoCompartmentalLayer = _load_twolayer_module(args.lowmemory)
    key = random.PRNGKey(args.seed)
    network = JAXEPropNetworkTwoLayer.load(args.pkl, key=key)
    T = network.T
    n_inputs = network.n_inputs
    n_extra, n_hidden, n_outputs = network.n_extra, network.n_hidden, network.n_outputs

    if "SHD_DATA_PATH" in os.environ:
        data_path = os.environ["SHD_DATA_PATH"]
    elif os.path.exists("/share/neurocomputation/Tim/SHD_data"):
        data_path = "/share/neurocomputation/Tim/SHD_data"
    else:
        data_path = os.path.expanduser("~/Documents/Heidelberg_Data")

    if not os.path.isdir(data_path):
        print(f"Error: SHD data path not found: {data_path}")
        sys.exit(1)

    train_raw, test_raw = load_shd_data(data_path, train_samples_per_class=None, test_samples_per_class=None)
    train_data = [(create_shd_input_jax(x, T=T), label) for x, label in train_raw]
    test_data = [(create_shd_input_jax(x, T=T), label) for x, label in test_raw]
    data = train_data if args.split == "train" else test_data
    raw_data = train_raw if args.split == "train" else test_raw
    if args.sample_idx >= len(data):
        print(f"Error: sample_idx {args.sample_idx} >= len(data) {len(data)}")
        sys.exit(1)
    target = data[args.sample_idx][1]
    if args.no_kernel:
        kw = dict(T=T, use_kernel=False)
        if args.spike_amplitude is not None:
            kw["spike_amplitude"] = args.spike_amplitude
        x = create_shd_input_jax(raw_data[args.sample_idx][0], **kw)
        print(f"Using input with use_kernel=False (spike_amplitude={kw.get('spike_amplitude', 5.0)})")
    else:
        x = data[args.sample_idx][0]
    x_j = jnp.asarray(x)
    params = network.get_params()

    traj = _compute_backward_trajectories(network, params, x_j, target, JAXTwoCompartmentalLayer)
    for k, v in traj.items():
        traj[k] = np.asarray(v)

    t_ms = np.arange(T)
    out_dir = args.out_dir or _SCRIPT_DIR
    os.makedirs(out_dir, exist_ok=True)
    ne, nh, nr = args.neuron_extra, args.neuron_hidden, args.neuron_readout
    suffix = "_nokernel" if args.no_kernel else ""

    # ----- Extra layer: one neuron, 4 subplots (dend fwd, dend bwd, soma fwd, soma bwd) -----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Extra layer — neuron {ne} (sample {args.sample_idx}, target={target})" + (" [no kernel]" if args.no_kernel else ""), fontsize=12)

    ax = axes[0, 0]
    ax.plot(t_ms, traj["mu_e"][:, ne], label=r"$\mu$", color="C0")
    ax.plot(t_ms, traj["h_e"][:, ne], label="$h$", color="C1")
    ax.set_title("Dendritic forward")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_ms, traj["sigma_prime_extra"][:, ne], label=r"$\sigma'$", color="C0")
    ax.plot(t_ms, traj["h_prime_extra"][:, ne], label="$h'$", color="C1")
    n_in_dmu = min(3, traj["dmu_dw_extra"].shape[2])
    for j in range(n_in_dmu):
        ax.plot(t_ms, traj["dmu_dw_extra"][:, ne, j], label=rf"$d\mu/dw_{{in{j}}}$", alpha=0.8)
    ax.set_title("Dendritic backward (σ′, h′, dμ/dw first 3 inputs)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_ms, traj["v_e"][:, ne], label="$v$", color="C2")
    ax_t = ax.twinx()
    ax_t.fill_between(t_ms, 0, traj["extra_o"][:, ne], alpha=0.4, color="C3", label="$o$")
    ax_t.set_ylim(-0.1, 1.5)
    ax.set_title("Somatic forward")
    ax.legend(loc="upper left", fontsize=8)
    ax_t.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_ms, traj["sigma_prime_extra"][:, ne], label=r"$\sigma'$", color="C0")
    n_in_show = min(3, traj["E_soma_extra"].shape[1])
    for j in range(n_in_show):
        ax.plot(t_ms, traj["E_soma_extra"][:, j], label=f"$E_{{in{j}}}$", alpha=0.8)
    ax.set_title("Somatic backward (σ′ + first 3 of n_inputs eligibility traces)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    for ax in axes[1, :]:
        ax.set_xlabel("Time (ms)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"extra_neuron_{ne}_sample{args.sample_idx}{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # ----- Extra: h′ and dμ/dw only (with t′_hidden sampling times) -----
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(
        f"Extra layer — h′ and dμ/dw (neuron {ne}); vertical lines = t′_hidden (dend path samples here)"
        + (" [no kernel]" if args.no_kernel else ""),
        fontsize=11,
    )
    t_prime_h = np.asarray(traj["t_prime_h"])
    uniq_tprime = np.unique(t_prime_h)
    # same x-scale as t_ms (time-step indices)
    max_vlines = 80
    if len(uniq_tprime) > max_vlines:
        step = max(1, len(uniq_tprime) // max_vlines)
        vline_ticks = uniq_tprime[::step]
    else:
        vline_ticks = uniq_tprime
    for ax in axes:
        for _t in vline_ticks:
            ax.axvline(_t, color="gray", alpha=0.25, linewidth=0.8)
    ax = axes[0]
    ax.plot(t_ms, traj["h_prime_extra"][:, ne], label=rf"$h'_{{extra}}$", color="C1")
    ax.set_ylabel(r"$h'$")
    ax.set_title(r"$h'_{extra}$ (at t′_extra); dend path uses h′ at t′_hidden(t)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    n_in_dmu = min(3, traj["dmu_dw_extra"].shape[2])
    for j in range(n_in_dmu):
        ax.plot(t_ms, traj["dmu_dw_extra"][:, ne, j], label=rf"$d\mu/dw_{{in{j}}}$", alpha=0.8)
    ax.set_ylabel(r"$d\mu/dw$")
    ax.set_xlabel("Time (ms)")
    ax.set_title(r"dμ/dw_extra (first 3); soma path at t, dend path at t′_hidden(t)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path_h_dmudw = os.path.join(out_dir, f"extra_hprime_dmudw_neuron_{ne}_sample{args.sample_idx}{suffix}.png")
    plt.savefig(out_path_h_dmudw, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path_h_dmudw}")

    # ----- Hidden layer: one neuron, 4 subplots -----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Hidden layer — neuron {nh} (sample {args.sample_idx}, target={target})" + (" [no kernel]" if args.no_kernel else ""), fontsize=12)

    ax = axes[0, 0]
    ax.plot(t_ms, traj["mu_h"][:, nh], label=r"$\mu$", color="C0")
    ax.plot(t_ms, traj["h_h"][:, nh], label="$h$", color="C1")
    ax.set_title("Dendritic forward")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_ms, traj["sigma_prime_hidden"][:, nh], label=r"$\sigma'$", color="C0")
    ax.plot(t_ms, traj["h_prime_hidden"][:, nh], label="$h'$", color="C1")
    n_in_dmu_h = min(3, traj["dmu_dw_hidden"].shape[2])
    for j in range(n_in_dmu_h):
        ax.plot(t_ms, traj["dmu_dw_hidden"][:, nh, j], label=rf"$d\mu/dw_{{ex{j}}}$", alpha=0.8)
    ax.set_title("Dendritic backward (σ′, h′, dμ/dw first 3 inputs)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_ms, traj["v_h"][:, nh], label="$v$", color="C2")
    ax_t = ax.twinx()
    ax_t.fill_between(t_ms, 0, traj["hidden_o"][:, nh], alpha=0.4, color="C3", label="$o$")
    ax_t.set_ylim(-0.1, 1.5)
    ax.set_title("Somatic forward")
    ax.legend(loc="upper left", fontsize=8)
    ax_t.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_ms, traj["sigma_prime_hidden"][:, nh], label=r"$\sigma'$", color="C0")
    n_in_show_h = min(3, traj["E_soma_hidden"].shape[1])
    for j in range(n_in_show_h):
        ax.plot(t_ms, traj["E_soma_hidden"][:, j], label=f"$E_{{ex{j}}}$", alpha=0.8)
    ax.set_title("Somatic backward (σ′ + first 3 of n_extra eligibility traces)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    for ax in axes[1, :]:
        ax.set_xlabel("Time (ms)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"hidden_neuron_{nh}_sample{args.sample_idx}{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # ----- Readout: somatic only — 2 subplots (forward v,o; backward σ', E for one output) -----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Readout (somatic) — neuron {nr} (sample {args.sample_idx}, target={target})" + (" [no kernel]" if args.no_kernel else ""), fontsize=12)

    ax = axes[0]
    ax.plot(t_ms, traj["readout_v"][:, nr], label="$v$", color="C2")
    ax_t = ax.twinx()
    ax_t.fill_between(t_ms, 0, traj["readout_o"][:, nr], alpha=0.4, color="C3", label="$o$")
    ax_t.set_ylim(-0.1, 1.5)
    ax.set_title("Somatic forward")
    ax.legend(loc="upper left", fontsize=8)
    ax_t.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_ms, traj["sigma_prime_readout"][:, nr], label=r"$\sigma'$", color="C0")
    n_h_show = min(5, traj["E_readout"].shape[1])
    for j in range(n_h_show):
        ax.plot(t_ms, traj["E_readout"][:, j], label=f"$E_{{h{j}}}$", alpha=0.8)
    ax.set_title("Somatic backward (σ′ + first 5 of n_hidden eligibility traces)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel("Time (ms)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"readout_neuron_{nr}_sample{args.sample_idx}{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
