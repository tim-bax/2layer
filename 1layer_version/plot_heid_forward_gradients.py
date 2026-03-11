#!/usr/bin/env python3
"""
Plot forward pass (dendrite + soma) and gradient components for one random hidden neuron
and one random readout neuron, using a saved heid_2comp model (e.g. heid_2comp_58_65.pkl).

- Effective error: eff_err(t,i) = Σ_j σ'_readout(t,j)·global_error_j·w_readout(j,i); it *does*
  vary with t when σ'_readout varies. A separate figure plots effective error and σ'_readout
  so you can see the time dependence.
- Readout: forward (voltage, spikes, hidden input) and gradient components (σ'_readout,
  global error, E_readout, per-t factor) for one output neuron.

Gradient-per-timestep (hidden): shows |product| so "higher" = stronger contribution.
"""
import os
import sys
import pickle
import random as pyrandom
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project root for data package
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, _SCRIPT_DIR)

# Load model (sets JAX env etc.)
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from model import JAXEPropNetwork

# Heidelberg / SHD data path (used if SHD_DATA_PATH env is not set)
HEIDELBERG_DATA_PATH = "/Users/tbax/Documents/Heidelberg_Data"

# Which data sample to plot: int = use that index, None = random each run
# Override via CLI: python plot_heid_forward_gradients.py --sample 42  or  --sample random
SAMPLE_INDEX = None

# Optional: load SHD for one sample; fallback to random input if data unavailable
def get_one_sample(n_inputs: int, T: int, pkl_path: str, sample_index=None):
    """
    sample_index: int = use train_raw[sample_index]; None = pick random.
    Returns (x_input, label, sample_index_used).
    """
    try:
        from data import load_shd_data, create_shd_input_jax
        data_path = (
            os.environ.get("SHD_DATA_PATH") or
            (HEIDELBERG_DATA_PATH if os.path.exists(HEIDELBERG_DATA_PATH) else None) or
            ("/share/neurocomputation/Tim/SHD_data" if os.path.exists("/share/neurocomputation/Tim/SHD_data") else None) or
            os.path.expanduser("~/data/hdspikes")
        )
        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)
        train_raw, _ = load_shd_data(
            data_path,
            train_samples_per_class=10,
            test_samples_per_class=None,
        )
        n_total = len(train_raw)
        if sample_index is None:
            # Use Python's random so choice isn't fixed by np.random.seed in model/2comp_uniform
            sample_index = pyrandom.randint(0, n_total - 1)
        else:
            sample_index = int(sample_index)
            if sample_index < 0 or sample_index >= n_total:
                raise ValueError(f"sample_index must be in [0, {n_total}), got {sample_index}")
        x_raw, label = train_raw[sample_index]
        x = create_shd_input_jax(x_raw, T=T)
        x = jnp.array(x)
        return x, int(label), sample_index
    except Exception as e:
        print(f"Using random input (data load failed: {e})", flush=True)
        np.random.seed(42)
        x = jnp.array(np.random.randn(T, n_inputs).astype(np.float64) * 0.1)
        return x, 0, -1


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot forward/gradient for one sample and one hidden neuron")
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Data sample index (int) or 'random'. Default: use SAMPLE_INDEX in script (currently %s)" % SAMPLE_INDEX,
    )
    args = parser.parse_args()

    # Resolve sample index: CLI > script constant; "random" -> None
    sample_index = SAMPLE_INDEX
    if args.sample is not None:
        if args.sample.strip().lower() == "random":
            sample_index = None
        else:
            try:
                sample_index = int(args.sample)
            except ValueError:
                parser.error("--sample must be an int or 'random'")

    pkl_path = os.path.join(_SCRIPT_DIR, "heid_2comp_58_65.pkl")
    if not os.path.isfile(pkl_path):
        print(f"Model not found: {pkl_path}")
        return

    print("Loading model...", flush=True)
    network = JAXEPropNetwork.load(pkl_path)
    params = network.get_params()
    T = network.T
    n_inputs = network.n_inputs
    n_hidden = network.n_hidden

    print("Loading one sample...", flush=True)
    x_input, target, sample_idx_used = get_one_sample(n_inputs, T, pkl_path, sample_index=sample_index)
    if sample_idx_used >= 0:
        print(f"  Using data sample index {sample_idx_used} (target class {target})", flush=True)
    else:
        print(f"  Using random input (target class {target})", flush=True)
    target = jnp.array(target, dtype=jnp.int32)

    # Forward pass: use same JIT path as training/eval so readout gets correct inputs
    T_p = network.hidden_layer.T_p
    mu, v_final, h, hidden_o, readout_v, readout_o, mu_history, t_prime_history, v_history = (
        network._forward_with_params_jit(params, x_input, network.config, T, T_p)
    )

    # Diagnostics: why would readout be flat?
    readout_input = hidden_o @ params.w_readout.T  # (T, n_outputs)
    hidden_spikes_total = int(jnp.sum(hidden_o))
    hidden_spikes_per = np.array(jnp.sum(hidden_o, axis=0))  # (n_hidden,)
    ro_in_min, ro_in_max = float(jnp.min(readout_input)), float(jnp.max(readout_input))
    ro_v_min, ro_v_max = float(jnp.min(readout_v)), float(jnp.max(readout_v))
    ro_o_sum = int(jnp.sum(readout_o))
    readout_spikes_per = np.array(jnp.sum(readout_o, axis=0))  # (n_outputs,) — each readout neuron drives effective error
    w_ro_min, w_ro_max = float(jnp.min(params.w_readout)), float(jnp.max(params.w_readout))
    print(f"  hidden_o: {hidden_o.shape}, total hidden spikes = {hidden_spikes_total}", flush=True)
    print(f"  hidden spikes per neuron (all {n_hidden}): {hidden_spikes_per.tolist()}", flush=True)
    print(f"  w_readout: {params.w_readout.shape}, range [{w_ro_min:.4f}, {w_ro_max:.4f}]", flush=True)
    print(f"  readout input (hidden_o @ w.T): range [{ro_in_min:.4f}, {ro_in_max:.4f}]", flush=True)
    print(f"  readout_v: range [{ro_v_min:.4f}, {ro_v_max:.4f}]", flush=True)
    print(f"  readout_o: total spikes = {ro_o_sum}", flush=True)
    print(f"  readout spikes per neuron (all {network.n_outputs}, each drives effective error): {readout_spikes_per.tolist()}", flush=True)

    # Pick one random hidden neuron and one random output neuron for readout plots
    np.random.seed(318)
    neuron_idx = int(np.random.randint(0, n_hidden))
    n_outputs = network.n_outputs
    out_idx = int(np.random.randint(0, n_outputs))

    # Gradient components (same logic as _train_step_impl)
    global_errors = network.compute_global_errors(readout_o, target)
    print(f"  global_errors (per readout neuron, drive effective error): {np.array(global_errors).tolist()}", flush=True)
    JAXTwoCompartmentalLayer = type(network.hidden_layer)

    E_readout = JAXTwoCompartmentalLayer.compute_eligibility_traces(hidden_o, network.readout_layer.alpha)
    v_input_vals = readout_v - network.config.v_th
    sigma_prime_readout = JAXTwoCompartmentalLayer.surrogate_sigma(v_input_vals, network.config.beta_s)

    effective_error_time_series = jnp.einsum("tj,j,ji->ti", sigma_prime_readout, global_errors, params.w_readout)
    soma_input_vals = v_history + network.config.gamma * h - network.config.v_th
    sigma_prime_hidden = JAXTwoCompartmentalLayer.surrogate_sigma(soma_input_vals, network.config.beta_s)

    t_prime_indices = t_prime_history.astype(jnp.int32)
    neuron_indices = jnp.arange(n_hidden)[None, :]
    mu_at_tprime = mu_history[t_prime_indices, neuron_indices]
    dend_input_vals = mu_at_tprime - network.config.mu_th
    h_prime = JAXTwoCompartmentalLayer.surrogate_sigma(dend_input_vals, network.config.beta_d)

    E_soma = JAXTwoCompartmentalLayer.compute_eligibility_traces(x_input, network.hidden_layer.alpha_s)
    dmu_tprime_dw = JAXTwoCompartmentalLayer.compute_dmu_tprime_dw(
        x_input, h, t_prime_history, network.hidden_layer.alpha
    )

    # Convert to numpy for plotting
    t_axis = np.arange(T)
    mu_n = np.array(mu_history[:, neuron_idx])
    v_n = np.array(v_history[:, neuron_idx])
    eff_err_n = np.array(effective_error_time_series[:, neuron_idx])
    sigma_p_n = np.array(sigma_prime_hidden[:, neuron_idx])
    h_prime_n = np.array(h_prime[:, neuron_idx])
    # E_soma is (T, n_inputs); for one neuron we show norm over inputs per t (eligibility magnitude)
    E_soma_norm_t = np.array(jnp.linalg.norm(E_soma, axis=1))
    # dmu_tprime_dw is (T, n_neurons, n_inputs); for neuron_idx take (T, n_inputs) and norm per t
    dmu_dw_n = np.array(dmu_tprime_dw[:, neuron_idx, :])  # (T, n_inputs)
    dmu_dw_norm_t = np.linalg.norm(dmu_dw_n, axis=1)

    # Per-t somatic gradient contribution (scalar per t): sigma' * effective_error for this neuron
    # (Full somatic grad also multiplies E_soma and sums over t and input dims.)
    soma_component_t = np.array(sigma_prime_hidden[:, neuron_idx] * effective_error_time_series[:, neuron_idx])
    # Per-t dendritic contribution (scalar): gamma * sigma' * h' * effective_error * ||dmu_dw||
    dend_component_t = np.array(
        network.config.gamma * sigma_prime_hidden[:, neuron_idx] * h_prime[:, neuron_idx]
        * effective_error_time_series[:, neuron_idx] * jnp.linalg.norm(dmu_tprime_dw[:, neuron_idx, :], axis=1)
    )
    dend_component_t = np.array(dend_component_t)
    h_n = np.array(h[:, neuron_idx])  # plateau indicator (0/1) for this neuron

    # Readout: forward and gradient components for one output neuron
    readout_input_n = np.array(readout_input[:, out_idx])  # what this output neuron receives (hidden_o @ w.T)
    readout_v_n = np.array(readout_v[:, out_idx])
    readout_o_n = np.array(readout_o[:, out_idx])
    hidden_o_n = np.array(hidden_o[:, neuron_idx])  # input from our hidden neuron to readout
    sigma_prime_readout_n = np.array(sigma_prime_readout[:, out_idx])
    global_err_out = float(global_errors[out_idx])
    E_readout_n = np.array(E_readout[:, neuron_idx])  # eligibility from hidden neuron_idx for readout
    readout_per_t = np.array(
        sigma_prime_readout[:, out_idx] * E_readout[:, neuron_idx] * global_errors[out_idx]
    )

    # --- Figure 1: Forward pass (dendrite + soma) ---
    fig1, (ax_dend, ax_soma) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    ax_dend.plot(t_axis, mu_n, color="C0", label=r"$\mu$ (dendrite)")
    ax_dend.set_ylabel("Dendritic potential")
    ax_dend.set_title(f"Forward pass — hidden neuron {neuron_idx} (dendrite)")
    ax_dend.legend(loc="lower right")
    ax_dend.grid(True, alpha=0.3)

    ax_soma.plot(t_axis, v_n, color="C1", label=r"$v$ (soma)")
    ax_soma.set_ylabel("Somatic potential")
    ax_soma.set_xlabel("Time step")
    ax_soma.set_title(f"Forward pass — hidden neuron {neuron_idx} (soma)")
    ax_soma.legend(loc="lower right")
    ax_soma.grid(True, alpha=0.3)

    fig1.suptitle(f"Forward pass for one random hidden neuron (index {neuron_idx})", fontsize=12)
    fig1.tight_layout()
    out1 = os.path.join(_SCRIPT_DIR, "heid_forward_pass_one_neuron.png")
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)
    print(f"Saved {out1}")

    # --- Figure 2: Gradient components for the same neuron ---
    fig2, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # Soma gradient components
    ax_s = axes[0]
    ax_s.plot(t_axis, sigma_p_n, color="C0", alpha=0.8, label=r"$\sigma'$ (soma surrogate)")
    ax_s.plot(t_axis, eff_err_n, color="C1", alpha=0.8, label="Effective error")
    ax_s_twin = ax_s.twinx()
    ax_s_twin.plot(t_axis, E_soma_norm_t, color="C2", alpha=0.7, linestyle="--", label=r"$\|E_{soma}\|$")
    ax_s_twin.set_ylabel(r"$\|E_{soma}\|$", color="C2")
    ax_s.set_ylabel("Value")
    ax_s.set_title(f"Somatic gradient components — neuron {neuron_idx}")
    ax_s.legend(loc="lower right")
    ax_s_twin.legend(loc="lower right")
    ax_s.grid(True, alpha=0.3)

    # Dendrite gradient components
    ax_d = axes[1]
    ax_d.plot(t_axis, sigma_p_n, color="C0", alpha=0.8, label=r"$\sigma'$")
    ax_d.plot(t_axis, h_prime_n, color="C3", alpha=0.8, label=r"$h'$ (dendritic surrogate)")
    ax_d.plot(t_axis, eff_err_n, color="C1", alpha=0.8, label="Effective error")
    ax_d_twin = ax_d.twinx()
    ax_d_twin.plot(t_axis, dmu_dw_norm_t, color="C4", alpha=0.7, linestyle="--", label=r"$\|\partial\mu_{t'}/\partial w\|$")
    ax_d_twin.set_ylabel(r"$\|\partial\mu_{t'}/\partial w\|$", color="C4")
    ax_d.set_ylabel("Value")
    ax_d.set_xlabel("Time step")
    ax_d.set_title(f"Dendritic gradient components — neuron {neuron_idx}")
    ax_d.legend(loc="lower right")
    ax_d_twin.legend(loc="lower right")
    ax_d.grid(True, alpha=0.3)

    fig2.suptitle(
        f"Gradient components for hidden neuron {neuron_idx}. "
        r"Effective error$(t,i)=\sum_j \sigma'_{\mathrm{readout}}(t,j)\cdot$ global_error$_j\cdot w_{\mathrm{readout}}(j,i)$; varies with $t$ when $\sigma'_{\mathrm{readout}}$ varies.",
        fontsize=8,
    )
    fig2.tight_layout()
    out2 = os.path.join(_SCRIPT_DIR, "heid_gradient_components_one_neuron.png")
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"Saved {out2}")

    # --- Figure: Effective error vs readout surrogate (why effective error can look flat) ---
    fig_eff, (ax_ee, ax_sr) = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    ax_ee.plot(t_axis, eff_err_n, color="C1", label=r"Effective error (hidden " + str(neuron_idx) + r")")
    ax_ee.set_ylabel("Effective error")
    ax_ee.set_title(r"Effective error$(t)$ for this hidden neuron (should vary with $t$)")
    ax_ee.legend(loc="lower right")
    ax_ee.grid(True, alpha=0.3)
    ax_sr.plot(t_axis, sigma_prime_readout_n, color="C0", label=r"$\sigma'_{\mathrm{readout}}(t)$ output " + str(out_idx))
    ax_sr.set_ylabel(r"$\sigma'_{\mathrm{readout}}$")
    ax_sr.set_xlabel("Time step")
    ax_sr.set_title(r"Readout surrogate for output " + str(out_idx) + r" (ingredient of effective error)")
    ax_sr.legend(loc="lower right")
    ax_sr.grid(True, alpha=0.3)
    fig_eff.suptitle(
        r"Effective error$(t,i)=\sum_j \sigma'_{\mathrm{readout}}(t,j)\cdot \mathrm{global\_error}_j \cdot w_{\mathrm{readout}}(j,i)$",
        fontsize=9,
    )
    fig_eff.tight_layout()
    out_eff = os.path.join(_SCRIPT_DIR, "heid_effective_error_vs_readout_surrogate.png")
    fig_eff.savefig(out_eff, dpi=150)
    plt.close(fig_eff)
    print(f"Saved {out_eff}")

    # --- Figure: Readout forward (one output neuron) ---
    fig_ro_fwd, axes_ro = plt.subplots(4, 1, sharex=True, figsize=(10, 7))
    axes_ro[0].plot(t_axis, readout_input_n, color="C4", label=r"input to readout ($h \cdot w^{\top}$)")
    axes_ro[0].set_ylabel("Input")
    axes_ro[0].set_title(f"Readout forward — output neuron {out_idx} (if input is flat, readout receives nothing)")
    axes_ro[0].legend(loc="lower right")
    axes_ro[0].grid(True, alpha=0.3)
    axes_ro[1].plot(t_axis, readout_v_n, color="C0", label=r"$v_{\mathrm{readout}}$")
    axes_ro[1].set_ylabel("Voltage")
    axes_ro[1].legend(loc="lower right")
    axes_ro[1].grid(True, alpha=0.3)
    axes_ro[2].plot(t_axis, readout_o_n, color="C1", drawstyle="steps-post", label=r"spikes $o$")
    axes_ro[2].set_ylabel("Spike")
    axes_ro[2].legend(loc="lower right")
    axes_ro[2].set_ylim(-0.1, 1.2)
    axes_ro[2].grid(True, alpha=0.3)
    axes_ro[3].plot(t_axis, hidden_o_n, color="C2", drawstyle="steps-post", label=rf"hidden spike (neuron {neuron_idx})")
    axes_ro[3].set_ylabel("Hidden spike")
    axes_ro[3].set_xlabel("Time step")
    axes_ro[3].legend(loc="lower right")
    axes_ro[3].set_ylim(-0.1, 1.2)
    axes_ro[3].grid(True, alpha=0.3)
    fig_ro_fwd.suptitle(f"Readout forward pass — output {out_idx} (target class {int(np.array(target).item())})", fontsize=11)
    fig_ro_fwd.tight_layout()
    out_ro_fwd = os.path.join(_SCRIPT_DIR, "heid_readout_forward_one_neuron.png")
    fig_ro_fwd.savefig(out_ro_fwd, dpi=150)
    plt.close(fig_ro_fwd)
    print(f"Saved {out_ro_fwd}")

    # --- Figure: All readout neurons — spike counts and global errors (each drives effective error) ---
    global_errors_np = np.array(global_errors)
    fig_ro_all, (ax_spikes, ax_err) = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    out_neurons = np.arange(network.n_outputs)
    ax_spikes.bar(out_neurons, readout_spikes_per, color="C0", alpha=0.8, label="spike count")
    ax_spikes.axvline(out_idx, color="red", linestyle="--", alpha=0.7, label=f"plotted neuron {out_idx}")
    ax_spikes.set_ylabel("Readout spike count")
    ax_spikes.set_title("Readout spikes per output neuron (all drive effective error)")
    ax_spikes.legend(loc="lower right")
    ax_spikes.grid(True, alpha=0.3, axis="y")
    ax_err.bar(out_neurons, global_errors_np, color="C1", alpha=0.8)
    ax_err.axvline(out_idx, color="red", linestyle="--", alpha=0.7)
    ax_err.set_ylabel("Global error")
    ax_err.set_xlabel("Output neuron index")
    ax_err.set_title("Global error per readout neuron (weight in effective error)")
    ax_err.grid(True, alpha=0.3, axis="y")
    fig_ro_all.suptitle(f"All readout neurons — target class {int(np.array(target).item())}", fontsize=11)
    fig_ro_all.tight_layout()
    out_ro_all = os.path.join(_SCRIPT_DIR, "heid_readout_all_neurons.png")
    fig_ro_all.savefig(out_ro_all, dpi=150)
    plt.close(fig_ro_all)
    print(f"Saved {out_ro_all}")

    # --- Figure: Readout gradient components (one output neuron, same hidden neuron) ---
    fig_ro_grad, axes_ro_g = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    ax_ro = axes_ro_g[0]
    ax_ro.plot(t_axis, sigma_prime_readout_n, color="C0", alpha=0.8, label=r"$\sigma'_{\mathrm{readout}}$")
    ax_ro.axhline(global_err_out, color="C1", linestyle="--", alpha=0.8, label=r"global error (const)")
    ax_ro_twin = ax_ro.twinx()
    ax_ro_twin.plot(t_axis, E_readout_n, color="C2", alpha=0.7, linestyle=":", label=r"$E_{\mathrm{readout}}$ (hidden " + str(neuron_idx) + ")")
    ax_ro_twin.set_ylabel(r"$E_{\mathrm{readout}}$", color="C2")
    ax_ro.set_ylabel("Value")
    ax_ro.set_title(f"Readout gradient components — output {out_idx}, hidden {neuron_idx}")
    ax_ro.legend(loc="lower right")
    ax_ro_twin.legend(loc="lower right")
    ax_ro.grid(True, alpha=0.3)
    axes_ro_g[1].plot(t_axis, readout_per_t, color="C3", label=r"$\sigma' \cdot E_{\mathrm{readout}} \cdot$ global error (per $t$)")
    axes_ro_g[1].set_ylabel("Per-t factor")
    axes_ro_g[1].set_xlabel("Time step")
    axes_ro_g[1].set_title(r"Per-t readout gradient factor for weight (output " + str(out_idx) + r", hidden " + str(neuron_idx) + r"); full grad = (1/T)$\sum_t$ this")
    axes_ro_g[1].legend(loc="lower right")
    axes_ro_g[1].grid(True, alpha=0.3)
    fig_ro_grad.suptitle(f"Readout gradient: grad[out,hid] = (1/T) $\\sum_t$ $\\sigma'$[t,out] $\\cdot$ E[t,hid] $\\cdot$ global_error[out]", fontsize=10)
    fig_ro_grad.tight_layout()
    out_ro_grad = os.path.join(_SCRIPT_DIR, "heid_readout_gradient_components_one_neuron.png")
    fig_ro_grad.savefig(out_ro_grad, dpi=150)
    plt.close(fig_ro_grad)
    print(f"Saved {out_ro_grad}")

    # Per-t gradient figure: plot ABSOLUTE VALUE of the product so "higher" = stronger contribution.
    # When effective_error is negative (as in your run), the signed product is negative, so the curve
    # goes *down* during plateau even though each factor (σ', h', ‖E_soma‖, ‖∂μ/∂w‖) is higher — the
    # magnitude is larger. Plotting |product| makes this consistent with the components plot.
    soma_mag_t = np.abs(soma_component_t)
    dend_mag_t = np.abs(dend_component_t)
    fig3, (ax_s, ax_d) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    y_s_max = max(soma_mag_t.max(), 1e-10)
    ax_s.fill_between(t_axis, 0, y_s_max, where=(h_n > 0.5), alpha=0.2, color="gray", label="plateau (h=1)")
    ax_s.plot(t_axis, soma_mag_t, color="C0", label=r"$|\sigma' \cdot$ effective error$|$")
    ax_s.set_ylabel(r"$|\sigma' \cdot$ effective error$|$")
    ax_s.set_title(f"Per-t somatic gradient factor (magnitude) — neuron {neuron_idx}")
    ax_s.legend(loc="lower right", fontsize=8)
    ax_s.grid(True, alpha=0.3)
    y_d_max = max(dend_mag_t.max(), 1e-10)
    ax_d.fill_between(t_axis, 0, y_d_max, where=(h_n > 0.5), alpha=0.2, color="gray", label="plateau (h=1)")
    ax_d.plot(t_axis, dend_mag_t, color="C1", label=r"$|\gamma \cdot \sigma' \cdot h' \cdot$ err $\cdot \|\partial\mu/\partial w\|$|")
    ax_d.set_ylabel(r"$|$dendritic factor$|$")
    ax_d.set_xlabel("Time step")
    ax_d.set_title(f"Per-t dendritic gradient factor (magnitude) — neuron {neuron_idx}")
    ax_d.legend(loc="lower right", fontsize=8)
    ax_d.grid(True, alpha=0.3)
    fig3.suptitle(
        "Per-timestep gradient factor magnitude (full grad = sum over t). "
        "When effective error is negative, signed product is negative; |product| is shown so plateau = higher contribution.",
        fontsize=9,
    )
    fig3.tight_layout()
    out3 = os.path.join(_SCRIPT_DIR, "heid_gradient_per_t_one_neuron.png")
    fig3.savefig(out3, dpi=150)
    plt.close(fig3)
    print(f"Saved {out3}")

    print("Done.")


if __name__ == "__main__":
    main()
