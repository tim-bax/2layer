"""
Run the N-layer model with 3 hidden layers (3, 2, 1 neurons), mock input with spikes at 10, 110, 210 ms,
and plot the forward pass and backward-pass quantities: one figure per layer, each with one subplot per neuron,
each with forward (dendritic μ,h; somatic v,o) and backward (dendritic σ′, h′, dμ/dw; somatic σ′, E) panels.
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random

from nlayer import JAXEPropNetworkNLayer, JAXTwoCompartmentalLayer

# --- Config ---
T = 300  # ms
n_inputs = 3
layer_sizes = [3, 2, 1]  # L1: 3 neurons, L2: 2 neurons, L3: 1 neuron
n_outputs = 1
spike_times = [10, 110, 210]  # ms

# --- Build network and set custom weights ---
key = random.PRNGKey(42)
net = JAXEPropNetworkNLayer(key, n_inputs=n_inputs, layer_sizes=layer_sizes, n_outputs=n_outputs, T=T)

# Plateau duration 100 ms for all layers (same for all neurons per layer)
plateau_steps = int(100.0 / net.config.dt)  # 100 ms in time steps
for ell, n in enumerate(layer_sizes):
    net.layers[ell].T_p = jnp.ones(n, dtype=jnp.int32) * plateau_steps

# L1: 3 neurons, 3 inputs — dendritic: (1,0,0) for all three (all get input from channel 0 only)
# Somatic: (0,0,1), (0,1,0), (1,0,0) — neuron 0 from input 2, neuron 1 from input 1, neuron 2 from input 0
w_dend_L1 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
w_soma_L1 = np.zeros((3, 3))
# Somatic: row 0 = (0,0,1), row 1 = (0,1,0), row 2 = (1,0,0)
w_soma_L1[0, 2] = 1.0
w_soma_L1[1, 1] = 1.0
w_soma_L1[2, 0] = 1.0

# L2: 2 neurons, 3 inputs — (0,0)=1, (1,0)=1, (0,1)=1
w_dend_L2 = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
w_soma_L2 = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

# L3: 1 neuron, 2 inputs — (0,0)=1, (0,1)=1
w_dend_L3 = np.array([[1.0, 1.0]])
w_soma_L3 = np.array([[1.0, 1.0]])

# Readout: 1 output, 1 input
w_readout = np.array([[1.0]])

params = (
    (jnp.array(w_dend_L1), jnp.array(w_dend_L2), jnp.array(w_dend_L3)),
    (jnp.array(w_soma_L1), jnp.array(w_soma_L2), jnp.array(w_soma_L3)),
    jnp.array(w_readout),
)
net.set_params(params)

# --- Mock input: 3 channels, spikes at 10, 110, 210 ms ---
x_input = np.zeros((T, n_inputs))
for ch, t_spike in enumerate(spike_times):
    if t_spike < T:
        x_input[t_spike, ch] = 1.0
x_input = jnp.array(x_input)

# --- Forward pass (get per-layer trajectories) ---
layer_outputs, readout_v, readout_o = net._forward_with_params(params, x_input)

# layer_outputs[ell] = (mu, v_fin, h, o, mu_hist, t_prime, v_hist)
# mu_hist, v_hist, h, o are (T, n_neurons)

# --- Backward-pass quantities (σ′, h′, E_soma, dμ/dw) per layer ---
gamma = net.config.gamma
sigma_primes = []
h_primes = []
E_somas = []
dmu_dws = []
for ell in range(net.L):
    mu_ell, v_fin_ell, h_ell, o_ell, mu_hist_ell, t_prime_ell, v_hist_ell = layer_outputs[ell]
    inp_ell = x_input if ell == 0 else layer_outputs[ell - 1][3]
    soma_in_ell = v_hist_ell + gamma * h_ell - net.config.v_th
    sigma_primes.append(
        np.asarray(JAXTwoCompartmentalLayer.surrogate_sigma(soma_in_ell, net.config.beta_s))
    )
    t_prime_int = t_prime_ell.astype(jnp.int32)
    n_ell = net.layer_sizes[ell]
    mu_at_tprime = mu_hist_ell[t_prime_int, jnp.arange(n_ell)]
    h_primes.append(
        np.asarray(JAXTwoCompartmentalLayer.surrogate_sigma(
            mu_at_tprime - net.config.mu_th, net.config.beta_d
        ))
    )
    E_somas.append(
        np.asarray(JAXTwoCompartmentalLayer.compute_eligibility_traces(
            inp_ell, net.layers[ell].alpha_s
        ))
    )
    dmu_dws.append(
        np.asarray(JAXTwoCompartmentalLayer.compute_dmu_tprime_dw(
            inp_ell, h_ell, t_prime_ell, net.layers[ell].alpha
        ))
    )
# Shapes: sigma_primes[ell] (T, n_ell), h_primes[ell] (T, n_ell), E_somas[ell] (T, n_in), dmu_dws[ell] (T, n_ell, n_in)

# --- Time nesting (nlayer._compute_gradients) ---
# L3 dendritic gradient: uses σ′₃, h′₃, dμ/dw₃ (main contribution at t′₃).
# Gradient through L3 dend → L2: L2's σ′, h′, E, dμ/dw are evaluated at t′₃ (nlayer line 238-241).
# Gradient through L2 dend → L1: L1's σ′, h′, E, dμ/dw are evaluated at t′₂.
# So: L3 dend at t′₃; L2 (for that path) at t′₃; L1 (for that path) at t′₂. Nesting: t′₁ at t′₂ at t′₃.
# Plots show the *forward* trajectories σ′(t), h′(t), dμ/dw(t): trace length = this layer's plateau only.
# The "effective" span for L1 in the dend-dend-dend path is t′₁→t′₃ (same value from t′₁ used in the sum
# over t up to t′₃), but that long span is NOT the length of the dμ/dw trace — we depict the forward series.

# Plateau start times per layer (one per neuron), for vertical lines on backward plots
t_prime_final = [
    np.asarray(layer_outputs[ell][5])[-1, :] for ell in range(net.L)
]  # t_prime_final[ell] shape (layer_sizes[ell],)

# --- Plotting: one figure per layer (4 cols: forward dend, forward soma, backward dend, backward soma) ---
t_ms = np.arange(T)

for ell, n_neurons in enumerate(layer_sizes):
    mu, v_fin, h, o, mu_hist, t_prime, v_hist = layer_outputs[ell]
    mu_hist = np.asarray(mu_hist)
    v_hist = np.asarray(v_hist)
    h = np.asarray(h)
    o = np.asarray(o)
    sp = sigma_primes[ell]   # (T, n_neurons)
    hp = h_primes[ell]       # (T, n_neurons)
    E = E_somas[ell]         # (T, n_in)
    dmu_dw = dmu_dws[ell]    # (T, n_neurons, n_in)
    n_in = E.shape[1]

    # Times at which this layer's backward quantities are sampled when gradient comes from above
    t_self = t_prime_final[ell]           # this layer's t′ (plateau start)
    t_above = t_prime_final[ell + 1] if ell < net.L - 1 else None  # layer above's t′

    fig, axes = plt.subplots(n_neurons, 4, figsize=(16, 2.5 * n_neurons), squeeze=False)
    fig.suptitle(f"Layer L{ell + 1} (n={n_neurons}): forward + backward", fontsize=12)

    for n in range(n_neurons):
        # Col 0: Forward dendritic (μ, h)
        ax = axes[n, 0]
        ax.plot(t_ms, mu_hist[:, n], label=r"$\mu$", color="C0")
        ax.plot(t_ms, h[:, n], label="$h$", color="C1")
        ax.set_ylabel(f"Neuron {n}")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("Fwd: dendritic ($\\mu$, $h$)")
        ax.grid(True, alpha=0.3)

        # Col 1: Forward somatic (v, o)
        ax = axes[n, 1]
        ax.plot(t_ms, v_hist[:, n], label="$v$", color="C2")
        ax_t = ax.twinx()
        ax_t.fill_between(t_ms, 0, o[:, n], alpha=0.4, color="C3", label="$o$")
        ax_t.set_ylim(-0.1, 1.5)
        ax.set_ylabel(f"Neuron {n}")
        ax.legend(loc="upper left", fontsize=7)
        ax_t.legend(loc="upper right", fontsize=7)
        ax.set_title("Fwd: somatic ($v$, $o$)")
        ax.grid(True, alpha=0.3)

        # Col 2: Backward dendritic (σ′, h′, dμ/dw) — forward trajectory (trace length = this layer's plateau)
        # Effective grad window: same value (from t′_L) is used when sampled at t′_{L+1}; shade t′_L..t′_{L+1}
        ax = axes[n, 2]
        if t_above is not None:
            t_lo, t_hi = min(t_self), max(t_above)
            ax.axvspan(t_lo, t_hi, alpha=0.12, color="gray", label="effective grad window (dend path)")
        ax.plot(t_ms, sp[:, n], label=r"$\sigma'$", color="C0")
        ax.plot(t_ms, hp[:, n], label="$h'$", color="C1")
        for j in range(n_in):
            ax.plot(t_ms, dmu_dw[:, n, j], label=rf"$d\mu/dw_{{.,{j}}}$", alpha=0.8)
        for t in t_self:
            ax.axvline(t, color="gray", linestyle="--", alpha=0.7, linewidth=0.8)
        if t_above is not None:
            for t in t_above:
                ax.axvline(t, color="black", linestyle=":", alpha=0.8, linewidth=1)
        ax.set_ylabel(f"Neuron {n}")
        ax.legend(loc="upper right", fontsize=6)
        ax.set_title("Bwd: dendritic — trace = fwd plateau; shaded = effective window (t'$_L$..t'$_{L+1}$)")
        ax.grid(True, alpha=0.3)

        # Col 3: Backward somatic (σ′, E)
        ax = axes[n, 3]
        if t_above is not None:
            t_lo, t_hi = min(t_self), max(t_above)
            ax.axvspan(t_lo, t_hi, alpha=0.12, color="gray")
        ax.plot(t_ms, sp[:, n], label=r"$\sigma'$", color="C0")
        for j in range(n_in):
            ax.plot(t_ms, E[:, j], label=f"$E_{{in{j}}}$", alpha=0.8)
        for t in t_self:
            ax.axvline(t, color="gray", linestyle="--", alpha=0.7, linewidth=0.8)
        if t_above is not None:
            for t in t_above:
                ax.axvline(t, color="black", linestyle=":", alpha=0.8, linewidth=1)
        ax.set_ylabel(f"Neuron {n}")
        ax.legend(loc="upper right", fontsize=6)
        ax.set_title("Bwd: somatic — gray $t'_{L}$, black $t'_{L+1}$ (sampled)")
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (ms)")
    plt.tight_layout()
    out_path = os.path.join(_SCRIPT_DIR, f"forward_backward_L{ell + 1}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # Forward-only figure (same data): Dendritic (μ, h) + Somatic (v, o) — used for forward_L*.png
    fig_fwd, axes_fwd = plt.subplots(n_neurons, 2, figsize=(10, 2.5 * n_neurons), squeeze=False)
    fig_fwd.suptitle(f"Layer L{ell + 1} (n={n_neurons})", fontsize=12)
    for n in range(n_neurons):
        ax = axes_fwd[n, 0]
        ax.plot(t_ms, mu_hist[:, n], label=r"$\mu$", color="C0")
        ax.plot(t_ms, h[:, n], label="$h$", color="C1")
        ax.set_ylabel(f"Neuron {n}")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("Dendritic")
        ax.grid(True, alpha=0.3)
        ax = axes_fwd[n, 1]
        ax.plot(t_ms, v_hist[:, n], label="$v$", color="C2")
        ax_t = ax.twinx()
        ax_t.fill_between(t_ms, 0, o[:, n], alpha=0.4, color="C3", label="$o$")
        ax_t.set_ylim(-0.1, 1.5)
        ax.set_ylabel(f"Neuron {n}")
        ax.legend(loc="upper left", fontsize=7)
        ax_t.legend(loc="upper right", fontsize=7)
        ax.set_title("Somatic")
        ax.grid(True, alpha=0.3)
    for ax in axes_fwd[-1, :]:
        ax.set_xlabel("Time (ms)")
    plt.tight_layout()
    out_fwd = os.path.join(_SCRIPT_DIR, f"forward_L{ell + 1}.png")
    plt.savefig(out_fwd, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_fwd}")

print("Done.")
