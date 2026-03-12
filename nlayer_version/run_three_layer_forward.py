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

# L1: 3 neurons, 3 inputs — each neuron i gets input i with weight 1
w_dend_L1 = np.zeros((3, 3))
w_soma_L1 = np.zeros((3, 3))
for i in range(3):
    w_dend_L1[i, i] = 1.0
    w_soma_L1[i, i] = 1.0

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

        # Col 2: Backward dendritic (σ′, h′, dμ/dw for this neuron)
        ax = axes[n, 2]
        ax.plot(t_ms, sp[:, n], label=r"$\sigma'$", color="C0")
        ax.plot(t_ms, hp[:, n], label="$h'$", color="C1")
        for j in range(n_in):
            ax.plot(t_ms, dmu_dw[:, n, j], label=rf"$d\mu/dw_{{.,{j}}}$", alpha=0.8)
        ax.set_ylabel(f"Neuron {n}")
        ax.legend(loc="upper right", fontsize=6)
        ax.set_title("Bwd: dendritic ($\\sigma'$, $h'$, $d\\mu/dw$)")
        ax.grid(True, alpha=0.3)

        # Col 3: Backward somatic (σ′ for this neuron, E from input)
        ax = axes[n, 3]
        ax.plot(t_ms, sp[:, n], label=r"$\sigma'$", color="C0")
        for j in range(n_in):
            ax.plot(t_ms, E[:, j], label=f"$E_{{in{j}}}$", alpha=0.8)
        ax.set_ylabel(f"Neuron {n}")
        ax.legend(loc="upper right", fontsize=6)
        ax.set_title("Bwd: somatic ($\\sigma'$, $E$)")
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (ms)")
    plt.tight_layout()
    out_path = os.path.join(_SCRIPT_DIR, f"forward_backward_L{ell + 1}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

print("Done.")
