"""
Sample: causal chain L1 plateau→spike → L2 plateau→spike → L3 plateau→spike (with external + somatic
input during plateaus), and the effective L1 dμ/dw integrand time series for all 8 gradient paths
(soma/dend at L3, L2, L1) with correct inherited times. Includes a tree diagram and 8 integrand plots.

What is going on:
- Forward: Input spikes at 10, 110, 210 ms drive L1→L2→L3→readout. Each layer uses two-compartment
  neurons: dendritic potential μ, plateau h (0/1), somatic v, spike o. When μ ≥ μ_th, h=1 for T_p steps.
- Plateau length: T_p is set to 100 time steps; with config.dt=1 ms that is 100 ms per plateau.
- t′ (t_prime): For each neuron, t′ is the time step when its current plateau started. During the
  plateau, t′ stays fixed; after the plateau ends, t′ is updated to current t each step (so at the
  end of the trial you see t′ = 299 for neurons that are no longer in a plateau).
- Backward: The gradient is a sum over time t. Each of the 8 paths is soma (s) or dendrite (d) at
  L3, L2, L1. L1 dμ/dw only appears on paths 2,4,6,8 (L1 dend). The integrand we plot is the
  contribution to that path’s gradient at each t (scaled by γ, no 1/T).
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

# --- Config: same as run_three_layer_forward so forward pass produces clear activity ---
T = 300
n_inputs = 3
layer_sizes = [3, 2, 1]
n_outputs = 2   # need ≥2 so global_errors = target_oh - probs is non-zero (with 1 output probs≡1)
target = 0
spike_times = [10, 110, 210]

key = random.PRNGKey(42)
net = JAXEPropNetworkNLayer(key, n_inputs=n_inputs, layer_sizes=layer_sizes, n_outputs=n_outputs, T=T)
plateau_steps = int(100.0 / net.config.dt)
for ell, n in enumerate(layer_sizes):
    net.layers[ell].T_p = jnp.ones(n, dtype=jnp.int32) * plateau_steps
# Plateau length = plateau_steps * dt = 100 ms (dt=1 ms by default)

# Same weights as run_three_layer_forward (L1 dend all from ch0; soma cross-wired; L2,L3 identity-like)
w_dend_L1 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
w_soma_L1 = np.zeros((3, 3))
w_soma_L1[0, 2], w_soma_L1[1, 1], w_soma_L1[2, 0] = 1.0, 1.0, 1.0
w_dend_L2 = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
w_soma_L2 = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
w_dend_L3 = np.array([[1.0, 1.0]])
w_soma_L3 = np.array([[1.0, 1.0]])
w_readout = np.array([[1.0], [0.5]])  # (n_outputs, n_L3): class 0 driven more so probs ≠ [1,0]

params = (
    (jnp.array(w_dend_L1), jnp.array(w_dend_L2), jnp.array(w_dend_L3)),
    (jnp.array(w_soma_L1), jnp.array(w_soma_L2), jnp.array(w_soma_L3)),
    jnp.array(w_readout),
)
net.set_params(params)

x_input = np.zeros((T, n_inputs))
for ch, t in enumerate(spike_times):
    if t < T:
        x_input[t, ch] = 1.0
x_input = jnp.array(x_input)

# --- Forward pass ---
layer_outputs, readout_v, readout_o = net._forward_with_params(params, x_input)
dend, soma, w_readout = params
gamma = net.config.gamma

# --- Backward quantities (same as nlayer) ---
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
        np.asarray(JAXTwoCompartmentalLayer.surrogate_sigma(mu_at_tprime - net.config.mu_th, net.config.beta_d))
    )
    E_somas.append(
        np.asarray(JAXTwoCompartmentalLayer.compute_eligibility_traces(inp_ell, net.layers[ell].alpha_s))
    )
    dmu_dws.append(
        np.asarray(JAXTwoCompartmentalLayer.compute_dmu_tprime_dw(
            inp_ell, h_ell, t_prime_ell, net.layers[ell].alpha
        ))
    )

# Full t_prime history (T, n_ell) for time-varying paths; last row for paths that need a single t′
t_prime_1_history = np.asarray(layer_outputs[0][5])   # (T, n_1)
t_prime_2_history = np.asarray(layer_outputs[1][5])   # (T, n_2)
t_prime_3_history = np.asarray(layer_outputs[2][5])   # (T, n_3)
t_prime_1_int = t_prime_1_history[-1, :]  # (3,) for L1-at-t′₁/t′₂ indexing
t_prime_2_int = t_prime_2_history[-1, :]  # (2,)
t_prime_3_int = t_prime_3_history[-1, :]  # (1,)
n_1, n_2, n_3 = layer_sizes[0], layer_sizes[1], layer_sizes[2]

# Global error and effective error at L3
global_errors = np.asarray(net.compute_global_errors(readout_o, target))
effective_3 = np.asarray(
    jnp.einsum("tj,j,ji->ti", JAXTwoCompartmentalLayer.surrogate_sigma(readout_v - net.config.v_th, net.config.beta_s), global_errors, w_readout)
)
# effective_3 (T, 1)

# --- Per-path effective errors at L2 and L1, then L1 dμ/dw integrand ---
# Path index: (L3 branch, L2 branch, L1 branch) 0=soma, 1=dend → path 0..7
# Path 1 (0,0,0): L3s L2s L1s all t
# Path 2 (0,0,1): L3s L2s L1d all t for L3,L2; L1 dend at t′₁
# Path 3 (0,1,0): L3s L2d L1s L2 at t′₂, L1 soma at t′₂
# Path 4 (0,1,1): L3s L2d L1d L2 at t′₂, L1 at t′₂
# Path 5 (1,0,0): L3d L2s L1s at t′₃
# Path 6 (1,0,1): L3d L2s L1d at t′₃ for L3,L2; L1 dend at t′₁
# Path 7 (1,1,0): L3d L2d L1s at t′₃, L2 at t′₃, L1 at t′₃
# Path 8 (1,1,1): L3d L2d L1d at t′₃, L2 at t′₃, L1 at t′₂

sp1, sp2, sp3 = sigma_primes[0], sigma_primes[1], sigma_primes[2]
hp1, hp2, hp3 = h_primes[0], h_primes[1], h_primes[2]
dmu1 = dmu_dws[0]  # (T, n_1, n_inputs)
tp1 = np.asarray(t_prime_1_int, dtype=int)
tp2 = np.asarray(t_prime_2_int, dtype=int)
tp3 = np.asarray(t_prime_3_int, dtype=int)

# L1 at t′₁ (one time per L1 neuron)
sp1_at_t1 = sp1[tp1, np.arange(n_1)]
hp1_at_t1 = hp1[tp1, np.arange(n_1)]
dmu1_at_t1 = dmu1[tp1, np.arange(n_1), :]  # (n_1, n_inputs)

# L1 at t′₂ (one time per L2 neuron)
sp1_at_t2 = sp1[tp2]
hp1_at_t2 = hp1[tp2]
dmu1_at_t2 = dmu1[tp2]  # (n_2, n_1, n_inputs)

W_s2, W_d2 = np.asarray(soma[1]), np.asarray(dend[1])
W_s3, W_d3 = np.asarray(soma[2]), np.asarray(dend[2])

e2_soma = np.einsum("ik,ti,ti->tk", W_s3, sp3, effective_3)  # (T, n_2)
e1_from_L2_soma = np.einsum("ik,ti,ti->tk", W_s2, sp2, e2_soma)  # (T, n_1)

integrand_L1_dend = np.zeros((8, T))
scale_plot = gamma

# Path 2: L1 dend at t′₁; (T,n_1,1)*(1,n_1,n_in) -> (T,n_1,n_in)
part2 = (e1_from_L2_soma * sp1 * hp1_at_t1[None, :])[:, :, None] * dmu1_at_t1[None, :, :]
integrand_L1_dend[1] = np.sum(part2 * scale_plot, axis=(1, 2))

# Path 4: L1 dend at t′₂; need (T, n_2, n_1, n_in) then sum over (1,2,3)
sp2_at_t2 = sp2[tp2, np.arange(n_2)]
hp2_at_t2 = hp2[tp2, np.arange(n_2)]
coeff_L2d = W_d2 * sp2_at_t2[:, None] * hp2_at_t2[:, None]
term4 = (e2_soma[:, :, None] * coeff_L2d[None] * sp1_at_t2[None] * hp1_at_t2[None])[:, :, :, None] * dmu1_at_t2[None]
integrand_L1_dend[3] = np.sum(term4 * scale_plot, axis=(1, 2, 3))

# Path 6: at each t, L3 dend at t′₃(t); e_1 at t′₃(t), L1 dend at t′₁ (so integrand non-zero when L3 plateau active)
for t in range(T):
    t3 = int(t_prime_3_history[t, 0])
    _e2_dend_t = (sp3[t3, :] * hp3[t3, :] * effective_3[t3, :]).flatten()
    e2_dend_coeff_t = (W_d3.T @ _e2_dend_t) if _e2_dend_t.size > 0 else np.zeros(n_2)
    e1_at_t = (W_s2 * sp2[t3, :][:, None] * e2_dend_coeff_t[:, None]).sum(axis=0)
    integrand_L1_dend[5, t] = np.sum(e1_at_t * sp1_at_t1 * hp1_at_t1 * dmu1_at_t1 * scale_plot)

# Path 8: at each t, L3 dend at t′₃(t), L2 and L1 at t′₂(t′₃(t))
for t in range(T):
    t3 = int(t_prime_3_history[t, 0])
    t2 = t_prime_2_history[t3, :]  # (n_2,)
    _e2_dend_t = (sp3[t3, :] * hp3[t3, :] * effective_3[t3, :]).flatten()
    e2_dend_coeff_t = (W_d3.T @ _e2_dend_t) if _e2_dend_t.size > 0 else np.zeros(n_2)
    sp2_t2 = sp2[t2, np.arange(n_2)]
    hp2_t2 = hp2[t2, np.arange(n_2)]
    sp1_t2 = sp1[t2]           # (n_2, n_1)
    hp1_t2 = hp1[t2]
    dmu1_t2 = dmu1[t2]         # (n_2, n_1, n_in)
    coeff_t = (e2_dend_coeff_t[:, None] * W_d2 * sp2_t2[:, None] * hp2_t2[:, None]
               * sp1_t2 * hp1_t2 * scale_plot)
    integrand_L1_dend[7, t] = np.sum(coeff_t[:, :, None] * dmu1_t2)

# --- Diagnostics: are quantities non-zero? ---
print("Diagnostics (max abs):")
print(f"  global_errors: {np.max(np.abs(global_errors)):.2e}")
print(f"  effective_3:   {np.max(np.abs(effective_3)):.2e}")
print(f"  e2_soma:       {np.max(np.abs(e2_soma)):.2e}")
print(f"  e1_from_L2_soma: {np.max(np.abs(e1_from_L2_soma)):.2e}")
print(f"  sp1, hp1_at_t1, dmu1_at_t1: {np.max(np.abs(sp1)):.2e}, {np.max(np.abs(hp1_at_t1)):.2e}, {np.max(np.abs(dmu1_at_t1)):.2e}")
for p in range(8):
    m = np.max(np.abs(integrand_L1_dend[p]))
    print(f"  path {p} integrand: {m:.2e}")
# Normalize for display: scale so at least one path has visible range (avoid division by zero)
total_scale = np.max(np.abs(integrand_L1_dend))
if total_scale < 1e-15:
    total_scale = 1.0
integrand_display = integrand_L1_dend / total_scale  # same shape, max abs = 1 for visibility

# --- Tree diagram (print to console; see explanation in docstring / chat) ---
TREE_TEXT = r"""
Gradient path tree (readout δ → L3 → L2 → L1). Each branch: soma (s) or dendrite (d). Label = inherited time.

                                    δ
                                   / \
                          L3 soma/t    L3 dend/t′₃
                          /     \           /    \
                    L2s/t    L2d/t′₂   L2s/t′₃  L2d/t′₃
                    /  \      /  \       /  \      /  \
                  L1s  L1d  L1s  L1d   L1s  L1d  L1s  L1d
                  t   t′₁   t′₂  t′₂   t′₃  t′₁  t′₃  t′₂

Paths 1–8 (left to right): (L3s,L2s,L1s), (L3s,L2s,L1d), (L3s,L2d,L1s), (L3s,L2d,L1d),
                           (L3d,L2s,L1s), (L3d,L2s,L1d), (L3d,L2d,L1s), (L3d,L2d,L1d).
L1 dμ/dw only non-zero for paths 2,4,6,8 (L1 dend). Path 1,3,5,7 use L1 soma (E₁), so integrand = 0.
"""
print(TREE_TEXT)

path_names = [
    "L3s→L2s→L1s (all t)",
    "L3s→L2s→L1d (t′₁)",
    "L3s→L2d→L1s (t′₂)",
    "L3s→L2d→L1d (t′₂)",
    "L3d→L2s→L1s (t′₃)",
    "L3d→L2s→L1d (t′₁@t′₃)",
    "L3d→L2d→L1s (t′₃)",
    "L3d→L2d→L1d (t′₂@t′₃)",
]

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
t_ms = np.arange(T)
for p in range(8):
    ax = axes[p // 4, p % 4]
    ax.plot(t_ms, integrand_display[p], color="C0")
    for t in np.unique(tp1):
        ax.axvline(t, color="gray", linestyle="--", alpha=0.5)
    for t in np.unique(tp2):
        ax.axvline(t, color="gray", linestyle=":", alpha=0.5)
    for t in np.unique(tp3):
        ax.axvline(t, color="black", linestyle=":", alpha=0.7)
    ax.set_title(path_names[p], fontsize=8)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("integrand (norm)")
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)
plt.suptitle(f"L1 dμ/dw integrand per path (normalized by max abs = {total_scale:.2e}); paths 1,3,5,7 = 0 (L1 soma)", fontsize=10)
plt.tight_layout()
out = os.path.join(_SCRIPT_DIR, "gradient_paths_eight_integrands.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

dt_ms = float(net.config.dt)
print(f"Plateau: T_p = {plateau_steps} steps = {plateau_steps * dt_ms:.0f} ms (dt = {dt_ms} ms)")
print(f"t′₁={tp1}, t′₂={tp2}, t′₃={tp3}")
print(f"L1 spikes (sum o): {np.sum(layer_outputs[0][3])}, L2: {np.sum(layer_outputs[1][3])}, L3: {np.sum(layer_outputs[2][3])}, readout: {np.sum(readout_o)}")
print("Done.")
