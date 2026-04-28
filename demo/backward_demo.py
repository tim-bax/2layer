#!/usr/bin/env python3
"""
Backward-pass demo for the same one-neuron setup as demo.py.

For each compartment (dendrite and soma), plot three rows:
  1. Eligibility trace
        - Soma:     dv/dw        = E_soma
        - Dendrite: dmu_t'/dw    (frozen at plateau start — the e-prop one)
  2. Surrogate gradient
        - Soma:     do/dv         = sigma'(v + gamma*h - v_th)
        - Dendrite: dh/dmu_t'     = sigma'(mu_at_t' - mu_th)
  3. Total gradient per timestep = surrogate * eligibility
        (error term intentionally omitted — this is the local factor only)

Three input channels are shown as separate colored lines.

Outputs:
  demo/backward_dendrite.png
  demo/backward_soma.png
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
_NO_HISTORY = os.path.join(_ROOT, "no_history")
if _NO_HISTORY not in sys.path:
    sys.path.insert(0, _NO_HISTORY)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import NeuronConfig, surrogate_sigma  # noqa: E402
from two_comp_neuron import TwoCompNeuron  # noqa: E402
from demo import build_mock_input  # noqa: E402  (reuse mock-input helper)


def run_backward():
    """Replay forward pass + record all quantities needed for backward plots."""
    config = NeuronConfig()
    T = 300
    spike_times = [10, 80, 200]
    n_inputs = 3
    n_neurons = 1

    w_dend = jnp.array([[0.7, 1.0, 0.2]])
    w_soma = jnp.array([[0.6, 0.3, 0.5]])

    neuron = TwoCompNeuron(jax.random.PRNGKey(0), n_neurons, n_inputs, config)
    neuron.w_dend = w_dend
    neuron.w_soma = w_soma
    T_p = jnp.array([150], dtype=jnp.int32)

    x_input = build_mock_input(T, spike_times, n_inputs)
    dend_inputs = x_input @ w_dend.T
    soma_inputs = x_input @ w_soma.T

    # same tiny noise as the forward demo (same seed)
    rng = np.random.default_rng(0)
    noise_std = 0.01
    dend_inputs = dend_inputs + jnp.array(rng.normal(0, noise_std, size=(T, 1)))
    soma_inputs = soma_inputs + jnp.array(rng.normal(0, noise_std, size=(T, 1)))

    alpha_s = neuron.alpha_s
    alpha_d = neuron.alpha_d

    carry = neuron.init_carry()
    # eligibility state — initialise to zeros
    E_soma = jnp.zeros(n_inputs)
    dmu_dw = jnp.zeros((n_neurons, n_inputs))
    dmu_dw_atp = jnp.zeros((n_neurons, n_inputs))

    v_pre_hist = np.zeros(T)
    h_hist = np.zeros(T, dtype=np.float64)
    mu_atp_hist = np.zeros(T)
    E_soma_hist = np.zeros((T, n_inputs))
    dmu_atp_hist = np.zeros((T, n_inputs))

    for t in range(T):
        carry, o, v_pre_reset, h, h_prev, mu_at_tp = TwoCompNeuron.forward_step(
            carry,
            dend_inputs[t],
            soma_inputs[t],
            jnp.array(t, dtype=jnp.int32),
            alpha_s,
            alpha_d,
            T_p,
            config,
        )

        x_t = x_input[t].astype(jnp.float64)
        E_soma = TwoCompNeuron.update_somatic_eligibility(E_soma, x_t, alpha_s)
        dmu_dw, dmu_dw_atp = TwoCompNeuron.update_dendritic_eligibility(
            dmu_dw, dmu_dw_atp, x_t, h_prev, alpha_d,
        )

        v_pre_hist[t] = float(v_pre_reset[0])
        h_hist[t] = float(h[0])
        mu_atp_hist[t] = float(mu_at_tp[0])
        E_soma_hist[t] = np.asarray(E_soma)
        dmu_atp_hist[t] = np.asarray(dmu_dw_atp[0])

    # Surrogate gradients (same formulas as used in the training code)
    v_input_vals = v_pre_hist + config.gamma * h_hist - config.v_th
    sp_hidden = np.asarray(
        surrogate_sigma(jnp.array(v_input_vals), config.beta_s)
    )
    dend_input_vals = mu_atp_hist - config.mu_th
    hp_hidden = np.asarray(
        surrogate_sigma(jnp.array(dend_input_vals), config.beta_d)
    )

    soma_total = sp_hidden[:, None] * E_soma_hist      # (T, n_inputs)
    # Dendritic credit reaches the output via the dynamic threshold
    # v_th - gamma*h, so do/dh = gamma * sp_hidden. Full chain (error omitted):
    #   gamma * sp_hidden * hp_hidden * dmu_t'/dw
    dend_total = (
        config.gamma
        * sp_hidden[:, None]
        * hp_hidden[:, None]
        * dmu_atp_hist
    )                                                   # (T, n_inputs)

    return {
        "T": T,
        "config": config,
        "spike_times": spike_times,
        "h": h_hist,
        "E_soma": E_soma_hist,
        "sp_hidden": sp_hidden,
        "soma_total": soma_total,
        "dmu_atp": dmu_atp_hist,
        "hp_hidden": hp_hidden,
        "dend_total": dend_total,
    }


def _plot_compartment(res: dict, which: str, out_path: str):
    T = res["T"]
    t_axis = np.arange(T)
    colors = ["C0", "C2", "C4"]
    labels = [f"ch{k}" for k in range(3)]

    if which == "dend":
        elig = res["dmu_atp"]
        surrogate = res["hp_hidden"]
        total = res["dend_total"]
        elig_label = r"$\partial \mu_{t'}/\partial w_\mathrm{dend}$"
        surr_label = r"$\partial h / \partial \mu_{t'}$"
        total_label = (
            r"$\gamma \cdot \partial o/\partial v "
            r"\cdot \partial h / \partial \mu_{t'} "
            r"\cdot \partial \mu_{t'}/\partial w_\mathrm{dend}$"
        )
        title_pre = "Dendrite"
    elif which == "soma":
        elig = res["E_soma"]
        surrogate = res["sp_hidden"]
        total = res["soma_total"]
        elig_label = r"$\partial v/\partial w_\mathrm{soma}$"
        surr_label = r"$\partial o/\partial v$"
        total_label = (
            r"$\partial o/\partial v \cdot "
            r"\partial v/\partial w_\mathrm{soma}$"
        )
        title_pre = "Soma"
    else:
        raise ValueError(which)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    # 1. Eligibility trace — one line per input channel
    for k in range(3):
        axes[0].plot(t_axis, elig[:, k], color=colors[k], label=labels[k])
    axes[0].set_ylabel(elig_label)
    axes[0].set_title(f"{title_pre}: eligibility trace")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 2. Surrogate gradient — scalar (single line)
    axes[1].plot(t_axis, surrogate, color="C3")
    axes[1].set_ylabel(surr_label)
    axes[1].set_title(f"{title_pre}: surrogate gradient")
    axes[1].grid(True, alpha=0.3)

    # 3. Total per-timestep gradient = surrogate * eligibility (error omitted)
    for k in range(3):
        axes[2].plot(t_axis, total[:, k], color=colors[k], label=labels[k])
    axes[2].set_ylabel(total_label)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title(f"{title_pre}: total gradient per timestep (error omitted)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    res = run_backward()
    _plot_compartment(
        res, "dend", os.path.join(_SCRIPT_DIR, "backward_dendrite.png")
    )
    _plot_compartment(
        res, "soma", os.path.join(_SCRIPT_DIR, "backward_soma.png")
    )


if __name__ == "__main__":
    main()
