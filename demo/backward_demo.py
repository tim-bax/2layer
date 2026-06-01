#!/usr/bin/env python3
"""
Backward-pass demo for the same one-neuron setup as demo.py.

Soma figure (3 rows): surrogate gradient, somatic eligibility, total local factor.

Dendrite figure (4 rows): somatic surrogate (reference), dendritic surrogate,
dendritic eligibility (∂μ_{t'}/∂w), dendritic total local factor.

Each subplot row uses the same height in inches in both figures (``_BACKWARD_ROW_H_IN``).

Five input channels are shown as separate colored lines in multi-channel panels.

Outputs:
  demo/backward_dendrite.png and .svg
  demo/backward_soma.png and .svg
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

_TRACE_LW = 2.2
_DEND_SURR_COLOR = "#2ca02c"
# Same vertical size per row in soma (3 panels) and dendrite (4 panels) figures.
_BACKWARD_ROW_H_IN = 3.15


def run_backward():
    """Replay forward pass + record all quantities needed for backward plots."""
    config = NeuronConfig()
    T = 300
    spike_times = [10, 80, 200, 65, 250]
    n_inputs = 5
    n_neurons = 1

    w_dend = jnp.array([[0.7, 1.1, 0.2, 0.0, 0.3]])
    w_soma = jnp.array([[0.8, 0.3, 0.5, 0.3, 1.0]])

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
        "n_inputs": n_inputs,
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


def _channel_colors(n_in: int):
    tab = plt.cm.tab10.colors
    return [tab[i % len(tab)] for i in range(n_in)]


def _plot_compartment(res: dict, which: str, out_path: str):
    T = res["T"]
    t_axis = np.arange(T)

    if which == "dend":
        title_pre = "Dendrite"
        soma_surrogate = res["sp_hidden"]
        surrogate = res["hp_hidden"]
        dend_elig = res["dmu_atp"]
        total = res["dend_total"]
        soma_surr_label = r"$\partial o/\partial v$"
        surr_label = r"$\partial h / \partial \mu_{t'}$"
        dend_elig_label = r"$\partial \mu_{t'}/\partial w_\mathrm{dend}$"
        total_label = (
            r"$\gamma \cdot \partial o/\partial v "
            r"\cdot \partial h / \partial \mu_{t'} "
            r"\cdot \partial \mu_{t'}/\partial w_\mathrm{dend}$"
        )
        nrows = 4
    elif which == "soma":
        title_pre = "Soma"
        surrogate = res["sp_hidden"]
        soma_elig = res["E_soma"]
        total = res["soma_total"]
        surr_label = r"$\partial o/\partial v$"
        soma_elig_label = r"$\partial v/\partial w_\mathrm{soma}$"
        total_label = (
            r"$\partial o/\partial v \cdot "
            r"\partial v/\partial w_\mathrm{soma}$"
        )
        nrows = 3
    else:
        raise ValueError(which)

    fig_h = nrows * _BACKWARD_ROW_H_IN
    n_in = int(res["n_inputs"])
    colors = _channel_colors(n_in)
    labels = [f"ch{k}" for k in range(n_in)]

    fig, axes = plt.subplots(
        nrows,
        1,
        sharex=True,
        figsize=(10, fig_h),
        gridspec_kw={"height_ratios": [1.0] * nrows},
    )
    surr_color = _DEND_SURR_COLOR if which == "dend" else "C3"

    def _plot_multiline(ax, ydata, ylabel, ptitle, legend=True):
        for k in range(n_in):
            ax.plot(
                t_axis, ydata[:, k], color=colors[k], label=labels[k],
                linewidth=_TRACE_LW,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(ptitle)
        if legend:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    i = 0
    if which == "dend":
        axes[i].plot(t_axis, soma_surrogate, color="C3", linewidth=_TRACE_LW)
        axes[i].set_ylabel(soma_surr_label)
        axes[i].set_title("Somatic surrogate gradient (same hidden neuron)")
        axes[i].grid(True, alpha=0.3)
        i += 1

    axes[i].plot(t_axis, surrogate, color=surr_color, linewidth=_TRACE_LW)
    axes[i].set_ylabel(surr_label)
    axes[i].set_title(f"{title_pre}: surrogate gradient")
    axes[i].grid(True, alpha=0.3)
    i += 1

    if which == "dend":
        _plot_multiline(
            axes[i],
            dend_elig,
            dend_elig_label,
            f"{title_pre}: eligibility trace",
        )
        i += 1
    else:
        _plot_multiline(
            axes[i],
            soma_elig,
            soma_elig_label,
            f"{title_pre}: eligibility trace",
        )
        i += 1

    _plot_multiline(
        axes[i],
        total,
        total_label,
        f"{title_pre}: total gradient per timestep (error omitted)",
    )
    axes[i].set_xlabel("Time (ms)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    svg_path = os.path.splitext(out_path)[0] + ".svg"
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Saved {svg_path}")


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
