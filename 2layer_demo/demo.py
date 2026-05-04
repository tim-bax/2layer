#!/usr/bin/env python3
"""
Forward-pass demo for 2layer_no_history with one extra and one hidden neuron.

Architecture:
  input (4 channels) -> extra neuron -> hidden neuron

Fixed cross-layer weights (extra -> hidden):
  w_dend_h = 1.0
  w_soma_h = 0.6

Outputs:
  2layer_demo/forward_extra.png
  2layer_demo/forward_hidden.png
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
_TWO_LAYER = os.path.join(_ROOT, "2layer_no_history")
if _TWO_LAYER not in sys.path:
    sys.path.insert(0, _TWO_LAYER)

from config import NeuronConfig  # noqa: E402
from two_comp_neuron import TwoCompNeuron  # noqa: E402


def build_mock_input(T: int, spike_times_ms, n_inputs: int) -> jnp.ndarray:
    assert len(spike_times_ms) == n_inputs
    x = np.zeros((T, n_inputs), dtype=np.float64)
    for ch, t in enumerate(spike_times_ms):
        x[t, ch] = 1.0
    return jnp.array(x)


def run_forward_demo():
    cfg = NeuronConfig()
    T = 300
    spike_times = [10, 80, 200, 250]
    n_inputs = 4

    # input -> extra
    w_dend_e = jnp.array([[0.7, 1.0, 0.2, 0.8]], dtype=jnp.float64)
    w_soma_e = jnp.array([[1.6, 0.6, 0.6, 1.0]], dtype=jnp.float64)

    # extra -> hidden (requested)
    w_dend_h = jnp.array([[1.0]], dtype=jnp.float64)
    w_soma_h = jnp.array([[0.6]], dtype=jnp.float64)

    extra = TwoCompNeuron(jax.random.PRNGKey(0), 1, n_inputs, cfg)
    hidden = TwoCompNeuron(jax.random.PRNGKey(1), 1, 1, cfg)
    extra.w_dend = w_dend_e
    extra.w_soma = w_soma_e
    hidden.w_dend = w_dend_h
    hidden.w_soma = w_soma_h

    T_p_e = jnp.array([150], dtype=jnp.int32)
    T_p_h = jnp.array([150], dtype=jnp.int32)

    x_input = build_mock_input(T, spike_times, n_inputs)
    dend_in_e = x_input @ w_dend_e.T
    soma_in_e = x_input @ w_soma_e.T

    rng = np.random.default_rng(0)
    noise_std = 0.01
    dend_in_e = dend_in_e + jnp.array(rng.normal(0, noise_std, size=(T, 1)))
    soma_in_e = soma_in_e + jnp.array(rng.normal(0, noise_std, size=(T, 1)))

    e_carry = extra.init_carry()
    h_carry = hidden.init_carry()

    mu_e = np.zeros(T)
    v_e = np.zeros(T)
    vpre_e = np.zeros(T)
    h_e = np.zeros(T, dtype=np.int32)
    o_e = np.zeros(T, dtype=np.int32)

    mu_h = np.zeros(T)
    v_h = np.zeros(T)
    vpre_h = np.zeros(T)
    h_h = np.zeros(T, dtype=np.int32)
    o_h = np.zeros(T, dtype=np.int32)

    for t in range(T):
        e_carry, e_o, e_vpre, e_h_now, _eh_prev, _emu_tp = TwoCompNeuron.forward_step(
            e_carry,
            dend_in_e[t],
            soma_in_e[t],
            jnp.array(t, dtype=jnp.int32),
            extra.alpha_s,
            extra.alpha_d,
            T_p_e,
            cfg,
        )
        e_mu, e_v, _eh, _etp, _ematp, _eE, _edmu, _edmu_atp = e_carry
        e_o_f = e_o.astype(jnp.float64)

        dend_in_h_t = e_o_f @ w_dend_h.T
        soma_in_h_t = e_o_f @ w_soma_h.T
        h_carry, h_o_now, h_vpre_now, h_h_now, _hh_prev, _hmu_tp = TwoCompNeuron.forward_step(
            h_carry,
            dend_in_h_t,
            soma_in_h_t,
            jnp.array(t, dtype=jnp.int32),
            hidden.alpha_s,
            hidden.alpha_d,
            T_p_h,
            cfg,
        )
        h_mu_now, h_v_now, _hh, _htp, _hmatp, _hE, _hdmu, _hdmu_atp = h_carry

        mu_e[t] = float(e_mu[0])
        v_e[t] = float(e_v[0])
        vpre_e[t] = float(e_vpre[0])
        h_e[t] = int(e_h_now[0])
        o_e[t] = int(e_o[0])

        mu_h[t] = float(h_mu_now[0])
        v_h[t] = float(h_v_now[0])
        vpre_h[t] = float(h_vpre_now[0])
        h_h[t] = int(h_h_now[0])
        o_h[t] = int(h_o_now[0])

    return {
        "T": T,
        "cfg": cfg,
        "spike_times": spike_times,
        "T_p_e": int(T_p_e[0]),
        "T_p_h": int(T_p_h[0]),
        "mu_e": mu_e, "v_e": v_e, "vpre_e": vpre_e, "h_e": h_e, "o_e": o_e,
        "mu_h": mu_h, "v_h": v_h, "vpre_h": vpre_h, "h_h": h_h, "o_h": o_h,
    }


def _plot_single_neuron(mu, v, vpre, h, o, cfg, title_prefix, T_p, out_path):
    t = np.arange(len(mu))
    out_spikes = np.where(o > 0)[0].tolist()
    dyn_th = cfg.v_th - cfg.gamma * h

    fig, (ax_d, ax_s) = plt.subplots(2, 1, figsize=(10, 6))
    ax_d.plot(t, mu, color="C0", label=r"$\mu$")
    ax_d.plot(t, h, color="C1", drawstyle="steps-post", label=r"$h$")
    ax_d.axhline(cfg.mu_th, color="C0", linestyle=":", alpha=0.5, label=r"$\mu_{th}$")
    for i, ts in enumerate(out_spikes):
        ax_d.axvline(ts, color="k", linestyle="--", alpha=0.6, label="spike" if i == 0 else None)
    ax_d.set_title(f"{title_prefix} dendrite ($T_p={T_p}$ ms)")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc="upper right", fontsize=8)

    ax_s.plot(t, vpre, color="C3", alpha=0.5, label=r"$v$ pre-reset")
    ax_s.plot(t, v, color="C3", label=r"$v$ post-reset")
    ax_s.plot(t, dyn_th, color="k", linestyle="--", label=r"$v_{th}-\gamma h$")
    for i, ts in enumerate(out_spikes):
        ax_s.axvline(ts, color="k", linestyle="--", alpha=0.6, label="spike" if i == 0 else None)
    ax_s.set_title(f"{title_prefix} soma")
    ax_s.set_xlabel("Time (ms)")
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    res = run_forward_demo()
    print(f"Extra spikes: {np.where(res['o_e'] > 0)[0].tolist()}")
    print(f"Hidden spikes: {np.where(res['o_h'] > 0)[0].tolist()}")
    print("extra->hidden weights: dendrite=1.0, soma=0.6")

    _plot_single_neuron(
        res["mu_e"], res["v_e"], res["vpre_e"], res["h_e"], res["o_e"],
        res["cfg"], "Extra neuron", res["T_p_e"],
        os.path.join(_SCRIPT_DIR, "forward_extra.png"),
    )
    _plot_single_neuron(
        res["mu_h"], res["v_h"], res["vpre_h"], res["h_h"], res["o_h"],
        res["cfg"], "Hidden neuron", res["T_p_h"],
        os.path.join(_SCRIPT_DIR, "forward_hidden.png"),
    )


if __name__ == "__main__":
    main()
