#!/usr/bin/env python3
"""
Minimal demo of one two-compartmental hidden neuron driven by mock input.

Setup:
  - 3 input channels, each firing a single spike at t = 10, 80, 200 ms.
  - Dendritic weights: [1.0, 1.0, 0.2]
  - Somatic   weights: [0.2, 0.3, 0.5]
  - Plateau window T_p fixed to 200 ms (reproducible).

Plots two subplots:
  1. dendritic voltage mu and plateau state h
  2. somatic voltage v and dynamic threshold v_th - gamma * h

Input spike times are shown as dotted vertical lines; hidden output
spike times are shown as dashed vertical lines.
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

from config import NeuronConfig  # noqa: E402
from two_comp_neuron import TwoCompNeuron  # noqa: E402


def build_mock_input(T: int, spike_times_ms, n_inputs: int) -> jnp.ndarray:
    """One input channel per spike time. Channel i fires a single spike at
    spike_times_ms[i] (dt = 1 ms). Returns (T, n_inputs) float array."""
    assert len(spike_times_ms) == n_inputs
    x = np.zeros((T, n_inputs), dtype=np.float64)
    for ch, t in enumerate(spike_times_ms):
        x[t, ch] = 1.0
    return jnp.array(x)


def run_demo():
    config = NeuronConfig()

    T = 300
    spike_times = [10, 80, 200]
    n_inputs = 3
    n_neurons = 1

    w_dend = jnp.array([[0.3, 1.0, 0.2]])
    w_soma = jnp.array([[0.8, 0.3, 0.5]])

    neuron = TwoCompNeuron(jax.random.PRNGKey(0), n_neurons, n_inputs, config)
    neuron.w_dend = w_dend
    neuron.w_soma = w_soma
    # Deterministic plateau window (150 ms).
    T_p = jnp.array([150], dtype=jnp.int32)

    x_input = build_mock_input(T, spike_times, n_inputs)
    dend_inputs = x_input @ w_dend.T  # (T, 1)
    soma_inputs = x_input @ w_soma.T  # (T, 1)

    # Tiny gaussian background noise on both compartments so the traces
    # look like real sub-threshold voltages instead of flat zero lines.
    rng = np.random.default_rng(0)
    noise_std = 0.01
    dend_inputs = dend_inputs + jnp.array(rng.normal(0, noise_std, size=(T, 1)))
    soma_inputs = soma_inputs + jnp.array(rng.normal(0, noise_std, size=(T, 1)))

    alpha_s = neuron.alpha_s
    alpha_d = neuron.alpha_d

    carry = neuron.init_carry()
    mu_hist = np.zeros(T)
    v_hist = np.zeros(T)
    v_pre_hist = np.zeros(T)
    h_hist = np.zeros(T, dtype=np.int32)
    o_hist = np.zeros(T, dtype=np.int32)

    for t in range(T):
        carry, o, v_pre_reset, h, _h_prev, _mu_at_tp = TwoCompNeuron.forward_step(
            carry,
            dend_inputs[t],
            soma_inputs[t],
            jnp.array(t, dtype=jnp.int32),
            alpha_s,
            alpha_d,
            T_p,
            config,
        )
        mu, v, _h_c, _tp, _matp, _E, _dmu, _dmu_atp = carry
        mu_hist[t] = float(mu[0])
        v_hist[t] = float(v[0])
        v_pre_hist[t] = float(v_pre_reset[0])
        h_hist[t] = int(h[0])
        o_hist[t] = int(o[0])

    return {
        "T": T,
        "config": config,
        "spike_times": spike_times,
        "mu": mu_hist,
        "v": v_hist,
        "v_pre": v_pre_hist,
        "h": h_hist,
        "o": o_hist,
        "T_p": int(T_p[0]),
    }


def plot_demo(res: dict, out_path: str):
    T = res["T"]
    cfg = res["config"]
    t_axis = np.arange(T)
    mu = res["mu"]
    v = res["v"]
    v_pre = res["v_pre"]
    h = res["h"]
    o = res["o"]
    in_spikes = res["spike_times"]
    out_spikes = np.where(o > 0)[0].tolist()
    dyn_threshold = cfg.v_th - cfg.gamma * h

    fig, (ax_d, ax_s) = plt.subplots(2, 1, figsize=(10, 6))

    # --- Subplot 1: dendritic voltage mu + plateau state h ---
    ax_d.plot(t_axis, mu, color="C0", label=r"$\mu$ (dendritic voltage)")
    ax_d.plot(t_axis, h, color="C1", drawstyle="steps-post",
              label=r"plateau $h$")
    ax_d.axhline(cfg.mu_th, color="C0", linestyle=":", alpha=0.5,
                 label=rf"$\mu_\mathrm{{th}}={cfg.mu_th}$")
    ax_d.set_ylabel(r"$\mu$")

    for i, ts in enumerate(out_spikes):
        ax_d.axvline(ts, color="k", linestyle="--", alpha=0.7,
                     label="hidden spike" if i == 0 else None)

    ax_d.set_title(r"Dendrite: voltage $\mu$ and plateau state $h$"
                   rf"  ($T_p={res['T_p']}$ ms)")
    ax_d.set_xlabel("Time (ms)")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc="upper right", fontsize=8)

    # --- Subplot 2: somatic voltage v + dynamic threshold v_th - gamma*h ---
    ax_s.plot(t_axis, v_pre, color="C3", alpha=0.55,
              label=r"$v$ (pre-reset)")
    ax_s.plot(t_axis, v, color="C3", label=r"$v$ (post-reset)")
    ax_s.plot(t_axis, dyn_threshold, color="k", linestyle="--",
              label=r"dynamic threshold $v_\mathrm{th}-\gamma h$")

    for i, ts in enumerate(out_spikes):
        ax_s.axvline(ts, color="k", linestyle="--", alpha=0.7,
                     label="hidden spike" if i == 0 else None)

    ax_s.set_ylabel(r"$v$")
    ax_s.set_xlabel("Time (ms)")
    ax_s.set_title(r"Soma: voltage $v$ and dynamic threshold "
                   rf"($v_\mathrm{{th}}={cfg.v_th}$, $\gamma={cfg.gamma}$)")
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    res = run_demo()
    print(f"Hidden output spikes at t = {np.where(res['o'] > 0)[0].tolist()} ms")
    print(f"Plateau active for {int(res['h'].sum())} ms total")
    out_path = os.path.join(_SCRIPT_DIR, "demo.png")
    plot_demo(res, out_path)


if __name__ == "__main__":
    main()
