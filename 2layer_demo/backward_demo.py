#!/usr/bin/env python3
"""
Backward-pass demo for 2layer_no_history with one extra and one hidden neuron.

Produces local-gradient plots (error term omitted) for both neurons:
  - extra:  soma + dendrite
  - hidden: soma + dendrite

Fixed cross-layer weights (extra -> hidden):
  w_dend_h = 1.0
  w_soma_h = 0.6
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
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import NeuronConfig, surrogate_sigma  # noqa: E402
from two_comp_neuron import TwoCompNeuron  # noqa: E402
from demo import build_mock_input  # noqa: E402


def run_backward():
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

    E_soma_e = jnp.zeros(n_inputs)
    dmu_e = jnp.zeros((1, n_inputs))
    dmu_e_atp = jnp.zeros((1, n_inputs))

    E_soma_h = jnp.zeros(1)
    dmu_h = jnp.zeros((1, 1))
    dmu_h_atp = jnp.zeros((1, 1))

    # Mirrors of extra-layer quantities at hidden t'_h(t), matching the
    # mechanism in 2layer_no_history/network.py.
    sp_e_at_tph = 0.0
    hp_e_at_tph = 0.0
    E_e_at_tph = np.zeros(n_inputs, dtype=np.float64)
    dmu_e_at_tph = np.zeros(n_inputs, dtype=np.float64)

    # extra traces
    vpre_e = np.zeros(T)
    h_e = np.zeros(T)
    mu_tp_e = np.zeros(T)
    E_soma_e_hist = np.zeros((T, n_inputs))
    dmu_e_atp_hist = np.zeros((T, n_inputs))

    # hidden traces
    vpre_h = np.zeros(T)
    h_h = np.zeros(T)
    mu_tp_h = np.zeros(T)
    E_soma_h_hist = np.zeros((T, 1))
    dmu_h_atp_hist = np.zeros((T, 1))
    h_h_prev_hist = np.zeros(T, dtype=np.int32)

    # Path-wise terms for extra layer (error/readout omitted):
    # p1 contributes to extra dendritic gradient
    # p2 contributes to extra soma and extra dendritic gradients
    extra_p1_dend_hist = np.zeros((T, n_inputs))
    extra_p2_soma_hist = np.zeros((T, n_inputs))
    extra_p2_dend_hist = np.zeros((T, n_inputs))

    # Per-timestep scalar factors for decomposition plots.
    sp_e_hist = np.zeros(T)
    hp_e_hist = np.zeros(T)
    sp_h_hist = np.zeros(T)
    hp_h_hist = np.zeros(T)
    sp_e_at_tph_hist = np.zeros(T)
    hp_e_at_tph_hist = np.zeros(T)
    E_e_at_tph_hist = np.zeros((T, n_inputs))
    dmu_e_at_tph_hist = np.zeros((T, n_inputs))

    for t in range(T):
        e_carry, e_o, e_vpre, e_h_now, e_h_prev, e_mu_tp = TwoCompNeuron.forward_step(
            e_carry, dend_in_e[t], soma_in_e[t], jnp.array(t, dtype=jnp.int32),
            extra.alpha_s, extra.alpha_d, T_p_e, cfg,
        )
        x_t = x_input[t].astype(jnp.float64)
        E_soma_e = TwoCompNeuron.update_somatic_eligibility(E_soma_e, x_t, extra.alpha_s)
        dmu_e, dmu_e_atp = TwoCompNeuron.update_dendritic_eligibility(
            dmu_e, dmu_e_atp, x_t, e_h_prev, extra.alpha_d,
        )

        e_o_f = e_o.astype(jnp.float64)
        dend_in_h_t = e_o_f @ w_dend_h.T
        soma_in_h_t = e_o_f @ w_soma_h.T
        h_carry, h_o, h_vpre_now, h_h_now, h_h_prev, h_mu_tp = TwoCompNeuron.forward_step(
            h_carry, dend_in_h_t, soma_in_h_t, jnp.array(t, dtype=jnp.int32),
            hidden.alpha_s, hidden.alpha_d, T_p_h, cfg,
        )
        E_soma_h = TwoCompNeuron.update_somatic_eligibility(E_soma_h, e_o_f, hidden.alpha_s)
        dmu_h, dmu_h_atp = TwoCompNeuron.update_dendritic_eligibility(
            dmu_h, dmu_h_atp, e_o_f, h_h_prev, hidden.alpha_d,
        )

        # Current surrogates (scalar in this 1-extra/1-hidden demo).
        sp_e_t = float(surrogate_sigma(e_vpre[0] + cfg.gamma * e_h_now[0] - cfg.v_th, cfg.beta_s))
        hp_e_t = float(surrogate_sigma(e_mu_tp[0] - cfg.mu_th, cfg.beta_d))
        sp_h_t = float(surrogate_sigma(h_vpre_now[0] + cfg.gamma * h_h_now[0] - cfg.v_th, cfg.beta_s))
        hp_h_t = float(surrogate_sigma(h_mu_tp[0] - cfg.mu_th, cfg.beta_d))

        # Refresh mirrors exactly when hidden plateau is not ongoing into this step.
        if int(h_h_prev[0]) == 0:
            sp_e_at_tph = sp_e_t
            hp_e_at_tph = hp_e_t
            E_e_at_tph = np.asarray(E_soma_e)
            dmu_e_at_tph = np.asarray(dmu_e_atp[0])

        # Path 1 for extra dendrite (from hidden soma branch), dropping sp_r and w_readout.
        # gamma * sp_h * w_soma_h * sp_e * hp_e * dmu_e_atp
        extra_p1_dend_hist[t] = (
            cfg.gamma
            * sp_h_t
            * float(w_soma_h[0, 0])
            * sp_e_t
            * hp_e_t
            * np.asarray(dmu_e_atp[0])
        )

        # Path 2 for extra soma (uses mirrored extra quantities at hidden t'_h).
        # gamma * sp_h * hp_h * w_dend_h * sp_e(t'_h) * E_e(t'_h)
        extra_p2_soma_hist[t] = (
            cfg.gamma
            * sp_h_t
            * hp_h_t
            * float(w_dend_h[0, 0])
            * sp_e_at_tph
            * E_e_at_tph
        )

        # Path 2 for extra dendrite (adds one more gamma*hp_e(t'_h) and dmu_e(t'_h)).
        extra_p2_dend_hist[t] = (
            cfg.gamma
            * sp_h_t
            * hp_h_t
            * float(w_dend_h[0, 0])
            * sp_e_at_tph
            * hp_e_at_tph
            * cfg.gamma
            * dmu_e_at_tph
        )

        sp_e_hist[t] = sp_e_t
        hp_e_hist[t] = hp_e_t
        sp_h_hist[t] = sp_h_t
        hp_h_hist[t] = hp_h_t
        sp_e_at_tph_hist[t] = sp_e_at_tph
        hp_e_at_tph_hist[t] = hp_e_at_tph
        E_e_at_tph_hist[t] = E_e_at_tph
        dmu_e_at_tph_hist[t] = dmu_e_at_tph

        vpre_e[t] = float(e_vpre[0])
        h_e[t] = float(e_h_now[0])
        mu_tp_e[t] = float(e_mu_tp[0])
        E_soma_e_hist[t] = np.asarray(E_soma_e)
        dmu_e_atp_hist[t] = np.asarray(dmu_e_atp[0])

        vpre_h[t] = float(h_vpre_now[0])
        h_h[t] = float(h_h_now[0])
        mu_tp_h[t] = float(h_mu_tp[0])
        E_soma_h_hist[t] = np.asarray(E_soma_h)
        dmu_h_atp_hist[t] = np.asarray(dmu_h_atp[0])
        h_h_prev_hist[t] = int(h_h_prev[0])

    sp_e = np.asarray(surrogate_sigma(jnp.array(vpre_e + cfg.gamma * h_e - cfg.v_th), cfg.beta_s))
    hp_e = np.asarray(surrogate_sigma(jnp.array(mu_tp_e - cfg.mu_th), cfg.beta_d))
    sp_h = np.asarray(surrogate_sigma(jnp.array(vpre_h + cfg.gamma * h_h - cfg.v_th), cfg.beta_s))
    hp_h = np.asarray(surrogate_sigma(jnp.array(mu_tp_h - cfg.mu_th), cfg.beta_d))

    return {
        "T": T,
        "E_soma_e": E_soma_e_hist,
        "dmu_e_atp": dmu_e_atp_hist,
        "sp_e": sp_e,
        "hp_e": hp_e,
        "E_soma_h": E_soma_h_hist,
        "dmu_h_atp": dmu_h_atp_hist,
        "sp_h": sp_h,
        "hp_h": hp_h,
        "extra_p1_dend": extra_p1_dend_hist,
        "extra_p2_soma": extra_p2_soma_hist,
        "extra_p2_dend": extra_p2_dend_hist,
        "h_h_prev": h_h_prev_hist,
        "sp_e_hist": sp_e_hist,
        "hp_e_hist": hp_e_hist,
        "sp_h_hist": sp_h_hist,
        "hp_h_hist": hp_h_hist,
        "sp_e_at_tph_hist": sp_e_at_tph_hist,
        "hp_e_at_tph_hist": hp_e_at_tph_hist,
        "E_e_at_tph_hist": E_e_at_tph_hist,
        "dmu_e_at_tph_hist": dmu_e_at_tph_hist,
        "w_dend_h": float(w_dend_h[0, 0]),
        "w_soma_h": float(w_soma_h[0, 0]),
        "gamma": float(cfg.gamma),
    }


def _plot_compartment(elig, surrogate, total, labels, title_prefix, elig_label, surr_label, total_label, out_path):
    T = elig.shape[0]
    t = np.arange(T)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    for k in range(elig.shape[1]):
        axes[0].plot(t, elig[:, k], label=labels[k])
    axes[0].set_title(f"{title_prefix}: eligibility")
    axes[0].set_ylabel(elig_label)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, surrogate, color="C3")
    axes[1].set_title(f"{title_prefix}: surrogate")
    axes[1].set_ylabel(surr_label)
    axes[1].grid(True, alpha=0.3)

    for k in range(total.shape[1]):
        axes[2].plot(t, total[:, k], label=labels[k])
    axes[2].set_title(f"{title_prefix}: surrogate * eligibility (error omitted)")
    axes[2].set_ylabel(total_label)
    axes[2].set_xlabel("Time (ms)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_extra_dendrite_channel_parts(res: dict, ch: int, out_path: str):
    """Per-channel decomposition of extra dendritic full gradient."""
    T = res["T"]
    t = np.arange(T)

    gamma = res["gamma"]
    w_soma_h = res["w_soma_h"]
    w_dend_h = res["w_dend_h"]

    sp_h = res["sp_h_hist"]
    hp_h = res["hp_h_hist"]
    sp_e = res["sp_e_hist"]
    hp_e = res["hp_e_hist"]
    dmu_e = res["dmu_e_atp"][:, ch]

    sp_e_tph = res["sp_e_at_tph_hist"]
    hp_e_tph = res["hp_e_at_tph_hist"]
    dmu_e_tph = res["dmu_e_at_tph_hist"][:, ch]

    # Path components
    p1 = res["extra_p1_dend"][:, ch]
    p2 = res["extra_p2_dend"][:, ch]
    total = p1 + p2

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 10))

    axes[0].plot(t, np.full(T, gamma), label="gamma")
    axes[0].plot(t, np.full(T, w_soma_h), label="w_soma_h")
    axes[0].plot(t, np.full(T, w_dend_h), label="w_dend_h")
    axes[0].set_title(f"Extra dendrite ch{ch}: constant factors")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, sp_h, label="sp_h(t)")
    axes[1].plot(t, hp_h, label="hp_h(t)")
    axes[1].plot(t, sp_e, label="sp_e(t)")
    axes[1].plot(t, hp_e, label="hp_e(t)")
    axes[1].plot(t, dmu_e, label=f"dmu_e_atp(t,ch{ch})")
    axes[1].set_title("Path-1 time-varying factors")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, sp_e_tph, label=r"sp_e(t'_h)")
    axes[2].plot(t, hp_e_tph, label=r"hp_e(t'_h)")
    axes[2].plot(t, dmu_e_tph, label=rf"dmu_e(t'_h,ch{ch})")
    axes[2].set_title("Path-2 mirrored factors at hidden plateau-init time")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, p1, label="p1_dend(ch)")
    axes[3].plot(t, p2, label="p2_dend(ch)")
    axes[3].plot(t, total, label="p1+p2 (full extra dend)")
    axes[3].set_title("Composed gradient terms")
    axes[3].set_xlabel("Time (ms)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    res = run_backward()

    extra_soma_total = res["sp_e"][:, None] * res["E_soma_e"]
    extra_dend_total = res["sp_e"][:, None] * res["hp_e"][:, None] * res["dmu_e_atp"]
    hidden_soma_total = res["sp_h"][:, None] * res["E_soma_h"]
    hidden_dend_total = res["sp_h"][:, None] * res["hp_h"][:, None] * res["dmu_h_atp"]

    # Parameter-specific full local gradients (error/readout omitted).
    # Hidden layer:
    #  - w_soma_h gets only soma branch
    #  - w_dend_h gets only dendritic branch
    hidden_full_soma = hidden_soma_total
    hidden_full_dend = hidden_dend_total
    # Extra layer:
    #  - w_soma_e gets path-2 soma contribution
    #  - w_dend_e gets path-1 dend + path-2 dend
    extra_full_soma = res["extra_p2_soma"]
    extra_full_dend = res["extra_p1_dend"] + res["extra_p2_dend"]

    _plot_compartment(
        res["E_soma_e"], res["sp_e"], extra_full_soma,
        labels=["ch0", "ch1", "ch2", "ch3"],
        title_prefix="Extra soma",
        elig_label=r"$\partial v / \partial w_{soma,e}$",
        surr_label=r"$\partial o / \partial v$",
        total_label=r"full gradient for $w_{soma,e}$ (p2, no error/readout)",
        out_path=os.path.join(_SCRIPT_DIR, "backward_extra_soma.png"),
    )
    _plot_compartment(
        res["dmu_e_atp"], res["hp_e"], extra_full_dend,
        labels=["ch0", "ch1", "ch2", "ch3"],
        title_prefix="Extra dendrite",
        elig_label=r"$\partial \mu_{t'} / \partial w_{dend,e}$",
        surr_label=r"$\partial h / \partial \mu_{t'}$",
        total_label=r"full gradient for $w_{dend,e}$ (p1+p2, no error/readout)",
        out_path=os.path.join(_SCRIPT_DIR, "backward_extra_dendrite.png"),
    )
    _plot_compartment(
        res["E_soma_h"], res["sp_h"], hidden_full_soma,
        labels=["extra->hidden"],
        title_prefix="Hidden soma",
        elig_label=r"$\partial v / \partial w_{soma,h}$",
        surr_label=r"$\partial o / \partial v$",
        total_label=r"full gradient for $w_{soma,h}$ (no error/readout)",
        out_path=os.path.join(_SCRIPT_DIR, "backward_hidden_soma.png"),
    )
    _plot_compartment(
        res["dmu_h_atp"], res["hp_h"], hidden_full_dend,
        labels=["extra->hidden"],
        title_prefix="Hidden dendrite",
        elig_label=r"$\partial \mu_{t'} / \partial w_{dend,h}$",
        surr_label=r"$\partial h / \partial \mu_{t'}$",
        total_label=r"full gradient for $w_{dend,h}$ (no error/readout)",
        out_path=os.path.join(_SCRIPT_DIR, "backward_hidden_dendrite.png"),
    )

    # Per-channel decomposition plots for extra dendritic full gradient.
    for ch in range(4):
        _plot_extra_dendrite_channel_parts(
            res,
            ch,
            os.path.join(_SCRIPT_DIR, f"backward_extra_dendrite_ch{ch}_parts.png"),
        )


if __name__ == "__main__":
    main()
