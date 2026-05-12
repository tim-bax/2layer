#!/usr/bin/env python3
"""
Forward + backward (max-over-time readout) visualization for the `no_history` model.

Mirrors `demo/demo.py` style (single hidden, mock spikes, deterministic T_p) but wires
the full two-layer forward from `no_history/network.py` and decomposes the same gradient
factors used in `_max_loss_and_grads`.

Outputs (default directory: same folder as this script):
  - forward_pass.png          — hidden μ, h, v_pre, readout v_j(t), markers at t*_j
  - backward_readout.png      — ∂L/∂logit, ∂L/∂v_max, M, E_readout, ∂v_max/∂w, grad w
  - backward_soma.png         — φ, ψ_soma, E_soma, φ·ψ_soma·E (per input k)
  - backward_dendrite.png     — sliding argmax g, μ at g, ψ_dend, ∂μ/∂w at g, γ·φ·ψ·∂μ/∂w

Run from repo root or from `demo/`:
  python demo/demo_no_history_forward_backward.py
"""
from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp
from jax import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
_NO_HISTORY = os.path.join(_ROOT, "no_history")
if _NO_HISTORY not in sys.path:
    sys.path.insert(0, _NO_HISTORY)

from config import NeuronConfig, surrogate_sigma  # noqa: E402
from network import (  # noqa: E402
    Network,
    _forward_and_accum,
    _max_loss_and_grads,
    _sliding_argmax_mu,
)
from two_comp_neuron import TwoCompNeuron  # noqa: E402


def build_mock_input(T: int, spike_times_ms, n_inputs: int) -> jnp.ndarray:
    assert len(spike_times_ms) == n_inputs
    x = np.zeros((T, n_inputs), dtype=np.float64)
    for ch, t in enumerate(spike_times_ms):
        x[t, ch] = 1.0
    return jnp.array(x)


def _time_axis_ms(T: int, dt_ms: float) -> np.ndarray:
    return np.arange(T, dtype=np.float64) * float(dt_ms)


def run_forward_backward(
    *,
    T: int,
    n_inputs: int,
    n_hidden: int,
    n_outputs: int,
    spike_times_ms,
    label: int,
    seed: int,
    tau_plat_steps: int,
    noise_std: float,
    loss_temperature: float,
    loss_logit_bias: float,
    label_smoothing: float,
) -> dict:
    cfg = NeuronConfig(
        dt=1.0,
        loss_temperature=loss_temperature,
        loss_count_bias=loss_logit_bias,
        loss_label_smoothing=label_smoothing,
    )
    key = random.PRNGKey(seed)
    net = Network(key, n_inputs, n_hidden, n_outputs, cfg, optimizer="sgd")

    w_d = jnp.ones((n_hidden, n_inputs), dtype=jnp.float64) * 0.65
    row = jnp.array([1.8, 1.35, 1.25][:n_inputs], dtype=jnp.float64)
    w_s = jnp.tile(row[None, :], (n_hidden, 1))
    w_r = random.normal(random.PRNGKey(seed + 1), (n_outputs, n_hidden)) * 0.65
    net.hidden.w_dend = w_d
    net.hidden.w_soma = w_s
    net.readout.w = w_r.astype(jnp.float64)

    if tau_plat_steps > 0:
        net.hidden.T_p = jnp.full((n_hidden,), tau_plat_steps, dtype=jnp.int32)

    x_raw_np = np.array(build_mock_input(T, spike_times_ms, n_inputs))
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        x_raw_np = x_raw_np + rng.normal(0, noise_std, size=x_raw_np.shape)
    x_input = jnp.array(x_raw_np)

    fwd_key = random.PRNGKey(seed + 7)
    v_readout, mu_tr, dmu_tr, sp_hidden, E_soma_tr, E_readout_tr = _forward_and_accum(
        x_input,
        net.hidden.w_dend,
        net.hidden.w_soma,
        net.readout.w,
        net.hidden.alpha_s,
        net.hidden.alpha_d,
        net.readout.alpha_m,
        net.hidden.T_p,
        cfg,
        net._h_carry(),
        net._r_carry(),
        fwd_key,
        0.0,
    )

    target_vec = net._smooth_targets(jnp.array(label, dtype=jnp.int32))
    loss, pred, grad_r, grad_s, grad_d = _max_loss_and_grads(
        v_readout,
        mu_tr,
        dmu_tr,
        sp_hidden,
        E_soma_tr,
        E_readout_tr,
        net.readout.w,
        net.hidden.T_p,
        net.readout.alpha_m,
        target_vec,
        cfg.loss_temperature,
        cfg.loss_count_bias,
        cfg,
    )

    Tsteps = int(v_readout.shape[0])
    J = int(v_readout.shape[1])
    N = int(mu_tr.shape[1])

    t_star = np.asarray(jnp.argmax(v_readout, axis=0))
    v_max = np.asarray(v_readout[t_star, np.arange(J)])
    logits = v_max / float(loss_temperature) + float(loss_logit_bias)
    logits = logits - logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()
    global_error = np.asarray(target_vec - probs)

    t_grid = jnp.arange(Tsteps, dtype=jnp.int32)[:, None]
    t_star_row = jnp.asarray(t_star)[None, :]
    alpha_m = float(net.readout.alpha_m)
    M = np.asarray(
        (t_grid <= t_star_row).astype(jnp.float64)
        * (alpha_m ** jnp.maximum(0, t_star_row - t_grid))
    )
    phi = np.asarray(
        jnp.einsum("j,tj,ji->ti", jnp.asarray(global_error), jnp.asarray(M), net.readout.w)
    )

    G = np.asarray(_sliding_argmax_mu(mu_tr, net.hidden.T_p))
    n_broadcast = np.broadcast_to(np.arange(N, dtype=np.int32), G.shape)
    mu_at_g = np.asarray(mu_tr[G, n_broadcast])
    hp_heur = np.asarray(surrogate_sigma(jnp.asarray(mu_at_g) - cfg.mu_th, cfg.beta_d))
    dmu_pick = np.asarray(dmu_tr[G, n_broadcast, :])

    T_scale = max(1.0, float(Tsteps))
    dL_dlogit = global_error
    dL_dv_max = global_error / float(loss_temperature)

    E_at_star = np.asarray(E_readout_tr[t_star, :])
    dvmax_dw = E_at_star
    grad_readout_impl = np.asarray(grad_r)

    hidden_carry_mu = np.asarray(mu_tr[:, 0] if N == 1 else mu_tr.mean(axis=1))
    hidden_h = []
    hidden_v_pre = []
    hidden_o = []
    dend_in = x_input @ net.hidden.w_dend.T
    soma_in = x_input @ net.hidden.w_soma.T
    carry = net._h_carry()
    for t in range(Tsteps):
        carry, o, v_pre, h, h_prev, _matp = TwoCompNeuron.forward_step(
            carry,
            dend_in[t],
            soma_in[t],
            jnp.int32(t),
            net.hidden.alpha_s,
            net.hidden.alpha_d,
            net.hidden.T_p,
            cfg,
        )
        mu, v, h_c, tp_c, matp_c, Es, dmu, dmu_atp = carry
        x_t = x_input[t]
        E_new = TwoCompNeuron.update_somatic_eligibility(
            Es, x_t.astype(jnp.float64), net.hidden.alpha_s,
        )
        dmu_new, dmu_atp_new = TwoCompNeuron.update_dendritic_eligibility(
            dmu, dmu_atp, x_t.astype(jnp.float64), h_prev, net.hidden.alpha_d,
        )
        carry = (mu, v, h_c, tp_c, matp_c, E_new, dmu_new, dmu_atp_new)

        hidden_h.append(np.asarray(h))
        hidden_v_pre.append(np.asarray(v_pre))
        hidden_o.append(np.asarray(o))

    return {
        "config": cfg,
        "T": Tsteps,
        "t_ms": _time_axis_ms(Tsteps, float(cfg.dt)),
        "n_inputs": n_inputs,
        "n_hidden": N,
        "n_outputs": J,
        "x_input": np.asarray(x_input),
        "v_readout": np.asarray(v_readout),
        "mu_tr": np.asarray(mu_tr),
        "sp_hidden": np.asarray(sp_hidden),
        "E_soma_tr": np.asarray(E_soma_tr),
        "E_readout_tr": np.asarray(E_readout_tr),
        "dmu_tr": np.asarray(dmu_tr),
        "t_star": t_star,
        "v_max": v_max,
        "probs": probs,
        "global_error": global_error,
        "dL_dlogit": dL_dlogit,
        "dL_dv_max": dL_dv_max,
        "M": M,
        "phi": phi,
        "G": G,
        "mu_at_g": mu_at_g,
        "hp_heur": hp_heur,
        "dmu_pick": dmu_pick,
        "dvmax_dw": dvmax_dw,
        "grad_readout": grad_readout_impl,
        "grad_soma": np.asarray(grad_s),
        "grad_dend": np.asarray(grad_d),
        "loss": float(loss),
        "pred": int(pred),
        "label": int(label),
        "w_readout": np.asarray(net.readout.w),
        "alpha_m": alpha_m,
        "T_scale": T_scale,
        "gamma": float(cfg.gamma),
        "hidden_mu_mean": hidden_carry_mu,
        "hidden_h_stack": np.stack(hidden_h, axis=0) if hidden_h else np.zeros((0, N)),
        "hidden_v_pre_stack": np.stack(hidden_v_pre, axis=0) if hidden_v_pre else np.zeros((0, N)),
        "hidden_o_stack": np.stack(hidden_o, axis=0) if hidden_o else np.zeros((0, N), dtype=np.int32),
        "spike_times_ms": spike_times_ms,
    }


def plot_forward(res: dict, out_path: str) -> None:
    t = res["t_ms"]
    cfg = res["config"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    mu = res["mu_tr"]
    vread = res["v_readout"]
    T, J = vread.shape
    N = mu.shape[1]

    axes[0].set_title("Hidden: dendritic μ (replay); plateau $h_0$")
    for i in range(min(N, 3)):
        axes[0].plot(t, mu[:, i], label=rf"$\mu_{{{i}}}$", alpha=0.85)
    if res["hidden_h_stack"].size:
        axes[0].plot(
            t, res["hidden_h_stack"][:, 0], drawstyle="steps-post",
            color="C2", label=r"$h_0$",
        )
    axes[0].axhline(cfg.mu_th, color="k", linestyle=":", alpha=0.45, label=rf"$\mu_\mathrm{{th}}$")
    for ts in res["spike_times_ms"]:
        axes[0].axvline(ts, color="gray", linestyle=":", alpha=0.6)
    axes[0].set_ylabel(r"$\mu$ / $h$")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].set_title(r"Hidden soma $v_\mathrm{pre}$ (first neuron)")
    axes[1].plot(t, res["hidden_v_pre_stack"][:, 0], color="C3", label=r"$v_\mathrm{pre}$")
    thr = cfg.v_th - cfg.gamma * res["hidden_h_stack"][:, 0]
    axes[1].plot(t, thr, "k--", alpha=0.6, label=r"$v_\mathrm{th}-\gamma h$")
    out_spikes = np.where(res["hidden_o_stack"][:, 0] > 0)[0]
    for i, ix in enumerate(out_spikes):
        axes[1].axvline(t[ix], color="k", linestyle="--", alpha=0.55,
                        label="hidden spike" if i == 0 else None)
    axes[1].set_ylabel(r"$v$")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].set_title(r"Readout membrane $v_j(t)$ (non-spiking); $t^*_j=\arg\max_t v_j$")
    for j in range(J):
        axes[2].plot(t, vread[:, j], label=rf"$v_{{{j}}}$", alpha=0.85)
        tj = res["t_star"][j]
        axes[2].axvline(float(t[tj]), color=f"C{j % 10}", linestyle="--", alpha=0.45)
    axes[2].set_xlabel(f"Time (ms), dt = {cfg.dt}")
    axes[2].set_ylabel(r"$v_j$")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right", fontsize=8, ncol=min(4, J))

    fig.suptitle(
        rf"Forward  |  loss={res['loss']:.4f}  pred={res['pred']}  label={res['label']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_backward_readout(res: dict, out_path: str) -> None:
    J = res["n_outputs"]
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    ax = axs[0, 0]
    ax.bar(np.arange(J), res["dL_dlogit"], color="C0", alpha=0.85)
    ax.set_title(r"$\partial L/\partial \mathrm{logit}_j$  (= $y_j - p_j$, smoothed target)")
    ax.set_xticks(np.arange(J))
    ax.set_xlabel("readout index j")

    ax = axs[0, 1]
    ax.bar(np.arange(J), res["dL_dv_max"], color="C1", alpha=0.85)
    ax.set_title(r"$\partial L/\partial v^{\max}_j = (\partial L/\partial \mathrm{logit}_j)\,/\,T_\mathrm{loss}$")
    ax.set_xticks(np.arange(J))

    ax = axs[1, 0]
    im = ax.imshow(res["M"].T, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title(
        r"$M_{j,t}=\mathbb{1}[t\leq t^*_j]\,\alpha_m^{\,t^*_j-t}$ (readout temporal reachback)"
    )
    ax.set_xlabel("time t")
    ax.set_ylabel("readout j")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[1, 1]
    im = ax.imshow(res["E_readout_tr"].T, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title(r"$E^\mathrm{readout}_i(t)$ (filtered hidden drive; same for all $j$)")
    ax.set_xlabel("time t")
    ax.set_ylabel("hidden i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[2, 0]
    im = ax.imshow(res["dvmax_dw"], aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title(r"$\partial v^{\max}_j / \partial w_{ji} = E_i(t^*_j)$")
    ax.set_xlabel("hidden i")
    ax.set_ylabel("readout j")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[2, 1]
    im = ax.imshow(res["grad_readout"], aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title(
        r"$\partial L/\partial w_{ji}$ implemented = $(y_j-p_j)\,E_i(t^*_j)$ "
        "\n(chain w.r.t. logits, not $v^\mathrm{max}$ — see left column)"
    )
    ax.set_xlabel("hidden i")
    ax.set_ylabel("readout j")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Backward: readout chain", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_backward_soma(res: dict, out_path: str) -> None:
    t = res["t_ms"]
    phi = res["phi"]
    psi = res["sp_hidden"]
    E = res["E_soma_tr"]
    Ts, N = phi.shape
    K = E.shape[1]

    integrand = phi[:, :, None] * psi[:, :, None] * E[:, None, :]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    ax = axs[0, 0]
    im = ax.imshow(phi.T, aspect="auto", origin="lower", extent=[t[0], t[-1], -0.5, N - 0.5])
    ax.set_title(r"$\phi_i(t)=\sum_j e_j\,M_{j,t}\,w_{ji}$ (learning signal onto hidden $i$)")
    ax.set_ylabel("hidden i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[0, 1]
    im = ax.imshow(psi.T, aspect="auto", origin="lower", extent=[t[0], t[-1], -0.5, N - 0.5])
    ax.set_title(r"$\psi_i^\mathrm{soma}(t)=\sigma'(^{\cdot} v_i^\mathrm{pre}+\gamma h - v_\mathrm{th})$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[1, 0]
    for k in range(K):
        ax.plot(
            t, E[:, k],
            label=rf"$E_{{{k}}}^{{\mathrm{{soma}}}}$",
            alpha=0.8,
        )
    ax.set_title(r"$E_k^\mathrm{soma}(t)$ somatic eligibility (same for all hidden in this model)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("eligibility")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=min(4, K))

    ax = axs[1, 1]
    for k in range(K):
        c = np.sum(integrand[:, :, k], axis=1) / res["T_scale"]
        ax.plot(t, c, label=rf"$\sum_i \phi\psi E$ k={k}", alpha=0.85)
    ax.set_title(
        r"Per-time contribution to $\partial L/\partial w^\mathrm{soma}_{ik}$: "
        r"$\phi_i \psi_i^\mathrm{soma} E_k$ (sum over $i$, scaled by $1/T$ in code)"
    )
    ax.set_xlabel("time (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Backward: soma weights (e-prop factors)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_backward_dend(res: dict, out_path: str) -> None:
    t = res["t_ms"]
    phi = res["phi"]
    psi_s = res["sp_hidden"]
    g = res["G"]
    hp = res["hp_heur"]
    dmu = res["dmu_pick"]
    gam = res["gamma"]
    Ts, N, K = dmu.shape

    dend_chain = phi * psi_s * hp * gam
    integrand = dend_chain[:, :, None] * dmu

    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

    ax = axs[0, 0]
    im = ax.imshow(g.T, aspect="auto", origin="lower", extent=[t[0], t[-1], -0.5, N - 0.5])
    ax.set_title(r"$g_i(t)$ — argmax index of $\mu$ in $[t-T_{p,i},\,t]$ (heuristic, backward-only)")
    ax.set_ylabel("hidden i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[0, 1]
    im = ax.imshow(res["mu_at_g"].T, aspect="auto", origin="lower",
                   extent=[t[0], t[-1], -0.5, N - 0.5])
    ax.set_title(r"$\mu_i(g_i(t))$ at argmax time in window")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[1, 0]
    im = ax.imshow(hp.T, aspect="auto", origin="lower", extent=[t[0], t[-1], -0.5, N - 0.5])
    ax.set_title(r"$\psi_i^\mathrm{dend}(t)=\sigma'(\mu_i(g_i(t))-\mu_\mathrm{th})$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[1, 1]
    k_show = min(2, K - 1)
    im = ax.imshow(dmu[:, :, k_show].T, aspect="auto", origin="lower",
                   extent=[t[0], t[-1], -0.5, N - 0.5])
    ax.set_title(rf"$\partial\mu_i/\partial w^{{\mathrm{{dend}}}}_{{i,{k_show}}}$ at $g_i(t)$ (slice $k={k_show}$)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[2, 0]
    for k in range(K):
        c = np.sum(integrand[:, :, k], axis=1) / res["T_scale"]
        ax.plot(t, c, label=rf"sum$_i$ terms, k={k}", alpha=0.85)
    ax.set_title(
        r"$\gamma\,\phi\,\psi^\mathrm{soma}\,\psi^\mathrm{dend}\,\partial\mu/\partial w^\mathrm{dend}$ "
        r"(per $k$, sum over $i$; matches code up to $1/T$)"
    )
    ax.set_xlabel("time (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")

    ax = axs[2, 1]
    im = ax.imshow(
        (dend_chain / (np.abs(dend_chain).max() + 1e-12)).T,
        aspect="auto", origin="lower", extent=[t[0], t[-1], -0.5, N - 0.5],
        vmin=-1, vmax=1, cmap="RdBu_r",
    )
    ax.set_title(r"$\gamma\,\phi\,\psi^\mathrm{soma}\,\psi^\mathrm{dend}$ (normalized for display)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Backward: dendrite weights (heuristic μ-window)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser(description="Plot no_history forward + backward gradient factors.")
    p.add_argument("--out_dir", type=str, default=_SCRIPT_DIR, help="Directory for PNG outputs.")
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--n_hidden", type=int, default=1)
    p.add_argument("--n_outputs", type=int, default=4)
    p.add_argument("--label", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tau_plat_steps", type=int, default=150)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--loss_temperature", type=float, default=2.7)
    p.add_argument("--loss_logit_bias", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    args = p.parse_args()

    spike_times = [10, 80, 200]
    n_inputs = len(spike_times)

    res = run_forward_backward(
        T=args.T,
        n_inputs=n_inputs,
        n_hidden=args.n_hidden,
        n_outputs=args.n_outputs,
        spike_times_ms=spike_times,
        label=args.label,
        seed=args.seed,
        tau_plat_steps=args.tau_plat_steps,
        noise_std=args.noise_std,
        loss_temperature=args.loss_temperature,
        loss_logit_bias=args.loss_logit_bias,
        label_smoothing=args.label_smoothing,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    plot_forward(res, os.path.join(args.out_dir, "forward_pass.png"))
    plot_backward_readout(res, os.path.join(args.out_dir, "backward_readout.png"))
    plot_backward_soma(res, os.path.join(args.out_dir, "backward_soma.png"))
    plot_backward_dend(res, os.path.join(args.out_dir, "backward_dendrite.png"))


if __name__ == "__main__":
    main()
