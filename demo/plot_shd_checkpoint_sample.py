#!/usr/bin/env python3
"""
Load a trained no_history checkpoint (from run_shd.py --save_model), draw a random
SHD sample with the same binning as recorded in the checkpoint meta, run the hidden
forward pass, pick a hidden neuron that entered a plateau at least once, and plot:

  1) Input raster (one event row per input channel).
  2) That hidden unit's plateau state h (step) and somatic spikes o during the trial.

Usage (from repo root)::

    python demo/plot_shd_checkpoint_sample.py --checkpoint outputs/shd_run1

Requires the checkpoint pair ``<path>.npz`` and ``<path>.meta.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
_NO_HISTORY = os.path.join(_ROOT, "no_history")
_DATA_ROOT = _ROOT
for _p in (_NO_HISTORY, _DATA_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _peek_checkpoint_meta(checkpoint_base: str) -> dict:
    base = checkpoint_base[:-4] if checkpoint_base.endswith(".npz") else checkpoint_base
    meta_path = base + ".meta.json"
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing meta JSON: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_jax_precision(meta: dict, override: str | None) -> str:
    if override:
        return override
    return str(meta.get("extra", {}).get("precision", "64"))


def main():
    p = argparse.ArgumentParser(description="Raster + plateau plot for one SHD sample via saved model.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(_ROOT, "outputs", "shd_trained"),
        help="Checkpoint prefix (same as run_shd.py --save_model; loads .npz + .meta.json).",
    )
    p.add_argument("--split", choices=("train", "test"), default="test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Fixed dataset index; if omitted, random (controlled by --seed).",
    )
    p.add_argument(
        "--max_tries",
        type=int,
        default=128,
        help="Resample random indices until a hidden neuron with ≥1 plateau step is found.",
    )
    p.add_argument("--precision", choices=("32", "64"), default=None,
                   help="JAX x64; default read from checkpoint meta extra.precision.")
    p.add_argument("--out", type=str, default=os.path.join(_SCRIPT_DIR, "shd_checkpoint_raster.png"))
    args = p.parse_args()

    meta0 = _peek_checkpoint_meta(args.checkpoint)
    prec = _resolve_jax_precision(meta0, args.precision)

    import jax
    jax.config.update("jax_enable_x64", prec == "64")
    import jax.numpy as jnp
    from jax import jit, lax
    from jax import random

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from checkpoint import load_checkpoint
    from data.shd_binned import load_shd_binned

    @jit
    def hidden_traces_scan(
        x_input,
        w_dend,
        w_soma,
        w_readout,
        alpha_s,
        alpha_d,
        alpha_m,
        T_p,
        config,
    ):
        """Same dynamics as network._predict_only; returns (T, N) plateau h and hidden spikes o."""
        dend_inputs = x_input @ w_dend.T
        soma_inputs = x_input @ w_soma.T
        t_steps = x_input.shape[0]
        n_hidden = w_dend.shape[0]
        n_outputs = w_readout.shape[0]
        time_indices = jnp.arange(t_steps, dtype=jnp.int32)

        def step(carry, inputs):
            mu, v, h, t_prime, mu_at_tp, r_v, r_counts = carry
            dend_in, soma_in, t = inputs

            t_prime_new = jnp.where(t == 0, 0, jnp.where(h == 1, t_prime, t))
            mu_new = jnp.where(t > 0, alpha_d * mu + (1 - h) * dend_in, dend_in)
            mu_at_tp_new = jnp.where(h == 0, mu_new, mu_at_tp)

            plat_dur = t - t_prime_new
            h_new = jnp.where(
                (mu_at_tp_new >= config.mu_th)
                & (plat_dur <= T_p)
                & (plat_dur >= 0),
                1,
                0,
            ).astype(jnp.int32)

            v_pre = jnp.where(t > 0, alpha_s * v + soma_in, soma_in)
            o_h = jnp.where(v_pre >= config.v_th - config.gamma * h_new, 1, 0).astype(jnp.int32)
            v_new = v_pre * (1 - o_h)

            r_in = o_h.astype(jnp.float64) @ w_readout.T
            r_v_new = alpha_m * r_v + r_in
            r_o = jnp.where(r_v_new >= config.v_th, 1, 0).astype(jnp.int32)
            r_v_new = r_v_new * (1 - r_o)
            r_counts_new = r_counts + r_o

            new_carry = (mu_new, v_new, h_new, t_prime_new, mu_at_tp_new, r_v_new, r_counts_new)
            out_h = h_new.astype(jnp.float64)
            out_o = o_h.astype(jnp.float64)
            return new_carry, (out_h, out_o)

        init = (
            jnp.zeros(n_hidden),
            jnp.zeros(n_hidden),
            jnp.zeros(n_hidden, dtype=jnp.int32),
            jnp.zeros(n_hidden, dtype=jnp.int32),
            jnp.zeros(n_hidden),
            jnp.zeros(n_outputs),
            jnp.zeros(n_outputs),
        )
        _, (h_tr, o_tr) = lax.scan(step, init, (dend_inputs, soma_inputs, time_indices))
        return h_tr, o_tr

    net, meta = load_checkpoint(args.checkpoint, key=random.PRNGKey(args.seed))

    dm = meta.get("extra", {}).get("data", {})
    bin_size_ms = float(dm.get("bin_size_ms", 4.0))
    collapse_factor = int(dm.get("collapse_factor", 5))
    max_duration_ms = float(dm.get("max_duration_ms", 1400.0))
    binarize = bool(dm.get("binarize", False))
    input_scale = float(dm.get("input_scale", 1.0))

    np_dtype = np.float64 if prec == "64" else np.float32
    X_tr, y_tr, _, X_te, y_te, _ = load_shd_binned(
        bin_size_ms=bin_size_ms,
        collapse_factor=collapse_factor,
        max_duration_ms=max_duration_ms,
        binarize=binarize,
        dtype=np_dtype,
    )
    if args.split == "test":
        Xs, ys = X_te, y_te
    else:
        Xs, ys = X_tr, y_tr

    if input_scale != 1.0:
        Xs = Xs * input_scale

    n_samples = len(ys)
    rng = np.random.default_rng(args.seed)

    w_d = net.hidden.w_dend
    w_s = net.hidden.w_soma
    w_r = net.readout.w
    alpha_s = net.hidden.alpha_s
    alpha_d = net.hidden.alpha_d
    alpha_m = net.readout.alpha_m
    T_p = net.hidden.T_p
    cfg = net.config

    idx = None
    h_np = None
    o_np = None
    x_np = None
    y_true = None

    tries = max(1, args.max_tries)
    for attempt in range(tries):
        if args.sample_index is not None:
            idx = int(args.sample_index) % n_samples
        else:
            idx = int(rng.integers(0, n_samples))

        x_np = np.asarray(Xs[idx], dtype=np_dtype)
        if x_np.ndim != 2 or x_np.shape[1] != net.n_inputs:
            raise ValueError(
                f"Sample shape {x_np.shape} incompatible with checkpoint n_inputs={net.n_inputs}. "
                "Check SHD preprocessing matches meta extra.data."
            )
        x_j = jnp.asarray(x_np)
        h_tr, o_tr = hidden_traces_scan(x_j, w_d, w_s, w_r, alpha_s, alpha_d, alpha_m, T_p, cfg)
        h_np = np.asarray(h_tr)
        o_np = np.asarray(o_tr)
        plateau_steps = h_np.sum(axis=0)
        cand = np.where(plateau_steps > 0)[0]
        if len(cand) > 0:
            sel = int(rng.choice(cand))
            y_true = int(ys[idx])
            break

        if args.sample_index is not None:
            raise RuntimeError(
                f"Fixed --sample_index={args.sample_index} has no hidden plateau on any neuron. "
                "Pick another index or omit --sample_index."
            )
    else:
        raise RuntimeError(
            f"No sample with ≥1 plateau found in {tries} random draws. "
            "Try another --seed, use --split train, or train longer."
        )

    pred = int(net.predict(jnp.asarray(x_np)))

    dt = float(cfg.dt)
    t_ms = np.arange(h_np.shape[0], dtype=np.float64) * dt

    # --- Figure: raster + plateau / spikes ---
    fig, (ax_r, ax_h) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.08},
    )
    n_in = x_np.shape[1]
    events = []
    for k in range(n_in):
        times = []
        for t_bin in range(x_np.shape[0]):
            c = float(x_np[t_bin, k])
            if c <= 0:
                continue
            n_ev = int(min(max(round(c), 1), 12))
            times.extend([t_bin] * n_ev)
        events.append(np.asarray(times, dtype=np.float64) * dt)

    ax_r.eventplot(events, lineoffsets=np.arange(n_in), linelengths=0.85, colors="black", linewidths=0.6)
    ax_r.set_ylabel("input channel")
    ax_r.set_title(
        rf"SHD sample idx={idx} ({args.split})  label={y_true}  pred={pred}  "
        rf"hidden $i={sel}$  $T_{{p,i}}={int(np.asarray(T_p[sel]))}$ bins "
        rf"({float(np.asarray(T_p[sel])) * dt:.1f} ms)"
    )
    ax_r.set_ylim(-0.5, n_in - 0.5)
    ax_r.invert_yaxis()

    h_i = np.asarray(h_np[:, sel])
    o_i = np.asarray(o_np[:, sel])
    ax_h.fill_between(
        t_ms, 0.0, h_i, step="post", color="0.75", alpha=0.85, label="plateau $h$",
    )
    ax_h.step(t_ms, h_i, where="post", color="0.35", linewidth=1.0)
    spike_t = t_ms[o_i > 0.5]
    ax_h.vlines(spike_t, ymin=-0.05, ymax=1.05, colors="C3", linewidths=1.2, label="hidden soma spike")
    ax_h.set_ylim(-0.15, 1.15)
    ax_h.set_ylabel(f"hidden {sel}\n$h$ / spikes")
    ax_h.set_xlabel("time (ms)")
    ax_h.legend(loc="upper right", fontsize=8)
    ax_h.grid(True, alpha=0.25)

    fig.tight_layout()
    out_png = args.out
    fig.savefig(out_png, dpi=150)
    svg_path = os.path.splitext(out_png)[0] + ".svg"
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    main()
