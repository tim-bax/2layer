#!/usr/bin/env python3
"""
Sequence-discrimination demo for the no_history two-compartment neuron.

Four panels in a 2×2 grid (same layout each): somatic voltage (pre-reset),
dynamic firing threshold v_th - gamma*h, and translucent grey bars at input times.
Optional Gaussian noise on dend/soma drives (see ``--no-noise``).

Timing: sequence length 80 ms (dt=1), spikes at t=10 and t=50.
Plateau window T_p = 50 (ms).

  Shared weights (channel 0=A, channel 1=B): w_dend=(1.1, 0.1), w_soma=(0.3, 0.6).
  Only the spike pattern changes: A->B, B->A, A->A, B->B.

  python sequence_discrimination_demo.py              # default drive noise
  python sequence_discrimination_demo.py --no-noise  # clean drives; saves …_nonoise.png

Outputs: demo/sequence_discrimination.png (default) and .svg; with ``--no-noise``,
  demo/sequence_discrimination_nonoise.png and .svg unless ``-o`` is set.
"""
from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp
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

from config import NeuronConfig  # noqa: E402
from two_comp_neuron import TwoCompNeuron  # noqa: E402

_TRACE_LW = 2.2
_NOISE_STD = 0.012
# Semi-transparent grey vertical bands marking input times (ms); width ~ one dt bin.
_INPUT_BAR_FACE = (0.55, 0.55, 0.55, 0.28)
_INPUT_BAR_WIDTH_MS = 1.0


def build_input_from_events(T: int, n_inputs: int, events: list[tuple[int, int]]) -> jnp.ndarray:
    """One-hot spikes: events are (time_ms, channel_index)."""
    x = np.zeros((T, n_inputs), dtype=np.float64)
    for t_ms, ch in events:
        if not (0 <= t_ms < T and 0 <= ch < n_inputs):
            raise ValueError(f"bad event ({t_ms}, {ch}) for T={T}, K={n_inputs}")
        x[t_ms, ch] = 1.0
    return jnp.array(x)


def simulate_case(
    *,
    w_dend: jnp.ndarray,
    w_soma: jnp.ndarray,
    events: list[tuple[int, int]],
    T: int,
    T_p_ms: int,
    noise_std: float,
    rng: np.random.Generator,
) -> dict:
    n_inputs = w_dend.shape[1]
    config = NeuronConfig()
    n_neurons = 1

    neuron = TwoCompNeuron(jax.random.PRNGKey(0), n_neurons, n_inputs, config)
    neuron.w_dend = w_dend
    neuron.w_soma = w_soma
    T_p = jnp.array([T_p_ms], dtype=jnp.int32)

    x_input = build_input_from_events(T, n_inputs, events)
    dend_inputs = x_input @ w_dend.T
    soma_inputs = x_input @ w_soma.T

    noise_d = jnp.array(rng.normal(0, noise_std, size=(T, 1)))
    noise_s = jnp.array(rng.normal(0, noise_std, size=(T, 1)))
    dend_inputs = dend_inputs + noise_d
    soma_inputs = soma_inputs + noise_s

    alpha_s = neuron.alpha_s
    alpha_d = neuron.alpha_d
    carry = neuron.init_carry()

    mu_hist = np.zeros(T)
    v_hist = np.zeros(T)
    v_pre_hist = np.zeros(T)
    h_hist = np.zeros(T, dtype=np.int32)

    for t in range(T):
        carry, _o, v_pre_reset, h, _h_prev, _mu_at_tp = TwoCompNeuron.forward_step(
            carry,
            dend_inputs[t],
            soma_inputs[t],
            jnp.array(t, dtype=jnp.int32),
            alpha_s,
            alpha_d,
            T_p,
            config,
        )
        mu, v, _hc, _tp, _matp, _E, _dmu, _dmu_atp = carry
        mu_hist[t] = float(mu[0])
        v_hist[t] = float(v[0])
        v_pre_hist[t] = float(v_pre_reset[0])
        h_hist[t] = int(h[0])

    dyn_th = config.v_th - config.gamma * h_hist.astype(np.float64)
    return {
        "T": T,
        "config": config,
        "events": events,
        "mu": mu_hist,
        "v_pre": v_pre_hist,
        "h": h_hist,
        "dyn_threshold": dyn_th,
        "T_p": T_p_ms,
    }


def plot_four_panel(
    results: list[tuple[str, dict]],
    out_path: str,
    *,
    noise_std: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes_flat = axes.ravel()
    t0 = np.arange(results[0][1]["T"])

    half_w = 0.5 * _INPUT_BAR_WIDTH_MS

    for ax, (title, res) in zip(axes_flat, results):
        v_pre = res["v_pre"]
        dyn = res["dyn_threshold"]

        for t_ms, _ch in res["events"]:
            ax.axvspan(
                t_ms - half_w,
                t_ms + half_w,
                facecolor=_INPUT_BAR_FACE,
                edgecolor="none",
                zorder=0,
            )

        v_label = (
            r"$v$ (soma, pre-reset, noisy drive)"
            if noise_std > 0
            else r"$v$ (soma, pre-reset, no drive noise)"
        )
        ax.plot(
            t0,
            v_pre,
            color="#c0392b",
            linewidth=_TRACE_LW,
            label=v_label,
            zorder=2,
        )
        ax.plot(
            t0,
            dyn,
            color="k",
            linestyle="--",
            linewidth=_TRACE_LW,
            label=r"$v_\mathrm{th} - \gamma h$",
            zorder=2,
        )

        label_ab = ("A", "B")
        ax.set_ylabel(r"$v$")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, zorder=1)

        ax.legend(loc="upper right", fontsize=8)

        y1, y2 = ax.get_ylim()
        y_text = y2 - 0.06 * (y2 - y1)
        for t_ms, ch in res["events"]:
            ax.text(
                t_ms + 0.9,
                y_text,
                label_ab[ch],
                fontsize=9,
                color="gray",
                va="top",
            )

    for ax in axes[1, :]:
        ax.set_xlabel("Time (ms)")
    noise_note = (
        rf"$\mathrm{{noise}}={noise_std}$ on dend/soma drive"
        if noise_std > 0
        else r"no noise on dend/soma drive"
    )
    fig.suptitle(
        rf"Two-compartment sequence discrimination ($T={results[0][1]['T']}$ ms, "
        rf"$T_p={results[0][1]['T_p']}$ ms; dend $[1.1,0.1]$, soma $[0.3,0.6]$ for $(A,B)$; "
        rf"{noise_note})",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    svg_path = os.path.splitext(out_path)[0] + ".svg"
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Saved {svg_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-compartment sequence discrimination figure (no_history model).",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Turn off Gaussian noise on dendritic and somatic drives (cleaner traces).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Output PNG path (SVG written alongside). Default: sequence_discrimination.png "
        "or sequence_discrimination_nonoise.png with --no-noise.",
    )
    args = parser.parse_args()

    noise_std = 0.0 if args.no_noise else _NOISE_STD
    if args.output:
        out_path = (
            args.output
            if os.path.isabs(args.output)
            else os.path.join(_SCRIPT_DIR, args.output)
        )
    else:
        base = (
            "sequence_discrimination_nonoise.png"
            if args.no_noise
            else "sequence_discrimination.png"
        )
        out_path = os.path.join(_SCRIPT_DIR, base)

    T = 80
    T_p_ms = 50

    # Channel 0 = A, channel 1 = B (same weights for all four panels).
    w_dend = jnp.array([[1.1, 0.1]])
    w_soma = jnp.array([[0.3, 0.6]])

    rng = np.random.default_rng(42)

    cases: list[tuple[str, list[tuple[int, int]]]] = [
        (r"A $\to$ B", [(10, 0), (50, 1)]),
        (r"B $\to$ A", [(10, 1), (50, 0)]),
        (r"A $\to$ A", [(10, 0), (50, 0)]),
        (r"B $\to$ B", [(10, 1), (50, 1)]),
    ]

    results: list[tuple[str, dict]] = []
    for title, ev in cases:
        r = simulate_case(
            w_dend=w_dend,
            w_soma=w_soma,
            events=ev,
            T=T,
            T_p_ms=T_p_ms,
            noise_std=noise_std,
            rng=rng,
        )
        results.append((title, r))

    plot_four_panel(results, out_path, noise_std=noise_std)


if __name__ == "__main__":
    main()
