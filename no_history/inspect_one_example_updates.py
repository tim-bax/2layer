#!/usr/bin/env python3
import argparse
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data import create_shd_input_jax, load_shd_data
from config import NeuronConfig
from network import Network


def _summarize_abs(name: str, arr: jnp.ndarray) -> dict:
    flat = jnp.abs(arr).reshape(-1)
    return {
        "name": name,
        "count": int(flat.size),
        "l2": float(jnp.linalg.norm(flat)),
        "mean": float(jnp.mean(flat)),
        "median": float(jnp.median(flat)),
        "max": float(jnp.max(flat)),
        "min": float(jnp.min(flat)),
    }


def _print_stats(stats: dict):
    print(
        f"{stats['name']:>10s} | n={stats['count']:8d} | "
        f"mean={stats['mean']:.3e} | median={stats['median']:.3e} | "
        f"max={stats['max']:.3e} | l2={stats['l2']:.3e}"
    )


def parse_args():
    p = argparse.ArgumentParser(description="Inspect one-example update scales for no_history model.")
    p.add_argument("--T", type=int, default=700)
    p.add_argument("--n_hidden", type=int, default=64)
    p.add_argument("--n_outputs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Ignored; accepted for CLI compatibility with run_shd.py.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gradient_clip", type=float, default=5.0)
    p.add_argument("--loss_temperature", type=float, default=2.7)
    p.add_argument("--loss_count_bias", type=float, default=0.18)
    p.add_argument("--loss_label_smoothing", type=float, default=0.13)
    p.add_argument("--beta_s", type=float, default=1.0)
    p.add_argument("--beta_d", type=float, default=1.5)
    p.add_argument("--weight_scale", type=float, default=0.25)
    p.add_argument("--no_kernel", action="store_true")
    p.add_argument("--spike_amplitude", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--sample_idx", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    key = random.PRNGKey(args.seed)

    print("Loading SHD data...", flush=True)
    train_raw, _ = load_shd_data()
    input_kw = {
        "T": args.T,
        "use_kernel": not args.no_kernel,
        "spike_amplitude": args.spike_amplitude,
    }
    train_data = [(create_shd_input_jax(x, **input_kw), y) for x, y in train_raw]
    n_inputs = train_data[0][0].shape[1]
    idx = int(np.clip(args.sample_idx, 0, len(train_data) - 1))
    x, y = train_data[idx]

    config = NeuronConfig(
        beta_s=args.beta_s,
        beta_d=args.beta_d,
        weight_scale=args.weight_scale,
        loss_temperature=args.loss_temperature,
        loss_count_bias=args.loss_count_bias,
        loss_label_smoothing=args.loss_label_smoothing,
    )
    net = Network(
        key, n_inputs, args.n_hidden, args.n_outputs, config,
        optimizer=args.optimizer, beta1=args.beta1, beta2=args.beta2, adam_eps=args.adam_eps,
        dropout_rate=args.dropout, weight_decay=args.weight_decay,
    )

    w_d_before = net.hidden.w_dend.copy()
    w_s_before = net.hidden.w_soma.copy()
    w_r_before = net.readout.w.copy()

    loss, pred, gnorms = net.train_step(
        jnp.array(x), int(y), lr=args.lr, clip_value=args.gradient_clip
    )

    delta_d = net.hidden.w_dend - w_d_before
    delta_s = net.hidden.w_soma - w_s_before
    delta_r = net.readout.w - w_r_before

    print("\nOne-example update analysis")
    print(
        f"sample_idx={idx}  target={int(y)}  pred_before_update={pred}  "
        f"loss={loss:.6f}  optimizer={args.optimizer}"
    )
    print("\nAbsolute per-synapse |delta_w| stats")
    s_soma = _summarize_abs("soma", delta_s)
    s_dend = _summarize_abs("dend", delta_d)
    s_read = _summarize_abs("readout", delta_r)
    _print_stats(s_soma)
    _print_stats(s_dend)
    _print_stats(s_read)

    print("\nGradient L2 norms returned by train_step (pre-optimizer transform)")
    print(
        f"readout={gnorms['readout']:.3e}  soma={gnorms['soma']:.3e}  dend={gnorms['dend']:.3e}"
    )

    eps = 1e-20
    print("\nScale ratios using mean |delta_w|")
    print(f"soma / dend    = {s_soma['mean'] / (s_dend['mean'] + eps):.3e}")
    print(f"soma / readout = {s_soma['mean'] / (s_read['mean'] + eps):.3e}")
    print(f"dend / readout = {s_dend['mean'] / (s_read['mean'] + eps):.3e}")

    print("\nScale ratios using max |delta_w|")
    print(f"soma / dend    = {s_soma['max'] / (s_dend['max'] + eps):.3e}")
    print(f"soma / readout = {s_soma['max'] / (s_read['max'] + eps):.3e}")
    print(f"dend / readout = {s_dend['max'] / (s_read['max'] + eps):.3e}")


if __name__ == "__main__":
    main()
