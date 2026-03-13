#!/usr/bin/env python3
"""
Load a saved 2-layer SHD model (.pkl), run 10 samples (from test set), and report gradient
component norms for each part: global error, readout gradient, hidden soma/dendrite,
extra soma/dendrite, and related diagnostics (effective error, surrogates, eligibility).
"""
import os
import sys
import argparse
import importlib.util

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax import random

from data import load_shd_data, create_shd_input_jax


def _load_twolayer_module(lowmemory: bool):
    basename = "2layer_lowmemory.py" if lowmemory else "2layer.py"
    path = os.path.join(_SCRIPT_DIR, basename)
    spec = importlib.util.spec_from_file_location("twolayer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.JAXEPropNetworkTwoLayer


def main():
    p = argparse.ArgumentParser(description="Assess gradient components on 10 samples for a saved 2-layer SHD model")
    p.add_argument("--pkl", type=str, default=os.path.join(_SCRIPT_DIR, "model", "shd_two_layer_65_10.pkl"),
                   help="Path to .pkl model")
    p.add_argument("--n_samples", type=int, default=10, help="Number of samples to run")
    p.add_argument("--split", type=str, default="test", choices=("train", "test"),
                   help="Use 'train' or 'test' set for samples")
    p.add_argument("--lowmemory", action="store_true", help="Use 2layer_lowmemory.py")
    p.add_argument("--no_kernel", action="store_true", help="Input: use_kernel=False (match run_shd --no_kernel)")
    p.add_argument("--spike_amplitude", type=float, default=None, help="Spike amplitude when --no_kernel (default: 5.0)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not os.path.isfile(args.pkl):
        print(f"Error: model file not found: {args.pkl}")
        sys.exit(1)

    JAXEPropNetworkTwoLayer = _load_twolayer_module(args.lowmemory)
    key = random.PRNGKey(args.seed)
    network = JAXEPropNetworkTwoLayer.load(args.pkl, key=key)
    T = network.T
    n_inputs = network.n_inputs

    if "SHD_DATA_PATH" in os.environ:
        data_path = os.environ["SHD_DATA_PATH"]
    elif os.path.exists("/share/neurocomputation/Tim/SHD_data"):
        data_path = "/share/neurocomputation/Tim/SHD_data"
    else:
        data_path = os.path.expanduser("~/Documents/Heidelberg_Data")

    if not os.path.isdir(data_path):
        print(f"Error: SHD data path not found: {data_path}")
        print("Set SHD_DATA_PATH or place data in ~/Documents/Heidelberg_Data")
        sys.exit(1)

    print("Loading SHD data...")
    train_raw, test_raw = load_shd_data(data_path, train_samples_per_class=None, test_samples_per_class=None)
    input_kw = {"T": T}
    if args.no_kernel:
        input_kw["use_kernel"] = False
        if args.spike_amplitude is not None:
            input_kw["spike_amplitude"] = args.spike_amplitude
        print("Using input with use_kernel=False", flush=True)
    train_data = [(create_shd_input_jax(x, **input_kw), label) for x, label in train_raw]
    test_data = [(create_shd_input_jax(x, **input_kw), label) for x, label in test_raw]
    if args.split == "train":
        data = train_data
    else:
        data = test_data
    n_available = len(data)
    n_samples = min(args.n_samples, n_available)
    samples = data[:n_samples]
    print(f"Using first {n_samples} samples from {args.split} set (available: {n_available})")
    print(f"Model: T={T}, n_inputs={n_inputs}, n_extra={network.n_extra}, n_hidden={network.n_hidden}, n_outputs={network.n_outputs}\n")

    params = network.get_params()
    clip = getattr(network, "gradient_clip", 5.0)

    # Keys we care about for "gradient parts" and related diagnostics
    component_keys = [
        "global_errors",
        "effective_error_hidden",
        "sigma_prime_readout",
        "sigma_prime_hidden",
        "sigma_prime_extra",
        "h_prime_hidden",
        "h_prime_extra",
        "E_readout",
        "E_soma_hidden",
        "E_soma_extra",
        "dmu_dw_hidden",
        "dmu_dw_extra",
        "grad_readout",
        "grad_soma_hidden",
        "grad_dend_hidden",
        "grad_extra_soma",
        "grad_extra_dend",
    ]

    rows = []
    for i, (x, target) in enumerate(samples):
        x_j = jnp.asarray(x)
        d = network.get_single_sample_diagnostics(params, x_j, target)
        rows.append(d)

    # Per-sample table: main gradient parts + loss/target/pred
    main_cols = ["loss", "global_errors", "grad_readout", "grad_soma_hidden", "grad_dend_hidden", "grad_extra_soma", "grad_extra_dend"]
    print("Per-sample (target, prediction, loss, global_error, gradient norms):")
    print("-" * 95)
    header = f"{'i':>3} {'tgt':>3} {'pred':>4}  " + "  ".join(f"{k:>11}" for k in main_cols)
    print(header)
    for i, d in enumerate(rows):
        parts = [f"{i:3d}", f"{d['target']:3d}", f"{d['prediction']:4d}"]
        for k in main_cols:
            parts.append(f"{d[k]:11.4e}")
        print("  ".join(parts))
    print()

    # Summary stats over samples for each component
    print("Summary over samples (mean ± std, min, max):")
    print("-" * 80)
    for k in component_keys:
        vals = [r[k] for r in rows]
        mean, std = np.mean(vals), np.std(vals)
        mn, mx = np.min(vals), np.max(vals)
        print(f"  {k:25s}  mean={mean:10.4e}  std={std:10.4e}  min={mn:10.4e}  max={mx:10.4e}")
    print()

    # Gradient-only summary (the "parts" the user asked for)
    grad_keys = ["grad_readout", "grad_soma_hidden", "grad_dend_hidden", "grad_extra_soma", "grad_extra_dend"]
    print("Gradient norms (Frobenius) by part:")
    print("-" * 60)
    for k in grad_keys:
        vals = [r[k] for r in rows]
        print(f"  {k:22s}  mean={np.mean(vals):.4e}  std={np.std(vals):.4e}  range=[{np.min(vals):.4e}, {np.max(vals):.4e}]")
    print("Done.")


if __name__ == "__main__":
    main()
