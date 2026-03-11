#!/usr/bin/env python3
"""
Check initial activity of the 1-layer 2comp network before training.
Uses the same weight init as run_shd (weight_scale, readout_weight_scale).
Runs forward passes on N samples and reports readout/hidden activity so you can
tune init: aim for some variance in readout (not all silent or all ties) and
moderate hidden activity (not silent, not saturated).
"""
import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
jax.config.update("jax_enable_x64", True)

from data import load_shd_data, create_shd_input_jax

sys.path.insert(0, _SCRIPT_DIR)
import importlib.util

def _load_module():
    path = os.path.join(_SCRIPT_DIR, "model.py")
    spec = importlib.util.spec_from_file_location("onelayer_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    p = argparse.ArgumentParser(description="Check initial readout/hidden activity (no training)")
    p.add_argument("--T", type=int, default=700)
    p.add_argument("--n_hidden", type=int, default=64)
    p.add_argument("--n_outputs", type=int, default=20)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--weight_scale", type=float, default=0.25, help="Hidden layer init scale")
    p.add_argument("--readout_weight_scale", type=float, default=None, help="Readout init scale (default: same as weight_scale)")
    p.add_argument("--n_samples", type=int, default=200, help="Number of samples to run forward on")
    p.add_argument("--data_path", type=str, default=None, help="SHD data dir (default: env SHD_DATA_PATH or ~/data/hdspikes)")
    args = p.parse_args()

    readout_scale = args.readout_weight_scale if args.readout_weight_scale is not None else args.weight_scale

    # Data path: explicit > env > fallbacks
    if args.data_path:
        data_path = args.data_path
    elif os.environ.get("SHD_DATA_PATH"):
        data_path = os.environ["SHD_DATA_PATH"]
    elif os.path.isdir(os.path.expanduser("~/data/hdspikes")):
        data_path = os.path.expanduser("~/data/hdspikes")
    elif os.path.isdir(os.path.expanduser("~/Documents/Heidelberg_Data")):
        data_path = os.path.expanduser("~/Documents/Heidelberg_Data")
    elif os.path.isdir("/share/neurocomputation/Tim/SHD_data"):
        data_path = "/share/neurocomputation/Tim/SHD_data"
    else:
        data_path = os.path.expanduser("~/data/hdspikes")

    if not os.path.isdir(data_path):
        print("SHD data path not found. Tried:", flush=True)
        print("  --data_path, then SHD_DATA_PATH, then ~/data/hdspikes, ~/Documents/Heidelberg_Data", flush=True)
        print(f"  Result: {data_path}", flush=True)
        print("Run with: python check_init_activity.py --data_path /path/to/shd  (or set SHD_DATA_PATH)", flush=True)
        return 1

    mod = _load_module()
    JAXEPropNetwork = mod.JAXEPropNetwork
    initialize_numpy_weights = mod.initialize_numpy_weights
    NeuronConfig = mod.NeuronConfig

    key = jax.random.PRNGKey(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    print("Loading SHD data (train split, small subset)...", flush=True)
    train_raw, _ = load_shd_data(data_path, train_samples_per_class=50, test_samples_per_class=0)
    train_data = [(create_shd_input_jax(x, T=args.T), y) for x, y in train_raw]
    n_use = min(args.n_samples, len(train_data))
    train_data = train_data[:n_use]
    n_inputs = train_data[0][0].shape[1]  # (T, n_inputs) after create_shd_input_jax
    print(f"Using {n_use} samples, n_inputs={n_inputs}", flush=True)

    print(f"Initializing network: weight_scale={args.weight_scale}, readout_weight_scale={readout_scale}", flush=True)
    w_dend_np, w_soma_np, w_readout_np = initialize_numpy_weights(
        n_inputs=n_inputs,
        n_hidden=args.n_hidden,
        n_outputs=args.n_outputs,
        weight_scale=args.weight_scale,
        readout_weight_scale=readout_scale,
    )
    config = NeuronConfig()
    n_inputs = train_data[0][0].shape[1]
    network = JAXEPropNetwork(
        key,
        n_inputs=n_inputs,
        n_hidden=args.n_hidden,
        n_outputs=args.n_outputs,
        T=args.T,
        beta_s=1.0,
        beta_d=1.5,
        weight_scale=args.weight_scale,
    )
    network.hidden_layer.w_dend = jax.numpy.array(w_dend_np)
    network.hidden_layer.w_soma = jax.numpy.array(w_soma_np)
    network.readout_layer.w = jax.numpy.array(w_readout_np)

    # Forward on all samples and collect spike counts
    readout_counts_list = []   # each (n_outputs,)
    hidden_counts_list = []    # each (n_hidden,)
    for x, _ in train_data:
        x_jax = jax.numpy.array(np.asarray(x))
        mu, v, h, hidden_o, readout_v, readout_o, _, _, _ = network.forward(x_jax)
        ro_counts = np.array(jax.numpy.sum(readout_o, axis=0))   # (n_outputs,)
        hid_counts = np.array(jax.numpy.sum(hidden_o, axis=0))   # (n_hidden,)
        readout_counts_list.append(ro_counts)
        hidden_counts_list.append(hid_counts)

    readout_counts = np.array(readout_counts_list)   # (n_samples, n_outputs)
    hidden_counts = np.array(hidden_counts_list)     # (n_samples, n_hidden)

    # ---- Readout stats ----
    ro_per_neuron = np.mean(readout_counts, axis=0)   # (n_outputs,)
    ro_std = np.std(readout_counts, axis=0)
    ro_total_per_sample = np.sum(readout_counts, axis=1)  # (n_samples,)
    n_silent_ro = np.sum(np.max(readout_counts, axis=1) == 0)  # samples where every readout neuron had 0 spikes
    n_tie = np.sum(np.max(readout_counts, axis=1) == np.min(readout_counts, axis=1))  # all outputs same count
    ro_min_per_sample = np.min(readout_counts, axis=1)
    ro_max_per_sample = np.max(readout_counts, axis=1)
    n_varied = np.sum(ro_max_per_sample > ro_min_per_sample)  # at least two different counts
    # "Nearly silent": total readout so low that effective_error is tiny (e.g. < 20 spikes total)
    low_total = 20
    n_low_total = np.sum(ro_total_per_sample < low_total)
    # Output neurons that fire in almost no samples (silent in >90% of samples)
    pct_silent_per_neuron = np.mean(readout_counts == 0, axis=0) * 100  # (n_outputs,)
    n_often_silent = np.sum(pct_silent_per_neuron > 90)

    print()
    print("=" * 60)
    print("READOUT (output layer) activity at init")
    print("=" * 60)
    print(f"  Mean spikes per output neuron (over samples): min={ro_per_neuron.min():.2f}, max={ro_per_neuron.max():.2f}, mean={ro_per_neuron.mean():.2f}")
    print(f"  Std per output neuron (over samples):         min={ro_std.min():.2f}, max={ro_std.max():.2f}")
    print(f"  Total readout spikes per sample:              mean={ro_total_per_sample.mean():.1f}, std={ro_total_per_sample.std():.1f}, min={ro_total_per_sample.min():.0f}, max={ro_total_per_sample.max():.0f}")
    print(f"  Samples with at least one readout spike:      {n_use - n_silent_ro}/{n_use} ({100*(n_use-n_silent_ro)/n_use:.1f}%)")
    print(f"  Samples with total readout < {low_total} spikes (nearly silent): {n_low_total}/{n_use} ({100*n_low_total/n_use:.1f}%)  <- want 0")
    print(f"  Output neurons silent in >90% of samples:    {n_often_silent}/{args.n_outputs}  <- want 0")
    print(f"  Samples with varied readout (max != min):     {n_varied}/{n_use} ({100*n_varied/n_use:.1f}%)  <- want high for learning")
    print(f"  Samples with all outputs same count (tie):     {n_tie}/{n_use} ({100*n_tie/n_use:.1f}%)  <- want low")
    if n_silent_ro > 0:
        print(f"  WARNING: {n_silent_ro} samples had completely silent readout (all 0). Consider larger readout_weight_scale.")
    if n_low_total > 0:
        print(f"  WARNING: {n_low_total} samples had very low total readout (<{low_total} spikes). Effective_error will be tiny; consider larger readout_weight_scale.")
    if n_often_silent > 0:
        print(f"  WARNING: {n_often_silent} readout neurons are silent in >90% of samples. Consider larger readout_weight_scale.")
    if n_varied < n_use * 0.5:
        print(f"  WARNING: Less than half of samples have varied readout. Consider increasing readout_weight_scale (e.g. 0.4--0.5).")
    if ro_total_per_sample.mean() > 500:
        print(f"  NOTE: Very high readout activity (mean total {ro_total_per_sample.mean():.0f}). May be fine or consider slightly lower scale.")
    print()

    # ---- Hidden stats ----
    hid_per_neuron = np.mean(hidden_counts, axis=0)   # (n_hidden,)
    hid_total_per_sample = np.sum(hidden_counts, axis=1)
    n_hidden_silent = np.sum(np.max(hidden_counts, axis=0) == 0)  # neurons that never fired in any sample
    hid_max_per_neuron = np.max(hidden_counts, axis=0)

    print("=" * 60)
    print("HIDDEN layer activity at init")
    print("=" * 60)
    print(f"  Mean spikes per hidden neuron (over samples): min={hid_per_neuron.min():.2f}, max={hid_per_neuron.max():.2f}, mean={hid_per_neuron.mean():.2f}")
    print(f"  Total hidden spikes per sample:                mean={hid_total_per_sample.mean():.1f}, std={hid_total_per_sample.std():.1f}, min={hid_total_per_sample.min():.0f}, max={hid_total_per_sample.max():.0f}")
    print(f"  Hidden neurons that never fired (over {n_use} samples): {n_hidden_silent}/{args.n_hidden}")
    if n_hidden_silent > args.n_hidden * 0.5:
        print(f"  WARNING: Many hidden neurons are silent. Consider larger weight_scale (e.g. 0.35).")
    if hid_total_per_sample.mean() > 0.5 * args.T * args.n_hidden:
        print(f"  NOTE: Very high hidden activity (saturation risk). Consider slightly lower weight_scale.")
    print()
    print("Summary: aim for varied readout (high % varied, low % tie) and non-silent hidden; adjust weight_scale / readout_weight_scale and re-run.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
