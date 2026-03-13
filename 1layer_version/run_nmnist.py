#!/usr/bin/env python3
"""
Run 1-layer 2comp model on NMNIST. Loads NMNIST data, converts to (T, n_inputs), trains the standalone model.
Same model as run_shd; only data loading is dataset-specific.
"""
import argparse
import os
import sys
from datetime import datetime

# Project root so we can import the shared data package
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Enable float64 for consistent numerics across all scripts (optional: set JAX_ENABLE_X64=0 to disable)
import jax
jax.config.update("jax_enable_x64", True)

from data import load_nmnist_data, create_nmnist_input_jax

# Standalone model (no dataset code)
sys.path.insert(0, _SCRIPT_DIR)
from model import (
    JAXEPropNetwork,
    train_network_jax,
    print_summary_statistics,
    initialize_numpy_weights,
    RANDOM_SEED,
)
from jax import random
import numpy as np

T_NMNIST = 300
N_HIDDEN = 30  # match typical NMNIST 1-layer setup from 2comp.py
N_OUTPUTS = 10


def main():
    p = argparse.ArgumentParser(description="Run 1-layer 2comp model on NMNIST")
    p.add_argument("--no_kernel", action="store_true", help="Input: use_kernel=False (no alpha kernel)")
    p.add_argument("--spike_amplitude", type=float, default=None, help="Spike amplitude when --no_kernel (default: 5.0)")
    args = p.parse_args()

    key = random.PRNGKey(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if "NMNIST_DATA_PATH" in os.environ:
        data_path = os.environ["NMNIST_DATA_PATH"]
    elif os.path.exists("/share/neurocomputation/Tim/NMNIST_data"):
        data_path = "/share/neurocomputation/Tim/NMNIST_data"
    else:
        data_path = os.path.expanduser("~/Documents/NMNIST_data")

    model_dir = os.path.join(_SCRIPT_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(model_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    epochs = int(os.getenv("EPOCHS", "10"))
    batch_size = int(os.getenv("BATCH_SIZE", "32"))

    print("Loading NMNIST data...", flush=True)
    train_raw, test_raw = load_nmnist_data(
        data_path,
        train_samples_per_class=None,
        test_samples_per_class=None,
    )
    print(f"Converting to (T={T_NMNIST}, n_inputs) format...", flush=True)
    input_kw = {"T": T_NMNIST}
    if args.no_kernel:
        input_kw["use_kernel"] = False
        if args.spike_amplitude is not None:
            input_kw["spike_amplitude"] = args.spike_amplitude
        print("Using input with use_kernel=False", flush=True)
    train_data = [(create_nmnist_input_jax(x, **input_kw), label) for x, label in train_raw]
    test_data = [(create_nmnist_input_jax(x, **input_kw), label) for x, label in test_raw]
    n_inputs = train_data[0][0].shape[1]

    print(f"Train: {len(train_data)}, Test: {len(test_data)}, n_inputs: {n_inputs}", flush=True)
    print("Creating network and initializing weights...", flush=True)
    np.random.seed(RANDOM_SEED)
    w_dend_np, w_soma_np, w_readout_np = initialize_numpy_weights(
        n_inputs=n_inputs, n_hidden=N_HIDDEN, n_outputs=N_OUTPUTS
    )
    network = JAXEPropNetwork(
        key, n_inputs=n_inputs, n_hidden=N_HIDDEN, n_outputs=N_OUTPUTS, T=T_NMNIST
    )
    network.hidden_layer.w_dend = jax.numpy.array(w_dend_np)
    network.hidden_layer.w_soma = jax.numpy.array(w_soma_np)
    network.readout_layer.w = jax.numpy.array(w_readout_np)

    hyperparams_lines = [
        "", "1-layer NMNIST run", "=" * 80,
        f"Random seed: {RANDOM_SEED}",
        f"Epochs: {epochs}, Batch size: {batch_size}",
        "=" * 80, "",
    ]
    with open(os.path.join(run_dir, "hyperparameters.txt"), "w") as f:
        f.write("\n".join(hyperparams_lines))

    print("Training...", flush=True)
    epoch_results, final_test_results, best_model_path, best_accuracy = train_network_jax(
        network, train_data, test_data,
        epochs=epochs, batch_size=batch_size, run_dir=run_dir,
    )
    model_save_path = os.path.join(run_dir, "nmnist_2comp_model_jax.pkl")
    network.save(model_save_path)
    print_summary_statistics(network, epoch_results, final_test_results, run_dir)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
