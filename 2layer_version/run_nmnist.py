#!/usr/bin/env python3
"""
Run 2-layer model on NMNIST. Loads NMNIST data, converts to (T, n_inputs), trains the standalone model.
"""
import argparse
import os
import sys
import importlib.util
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Enable float64 for consistent numerics across all scripts (optional: set JAX_ENABLE_X64=0 to disable)
import jax
jax.config.update("jax_enable_x64", True)

from data import load_nmnist_data, create_nmnist_input_jax

sys.path.insert(0, _SCRIPT_DIR)
from jax import random
import numpy as np

_spec2 = importlib.util.spec_from_file_location("twolayer", os.path.join(_SCRIPT_DIR, "2layer.py"))
_net = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_net)
JAXEPropNetworkTwoLayer = _net.JAXEPropNetworkTwoLayer
train_network_two_layer = _net.train_network_two_layer

T_NMNIST = 300
N_EXTRA = 10
N_HIDDEN = 10
N_OUTPUTS = 10
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))


def main():
    p = argparse.ArgumentParser(description="Run 2-layer model on NMNIST")
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

    run_dir = os.path.join(_SCRIPT_DIR, "model", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    epochs = int(os.getenv("EPOCHS", "3"))
    batch_size = int(os.getenv("BATCH_SIZE", "1"))

    print("Loading NMNIST data...", flush=True)
    train_raw, test_raw = load_nmnist_data(data_path, train_samples_per_class=None, test_samples_per_class=None)
    input_kw = {"T": T_NMNIST}
    if args.no_kernel:
        input_kw["use_kernel"] = False
        if args.spike_amplitude is not None:
            input_kw["spike_amplitude"] = args.spike_amplitude
        print("Using input with use_kernel=False", flush=True)
    train_data = [(create_nmnist_input_jax(x, **input_kw), label) for x, label in train_raw]
    test_data = [(create_nmnist_input_jax(x, **input_kw), label) for x, label in test_raw]
    n_inputs = train_data[0][0].shape[1]

    network = JAXEPropNetworkTwoLayer(
        key, n_inputs=n_inputs, n_extra=N_EXTRA, n_hidden=N_HIDDEN, n_outputs=N_OUTPUTS, T=T_NMNIST
    )
    print(f"Train: {len(train_data)}, Test: {len(test_data)}, n_inputs: {n_inputs}", flush=True)
    print("Training...", flush=True)
    train_network_two_layer(network, train_data, test_data, run_dir, epochs, batch_size, "nmnist_two_layer", RANDOM_SEED)
    print(f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
