#!/usr/bin/env python3
"""
Run 1-layer 2comp model on SHD. Loads SHD data, converts to (T, n_inputs), trains the standalone model.
Edit the HYPERPARAMETERS block below or pass CLI args to change settings.
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

from data import load_shd_data, create_shd_input_jax

# Standalone model (no dataset code)
sys.path.insert(0, _SCRIPT_DIR)
from model import (
    JAXEPropNetwork,
    train_network_jax,
    print_summary_statistics,
    initialize_numpy_weights,
    NeuronConfig,
)
from jax import random, jit
import jax.numpy as jnp
import numpy as np
from typing import List

# JAX-kernel conversion (same as compare.py) for --compare_style to replicate compare's result
@jit
def _alpha_kernel_jax(t_vals, tau):
    k = (t_vals / tau) * jnp.exp(-t_vals / tau)
    return jnp.where(t_vals < 0, 0.0, k)


def _create_shd_input_compare_style(shd_data: List[np.ndarray], T: int, tau_alpha: float = 3.3, spike_amplitude: float = 5.0) -> np.ndarray:
    """Same conversion as compare.py (JAX kernel) so run_shd can replicate compare's loss/trajectory."""
    n_units = len(shd_data)
    x_input = np.zeros((T, n_units))
    kernel_len = int(10 * tau_alpha)
    t_vals = np.arange(kernel_len)
    k = _alpha_kernel_jax(jnp.array(t_vals), tau_alpha)
    peak_value = np.exp(-1)
    k_normalized = np.array(k) * (spike_amplitude / peak_value)
    for unit_idx, spike_times in enumerate(shd_data):
        for spike_time in spike_times:
            spike_time_int = int(spike_time)
            if 0 <= spike_time_int < T:
                kernel_start = spike_time_int
                kernel_end = min(kernel_start + kernel_len, T)
                kernel_length_used = kernel_end - kernel_start
                if kernel_length_used > 0:
                    x_input[kernel_start:kernel_end, unit_idx] += k_normalized[:kernel_length_used]
    return x_input

# -----------------------------------------------------------------------------
# HYPERPARAMETERS — edit these or override via command line
# -----------------------------------------------------------------------------
# Architecture & data
T_SHD = 700
N_HIDDEN = 64
N_OUTPUTS = 20
RANDOM_SEED = 12
EPOCHS = 30
BATCH_SIZE = 32
# Learning rates (hidden dendritic, hidden somatic, readout)
LR_HIDDEN_DEND = 0.045
LR_HIDDEN_SOMA = 0.00015
LR_READOUT = 0.035
# Regularization & training
WEIGHT_DECAY = 0.00001
GRADIENT_CLIP = 5.0
# Loss (softmax temperature, count bias, label smoothing)
LOSS_TEMPERATURE = 2.7
LOSS_COUNT_BIAS = 0.18
LOSS_LABEL_SMOOTHING = 0.13
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Run 1-layer 2comp model on SHD")
    p.add_argument("--T", type=int, default=None, help=f"Time steps (default: {T_SHD})")
    p.add_argument("--n_hidden", type=int, default=None, help=f"Hidden units (default: {N_HIDDEN})")
    p.add_argument("--n_outputs", type=int, default=None, help=f"Output classes (default: {N_OUTPUTS})")
    p.add_argument("--seed", type=int, default=None, help=f"Random seed (default: {RANDOM_SEED})")
    p.add_argument("--epochs", type=int, default=None, help=f"Epochs (default: {EPOCHS})")
    p.add_argument("--batch_size", type=int, default=None, help=f"Batch size (default: {BATCH_SIZE})")
    p.add_argument("--lr_hidden_dend", type=float, default=None)
    p.add_argument("--lr_hidden_soma", type=float, default=None)
    p.add_argument("--lr_readout", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--gradient_clip", type=float, default=None)
    p.add_argument("--loss_temperature", type=float, default=None)
    p.add_argument("--loss_count_bias", type=float, default=None)
    p.add_argument("--loss_label_smoothing", type=float, default=None)
    p.add_argument("--compare_style", action="store_true", help="Use same JAX-kernel conversion as compare.py (replicate compare's result)")
    args = p.parse_args()
    def _int(name, default): v = getattr(args, name); return v if v is not None else default
    def _float(name, default): v = getattr(args, name); return v if v is not None else default
    return {
        "T_SHD": _int("T", T_SHD),
        "N_HIDDEN": _int("n_hidden", N_HIDDEN),
        "N_OUTPUTS": _int("n_outputs", N_OUTPUTS),
        "RANDOM_SEED": _int("seed", RANDOM_SEED),
        "EPOCHS": _int("epochs", EPOCHS),
        "BATCH_SIZE": _int("batch_size", BATCH_SIZE),
        "LR_HIDDEN_DEND": _float("lr_hidden_dend", LR_HIDDEN_DEND),
        "LR_HIDDEN_SOMA": _float("lr_hidden_soma", LR_HIDDEN_SOMA),
        "LR_READOUT": _float("lr_readout", LR_READOUT),
        "WEIGHT_DECAY": _float("weight_decay", WEIGHT_DECAY),
        "GRADIENT_CLIP": _float("gradient_clip", GRADIENT_CLIP),
        "LOSS_TEMPERATURE": _float("loss_temperature", LOSS_TEMPERATURE),
        "LOSS_COUNT_BIAS": _float("loss_count_bias", LOSS_COUNT_BIAS),
        "LOSS_LABEL_SMOOTHING": _float("loss_label_smoothing", LOSS_LABEL_SMOOTHING),
        "COMPARE_STYLE": getattr(args, "compare_style", False),
    }


def main():
    cfg = parse_args()
    T, n_hidden, n_outputs = cfg["T_SHD"], cfg["N_HIDDEN"], cfg["N_OUTPUTS"]
    seed, epochs, batch_size = cfg["RANDOM_SEED"], cfg["EPOCHS"], cfg["BATCH_SIZE"]

    key = random.PRNGKey(seed)
    np.random.seed(seed)

    # Data path
    if "SHD_DATA_PATH" in os.environ:
        data_path = os.environ["SHD_DATA_PATH"]
    elif os.path.exists("/share/neurocomputation/Tim/SHD_data"):
        data_path = "/share/neurocomputation/Tim/SHD_data"
    else:
        data_path = os.path.expanduser("~/data/hdspikes")

    model_dir = os.path.join(_SCRIPT_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(model_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    print("Loading SHD data...", flush=True)
    train_raw, test_raw = load_shd_data(
        data_path,
        train_samples_per_class=None,
        test_samples_per_class=None,
    )
    compare_style = cfg.get("COMPARE_STYLE", False)
    if compare_style:
        print(f"Converting to (T={T}, n_inputs) format (compare-style JAX kernel)...", flush=True)
        create_fn = lambda x, T=T: _create_shd_input_compare_style(x, T)
    else:
        print(f"Converting to (T={T}, n_inputs) format (data package NumPy kernel)...", flush=True)
        create_fn = lambda x, T=T: create_shd_input_jax(x, T=T)
    train_data = [(create_fn(x), label) for x, label in train_raw]
    test_data = [(create_fn(x), label) for x, label in test_raw]
    n_inputs = train_data[0][0].shape[1]

    print(f"Train: {len(train_data)}, Test: {len(test_data)}, n_inputs: {n_inputs}", flush=True)
    print(f"Creating network and initializing weights...", flush=True)
    np.random.seed(seed)
    w_dend_np, w_soma_np, w_readout_np = initialize_numpy_weights(
        n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs
    )
    network = JAXEPropNetwork(
        key, n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs, T=T,
        learning_rate_hidden_dendritic=cfg["LR_HIDDEN_DEND"],
        learning_rate_hidden_somatic=cfg["LR_HIDDEN_SOMA"],
        learning_rate_readout=cfg["LR_READOUT"],
        weight_decay=cfg["WEIGHT_DECAY"],
        gradient_clip=cfg["GRADIENT_CLIP"],
        loss_temperature=cfg["LOSS_TEMPERATURE"],
        loss_count_bias=cfg["LOSS_COUNT_BIAS"],
        loss_label_smoothing=cfg["LOSS_LABEL_SMOOTHING"],
    )
    network.hidden_layer.w_dend = jax.numpy.array(w_dend_np)
    network.hidden_layer.w_soma = jax.numpy.array(w_soma_np)
    network.readout_layer.w = jax.numpy.array(w_readout_np)

    hyperparams_lines = [
        "", "1-layer SHD run", "=" * 80,
        f"Random seed: {seed}",
        f"Conversion: {'compare-style (JAX kernel)' if compare_style else 'data package (NumPy kernel)'}",
        f"Epochs: {epochs}, Batch size: {batch_size}",
        f"LR hidden dend: {cfg['LR_HIDDEN_DEND']}, soma: {cfg['LR_HIDDEN_SOMA']}, readout: {cfg['LR_READOUT']}",
        f"Loss: temp={cfg['LOSS_TEMPERATURE']}, bias={cfg['LOSS_COUNT_BIAS']}, smoothing={cfg['LOSS_LABEL_SMOOTHING']}",
        "=" * 80, "",
    ]
    with open(os.path.join(run_dir, "hyperparameters.txt"), "w") as f:
        f.write("\n".join(hyperparams_lines))

    print("Training...", flush=True)
    epoch_results, final_test_results, best_model_path, best_accuracy = train_network_jax(
        network, train_data, test_data,
        epochs=epochs, batch_size=batch_size, run_dir=run_dir,
    )
    model_save_path = os.path.join(run_dir, "shd_2comp_model_jax.pkl")
    network.save(model_save_path)
    print_summary_statistics(network, epoch_results, final_test_results, run_dir)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
