#!/usr/bin/env python3
"""
Run 2-layer model on SHD. Loads SHD data, converts to (T, n_inputs), trains the standalone model.
Edit the HYPERPARAMETERS block below or pass CLI args to change settings.
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

from data import load_shd_data, create_shd_input_jax

sys.path.insert(0, _SCRIPT_DIR)
from jax import random
import numpy as np

# Load 2layer network: 2layer.py (default) or 2layer_lowmemory.py (--lowmemory)
def _load_twolayer_module(lowmemory: bool):
    basename = "2layer_lowmemory.py" if lowmemory else "2layer.py"
    path = os.path.join(_SCRIPT_DIR, basename)
    spec = importlib.util.spec_from_file_location("twolayer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.JAXEPropNetworkTwoLayer, mod.train_network_two_layer

# -----------------------------------------------------------------------------
# HYPERPARAMETERS — edit these or override via command line
# -----------------------------------------------------------------------------
# Architecture & data
T_SHD = 700
N_EXTRA = 42
N_HIDDEN = 40
N_OUTPUTS = 20
RANDOM_SEED = 12
EPOCHS = 3
BATCH_SIZE = 1
# Learning rates (extra layer, hidden layer, readout)
LR_EXTRA_DEND = 0.05
LR_EXTRA_SOMA = 0.0025
LR_HIDDEN_DEND = 0.05
LR_HIDDEN_SOMA = 0.0025
LR_READOUT = 0.025
# Regularization & training
WEIGHT_DECAY = 0.00001
GRADIENT_CLIP = 5.0
# Loss (softmax temperature, count bias, label smoothing)
LOSS_TEMPERATURE = 5.0
LOSS_COUNT_BIAS = 0.1
LOSS_LABEL_SMOOTHING = 0.2
# Spike dropout at train time (0 = off)
SPIKE_DROPOUT = 0.1
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Run 2-layer model on SHD")
    p.add_argument("--T", type=int, default=None, help=f"Time steps (default: {T_SHD})")
    p.add_argument("--n_extra", type=int, default=None, help=f"Extra 2comp units (default: {N_EXTRA})")
    p.add_argument("--n_hidden", type=int, default=None, help=f"Hidden units (default: {N_HIDDEN})")
    p.add_argument("--n_outputs", type=int, default=None, help=f"Output classes (default: {N_OUTPUTS})")
    p.add_argument("--seed", type=int, default=None, help=f"Random seed (default: {RANDOM_SEED})")
    p.add_argument("--epochs", type=int, default=None, help=f"Epochs (default: {EPOCHS})")
    p.add_argument("--batch_size", type=int, default=None, help=f"Batch size (default: {BATCH_SIZE})")
    p.add_argument("--lr_extra_dend", type=float, default=None)
    p.add_argument("--lr_extra_soma", type=float, default=None)
    p.add_argument("--lr_hidden_dend", type=float, default=None)
    p.add_argument("--lr_hidden_soma", type=float, default=None)
    p.add_argument("--lr_readout", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--gradient_clip", type=float, default=None)
    p.add_argument("--loss_temperature", type=float, default=None)
    p.add_argument("--loss_count_bias", type=float, default=None)
    p.add_argument("--loss_label_smoothing", type=float, default=None)
    p.add_argument("--spike_dropout", type=float, default=None, help="Train-time spike dropout 0--1 (default 0.1)")
    p.add_argument("--pkl", type=str, default=None, help="Path to existing .pkl model to resume training (optional)")
    p.add_argument("--lowmemory", action="store_true", help="Use 2layer_lowmemory.py (lower memory, may be slower)")
    p.add_argument("--no_kernel", action="store_true", help="Input: use_kernel=False (no alpha kernel; instantaneous spike_amplitude per bin)")
    p.add_argument("--spike_amplitude", type=float, default=None, help="Spike amplitude when --no_kernel (default: 5.0)")
    args = p.parse_args()
    def _int(name, default): v = getattr(args, name); return v if v is not None else default
    def _float(name, default): v = getattr(args, name); return v if v is not None else default
    return {
        "LOWMEMORY": getattr(args, "lowmemory", False),
        "T_SHD": _int("T", T_SHD),
        "N_EXTRA": _int("n_extra", N_EXTRA),
        "N_HIDDEN": _int("n_hidden", N_HIDDEN),
        "N_OUTPUTS": _int("n_outputs", N_OUTPUTS),
        "RANDOM_SEED": _int("seed", RANDOM_SEED),
        "EPOCHS": _int("epochs", EPOCHS),
        "BATCH_SIZE": _int("batch_size", BATCH_SIZE),
        "LR_EXTRA_DEND": _float("lr_extra_dend", LR_EXTRA_DEND),
        "LR_EXTRA_SOMA": _float("lr_extra_soma", LR_EXTRA_SOMA),
        "LR_HIDDEN_DEND": _float("lr_hidden_dend", LR_HIDDEN_DEND),
        "LR_HIDDEN_SOMA": _float("lr_hidden_soma", LR_HIDDEN_SOMA),
        "LR_READOUT": _float("lr_readout", LR_READOUT),
        "WEIGHT_DECAY": _float("weight_decay", WEIGHT_DECAY),
        "GRADIENT_CLIP": _float("gradient_clip", GRADIENT_CLIP),
        "LOSS_TEMPERATURE": _float("loss_temperature", LOSS_TEMPERATURE),
        "LOSS_COUNT_BIAS": _float("loss_count_bias", LOSS_COUNT_BIAS),
        "LOSS_LABEL_SMOOTHING": _float("loss_label_smoothing", LOSS_LABEL_SMOOTHING),
        "SPIKE_DROPOUT": _float("spike_dropout", SPIKE_DROPOUT),
        "PKL_PATH": getattr(args, "pkl", None),
        "NO_KERNEL": getattr(args, "no_kernel", False),
        "SPIKE_AMPLITUDE": getattr(args, "spike_amplitude", None),
    }


def main():
    cfg = parse_args()
    T, n_extra, n_hidden, n_outputs = cfg["T_SHD"], cfg["N_EXTRA"], cfg["N_HIDDEN"], cfg["N_OUTPUTS"]
    seed, epochs, batch_size = cfg["RANDOM_SEED"], cfg["EPOCHS"], cfg["BATCH_SIZE"]

    key = random.PRNGKey(seed)
    np.random.seed(seed)

    if "SHD_DATA_PATH" in os.environ:
        data_path = os.environ["SHD_DATA_PATH"]
    elif os.path.exists("/share/neurocomputation/Tim/SHD_data"):
        data_path = "/share/neurocomputation/Tim/SHD_data"
    else:
        data_path = os.path.expanduser("~/Documents/Heidelberg_Data")

    run_dir = os.path.join(_SCRIPT_DIR, "model", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)

    JAXEPropNetworkTwoLayer, train_network_two_layer = _load_twolayer_module(cfg["LOWMEMORY"])
    if cfg["LOWMEMORY"]:
        print("Using 2layer_lowmemory.py", flush=True)

    print("Loading SHD data...", flush=True)
    train_raw, test_raw = load_shd_data(data_path, train_samples_per_class=None, test_samples_per_class=None)
    input_kw = {"T": T}
    if cfg.get("NO_KERNEL"):
        input_kw["use_kernel"] = False
        if cfg.get("SPIKE_AMPLITUDE") is not None:
            input_kw["spike_amplitude"] = cfg["SPIKE_AMPLITUDE"]
        print("Using input with use_kernel=False", flush=True)
    train_data = [(create_shd_input_jax(x, **input_kw), label) for x, label in train_raw]
    test_data = [(create_shd_input_jax(x, **input_kw), label) for x, label in test_raw]
    n_inputs = train_data[0][0].shape[1]

    pkl_path = cfg.get("PKL_PATH")
    if pkl_path and os.path.isfile(pkl_path):
        print(f"Loading existing model from {pkl_path} (resume training)...", flush=True)
        network = JAXEPropNetworkTwoLayer.load(pkl_path, key=key)
        if network.T != T or network.n_inputs != n_inputs:
            raise ValueError(
                f"Resume failed: loaded model has T={network.T}, n_inputs={network.n_inputs}; "
                f"data has T={T}, n_inputs={n_inputs}. Use same T (and data) as when the pkl was saved."
            )
        n_extra, n_hidden, n_outputs = network.n_extra, network.n_hidden, network.n_outputs
        network.learning_rate_extra_dendritic = cfg["LR_EXTRA_DEND"]
        network.learning_rate_extra_soma = cfg["LR_EXTRA_SOMA"]
        network.learning_rate_hidden_dendritic = cfg["LR_HIDDEN_DEND"]
        network.learning_rate_hidden_somatic = cfg["LR_HIDDEN_SOMA"]
        network.learning_rate_readout = cfg["LR_READOUT"]
        network.weight_decay = cfg["WEIGHT_DECAY"]
        network.gradient_clip = cfg["GRADIENT_CLIP"]
        network.loss_temperature = cfg["LOSS_TEMPERATURE"]
        network.loss_count_bias = cfg["LOSS_COUNT_BIAS"]
        network.loss_label_smoothing = cfg["LOSS_LABEL_SMOOTHING"]
        print(f"  Applied LRs from config.", flush=True)
    else:
        if pkl_path:
            print(f"Warning: --pkl {pkl_path} not found or not a file; initializing new weights.", flush=True)
        network = JAXEPropNetworkTwoLayer(
            key, n_inputs=n_inputs, n_extra=n_extra, n_hidden=n_hidden, n_outputs=n_outputs, T=T,
            learning_rate_extra_dendritic=cfg["LR_EXTRA_DEND"],
            learning_rate_extra_soma=cfg["LR_EXTRA_SOMA"],
            learning_rate_hidden_dendritic=cfg["LR_HIDDEN_DEND"],
            learning_rate_hidden_somatic=cfg["LR_HIDDEN_SOMA"],
            learning_rate_readout=cfg["LR_READOUT"],
            weight_decay=cfg["WEIGHT_DECAY"],
            gradient_clip=cfg["GRADIENT_CLIP"],
            loss_temperature=cfg["LOSS_TEMPERATURE"],
            loss_count_bias=cfg["LOSS_COUNT_BIAS"],
            loss_label_smoothing=cfg["LOSS_LABEL_SMOOTHING"],
        )
    print(f"Train: {len(train_data)}, Test: {len(test_data)}", flush=True)
    hyperparams_lines = [
        "", "2-layer SHD run", "=" * 80,
        f"Random seed: {seed}",
        f"Epochs: {epochs}, Batch size: {batch_size}",
        f"Input: use_kernel={not cfg.get('NO_KERNEL', False)}" + (f", spike_amplitude={cfg.get('SPIKE_AMPLITUDE')}" if cfg.get("SPIKE_AMPLITUDE") is not None else ""),
        f"LR extra dend: {cfg['LR_EXTRA_DEND']}, extra soma: {cfg['LR_EXTRA_SOMA']}",
        f"LR hidden dend: {cfg['LR_HIDDEN_DEND']}, hidden soma: {cfg['LR_HIDDEN_SOMA']}, readout: {cfg['LR_READOUT']}",
        f"Loss: temp={cfg['LOSS_TEMPERATURE']}, bias={cfg['LOSS_COUNT_BIAS']}, smoothing={cfg['LOSS_LABEL_SMOOTHING']}",
        f"Spike dropout: {cfg['SPIKE_DROPOUT']}",
        "=" * 80, "",
    ]
    for line in hyperparams_lines:
        print(line, flush=True)
    print("Training...", flush=True)
    train_network_two_layer(network, train_data, test_data, run_dir, epochs, batch_size, "shd_two_layer", seed, spike_dropout_prob=cfg["SPIKE_DROPOUT"])
    print(f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
