#!/usr/bin/env python3
"""
Run the 2-layer model on SHD using the count-binning preprocessing from
data/shd_binned.py (Bittar & Garner / Fabre et al. 2025 style).

Differences from run_shd.py:
  - No alpha-kernel convolution. The network sees raw per-bin spike counts.
  - `bin_size_ms` and `collapse_factor` are first-class CLI knobs.
  - T (number of timesteps) is derived from `max_duration_ms / bin_size_ms`
    rather than being a user-specified constant.
  - `n_inputs` is derived from `700 / collapse_factor`.

Defaults reproduce the paper: bin_size_ms=4, collapse_factor=5, max_duration_ms=1400.
=> T = 350 timesteps, n_inputs = 140 channels, count-valued input tensor.
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

import jax
jax.config.update("jax_enable_x64", True)

from data.shd_binned import load_shd_binned

sys.path.insert(0, _SCRIPT_DIR)
from jax import random
import numpy as np


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
# Preprocessing (paper defaults)
BIN_SIZE_MS = 4.0
COLLAPSE_FACTOR = 5
MAX_DURATION_MS = 1400.0
BINARIZE = False
INPUT_SCALE = 1.0
# Architecture & data
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
SPIKE_DROPOUT = 0.0
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Run 2-layer model on SHD with count-bin preprocessing (sparch / Fabre et al. 2025 style)."
    )
    p.add_argument("--bin_size_ms", type=float, default=None,
                   help=f"Time bin width in ms (default: {BIN_SIZE_MS}; paper: 4, also tries 10, 14)")
    p.add_argument("--collapse_factor", type=int, default=None,
                   help=f"Channel collapsing: sum-pool every N input channels (default: {COLLAPSE_FACTOR}; paper: 5 -> 140 channels)")
    p.add_argument("--max_duration_ms", type=float, default=None,
                   help=f"Fixed window length in ms (default: {MAX_DURATION_MS})")
    p.add_argument("--binarize", action="store_true", default=False,
                   help="Cap per-bin counts to {0,1} instead of feeding actual counts")
    p.add_argument("--input_scale", type=float, default=None,
                   help=f"Multiplicative scale applied to the binned input (default: {INPUT_SCALE})")

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
    p.add_argument("--spike_dropout", type=float, default=None,
                   help="Train-time spike dropout 0--1 (default 0.0)")
    p.add_argument("--pkl", type=str, default=None,
                   help="Path to existing .pkl model to resume training (optional)")
    p.add_argument("--lowmemory", action="store_true",
                   help="Use 2layer_lowmemory.py (lower memory, may be slower)")

    args = p.parse_args()

    def _val(name, default):
        v = getattr(args, name)
        return v if v is not None else default

    return {
        "BIN_SIZE_MS": _val("bin_size_ms", BIN_SIZE_MS),
        "COLLAPSE_FACTOR": _val("collapse_factor", COLLAPSE_FACTOR),
        "MAX_DURATION_MS": _val("max_duration_ms", MAX_DURATION_MS),
        "BINARIZE": getattr(args, "binarize", False) or BINARIZE,
        "INPUT_SCALE": _val("input_scale", INPUT_SCALE),

        "LOWMEMORY": getattr(args, "lowmemory", False),
        "N_EXTRA": _val("n_extra", N_EXTRA),
        "N_HIDDEN": _val("n_hidden", N_HIDDEN),
        "N_OUTPUTS": _val("n_outputs", N_OUTPUTS),
        "RANDOM_SEED": _val("seed", RANDOM_SEED),
        "EPOCHS": _val("epochs", EPOCHS),
        "BATCH_SIZE": _val("batch_size", BATCH_SIZE),
        "LR_EXTRA_DEND": _val("lr_extra_dend", LR_EXTRA_DEND),
        "LR_EXTRA_SOMA": _val("lr_extra_soma", LR_EXTRA_SOMA),
        "LR_HIDDEN_DEND": _val("lr_hidden_dend", LR_HIDDEN_DEND),
        "LR_HIDDEN_SOMA": _val("lr_hidden_soma", LR_HIDDEN_SOMA),
        "LR_READOUT": _val("lr_readout", LR_READOUT),
        "WEIGHT_DECAY": _val("weight_decay", WEIGHT_DECAY),
        "GRADIENT_CLIP": _val("gradient_clip", GRADIENT_CLIP),
        "LOSS_TEMPERATURE": _val("loss_temperature", LOSS_TEMPERATURE),
        "LOSS_COUNT_BIAS": _val("loss_count_bias", LOSS_COUNT_BIAS),
        "LOSS_LABEL_SMOOTHING": _val("loss_label_smoothing", LOSS_LABEL_SMOOTHING),
        "SPIKE_DROPOUT": _val("spike_dropout", SPIKE_DROPOUT),
        "PKL_PATH": getattr(args, "pkl", None),
    }


def _resolve_data_path():
    if "SHD_DATA_PATH" in os.environ:
        return os.environ["SHD_DATA_PATH"]
    if os.path.exists("/share/neurocomputation/Tim/SHD_data"):
        return "/share/neurocomputation/Tim/SHD_data"
    return os.path.expanduser("~/Documents/Heidelberg_Data")


def main():
    cfg = parse_args()
    seed = cfg["RANDOM_SEED"]
    epochs = cfg["EPOCHS"]
    batch_size = cfg["BATCH_SIZE"]

    bin_size_ms = float(cfg["BIN_SIZE_MS"])
    collapse_factor = int(cfg["COLLAPSE_FACTOR"])
    max_duration_ms = float(cfg["MAX_DURATION_MS"])

    T = max(1, int(np.ceil(max_duration_ms / bin_size_ms)))
    n_inputs = (700 + collapse_factor - 1) // collapse_factor
    n_extra = cfg["N_EXTRA"]
    n_hidden = cfg["N_HIDDEN"]
    n_outputs = cfg["N_OUTPUTS"]

    key = random.PRNGKey(seed)
    np.random.seed(seed)

    data_path = _resolve_data_path()
    run_dir = os.path.join(_SCRIPT_DIR, "model_binned",
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)

    JAXEPropNetworkTwoLayer, train_network_two_layer = _load_twolayer_module(cfg["LOWMEMORY"])
    if cfg["LOWMEMORY"]:
        print("Using 2layer_lowmemory.py", flush=True)

    print("Loading SHD data with count-bin preprocessing...", flush=True)
    print(
        f"  bin_size_ms={bin_size_ms}, collapse_factor={collapse_factor}, "
        f"max_duration_ms={max_duration_ms}  =>  T={T}, n_inputs={n_inputs}, "
        f"binarize={cfg['BINARIZE']}, input_scale={cfg['INPUT_SCALE']}",
        flush=True,
    )
    X_tr, y_tr, _, X_te, y_te, _ = load_shd_binned(
        bin_size_ms=bin_size_ms,
        collapse_factor=collapse_factor,
        max_duration_ms=max_duration_ms,
        binarize=cfg["BINARIZE"],
        dtype=np.float64,
        data_path=data_path,
    )

    if cfg["INPUT_SCALE"] != 1.0:
        X_tr = X_tr * cfg["INPUT_SCALE"]
        X_te = X_te * cfg["INPUT_SCALE"]

    assert X_tr.shape[1] == T and X_tr.shape[2] == n_inputs, (
        f"shape mismatch: got {X_tr.shape}, expected (?, {T}, {n_inputs})"
    )

    train_data = [(X_tr[i], int(y_tr[i])) for i in range(len(y_tr))]
    test_data = [(X_te[i], int(y_te[i])) for i in range(len(y_te))]

    pkl_path = cfg.get("PKL_PATH")
    if pkl_path and os.path.isfile(pkl_path):
        print(f"Loading existing model from {pkl_path} (resume training)...", flush=True)
        network = JAXEPropNetworkTwoLayer.load(pkl_path, key=key)
        if network.T != T or network.n_inputs != n_inputs:
            raise ValueError(
                f"Resume failed: loaded model has T={network.T}, n_inputs={network.n_inputs}; "
                f"data has T={T}, n_inputs={n_inputs}. Use the same bin_size_ms/collapse_factor "
                f"as when the pkl was saved."
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
        print("  Applied LRs from config.", flush=True)
    else:
        if pkl_path:
            print(f"Warning: --pkl {pkl_path} not found or not a file; initializing new weights.",
                  flush=True)
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
        "", "2-layer SHD run (binned preprocessing)", "=" * 80,
        f"Random seed: {seed}",
        f"Epochs: {epochs}, Batch size: {batch_size}",
        f"Preprocessing: bin={bin_size_ms}ms, collapse={collapse_factor} -> 700/{collapse_factor}={n_inputs} ch, "
        f"window={max_duration_ms}ms -> T={T}, binarize={cfg['BINARIZE']}, scale={cfg['INPUT_SCALE']}",
        f"LR extra dend: {cfg['LR_EXTRA_DEND']}, extra soma: {cfg['LR_EXTRA_SOMA']}",
        f"LR hidden dend: {cfg['LR_HIDDEN_DEND']}, hidden soma: {cfg['LR_HIDDEN_SOMA']}, readout: {cfg['LR_READOUT']}",
        f"Loss: temp={cfg['LOSS_TEMPERATURE']}, bias={cfg['LOSS_COUNT_BIAS']}, smoothing={cfg['LOSS_LABEL_SMOOTHING']}",
        f"Spike dropout: {cfg['SPIKE_DROPOUT']}",
        "=" * 80, "",
    ]
    for line in hyperparams_lines:
        print(line, flush=True)
    print("Training...", flush=True)
    train_network_two_layer(
        network, train_data, test_data, run_dir, epochs, batch_size,
        "shd_two_layer_binned", seed,
        spike_dropout_prob=cfg["SPIKE_DROPOUT"],
    )
    print(f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
