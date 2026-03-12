#!/usr/bin/env python3
"""
Run N-layer e-prop model on SHD. Number of layers is set by N_LAYERS (variable).
Architecture: input → L1 → L2 → ... → L_N_LAYERS → readout.
Layer sizes: each hidden layer has N_HIDDEN units (or use --layer_sizes for a custom list).
"""
import argparse
import os
import sys
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
jax.config.update("jax_enable_x64", True)

from data import load_shd_data, create_shd_input_jax

sys.path.insert(0, _SCRIPT_DIR)
from nlayer import JAXEPropNetworkNLayer, train_network_n_layer
from jax import random
import numpy as np

# -----------------------------------------------------------------------------
# HYPERPARAMETERS — set N_LAYERS and optionally layer sizes
# -----------------------------------------------------------------------------
T_SHD = 700
N_LAYERS = 2
N_HIDDEN = 40
N_OUTPUTS = 20
RANDOM_SEED = 12
EPOCHS = 3
BATCH_SIZE = 1
LR_DEND = 0.05
LR_SOMA = 0.0025
LR_READOUT = 0.025
WEIGHT_DECAY = 0.00001
GRADIENT_CLIP = 5.0
LOSS_TEMPERATURE = 5.0
LOSS_COUNT_BIAS = 0.1
LOSS_LABEL_SMOOTHING = 0.2
BETA_S = 0.36   # Somatic surrogate (super-spike); larger = gradient more concentrated near threshold
BETA_D = 0.75   # Dendritic surrogate
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Run N-layer e-prop on SHD")
    p.add_argument("--T", type=int, default=None, help=f"Time steps (default: {T_SHD})")
    p.add_argument("--n_layers", type=int, default=None, help=f"Number of 2comp layers (default: {N_LAYERS})")
    p.add_argument("--n_hidden", type=int, default=None, help=f"Neurons per layer (default: {N_HIDDEN})")
    p.add_argument("--layer_sizes", type=str, default=None,
                   help="Comma-separated layer sizes, e.g. 42,40,40 or '17, 14, 12' (no spaces, or quote the list)")
    p.add_argument("--n_outputs", type=int, default=None, help=f"Output classes (default: {N_OUTPUTS})")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr_dend", type=float, default=None)
    p.add_argument("--lr_soma", type=float, default=None)
    p.add_argument("--lr_readout", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--gradient_clip", type=float, default=None)
    p.add_argument("--loss_temperature", type=float, default=None)
    p.add_argument("--loss_count_bias", type=float, default=None)
    p.add_argument("--loss_label_smoothing", type=float, default=None)
    p.add_argument("--beta_s", type=float, default=None, help=f"Somatic surrogate beta (default: {BETA_S})")
    p.add_argument("--beta_d", type=float, default=None, help=f"Dendritic surrogate beta (default: {BETA_D})")
    args = p.parse_args()

    def _int(name, default):
        v = getattr(args, name)
        return v if v is not None else default

    def _float(name, default):
        v = getattr(args, name)
        return v if v is not None else default

    n_layers = _int("n_layers", N_LAYERS)
    n_hidden = _int("n_hidden", N_HIDDEN)
    if args.layer_sizes is not None:
        layer_sizes = [int(x.strip()) for x in args.layer_sizes.split(",")]
        n_layers = len(layer_sizes)
    else:
        layer_sizes = [n_hidden] * n_layers

    return {
        "T_SHD": _int("T", T_SHD),
        "N_LAYERS": n_layers,
        "LAYER_SIZES": layer_sizes,
        "N_OUTPUTS": _int("n_outputs", N_OUTPUTS),
        "RANDOM_SEED": _int("seed", RANDOM_SEED),
        "EPOCHS": _int("epochs", EPOCHS),
        "BATCH_SIZE": _int("batch_size", BATCH_SIZE),
        "LR_DEND": _float("lr_dend", LR_DEND),
        "LR_SOMA": _float("lr_soma", LR_SOMA),
        "LR_READOUT": _float("lr_readout", LR_READOUT),
        "WEIGHT_DECAY": _float("weight_decay", WEIGHT_DECAY),
        "GRADIENT_CLIP": _float("gradient_clip", GRADIENT_CLIP),
        "LOSS_TEMPERATURE": _float("loss_temperature", LOSS_TEMPERATURE),
        "LOSS_COUNT_BIAS": _float("loss_count_bias", LOSS_COUNT_BIAS),
        "LOSS_LABEL_SMOOTHING": _float("loss_label_smoothing", LOSS_LABEL_SMOOTHING),
        "BETA_S": _float("beta_s", BETA_S),
        "BETA_D": _float("beta_d", BETA_D),
    }


def evaluate(net: JAXEPropNetworkNLayer, test_data: list) -> float:
    correct = 0
    for x, label in test_data:
        _, readout_o = net.forward(x)
        pred = int(np.argmax(np.sum(np.asarray(readout_o), axis=0)))
        if pred == int(label):
            correct += 1
    return correct / len(test_data) if test_data else 0.0


def main():
    cfg = parse_args()
    T = cfg["T_SHD"]
    layer_sizes = cfg["LAYER_SIZES"]
    n_outputs = cfg["N_OUTPUTS"]
    seed = cfg["RANDOM_SEED"]
    epochs = cfg["EPOCHS"]
    batch_size = cfg["BATCH_SIZE"]

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

    print("Loading SHD data...", flush=True)
    train_raw, test_raw = load_shd_data(data_path, train_samples_per_class=None, test_samples_per_class=None)
    train_data = [(create_shd_input_jax(x, T=T), label) for x, label in train_raw]
    test_data = [(create_shd_input_jax(x, T=T), label) for x, label in test_raw]
    n_inputs = train_data[0][0].shape[1]

    network = JAXEPropNetworkNLayer(
        key,
        n_inputs=n_inputs,
        layer_sizes=layer_sizes,
        n_outputs=n_outputs,
        T=T,
        learning_rate_dendritic=cfg["LR_DEND"],
        learning_rate_somatic=cfg["LR_SOMA"],
        learning_rate_readout=cfg["LR_READOUT"],
        weight_decay=cfg["WEIGHT_DECAY"],
        gradient_clip=cfg["GRADIENT_CLIP"],
        loss_temperature=cfg["LOSS_TEMPERATURE"],
        loss_count_bias=cfg["LOSS_COUNT_BIAS"],
        loss_label_smoothing=cfg["LOSS_LABEL_SMOOTHING"],
        beta_s=cfg["BETA_S"],
        beta_d=cfg["BETA_D"],
    )

    print(f"N-layer SHD: L={cfg['N_LAYERS']} layers, sizes={layer_sizes}", flush=True)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}", flush=True)
    print(f"LR dend: {cfg['LR_DEND']}, soma: {cfg['LR_SOMA']}, readout: {cfg['LR_READOUT']}", flush=True)
    print("Training...", flush=True)

    train_inputs = np.stack([x for x, _ in train_data])
    train_targets = np.array([y for _, y in train_data])
    epoch_losses = train_network_n_layer(network, train_inputs, train_targets, epochs, batch_size)

    for ep, loss in enumerate(epoch_losses):
        acc = evaluate(network, test_data)
        print(f"Epoch {ep + 1}/{epochs}  loss={loss:.4f}  test_acc={acc:.4f}", flush=True)

    final_acc = evaluate(network, test_data)
    print(f"Done. Final test accuracy: {final_acc:.4f}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)

    with open(os.path.join(run_dir, "hyperparameters.txt"), "w") as f:
        f.write(f"n_layers={cfg['N_LAYERS']}\nlayer_sizes={layer_sizes}\nn_inputs={n_inputs}\nn_outputs={n_outputs}\n")
        f.write(f"T={T}\nepochs={epochs}\nbatch_size={batch_size}\nseed={seed}\n")


if __name__ == "__main__":
    main()
