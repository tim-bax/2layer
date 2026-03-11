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

# Standalone model: model.py (default) or model_lowmemory.py (--lowmemory)
sys.path.insert(0, _SCRIPT_DIR)
import importlib.util

def _load_onelayer_module(lowmemory: bool):
    """Load model.py or model_lowmemory.py; same API (JAXEPropNetwork, train_network_jax, etc.)."""
    basename = "model_lowmemory.py" if lowmemory else "model.py"
    path = os.path.join(_SCRIPT_DIR, basename)
    spec = importlib.util.spec_from_file_location("onelayer_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return (
        mod.JAXEPropNetwork,
        mod.train_network_jax,
        mod.print_summary_statistics,
        mod.initialize_numpy_weights,
        mod.NeuronConfig,
    )

from jax import random
import numpy as np

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
# Readout-only warmup: first N epochs only readout updates (hidden frozen); 0 = off
WARMUP_READOUT_EPOCHS = 1
# Learning rates (hidden dendritic, hidden somatic, readout)
LR_HIDDEN_DEND = 0.045
LR_HIDDEN_SOMA = 0.00015
LR_READOUT = 0.035
# Scale dendritic gradient (1.0 = no change; >1 boosts dendrite updates to balance with soma)
GRAD_DEND_SCALE = 1.0
# Regularization & training
WEIGHT_DECAY = 0.00001
GRADIENT_CLIP = 5.0
# Spike dropout: fraction of non-zero input bins to zero out at train time (0 = off)
SPIKE_DROPOUT = 0.1
# Loss (softmax temperature, count bias, label smoothing)
LOSS_TEMPERATURE = 2.7
LOSS_COUNT_BIAS = 0.18
LOSS_LABEL_SMOOTHING = 0.13
# Surrogate gradient: larger beta = gradient more concentrated near spike (defaults 0.36 somatic, 0.75 dend)
BETA_S = 1.0
BETA_D = 1.5
# Weight init: hidden layer (dend/soma) and optionally readout (larger readout = non-zero initial output, larger effective_error early)
WEIGHT_SCALE = 0.25
READOUT_WEIGHT_SCALE = None  # None = use WEIGHT_SCALE; set e.g. 0.4 or 0.5 so readout has variance from the start
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Run 1-layer 2comp model on SHD")
    p.add_argument("--T", type=int, default=None, help=f"Time steps (default: {T_SHD})")
    p.add_argument("--n_hidden", type=int, default=None, help=f"Hidden units (default: {N_HIDDEN})")
    p.add_argument("--n_outputs", type=int, default=None, help=f"Output classes (default: {N_OUTPUTS})")
    p.add_argument("--seed", type=int, default=None, help=f"Random seed (default: {RANDOM_SEED})")
    p.add_argument("--epochs", type=int, default=None, help=f"Epochs (default: {EPOCHS})")
    p.add_argument("--batch_size", type=int, default=None, help=f"Batch size (default: {BATCH_SIZE})")
    p.add_argument("--warmup_readout_epochs", type=int, default=None, help="First N epochs only readout (0=off)")
    p.add_argument("--lr_hidden_dend", type=float, default=None)
    p.add_argument("--lr_hidden_soma", type=float, default=None)
    p.add_argument("--lr_readout", type=float, default=None)
    p.add_argument("--grad_dend_scale", type=float, default=None, help="Scale factor for dendritic gradient (default 1.0)")
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--gradient_clip", type=float, default=None)
    p.add_argument("--spike_dropout", type=float, default=None, help="Train-time spike dropout rate 0--1 (default 0.1)")
    p.add_argument("--loss_temperature", type=float, default=None)
    p.add_argument("--loss_count_bias", type=float, default=None)
    p.add_argument("--loss_label_smoothing", type=float, default=None)
    p.add_argument("--beta_s", type=float, default=None, help="Somatic surrogate beta (larger = more spike-near)")
    p.add_argument("--beta_d", type=float, default=None, help="Dendritic surrogate beta")
    p.add_argument("--weight_scale", type=float, default=None, help="Weight init scale for hidden (e.g. 0.25--0.4)")
    p.add_argument("--readout_weight_scale", type=float, default=None, help="Weight init for readout only (default: same as weight_scale; use 0.4--0.5 for non-zero initial readout)")
    p.add_argument("--pkl", type=str, default=None, help="Path to existing .pkl model to resume training (optional)")
    p.add_argument("--lowmemory", action="store_true", help="Use model_lowmemory.py (lower memory, may be slower)")
    args = p.parse_args()
    def _int(name, default): v = getattr(args, name); return v if v is not None else default
    def _float(name, default): v = getattr(args, name); return v if v is not None else default
    return {
        "LOWMEMORY": getattr(args, "lowmemory", False),
        "T_SHD": _int("T", T_SHD),
        "N_HIDDEN": _int("n_hidden", N_HIDDEN),
        "N_OUTPUTS": _int("n_outputs", N_OUTPUTS),
        "RANDOM_SEED": _int("seed", RANDOM_SEED),
        "EPOCHS": _int("epochs", EPOCHS),
        "BATCH_SIZE": _int("batch_size", BATCH_SIZE),
        "WARMUP_READOUT_EPOCHS": _int("warmup_readout_epochs", WARMUP_READOUT_EPOCHS),
        "LR_HIDDEN_DEND": _float("lr_hidden_dend", LR_HIDDEN_DEND),
        "LR_HIDDEN_SOMA": _float("lr_hidden_soma", LR_HIDDEN_SOMA),
        "LR_READOUT": _float("lr_readout", LR_READOUT),
        "GRAD_DEND_SCALE": _float("grad_dend_scale", GRAD_DEND_SCALE),
        "WEIGHT_DECAY": _float("weight_decay", WEIGHT_DECAY),
        "GRADIENT_CLIP": _float("gradient_clip", GRADIENT_CLIP),
        "SPIKE_DROPOUT": _float("spike_dropout", SPIKE_DROPOUT),
        "LOSS_TEMPERATURE": _float("loss_temperature", LOSS_TEMPERATURE),
        "LOSS_COUNT_BIAS": _float("loss_count_bias", LOSS_COUNT_BIAS),
        "LOSS_LABEL_SMOOTHING": _float("loss_label_smoothing", LOSS_LABEL_SMOOTHING),
        "BETA_S": _float("beta_s", BETA_S),
        "BETA_D": _float("beta_d", BETA_D),
        "WEIGHT_SCALE": _float("weight_scale", WEIGHT_SCALE),
        "READOUT_WEIGHT_SCALE": _float("readout_weight_scale", READOUT_WEIGHT_SCALE),
        "PKL_PATH": getattr(args, "pkl", None),
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

    JAXEPropNetwork, train_network_jax, print_summary_statistics, initialize_numpy_weights, NeuronConfig = _load_onelayer_module(cfg["LOWMEMORY"])
    if cfg["LOWMEMORY"]:
        print("Using model_lowmemory.py", flush=True)

    print("Loading SHD data...", flush=True)
    train_raw, test_raw = load_shd_data(
        data_path,
        train_samples_per_class=None,
        test_samples_per_class=None,
    )
    print(f"Converting to (T={T}, n_inputs) format...", flush=True)
    train_data = [
        (create_shd_input_jax(x, T=T), label) for x, label in train_raw
    ]
    test_data = [
        (create_shd_input_jax(x, T=T), label) for x, label in test_raw
    ]
    n_inputs = train_data[0][0].shape[1]

    print(f"Train: {len(train_data)}, Test: {len(test_data)}, n_inputs: {n_inputs}", flush=True)

    pkl_path = cfg.get("PKL_PATH")
    if pkl_path and os.path.isfile(pkl_path):
        print(f"Loading existing model from {pkl_path} (resume training)...", flush=True)
        network = JAXEPropNetwork.load(pkl_path, key=key)
        if network.T != T or network.n_inputs != n_inputs:
            raise ValueError(
                f"Resume failed: loaded model has T={network.T}, n_inputs={network.n_inputs}; "
                f"data has T={T}, n_inputs={n_inputs}. Use same T (and data) as when the pkl was saved."
            )
        n_hidden, n_outputs = network.n_hidden, network.n_outputs
        network.learning_rate_hidden_dendritic = cfg["LR_HIDDEN_DEND"]
        network.learning_rate_hidden_somatic = cfg["LR_HIDDEN_SOMA"]
        network.learning_rate_readout = cfg["LR_READOUT"]
        network.grad_dend_scale = cfg["GRAD_DEND_SCALE"]
        network.weight_decay = cfg["WEIGHT_DECAY"]
        network.gradient_clip = cfg["GRADIENT_CLIP"]
        network.loss_temperature = cfg["LOSS_TEMPERATURE"]
        network.loss_count_bias = cfg["LOSS_COUNT_BIAS"]
        network.loss_label_smoothing = cfg["LOSS_LABEL_SMOOTHING"]
        print(f"  Applied LRs: dend={cfg['LR_HIDDEN_DEND']}, soma={cfg['LR_HIDDEN_SOMA']}, readout={cfg['LR_READOUT']}", flush=True)
    else:
        if pkl_path:
            print(f"Warning: --pkl {pkl_path} not found or not a file; initializing new weights.", flush=True)
        print(f"Creating network and initializing weights...", flush=True)
        np.random.seed(seed)
        weight_scale = cfg["WEIGHT_SCALE"]
        readout_scale = cfg["READOUT_WEIGHT_SCALE"]
        try:
            w_dend_np, w_soma_np, w_readout_np = initialize_numpy_weights(
                n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs,
                weight_scale=weight_scale,
                readout_weight_scale=readout_scale,
            )
        except TypeError as e:
            if "readout_weight_scale" in str(e) or "unexpected keyword" in str(e).lower():
                w_dend_np, w_soma_np, w_readout_np = initialize_numpy_weights(
                    n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs,
                    weight_scale=weight_scale,
                )
                if readout_scale is not None:
                    print("Note: readout_weight_scale ignored (layer module does not support it).", flush=True)
            else:
                raise
        network = JAXEPropNetwork(
            key, n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs, T=T,
            learning_rate_hidden_dendritic=cfg["LR_HIDDEN_DEND"],
            learning_rate_hidden_somatic=cfg["LR_HIDDEN_SOMA"],
            learning_rate_readout=cfg["LR_READOUT"],
            grad_dend_scale=cfg["GRAD_DEND_SCALE"],
            weight_decay=cfg["WEIGHT_DECAY"],
            gradient_clip=cfg["GRADIENT_CLIP"],
            loss_temperature=cfg["LOSS_TEMPERATURE"],
            loss_count_bias=cfg["LOSS_COUNT_BIAS"],
            loss_label_smoothing=cfg["LOSS_LABEL_SMOOTHING"],
            beta_s=cfg["BETA_S"],
            beta_d=cfg["BETA_D"],
            weight_scale=weight_scale,
        )
        network.hidden_layer.w_dend = jax.numpy.array(w_dend_np)
        network.hidden_layer.w_soma = jax.numpy.array(w_soma_np)
        network.readout_layer.w = jax.numpy.array(w_readout_np)

    hyperparams_lines = [
        "", "1-layer SHD run", "=" * 80,
        f"Random seed: {seed}",
        f"Epochs: {epochs}, Batch size: {batch_size}",
        f"Warmup readout epochs: {cfg['WARMUP_READOUT_EPOCHS']}",
        f"grad_dend_scale: {cfg['GRAD_DEND_SCALE']}, spike_dropout: {cfg['SPIKE_DROPOUT']}",
        f"LR hidden dend: {cfg['LR_HIDDEN_DEND']}, soma: {cfg['LR_HIDDEN_SOMA']}, readout: {cfg['LR_READOUT']}",
        f"Loss: temp={cfg['LOSS_TEMPERATURE']}, bias={cfg['LOSS_COUNT_BIAS']}, smoothing={cfg['LOSS_LABEL_SMOOTHING']}",
        f"beta_s: {cfg['BETA_S']}, beta_d: {cfg['BETA_D']}, weight_scale: {cfg['WEIGHT_SCALE']}, readout_weight_scale: {cfg['READOUT_WEIGHT_SCALE']}",
        "=" * 80, "",
    ]
    with open(os.path.join(run_dir, "hyperparameters.txt"), "w") as f:
        f.write("\n".join(hyperparams_lines))

    print("Training...", flush=True)
    epoch_results, final_test_results, best_model_path, best_accuracy = train_network_jax(
        network, train_data, test_data,
        epochs=epochs, batch_size=batch_size, run_dir=run_dir,
        warmup_readout_epochs=cfg["WARMUP_READOUT_EPOCHS"],
        spike_dropout_prob=cfg["SPIKE_DROPOUT"],
    )
    model_save_path = os.path.join(run_dir, "shd_2comp_model_jax.pkl")
    network.save(model_save_path)
    print_summary_statistics(network, epoch_results, final_test_results, run_dir)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
