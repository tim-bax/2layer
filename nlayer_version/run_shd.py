#!/usr/bin/env python3
"""
Run N-layer e-prop model on SHD. Number of layers is set by N_LAYERS (variable).
Architecture: input → L1 → L2 → ... → L_N_LAYERS → readout.
Layer sizes: each hidden layer has N_HIDDEN units (or use --layer_sizes for a custom list).
"""
import argparse
import json
import os
import sys
import time
import pickle
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
jax.config.update("jax_enable_x64", True)

from data import load_shd_data, create_shd_input_jax

sys.path.insert(0, _SCRIPT_DIR)
from nlayer import JAXEPropNetworkNLayer
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


def predict(net: JAXEPropNetworkNLayer, x) -> int:
    """Predicted class for one input (T, n_inputs)."""
    _, readout_o = net.forward(x)
    counts = np.sum(np.asarray(readout_o), axis=0)
    scaled = counts / net.loss_temperature + net.loss_count_bias
    probs = np.exp(scaled - np.max(scaled)) / np.sum(np.exp(scaled - np.max(scaled)))
    return int(np.argmax(probs))


def evaluate_full(net: JAXEPropNetworkNLayer, data: list) -> dict:
    """Full evaluation: accuracy %, avg loss, correct, total, confusion matrix, per-class accuracy (like 2layer)."""
    correct = total = 0
    test_losses = []
    n_out = net.n_outputs
    confusion = np.zeros((n_out, n_out))
    params = net.get_params()
    for x_raw, target in data:
        x = np.asarray(x_raw) if not hasattr(x_raw, "shape") else x_raw
        target = int(target)
        loss_val = float(net._loss_impl(params, x, target))
        test_losses.append(loss_val)
        pred = predict(net, x)
        if pred == target:
            correct += 1
        total += 1
        confusion[target, pred] += 1
    acc = correct / total * 100 if total else 0.0
    avg_loss = float(np.mean(test_losses)) if test_losses else 0.0
    per_class = {}
    for i in range(n_out):
        ct = np.sum(confusion[i, :])
        if ct > 0:
            per_class[i] = confusion[i, i] / ct * 100
    return {
        "accuracy": acc,
        "avg_loss": avg_loss,
        "correct": correct,
        "total": total,
        "confusion_matrix": confusion,
        "per_class_accuracy": per_class,
    }


def save_model(net: JAXEPropNetworkNLayer, path: str, cfg: dict, n_inputs: int):
    """Save network weights and config to a pickle file (for later load)."""
    params = net.get_params()
    dend, soma, w_readout = params
    state = {
        "n_inputs": n_inputs,
        "layer_sizes": list(net.layer_sizes),
        "n_outputs": net.n_outputs,
        "T": net.T,
        "w_dend": [np.asarray(w) for w in dend],
        "w_soma": [np.asarray(w) for w in soma],
        "w_readout": np.asarray(w_readout),
        "cfg": cfg,
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)


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

    # --- Print hyperparameters ---
    hp_lines = [
        "",
        "N-layer SHD hyperparameters",
        "=" * 60,
        f"  Architecture:  n_inputs={n_inputs}, layer_sizes={layer_sizes}, n_outputs={n_outputs}",
        f"  T={T},  epochs={epochs},  batch_size={batch_size},  seed={seed}",
        f"  LR_dend={cfg['LR_DEND']},  LR_soma={cfg['LR_SOMA']},  LR_readout={cfg['LR_READOUT']}",
        f"  weight_decay={cfg['WEIGHT_DECAY']},  gradient_clip={cfg['GRADIENT_CLIP']}",
        f"  loss_temperature={cfg['LOSS_TEMPERATURE']},  loss_count_bias={cfg['LOSS_COUNT_BIAS']},  loss_label_smoothing={cfg['LOSS_LABEL_SMOOTHING']}",
        f"  beta_s={cfg['BETA_S']},  beta_d={cfg['BETA_D']}",
        f"  Train samples: {len(train_data)},  Test samples: {len(test_data)}",
        "=" * 60,
        "",
    ]
    print("\n".join(hp_lines), flush=True)

    # --- Write hyperparameters file ---
    with open(os.path.join(run_dir, "hyperparameters.txt"), "w") as f:
        f.write("\n".join(hp_lines))
        f.write("\n".join([f"{k}={v}" for k, v in cfg.items()]))

    # --- Training loop (2layer-style: samples/s, ETA, per-epoch summary, confusion at end) ---
    train_inputs = np.stack([x for x, _ in train_data])
    train_targets = np.array([y for _, y in train_data])
    n_train = train_inputs.shape[0]
    n_batches_per_epoch = (n_train + batch_size - 1) // batch_size

    best_accuracy = -1.0
    best_model_path = None
    epoch_results = []
    model_path = os.path.join(run_dir, "nlayer_model.pkl")
    model_best_path = os.path.join(run_dir, "nlayer_model_best.pkl")

    for epoch in range(epochs):
        np.random.seed(seed + epoch)
        perm = np.random.permutation(n_train)
        epoch_start = time.time()
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0

        for batch_idx in range(n_batches_per_epoch):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            idx = perm[start_idx:end_idx]
            x_batch = train_inputs[idx]
            y_batch = train_targets[idx]

            for b in range(x_batch.shape[0]):
                loss, _ = network.train_step(x_batch[b], int(y_batch[b]))
                epoch_losses.append(loss)
                pred = predict(network, x_batch[b])
                if pred == int(y_batch[b]):
                    epoch_correct += 1
                epoch_total += 1

            # Progress: every 5000 samples, at start, or at end of epoch (like 2layer)
            samples_done = end_idx
            if samples_done % 5000 == 0 or samples_done == 1 or end_idx == n_train:
                elapsed = time.time() - epoch_start
                rate = samples_done / elapsed if elapsed > 0 else 0
                eta = (n_train - samples_done) / rate if rate > 0 else 0
                print(f"  Epoch {epoch+1}: {samples_done}/{n_train} samples ({100*samples_done/n_train:.0f}%) | "
                      f"{rate:.1f} samples/s | ETA this epoch: {eta:.0f}s", flush=True)

        epoch_avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        epoch_accuracy = epoch_correct / epoch_total * 100 if epoch_total else 0.0
        res = evaluate_full(network, test_data)
        epoch_results.append({
            "epoch": epoch + 1,
            "train_avg_loss": epoch_avg_loss,
            "train_accuracy": epoch_accuracy,
            "train_correct": epoch_correct,
            "train_total": epoch_total,
            "test_accuracy": res["accuracy"],
            "test_avg_loss": res["avg_loss"],
            "correct": res["correct"],
            "total": res["total"],
        })

        print(f"\n{'='*80}", flush=True)
        print(f"EPOCH {epoch+1}/{epochs} SUMMARY", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Train - Average Loss: {epoch_avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}% ({epoch_correct}/{epoch_total})", flush=True)
        print(f"Test - Average Loss: {res['avg_loss']:.4f}, Accuracy: {res['accuracy']:.2f}% ({res['correct']}/{res['total']})", flush=True)
        if res["accuracy"] > best_accuracy:
            best_accuracy = res["accuracy"]
            save_model(network, model_best_path, cfg, n_inputs)
            print(f"  Saved best model to {model_best_path}", flush=True)

    # Final model and evaluation
    save_model(network, model_path, cfg, n_inputs)
    final_res = evaluate_full(network, test_data)
    with open(os.path.join(run_dir, "training_summary.json"), "w") as f:
        json.dump({
            "final_test_accuracy": final_res["accuracy"],
            "final_test_avg_loss": final_res["avg_loss"],
            "best_accuracy": best_accuracy,
            "best_model_path": model_best_path,
            "epoch_results": epoch_results,
        }, f, indent=2)

    print(f"\n{'='*80}", flush=True)
    print("FINAL TEST EVALUATION", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Test - Average Loss: {final_res['avg_loss']:.4f}, Accuracy: {final_res['accuracy']:.2f}% ({final_res['correct']}/{final_res['total']})", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Model (final): {model_path}", flush=True)
    print(f"Model (best): {model_best_path}", flush=True)

    # Confusion matrix and per-class accuracy
    cm = final_res["confusion_matrix"]
    print(f"\nConfusion matrix (rows=true class, cols=predicted):", flush=True)
    print(cm.astype(int), flush=True)
    print("\nPer-class accuracy (%):", flush=True)
    for i in sorted(final_res["per_class_accuracy"].keys()):
        print(f"  Class {i}: {final_res['per_class_accuracy'][i]:.1f}%", flush=True)


if __name__ == "__main__":
    main()
