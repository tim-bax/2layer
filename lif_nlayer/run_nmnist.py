#!/usr/bin/env python3
"""Multi-layer LIF e-prop on N-MNIST (standalone lif_nlayer folder)."""
import argparse
import os
import sys
import time

import jax


def _precision_from_argv(argv):
    default = "64"
    for i, arg in enumerate(argv):
        if arg.startswith("--precision="):
            return arg.split("=", 1)[1]
        if arg == "--precision" and i + 1 < len(argv):
            return argv[i + 1]
    return default


_PRECISION = _precision_from_argv(sys.argv[1:])
if _PRECISION not in ("32", "64"):
    raise ValueError(f"Invalid --precision '{_PRECISION}'. Expected '32' or '64'.")
jax.config.update("jax_enable_x64", _PRECISION == "64")
import jax.numpy as jnp
from jax import random
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data import create_nmnist_input_jax, load_nmnist_data
from config import NeuronConfig
from network import Network, parse_hidden_sizes


def parse_args():
    p = argparse.ArgumentParser(description="Multi-layer LIF e-prop on N-MNIST")
    p.add_argument("--T", type=int, default=300, help="Number of time steps")
    p.add_argument(
        "--hidden",
        type=str,
        default="64,64",
        help="Comma-separated hidden widths, e.g. 128,64",
    )
    p.add_argument("--n-classes", type=int, default=10, dest="n_classes")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_clip", type=float, default=5.0)
    p.add_argument("--loss_temperature", type=float, default=2.7)
    p.add_argument("--loss_count_bias", type=float, default=0.18)
    p.add_argument("--loss_label_smoothing", type=float, default=0.13)
    p.add_argument("--beta_s", type=float, default=1.0)
    p.add_argument("--beta_d", type=float, default=1.5)
    p.add_argument("--weight_scale", type=float, default=0.25)
    p.add_argument("--no_kernel", action="store_true")
    p.add_argument("--spike_amplitude", type=float, default=5.0,
                   help="Used when --no_kernel (default 5.0)")
    p.add_argument("--dropout_hidden", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument(
        "--precision",
        choices=["32", "64"],
        default=_PRECISION,
    )
    p.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="NMNIST root (default: NMNIST_DATA_PATH env or ~/Documents/NMNIST_data)",
    )
    return p.parse_args()


def evaluate(net, dataset, batch_size=1):
    n = len(dataset)
    if batch_size <= 1:
        correct = sum(1 for x, y in dataset if net.predict(x) == int(y))
        return 100.0 * correct / max(n, 1)

    correct = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = dataset[start:end]
        actual = len(batch)
        xs = [x for x, y in batch]
        ys = jnp.array([int(y) for x, y in batch])
        if actual < batch_size:
            xs += [xs[0]] * (batch_size - actual)
        preds = net.batch_predict(jnp.stack(xs))
        correct += int(jnp.sum(preds[:actual] == ys))
    return 100.0 * correct / max(n, 1)


def main():
    args = parse_args()
    if args.precision != _PRECISION:
        raise ValueError(
            f"--precision mismatch during startup ({args.precision} vs {_PRECISION})."
        )

    hidden_sizes = parse_hidden_sizes(args.hidden)
    np.random.seed(args.seed)
    key = random.PRNGKey(args.seed)
    B = args.batch_size

    if args.data_path:
        data_path = args.data_path
    elif "NMNIST_DATA_PATH" in os.environ:
        data_path = os.environ["NMNIST_DATA_PATH"]
    elif os.path.exists("/share/neurocomputation/Tim/NMNIST_data"):
        data_path = "/share/neurocomputation/Tim/NMNIST_data"
    else:
        data_path = os.path.expanduser("~/Documents/NMNIST_data")

    print("Loading N-MNIST...", flush=True)
    train_raw, test_raw = load_nmnist_data(
        data_path, train_samples_per_class=None, test_samples_per_class=None
    )
    input_kw: dict = {"T": args.T}
    if args.no_kernel:
        input_kw["use_kernel"] = False
        input_kw["spike_amplitude"] = args.spike_amplitude
    train_data = [(create_nmnist_input_jax(x, **input_kw), y) for x, y in train_raw]
    test_data = [(create_nmnist_input_jax(x, **input_kw), y) for x, y in test_raw]
    n_inputs = train_data[0][0].shape[1]
    print(
        f"Train: {len(train_data)}  Test: {len(test_data)}  n_inputs={n_inputs}  "
        f"T={args.T}  batch={B}  precision=float{args.precision}",
        flush=True,
    )

    config = NeuronConfig(
        beta_s=args.beta_s,
        beta_d=args.beta_d,
        weight_scale=args.weight_scale,
        loss_temperature=args.loss_temperature,
        loss_count_bias=args.loss_count_bias,
        loss_label_smoothing=args.loss_label_smoothing,
    )
    net = Network(
        key,
        n_inputs,
        hidden_sizes,
        args.n_classes,
        config,
        optimizer=args.optimizer,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        dropout_hidden=args.dropout_hidden,
        weight_decay=args.weight_decay,
    )
    hid_str = " -> ".join(str(h) for h in hidden_sizes)
    print(
        f"Network: {n_inputs} -> {hid_str} -> {args.n_classes} (LIF readout)  "
        f"opt={args.optimizer} lr={args.lr}",
        flush=True,
    )

    pre_acc = evaluate(net, test_data, B)
    print(f"Pre-training test accuracy: {pre_acc:.2f}%", flush=True)

    n_train = len(train_data)
    n_batches = n_train // B
    samples_per_epoch = n_batches * B
    log_interval = 500
    log_every = max(1, log_interval // B)

    for epoch in range(1, args.epochs + 1):
        idx = np.random.permutation(n_train)
        losses = []
        correct = 0
        epoch_t0 = time.time()
        batch_t0 = time.time()

        for bi in range(n_batches):
            start = bi * B
            batch_idx = idx[start : start + B]

            if B == 1:
                x, y = train_data[int(batch_idx[0])]
                loss, pred, _ = net.train_step(
                    jnp.array(x), int(y), lr=args.lr, clip_value=args.gradient_clip,
                )
                batch_correct = int(pred == int(y))
            else:
                x_batch = jnp.stack([train_data[int(i)][0] for i in batch_idx])
                y_batch = jnp.array([int(train_data[int(i)][1]) for i in batch_idx])
                loss, preds, _ = net.batch_train_step(
                    x_batch, y_batch, lr=args.lr, clip_value=args.gradient_clip,
                )
                batch_correct = int(jnp.sum(preds == y_batch))

            losses.append(loss)
            correct += batch_correct

            if (bi + 1) % log_every == 0:
                elapsed = time.time() - batch_t0
                samples_done = (bi + 1) * B
                sps = (log_every * B) / max(elapsed, 1e-6)
                avg_loss = float(np.mean(losses[-log_every:]))
                acc_so_far = 100.0 * correct / samples_done
                print(
                    f"  [{samples_done:5d}/{samples_per_epoch}] loss={avg_loss:.4f} "
                    f"acc={acc_so_far:.1f}% | {sps:.1f} samples/s",
                    flush=True,
                )
                batch_t0 = time.time()

        epoch_elapsed = time.time() - epoch_t0
        train_acc = 100.0 * correct / max(samples_per_epoch, 1)
        test_acc = evaluate(net, test_data, B)
        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} "
            f"train_acc={train_acc:.2f}% test_acc={test_acc:.2f}% ({epoch_elapsed:.1f}s)",
            flush=True,
        )

    final_acc = evaluate(net, test_data, B)
    print(f"\nFinal test accuracy: {final_acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()
