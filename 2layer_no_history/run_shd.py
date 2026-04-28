#!/usr/bin/env python3
"""Two-layer no_history network on SHD.

Mirrors no_history/run_shd.py, with one extra hidden 2-comp layer and two
dropout knobs:
  --dropout_extra   : drops extra-layer spikes on their way to hidden
  --dropout_hidden  : drops hidden-layer spikes on their way to readout
                      (this is the dropout from no_history)
"""
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

from data import create_shd_input_jax, load_shd_data
from config import NeuronConfig
from network import Network


def parse_args():
    p = argparse.ArgumentParser(description="Two-layer no_history model on SHD")
    p.add_argument("--T", type=int, default=700)
    p.add_argument("--n_extra", type=int, default=64,
                   help="Number of neurons in the extra (first) hidden 2-comp layer.")
    p.add_argument("--n_hidden", type=int, default=64,
                   help="Number of neurons in the second hidden 2-comp layer.")
    p.add_argument("--n_outputs", type=int, default=20)
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
    p.add_argument("--spike_amplitude", type=float, default=1.0)
    p.add_argument("--dropout_extra", type=float, default=0.0,
                   help="Dropout rate on extra-layer spikes (extra → hidden).")
    p.add_argument("--dropout_hidden", type=float, default=0.0,
                   help="Dropout rate on hidden-layer spikes (hidden → readout). "
                        "This is the dropout knob from no_history.")
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="Decoupled weight decay (AdamW-style for Adam, "
                        "L2-equivalent for SGD). Typical: 1e-5 to 1e-3.")
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument(
        "--precision",
        choices=["32", "64"],
        default=_PRECISION,
        help="Floating-point precision for JAX computations.",
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
            f"--precision mismatch during startup ({args.precision} vs {_PRECISION}). "
            "Pass --precision only once."
        )
    np.random.seed(args.seed)
    key = random.PRNGKey(args.seed)
    B = args.batch_size

    print("Loading SHD data...", flush=True)
    train_raw, test_raw = load_shd_data()
    input_kw = {
        "T": args.T,
        "use_kernel": not args.no_kernel,
        "spike_amplitude": args.spike_amplitude,
    }
    train_data = [(create_shd_input_jax(x, **input_kw), y) for x, y in train_raw]
    test_data = [(create_shd_input_jax(x, **input_kw), y) for x, y in test_raw]
    n_inputs = train_data[0][0].shape[1]
    print(
        f"Train: {len(train_data)}  Test: {len(test_data)}  "
        f"n_inputs: {n_inputs}  T: {args.T}  batch_size: {B}  "
        f"precision=float{args.precision}",
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
        key, n_inputs, args.n_extra, args.n_hidden, args.n_outputs, config,
        optimizer=args.optimizer, beta1=args.beta1, beta2=args.beta2, adam_eps=args.adam_eps,
        dropout_extra=args.dropout_extra, dropout_hidden=args.dropout_hidden,
        weight_decay=args.weight_decay,
    )
    opt_str = f"adam(β1={args.beta1},β2={args.beta2})" if args.optimizer == "adam" else "sgd"
    drop_str = ""
    if args.dropout_extra > 0 or args.dropout_hidden > 0:
        drop_str = f"  dropout_extra={args.dropout_extra} dropout_hidden={args.dropout_hidden}"
    wd_str = f"  weight_decay={args.weight_decay}" if args.weight_decay > 0 else ""
    print(
        f"Network: {n_inputs} -> {args.n_extra} (2-comp) -> {args.n_hidden} (2-comp) "
        f"-> {args.n_outputs} (LIF readout)  "
        f"optimizer={opt_str}  lr={args.lr}{drop_str}{wd_str}",
        flush=True,
    )

    pre_acc = evaluate(net, test_data, B)
    print(f"Pre-training test accuracy: {pre_acc:.2f}%", flush=True)

    dev = jax.local_devices()[0]
    if hasattr(dev, "memory_stats") and dev.memory_stats() is not None:
        ms = dev.memory_stats()
        print(
            f"GPU memory: {ms['bytes_in_use']/1e6:.1f} MB in use, "
            f"{ms['peak_bytes_in_use']/1e6:.1f} MB peak, "
            f"{ms['bytes_limit']/1e6:.1f} MB pool",
            flush=True,
        )
    else:
        print(f"Device: {dev} (no memory stats available)", flush=True)

    n_train = len(train_data)
    n_batches = n_train // B
    samples_per_epoch = n_batches * B
    log_interval = 1000
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
                loss, pred, gnorms = net.train_step(
                    jnp.array(x), int(y), lr=args.lr, clip_value=args.gradient_clip,
                )
                batch_correct = int(pred == int(y))
            else:
                x_batch = jnp.stack([train_data[int(i)][0] for i in batch_idx])
                y_batch = jnp.array([int(train_data[int(i)][1]) for i in batch_idx])
                loss, preds, gnorms = net.batch_train_step(
                    x_batch, y_batch, lr=args.lr, clip_value=args.gradient_clip,
                )
                batch_correct = int(jnp.sum(preds == y_batch))

            losses.append(loss)
            correct += batch_correct

            if bi == 0 and hasattr(dev, "memory_stats") and dev.memory_stats() is not None:
                ms = dev.memory_stats()
                print(
                    f"GPU after 1st train batch: {ms['bytes_in_use']/1e6:.1f} MB in use, "
                    f"{ms['peak_bytes_in_use']/1e6:.1f} MB peak",
                    flush=True,
                )

            if (bi + 1) % log_every == 0:
                elapsed = time.time() - batch_t0
                samples_done = (bi + 1) * B
                sps = (log_every * B) / max(elapsed, 1e-6)
                avg_loss = float(np.mean(losses[-log_every:]))
                acc_so_far = 100.0 * correct / samples_done
                remaining = (samples_per_epoch - samples_done) / max(sps, 1e-6)
                print(
                    f"  [{samples_done:5d}/{samples_per_epoch}] loss={avg_loss:.4f} "
                    f"acc={acc_so_far:.1f}% | {sps:.1f} samples/s, "
                    f"~{remaining:.0f}s remaining",
                    flush=True,
                )
                batch_t0 = time.time()

        epoch_elapsed = time.time() - epoch_t0
        train_acc = 100.0 * correct / max(samples_per_epoch, 1)
        test_acc = evaluate(net, test_data, B)
        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} "
            f"train_acc={train_acc:.2f}% test_acc={test_acc:.2f}% "
            f"({epoch_elapsed:.1f}s)",
            flush=True,
        )

    final_acc = evaluate(net, test_data, B)
    print(f"\nFinal test accuracy: {final_acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()
