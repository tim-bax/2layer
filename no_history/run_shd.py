#!/usr/bin/env python3
import argparse
import os
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
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
    p = argparse.ArgumentParser(description="No-history model on SHD")
    p.add_argument("--T", type=int, default=700)
    p.add_argument("--n_hidden", type=int, default=64)
    p.add_argument("--n_outputs", type=int, default=20)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gradient_clip", type=float, default=5.0)
    p.add_argument("--loss_temperature", type=float, default=2.7)
    p.add_argument("--loss_count_bias", type=float, default=0.18)
    p.add_argument("--loss_label_smoothing", type=float, default=0.13)
    p.add_argument("--beta_s", type=float, default=1.0)
    p.add_argument("--beta_d", type=float, default=1.5)
    p.add_argument("--weight_scale", type=float, default=0.25)
    p.add_argument("--no_kernel", action="store_true")
    p.add_argument("--spike_amplitude", type=float, default=1.0)
    return p.parse_args()


def evaluate(net: Network, dataset):
    correct = 0
    for x, y in dataset:
        if net.predict(x) == int(y):
            correct += 1
    return 100.0 * correct / max(len(dataset), 1)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    key = random.PRNGKey(args.seed)

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
        f"n_inputs: {n_inputs}  T: {args.T}",
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
    net = Network(key, n_inputs, args.n_hidden, args.n_outputs, config)
    print(
        f"Network: {n_inputs} -> {args.n_hidden} (2-comp) -> {args.n_outputs} (LIF readout)",
        flush=True,
    )

    pre_acc = evaluate(net, test_data)
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
    log_interval = 1000
    for epoch in range(1, args.epochs + 1):
        idx = np.random.permutation(n_train)
        losses = []
        correct = 0
        epoch_t0 = time.time()
        batch_t0 = time.time()

        for step, i in enumerate(idx, 1):
            x, y = train_data[int(i)]
            loss, pred = net.train_step(
                jnp.array(x), int(y), lr=args.lr, clip_value=args.gradient_clip
            )
            losses.append(loss)
            if pred == int(y):
                correct += 1

            if step == 1 and hasattr(dev, "memory_stats") and dev.memory_stats() is not None:
                ms = dev.memory_stats()
                print(
                    f"GPU after 1st train step: {ms['bytes_in_use']/1e6:.1f} MB in use, "
                    f"{ms['peak_bytes_in_use']/1e6:.1f} MB peak",
                    flush=True,
                )

            if step % log_interval == 0:
                elapsed = time.time() - batch_t0
                sps = log_interval / max(elapsed, 1e-6)
                avg_loss = float(np.mean(losses[-log_interval:]))
                acc_so_far = 100.0 * correct / step
                remaining = (n_train - step) / max(sps, 1e-6)
                print(
                    f"  [{step:5d}/{n_train}] loss={avg_loss:.4f} "
                    f"acc={acc_so_far:.1f}% | {sps:.2f} samples/s, "
                    f"~{remaining:.0f}s remaining",
                    flush=True,
                )
                batch_t0 = time.time()

        epoch_elapsed = time.time() - epoch_t0
        train_acc = 100.0 * correct / max(n_train, 1)
        test_acc = evaluate(net, test_data)
        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} "
            f"train_acc={train_acc:.2f}% test_acc={test_acc:.2f}% "
            f"({epoch_elapsed:.1f}s)",
            flush=True,
        )

    final_acc = evaluate(net, test_data)
    print(f"\nFinal test accuracy: {final_acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()
