#!/usr/bin/env python3
import argparse
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
from jax import random
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data import create_shd_input_jax, load_shd_data
from neuron import NeuronConfig
from network import TwoLayerEProp
from training import evaluate, train


def parse_args():
    p = argparse.ArgumentParser(description="Minimal modular 2-layer SHD run")
    p.add_argument("--T", type=int, default=700)
    p.add_argument("--n_extra", type=int, default=42)
    p.add_argument("--n_hidden", type=int, default=40)
    p.add_argument("--n_outputs", type=int, default=20)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr_extra_dend", type=float, default=0.05)
    p.add_argument("--lr_extra_soma", type=float, default=0.0025)
    p.add_argument("--lr_hidden_dend", type=float, default=0.05)
    p.add_argument("--lr_hidden_soma", type=float, default=0.0025)
    p.add_argument("--lr_readout", type=float, default=0.025)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--gradient_clip", type=float, default=5.0)
    p.add_argument("--loss_temperature", type=float, default=5.0)
    p.add_argument("--loss_count_bias", type=float, default=0.1)
    p.add_argument("--loss_label_smoothing", type=float, default=0.2)
    p.add_argument("--beta_s", type=float, default=1.0)
    p.add_argument("--beta_d", type=float, default=1.5)
    p.add_argument("--weight_scale", type=float, default=0.25)
    p.add_argument("--spike_dropout", type=float, default=0.0)
    p.add_argument("--no_kernel", action="store_true")
    p.add_argument("--spike_amplitude", type=float, default=1.0)
    p.add_argument("--full_history", action="store_true",
                   help="Use full-history gradient path instead of low-memory online default")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    key = random.PRNGKey(args.seed)

    print("Loading SHD data...", flush=True)
    train_raw, test_raw = load_shd_data()
    input_kw = {"T": args.T, "use_kernel": not args.no_kernel, "spike_amplitude": args.spike_amplitude}
    train_data = [(create_shd_input_jax(x, **input_kw), y) for x, y in train_raw]
    test_data = [(create_shd_input_jax(x, **input_kw), y) for x, y in test_raw]
    n_inputs = train_data[0][0].shape[1]
    print(f"Train: {len(train_data)} Test: {len(test_data)} n_inputs: {n_inputs}", flush=True)

    config = NeuronConfig(beta_s=args.beta_s, beta_d=args.beta_d, weight_scale=args.weight_scale)
    model = TwoLayerEProp(
        key=key,
        n_inputs=n_inputs,
        n_extra=args.n_extra,
        n_hidden=args.n_hidden,
        n_outputs=args.n_outputs,
        T=args.T,
        neuron_config=config,
        learning_rate_extra_dendritic=args.lr_extra_dend,
        learning_rate_extra_soma=args.lr_extra_soma,
        learning_rate_hidden_dendritic=args.lr_hidden_dend,
        learning_rate_hidden_somatic=args.lr_hidden_soma,
        learning_rate_readout=args.lr_readout,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        loss_temperature=args.loss_temperature,
        loss_count_bias=args.loss_count_bias,
        loss_label_smoothing=args.loss_label_smoothing,
        use_low_memory=not args.full_history,
    )

    history = train(
        model=model,
        train_data=train_data,
        test_data=test_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        spike_dropout_prob=args.spike_dropout,
    )
    final_test = evaluate(model, test_data)
    print(
        f"Final test accuracy: {final_test['accuracy']:.2f}% "
        f"({final_test['correct']}/{final_test['total']})",
        flush=True,
    )
    if history:
        last = history[-1]
        print(
            f"Last epoch: loss={last['train_loss']:.4f}, "
            f"train_acc={last['train_accuracy']:.2f}%, test_acc={last['test_accuracy']:.2f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
