#!/usr/bin/env python3
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

from data.shd_binned import load_shd_binned
from config import NeuronConfig
from network import Network


def apply_temporal_jitter(x_input, jitter_range: int):
    """Shift one sample in time by a uniform integer offset in [-range, +range].

    - One jitter value per sample (not per neuron/channel).
    - Shifted indices are clamped to [0, T-1] (no wrap, no drop).
    - Collisions from clamping/overlap are summed via np.add.at.
    """
    if jitter_range <= 0:
        return np.asarray(x_input)
    x_np = np.asarray(x_input)
    T = x_np.shape[0]
    shift = np.random.randint(-jitter_range, jitter_range + 1)
    shifted_t = np.clip(np.arange(T) + shift, 0, T - 1)
    out = np.zeros_like(x_np)
    np.add.at(out, shifted_t, x_np)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="No-history model on SHD (count-bin preprocessing)")
    p.add_argument("--bin_size_ms", type=float, default=4.0,
                   help="Time bin width in ms (paper default 4.0; also try 10, 14).")
    p.add_argument("--collapse_factor", type=int, default=5,
                   help="Sum-pool every N consecutive input channels (paper default 5: 700 -> 140).")
    p.add_argument("--max_duration_ms", type=float, default=1400.0,
                   help="Fixed window length in ms (paper default 1400). Sets T = ceil(max_duration_ms / bin_size_ms).")
    p.add_argument("--binarize", action="store_true",
                   help="Cap per-bin counts at 1 instead of feeding raw counts.")
    p.add_argument("--input_scale", type=float, default=1.0,
                   help="Multiplicative scale on the binned input (default 1.0; raw counts).")
    p.add_argument("--n_hidden", type=int, default=64)
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
    p.add_argument("--tau_soma", type=float, default=15.0,
                   help="Soma membrane time constant (physical ms; default 15.0).")
    p.add_argument("--tau_dend", type=float, default=15.0,
                   help="Dendritic membrane time constant (physical ms; default 15.0).")
    p.add_argument("--tau_m", type=float, default=20.0,
                   help="Readout LIF membrane time constant (physical ms; default 20.0).")
    p.add_argument("--tau_plat_min", type=float, default=100.0,
                   help="Plateau duration min (physical ms; default 100).")
    p.add_argument("--tau_plat_max", type=float, default=350.0,
                   help="Plateau duration max (physical ms; default 350).")
    p.add_argument("--mu_th", type=float, default=1.0,
                   help="Dendritic plateau threshold (default 1.0). Lower if plateaus rarely fire.")
    p.add_argument("--v_th", type=float, default=1.0,
                   help="Somatic spike threshold (default 1.0). Lower if neurons rarely spike.")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="Plateau-induced threshold reduction (default 0.5). Effective v_th = v_th - gamma*h. "
                        "Used as the per-neuron initial value of gamma_h.")
    p.add_argument("--train_gamma", action="store_true",
                   help="Make per-neuron gamma_h trainable (reparam: gamma_h = v_th * sigmoid(rho_h)).")
    p.add_argument("--lr_gamma", type=float, default=None,
                   help="Learning rate for rho_h. Defaults to --lr if unset. "
                        "Has no effect unless --train_gamma is set.")
    p.add_argument("--use_bn", action="store_true",
                   help="Apply batch-norm to soma and dend pre-activations (SpArch-style). "
                        "Uses frozen-stats e-prop approximation: μ, σ² treated as stop_gradient "
                        "so the gradient code remains per-sample / online. Effectively useless "
                        "with batch_size=1; use --batch_size >= 8 or so.")
    p.add_argument("--no_train_bn", action="store_true",
                   help="If set with --use_bn, BN normalises but γ_BN, β_BN stay frozen at 1, 0.")
    p.add_argument("--lr_bn", type=float, default=None,
                   help="Learning rate for γ_BN, β_BN. Defaults to --lr if unset.")
    p.add_argument("--bn_momentum", type=float, default=0.1,
                   help="Running-stats momentum for batch-norm (0.1 matches PyTorch default).")
    p.add_argument("--bn_eps", type=float, default=1e-5,
                   help="Numerical stabiliser for batch-norm.")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--augment_jitter",
        action="store_true",
        help="Enable temporal jitter augmentation on training inputs only.",
    )
    p.add_argument(
        "--jitter_range",
        type=int,
        default=10,
        help="Temporal jitter range in timesteps (uniform in [-range, +range]).",
    )
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="Decoupled weight decay (AdamW-style for Adam, "
                        "L2-equivalent for SGD). Typical: 1e-5 to 1e-3.")
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--lr_patience", type=int, default=5,
                   help="ReduceLROnPlateau patience (epochs without test-acc improvement). "
                        "0 disables scheduling.")
    p.add_argument("--lr_factor", type=float, default=0.7,
                   help="ReduceLROnPlateau multiplier (lr := lr * factor). "
                        "Set to 1.0 to disable.")
    p.add_argument("--lr_min", type=float, default=1e-6,
                   help="LR floor; scheduler will not reduce below this.")
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="Stop training if no test-acc improvement for this many epochs. "
                        "0 disables.")
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
    if args.jitter_range < 0:
        raise ValueError("--jitter_range must be >= 0")
    np.random.seed(args.seed)
    key = random.PRNGKey(args.seed)
    B = args.batch_size

    print("Loading SHD data with count-bin preprocessing...", flush=True)
    dtype = np.float64 if args.precision == "64" else np.float32
    X_tr, y_tr, _, X_te, y_te, _ = load_shd_binned(
        bin_size_ms=args.bin_size_ms,
        collapse_factor=args.collapse_factor,
        max_duration_ms=args.max_duration_ms,
        binarize=args.binarize,
        dtype=dtype,
    )
    if args.input_scale != 1.0:
        X_tr = X_tr * args.input_scale
        X_te = X_te * args.input_scale
    train_data = [(X_tr[i], int(y_tr[i])) for i in range(len(y_tr))]
    test_data = [(X_te[i], int(y_te[i])) for i in range(len(y_te))]
    T = train_data[0][0].shape[0]
    n_inputs = train_data[0][0].shape[1]
    print(
        f"Train: {len(train_data)}  Test: {len(test_data)}  "
        f"n_inputs: {n_inputs}  T: {T}  batch_size: {B}  "
        f"precision=float{args.precision}  "
        f"bin={args.bin_size_ms}ms  collapse={args.collapse_factor}  "
        f"binarize={args.binarize}  input_scale={args.input_scale}",
        flush=True,
    )

    config = NeuronConfig(
        dt=args.bin_size_ms,
        tau_soma=args.tau_soma,
        tau_dend=args.tau_dend,
        tau_m=args.tau_m,
        tau_plat_min=args.tau_plat_min,
        tau_plat_max=args.tau_plat_max,
        mu_th=args.mu_th,
        v_th=args.v_th,
        gamma=args.gamma,
        beta_s=args.beta_s,
        beta_d=args.beta_d,
        weight_scale=args.weight_scale,
        loss_temperature=args.loss_temperature,
        loss_count_bias=args.loss_count_bias,
        loss_label_smoothing=args.loss_label_smoothing,
        bn_eps=args.bn_eps,
        bn_momentum=args.bn_momentum,
    )
    alpha_s = float(np.exp(-config.dt / config.tau_soma))
    alpha_d = float(np.exp(-config.dt / config.tau_dend))
    alpha_m = float(np.exp(-config.dt / config.tau_m))
    print(
        f"Neuron dynamics: dt={config.dt}ms (=bin_size_ms), "
        f"tau_soma={config.tau_soma}ms -> alpha_s={alpha_s:.4f}, "
        f"tau_dend={config.tau_dend}ms -> alpha_d={alpha_d:.4f}, "
        f"tau_m={config.tau_m}ms -> alpha_m={alpha_m:.4f}, "
        f"plateau in [{config.tau_plat_min}, {config.tau_plat_max}]ms "
        f"= [{int(config.tau_plat_min/config.dt)}, {int(config.tau_plat_max/config.dt)}] steps",
        flush=True,
    )
    net = Network(
        key, n_inputs, args.n_hidden, args.n_outputs, config,
        optimizer=args.optimizer, beta1=args.beta1, beta2=args.beta2, adam_eps=args.adam_eps,
        dropout_rate=args.dropout, weight_decay=args.weight_decay,
        train_gamma=args.train_gamma, lr_gamma=args.lr_gamma,
        use_bn=args.use_bn, train_bn=not args.no_train_bn, lr_bn=args.lr_bn,
    )
    opt_str = f"adam(β1={args.beta1},β2={args.beta2})" if args.optimizer == "adam" else "sgd"
    drop_str = f"  dropout={args.dropout}" if args.dropout > 0 else ""
    jitter_str = ""
    if args.augment_jitter:
        jitter_str = f"  augment_jitter=True(range=±{args.jitter_range})"
    wd_str = f"  weight_decay={args.weight_decay}" if args.weight_decay > 0 else ""
    gamma_str = ""
    if args.train_gamma:
        eff_lr_gamma = args.lr if args.lr_gamma is None else args.lr_gamma
        gamma_str = f"  train_gamma=True(init={args.gamma}, lr_gamma={eff_lr_gamma:.2e})"
    bn_str = ""
    if args.use_bn:
        eff_lr_bn = args.lr if args.lr_bn is None else args.lr_bn
        bn_str = (f"  use_bn=True(momentum={args.bn_momentum}, "
                  f"train_bn={not args.no_train_bn}, lr_bn={eff_lr_bn:.2e})")
    print(
        f"Network: {n_inputs} -> {args.n_hidden} (2-comp) -> {args.n_outputs} (LIF readout)  "
        f"optimizer={opt_str}  lr={args.lr}{drop_str}{jitter_str}{wd_str}{gamma_str}{bn_str}",
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

    current_lr = args.lr
    best_test_acc = 0.0
    best_epoch = 0
    epochs_since_lr_drop = 0
    epochs_without_improvement = 0

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
                if args.augment_jitter:
                    x = apply_temporal_jitter(x, args.jitter_range)
                loss, pred, gnorms = net.train_step(
                    jnp.array(x), int(y), lr=current_lr, clip_value=args.gradient_clip,
                )
                batch_correct = int(pred == int(y))
            else:
                x_batch_np = [
                    apply_temporal_jitter(train_data[int(i)][0], args.jitter_range)
                    if args.augment_jitter
                    else train_data[int(i)][0]
                    for i in batch_idx
                ]
                x_batch = jnp.stack(x_batch_np)
                y_batch = jnp.array([int(train_data[int(i)][1]) for i in batch_idx])
                loss, preds, gnorms = net.batch_train_step(
                    x_batch, y_batch, lr=current_lr, clip_value=args.gradient_clip,
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

        improved = test_acc > best_test_acc
        if improved:
            best_test_acc = test_acc
            best_epoch = epoch
            epochs_since_lr_drop = 0
            epochs_without_improvement = 0
            marker = "  *best"
        else:
            epochs_since_lr_drop += 1
            epochs_without_improvement += 1
            marker = ""

        gamma_log = ""
        if args.train_gamma:
            gh = np.asarray(net.gamma_h)
            gamma_log = (f" gamma_h(min/mean/max)="
                         f"{gh.min():.3f}/{gh.mean():.3f}/{gh.max():.3f}")
        bn_log = ""
        if args.use_bn:
            gs = np.asarray(net.gamma_BN_s); bs = np.asarray(net.beta_BN_s)
            gd = np.asarray(net.gamma_BN_d); bd = np.asarray(net.beta_BN_d)
            rms = np.asarray(net.running_mean_s); rvs = np.asarray(net.running_var_s)
            rmd = np.asarray(net.running_mean_d); rvd = np.asarray(net.running_var_d)
            bn_log = (
                f"\n  BN soma  γ[{gs.min():+.2f},{gs.max():+.2f}] β[{bs.min():+.2f},{bs.max():+.2f}]"
                f"  μ̄={rms.mean():+.3f}±{rms.std():.3f}  σ̄²={rvs.mean():.4f}±{rvs.std():.4f}"
                f"\n  BN dend  γ[{gd.min():+.2f},{gd.max():+.2f}] β[{bd.min():+.2f},{bd.max():+.2f}]"
                f"  μ̄={rmd.mean():+.3f}±{rmd.std():.3f}  σ̄²={rvd.mean():.4f}±{rvd.std():.4f}"
            )
        print(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} "
            f"train_acc={train_acc:.2f}% test_acc={test_acc:.2f}% "
            f"lr={current_lr:.2e} ({epoch_elapsed:.1f}s){marker}{gamma_log}{bn_log}",
            flush=True,
        )

        if (args.lr_factor < 1.0
                and args.lr_patience > 0
                and current_lr > args.lr_min
                and epochs_since_lr_drop >= args.lr_patience):
            new_lr = max(current_lr * args.lr_factor, args.lr_min)
            if new_lr < current_lr:
                print(f"  LR scheduler: {current_lr:.2e} -> {new_lr:.2e} "
                      f"(no improvement for {epochs_since_lr_drop} epochs)", flush=True)
                current_lr = new_lr
                epochs_since_lr_drop = 0

        if (args.early_stop_patience > 0
                and epochs_without_improvement >= args.early_stop_patience):
            print(f"  Early stopping: no improvement for "
                  f"{epochs_without_improvement} epochs.", flush=True)
            break

    final_acc = evaluate(net, test_data, B)
    print(
        f"\nFinal test accuracy: {final_acc:.2f}%  |  "
        f"Best test accuracy: {best_test_acc:.2f}% (epoch {best_epoch})",
        flush=True,
    )


if __name__ == "__main__":
    main()
