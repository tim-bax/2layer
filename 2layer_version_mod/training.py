import time
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np

from data import apply_spike_dropout
from network import TwoLayerEProp


def evaluate(model: TwoLayerEProp, dataset: List[Tuple[np.ndarray, int]]) -> Dict[str, float]:
    total = len(dataset)
    correct = 0
    for x, y in dataset:
        if model.predict(x) == int(y):
            correct += 1
    accuracy = 100.0 * correct / max(total, 1)
    return {"accuracy": accuracy, "correct": correct, "total": total}


def train(
    model: TwoLayerEProp,
    train_data: List[Tuple[np.ndarray, int]],
    test_data: List[Tuple[np.ndarray, int]],
    epochs: int,
    batch_size: int = 1,
    spike_dropout_prob: float = 0.0,
) -> List[Dict[str, float]]:
    del batch_size
    history: List[Dict[str, float]] = []
    n_train = len(train_data)
    log_interval = 1000
    for epoch in range(epochs):
        idx = np.random.permutation(n_train)
        losses = []
        correct = 0
        epoch_t0 = time.time()
        batch_t0 = time.time()
        for step, i in enumerate(idx, 1):
            x, y = train_data[int(i)]
            x_used = apply_spike_dropout(x, spike_dropout_prob) if spike_dropout_prob > 0 else x
            loss, readout_o = model.train_step(x_used, int(y))
            losses.append(loss)
            pred = int(jnp.argmax(jnp.sum(readout_o, axis=0)))
            if pred == int(y):
                correct += 1
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
        test_metrics = evaluate(model, test_data)
        result = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "train_accuracy": train_acc,
            "test_accuracy": test_metrics["accuracy"],
        }
        history.append(result)
        print(
            f"Epoch {result['epoch']:03d} | train_loss={result['train_loss']:.4f} "
            f"train_acc={result['train_accuracy']:.2f}% test_acc={result['test_accuracy']:.2f}% "
            f"({epoch_elapsed:.1f}s)",
            flush=True,
        )
    return history
