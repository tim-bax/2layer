from typing import Dict, List, Tuple

import numpy as np

from data import apply_spike_dropout
from layer import OneLayerEProp


def evaluate(model: OneLayerEProp, dataset: List[Tuple[np.ndarray, int]]) -> Dict[str, float]:
    total = len(dataset)
    correct = 0
    for x, y in dataset:
        pred = model.predict(x)
        if pred == int(y):
            correct += 1
    accuracy = 100.0 * correct / max(total, 1)
    return {"accuracy": accuracy, "correct": correct, "total": total}


def train(
    model: OneLayerEProp,
    train_data: List[Tuple[np.ndarray, int]],
    test_data: List[Tuple[np.ndarray, int]],
    epochs: int,
    batch_size: int = 1,
    spike_dropout_prob: float = 0.0,
    warmup_readout_epochs: int = 0,
) -> List[Dict[str, float]]:
    del batch_size
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        idx = np.random.permutation(len(train_data))
        losses: List[float] = []
        correct = 0

        for i in idx:
            x, y = train_data[int(i)]
            x_used = apply_spike_dropout(x, spike_dropout_prob) if spike_dropout_prob > 0 else x

            if epoch < warmup_readout_epochs:
                loss, _, _ = model.train_step(x_used, int(y), lr_dend=0.0, lr_soma=0.0)
            else:
                loss, _, _ = model.train_step(x_used, int(y))

            losses.append(float(loss))
            pred = model.predict(x)
            if pred == int(y):
                correct += 1

        train_acc = 100.0 * correct / max(len(train_data), 1)
        test_metrics = evaluate(model, test_data)
        result = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "train_accuracy": train_acc,
            "test_accuracy": test_metrics["accuracy"],
        }
        history.append(result)
        print(
            f"Epoch {result['epoch']:03d} | "
            f"train_loss={result['train_loss']:.4f} "
            f"train_acc={result['train_accuracy']:.2f}% "
            f"test_acc={result['test_accuracy']:.2f}%",
            flush=True,
        )

    return history
