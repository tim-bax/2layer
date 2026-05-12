"""Save and load a trained no_history ``Network`` (weights, T_p, NeuronConfig).

Writes a pair of files:

  - ``<path>.npz`` — ``w_dend``, ``w_soma``, ``w_readout``, ``T_p``
  - ``<path>.meta.json`` — architecture, ``NeuronConfig`` scalars, optional metadata

Example::

    from checkpoint import load_checkpoint
    from jax import random

    net, meta = load_checkpoint("runs/shd_model", key=random.PRNGKey(0))
    y = net.predict(x_t_k)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax import random

from config import NeuronConfig
from network import Network

_NEURON_CONFIG_FIELDS = (
    "mu_th",
    "v_th",
    "gamma",
    "tau_soma",
    "tau_dend",
    "tau_plat_min",
    "tau_plat_max",
    "dt",
    "tau_m",
    "v_reset",
    "beta_s",
    "beta_d",
    "weight_scale",
    "loss_temperature",
    "loss_count_bias",
    "loss_label_smoothing",
)


def config_to_dict(cfg: NeuronConfig) -> Dict[str, float]:
    return {name: float(getattr(cfg, name)) for name in _NEURON_CONFIG_FIELDS}


def config_from_dict(d: Dict[str, Any]) -> NeuronConfig:
    return NeuronConfig(**{name: float(d[name]) for name in _NEURON_CONFIG_FIELDS})


def _paths(path: str) -> Tuple[str, str]:
    base = path[:-4] if path.endswith(".npz") else path
    return base + ".npz", base + ".meta.json"


def save_checkpoint(
    net: Network,
    path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Persist ``net`` to ``.npz`` + ``.meta.json``. Returns ``(npz_path, meta_path)``."""
    npz_path, meta_path = _paths(path)
    d = os.path.dirname(npz_path)
    if d:
        os.makedirs(d, exist_ok=True)

    arrays = {
        "w_dend": np.asarray(net.hidden.w_dend),
        "w_soma": np.asarray(net.hidden.w_soma),
        "w_readout": np.asarray(net.readout.w),
        "T_p": np.asarray(net.hidden.T_p, dtype=np.int32),
    }
    np.savez(npz_path, **arrays)

    meta: Dict[str, Any] = {
        "format": "no_history_network_v1",
        "n_inputs": int(net.n_inputs),
        "n_hidden": int(net.n_hidden),
        "n_outputs": int(net.n_outputs),
        "dropout_rate": float(net.dropout_rate),
        "weight_decay": float(net.weight_decay),
        "optimizer": str(net.optimizer),
        "config": config_to_dict(net.config),
    }
    if net.optimizer == "adam":
        meta["adam"] = {
            "beta1": float(net.beta1),
            "beta2": float(net.beta2),
            "adam_eps": float(net.adam_eps),
        }
    if extra is not None:
        meta["extra"] = extra

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return npz_path, meta_path


def load_checkpoint(
    path: str,
    *,
    key: Optional[jnp.ndarray] = None,
    optimizer: Optional[str] = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_eps: float = 1e-8,
    dropout_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
) -> Tuple[Network, Dict[str, Any]]:
    """Rebuild a ``Network`` and load weights from checkpoint.

    New random ``key`` only re-seeds dropout / fresh Adam moments; weights are
    overwritten from disk.
    """
    npz_path, meta_path = _paths(path)
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(meta_path)

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    if meta.get("format") != "no_history_network_v1":
        raise ValueError(f"Unknown checkpoint format: {meta.get('format')}")

    cfg = config_from_dict(meta["config"])
    opt = optimizer if optimizer is not None else str(meta.get("optimizer", "sgd"))
    dr = float(meta["dropout_rate"] if dropout_rate is None else dropout_rate)
    wd = float(meta["weight_decay"] if weight_decay is None else weight_decay)

    if key is None:
        key = random.PRNGKey(0)

    if opt == "adam":
        a = meta.get("adam", {})
        b1 = float(a.get("beta1", beta1))
        b2 = float(a.get("beta2", beta2))
        eps = float(a.get("adam_eps", adam_eps))
        net = Network(
            key,
            int(meta["n_inputs"]),
            int(meta["n_hidden"]),
            int(meta["n_outputs"]),
            cfg,
            optimizer="adam",
            beta1=b1,
            beta2=b2,
            adam_eps=eps,
            dropout_rate=dr,
            weight_decay=wd,
        )
    else:
        net = Network(
            key,
            int(meta["n_inputs"]),
            int(meta["n_hidden"]),
            int(meta["n_outputs"]),
            cfg,
            optimizer="sgd",
            dropout_rate=dr,
            weight_decay=wd,
        )

    with np.load(npz_path) as z:
        net.hidden.w_dend = jnp.asarray(z["w_dend"])
        net.hidden.w_soma = jnp.asarray(z["w_soma"])
        net.readout.w = jnp.asarray(z["w_readout"])
        net.hidden.T_p = jnp.asarray(z["T_p"], dtype=jnp.int32)

    return net, meta
