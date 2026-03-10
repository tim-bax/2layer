"""
1-layer low-memory variant. Same API as model.py (JAXEPropNetwork, train_network_jax, etc.).
Currently re-exports model; replace this file with a low-memory implementation when needed.
"""
from model import (
    JAXEPropNetwork,
    train_network_jax,
    print_summary_statistics,
    initialize_numpy_weights,
    NeuronConfig,
)

__all__ = [
    "JAXEPropNetwork",
    "train_network_jax",
    "print_summary_statistics",
    "initialize_numpy_weights",
    "NeuronConfig",
]
