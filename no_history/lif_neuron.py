import jax.numpy as jnp
from jax import random

from config import NeuronConfig


class LIFNeuron:
    def __init__(self, key: jnp.ndarray, n_neurons: int, n_inputs: int, config: NeuronConfig):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.config = config
        self.alpha_m = jnp.exp(-config.dt / config.tau_m)

        xavier_std = jnp.sqrt(2.0 / n_inputs)
        scale = xavier_std * config.weight_scale
        self.w = random.normal(key, (n_neurons, n_inputs)) * scale

    def init_carry(self):
        return (
            jnp.zeros(self.n_neurons),          # v
            jnp.zeros(self.n_neurons),          # readout_counts
            jnp.zeros(self.n_inputs),           # E_readout
        )

    @staticmethod
    def forward_step(carry, spike_input, w, alpha_m, v_th):
        """Pure-function forward step for one timestep. JIT-friendly."""
        v_prev, readout_counts, E_readout_prev = carry

        readout_in = spike_input @ w.T
        v = alpha_m * v_prev + readout_in
        o = jnp.where(v >= v_th, 1, 0).astype(jnp.int32)
        v_pre_reset = v
        v = v * (1 - o)
        readout_counts = readout_counts + o
        E_readout = alpha_m * E_readout_prev + spike_input

        new_carry = (v, readout_counts, E_readout)
        return new_carry, o, v_pre_reset, E_readout

    @staticmethod
    def forward_step_integrate_only(carry, spike_input, w, alpha_m):
        """Leaky integrate readout without threshold, reset, or spike counting.

        Same recurrent filter E_readout as the spiking step; per-class membrane
        v_j is tracked for max-over-time loss and exact ∂L/∂w readout.

        Args:
            carry: (v, readout_counts, E_readout_prev) same 3-tuple as forward_step;
                   readout_counts are left unchanged.
        """
        v_prev, readout_counts, E_readout_prev = carry

        readout_in = spike_input @ w.T
        v = alpha_m * v_prev + readout_in
        E_readout = alpha_m * E_readout_prev + spike_input
        new_carry = (v, readout_counts, E_readout)
        return new_carry, v, E_readout
