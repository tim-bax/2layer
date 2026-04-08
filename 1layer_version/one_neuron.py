import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax, random
from flax import struct


RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))


@struct.dataclass
class NetworkParams:
    w_dend: jnp.ndarray
    w_soma: jnp.ndarray
    w_readout: jnp.ndarray


@struct.dataclass
class NeuronConfig:
    mu_th: float = 1.0
    v_th: float = 1.0
    gamma: float = 0.5
    tau_soma: float = 15.0
    tau_dend: float = 15.0
    tau_plat_min: float = 100.0
    tau_plat_max: float = 350.0
    dt: float = 1.0
    tau_m: float = 20.0
    v_reset: float = 0.0
    beta_s: float = 0.36
    beta_d: float = 0.75
    beta: float = 0.36
    weight_scale: float = 0.15


class JAXTwoCompartmentalLayer:
    def __init__(self, key, n_neurons: int, n_inputs: int, config: NeuronConfig):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.config = config
        self.alpha_s = jnp.exp(-config.dt / config.tau_soma)
        self.alpha = jnp.exp(-config.dt / config.tau_dend)

        key1, key2, key3 = random.split(key, 3)
        tau_plat_values = random.uniform(
            key3,
            shape=(n_neurons,),
            minval=config.tau_plat_min,
            maxval=config.tau_plat_max,
        )
        self.T_p = (tau_plat_values / config.dt).astype(jnp.int32)

        xavier_std = jnp.sqrt(2.0 / n_inputs)
        scale = xavier_std * config.weight_scale
        self.w_dend = random.normal(key1, (n_neurons, n_inputs)) * scale
        self.w_soma = random.normal(key2, (n_neurons, n_inputs)) * scale

    @staticmethod
    @jit(static_argnames=["T"])
    def forward_pass(
        x_input: jnp.ndarray,
        w_dend: jnp.ndarray,
        w_soma: jnp.ndarray,
        config: NeuronConfig,
        T: int,
        T_p: jnp.ndarray,
    ):
        n_neurons = w_dend.shape[0]
        alpha_s = jnp.exp(-config.dt / config.tau_soma)
        alpha = jnp.exp(-config.dt / config.tau_dend)
        dend_inputs = x_input @ w_dend.T
        soma_inputs = x_input @ w_soma.T
        neuron_indices_static = jnp.arange(n_neurons)

        def step(carry, inputs):
            mu_prev, v_prev, h_prev, t_prime_prev, mu_history = carry
            dend_in, soma_in, t = inputs

            t_prime = jnp.where(t == 0, 0, jnp.where(h_prev == 1, t_prime_prev, t)).astype(jnp.int64)
            mu = jnp.where(t > 0, alpha * mu_prev + (1 - h_prev) * dend_in, dend_in)

            t_prime_int = t_prime.astype(jnp.int32)
            mu_at_tprime_from_history = mu_history[t_prime_int, neuron_indices_static]
            mu_at_initiation = jnp.where(t_prime < t, mu_at_tprime_from_history, mu)

            plateau_duration = t - t_prime
            h = jnp.where(
                (mu_at_initiation >= config.mu_th) & (plateau_duration <= T_p) & (plateau_duration >= 0),
                1,
                0,
            ).astype(jnp.int64)

            mu_history = mu_history.at[t].set(mu)
            v = jnp.where(t > 0, alpha_s * v_prev + soma_in, soma_in)
            o = jnp.where(v >= config.v_th - config.gamma * h, 1, 0).astype(jnp.int64)
            v_for_history = v
            v = v * (1 - o)
            t_prime_next = t_prime
            return (mu, v, h, t_prime_next, mu_history), (mu, v_for_history, h, o, t_prime)

        init_mu_history = jnp.zeros((T, n_neurons))
        init_state = (
            jnp.zeros(n_neurons),
            jnp.zeros(n_neurons),
            jnp.zeros(n_neurons, dtype=jnp.int64),
            jnp.zeros(n_neurons, dtype=jnp.int64),
            init_mu_history,
        )
        time_indices = jnp.arange(T, dtype=jnp.int64)
        _, outputs = lax.scan(step, init_state, (dend_inputs, soma_inputs, time_indices))
        mu, v_history, h, o, t_prime_history = outputs
        return mu, v_history[-1], h, o, mu, t_prime_history, v_history

    @staticmethod
    @jit
    def surrogate_sigma(x: jnp.ndarray, beta: float = 0.5) -> jnp.ndarray:
        return 1.0 / (1.0 + beta * jnp.abs(x)) ** 2

    @staticmethod
    @jit(static_argnames=[])
    def compute_dmu_tprime_dw(
        x_input: jnp.ndarray, h_history: jnp.ndarray, t_prime_history: jnp.ndarray, alpha: float
    ) -> jnp.ndarray:
        T = x_input.shape[0]
        n_neurons = h_history.shape[1]
        n_inputs = x_input.shape[1]
        time_indices = jnp.arange(T)[:, None]
        no_plateau_mask = t_prime_history == time_indices

        def step_dmu(carry, inputs):
            dmu_dw_prev = carry
            x_t, h_prev, no_plateau_t = inputs
            x_t_broadcast = jnp.broadcast_to(x_t[None, :], (n_neurons, n_inputs))
            h_prev_broadcast = h_prev[:, None]
            dmu_dw_new = jnp.where(
                no_plateau_t[:, None],
                alpha * dmu_dw_prev + (1 - h_prev_broadcast) * x_t_broadcast,
                jnp.zeros_like(dmu_dw_prev),
            )
            return dmu_dw_new, dmu_dw_new

        init_dmu_dw = jnp.zeros((n_neurons, n_inputs))
        h_prev = jnp.concatenate([jnp.zeros((1, n_neurons)), h_history[:-1]], axis=0)
        _, dmu_dw_standard = lax.scan(step_dmu, init_dmu_dw, (x_input, h_prev, no_plateau_mask))

        neuron_indices_2d = jnp.broadcast_to(jnp.arange(n_neurons)[None, :], (T, n_neurons))
        t_prime_int = t_prime_history.astype(jnp.int32)
        dmu_dw_plateau_values = dmu_dw_standard[t_prime_int, neuron_indices_2d]
        in_plateau_mask = ~no_plateau_mask
        dmu_dw_history = jnp.where(in_plateau_mask[:, :, None], dmu_dw_plateau_values, dmu_dw_standard)
        return dmu_dw_history

    @staticmethod
    @jit
    def compute_eligibility_traces(x_input: jnp.ndarray, alpha: float) -> jnp.ndarray:
        x_input_float = x_input.astype(jnp.float64)

        def scan_fn(carry, x_t):
            E_t = alpha * carry + x_t
            return E_t, E_t

        init_E = jnp.zeros_like(x_input_float[0], dtype=jnp.float64)
        _, E = lax.scan(scan_fn, init_E, x_input_float)
        return E


class JAXLIFLayer:
    def __init__(self, key, n_neurons: int, n_inputs: int, config: NeuronConfig):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.config = config
        self.alpha = jnp.exp(-config.dt / config.tau_m)
        xavier_std = jnp.sqrt(2.0 / n_inputs)
        scale = xavier_std * config.weight_scale
        self.w = random.normal(key, (n_neurons, n_inputs)) * scale

    @staticmethod
    @jit
    def forward_pass(spike_inputs: jnp.ndarray, w: jnp.ndarray, config: NeuronConfig, T: int):
        n_neurons = w.shape[0]
        alpha = jnp.exp(-config.dt / config.tau_m)
        inputs = spike_inputs @ w.T

        def step(carry, inp):
            v = alpha * carry + inp
            o = jnp.where(v >= config.v_th, 1, 0).astype(jnp.int64)
            v = v * (1 - o) + config.v_reset * o
            return v, (v, o)

        init_v = jnp.zeros(n_neurons)
        _, outputs = lax.scan(step, init_v, inputs)
        v, o = outputs
        return v, o
