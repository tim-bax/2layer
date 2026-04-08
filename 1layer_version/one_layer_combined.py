from typing import Optional, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, random

from one_neuron import (
    JAXLIFLayer,
    JAXTwoCompartmentalLayer,
    NetworkParams,
    NeuronConfig,
    RANDOM_SEED,
)


class OneLayerEProp:
    def __init__(
        self,
        key,
        n_inputs: int = 700,
        n_hidden: int = 64,
        n_outputs: int = 20,
        T: int = 700,
        neuron_config=None,
        learning_rate_hidden_dendritic: float = 0.045,
        learning_rate_hidden_somatic: float = 0.00015,
        learning_rate_readout: float = 0.035,
        weight_decay: float = 1e-5,
        gradient_clip: float = 5.0,
        loss_temperature: float = 2.7,
        loss_count_bias: float = 0.18,
        loss_label_smoothing: float = 0.13,
        grad_dend_scale: float = 1.0,
    ):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.T = T

        self.config = NeuronConfig() if neuron_config is None else neuron_config
        self.learning_rate_hidden_dendritic = float(learning_rate_hidden_dendritic)
        self.learning_rate_hidden_somatic = float(learning_rate_hidden_somatic)
        self.learning_rate_readout = float(learning_rate_readout)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = float(gradient_clip)
        self.loss_temperature = float(loss_temperature)
        self.loss_count_bias = float(loss_count_bias)
        self.loss_label_smoothing = float(loss_label_smoothing)
        self.grad_dend_scale = float(grad_dend_scale)

        key_hidden, key_readout = random.split(key)
        self.hidden_layer = JAXTwoCompartmentalLayer(key_hidden, n_hidden, n_inputs, self.config)
        self.readout_layer = JAXLIFLayer(key_readout, n_outputs, n_hidden, self.config)

        self._train_step_compiled = jit(self._train_step_impl, static_argnums=(2,))

    def get_params(self) -> NetworkParams:
        return NetworkParams(
            w_dend=self.hidden_layer.w_dend,
            w_soma=self.hidden_layer.w_soma,
            w_readout=self.readout_layer.w,
        )

    def set_params(self, params: NetworkParams):
        self.hidden_layer.w_dend = params.w_dend
        self.hidden_layer.w_soma = params.w_soma
        self.readout_layer.w = params.w_readout

    def _forward_with_params(self, params: NetworkParams, x_input: jnp.ndarray) -> Tuple:
        T_p = self.hidden_layer.T_p
        mu, v, h, hidden_o, mu_history, t_prime_history, v_history = JAXTwoCompartmentalLayer.forward_pass(
            x_input, params.w_dend, params.w_soma, self.config, self.T, T_p
        )
        readout_v, readout_o = JAXLIFLayer.forward_pass(hidden_o, params.w_readout, self.config, self.T)
        return mu, v, h, hidden_o, readout_v, readout_o, mu_history, t_prime_history, v_history

    def forward(self, x_input: jnp.ndarray) -> Tuple:
        if not isinstance(x_input, jnp.ndarray):
            x_input = jnp.array(x_input)
        return self._forward_with_params(self.get_params(), x_input)

    def _loss_impl(self, params: NetworkParams, x_input: jnp.ndarray, target: int) -> jnp.ndarray:
        _, _, _, _, _, readout_o, _, _, _ = self._forward_with_params(params, x_input)
        return self._loss_from_readout(readout_o, target)

    def _loss_from_readout(self, readout_o: jnp.ndarray, target: int) -> jnp.ndarray:
        readout_counts = jnp.sum(readout_o, axis=0)
        scaled_counts = readout_counts / self.loss_temperature + self.loss_count_bias
        exp_counts = jnp.exp(scaled_counts - jnp.max(scaled_counts))
        probabilities = exp_counts / jnp.sum(exp_counts)
        target_one_hot = jnp.zeros(self.n_outputs).at[target].set(1.0)
        target_one_hot = target_one_hot * (1 - self.loss_label_smoothing) + self.loss_label_smoothing / self.n_outputs
        return -jnp.sum(target_one_hot * jnp.log(probabilities + 1e-8))

    def compute_global_errors(self, readout_o: jnp.ndarray, target: int) -> jnp.ndarray:
        readout_counts = jnp.sum(readout_o, axis=0)
        scaled_counts = readout_counts / self.loss_temperature + self.loss_count_bias
        exp_counts = jnp.exp(scaled_counts - jnp.max(scaled_counts))
        probabilities = exp_counts / jnp.sum(exp_counts)
        target_one_hot = jnp.zeros(self.n_outputs).at[target].set(1.0)
        target_one_hot = target_one_hot * (1 - self.loss_label_smoothing) + self.loss_label_smoothing / self.n_outputs
        return target_one_hot - probabilities

    def _train_step_impl(
        self,
        params: NetworkParams,
        x_input: jnp.ndarray,
        target: int,
        lr_dend: float,
        lr_soma: float,
        lr_readout: float,
        clip_value: float = 5.0,
    ):
        mu, v, h, hidden_o, readout_v, readout_o, mu_history, t_prime_history, v_history = self._forward_with_params(
            params, x_input
        )
        global_errors = self.compute_global_errors(readout_o, target)

        E_readout = JAXTwoCompartmentalLayer.compute_eligibility_traces(hidden_o, self.readout_layer.alpha)
        sigma_prime_readout = JAXTwoCompartmentalLayer.surrogate_sigma(
            readout_v - self.config.v_th, self.config.beta_s
        )
        grad_readout_raw = jnp.einsum("ti,tj,i->ij", sigma_prime_readout, E_readout, global_errors) / self.T
        grad_readout = jnp.clip(grad_readout_raw, -clip_value, clip_value)

        T = x_input.shape[0]
        n_neurons = h.shape[1]
        effective_error = jnp.einsum("tj,j,ji->ti", sigma_prime_readout, global_errors, params.w_readout)

        soma_input_vals = v_history + self.config.gamma * h - self.config.v_th
        sigma_prime_hidden = JAXTwoCompartmentalLayer.surrogate_sigma(soma_input_vals, self.config.beta_s)

        t_prime_indices = t_prime_history.astype(jnp.int32)
        neuron_indices = jnp.arange(n_neurons)[None, :]
        mu_at_tprime = mu_history[t_prime_indices, neuron_indices]
        h_prime = JAXTwoCompartmentalLayer.surrogate_sigma(mu_at_tprime - self.config.mu_th, self.config.beta_d)

        E_soma = JAXTwoCompartmentalLayer.compute_eligibility_traces(x_input, self.hidden_layer.alpha_s)
        dmu_tprime_dw = JAXTwoCompartmentalLayer.compute_dmu_tprime_dw(
            x_input, h, t_prime_history, self.hidden_layer.alpha
        )

        grad_soma_raw = jnp.einsum("ti,tj,ti->ij", sigma_prime_hidden, E_soma, effective_error) / self.T
        grad_soma = jnp.clip(grad_soma_raw, -clip_value, clip_value)

        grad_dend_raw = (
            jnp.einsum(
                "ti,ti,tij,ti->ij",
                sigma_prime_hidden,
                h_prime,
                dmu_tprime_dw,
                effective_error * self.config.gamma,
            )
            / self.T
        )
        grad_dend = jnp.clip(grad_dend_raw, -clip_value, clip_value)

        new_w_dend = params.w_dend * (1 - self.weight_decay) + lr_dend * (self.grad_dend_scale * grad_dend)
        new_w_soma = params.w_soma * (1 - self.weight_decay) + lr_soma * grad_soma
        new_w_readout = params.w_readout * (1 - self.weight_decay) + lr_readout * grad_readout

        new_params = NetworkParams(
            w_dend=jnp.clip(new_w_dend, -1.0, 1.0),
            w_soma=jnp.clip(new_w_soma, -1.0, 1.0),
            w_readout=jnp.clip(new_w_readout, -1.0, 1.0),
        )
        loss = self._loss_from_readout(readout_o, target)
        return new_params, loss, hidden_o, readout_o

    def train_step(
        self,
        x_input: jnp.ndarray,
        target: int,
        lr_dend: Optional[float] = None,
        lr_soma: Optional[float] = None,
        lr_readout: Optional[float] = None,
    ):
        if not isinstance(x_input, jnp.ndarray):
            x_input = jnp.array(x_input)
        if lr_dend is None:
            lr_dend = self.learning_rate_hidden_dendritic
        if lr_soma is None:
            lr_soma = self.learning_rate_hidden_somatic
        if lr_readout is None:
            lr_readout = self.learning_rate_readout

        params = self.get_params()
        new_params, loss, hidden_o, readout_o = self._train_step_compiled(
            params, x_input, target, lr_dend, lr_soma, lr_readout, self.gradient_clip
        )
        self.set_params(new_params)
        return float(loss), hidden_o, readout_o


def demo():
    key = random.PRNGKey(RANDOM_SEED)
    model = OneLayerEProp(key, n_inputs=16, n_hidden=8, n_outputs=4, T=30)
    x = jnp.zeros((30, 16), dtype=jnp.float64)
    x = x.at[5, 1].set(1.0).at[6, 1].set(1.0).at[12, 3].set(1.0)
    target = 2
    loss, _, _ = model.train_step(x, target)
    print(f"one-step loss: {loss:.6f}")


if __name__ == "__main__":
    demo()
