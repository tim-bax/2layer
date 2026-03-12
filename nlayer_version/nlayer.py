"""
N-layer e-prop network: L configurable two-compartment layers (input → L1 → L2 → ... → L_L → readout).
Uses the same 2comp building blocks and recursive effective-error rule as 1/2-layer versions.
"""
import os
import sys
from typing import List, Tuple, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
# 2comp_uniform lives in 2layer_version
_COMP_UNIFORM_PATH = os.path.join(_PARENT_DIR, "2layer_version", "2comp_uniform.py")
if not os.path.isfile(_COMP_UNIFORM_PATH):
    _COMP_UNIFORM_PATH = os.path.join(_SCRIPT_DIR, "2comp_uniform.py")
if not os.path.isfile(_COMP_UNIFORM_PATH):
    raise FileNotFoundError(
        f"2comp_uniform.py not found. Looked in {_PARENT_DIR!r}/2layer_version and {_SCRIPT_DIR!r}."
    )
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
import importlib.util
_spec = importlib.util.spec_from_file_location("comp_uniform", _COMP_UNIFORM_PATH)
_comp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_comp)

NeuronConfig = _comp.NeuronConfig
JAXTwoCompartmentalLayer = _comp.JAXTwoCompartmentalLayer
JAXLIFLayer = _comp.JAXLIFLayer

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, jit, lax
import numpy as np


def _make_params_struct(dend_weights: Tuple[jnp.ndarray, ...], soma_weights: Tuple[jnp.ndarray, ...],
                        w_readout: jnp.ndarray) -> Tuple:
    """Params as a PyTree: (dend_tuple, soma_tuple, w_readout)."""
    return (dend_weights, soma_weights, w_readout)


class JAXEPropNetworkNLayer:
    """
    E-prop network with L two-compartment layers (L configurable).
    Architecture: input (n_inputs) → layer_0 (n_0) → ... → layer_{L-1} (n_{L-1}) → LIF readout (n_outputs).
    layer_sizes = [n_0, n_1, ..., n_{L-1}] gives the number of neurons per 2comp layer; L = len(layer_sizes).
    """

    def __init__(
        self,
        key,
        n_inputs: int,
        layer_sizes: List[int],
        n_outputs: int,
        T: int,
        learning_rate_dendritic: Optional[float] = None,
        learning_rate_somatic: Optional[float] = None,
        learning_rate_readout: Optional[float] = None,
        weight_decay: Optional[float] = None,
        gradient_clip: Optional[float] = None,
        loss_temperature: Optional[float] = None,
        loss_count_bias: Optional[float] = None,
        loss_label_smoothing: Optional[float] = None,
    ):
        self.n_inputs = n_inputs
        self.layer_sizes = list(layer_sizes)
        self.L = len(self.layer_sizes)
        self.n_outputs = n_outputs
        self.T = T
        self.config = NeuronConfig()

        if self.L < 1:
            raise ValueError("Need at least one 2comp layer (layer_sizes non-empty).")

        keys = random.split(key, self.L + 1)
        # Layer ell: n_neurons = layer_sizes[ell], n_inputs = layer_sizes[ell-1] if ell>0 else n_inputs
        self.layers: List[JAXTwoCompartmentalLayer] = []
        for ell in range(self.L):
            n_in = n_inputs if ell == 0 else self.layer_sizes[ell - 1]
            n_out = self.layer_sizes[ell]
            self.layers.append(
                JAXTwoCompartmentalLayer(keys[ell], n_out, n_in, self.config)
            )
        self.readout_layer = JAXLIFLayer(keys[self.L], n_outputs, self.layer_sizes[self.L - 1], self.config)

        # Learning rates: one per layer (dend, soma) + readout; default same for all layers
        default_lr_d = 0.05
        default_lr_s = 0.0025
        default_lr_r = 0.025
        self.learning_rates_dendritic = [learning_rate_dendritic or default_lr_d] * self.L
        self.learning_rates_somatic = [learning_rate_somatic or default_lr_s] * self.L
        self.learning_rate_readout = learning_rate_readout if learning_rate_readout is not None else default_lr_r
        self.weight_decay = weight_decay if weight_decay is not None else 0.00001
        self.gradient_clip = gradient_clip if gradient_clip is not None else 5.0
        self.loss_temperature = loss_temperature if loss_temperature is not None else 5.0
        self.loss_count_bias = loss_count_bias if loss_count_bias is not None else 0.1
        self.loss_label_smoothing = loss_label_smoothing if loss_label_smoothing is not None else 0.2

        self.activity_history: List = []

    def get_params(self) -> Tuple:
        dend = tuple(lyr.w_dend for lyr in self.layers)
        soma = tuple(lyr.w_soma for lyr in self.layers)
        return _make_params_struct(dend, soma, self.readout_layer.w)

    def set_params(self, params: Tuple):
        dend, soma, w_readout = params
        for ell in range(self.L):
            self.layers[ell].w_dend = dend[ell]
            self.layers[ell].w_soma = soma[ell]
        self.readout_layer.w = w_readout

    def _forward_with_params(self, params: Tuple, x_input: jnp.ndarray) -> Tuple:
        """Forward: x → L0 → L1 → ... → L_{L-1} → readout. Returns layer outputs + readout."""
        dend, soma, w_readout = params
        T = x_input.shape[0]
        layer_outputs = []
        inp = x_input
        for ell in range(self.L):
            mu, v_fin, h, o, mu_hist, t_prime, v_hist = JAXTwoCompartmentalLayer.forward_pass(
                inp, dend[ell], soma[ell], self.config, T, self.layers[ell].T_p
            )
            layer_outputs.append((mu, v_fin, h, o, mu_hist, t_prime, v_hist))
            inp = o
        readout_v, readout_o = JAXLIFLayer.forward_pass(inp, w_readout, self.config, T)
        return (layer_outputs, readout_v, readout_o)

    def _loss_impl(self, params: Tuple, x_input: jnp.ndarray, target: int) -> jnp.ndarray:
        _, readout_v, readout_o = self._forward_with_params(params, x_input)
        readout_counts = jnp.sum(readout_o, axis=0)
        scaled = readout_counts / self.loss_temperature + self.loss_count_bias
        exp_counts = jnp.exp(scaled - jnp.max(scaled))
        probs = exp_counts / jnp.sum(exp_counts)
        target_oh = jnp.zeros(self.n_outputs).at[target].set(1.0)
        target_oh = target_oh * (1 - self.loss_label_smoothing) + self.loss_label_smoothing / self.n_outputs
        return -jnp.sum(target_oh * jnp.log(probs + 1e-8))

    def compute_global_errors(self, readout_o: jnp.ndarray, target: int) -> jnp.ndarray:
        readout_counts = jnp.sum(readout_o, axis=0)
        scaled = readout_counts / self.loss_temperature + self.loss_count_bias
        exp_counts = jnp.exp(scaled - jnp.max(scaled))
        probs = exp_counts / jnp.sum(exp_counts)
        target_oh = jnp.zeros(self.n_outputs).at[target].set(1.0)
        target_oh = target_oh * (1 - self.loss_label_smoothing) + self.loss_label_smoothing / self.n_outputs
        return target_oh - probs

    def _compute_gradients(self, params: Tuple, x_input: jnp.ndarray, target: int,
                           clip_value: float = 5.0) -> Tuple:
        """Compute gradients for all layers (no weight update)."""
        layer_outputs, readout_v, readout_o = self._forward_with_params(params, x_input)
        dend, soma, w_readout = params
        T = x_input.shape[0]
        gamma = self.config.gamma

        global_errors = self.compute_global_errors(readout_o, target)

        # Readout gradients
        E_readout = JAXTwoCompartmentalLayer.compute_eligibility_traces(
            layer_outputs[self.L - 1][3], self.readout_layer.alpha
        )
        sigma_prime_readout = JAXTwoCompartmentalLayer.surrogate_sigma(
            readout_v - self.config.v_th, self.config.beta_s
        )
        grad_readout = jnp.clip(
            jnp.einsum('ti,tj,i->ij', sigma_prime_readout, E_readout, global_errors) / T,
            -clip_value, clip_value
        )

        # Effective error for top layer (index L-1)
        effective_top = jnp.einsum('tj,j,ji->ti', sigma_prime_readout, global_errors, w_readout)

        # Precompute per-layer quantities for all layers
        sigma_primes = []
        h_primes = []
        E_somas = []
        dmu_dws = []
        for ell in range(self.L):
            mu_ell, v_fin_ell, h_ell, o_ell, mu_hist_ell, t_prime_ell, v_hist_ell = layer_outputs[ell]
            inp_ell = x_input if ell == 0 else layer_outputs[ell - 1][3]
            soma_in_ell = v_hist_ell + gamma * h_ell - self.config.v_th
            sigma_primes.append(
                JAXTwoCompartmentalLayer.surrogate_sigma(soma_in_ell, self.config.beta_s)
            )
            t_prime_int = t_prime_ell.astype(jnp.int32)
            n_ell = self.layer_sizes[ell]
            mu_at_tprime = mu_hist_ell[t_prime_int, jnp.arange(n_ell)]
            h_primes.append(
                JAXTwoCompartmentalLayer.surrogate_sigma(mu_at_tprime - self.config.mu_th, self.config.beta_d)
            )
            E_somas.append(
                JAXTwoCompartmentalLayer.compute_eligibility_traces(inp_ell, self.layers[ell].alpha_s)
            )
            dmu_dws.append(
                JAXTwoCompartmentalLayer.compute_dmu_tprime_dw(
                    inp_ell, h_ell, t_prime_ell, self.layers[ell].alpha
                )
            )

        grad_dend_list = []
        grad_soma_list = []
        effective_curr = effective_top

        # Top layer (L-1) gradients
        ell = self.L - 1
        grad_soma_ell = jnp.einsum('ti,tj,ti->ij', sigma_primes[ell], E_somas[ell], effective_curr) / T
        grad_dend_ell = jnp.einsum('ti,ti,tij,ti->ij', sigma_primes[ell], h_primes[ell], dmu_dws[ell],
                                    effective_curr * gamma) / T
        grad_soma_ell = jnp.clip(grad_soma_ell, -clip_value, clip_value)
        grad_dend_ell = jnp.clip(grad_dend_ell, -clip_value, clip_value)
        grad_soma_list.append(grad_soma_ell)
        grad_dend_list.append(grad_dend_ell)

        # Layers L-2 down to 0: effective error from layer above (soma + dend path), then local e-prop
        for ell in range(self.L - 2, -1, -1):
            ell_next = ell + 1
            n_ell = self.layer_sizes[ell]
            n_next = self.layer_sizes[ell_next]
            t_prime_next_int = layer_outputs[ell_next][5].astype(jnp.int32)

            # Soma path: e_ell_soma(t) = W_soma_{ell+1}^T (sigma'_{ell+1}(t) * e_{ell+1}(t))
            e_ell_soma = jnp.einsum('ik,ti,ti->tk', soma[ell_next], sigma_primes[ell_next], effective_curr)
            # e_ell_soma is (T, n_ell) but we used dend[ell_next] which is (n_next, n_ell). So W^T (s'*e) = (n_ell,) per t. So shape (T, n_ell). Good.
            # Actually: dend[ell_next] = w_dend for layer ell_next, shape (n_next, n_ell). So einsum('ik,ti,ti->tk') gives (T, n_ell). Good.

            # Soma-path gradient (all t)
            grad_soma_ell = jnp.einsum('ti,tj,ti->ij', sigma_primes[ell], E_somas[ell], e_ell_soma) / T

            # Dend path: coefficient at t'_{ell+1}; then multiply by E_ell(t'_{ell+1}) and dmu_ell(t'_{ell+1})
            sigma_ell_at_tprime_next = sigma_primes[ell][t_prime_next_int]
            E_ell_at_tprime_next = E_somas[ell][t_prime_next_int]
            h_ell_at_tprime_next = h_primes[ell][t_prime_next_int]
            dmu_ell_at_tprime_next = dmu_dws[ell][t_prime_next_int]

            # Dend path at t'_{ell+1}: W_dend_{ell+1} * sigma'_{ell+1} * h'_{ell+1} * sigma'_ell(t') * e_{ell+1}
            term_p2 = (dend[ell_next] * sigma_primes[ell_next][:, :, None] * h_primes[ell_next][:, :, None]
                       * sigma_ell_at_tprime_next * effective_curr[:, :, None])
            grad_soma_ell = grad_soma_ell + (gamma / T) * jnp.einsum('tik,tij->kj', term_p2, E_ell_at_tprime_next)

            coeff_dend_p1 = jnp.einsum('ik,ti,tk,ti->tk', soma[ell_next], sigma_primes[ell_next],
                                       sigma_primes[ell] * h_primes[ell], effective_curr)
            grad_dend_ell = jnp.einsum('tk,tkj->kj', coeff_dend_p1 * gamma, dmu_dws[ell]) / T
            term_d2 = (dend[ell_next] * sigma_primes[ell_next][:, :, None] * h_primes[ell_next][:, :, None]
                       * sigma_ell_at_tprime_next * h_ell_at_tprime_next * effective_curr[:, :, None] * gamma)
            grad_dend_ell = grad_dend_ell + (gamma / T) * jnp.einsum('tik,tikj->kj', term_d2, dmu_ell_at_tprime_next)

            grad_soma_ell = jnp.clip(grad_soma_ell, -clip_value, clip_value)
            grad_dend_ell = jnp.clip(grad_dend_ell, -clip_value, clip_value)
            grad_soma_list.append(grad_soma_ell)
            grad_dend_list.append(grad_dend_ell)

            # Effective error for next (lower) layer: soma path only (dend path is already in gradients)
            effective_curr = e_ell_soma

        grad_soma_list.reverse()
        grad_dend_list.reverse()
        loss = self._loss_impl(params, x_input, target)
        return (tuple(grad_dend_list), tuple(grad_soma_list), grad_readout, loss, layer_outputs, readout_o)

    def _train_step_impl(self, params: Tuple, x_input: jnp.ndarray, target: int,
                         learning_rates_dend: jnp.ndarray, learning_rates_soma: jnp.ndarray,
                         lr_readout: float, clip_value: float = 5.0) -> Tuple:
        grad_dend_list, grad_soma_list, grad_readout, loss, layer_outputs, readout_o = self._compute_gradients(
            params, x_input, target, clip_value
        )
        dend, soma, w_readout = params
        wd = self.weight_decay
        new_dend = tuple(
            jnp.clip(dend[ell] * (1 - wd) + learning_rates_dend[ell] * grad_dend_list[ell], -1.0, 1.0)
            for ell in range(self.L)
        )
        new_soma = tuple(
            jnp.clip(soma[ell] * (1 - wd) + learning_rates_soma[ell] * grad_soma_list[ell], -1.0, 1.0)
            for ell in range(self.L)
        )
        new_readout = jnp.clip(w_readout * (1 - wd) + lr_readout * grad_readout, -1.0, 1.0)
        new_params = _make_params_struct(new_dend, new_soma, new_readout)
        return new_params, loss, layer_outputs, readout_o

    def get_balanced_learning_rates(self) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        if len(self.activity_history) < 10:
            return (jnp.array(self.learning_rates_dendritic), jnp.array(self.learning_rates_somatic),
                    self.learning_rate_readout)
        recent = np.array(self.activity_history[-10:])
        avg_act = np.mean(recent)
        std_act = np.std(recent)
        lr_r = self.learning_rate_readout * 0.5 if std_act > avg_act * 0.5 else self.learning_rate_readout
        return (jnp.array(self.learning_rates_dendritic), jnp.array(self.learning_rates_somatic), lr_r)

    def train_step(self, x_input: jnp.ndarray, target: int) -> Tuple[float, jnp.ndarray]:
        params = self.get_params()
        lr_d, lr_s, lr_r = self.get_balanced_learning_rates()
        new_params, loss, layer_outputs, readout_o = self._train_step_compiled(
            params, x_input, target, lr_d, lr_s, lr_r, self.gradient_clip
        )
        self.set_params(new_params)
        self.activity_history.append(np.array(jnp.sum(readout_o, axis=0)))
        if len(self.activity_history) > 50:
            self.activity_history = self.activity_history[-50:]
        return float(loss), readout_o

    def _train_step_compiled(self, params: Tuple, x_input: jnp.ndarray, target: int,
                            lr_d: jnp.ndarray, lr_s: jnp.ndarray, lr_r: float, clip_value: float):
        return self._train_step_impl(params, x_input, target, lr_d, lr_s, lr_r, clip_value)

    def forward(self, x_input: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if not isinstance(x_input, jnp.ndarray):
            x_input = jnp.array(x_input)
        params = self.get_params()
        _, readout_v, readout_o = self._forward_with_params(params, x_input)
        return readout_v, readout_o


# JIT-compiled train step (learning rates as arrays so no recompile per LR)
JAXEPropNetworkNLayer._train_step_compiled = jit(JAXEPropNetworkNLayer._train_step_impl)


def train_network_n_layer(
    net: JAXEPropNetworkNLayer,
    train_inputs: jnp.ndarray,
    train_targets: jnp.ndarray,
    epochs: int,
    batch_size: int = 1,
) -> List[float]:
    """Train the N-layer network; returns list of mean loss per epoch."""
    epoch_losses = []
    n_train = train_inputs.shape[0]
    for ep in range(epochs):
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            x_batch = train_inputs[idx]
            y_batch = train_targets[idx]
            for b in range(x_batch.shape[0]):
                loss, _ = net.train_step(x_batch[b], int(y_batch[b]))
                epoch_loss += loss
                n_batches += 1
        epoch_losses.append(epoch_loss / max(n_batches, 1))
    return epoch_losses
