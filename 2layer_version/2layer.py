"""
JAX two-layer e-prop network: same as 2comp_uniform (SHD, batching, uniform plateau)
plus a trainable extra 2-comp layer (input → extra 2comp → hidden 2comp → LIF readout).

Model matches the NumPy 2layer design but implemented in JAX for speed.

NMNIST (same setup as Numpy/run_2layer.py):
  DATASET=nmnist python 2layer.py
  Uses same arch (n_extra=10, n_hidden=10, n_outputs=10, T=300), data (all train/test),
  epochs=3, and hyperparameters aligned with two_layer_numpy (LRs, loss temp 5.0, bias 0.1, smoothing 0.2).
  Remaining differences: initial weights (JAX vs NumPy RNG), save paths (Jax/model/<timestamp>/).
"""
import os
import sys

# CRITICAL: Set TMPDIR before importing JAX (avoids "No space left on device" on HPC)
# JAX uses TMPDIR for CUDA/compilation cache; /tmp is often full on clusters
_current_tmpdir = os.environ.get("TMPDIR", "/tmp")
if _current_tmpdir == "/tmp" or not os.path.exists(_current_tmpdir):
    import getpass
    _username = getpass.getuser()
    if "SLURM_TMPDIR" in os.environ:
        _new_tmpdir = os.environ["SLURM_TMPDIR"]
    elif "SCRATCH" in os.environ:
        _new_tmpdir = os.path.join(os.environ["SCRATCH"], f"tmp_{_username}")
    elif os.path.exists("/scratch"):
        _new_tmpdir = os.path.join("/scratch", f"tmp_{_username}")
    else:
        _new_tmpdir = os.path.expanduser("~/tmp")
    try:
        os.makedirs(_new_tmpdir, exist_ok=True)
        _test = os.path.join(_new_tmpdir, ".jax_write_test")
        with open(_test, "w") as _f:
            _f.write("test")
        os.remove(_test)
        os.environ["TMPDIR"] = _new_tmpdir
    except (OSError, PermissionError):
        _fallback = os.path.expanduser("~/tmp")
        os.makedirs(_fallback, exist_ok=True)
        os.environ["TMPDIR"] = _fallback

import time
import pickle
import json
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap, lax
from typing import List, Tuple, Dict, Optional
from flax import struct

# Load 2comp_uniform from same dir as this script (or parent) so imports work regardless of cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
_COMP_UNIFORM_PATH = os.path.join(_SCRIPT_DIR, "2comp_uniform.py")
if not os.path.isfile(_COMP_UNIFORM_PATH):
    _COMP_UNIFORM_PATH = os.path.join(_PARENT_DIR, "2comp_uniform.py")
if not os.path.isfile(_COMP_UNIFORM_PATH):
    raise FileNotFoundError(
        f"2comp_uniform.py not found in {_SCRIPT_DIR!r} or parent. "
        f"Ensure 2comp_uniform.py is next to this script or in the parent directory."
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


@struct.dataclass
class NetworkParamsTwoLayer:
    """Parameters for two-layer network: extra 2comp + hidden 2comp + readout."""
    w_dend_extra: jnp.ndarray   # (n_extra, n_inputs)
    w_soma_extra: jnp.ndarray   # (n_extra, n_inputs)
    w_dend_hidden: jnp.ndarray  # (n_hidden, n_extra)
    w_soma_hidden: jnp.ndarray  # (n_hidden, n_extra)
    w_readout: jnp.ndarray      # (n_outputs, n_hidden)


class JAXEPropNetworkTwoLayer:
    """
    E-prop network with trainable extra 2comp layer (same architecture as NumPy 2layer_jax.py).
    input (n_inputs) → extra 2comp (n_extra) → hidden 2comp (n_hidden) → LIF readout (n_outputs).
    Uses same NeuronConfig, uniform T_p, batching, and time-dependent effective error as 2comp_uniform.
    """

    def __init__(self, key, n_inputs: int = 700, n_extra: int = 10, n_hidden: int = 64,
                 n_outputs: int = 20, T: int = 700,
                 learning_rate_extra_dendritic: Optional[float] = None,
                 learning_rate_extra_soma: Optional[float] = None,
                 learning_rate_hidden_dendritic: Optional[float] = None,
                 learning_rate_hidden_somatic: Optional[float] = None,
                 learning_rate_readout: Optional[float] = None,
                 weight_decay: Optional[float] = None,
                 gradient_clip: Optional[float] = None,
                 loss_temperature: Optional[float] = None,
                 loss_count_bias: Optional[float] = None,
                 loss_label_smoothing: Optional[float] = None):
        self.n_inputs = n_inputs
        self.n_extra = n_extra
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.T = T

        key_extra, key_hidden, key_readout = random.split(key, 3)
        self.config = NeuronConfig()

        self.extra_layer = JAXTwoCompartmentalLayer(key_extra, n_extra, n_inputs, self.config)
        self.hidden_layer = JAXTwoCompartmentalLayer(key_hidden, n_hidden, n_extra, self.config)
        self.readout_layer = JAXLIFLayer(key_readout, n_outputs, n_hidden, self.config)

        # Match NumPy two_layer_numpy.py defaults (NMNIST / run_2layer.py); overridable by caller
        self.learning_rate_extra_dendritic = 0.05
        self.learning_rate_extra_soma = 0.0025
        self.learning_rate_hidden_dendritic = 0.05
        self.learning_rate_hidden_somatic = 0.0025
        self.learning_rate_readout = 0.025
        self.weight_decay = 0.00001
        self.gradient_clip = 5.0
        self.loss_temperature = 5.0
        self.loss_count_bias = 0.1
        self.loss_label_smoothing = 0.2
        if learning_rate_extra_dendritic is not None:
            self.learning_rate_extra_dendritic = learning_rate_extra_dendritic
        if learning_rate_extra_soma is not None:
            self.learning_rate_extra_soma = learning_rate_extra_soma
        if learning_rate_hidden_dendritic is not None:
            self.learning_rate_hidden_dendritic = learning_rate_hidden_dendritic
        if learning_rate_hidden_somatic is not None:
            self.learning_rate_hidden_somatic = learning_rate_hidden_somatic
        if learning_rate_readout is not None:
            self.learning_rate_readout = learning_rate_readout
        if weight_decay is not None:
            self.weight_decay = weight_decay
        if gradient_clip is not None:
            self.gradient_clip = gradient_clip
        if loss_temperature is not None:
            self.loss_temperature = loss_temperature
        if loss_count_bias is not None:
            self.loss_count_bias = loss_count_bias
        if loss_label_smoothing is not None:
            self.loss_label_smoothing = loss_label_smoothing

        self.activity_history: List = []
        self.debug_stats = {
            'epoch_losses': [], 'epoch_accuracies': [],
            'hidden_spike_rates': [], 'readout_spike_rates': [],
            'plateau_activity': [],
            'confusion_matrix': np.zeros((n_outputs, n_outputs))
        }

        self._compile_functions()

    def _compile_functions(self):
        self._loss_compiled = jit(self._loss_impl)
        self._train_step_compiled = jit(self._train_step_impl, static_argnums=(2,))

    def get_params(self) -> NetworkParamsTwoLayer:
        return NetworkParamsTwoLayer(
            w_dend_extra=self.extra_layer.w_dend,
            w_soma_extra=self.extra_layer.w_soma,
            w_dend_hidden=self.hidden_layer.w_dend,
            w_soma_hidden=self.hidden_layer.w_soma,
            w_readout=self.readout_layer.w
        )

    def set_params(self, params: NetworkParamsTwoLayer):
        self.extra_layer.w_dend = params.w_dend_extra
        self.extra_layer.w_soma = params.w_soma_extra
        self.hidden_layer.w_dend = params.w_dend_hidden
        self.hidden_layer.w_soma = params.w_soma_hidden
        self.readout_layer.w = params.w_readout

    def _forward_with_params(self, params: NetworkParamsTwoLayer, x_input: jnp.ndarray) -> Tuple:
        """Forward: x → extra 2comp → hidden 2comp → readout."""
        T_p_extra = self.extra_layer.T_p
        T_p_hidden = self.hidden_layer.T_p

        mu_e, v_e, h_e, extra_o, mu_history_e, t_prime_e, v_history_e = JAXTwoCompartmentalLayer.forward_pass(
            x_input, params.w_dend_extra, params.w_soma_extra,
            self.config, self.T, T_p_extra
        )
        mu_h, v_h, h_h, hidden_o, mu_history_h, t_prime_h, v_history_h = JAXTwoCompartmentalLayer.forward_pass(
            extra_o, params.w_dend_hidden, params.w_soma_hidden,
            self.config, self.T, T_p_hidden
        )
        readout_v, readout_o = JAXLIFLayer.forward_pass(
            hidden_o, params.w_readout, self.config, self.T
        )
        return (mu_e, v_e, h_e, extra_o, mu_history_e, t_prime_e, v_history_e,
                mu_h, v_h, h_h, hidden_o, mu_history_h, t_prime_h, v_history_h,
                readout_v, readout_o)

    @staticmethod
    @jit(static_argnames=['T'])
    def _forward_with_params_jit(params: NetworkParamsTwoLayer, x_input: jnp.ndarray,
                                  config: NeuronConfig, T: int,
                                  T_p_extra: jnp.ndarray, T_p_hidden: jnp.ndarray) -> Tuple:
        mu_e, v_e, h_e, extra_o, mu_history_e, t_prime_e, v_history_e = JAXTwoCompartmentalLayer.forward_pass(
            x_input, params.w_dend_extra, params.w_soma_extra, config, T, T_p_extra
        )
        mu_h, v_h, h_h, hidden_o, mu_history_h, t_prime_h, v_history_h = JAXTwoCompartmentalLayer.forward_pass(
            extra_o, params.w_dend_hidden, params.w_soma_hidden, config, T, T_p_hidden
        )
        readout_v, readout_o = JAXLIFLayer.forward_pass(
            hidden_o, params.w_readout, config, T
        )
        return (mu_e, v_e, h_e, extra_o, mu_history_e, t_prime_e, v_history_e,
                mu_h, v_h, h_h, hidden_o, mu_history_h, t_prime_h, v_history_h,
                readout_v, readout_o)

    def forward(self, x_input: jnp.ndarray) -> Tuple:
        if not isinstance(x_input, jnp.ndarray):
            x_input = jnp.array(x_input)
        params = self.get_params()
        return self._forward_with_params_jit(
            params, x_input, self.config, self.T,
            self.extra_layer.T_p, self.hidden_layer.T_p
        )

    def _loss_impl(self, params: NetworkParamsTwoLayer, x_input: jnp.ndarray, target: int) -> jnp.ndarray:
        *_, readout_o = self._forward_with_params(params, x_input)
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

    def get_balanced_learning_rates(self) -> Tuple[float, float, float, float, float]:
        if len(self.activity_history) < 10:
            return (self.learning_rate_extra_dendritic, self.learning_rate_extra_soma,
                    self.learning_rate_hidden_dendritic, self.learning_rate_hidden_somatic,
                    self.learning_rate_readout)
        recent = np.array(self.activity_history[-10:])
        avg_act = np.mean(recent, axis=0)
        std, mean = np.std(avg_act), np.mean(avg_act)
        lr_readout = self.learning_rate_readout * 0.5 if std > mean * 0.5 else self.learning_rate_readout
        return (self.learning_rate_extra_dendritic, self.learning_rate_extra_soma,
                self.learning_rate_hidden_dendritic, self.learning_rate_hidden_somatic,
                lr_readout)

    def _compute_gradients(self, params: NetworkParamsTwoLayer, x_input: jnp.ndarray, target: int,
                           clip_value: float = 5.0) -> Tuple:
        """Compute gradients for a single sample (no weight update). Used for batch gradient averaging."""
        (mu_e, v_e, h_e, extra_o, mu_history_e, t_prime_e, v_history_e,
         mu_h, v_h, h_h, hidden_o, mu_history_h, t_prime_h, v_history_h,
         readout_v, readout_o) = self._forward_with_params(params, x_input)

        global_errors = self.compute_global_errors(readout_o, target)
        T = x_input.shape[0]
        n_hidden = self.n_hidden
        n_extra = self.n_extra
        n_inputs = self.n_inputs

        # === READOUT GRADIENTS (same as 2comp_uniform) ===
        E_readout = JAXTwoCompartmentalLayer.compute_eligibility_traces(hidden_o, self.readout_layer.alpha)
        sigma_prime_readout = JAXTwoCompartmentalLayer.surrogate_sigma(readout_v - self.config.v_th, self.config.beta_s)
        grad_readout_raw = jnp.einsum('ti,tj,i->ij', sigma_prime_readout, E_readout, global_errors) / self.T
        grad_readout = jnp.clip(grad_readout_raw, -clip_value, clip_value)

        # === HIDDEN LAYER (input = extra_o; same as 2comp_uniform with effective_error from readout) ===
        effective_error_hidden = jnp.einsum('tj,j,ji->ti', sigma_prime_readout, global_errors, params.w_readout)
        soma_input_h = v_history_h + self.config.gamma * h_h - self.config.v_th  # full trajectory v(t), not final v
        sigma_prime_hidden = JAXTwoCompartmentalLayer.surrogate_sigma(soma_input_h, self.config.beta_s)
        t_prime_h_int = t_prime_h.astype(jnp.int32)
        neuron_idx_h = jnp.arange(n_hidden)[None, :]
        mu_at_tprime_h = mu_history_h[t_prime_h_int, neuron_idx_h]
        h_prime_hidden = JAXTwoCompartmentalLayer.surrogate_sigma(mu_at_tprime_h - self.config.mu_th, self.config.beta_d)
        E_soma_hidden = JAXTwoCompartmentalLayer.compute_eligibility_traces(extra_o, self.hidden_layer.alpha_s)
        dmu_dw_hidden = JAXTwoCompartmentalLayer.compute_dmu_tprime_dw(
            extra_o, h_h, t_prime_h, self.hidden_layer.alpha
        )
        grad_soma_hidden = jnp.einsum('ti,tj,ti->ij', sigma_prime_hidden, E_soma_hidden, effective_error_hidden) / self.T
        grad_dend_hidden = jnp.einsum('ti,ti,tij,ti->ij', sigma_prime_hidden, h_prime_hidden, dmu_dw_hidden,
                                      effective_error_hidden * self.config.gamma) / self.T
        grad_soma_hidden = jnp.clip(grad_soma_hidden, -clip_value, clip_value)
        grad_dend_hidden = jnp.clip(grad_dend_hidden, -clip_value, clip_value)

        # === EXTRA LAYER (two-path gradient from hidden, matching NumPy 2layer_jax) ===
        soma_input_e = v_history_e + self.config.gamma * h_e - self.config.v_th  # full trajectory v(t), not final v
        sigma_prime_extra = JAXTwoCompartmentalLayer.surrogate_sigma(soma_input_e, self.config.beta_s)
        t_prime_e_int = t_prime_e.astype(jnp.int32)
        neuron_idx_e = jnp.arange(n_extra)[None, :]
        mu_at_tprime_e = mu_history_e[t_prime_e_int, neuron_idx_e]
        h_prime_extra = JAXTwoCompartmentalLayer.surrogate_sigma(mu_at_tprime_e - self.config.mu_th, self.config.beta_d)
        E_soma_extra = JAXTwoCompartmentalLayer.compute_eligibility_traces(x_input, self.extra_layer.alpha_s)
        dmu_dw_extra = JAXTwoCompartmentalLayer.compute_dmu_tprime_dw(
            x_input, h_e, t_prime_e, self.extra_layer.alpha
        )

        coeff_extra_soma_p1 = jnp.einsum('ik,ti,tk,ti->tk',
                                         params.w_soma_hidden, sigma_prime_hidden, sigma_prime_extra, effective_error_hidden)
        grad_extra_soma = jnp.einsum('tk,tj->kj', coeff_extra_soma_p1, E_soma_extra) / self.T

        sigma_extra_at_tprime_h = sigma_prime_extra[t_prime_h_int]
        E_extra_at_tprime_h = E_soma_extra[t_prime_h_int]
        term_p2 = (params.w_dend_hidden * sigma_prime_hidden[:, :, None] * h_prime_hidden[:, :, None]
                   * sigma_extra_at_tprime_h * effective_error_hidden[:, :, None])
        grad_extra_soma = grad_extra_soma + (self.config.gamma / self.T) * jnp.einsum('tik,tij->kj', term_p2, E_extra_at_tprime_h)

        coeff_extra_dend_p1 = jnp.einsum('ik,ti,tk,ti->tk',
                                         params.w_soma_hidden, sigma_prime_hidden, sigma_prime_extra * h_prime_extra, effective_error_hidden)
        grad_extra_dend = jnp.einsum('tk,tkj->kj', coeff_extra_dend_p1 * self.config.gamma, dmu_dw_extra) / self.T

        h_extra_at_tprime_h = h_prime_extra[t_prime_h_int]
        dmu_extra_at_tprime_h = dmu_dw_extra[t_prime_h_int]
        term_d2 = (params.w_dend_hidden * sigma_prime_hidden[:, :, None] * h_prime_hidden[:, :, None]
                   * sigma_extra_at_tprime_h * h_extra_at_tprime_h * effective_error_hidden[:, :, None] * self.config.gamma)
        grad_extra_dend = grad_extra_dend + (self.config.gamma / self.T) * jnp.einsum('tik,tikj->kj', term_d2, dmu_extra_at_tprime_h)

        grad_extra_soma = jnp.clip(grad_extra_soma, -clip_value, clip_value)
        grad_extra_dend = jnp.clip(grad_extra_dend, -clip_value, clip_value)

        loss = self._loss_impl(params, x_input, target)
        return (grad_extra_dend, grad_extra_soma, grad_dend_hidden, grad_soma_hidden, grad_readout,
                loss, hidden_o, readout_o, h_e, h_h)

    def _train_step_impl(self, params: NetworkParamsTwoLayer, x_input: jnp.ndarray, target: int,
                         lr_ed: float, lr_es: float, lr_hd: float, lr_hs: float, lr_r: float,
                         clip_value: float = 5.0) -> Tuple:
        (grad_extra_dend, grad_extra_soma, grad_dend_hidden, grad_soma_hidden, grad_readout,
         loss, hidden_o, readout_o, h_extra, h_hidden) = self._compute_gradients(params, x_input, target, clip_value)

        # === WEIGHT UPDATES ===
        new_w_dend_extra = jnp.clip(params.w_dend_extra * (1 - self.weight_decay) + lr_ed * grad_extra_dend, -1.0, 1.0)
        new_w_soma_extra = jnp.clip(params.w_soma_extra * (1 - self.weight_decay) + lr_es * grad_extra_soma, -1.0, 1.0)
        new_w_dend_hidden = jnp.clip(params.w_dend_hidden * (1 - self.weight_decay) + lr_hd * grad_dend_hidden, -1.0, 1.0)
        new_w_soma_hidden = jnp.clip(params.w_soma_hidden * (1 - self.weight_decay) + lr_hs * grad_soma_hidden, -1.0, 1.0)
        new_w_readout = jnp.clip(params.w_readout * (1 - self.weight_decay) + lr_r * grad_readout, -1.0, 1.0)

        new_params = NetworkParamsTwoLayer(
            w_dend_extra=new_w_dend_extra, w_soma_extra=new_w_soma_extra,
            w_dend_hidden=new_w_dend_hidden, w_soma_hidden=new_w_soma_hidden,
            w_readout=new_w_readout
        )
        return new_params, loss, hidden_o, readout_o, h_extra, h_hidden

    def train_step(self, x_input: jnp.ndarray, target: int) -> Tuple[float, jnp.ndarray, jnp.ndarray, int]:
        params = self.get_params()
        lr_ed, lr_es, lr_hd, lr_hs, lr_r = self.get_balanced_learning_rates()
        new_params, loss, hidden_o, readout_o, h_extra, h_hidden = self._train_step_compiled(
            params, x_input, target, lr_ed, lr_es, lr_hd, lr_hs, lr_r, self.gradient_clip
        )
        self.set_params(new_params)
        self.activity_history.append(np.array(jnp.sum(readout_o, axis=0)))
        if len(self.activity_history) > 50:
            self.activity_history = self.activity_history[-50:]
        plateau_total = int(jnp.sum(h_extra) + jnp.sum(h_hidden))
        return float(loss), hidden_o, readout_o, plateau_total

    def train_step_batch(self, x_batch: jnp.ndarray, target_batch: jnp.ndarray) -> Tuple[float, List, List]:
        """Batch training step: compute gradients per sample (same params), average gradients, one weight update.
        Uses a loop over batch elements to avoid OOM (vmap would materialize full batch of T x n x n_inputs tensors)."""
        params = self.get_params()
        lr_ed, lr_es, lr_hd, lr_hs, lr_r = self.get_balanced_learning_rates()
        clip_value = self.gradient_clip
        batch_size = x_batch.shape[0]

        # Accumulate gradients over batch (one sample at a time to limit memory)
        grad_ed_sum = jnp.zeros_like(params.w_dend_extra)
        grad_es_sum = jnp.zeros_like(params.w_soma_extra)
        grad_hd_sum = jnp.zeros_like(params.w_dend_hidden)
        grad_hs_sum = jnp.zeros_like(params.w_soma_hidden)
        grad_r_sum = jnp.zeros_like(params.w_readout)
        loss_sum = 0.0
        hidden_os_list = []
        readout_os_list = []

        for i in range(batch_size):
            (g_ed, g_es, g_hd, g_hs, g_r, loss, h_o, r_o, _, _) = self._compute_gradients(
                params, x_batch[i], target_batch[i], clip_value
            )
            grad_ed_sum = grad_ed_sum + g_ed
            grad_es_sum = grad_es_sum + g_es
            grad_hd_sum = grad_hd_sum + g_hd
            grad_hs_sum = grad_hs_sum + g_hs
            grad_r_sum = grad_r_sum + g_r
            loss_sum = loss_sum + loss
            hidden_os_list.append(h_o)
            readout_os_list.append(r_o)

        # Average gradients across batch
        grad_ed_avg = grad_ed_sum / batch_size
        grad_es_avg = grad_es_sum / batch_size
        grad_hd_avg = grad_hd_sum / batch_size
        grad_hs_avg = grad_hs_sum / batch_size
        grad_r_avg = grad_r_sum / batch_size

        # Clip averaged gradients
        grad_ed_clipped = jnp.clip(grad_ed_avg, -clip_value, clip_value)
        grad_es_clipped = jnp.clip(grad_es_avg, -clip_value, clip_value)
        grad_hd_clipped = jnp.clip(grad_hd_avg, -clip_value, clip_value)
        grad_hs_clipped = jnp.clip(grad_hs_avg, -clip_value, clip_value)
        grad_r_clipped = jnp.clip(grad_r_avg, -clip_value, clip_value)

        # One weight update with averaged gradients
        new_w_dend_extra = jnp.clip(params.w_dend_extra * (1 - self.weight_decay) + lr_ed * grad_ed_clipped, -1.0, 1.0)
        new_w_soma_extra = jnp.clip(params.w_soma_extra * (1 - self.weight_decay) + lr_es * grad_es_clipped, -1.0, 1.0)
        new_w_dend_hidden = jnp.clip(params.w_dend_hidden * (1 - self.weight_decay) + lr_hd * grad_hd_clipped, -1.0, 1.0)
        new_w_soma_hidden = jnp.clip(params.w_soma_hidden * (1 - self.weight_decay) + lr_hs * grad_hs_clipped, -1.0, 1.0)
        new_w_readout = jnp.clip(params.w_readout * (1 - self.weight_decay) + lr_r * grad_r_clipped, -1.0, 1.0)

        new_params = NetworkParamsTwoLayer(
            w_dend_extra=new_w_dend_extra, w_soma_extra=new_w_soma_extra,
            w_dend_hidden=new_w_dend_hidden, w_soma_hidden=new_w_soma_hidden,
            w_readout=new_w_readout
        )
        self.set_params(new_params)

        # Activity history = mean over batch (same as before)
        readout_activity_avg = np.array(jnp.mean(jnp.stack([jnp.sum(r, axis=0) for r in readout_os_list]), axis=0))
        self.activity_history.append(readout_activity_avg)
        if len(self.activity_history) > 50:
            self.activity_history = self.activity_history[-50:]

        avg_loss = float(loss_sum / batch_size)
        return avg_loss, hidden_os_list, readout_os_list

    def predict(self, x_input: jnp.ndarray) -> int:
        *_, readout_o = self.forward(x_input)
        counts = jnp.sum(readout_o, axis=0)
        scaled = counts / self.loss_temperature + self.loss_count_bias
        probs = jnp.exp(scaled - jnp.max(scaled)) / jnp.sum(jnp.exp(scaled - jnp.max(scaled)))
        return int(jnp.argmax(probs))

    def evaluate(self, test_data: List[Tuple]) -> Dict:
        correct = total = 0
        test_losses = []
        confusion = np.zeros((self.n_outputs, self.n_outputs))
        for x_raw, target in test_data:
            x = jnp.array(x_raw)
            *_, readout_o = self.forward(x)
            readout_counts = jnp.sum(readout_o, axis=0)
            scaled_counts = readout_counts / self.loss_temperature + self.loss_count_bias
            exp_counts = jnp.exp(scaled_counts - jnp.max(scaled_counts))
            probabilities = exp_counts / jnp.sum(exp_counts)
            target_one_hot = jnp.zeros(self.n_outputs).at[target].set(1.0)
            target_one_hot = target_one_hot * (1 - self.loss_label_smoothing) + self.loss_label_smoothing / self.n_outputs
            loss = float(-jnp.sum(target_one_hot * jnp.log(probabilities + 1e-8)))
            test_losses.append(loss)
            pred = int(jnp.argmax(probabilities))
            if pred == target:
                correct += 1
            total += 1
            confusion[target, pred] += 1
        acc = correct / total * 100 if total else 0.0
        avg_loss = float(np.mean(test_losses)) if test_losses else 0.0
        per_class = {}
        for i in range(self.n_outputs):
            ct = np.sum(confusion[i, :])
            if ct > 0:
                per_class[i] = confusion[i, i] / ct * 100
        return {'accuracy': acc, 'avg_loss': avg_loss, 'correct': correct, 'total': total, 'confusion_matrix': confusion, 'per_class_accuracy': per_class}

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'n_inputs': self.n_inputs, 'n_extra': self.n_extra, 'n_hidden': self.n_hidden,
                'n_outputs': self.n_outputs, 'T': self.T,
                'w_dend_extra': np.array(self.extra_layer.w_dend),
                'w_soma_extra': np.array(self.extra_layer.w_soma),
                'w_dend_hidden': np.array(self.hidden_layer.w_dend),
                'w_soma_hidden': np.array(self.hidden_layer.w_soma),
                'w_readout': np.array(self.readout_layer.w),
            }, f)
        print(f"Saved to {path}")

    @classmethod
    def load(cls, path: str, key=None):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        if key is None:
            key = random.PRNGKey(0)
        net = cls(key, d['n_inputs'], d['n_extra'], d['n_hidden'], d['n_outputs'], d['T'])
        net.extra_layer.w_dend = jnp.array(d['w_dend_extra'])
        net.extra_layer.w_soma = jnp.array(d['w_soma_extra'])
        net.hidden_layer.w_dend = jnp.array(d['w_dend_hidden'])
        net.hidden_layer.w_soma = jnp.array(d['w_soma_hidden'])
        net.readout_layer.w = jnp.array(d['w_readout'])
        print(f"Loaded from {path}")
        return net


def train_network_two_layer(network, train_data, test_data, run_dir, epochs, batch_size, model_name_prefix, random_seed=42):
    """Train two-layer network. train_data/test_data are list of (x, label) with x shape (T, n_inputs)."""
    n_train = len(train_data)
    temp_config = network.config
    hyperparams_lines = [
        "", "HYPERPARAMETERS (two-layer)", "=" * 80,
        f"Random seed: {random_seed}",
        "Architecture:",
        f"  n_inputs: {network.n_inputs}, n_extra: {network.n_extra}, n_hidden: {network.n_hidden}, n_outputs: {network.n_outputs}",
        "Training:",
        f"  Epochs: {epochs}, Batch size: {batch_size}",
        "Learning rates:",
        f"  extra dend: {network.learning_rate_extra_dendritic}, extra soma: {network.learning_rate_extra_soma}",
        f"  hidden dend: {network.learning_rate_hidden_dendritic}, hidden soma: {network.learning_rate_hidden_somatic}, readout: {network.learning_rate_readout}",
        f"  weight_decay: {network.weight_decay}, gradient_clip: {network.gradient_clip}",
        "Loss:",
        f"  temperature: {network.loss_temperature}, count_bias: {network.loss_count_bias}, label_smoothing: {network.loss_label_smoothing}",
        "=" * 80, ""
    ]
    with open(os.path.join(run_dir, "hyperparameters.txt"), "w") as f:
        f.write("\n".join(hyperparams_lines))

    best_accuracy = -1.0
    best_model_path = None
    epoch_results = []

    for epoch in range(epochs):
        np.random.seed(random_seed + epoch)
        np.random.shuffle(train_data)
        epoch_start = time.time()
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0
        n_batches = (n_train + batch_size - 1) // batch_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            batch_data = train_data[start_idx:end_idx]
            if batch_size == 1:
                x_arr, target = batch_data[0]
                x = jnp.asarray(x_arr)
                loss, hidden_o, readout_o, _ = network.train_step(x, target)
                epoch_losses.append(loss)
                pred = network.predict(x)
                if pred == target:
                    epoch_correct += 1
                epoch_total += 1
            else:
                batch_inputs = jnp.array([x for x, _ in batch_data])
                batch_targets = jnp.array([t for _, t in batch_data], dtype=jnp.int32)
                avg_loss, hidden_os_list, readout_os_list = network.train_step_batch(batch_inputs, batch_targets)
                for i in range(len(batch_data)):
                    epoch_losses.append(avg_loss)
                for i in range(batch_inputs.shape[0]):
                    pred = network.predict(batch_inputs[i])
                    if pred == batch_targets[i]:
                        epoch_correct += 1
                    epoch_total += 1
            idx = end_idx - 1
            if (idx + 1) % 5000 == 0 or idx == 0 or end_idx == n_train:
                elapsed = time.time() - epoch_start
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (n_train - idx - 1) / rate if rate > 0 else 0
                print(f"  Epoch {epoch+1}: {idx+1}/{n_train} samples ({100*(idx+1)/n_train:.0f}%) | "
                      f"{rate:.1f} samples/s | ETA this epoch: {eta:.0f}s", flush=True)
        epoch_avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        epoch_accuracy = epoch_correct / epoch_total * 100 if epoch_total else 0.0
        res = network.evaluate(test_data)
        epoch_results.append({
            "epoch": epoch + 1,
            "train_avg_loss": epoch_avg_loss,
            "train_accuracy": epoch_accuracy,
            "train_correct": epoch_correct,
            "train_total": epoch_total,
            "test_accuracy": res["accuracy"],
            "test_avg_loss": res["avg_loss"],
            "correct": res["correct"],
            "total": res["total"],
        })
        print(f"\n{'='*80}", flush=True)
        print(f"EPOCH {epoch+1}/{epochs} SUMMARY", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Train - Average Loss: {epoch_avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}% ({epoch_correct}/{epoch_total})", flush=True)
        print(f"Test - Average Loss: {res['avg_loss']:.4f}, Accuracy: {res['accuracy']:.2f}% ({res['correct']}/{res['total']})", flush=True)
        if res["accuracy"] > best_accuracy:
            best_accuracy = res["accuracy"]
            acc_str = f"{int(best_accuracy * 100) // 100:02d}_{int(best_accuracy * 100) % 100:02d}"
            best_model_path = os.path.join(run_dir, f"{model_name_prefix}_{acc_str}.pkl")
            network.save(best_model_path)
            print(f"  Saved best model to {best_model_path}", flush=True)

    final_res = network.evaluate(test_data)
    final_model_path = os.path.join(run_dir, f"{model_name_prefix}_jax_model.pkl")
    network.save(final_model_path)
    with open(os.path.join(run_dir, "training_summary.json"), "w") as f:
        json.dump({
            "final_test_accuracy": final_res["accuracy"],
            "final_test_loss": final_res["avg_loss"],
            "best_accuracy": best_accuracy,
            "best_model_path": best_model_path,
            "epoch_results": epoch_results,
        }, f, indent=2)
    return best_accuracy, best_model_path, final_model_path
