"""
Standalone 1-layer 2-compartment e-prop model. No dataset loading (SHD/NMNIST).
Expects train/test data as list of (x, label) with x shape (T, n_inputs).
Use run_shd.py or run_nmnist.py to load data and train.
"""
import sys
import time
from datetime import datetime
import os

# CRITICAL: Set TMPDIR before importing JAX to avoid "No space left on device" errors
# JAX uses TMPDIR for CUDA compilation cache files
# Override TMPDIR if it's set to /tmp (which is often full on HPC systems)
current_tmpdir = os.environ.get("TMPDIR", "/tmp")
if current_tmpdir == "/tmp" or not os.path.exists(current_tmpdir):
    # Try to use scratch space or job-specific temp directory
    # Use user-specific directory to avoid permission issues
    import getpass
    username = getpass.getuser()
    
    if "SLURM_TMPDIR" in os.environ:
        # SLURM_TMPDIR is job-specific and should have proper permissions
        new_tmpdir = os.environ["SLURM_TMPDIR"]
    elif "SCRATCH" in os.environ:
        # Use user-specific scratch directory
        new_tmpdir = os.path.join(os.environ["SCRATCH"], f"tmp_{username}")
    elif os.path.exists("/scratch"):
        # Use user-specific scratch directory
        new_tmpdir = os.path.join("/scratch", f"tmp_{username}")
    else:
        # Fallback: use user's home directory
        new_tmpdir = os.path.expanduser(f"~/tmp")
    
    # Create the directory if it doesn't exist (with user-specific permissions)
    try:
        os.makedirs(new_tmpdir, exist_ok=True)
        # Verify we can write to it
        test_file = os.path.join(new_tmpdir, ".jax_write_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        os.environ["TMPDIR"] = new_tmpdir
    except (OSError, PermissionError):
        # If we can't create/write to the directory, try user's home
        fallback_tmpdir = os.path.expanduser(f"~/tmp")
        os.makedirs(fallback_tmpdir, exist_ok=True)
        os.environ["TMPDIR"] = fallback_tmpdir

import jax
jax.config.update("jax_enable_x64", True)  # Consistent float64 across all scripts
import jax.numpy as jnp
import numpy as np
from jax import random, jit, grad, vmap, lax
from typing import List, Tuple, Dict, Optional
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flax import struct
import warnings
warnings.filterwarnings('ignore')

# Single source of truth: use the same layer as 2layer_version (and existing 2layer/2comp_uniform)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
_2LAYER_VER = os.path.join(_ROOT, "2layer_version")
if _2LAYER_VER not in sys.path:
    sys.path.insert(0, _2LAYER_VER)
import importlib.util
_spec_layer = importlib.util.spec_from_file_location(
    "_two_comp_layer", os.path.join(_2LAYER_VER, "2comp_uniform.py")
)
_layer_mod = importlib.util.module_from_spec(_spec_layer)
_spec_layer.loader.exec_module(_layer_mod)
NeuronConfig = _layer_mod.NeuronConfig
NetworkParams = _layer_mod.NetworkParams
JAXTwoCompartmentalLayer = _layer_mod.JAXTwoCompartmentalLayer
JAXLIFLayer = _layer_mod.JAXLIFLayer
alpha_kernel_jax = _layer_mod.alpha_kernel_jax
initialize_numpy_weights = _layer_mod.initialize_numpy_weights
RANDOM_SEED = _layer_mod.RANDOM_SEED
key = random.PRNGKey(RANDOM_SEED)

# (NeuronConfig, NetworkParams, JAXTwoCompartmentalLayer, JAXLIFLayer, alpha_kernel_jax, initialize_numpy_weights
#  are imported from 2layer_version/2comp_uniform.py above so 1-layer and 2-layer use the same layer.)

''' JAX E-PROP NETWORK (1-layer: hidden 2comp + readout) '''


class JAXEPropNetwork:
    """Complete JAX implementation of e-prop network"""
    
    def __init__(self, key, n_inputs: int = 700, n_hidden: int = 64, 
                 n_outputs: int = 20, T: int = 700,
                 learning_rate_hidden_dendritic=None,
                 learning_rate_hidden_somatic=None,
                 learning_rate_readout=None,
                 weight_decay=None,
                 gradient_clip=None,
                 loss_temperature=None,
                 loss_count_bias=None,
                 loss_label_smoothing=None,
                 neuron_config=None,
                 beta_s=None,
                 beta_d=None,
                 weight_scale=None):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.T = T
        
        # Split keys for different components
        key_hidden, key_readout = random.split(key)
        
        # Configuration: use provided config or build from defaults and overrides
        if neuron_config is not None:
            self.config = neuron_config
        else:
            self.config = NeuronConfig()
            if beta_s is not None:
                self.config = self.config.replace(beta_s=beta_s)
            if beta_d is not None:
                self.config = self.config.replace(beta_d=beta_d)
            if weight_scale is not None:
                self.config = self.config.replace(weight_scale=weight_scale)
        
        # Initialize layers
        self.hidden_layer = JAXTwoCompartmentalLayer(key_hidden, n_hidden, n_inputs, self.config)
        self.readout_layer = JAXLIFLayer(key_readout, n_outputs, n_hidden, self.config)
        
        # Learning parameters (overridable by caller)
        self.learning_rate_hidden_dendritic = 0.045
        self.learning_rate_hidden_somatic = 0.00015
        self.learning_rate_readout = 0.035
        self.weight_decay = 0.00001
        self.gradient_clip = 5.0
        self.loss_temperature = 2.7
        self.loss_count_bias = 0.18
        self.loss_label_smoothing = 0.13
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
        
        # Activity history for adaptive learning rates
        self.activity_history = []
        
        # Debug statistics
        self.debug_stats = {
            'epoch_losses': [],
            'epoch_accuracies': [],
            'hidden_spike_rates': [],
            'readout_spike_rates': [],
            'weight_norms_dend': [],
            'weight_norms_soma': [],
            'gradient_norms_dend': [],
            'gradient_norms_soma': [],
            'plateau_activity': [],
            'confusion_matrix': np.zeros((n_outputs, n_outputs))
        }
        
        # Compile key functions for speed
        self._compile_functions()
    
    def _compile_functions(self):
        """Compile frequently used functions"""
        # Note: _forward_impl is not JIT-compiled because it uses instance variables
        # that are captured at compilation time. Instead, we use _forward_with_params_jit
        # which takes parameters as arguments.
        
        # Loss function
        self._loss_compiled = jit(self._loss_impl)
        
        # Training step (target is static for JIT compilation, learning rates are dynamic)
        # static_argnums: 0=params, 1=x_input, 2=target, 3=lr_dend, 4=lr_soma, 5=lr_readout, 6=clip_value
        # Note: Learning rates are NOT static so they can change during training (adaptive behavior)
        self._train_step_compiled = jit(self._train_step_impl, static_argnums=(2,))
    
    def get_params(self) -> NetworkParams:
        """Get all network parameters as a single object"""
        return NetworkParams(
            w_dend=self.hidden_layer.w_dend,
            w_soma=self.hidden_layer.w_soma,
            w_readout=self.readout_layer.w
        )
    
    def set_params(self, params: NetworkParams):
        """Set network parameters"""
        self.hidden_layer.w_dend = params.w_dend
        self.hidden_layer.w_soma = params.w_soma
        self.readout_layer.w = params.w_readout
    
    def _forward_impl(self, x_input: jnp.ndarray) -> Tuple:
        """Internal forward pass implementation"""
        # Use per-neuron T_p values from hidden layer
        T_p = self.hidden_layer.T_p  # (n_neurons,)
        
        # Hidden layer
        mu, v, h, hidden_o, mu_history, t_prime_history, v_history = JAXTwoCompartmentalLayer.forward_pass(
            x_input, 
            self.hidden_layer.w_dend, 
            self.hidden_layer.w_soma,
            self.config,
            self.T,
            T_p
        )
        
        # Readout layer
        readout_v, readout_o = JAXLIFLayer.forward_pass(
            hidden_o,
            self.readout_layer.w,
            self.config,
            self.T
        )
        
        return mu, v, h, hidden_o, readout_v, readout_o, mu_history, t_prime_history, v_history
    
    def forward(self, x_input: jnp.ndarray) -> Tuple:
        """Forward pass through the network"""
        # Convert to JAX array if needed
        if not isinstance(x_input, jnp.ndarray):
            x_input = jnp.array(x_input)
        
        # Use _forward_with_params to ensure we use current weights
        # (JIT-compiled _forward_compiled captures weights at compilation time)
        params = self.get_params()
        T_p = self.hidden_layer.T_p  # (n_neurons,)
        return self._forward_with_params_jit(params, x_input, self.config, self.T, T_p)
    
    def _loss_impl(self, params: NetworkParams, x_input: jnp.ndarray, target: int) -> jnp.ndarray:
        """Compute loss for a single sample"""
        # Forward pass with given parameters
        mu, v, h, hidden_o, readout_v, readout_o, _, _, _ = self._forward_with_params(params, x_input)
        
        # Convert spike counts to probabilities
        readout_counts = jnp.sum(readout_o, axis=0)  # (n_outputs,)
        scaled_counts = readout_counts / self.loss_temperature + self.loss_count_bias
        exp_counts = jnp.exp(scaled_counts - jnp.max(scaled_counts))
        probabilities = exp_counts / jnp.sum(exp_counts)
        
        # Cross-entropy loss with label smoothing
        target_one_hot = jnp.zeros(self.n_outputs)
        target_one_hot = target_one_hot.at[target].set(1.0)
        
        target_one_hot = target_one_hot * (1 - self.loss_label_smoothing) + self.loss_label_smoothing / self.n_outputs
        
        loss = -jnp.sum(target_one_hot * jnp.log(probabilities + 1e-8))
        return loss
    
    def _forward_with_params(self, params: NetworkParams, x_input: jnp.ndarray) -> Tuple:
        """Forward pass with given parameters"""
        # Use per-neuron T_p values from hidden layer
        T_p = self.hidden_layer.T_p  # (n_neurons,)
        
        mu, v, h, hidden_o, mu_history, t_prime_history, v_history = JAXTwoCompartmentalLayer.forward_pass(
            x_input, params.w_dend, params.w_soma, self.config, self.T, T_p
        )
        
        readout_v, readout_o = JAXLIFLayer.forward_pass(
            hidden_o, params.w_readout, self.config, self.T
        )
        
        return mu, v, h, hidden_o, readout_v, readout_o, mu_history, t_prime_history, v_history
    
    @staticmethod
    @jit(static_argnames=['T'])
    def _forward_with_params_jit(params: NetworkParams, x_input: jnp.ndarray, 
                                  config: NeuronConfig, T: int, T_p: jnp.ndarray) -> Tuple:
        """JIT-compiled forward pass with parameters"""
        mu, v, h, hidden_o, mu_history, t_prime_history, v_history = JAXTwoCompartmentalLayer.forward_pass(
            x_input, params.w_dend, params.w_soma, config, T, T_p
        )
        
        readout_v, readout_o = JAXLIFLayer.forward_pass(
            hidden_o, params.w_readout, config, T
        )
        
        return mu, v, h, hidden_o, readout_v, readout_o, mu_history, t_prime_history, v_history
    
    def compute_global_errors(self, readout_o: jnp.ndarray, target: int) -> jnp.ndarray:
        """Compute global errors for e-prop"""
        readout_counts = jnp.sum(readout_o, axis=0)
        scaled_counts = readout_counts / self.loss_temperature + self.loss_count_bias
        exp_counts = jnp.exp(scaled_counts - jnp.max(scaled_counts))
        probabilities = exp_counts / jnp.sum(exp_counts)
        
        target_one_hot = jnp.zeros(self.n_outputs)
        target_one_hot = target_one_hot.at[target].set(1.0)
        target_one_hot = target_one_hot * (1 - self.loss_label_smoothing) + self.loss_label_smoothing / self.n_outputs
        
        return target_one_hot - probabilities  # (n_outputs,)
    
    def get_balanced_learning_rates(self) -> Tuple[float, float, float]:
        """Compute balanced learning rates based on recent activity"""
        if len(self.activity_history) < 10:
            return (self.learning_rate_hidden_dendritic, 
                    self.learning_rate_hidden_somatic, 
                    self.learning_rate_readout)
        
        recent_activity = np.array(self.activity_history[-10:])
        avg_activity = np.mean(recent_activity, axis=0)
        
        activity_std = np.std(avg_activity)
        activity_mean = np.mean(avg_activity)
        
        if activity_std > activity_mean * 0.5:
            readout_lr = self.learning_rate_readout * 0.5
        else:
            readout_lr = self.learning_rate_readout
        
        return (self.learning_rate_hidden_dendritic, 
                self.learning_rate_hidden_somatic, 
                readout_lr)
    
    def _train_step_impl(self, params: NetworkParams, x_input: jnp.ndarray, 
                        target: int, lr_dend: float, lr_soma: float, lr_readout: float, 
                        clip_value: float = 5.0):
        """Single training step with manual gradient computation for e-prop"""
        # Forward pass
        mu, v, h, hidden_o, readout_v, readout_o, mu_history, t_prime_history, v_history = self._forward_with_params(params, x_input)
        
        # Compute global errors
        global_errors = self.compute_global_errors(readout_o, target)
        
        # === READOUT LAYER GRADIENTS ===
        # Eligibility traces for readout
        # All output neurons receive the same input, so they share the same eligibility trace
        E_readout = JAXTwoCompartmentalLayer.compute_eligibility_traces(hidden_o, 
                                                                        self.readout_layer.alpha)  # (T, n_hidden)
        
        # Surrogate gradient for readout (using same beta as somatic: beta_s)
        v_input_vals = readout_v - self.config.v_th
        sigma_prime_readout = JAXTwoCompartmentalLayer.surrogate_sigma(v_input_vals, self.config.beta_s)  # (T, n_outputs)
        
        # Gradient for readout weights
        # Formula: grad[i,j] = Σ_t sigma'[t,i] * E[t,j] * error[i] / T
        # sigma_prime_readout: (T, n_outputs) - sigma'[t, i]
        # E_readout: (T, n_hidden) - E[t, j] (same for all output neurons)
        # global_errors: (n_outputs,) - error[i]
        # Result: (n_outputs, n_hidden)
        grad_readout_raw = jnp.einsum('ti,tj,i->ij', 
                                     sigma_prime_readout, E_readout, global_errors) / self.T
        grad_readout_clipped = jnp.clip(grad_readout_raw, -clip_value, clip_value)
        
        # === HIDDEN LAYER GRADIENTS ===
        # CORRECTED: Compute time-dependent effective error for each hidden neuron
        # effective_error(t) = Σ_j [global_error_j · σ'_readout_j(t) · w_{readout_j, hidden_i}]
        # This is time-dependent (T, n_hidden) instead of a scalar (n_hidden,)
        T = x_input.shape[0]
        n_neurons = h.shape[1]
        
        # Compute time-dependent effective error for each hidden neuron
        # sigma_prime_readout: (T, n_outputs)
        # global_errors: (n_outputs,)
        # params.w_readout: (n_outputs, n_hidden)
        # Result: (T, n_hidden) - effective error for each hidden neuron at each time step
        effective_error_time_series = jnp.einsum('tj,j,ji->ti', 
                                                 sigma_prime_readout, 
                                                 global_errors, 
                                                 params.w_readout)  # (T, n_hidden)
        
        # Somatic surrogate gradient: use v_history (pre-reset) so surrogate sees voltage at threshold crossing
        soma_input_vals = v_history + self.config.gamma * h - self.config.v_th
        sigma_prime_hidden = JAXTwoCompartmentalLayer.surrogate_sigma(soma_input_vals, self.config.beta_s)
        
        # Dendritic surrogate gradient - FIXED: Use mu at plateau initiation time
        # Get mu at t_prime for each neuron at each time step
        time_indices = jnp.arange(T)[:, None]  # (T, 1)
        
        # For each time step and neuron, get mu_history[t_prime[t, neuron], neuron]
        t_prime_indices = t_prime_history.astype(jnp.int32)  # (T, n_neurons)
        neuron_indices = jnp.arange(n_neurons)[None, :]  # (1, n_neurons)
        mu_at_tprime = mu_history[t_prime_indices, neuron_indices]  # (T, n_neurons)
        
        # Evaluate h' at mu_at_tprime (plateau initiation time), not current mu
        dend_input_vals = mu_at_tprime - self.config.mu_th  # (T, n_neurons)
        h_prime = JAXTwoCompartmentalLayer.surrogate_sigma(dend_input_vals, self.config.beta_d)  # (T, n_neurons)
        
        # Eligibility traces for hidden layer (somatic)
        E_soma = JAXTwoCompartmentalLayer.compute_eligibility_traces(x_input, self.hidden_layer.alpha_s)  # (T, n_inputs)
        
        # FIXED: Compute dmu_tprime_dw properly
        dmu_tprime_dw = JAXTwoCompartmentalLayer.compute_dmu_tprime_dw(
            x_input, h, t_prime_history, self.hidden_layer.alpha
        )  # (T, n_neurons, n_inputs)
        
        # Gradients for hidden layer
        # Somatic gradients: sigma' * E_soma * effective_error_time_series
        # sigma_prime_hidden: (T, n_neurons)
        # E_soma: (T, n_inputs)
        # effective_error_time_series: (T, n_neurons)
        grad_soma_raw = jnp.einsum('ti,tj,ti->ij',
                                  sigma_prime_hidden, E_soma, effective_error_time_series) / self.T
        grad_soma_clipped = jnp.clip(grad_soma_raw, -clip_value, clip_value)
        
        # Dendritic gradients - FIXED: Use dmu_tprime_dw instead of E_soma
        # grad_dend = gamma * sigma' * h' * dmu_tprime_dw * effective_error_time_series
        # sigma_prime_hidden: (T, n_neurons)
        # h_prime: (T, n_neurons)
        # dmu_tprime_dw: (T, n_neurons, n_inputs)
        # effective_error_time_series: (T, n_neurons)
        grad_dend_raw = jnp.einsum('ti,ti,tij,ti->ij',
                                  sigma_prime_hidden, h_prime, dmu_tprime_dw,
                                  effective_error_time_series * self.config.gamma) / self.T
        grad_dend_clipped = jnp.clip(grad_dend_raw, -clip_value, clip_value)
        
        # === UPDATE WEIGHTS ===
        # Apply learning rates and weight decay (using provided learning rates for adaptive behavior)
        new_w_dend = params.w_dend * (1 - self.weight_decay) + lr_dend * grad_dend_clipped
        new_w_soma = params.w_soma * (1 - self.weight_decay) + lr_soma * grad_soma_clipped
        new_w_readout = params.w_readout * (1 - self.weight_decay) + lr_readout * grad_readout_clipped
        
        # Clip weights
        new_w_dend = jnp.clip(new_w_dend, -1.0, 1.0)
        new_w_soma = jnp.clip(new_w_soma, -1.0, 1.0)
        new_w_readout = jnp.clip(new_w_readout, -1.0, 1.0)
        
        new_params = NetworkParams(
            w_dend=new_w_dend,
            w_soma=new_w_soma,
            w_readout=new_w_readout
        )
        
        # Compute loss for debugging
        loss = self._loss_impl(params, x_input, target)
        
        return new_params, loss, hidden_o, readout_o

    def get_single_sample_diagnostics(self, params: NetworkParams, x_input: jnp.ndarray, target: int) -> Dict:
        """Compute target, prediction, and gradient-component norms for one sample (for first/last sample logging)."""
        mu, v, h, hidden_o, readout_v, readout_o, mu_history, t_prime_history, v_history = self._forward_with_params(params, x_input)
        prediction = int(jnp.argmax(jnp.sum(readout_o, axis=0)))
        target_int = int(target) if hasattr(target, '__int__') else int(target)

        global_errors = self.compute_global_errors(readout_o, target)
        E_readout = JAXTwoCompartmentalLayer.compute_eligibility_traces(hidden_o, self.readout_layer.alpha)
        v_input_vals = readout_v - self.config.v_th
        sigma_prime_readout = JAXTwoCompartmentalLayer.surrogate_sigma(v_input_vals, self.config.beta_s)
        grad_readout_raw = jnp.einsum('ti,tj,i->ij', sigma_prime_readout, E_readout, global_errors) / self.T

        T = x_input.shape[0]
        n_neurons = h.shape[1]
        effective_error_time_series = jnp.einsum('tj,j,ji->ti', sigma_prime_readout, global_errors, params.w_readout)
        soma_input_vals = v_history + self.config.gamma * h - self.config.v_th
        sigma_prime_hidden = JAXTwoCompartmentalLayer.surrogate_sigma(soma_input_vals, self.config.beta_s)
        t_prime_indices = t_prime_history.astype(jnp.int32)
        neuron_indices = jnp.arange(n_neurons)[None, :]
        mu_at_tprime = mu_history[t_prime_indices, neuron_indices]
        dend_input_vals = mu_at_tprime - self.config.mu_th
        h_prime = JAXTwoCompartmentalLayer.surrogate_sigma(dend_input_vals, self.config.beta_d)
        E_soma = JAXTwoCompartmentalLayer.compute_eligibility_traces(x_input, self.hidden_layer.alpha_s)
        dmu_tprime_dw = JAXTwoCompartmentalLayer.compute_dmu_tprime_dw(
            x_input, h, t_prime_history, self.hidden_layer.alpha
        )
        grad_soma_raw = jnp.einsum('ti,tj,ti->ij', sigma_prime_hidden, E_soma, effective_error_time_series) / self.T
        grad_dend_raw = jnp.einsum('ti,ti,tij,ti->ij',
                                  sigma_prime_hidden, h_prime, dmu_tprime_dw,
                                  effective_error_time_series * self.config.gamma) / self.T

        loss = float(self._loss_impl(params, x_input, target))

        def n(x):
            return float(jnp.linalg.norm(x))

        return {
            "target": target_int,
            "prediction": prediction,
            "loss": loss,
            "global_errors": n(global_errors),
            "effective_error_time_series": n(effective_error_time_series),
            "sigma_prime_readout": n(sigma_prime_readout),
            "sigma_prime_hidden": n(sigma_prime_hidden),
            "h_prime": n(h_prime),
            "E_readout": n(E_readout),
            "E_soma": n(E_soma),
            "dmu_tprime_dw": n(dmu_tprime_dw),
            "grad_dend": n(grad_dend_raw),
            "grad_soma": n(grad_soma_raw),
            "grad_readout": n(grad_readout_raw),
        }

    def train_step(self, x_input: jnp.ndarray, target: int) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
        """Perform one training step and update network"""
        params = self.get_params()
        
        # Get balanced learning rates (adaptive based on activity history)
        lr_dend, lr_soma, lr_readout = self.get_balanced_learning_rates()
        
        new_params, loss, hidden_o, readout_o = self._train_step_compiled(
            params, x_input, target, lr_dend, lr_soma, lr_readout, self.gradient_clip
        )
        
        # Update network parameters
        self.set_params(new_params)
        
        # Track activity history for adaptive learning rates (matching NumPy version)
        readout_activity = np.array(jnp.sum(readout_o, axis=0))  # Sum over time for each output neuron
        self.activity_history.append(readout_activity)
        
        # Keep only last 50 entries (matching NumPy version)
        if len(self.activity_history) > 50:
            self.activity_history = self.activity_history[-50:]
        
        return float(loss), hidden_o, readout_o
    
    def train_step_batch(self, x_input_batch: jnp.ndarray, target_batch: jnp.ndarray) -> Tuple[float, List[jnp.ndarray], List[jnp.ndarray]]:
        """Perform training step on a batch of samples (faster than sequential train_step calls)
        
        Args:
            x_input_batch: (batch_size, T, n_inputs) input batch
            target_batch: (batch_size,) target labels batch (JAX array of integers)
            
        Returns:
            Average loss, list of hidden outputs, list of readout outputs
        """
        params = self.get_params()
        
        # Get balanced learning rates
        lr_dend, lr_soma, lr_readout = self.get_balanced_learning_rates()
        
        # Compute gradients for each sample, then average
        def compute_grads(x_input, target):
            """Compute gradients for a single sample - target is JAX array scalar"""
            # Forward pass
            mu, v, h, hidden_o, readout_v, readout_o, mu_history, t_prime_history, v_history = self._forward_with_params(params, x_input)
            
            # Compute global errors
            global_errors = self.compute_global_errors(readout_o, target)
            
            # === READOUT LAYER GRADIENTS ===
            E_readout = JAXTwoCompartmentalLayer.compute_eligibility_traces(hidden_o, self.readout_layer.alpha)
            v_input_vals = readout_v - self.config.v_th
            # Use beta_s for readout (matching _train_step_impl)
            sigma_prime_readout = JAXTwoCompartmentalLayer.surrogate_sigma(v_input_vals, self.config.beta_s)
            grad_readout = jnp.einsum('ti,tj,i->ij', sigma_prime_readout, E_readout, global_errors) / self.T
            
            # === HIDDEN LAYER GRADIENTS ===
            T = x_input.shape[0]
            n_neurons = h.shape[1]
            
            # Time-dependent effective error
            effective_error_time_series = jnp.einsum('tj,j,ji->ti', 
                                                     sigma_prime_readout, 
                                                     global_errors, 
                                                     params.w_readout)
            
            # Somatic surrogate gradient: use v_history (pre-reset) for correct surrogate at threshold
            soma_input_vals = v_history + self.config.gamma * h - self.config.v_th
            sigma_prime_hidden = JAXTwoCompartmentalLayer.surrogate_sigma(soma_input_vals, self.config.beta_s)
            
            # Dendritic surrogate gradient
            t_prime_indices = t_prime_history.astype(jnp.int32)
            neuron_indices = jnp.arange(n_neurons)[None, :]
            mu_at_tprime = mu_history[t_prime_indices, neuron_indices]
            dend_input_vals = mu_at_tprime - self.config.mu_th
            h_prime = JAXTwoCompartmentalLayer.surrogate_sigma(dend_input_vals, self.config.beta_d)
            
            # Eligibility traces for hidden layer (somatic)
            E_soma = JAXTwoCompartmentalLayer.compute_eligibility_traces(x_input, self.hidden_layer.alpha_s)
            
            # Compute dmu_tprime_dw
            dmu_tprime_dw = JAXTwoCompartmentalLayer.compute_dmu_tprime_dw(
                x_input, h, t_prime_history, self.hidden_layer.alpha
            )
            
            # Somatic gradients
            grad_soma = jnp.einsum('ti,tj,ti->ij',
                                  sigma_prime_hidden, E_soma, effective_error_time_series) / self.T
            
            # Dendritic gradients
            grad_dend = jnp.einsum('ti,ti,tij,ti->ij',
                                  sigma_prime_hidden, h_prime, dmu_tprime_dw,
                                  effective_error_time_series * self.config.gamma) / self.T
            
            # Loss
            loss = self._loss_impl(params, x_input, target)
            
            return grad_dend, grad_soma, grad_readout, loss, hidden_o, readout_o
        
        # Vectorize gradient computation across batch
        compute_grads_batched = vmap(compute_grads, in_axes=(0, 0))
        grad_dend_batch, grad_soma_batch, grad_readout_batch, losses, hidden_os, readout_os = compute_grads_batched(x_input_batch, target_batch)
        
        # Average gradients across batch
        grad_dend_avg = jnp.mean(grad_dend_batch, axis=0)
        grad_soma_avg = jnp.mean(grad_soma_batch, axis=0)
        grad_readout_avg = jnp.mean(grad_readout_batch, axis=0)
        
        # Clip averaged gradients
        grad_dend_clipped = jnp.clip(grad_dend_avg, -self.gradient_clip, self.gradient_clip)
        grad_soma_clipped = jnp.clip(grad_soma_avg, -self.gradient_clip, self.gradient_clip)
        grad_readout_clipped = jnp.clip(grad_readout_avg, -self.gradient_clip, self.gradient_clip)
        
        # Update weights using averaged gradients
        new_w_dend = params.w_dend * (1 - self.weight_decay) + lr_dend * grad_dend_clipped
        new_w_soma = params.w_soma * (1 - self.weight_decay) + lr_soma * grad_soma_clipped
        new_w_readout = params.w_readout * (1 - self.weight_decay) + lr_readout * grad_readout_clipped
        
        # Clip weights
        new_w_dend = jnp.clip(new_w_dend, -1.0, 1.0)
        new_w_soma = jnp.clip(new_w_soma, -1.0, 1.0)
        new_w_readout = jnp.clip(new_w_readout, -1.0, 1.0)
        
        new_params = NetworkParams(
            w_dend=new_w_dend,
            w_soma=new_w_soma,
            w_readout=new_w_readout
        )
        self.set_params(new_params)
        
        # Track activity history (use average across batch)
        readout_activity_avg = np.array(jnp.mean(jnp.sum(readout_os, axis=1), axis=0))  # Average over batch, sum over time
        self.activity_history.append(readout_activity_avg)
        if len(self.activity_history) > 50:
            self.activity_history = self.activity_history[-50:]
        
        # Convert outputs to lists for compatibility
        hidden_os_list = [hidden_os[i] for i in range(hidden_os.shape[0])]
        readout_os_list = [readout_os[i] for i in range(readout_os.shape[0])]
        
        avg_loss = float(jnp.mean(losses))
        return avg_loss, hidden_os_list, readout_os_list
    
    def predict(self, x_input: jnp.ndarray) -> int:
        """Make prediction"""
        _, _, _, _, _, readout_o, _, _, _ = self.forward(x_input)
        
        readout_counts = jnp.sum(readout_o, axis=0)
        scaled_counts = readout_counts / self.loss_temperature + self.loss_count_bias
        exp_counts = jnp.exp(scaled_counts - jnp.max(scaled_counts))
        probabilities = exp_counts / jnp.sum(exp_counts)
        
        return int(jnp.argmax(probabilities))
    
    def evaluate(self, test_data: List[Tuple[np.ndarray, int]]) -> Dict:
        """Evaluate network on test data"""
        correct = 0
        total = 0
        test_losses = []
        confusion_matrix = np.zeros((self.n_outputs, self.n_outputs))
        
        print(f"Evaluating on {len(test_data)} test samples...", flush=True)
        
        for idx, (x_input, target) in enumerate(test_data):
            if idx % 1000 == 0 and idx > 0:
                print(f"  Evaluation: {idx}/{len(test_data)} samples", flush=True)
            
            # x_input is already (T, n_inputs) from run script
            x_network_jax = jnp.array(np.asarray(x_input))
            
            # Forward pass
            _, _, _, _, _, readout_o, _, _, _ = self.forward(x_network_jax)
            
            # Compute loss
            readout_counts = jnp.sum(readout_o, axis=0)
            scaled_counts = readout_counts / self.loss_temperature + self.loss_count_bias
            exp_counts = jnp.exp(scaled_counts - jnp.max(scaled_counts))
            probabilities = exp_counts / jnp.sum(exp_counts)
            
            target_one_hot = jnp.zeros(self.n_outputs)
            target_one_hot = target_one_hot.at[target].set(1.0)
            target_one_hot = target_one_hot * (1 - self.loss_label_smoothing) + self.loss_label_smoothing / self.n_outputs
            
            loss = -jnp.sum(target_one_hot * jnp.log(probabilities + 1e-8))
            test_losses.append(float(loss))
            
            # Prediction with tie-breaking (matching NumPy behavior)
            # Convert to NumPy for consistent argmax behavior
            probabilities_np = np.array(probabilities)
            max_val = np.max(probabilities_np)
            if np.all(probabilities_np == max_val):
                # All probabilities equal - random choice
                prediction = np.random.randint(0, self.n_outputs)
            else:
                max_indices = np.where(probabilities_np == max_val)[0]
                if len(max_indices) > 1:
                    # Multiple tied - random choice among ties
                    prediction = np.random.choice(max_indices)
                else:
                    # Single maximum - deterministic
                    prediction = int(np.argmax(probabilities_np))
            
            if prediction == target:
                correct += 1
            total += 1
            confusion_matrix[target, prediction] += 1
        
        accuracy = correct / total * 100
        avg_loss = np.mean(test_losses)
        
        # Per-class accuracy
        per_class_accuracy = {}
        for i in range(self.n_outputs):
            class_total = np.sum(confusion_matrix[i, :])
            if class_total > 0:
                class_correct = confusion_matrix[i, i]
                per_class_accuracy[i] = (class_correct / class_total) * 100
        
        return {
            'accuracy': accuracy,
            'avg_loss': avg_loss,
            'correct': correct,
            'total': total,
            'confusion_matrix': confusion_matrix,
            'per_class_accuracy': per_class_accuracy
        }
    
    def save(self, filepath: str):
        """Save network to file"""
        save_data = {
            'n_inputs': self.n_inputs,
            'n_hidden': self.n_hidden,
            'n_outputs': self.n_outputs,
            'T': self.T,
            'w_dend': np.array(self.hidden_layer.w_dend),
            'w_soma': np.array(self.hidden_layer.w_soma),
            'w_readout': np.array(self.readout_layer.w),
            'debug_stats': self.debug_stats
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, key=None):
        """Load network from file"""
        if key is None:
            key = random.PRNGKey(0)
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create network
        network = cls(
            key,
            n_inputs=save_data['n_inputs'],
            n_hidden=save_data['n_hidden'],
            n_outputs=save_data['n_outputs'],
            T=save_data['T']
        )
        
        # Load weights
        network.hidden_layer.w_dend = jnp.array(save_data['w_dend'])
        network.hidden_layer.w_soma = jnp.array(save_data['w_soma'])
        network.readout_layer.w = jnp.array(save_data['w_readout'])
        
        # Load debug stats
        network.debug_stats = save_data['debug_stats']
        
        print(f"Model loaded from {filepath}")
        return network



''' TRAINING FUNCTION '''

def print_training_diagnostics_jax(network: JAXEPropNetwork, sample_idx: int, recent_losses: List[float], 
                                   recent_accuracies: List[float], recent_grads_dend: List[np.ndarray],
                                   recent_grads_soma: List[np.ndarray], recent_grads_readout: List[np.ndarray],
                                   initial_weight_norms: Dict = None):
    """Print detailed training diagnostics every 100 samples (JAX version)"""
    print(f"\n  {'-'*80}", flush=True)
    print(f"  DIAGNOSTICS at sample {sample_idx}", flush=True)
    print(f"  {'-'*80}", flush=True)
    
    # Training metrics
    if recent_losses:
        print(f"  Training (last {len(recent_losses)} samples):", flush=True)
        print(f"    Loss: {np.mean(recent_losses):.4f} (std: {np.std(recent_losses):.4f}, min: {np.min(recent_losses):.4f}, max: {np.max(recent_losses):.4f})", flush=True)
    if recent_accuracies:
        print(f"    Accuracy: {np.mean(recent_accuracies)*100:.2f}% ({np.sum(recent_accuracies)}/{len(recent_accuracies)})", flush=True)
    
    # Weight norms and changes (JAX uses different structure)
    dend_norms = [np.linalg.norm(np.array(network.hidden_layer.w_dend[i])) for i in range(network.n_hidden)]
    soma_norms = [np.linalg.norm(np.array(network.hidden_layer.w_soma[i])) for i in range(network.n_hidden)]
    readout_norms = [np.linalg.norm(np.array(network.readout_layer.w[i])) for i in range(network.n_outputs)]
    
    print(f"  Weight norms:", flush=True)
    print(f"    Dendritic: {np.mean(dend_norms):.4f} (std: {np.std(dend_norms):.4f}, range: [{np.min(dend_norms):.4f}, {np.max(dend_norms):.4f}])", flush=True)
    print(f"    Somatic: {np.mean(soma_norms):.4f} (std: {np.std(soma_norms):.4f}, range: [{np.min(soma_norms):.4f}, {np.max(soma_norms):.4f}])", flush=True)
    print(f"    Readout: {np.mean(readout_norms):.4f} (std: {np.std(readout_norms):.4f}, range: [{np.min(readout_norms):.4f}, {np.max(readout_norms):.4f}])", flush=True)
    
    # Weight changes (if initial norms provided)
    if initial_weight_norms is not None:
        dend_change = np.mean(dend_norms) - initial_weight_norms['dend']
        soma_change = np.mean(soma_norms) - initial_weight_norms['soma']
        readout_change = np.mean(readout_norms) - initial_weight_norms['readout']
        print(f"  Weight changes from initial:", flush=True)
        print(f"    Dendritic: {dend_change:+.4f} ({dend_change/initial_weight_norms['dend']*100:+.2f}%)", flush=True)
        print(f"    Somatic: {soma_change:+.4f} ({soma_change/initial_weight_norms['soma']*100:+.2f}%)", flush=True)
        print(f"    Readout: {readout_change:+.4f} ({readout_change/initial_weight_norms['readout']*100:+.2f}%)", flush=True)
    
    # Gradient norms
    if recent_grads_dend:
        grad_norms_dend = [np.linalg.norm(g) if g is not None else 0.0 for g in recent_grads_dend]
        print(f"  Gradient norms (last {len(recent_grads_dend)} samples):", flush=True)
        print(f"    Dendritic: {np.mean(grad_norms_dend):.6f} (std: {np.std(grad_norms_dend):.6f}, max: {np.max(grad_norms_dend):.6f})", flush=True)
    if recent_grads_soma:
        grad_norms_soma = [np.linalg.norm(g) if g is not None else 0.0 for g in recent_grads_soma]
        print(f"    Somatic: {np.mean(grad_norms_soma):.6f} (std: {np.std(grad_norms_soma):.6f}, max: {np.max(grad_norms_soma):.6f})", flush=True)
    if recent_grads_readout:
        grad_norms_readout = [np.linalg.norm(g) if g is not None else 0.0 for g in recent_grads_readout]
        print(f"    Readout: {np.mean(grad_norms_readout):.6f} (std: {np.std(grad_norms_readout):.6f}, max: {np.max(grad_norms_readout):.6f})", flush=True)
    
    # Spike rates
    if len(network.debug_stats['hidden_spike_rates']) > 0:
        recent_hidden_rates = network.debug_stats['hidden_spike_rates'][-min(100, len(network.debug_stats['hidden_spike_rates'])):]
        recent_readout_rates = network.debug_stats['readout_spike_rates'][-min(100, len(network.debug_stats['readout_spike_rates'])):]
        print(f"  Spike rates (Hz, last {len(recent_hidden_rates)} samples):", flush=True)
        print(f"    Hidden: {np.mean(recent_hidden_rates)*1000:.2f} Hz (std: {np.std(recent_hidden_rates)*1000:.2f})", flush=True)
        print(f"    Readout: {np.mean(recent_readout_rates)*1000:.2f} Hz (std: {np.std(recent_readout_rates)*1000:.2f})", flush=True)
    
    # Plateau activity (for two-compartmental neurons)
    if len(network.debug_stats['plateau_activity']) > 0:
        recent_plateau = network.debug_stats['plateau_activity'][-min(100, len(network.debug_stats['plateau_activity'])):]
        print(f"    Plateau activity: {np.mean(recent_plateau):.4f} (std: {np.std(recent_plateau):.4f})", flush=True)
    
    # Learning rates
    lr_dend, lr_soma, lr_readout = network.get_balanced_learning_rates()
    print(f"  Learning rates:", flush=True)
    print(f"    Dendritic: {lr_dend:.6f} (base: {network.learning_rate_hidden_dendritic:.6f})", flush=True)
    print(f"    Somatic: {lr_soma:.6f} (base: {network.learning_rate_hidden_somatic:.6f})", flush=True)
    print(f"    Readout: {lr_readout:.6f} (base: {network.learning_rate_readout:.6f})", flush=True)
    
    # Sample predictions (last 10)
    if len(network.debug_stats['epoch_accuracies']) >= 10:
        print(f"  Recent accuracy pattern (last 10): {[int(a) for a in network.debug_stats['epoch_accuracies'][-10:]]}", flush=True)
    
    print(f"  {'-'*80}", flush=True)

def _print_epoch_sample_diagnostics_1layer(d: Dict, label: str):
    """Print first/last sample diagnostics for 1-layer (target, prediction, gradient norms)."""
    print(f"  --- {label} ---", flush=True)
    print(f"    target: {d['target']}  prediction: {d['prediction']}  loss: {d['loss']:.6f}", flush=True)
    print(f"    Error:        global_errors: {d['global_errors']:.6f}  effective_error_time_series: {d['effective_error_time_series']:.6f}", flush=True)
    print(f"    Surrogate:    sigma_prime_readout: {d['sigma_prime_readout']:.6f}  sigma_prime_hidden: {d['sigma_prime_hidden']:.6f}  h_prime: {d['h_prime']:.6f}", flush=True)
    print(f"    Eligibility:  E_readout: {d['E_readout']:.6f}  E_soma: {d['E_soma']:.6f}  dmu_tprime_dw: {d['dmu_tprime_dw']:.6f}", flush=True)
    print(f"    Grad norms:   grad_dend: {d['grad_dend']:.6f}  grad_soma: {d['grad_soma']:.6f}  grad_readout: {d['grad_readout']:.6f}", flush=True)


def train_network_jax(network: JAXEPropNetwork, train_data: List[Tuple[np.ndarray, int]],
                     test_data: List[Tuple[np.ndarray, int]], epochs: int = 10, batch_size: int = 32,
                     run_dir: str = "model"):
    """Train the JAX network with epochs and test evaluation after each epoch.
    Expects train_data/test_data as list of (x, label) with x shape (T, n_inputs).
    
    Args:
        network: JAXEPropNetwork instance
        train_data: List of (input, label) tuples; input is (T, n_inputs) array
        test_data: List of (input, label) tuples; input is (T, n_inputs) array
        epochs: Number of training epochs
        batch_size: Batch size for training (default: 32). Use batch_size=1 for single-sample training.
    """
    print(f"Training JAX network with {len(train_data)} training samples...", flush=True)
    print(f"Test set: {len(test_data)} samples (evaluated after each epoch)", flush=True)
    print(f"Network architecture: {network.n_inputs} inputs → {network.n_hidden} hidden → {network.n_outputs} outputs", flush=True)
    print(f"Training configuration:", flush=True)
    print(f"  - Epochs: {epochs} (each epoch uses all {len(train_data)} training samples)", flush=True)
    print(f"  - Batch size: {batch_size} (using {'batched' if batch_size > 1 else 'single-sample'} training)", flush=True)
    print(f"  - Test evaluation: After each epoch", flush=True)
    
    # INITIAL EVALUATION: Test before training to verify baseline performance
    print(f"\n{'='*80}", flush=True)
    print(f"INITIAL EVALUATION (BEFORE TRAINING)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Evaluating on test set to establish baseline...", flush=True)
    initial_test_results = network.evaluate(test_data)
    print(f"INITIAL TEST RESULTS - Loss: {initial_test_results['avg_loss']:.4f}, Accuracy: {initial_test_results['accuracy']:.2f}% ({initial_test_results['correct']}/{initial_test_results['total']})", flush=True)
    print(f"Expected baseline accuracy for random guessing: ~5.0% (1/{network.n_outputs} = 1/{network.n_outputs})", flush=True)
    if initial_test_results['accuracy'] > 15.0:
        print(f"WARNING: Initial accuracy ({initial_test_results['accuracy']:.2f}%) is suspiciously high!", flush=True)
        print(f"This might indicate the model is already trained or there's a data leakage issue.", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Store initial weight norms for tracking changes
    initial_weight_norms = {
        'dend': np.mean([np.linalg.norm(np.array(network.hidden_layer.w_dend[i])) for i in range(network.n_hidden)]),
        'soma': np.mean([np.linalg.norm(np.array(network.hidden_layer.w_soma[i])) for i in range(network.n_hidden)]),
        'readout': np.mean([np.linalg.norm(np.array(network.readout_layer.w[i])) for i in range(network.n_outputs)])
    }
    print(f"Initial weight norms - Dend: {initial_weight_norms['dend']:.4f}, Soma: {initial_weight_norms['soma']:.4f}, Readout: {initial_weight_norms['readout']:.4f}", flush=True)
    
    epoch_results = []
    
    # Track recent gradients for diagnostics
    recent_grads_dend = []
    recent_grads_soma = []
    recent_grads_readout = []
    
    training_start_time = time.time()
    total_samples_processed = 0
    
    # Track best model
    best_accuracy = -1.0
    best_model_path = None
    
    for epoch in range(epochs):
        print(f"\n{'='*80}", flush=True)
        print(f"EPOCH {epoch+1}/{epochs}", flush=True)
        print(f"{'='*80}", flush=True)
        
        # Shuffle data - match NumPy's behavior exactly
        # NumPy: seed(42) set once at module level → network created (30 random calls) → 
        #        epoch 1 shuffle (uses random state after 30 calls) →
        #        epoch 2 shuffle (continues from epoch 1) →
        #        epoch 3 shuffle (continues from epoch 2)
        # JAX: seed(42) set once in main → weights initialized (30 random calls) →
        #      epoch 1 shuffle (uses random state after 30 calls) →
        #      epoch 2 shuffle (continues from epoch 1) →
        #      epoch 3 shuffle (continues from epoch 2)
        # IMPORTANT: Do NOT reset seed here - let random state continue naturally
        # This ensures the shuffle order matches NumPy exactly
        shuffled_train_data = train_data.copy()
        np.random.shuffle(shuffled_train_data)
        
        # First and last sample of epoch (shuffled order): print target, prediction, gradient-component norms
        params = network.get_params()
        x_first, target_first = shuffled_train_data[0]
        x_last, target_last = shuffled_train_data[-1]
        d_first = network.get_single_sample_diagnostics(params, jnp.array(np.asarray(x_first)), target_first)
        d_last = network.get_single_sample_diagnostics(params, jnp.array(np.asarray(x_last)), target_last)
        print(f"  Epoch {epoch+1} - First sample (shuffled):", flush=True)
        _print_epoch_sample_diagnostics_1layer(d_first, "First sample")
        print(f"  Epoch {epoch+1} - Last sample (shuffled):", flush=True)
        _print_epoch_sample_diagnostics_1layer(d_last, "Last sample")
        
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0
        epoch_start_time = time.time()
        
        # Process data in batches
        n_samples = len(shuffled_train_data)
        n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
        
        for batch_idx in range(n_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_data = shuffled_train_data[start_idx:end_idx]
            actual_batch_size = len(batch_data)
            
            total_samples_processed += actual_batch_size
            
            # Progress reporting (less frequent)
            if batch_idx % max(1, n_batches // 10) == 0 or batch_idx == n_batches - 1:
                elapsed = time.time() - epoch_start_time
                samples_processed = end_idx
                rate = samples_processed / elapsed if elapsed > 0 else 0
                remaining = (n_samples - samples_processed) / rate if rate > 0 else 0
                print(f"  Training: {samples_processed}/{n_samples} samples "
                      f"({samples_processed*100//n_samples}%) | "
                      f"Rate: {rate:.1f} samples/s | ETA: {remaining:.0f}s", flush=True)
            
            if batch_size == 1:
                # Single sample mode (for compatibility/debugging)
                x_input, target = batch_data[0]
                x_network_jax = jnp.array(np.asarray(x_input))  # already (T, n_inputs)
                loss, hidden_o, readout_o = network.train_step(x_network_jax, target)
                hidden_os = [hidden_o]
                readout_os = [readout_o]
                batch_targets = [target]
                batch_losses = [loss]
            else:
                # Batch mode: prepare batch inputs
                batch_inputs = []
                batch_targets = []
                
                for x_input, target in batch_data:
                    x_network = np.asarray(x_input)  # already (T, n_inputs)
                    batch_inputs.append(x_network)
                    batch_targets.append(target)
                
                # Convert to JAX arrays
                x_batch = jnp.array(batch_inputs)  # (batch_size, T, n_inputs)
                target_batch = jnp.array(batch_targets, dtype=jnp.int32)  # (batch_size,)
                
                # Training step on batch
                avg_loss, hidden_os, readout_os = network.train_step_batch(x_batch, target_batch)
                batch_losses = [avg_loss] * actual_batch_size  # Use average loss for each sample in batch
            
            # Process predictions and statistics for each sample in batch
            for i, (hidden_o, readout_o, target) in enumerate(zip(hidden_os, readout_os, batch_targets)):
                # Compute prediction with random tie-breaking (matching NumPy behavior exactly)
                readout_counts = jnp.sum(readout_o, axis=0)
                readout_counts_np = np.array(readout_counts)
                max_val = np.max(readout_counts_np)
                if np.all(readout_counts_np == max_val):
                    # All outputs have same count - random choice
                    prediction = np.random.randint(0, network.n_outputs)
                else:
                    max_indices = np.where(readout_counts_np == max_val)[0]
                    if len(max_indices) > 1:
                        # Multiple outputs tied - random choice among ties
                        prediction = np.random.choice(max_indices)
                    else:
                        # Single maximum - deterministic
                        prediction = int(np.argmax(readout_counts_np))
                
                # Track debug statistics (matching NumPy behavior)
                loss = batch_losses[i] if batch_size > 1 else batch_losses[0]
                network.debug_stats['epoch_losses'].append(loss)
                accuracy = 1.0 if prediction == target else 0.0
                network.debug_stats['epoch_accuracies'].append(accuracy)
                
                # Track spike rates
                hidden_spike_rate = np.mean(np.array(jnp.sum(hidden_o, axis=0))) / network.T
                readout_spike_rate = np.mean(np.array(jnp.sum(readout_o, axis=0))) / network.T
                network.debug_stats['hidden_spike_rates'].append(hidden_spike_rate)
                network.debug_stats['readout_spike_rates'].append(readout_spike_rate)
                
                # Track metrics
                epoch_losses.append(loss)
                if prediction == target:
                    epoch_correct += 1
                epoch_total += 1
            
        
        epoch_elapsed = time.time() - epoch_start_time
        epoch_avg_loss = np.mean(epoch_losses)
        epoch_accuracy = epoch_correct / epoch_total * 100 if epoch_total > 0 else 0.0
        
        # Overall progress
        total_elapsed = time.time() - training_start_time
        total_samples_to_process = epochs * len(train_data)
        overall_progress = total_samples_processed * 100 / total_samples_to_process if total_samples_to_process > 0 else 0
        
        if total_samples_processed > 0:
            eta_minutes = (total_samples_to_process - total_samples_processed) * total_elapsed / total_samples_processed / 60
            print(f"\n  Epoch {epoch+1} completed: {len(shuffled_train_data)} samples in {epoch_elapsed:.1f}s ({len(shuffled_train_data)/epoch_elapsed:.1f} samples/s)", flush=True)
            print(f"  Overall Progress: {total_samples_processed}/{total_samples_to_process} samples ({overall_progress:.1f}%) | "
                  f"Elapsed: {total_elapsed/60:.1f} min | ETA: {eta_minutes:.1f} min", flush=True)
        
        # Evaluate on test set after each epoch
        print(f"\n  Evaluating on test set after epoch {epoch+1}...", flush=True)
        test_results = network.evaluate(test_data)
        epoch_results.append({
            'epoch': epoch + 1,
            'samples_processed': total_samples_processed,
            'train_results': {
                'avg_loss': epoch_avg_loss,
                'accuracy': epoch_accuracy,
                'correct': epoch_correct,
                'total': epoch_total
            },
            'test_results': test_results
        })
        
        # Save best model if this is the best epoch so far
        if test_results['accuracy'] > best_accuracy:
            best_accuracy = test_results['accuracy']
            # Format accuracy as integer (e.g., 85.23 -> 8523, then format as 85_23)
            acc_int = int(best_accuracy * 100)
            acc_str = f"{acc_int // 100:02d}_{acc_int % 100:02d}"
            best_model_path = os.path.join(run_dir, f"heid_2comp_{acc_str}.pkl")
            network.save(best_model_path)
            print(f"  💾 Saved best model (accuracy: {best_accuracy:.2f}%) to {best_model_path}", flush=True)
        
        print(f"\n{'='*80}", flush=True)
        print(f"EPOCH {epoch+1}/{epochs} SUMMARY", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Train - Average Loss: {epoch_avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}% ({epoch_correct}/{epoch_total})", flush=True)
        print(f"Test - Average Loss: {test_results['avg_loss']:.4f}, Accuracy: {test_results['accuracy']:.2f}% ({test_results['correct']}/{test_results['total']})", flush=True)
    
    # Final evaluation
    print(f"\n{'='*80}", flush=True)
    print(f"FINAL EVALUATION ON TEST SET", flush=True)
    print(f"{'='*80}", flush=True)
    
    print(f"Test set: {len(test_data)} total samples, evaluating on all samples", flush=True)
    
    final_test_results = network.evaluate(test_data)
    print(f"FINAL TEST RESULTS - Loss: {final_test_results['avg_loss']:.4f}, "
          f"Accuracy: {final_test_results['accuracy']:.2f}% ({final_test_results['correct']}/{final_test_results['total']})", flush=True)
    
    return epoch_results, final_test_results, best_model_path, best_accuracy


def print_summary_statistics(network: JAXEPropNetwork, epoch_results: List[Dict], 
                            final_test_results: Dict, save_dir: str = "."):
    """Print and save comprehensive summary statistics"""
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    # Use final test results
    final_test = final_test_results
    
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {final_test['accuracy']:.2f}%")
    print(f"  Average Loss: {final_test['avg_loss']:.4f}")
    print(f"  Correct: {final_test['correct']}/{final_test['total']}")
    
    print(f"\nPer-Class Test Accuracy:")
    for digit, acc in sorted(final_test['per_class_accuracy'].items()):
        print(f"  Class {digit}: {acc:.2f}%")
    
    # Confusion matrix
    print(f"\nConfusion Matrix (rows = true, cols = predicted):")
    cm = final_test['confusion_matrix']
    print("      ", end="")
    for i in range(20):
        print(f"{i:6d}", end="")
    print()
    for i in range(20):
        print(f"{i:4d}  ", end="")
        for j in range(20):
            print(f"{int(cm[i, j]):6d}", end="")
        print()
    
    # Training history
    if len(network.debug_stats['epoch_accuracies']) > 0:
        n_samples = len(network.debug_stats['epoch_accuracies'])
        final_train_acc = np.mean(network.debug_stats['epoch_accuracies'][-1000:]) * 100
        print(f"\nTraining History:")
        print(f"  Final training accuracy (last 1000 samples): {final_train_acc:.2f}%")
        if len(network.debug_stats['plateau_activity']) > 0:
            print(f"  Average plateau activity: {np.mean(network.debug_stats['plateau_activity'][-1000:]):.4f}")
    
    # Epoch history
    if len(epoch_results) > 0:
        print(f"\nEpoch History (test set evaluations after each epoch):")
        final_epoch = epoch_results[-1]
        final_epoch_acc = final_epoch['test_results']['accuracy']
        best_epoch_acc = np.max([e['test_results']['accuracy'] for e in epoch_results])
        best_epoch_num = np.argmax([e['test_results']['accuracy'] for e in epoch_results]) + 1
        print(f"  Final epoch accuracy: {final_epoch_acc:.2f}%")
        print(f"  Best epoch accuracy: {best_epoch_acc:.2f}% (epoch {best_epoch_num})")
        print(f"  Number of epochs: {len(epoch_results)}")
    
    # Save statistics to JSON
    stats_dict = {
        'final_test_accuracy': float(final_test['accuracy']),
        'final_test_loss': float(final_test['avg_loss']),
        'test_correct': int(final_test['correct']),
        'test_total': int(final_test['total']),
        'per_class_accuracy': {str(k): float(v) for k, v in final_test['per_class_accuracy'].items()},
        'confusion_matrix': final_test['confusion_matrix'].tolist(),
        'epoch_results': [
            {
                'epoch': e['epoch'],
                'samples_processed': e['samples_processed'],
                'train_accuracy': float(e['train_results']['accuracy']),
                'train_avg_loss': float(e['train_results']['avg_loss']),
                'test_accuracy': float(e['test_results']['accuracy']),
                'test_avg_loss': float(e['test_results']['avg_loss']),
                'test_correct': int(e['test_results']['correct']),
                'test_total': int(e['test_results']['total'])
            }
            for e in epoch_results
        ],
        'training_history': {
            'final_train_accuracy': float(final_train_acc) if len(network.debug_stats['epoch_accuracies']) > 0 else None,
            'avg_plateau_activity': float(np.mean(network.debug_stats['plateau_activity'][-1000:])) if len(network.debug_stats['plateau_activity']) > 0 else None,
            'final_epoch_accuracy': float(final_epoch_acc) if len(epoch_results) > 0 else None,
            'best_epoch_accuracy': float(best_epoch_acc) if len(epoch_results) > 0 else None,
            'best_epoch_number': int(best_epoch_num) if len(epoch_results) > 0 else None
        }
    }
    
    stats_file = os.path.join(save_dir, 'training_summary.json')
    with open(stats_file, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"\nSummary statistics saved to {stats_file}")
    
    # Plot training curves
    if len(epoch_results) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = [e['epoch'] for e in epoch_results]
        samples_processed = [e['samples_processed'] for e in epoch_results]
        train_accs = [e['train_results']['accuracy'] for e in epoch_results]
        test_accs = [e['test_results']['accuracy'] for e in epoch_results]
        test_losses = [e['test_results']['avg_loss'] for e in epoch_results]
        
        ax1.plot(epochs, train_accs, 'o-', label='Train Accuracy', linewidth=2, markersize=6, color='green')
        ax1.plot(epochs, test_accs, 'o-', label='Test Accuracy', linewidth=2, markersize=6, color='blue')
        # Add final test result as a single point
        ax1.axhline(y=final_test['accuracy'], color='red', linestyle='--', linewidth=2, label=f'Final Test Accuracy ({final_test["accuracy"]:.2f}%)')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Accuracy Over Training (by Epoch)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(epochs, test_losses, 'o-', label='Test Loss', linewidth=2, markersize=6, color='orange')
        # Add final test result as a single point
        ax2.axhline(y=final_test['avg_loss'], color='red', linestyle='--', linewidth=2, label=f'Final Test Loss ({final_test["avg_loss"]:.4f})')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Test Loss Over Training (by Epoch)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plot_file = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {plot_file}")