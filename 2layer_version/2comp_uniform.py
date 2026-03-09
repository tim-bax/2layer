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

# Helper function to initialize weights exactly like NumPy network does
def initialize_numpy_weights(n_inputs: int, n_hidden: int, n_outputs: int):
    """
    Initialize weights using NumPy's exact initialization logic.
    This replicates the sequential np.random.normal() calls from self_2comp_efficient_debug.py
    
    IMPORTANT: This function does NOT reset the seed. It uses the current random state.
    The seed should be set ONCE at module level (np.random.seed(42)) before calling this.
    This ensures the random state matches NumPy's sequence exactly.
    """
    # Initialize hidden layer weights (dendritic and somatic)
    # Each neuron is initialized sequentially, matching NumPy behavior
    # NumPy creates: n_hidden neurons × 2 weight arrays = 2*n_hidden calls to np.random.normal
    w_dend = np.zeros((n_hidden, n_inputs))
    w_soma = np.zeros((n_hidden, n_inputs))
    
    for i in range(n_hidden):
        # Xavier initialization for each neuron (sequential calls)
        fan_in = n_inputs
        xavier_std = np.sqrt(2.0 / fan_in)
        w_dend[i] = np.random.normal(0.0, xavier_std * 0.15, size=n_inputs)
        w_soma[i] = np.random.normal(0.0, xavier_std * 0.15, size=n_inputs)
    
    # Initialize readout layer weights
    # Each readout neuron is initialized sequentially
    # NumPy creates: n_outputs readout neurons × 1 weight array = n_outputs calls to np.random.normal
    w_readout = np.zeros((n_outputs, n_hidden))
    for i in range(n_outputs):
        fan_in = n_hidden
        xavier_std = np.sqrt(2.0 / fan_in)
        w_readout[i] = np.random.normal(0.0, xavier_std * 0.15, size=n_hidden)
    
    # Total: (2*n_hidden + n_outputs) calls to np.random.normal consumed
    # For n_hidden=64, n_outputs=20: 2*64 + 20 = 148 calls
    # After this function, random state matches NumPy after network creation
    return w_dend, w_soma, w_readout

# Auto-detect and configure JAX platform (GPU if available, else CPU)
# This allows the code to work on both CPU and GPU clusters
devices = jax.devices()
if len(devices) > 0 and devices[0].platform == 'gpu':
    print(f"JAX: Using GPU - {devices}", flush=True)
    # JAX will auto-use GPU, no need to set platform
else:
    print(f"JAX: Using CPU - {devices}", flush=True)
    jax.config.update('jax_platform_name', 'cpu')

# Enable float64 precision to match NumPy's default (for exact numerical matching)
jax.config.update('jax_enable_x64', True)

# Set random seed (configurable via environment variable or default to 42)
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
key = random.PRNGKey(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

''' JAX-OPTIMIZED NEURON IMPLEMENTATIONS '''

@struct.dataclass
class NetworkParams:
    """Immutable parameters for the network"""
    w_dend: jnp.ndarray  # (n_hidden, n_inputs)
    w_soma: jnp.ndarray  # (n_hidden, n_inputs)
    w_readout: jnp.ndarray  # (n_outputs, n_hidden)

@struct.dataclass
class NeuronConfig:
    """Configuration parameters for neurons"""
    mu_th: float = 1.0
    v_th: float = 1.0
    gamma: float = 0.5
    tau_soma: float = 15.0
    tau_dend: float = 15.0
    tau_plat_min: float = 100.0  # Minimum plateau duration (ms)
    tau_plat_max: float = 350.0  # Maximum plateau duration (ms)
    dt: float = 1.0
    tau_m: float = 20.0
    v_reset: float = 0.0
    beta_s: float = 0.36
    beta_d: float = 0.75
    beta: float = 0.36

class JAXTwoCompartmentalLayer:
    """JAX implementation of two-compartmental neurons"""
    
    def __init__(self, key, n_neurons: int, n_inputs: int, config: NeuronConfig):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.config = config
        
        # Calculate decay constants
        self.alpha_s = jnp.exp(-config.dt / config.tau_soma)
        self.alpha = jnp.exp(-config.dt / config.tau_dend)
        
        # Generate per-neuron plateau durations from uniform distribution
        key1, key2, key3 = random.split(key, 3)
        # Draw tau_plat for each neuron from uniform distribution [tau_plat_min, tau_plat_max]
        tau_plat_values = random.uniform(
            key3, 
            shape=(n_neurons,), 
            minval=config.tau_plat_min, 
            maxval=config.tau_plat_max
        )
        # Convert to time steps (integer)
        self.T_p = (tau_plat_values / config.dt).astype(jnp.int32)  # (n_neurons,)
        
        # Initialize weights
        xavier_std = jnp.sqrt(2.0 / n_inputs)
        scale = xavier_std * 0.15
        
        self.w_dend = random.normal(key1, (n_neurons, n_inputs)) * scale
        self.w_soma = random.normal(key2, (n_neurons, n_inputs)) * scale
    
    @staticmethod
    @jit(static_argnames=['T'])
    def forward_pass(x_input: jnp.ndarray, w_dend: jnp.ndarray, w_soma: jnp.ndarray,
                    config: NeuronConfig, T: int, T_p: jnp.ndarray):
        """
        JIT-compiled forward pass using lax.scan
        x_input: (T, n_inputs)
        T_p: Plateau duration in time steps per neuron, shape (n_neurons,)
        Returns: (mu, v_final, h, o, mu_trajectory, t_prime_history, v_history).
        mu_trajectory is (T, n_neurons), same as mu from outputs; use for mu_at_tprime lookup in backward.
        v_history is (T, n_neurons) pre-reset for somatic surrogate.
        """
        n_neurons = w_dend.shape[0]
        
        # Calculate decay constants
        alpha_s = jnp.exp(-config.dt / config.tau_soma)
        alpha = jnp.exp(-config.dt / config.tau_dend)
        
        # Pre-compute inputs for efficiency
        dend_inputs = x_input @ w_dend.T  # (T, n_neurons)
        soma_inputs = x_input @ w_soma.T  # (T, n_neurons)
        
        # Pre-compute neuron_indices outside scan to avoid recreating it every step
        neuron_indices_static = jnp.arange(n_neurons)
        
        def step(carry, inputs):
            mu_prev, v_prev, h_prev, t_prime_prev, mu_history = carry
            dend_in, soma_in, t = inputs  # t is the time step index
            
            # Update t_prime: keep previous value if in plateau, otherwise set to current time
            # Match NumPy: at t=0, t_prime=0; at t>0, use h_prev to determine
            # Explicitly cast to int64 to match NumPy's Python int type (important when float64 is enabled)
            t_prime = jnp.where(t == 0, 0, jnp.where(h_prev == 1, t_prime_prev, t)).astype(jnp.int64)
            
            # Update dendritic potential
            # Match NumPy: at t=0, mu = input (no decay, no h term); at t>0, use standard formula
            mu = jnp.where(t > 0, alpha * mu_prev + (1 - h_prev) * dend_in, dend_in)
            
            # CRITICAL FIX: Compute h BEFORE storing mu_history[t] to match NumPy exactly
            # This ensures mu_history[t] hasn't been updated when computing h
            # Update plateau state
            # Get mu at plateau initiation time: mu_history[t_prime] if t_prime < t, else current mu
            # OPTIMIZED: Use vectorized advanced indexing
            t_prime_int = t_prime.astype(jnp.int32)
            # Access mu_history BEFORE updating it (matching NumPy behavior)
            # t_prime_int is (n_neurons,) with different values per neuron
            # We need mu_history[t_prime_int[i], i] for each neuron i
            # Use advanced indexing with broadcasting
            mu_at_tprime_from_history = mu_history[t_prime_int, neuron_indices_static]  # (n_neurons,)
            mu_at_initiation = jnp.where(
                t_prime < t,
                mu_at_tprime_from_history,
                mu
            )
            
            # Check if plateau should be active: mu_at_initiation >= threshold AND duration <= T_p
            # T_p is now per-neuron: (n_neurons,)
            plateau_duration = t - t_prime  # (n_neurons,)
            # Compare each neuron's plateau duration with its own T_p value
            # Explicitly cast to int64 to match NumPy's Python int type (important when float64 is enabled)
            h = jnp.where(
                (mu_at_initiation >= config.mu_th) & (plateau_duration <= T_p) & (plateau_duration >= 0),
                1, 0
            ).astype(jnp.int64)
            
            # OPTIMIZED: Store mu_history more efficiently
            # Instead of creating a new array with .at[t].set(), we'll accumulate mu in the output
            # and build mu_history after the scan. But we still need it during scan for h computation.
            # For now, we keep the .at[].set() but this is the main bottleneck.
            # TODO: Consider computing mu_history from outputs after scan if possible
            mu_history = mu_history.at[t].set(mu)
            
            # Update somatic potential
            # Match NumPy: at t=0, v = input (no decay); at t>0, use standard formula
            v = jnp.where(t > 0, alpha_s * v_prev + soma_in, soma_in)
            
            # Generate output spike
            # Explicitly cast to int64 to match NumPy's Python int type (important when float64 is enabled)
            o = jnp.where(v >= config.v_th - config.gamma * h, 1, 0).astype(jnp.int64)
            
            # Store v before reset (same as NumPy: "store v before reset so trace shows rise to threshold")
            v_for_history = v
            # Reset somatic potential if spike occurred
            v = v * (1 - o)
            
            # t_prime stays constant during plateau (already handled above)
            t_prime_next = t_prime
            
            return (mu, v, h, t_prime_next, mu_history), (mu, v_for_history, h, o, t_prime)
        
        # Initial state - T is now static, so we can use it directly
        init_mu_history = jnp.zeros((T, n_neurons))
        
        init_state = (
            jnp.zeros(n_neurons),      # mu
            jnp.zeros(n_neurons),      # v
            jnp.zeros(n_neurons, dtype=jnp.int64),  # h - use int64 to match NumPy's Python int
            jnp.zeros(n_neurons, dtype=jnp.int64),  # t_prime - use int64 to match NumPy's Python int
            init_mu_history  # mu_history - need full history to access mu[t_prime]
        )
        
        # Create time indices for scan - T is static now
        # Use int64 to match NumPy's Python int type (important when float64 is enabled)
        time_indices = jnp.arange(T, dtype=jnp.int64)
        
        # Scan over time with time indices
        final_state, outputs = lax.scan(
            step, 
            init_state, 
            (dend_inputs, soma_inputs, time_indices)
        )
        
        mu, v_history, h, o, t_prime_history = outputs  # v_history: (T, n_neurons)
        # mu is (T, n_neurons) = same as mu_history from carry; return mu once for backward (mu_at_tprime lookup)
        return mu, v_history[-1], h, o, mu, t_prime_history, v_history
    
    @staticmethod
    @jit
    def surrogate_sigma(x: jnp.ndarray, beta: float = 0.5) -> jnp.ndarray:
        """Super-spike surrogate gradient - vectorized"""
        return 1.0 / (1.0 + beta * jnp.abs(x))**2
    
    @staticmethod
    @jit(static_argnames=[])
    def compute_dmu_tprime_dw(x_input: jnp.ndarray, h_history: jnp.ndarray, 
                              t_prime_history: jnp.ndarray, alpha: float) -> jnp.ndarray:
        """
        Compute ∂μ_t'/∂ω for all time steps and all synapses
        Returns shape (T, n_neurons, n_inputs)
        Based on theoretical formulation from self_2comp_efficient_debug.py
        """
        T = x_input.shape[0]
        n_neurons = h_history.shape[1]
        n_inputs = x_input.shape[1]
        
        # Determine which time steps are not in plateau (t_prime == t means new plateau or no plateau)
        # For each neuron, check if t_prime[t] == t
        time_indices = jnp.arange(T)[:, None]  # (T, 1)
        no_plateau_mask = (t_prime_history == time_indices)  # (T, n_neurons)
        
        def step_dmu(carry, inputs):
            dmu_dw_prev = carry  # (n_neurons, n_inputs)
            x_t, h_prev, no_plateau_t = inputs  # x_t: (n_inputs,), h_prev: (n_neurons,), no_plateau_t: (n_neurons,)
            
            # For each neuron, compute update
            # dmu_dw = alpha * dmu_dw_prev + (1 - h_prev) * x_t
            # Broadcast x_t: (n_inputs,) -> (1, n_inputs) -> (n_neurons, n_inputs)
            x_t_broadcast = jnp.broadcast_to(x_t[None, :], (n_neurons, n_inputs))  # (n_neurons, n_inputs)
            h_prev_broadcast = h_prev[:, None]  # (n_neurons, 1)
            
            # CRITICAL FIX: Match NumPy behavior exactly
            # NumPy: During loop, only update where no_plateau_mask[t] is True
            # If in plateau, dmu_dw_history[t] stays at 0 (not updated)
            # When updating, use dmu_dw_history[t-1], which might be 0 if t-1 was in plateau
            # After loop, fill plateau periods from t_prime
            
            # For standard update: use previous value (which might be 0 if previous was in plateau)
            # For plateau: leave at 0 (will be filled later from t_prime)
            dmu_dw_new = jnp.where(
                no_plateau_t[:, None],  # (n_neurons, 1)
                alpha * dmu_dw_prev + (1 - h_prev_broadcast) * x_t_broadcast,  # Standard update
                jnp.zeros_like(dmu_dw_prev)  # Leave at 0 during plateau (matches NumPy)
            )
            
            return dmu_dw_new, dmu_dw_new
        
        # Initial state - MATCHING NUMPY: Initialize to zeros (NumPy has this "bug" but performs better)
        # NumPy version leaves dmu_dw[0] = 0, which might act as implicit regularization
        # At t=0: dmu_dw[0] = 0 (matching NumPy behavior)
        init_dmu_dw = jnp.zeros((n_neurons, n_inputs))
        
        # Prepare inputs for scan: x_input is (T, n_inputs), need to process per time step
        h_prev = jnp.concatenate([jnp.zeros((1, n_neurons)), h_history[:-1]], axis=0)  # (T, n_neurons)
        
        # Scan to compute standard updates
        _, dmu_dw_standard = lax.scan(
            step_dmu,
            init_dmu_dw,
            (x_input, h_prev, no_plateau_mask)
        )
        
        # Fill plateau periods: copy from t_prime time using vectorized indexing
        # CRITICAL: t_prime always points to plateau initiation time, which is NOT in plateau
        # So t_prime should always point to a standard update (not 0)
        # OPTIMIZED: Use vectorized indexing instead of nested vmaps for much better performance
        
        # dmu_dw_standard shape: (T, n_neurons, n_inputs) from scan
        # t_prime_history shape: (T, n_neurons) - each entry is a time index
        # We need to get dmu_dw_standard[t_prime[t, n], n, :] for all t, n
        
        # Create neuron index array that matches t_prime_history shape
        neuron_indices_2d = jnp.broadcast_to(jnp.arange(n_neurons)[None, :], (T, n_neurons))  # (T, n_neurons)
        
        # Use advanced indexing: for each (t, n), get dmu_dw_standard[t_prime[t, n], n, :]
        # Convert t_prime_history to int32 for indexing
        t_prime_int = t_prime_history.astype(jnp.int32)  # (T, n_neurons)
        
        # Advanced indexing: dmu_dw_standard[t_prime_int, neuron_indices_2d]
        # This gives us (T, n_neurons, n_inputs) where each entry is dmu_dw_standard[t_prime[t, n], n, :]
        dmu_dw_plateau_values = dmu_dw_standard[t_prime_int, neuron_indices_2d]  # (T, n_neurons, n_inputs)
        
        # Use plateau values where in plateau, otherwise use standard values
        in_plateau_mask = ~no_plateau_mask  # (T, n_neurons)
        dmu_dw_history = jnp.where(
            in_plateau_mask[:, :, None],  # (T, n_neurons, 1) - broadcast to n_inputs
            dmu_dw_plateau_values,  # (T, n_neurons, n_inputs)
            dmu_dw_standard  # (T, n_neurons, n_inputs)
        )
        
        return dmu_dw_history
    
    @staticmethod
    @jit
    def compute_eligibility_traces(x_input: jnp.ndarray, alpha: float) -> jnp.ndarray:
        """Compute eligibility traces efficiently"""
        T = x_input.shape[0]
        # Vectorized computation using convolution-like approach
        # E_t = ∑_{s≤t} α^(t-s) x_s
        # This can be computed efficiently with scan
        # Convert input to float64 to ensure consistent types (input may be int64 spikes)
        x_input_float = x_input.astype(jnp.float64)
        
        def scan_fn(carry, x_t):
            E_prev = carry
            E_t = alpha * E_prev + x_t
            return E_t, E_t
        
        # Initialize and scan - use float64 for eligibility traces
        init_E = jnp.zeros_like(x_input_float[0], dtype=jnp.float64)
        _, E = lax.scan(scan_fn, init_E, x_input_float)
        
        return E  # (T, n_inputs)


class JAXLIFLayer:
    """JAX implementation of LIF neurons"""
    
    def __init__(self, key, n_neurons: int, n_inputs: int, config: NeuronConfig):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.config = config
        
        # Calculate decay constant
        self.alpha = jnp.exp(-config.dt / config.tau_m)
        
        # Initialize weights
        xavier_std = jnp.sqrt(2.0 / n_inputs)
        scale = xavier_std * 0.15
        self.w = random.normal(key, (n_neurons, n_inputs)) * scale
    
    @staticmethod
    @jit
    def forward_pass(spike_inputs: jnp.ndarray, w: jnp.ndarray,
                    config: NeuronConfig, T: int):
        """
        JIT-compiled forward pass for LIF neurons
        spike_inputs: (T, n_inputs)
        Returns: (v, o) each of shape (T, n_neurons)
        """
        n_neurons = w.shape[0]
        alpha = jnp.exp(-config.dt / config.tau_m)
        
        # Pre-compute inputs
        inputs = spike_inputs @ w.T  # (T, n_neurons)
        
        def step(carry, inp):
            v_prev = carry
            v = alpha * v_prev + inp
            # Explicitly cast to int64 to match NumPy's Python int type (important when float64 is enabled)
            o = jnp.where(v >= config.v_th, 1, 0).astype(jnp.int64)
            v = v * (1 - o) + config.v_reset * o
            return v, (v, o)
        
        # Initial state
        init_v = jnp.zeros(n_neurons)
        
        # Scan over time
        final_v, outputs = lax.scan(step, init_v, inputs)
        
        v, o = outputs
        return v, o




''' JAX-OPTIMIZED UTILITY FUNCTIONS '''

@jit
def alpha_kernel_jax(t_vals: jnp.ndarray, tau: float) -> jnp.ndarray:
    """Alpha kernel in JAX"""
    k = (t_vals / tau) * jnp.exp(-t_vals / tau)
    return jnp.where(t_vals < 0, 0.0, k)
