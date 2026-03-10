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

# Try to import tables, fallback to h5py if there's a compatibility issue
USE_H5PY = False
try:
    import tables
except (ImportError, ValueError) as e:
    # Fallback to h5py if tables has numpy compatibility issues
    print(f"Warning: Could not use tables ({e}), using h5py instead...", flush=True)
    USE_H5PY = True

# Import h5py if needed (or always import it for fallback)
try:
    import h5py
except ImportError:
    if USE_H5PY:
        raise ImportError("h5py is required when tables is not available. Please install: pip install h5py")
    h5py = None

import urllib.request
import gzip
import shutil
from tensorflow.keras.utils import get_file

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

''' LOAD SHD DATA (MODIFIED FOR JAX) '''

class SHDDataLoader:
    def __init__(self, data_path: str, duration_ms: int = 1400):
        """Initialize SHD data loader
        
        Args:
            data_path: Path to directory containing SHD HDF5 files or cache directory
            duration_ms: Duration of sequences in milliseconds (default 1400ms = 1.4 seconds for SHD)
        """
        self.data_path = data_path
        self.duration_ms = duration_ms
        self.n_units = 700  # SHD has 700 input units
        
    def _get_and_gunzip(self, origin: str, filename: str, md5hash: str = None) -> str:
        """Download and decompress SHD dataset files"""
        # Use environment variable if set, otherwise try preferred locations
        if "SHD_CACHE_DIR" in os.environ:
            cache_dir = os.environ["SHD_CACHE_DIR"]
        elif os.path.exists("/share/neurocomputation/Tim/SHD_data"):
            cache_dir = "/share/neurocomputation/Tim/SHD_data"
        elif "SCRATCH" in os.environ:
            cache_dir = os.path.join(os.environ["SCRATCH"], "data")
        elif "TMPDIR" in os.environ:
            cache_dir = os.path.join(os.environ["TMPDIR"], "data")
        elif os.path.exists("/scratch"):
            cache_dir = "/scratch/data"
        elif os.path.exists("/tmp"):
            cache_dir = "/tmp/data"
        else:
            cache_dir = os.path.expanduser("~/data")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_subdir = "hdspikes"
        
        gz_file_path = get_file(filename, origin, md5_hash=md5hash, cache_dir=cache_dir, cache_subdir=cache_subdir)
        hdf5_file_path = gz_file_path[:-3]  # Remove .gz extension
        
        if not os.path.isfile(hdf5_file_path) or (os.path.exists(gz_file_path) and os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path)):
            print(f"Decompressing {gz_file_path}", flush=True)
            with gzip.open(gz_file_path, 'rb') as f_in, open(hdf5_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return hdf5_file_path
    
    def _load_sample_from_hdf5(self, units: np.ndarray, times: np.ndarray) -> List[np.ndarray]:
        """Convert SHD format (units, times arrays) to list of spike times per unit
        
        Args:
            units: Array of unit IDs for each spike
            times: Array of spike times in SECONDS (SHD dataset format)
            
        Returns:
            List of arrays, where each array contains spike times for that unit (in milliseconds)
        """
        # CRITICAL FIX: SHD dataset stores times in SECONDS, not milliseconds!
        # Convert from seconds to milliseconds by multiplying by 1000
        times_ms = times * 1000.0
        
        # Convert times from ms to time steps (assuming 1ms per time step)
        # Clip times to [0, duration_ms) range
        times_clipped = np.clip(times_ms, 0, self.duration_ms - 1)
        times_int = np.around(times_clipped).astype(int)
        
        # Group spikes by unit ID
        spike_data = [[] for _ in range(self.n_units)]
        for unit_id, spike_time in zip(units, times_int):
            if 0 <= unit_id < self.n_units:
                spike_data[unit_id].append(spike_time)
        
        # Convert to numpy arrays
        spike_data_arrays = [np.array(spikes) for spikes in spike_data]
        
        return spike_data_arrays

    def get_dataset(self, split: str = "train", max_samples_per_class: int = None, target_classes: List[int] = None) -> Tuple[List[List[np.ndarray]], List[int]]:
        """Get SHD dataset for training or testing with optional sample limit per class
        
        Args:
            split: "train" or "test"
            max_samples_per_class: If None, loads all available samples
            target_classes: List of class IDs to load (0-19: 0-9 English + 10-19 German), None for all classes
        """
        if target_classes is None:
            target_classes = list(range(20))  # All 20 classes (0-9 English + 0-9 German) by default
        
        # Determine HDF5 file path
        if split == "train":
            filename = "shd_train.h5.gz"
        elif split == "test":
            filename = "shd_test.h5.gz"
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train' or 'test'")
        
        # Download and decompress if needed
        base_url = "https://zenkelab.org/datasets"
        origin = f"{base_url}/{filename}"
        
        # Get MD5 hash if available
        md5hash = None
        try:
            response = urllib.request.urlopen(f"{base_url}/md5sums.txt")
            data = response.read()
            lines = data.decode('utf-8').split("\n")
            file_hashes = {line.split()[1]: line.split()[0] for line in lines if len(line.split()) == 2}
            md5hash = file_hashes.get(filename)
        except:
            pass  # Continue without MD5 hash if unavailable
        
        hdf5_file_path = self._get_and_gunzip(origin, filename, md5hash)
        
        print(f"Loading {split} dataset from {hdf5_file_path}...", flush=True)
        print(f"Target classes: {target_classes}", flush=True)
        
        # Open HDF5 file and load data (with fallback to h5py)
        if USE_H5PY:
            fileh = h5py.File(hdf5_file_path, mode='r')
            units = fileh['spikes']['units']
            times = fileh['spikes']['times']
            labels = fileh['labels']
        else:
            fileh = tables.open_file(hdf5_file_path, mode='r')
            units = fileh.root.spikes.units
            times = fileh.root.spikes.times
            labels = fileh.root.labels
        
        images = []
        labels_list = []
        
        # Filter by target classes and max_samples_per_class
        class_counts = {label: 0 for label in target_classes}
        
        load_start_time = time.time()
        n_samples = len(labels) if not USE_H5PY else labels.shape[0]
        for idx in range(n_samples):
            if USE_H5PY:
                label = int(labels[idx])
            else:
                label = int(labels[idx])
            
            # Skip if not in target classes
            if label not in target_classes:
                continue
            
            # Skip if we've reached max samples for this class
            if max_samples_per_class is not None and class_counts[label] >= max_samples_per_class:
                continue
            
            # Load and convert sample
            if USE_H5PY:
                sample_units = units[idx][:]
                sample_times = times[idx][:]
            else:
                sample_units = units[idx]
                sample_times = times[idx]
            spike_data = self._load_sample_from_hdf5(sample_units, sample_times)
            
            images.append(spike_data)
            labels_list.append(label)
            class_counts[label] += 1
            
            # Progress update
            if (len(images) % 100 == 0) and len(images) > 0:
                elapsed = time.time() - load_start_time
                rate = len(images) / elapsed if elapsed > 0 else 0
                remaining = (n_samples - idx) / rate if rate > 0 else 0
                print(f"  Loaded {len(images)} samples | Rate: {rate:.1f} samples/s | ETA: {remaining:.0f}s", flush=True)
        
        fileh.close()
        
        load_elapsed = time.time() - load_start_time
        print(f"Loaded {len(images)} total samples from {split} split in {load_elapsed:.1f}s ({len(images)/load_elapsed:.1f} samples/s)", flush=True)
        print(f"Class distribution: {dict(class_counts)}", flush=True)
        
        return images, labels_list


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
        Returns: (mu, v, h, o) each of shape (T, n_neurons)
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
            
            # Reset somatic potential if spike occurred
            v = v * (1 - o)
            
            # t_prime stays constant during plateau (already handled above)
            t_prime_next = t_prime
            
            return (mu, v, h, t_prime_next, mu_history), (mu, v, h, o, t_prime)
        
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
        
        mu, v, h, o, t_prime_history = outputs
        # Also return mu_history for backward pass
        _, _, _, _, mu_history_final = final_state
        return mu, v, h, o, mu_history_final, t_prime_history
    
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


''' JAX E-PROP NETWORK '''

class JAXEPropNetwork:
    """Complete JAX implementation of e-prop network"""
    
    def __init__(self, key, n_inputs: int = 700, n_hidden: int = 64, 
                 n_outputs: int = 20, T: int = 1400):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.T = T
        
        # Split keys for different components
        key_hidden, key_readout = random.split(key)
        
        # Configuration
        self.config = NeuronConfig()
        
        # Initialize layers
        self.hidden_layer = JAXTwoCompartmentalLayer(key_hidden, n_hidden, n_inputs, self.config)
        self.readout_layer = JAXLIFLayer(key_readout, n_outputs, n_hidden, self.config)
        
        # Learning parameters
        self.learning_rate_hidden_dendritic = 0.045
        self.learning_rate_hidden_somatic = 0.00015
        self.learning_rate_readout = 0.035
        self.weight_decay = 0.00001
        self.gradient_clip = 5.0
        
        # Loss function hyperparameters
        self.loss_temperature = 2.7        # 1.7 Temperature for softmax (spike count to probability)
        self.loss_count_bias = 0.18         # 0.18 Bias added to scaled counts
        self.loss_label_smoothing = 0.13   # 0.23 Label smoothing coefficient
        
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
        mu, v, h, hidden_o, mu_history, t_prime_history = JAXTwoCompartmentalLayer.forward_pass(
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
        
        return mu, v, h, hidden_o, readout_v, readout_o, mu_history, t_prime_history
    
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
        mu, v, h, hidden_o, readout_v, readout_o, _, _ = self._forward_with_params(params, x_input)
        
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
        
        mu, v, h, hidden_o, mu_history, t_prime_history = JAXTwoCompartmentalLayer.forward_pass(
            x_input, params.w_dend, params.w_soma, self.config, self.T, T_p
        )
        
        readout_v, readout_o = JAXLIFLayer.forward_pass(
            hidden_o, params.w_readout, self.config, self.T
        )
        
        return mu, v, h, hidden_o, readout_v, readout_o, mu_history, t_prime_history
    
    @staticmethod
    @jit(static_argnames=['T'])
    def _forward_with_params_jit(params: NetworkParams, x_input: jnp.ndarray, 
                                  config: NeuronConfig, T: int, T_p: jnp.ndarray) -> Tuple:
        """JIT-compiled forward pass with parameters"""
        mu, v, h, hidden_o, mu_history, t_prime_history = JAXTwoCompartmentalLayer.forward_pass(
            x_input, params.w_dend, params.w_soma, config, T, T_p
        )
        
        readout_v, readout_o = JAXLIFLayer.forward_pass(
            hidden_o, params.w_readout, config, T
        )
        
        return mu, v, h, hidden_o, readout_v, readout_o, mu_history, t_prime_history
    
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
        mu, v, h, hidden_o, readout_v, readout_o, mu_history, t_prime_history = self._forward_with_params(params, x_input)
        
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
        
        # Somatic surrogate gradient
        soma_input_vals = v + self.config.gamma * h - self.config.v_th
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
            mu, v, h, hidden_o, readout_v, readout_o, mu_history, t_prime_history = self._forward_with_params(params, x_input)
            
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
            
            # Somatic surrogate gradient
            soma_input_vals = v + self.config.gamma * h - self.config.v_th
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
        mu, v, h, hidden_o, readout_v, readout_o, _, _ = self.forward(x_input)
        
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
            
            x_network = create_shd_input_jax(x_input, self.T)
            x_network_jax = jnp.array(x_network)
            
            # Forward pass
            mu, v, h, hidden_o, readout_v, readout_o, _, _ = self.forward(x_network_jax)
            
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


''' JAX-OPTIMIZED UTILITY FUNCTIONS '''

@jit
def alpha_kernel_jax(t_vals: jnp.ndarray, tau: float) -> jnp.ndarray:
    """Alpha kernel in JAX"""
    k = (t_vals / tau) * jnp.exp(-t_vals / tau)
    return jnp.where(t_vals < 0, 0.0, k)

def create_shd_input_jax(shd_data: List[np.ndarray], T: int = 1400,
                           tau_alpha: float = 3.3, spike_amplitude: float = 5.0,
                           use_kernel: bool = True) -> np.ndarray:
    """Create SHD input optimized for JAX"""
    n_units = len(shd_data)
    x_input = np.zeros((T, n_units))
    
    if use_kernel:
        # Pre-compute kernel
        kernel_len = int(10 * tau_alpha)
        t_vals = np.arange(kernel_len)
        k = alpha_kernel_jax(jnp.array(t_vals), tau_alpha)
        peak_value = np.exp(-1)
        k_normalized = np.array(k) * (spike_amplitude / peak_value)
        
        # Apply kernel
        for unit_idx, spike_times in enumerate(shd_data):
            for spike_time in spike_times:
                spike_time_int = int(spike_time)
                if 0 <= spike_time_int < T:
                    kernel_start = spike_time_int
                    kernel_end = min(kernel_start + kernel_len, T)
                    kernel_length_used = kernel_end - kernel_start
                    
                    if kernel_length_used > 0:
                        x_input[kernel_start:kernel_end, unit_idx] += k_normalized[:kernel_length_used]
    else:
        # Direct spikes
        for unit_idx, spike_times in enumerate(shd_data):
            for spike_time in spike_times:
                if 0 <= spike_time < T:
                    x_input[int(spike_time), unit_idx] = spike_amplitude
    
    return x_input


''' UTILITY FUNCTIONS '''

def load_shd_data(data_path: str = None, train_samples_per_class: int = None, 
                    test_samples_per_class: int = None, target_classes: List[int] = None):
    """Load SHD data with specified samples per class
    
    Args:
        data_path: Path to cache directory. If None, auto-detects:
            - SHD_DATA_PATH environment variable
            - /share/neurocomputation/Tim/SHD_data (HPC)
            - ~/data/hdspikes (local)
        train_samples_per_class: If None, loads all available training samples
        test_samples_per_class: If None, loads all available test samples
        target_classes: List of class IDs to load (0-19: 0-9 English + 10-19 German), None for all classes
    """
    if target_classes is None:
        target_classes = list(range(20))  # All 20 classes (0-9 English + 10-19 German)
    
    if data_path is None:
        # Auto-detect data path (same logic as _get_and_gunzip)
        if "SHD_DATA_PATH" in os.environ:
            data_path = os.environ["SHD_DATA_PATH"]
        elif os.path.exists("/share/neurocomputation/Tim/SHD_data"):
            data_path = "/share/neurocomputation/Tim/SHD_data"
        else:
            data_path = os.path.expanduser("~/data/hdspikes")
    
    data_loader = SHDDataLoader(data_path)
    
    # Load training data (all samples if train_samples_per_class is None)
    train_images, train_labels = data_loader.get_dataset("train", max_samples_per_class=train_samples_per_class, target_classes=target_classes)
    
    # Load test data (all samples if test_samples_per_class is None)
    test_images, test_labels = data_loader.get_dataset("test", max_samples_per_class=test_samples_per_class, target_classes=target_classes)
    
    # No remapping needed - labels already match network classes (0-19: 0-9 English + 10-19 German)
    train_data = [(img, label) for img, label in zip(train_images, train_labels)]
    test_data = [(img, label) for img, label in zip(test_images, test_labels)]
    
    return train_data, test_data


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

def train_network_jax(network: JAXEPropNetwork, train_data: List[Tuple[np.ndarray, int]],
                     test_data: List[Tuple[np.ndarray, int]], epochs: int = 10, batch_size: int = 32,
                     run_dir: str = "model"):
    """Train the JAX network on SHD data with epochs and test evaluation after each epoch
    
    Args:
        network: JAXEPropNetwork instance
        train_data: List of (input, label) tuples
        test_data: List of (input, label) tuples
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
        
        # Optional: Verify shuffle order matches (for debugging)
        # Note: This requires running NumPy model first to get reference order
        
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
                x_network = create_shd_input_jax(x_input, network.T)
                x_network_jax = jnp.array(x_network)
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
                    x_network = create_shd_input_jax(x_input, network.T)
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


''' MAIN EXECUTION '''

if __name__ == "__main__":
    # Configuration - auto-detect data path (HPC-aware)
    if "SHD_DATA_PATH" in os.environ:
        data_path = os.environ["SHD_DATA_PATH"]
    elif os.path.exists("/share/neurocomputation/Tim/SHD_data"):
        data_path = "/share/neurocomputation/Tim/SHD_data"
    else:
        data_path = os.path.expanduser("~/data/hdspikes")
    # Create model directory in the script's directory (for saving new models)
    _script_dir = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(_script_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Check for existing model in main directory (for evaluation)
    # Models are saved to model/ folder, but check happens in main directory
    model_check_path = "shd_2comp_model_jax.pkl"
    
    # Print hyperparameters at the start (using actual values from classes)
    # Create temporary instances to get default values
    temp_config = NeuronConfig()
    temp_network = JAXEPropNetwork(random.PRNGKey(0), n_inputs=700, n_hidden=64, n_outputs=20, T=1400)
    # Training parameters (read from environment or use defaults)
    epochs = int(os.getenv("EPOCHS", "30"))
    batch_size = int(os.getenv("BATCH_SIZE", "4"))
    hyperparams_lines = [
        "", "HYPERPARAMETERS", "=" * 80,
        f"Random seed: {RANDOM_SEED}",
        "Learning rates:",
        f"  Hidden (dendritic): {temp_network.learning_rate_hidden_dendritic}",
        f"  Hidden (somatic): {temp_network.learning_rate_hidden_somatic}",
        f"  Readout: {temp_network.learning_rate_readout}",
        "Neuron parameters:",
        f"  tau_soma: {temp_config.tau_soma}",
        f"  tau_dend: {temp_config.tau_dend}",
        f"  tau_plat: uniform({temp_config.tau_plat_min}, {temp_config.tau_plat_max}) ms per neuron",
        f"  tau_m: {temp_config.tau_m}",
        f"  v_th: {temp_config.v_th}",
        f"  mu_th: {temp_config.mu_th}",
        f"  gamma: {temp_config.gamma}",
        f"  beta_s: {temp_config.beta_s}",
        f"  beta_d: {temp_config.beta_d}",
        "Loss parameters:",
        f"  Temperature: {temp_network.loss_temperature}",
        f"  Count bias: {temp_network.loss_count_bias}",
        f"  Label smoothing: {temp_network.loss_label_smoothing}",
        "Other:",
        f"  Weight decay: {temp_network.weight_decay}",
        f"  Gradient clip: {temp_network.gradient_clip}",
        f"  Sequence length (T): {temp_network.T} ms",
        "Training parameters:",
        f"  Epochs: {epochs}",
        f"  Batch size: {batch_size}",
        "=" * 80, ""
    ]
    for line in hyperparams_lines:
        print(line, flush=True)
    
    # Check if model exists in main directory
    if os.path.exists(model_check_path):
        print(f"Loading existing model from {model_check_path} (main directory)...", flush=True)
        network = JAXEPropNetwork.load(model_check_path, key)
        
        # Load test data for evaluation (all samples)
        _, test_data = load_shd_data(data_path, train_samples_per_class=None, test_samples_per_class=None)
        
        print(f"Evaluating on {len(test_data)} test samples...", flush=True)
        test_results = network.evaluate(test_data)
        
        print(f"\n{'='*80}", flush=True)
        print("MODEL EVALUATION", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Test Accuracy: {test_results['accuracy']:.2f}%", flush=True)
        print(f"Test Loss: {test_results['avg_loss']:.4f}", flush=True)
        print(f"Correct: {test_results['correct']}/{test_results['total']}", flush=True)
        
        print(f"\nPer-Class Accuracy:", flush=True)
        for digit, acc in sorted(test_results['per_class_accuracy'].items()):
            print(f"  Class {digit}: {acc:.2f}%", flush=True)
        
    else:
        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(model_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        model_save_path = os.path.join(run_dir, "shd_2comp_model_jax.pkl")
        save_dir = run_dir
        print(f"Created run directory: {run_dir}", flush=True)
        with open(os.path.join(run_dir, "hyperparameters.txt"), "w") as f:
            f.write("\n".join(hyperparams_lines))
        
        # Load data (all samples)
        print(f"Loading SHD data...", flush=True)
        train_data, test_data = load_shd_data(data_path, 
                                               train_samples_per_class=None,
                                               test_samples_per_class=None)
        
        print(f"\nDataset loaded:", flush=True)
        print(f"  Training: {len(train_data)} samples (all available)", flush=True)
        print(f"  Test: {len(test_data)} samples (all available)", flush=True)
        
        # Create network
        n_inputs = len(train_data[0][0])  # Number of input units (should be 700 for SHD)
        print(f"\nCreating JAX network with {n_inputs} inputs...", flush=True)
        
        # Initialize weights using NumPy's exact initialization logic
        # IMPORTANT: Set seed ONCE here (using configurable seed)
        # Then initialize weights, which will consume random calls
        # This ensures the random state after initialization matches NumPy exactly
        print(f"  Setting random seed ({RANDOM_SEED}) and initializing weights using NumPy's method...", flush=True)
        np.random.seed(RANDOM_SEED)  # Set seed once, using configurable seed
        
        # Verify random state matches NumPy after seed
        # (NumPy sets seed at module level, so state should be identical)
        expected_state_after_seed = np.random.get_state()
        w_dend_np, w_soma_np, w_readout_np = initialize_numpy_weights(
            n_inputs=n_inputs, n_hidden=64, n_outputs=20
        )
        state_after_init = np.random.get_state()
        print("  ✅ NumPy-style weights initialized (148 random calls consumed: 2*64 hidden + 20 readout)", flush=True)
        print(f"  Random state after init: {state_after_init[1][:3] if len(state_after_init) > 1 else 'N/A'}...", flush=True)
        
        # Create JAX network
        print(f"  Creating JAX network with seed {RANDOM_SEED}...", flush=True)
        network_key = random.PRNGKey(RANDOM_SEED)
        network = JAXEPropNetwork(network_key, n_inputs=n_inputs, n_hidden=64, 
                                n_outputs=20, T=1400)
        
        # Sync weights from NumPy initialization to JAX network
        print("  Syncing NumPy-initialized weights to JAX network...", flush=True)
        network.hidden_layer.w_dend = jnp.array(w_dend_np)
        network.hidden_layer.w_soma = jnp.array(w_soma_np)
        network.readout_layer.w = jnp.array(w_readout_np)
        print("  ✅ Weights synced - JAX network now uses NumPy initialization", flush=True)
        
        # Train network
        # Epochs and batch size are already read from environment variables above
        print(f"\nTraining JAX network with epochs={epochs}, batch_size={batch_size}...", flush=True)
        epoch_results, final_test_results, best_model_path, best_accuracy = train_network_jax(
            network, train_data, test_data, 
            epochs=epochs,
            batch_size=batch_size,
            run_dir=run_dir
        )
        
        # Save final model
        network.save(model_save_path)
        
        # Print and save summary statistics
        print_summary_statistics(network, epoch_results, final_test_results, save_dir)
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Final model saved to: {model_save_path}")
        if best_model_path:
            print(f"Best model saved to: {best_model_path} (accuracy: {best_accuracy:.2f}%)")
        print(f"To reload and evaluate: run the script again (it will detect the saved model)")

