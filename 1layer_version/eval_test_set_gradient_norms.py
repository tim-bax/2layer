#!/usr/bin/env python3
"""
Run the full test set through the loaded heid_2comp model; report accuracy and
gradient norm statistics (full gradient norms, per-readout-neuron norms, sigma' readout).
"""
import os
import sys
import argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from model import JAXEPropNetwork

HEIDELBERG_DATA_PATH = "/Users/tbax/Documents/Heidelberg_Data"


def get_data_path():
    return (
        os.environ.get("SHD_DATA_PATH") or
        (HEIDELBERG_DATA_PATH if os.path.exists(HEIDELBERG_DATA_PATH) else None) or
        ("/share/neurocomputation/Tim/SHD_data" if os.path.exists("/share/neurocomputation/Tim/SHD_data") else None) or
        os.path.expanduser("~/data/hdspikes")
    )


def load_test_data(T: int, test_samples_per_class=None):
    """Load test set as list of (x_jax, label)."""
    from data import load_shd_data, create_shd_input_jax
    data_path = get_data_path()
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    _, test_raw = load_shd_data(
        data_path,
        train_samples_per_class=None,
        test_samples_per_class=test_samples_per_class,
    )
    test_data = []
    for x_raw, label in test_raw:
        x = create_shd_input_jax(x_raw, T=T)
        test_data.append((jnp.array(x), int(label)))
    return test_data


def main():
    parser = argparse.ArgumentParser(description="Eval test set: accuracy + gradient norms")
    parser.add_argument("--pkl", type=str, default=None, help="Path to model pkl. If omitted, uses heid_2comp_58_65.pkl in this dir if it exists.")
    parser.add_argument("--test_per_class", type=int, default=None, help="Max test samples per class (default: all)")
    args = parser.parse_args()

    default_pkl = os.path.join(_SCRIPT_DIR, "heid_2comp_58_65.pkl")
    pkl_path = args.pkl if args.pkl is not None else default_pkl
    if not os.path.isfile(pkl_path):
        print(f"Model not found: {pkl_path}")
        pkls = [f for f in os.listdir(_SCRIPT_DIR) if f.endswith(".pkl")]
        if pkls:
            print(f"Available .pkl in this directory: {pkls}")
            print("Run with: python eval_test_set_gradient_norms.py --pkl <path>")
        return

    print("Loading model...", flush=True)
    network = JAXEPropNetwork.load(pkl_path)
    params = network.get_params()
    T = network.T
    n_outputs = network.n_outputs

    # Model weights summary (you "get" the weights when you select/load a model)
    print("Model weights (loaded):")
    print(f"  w_dend:    shape {params.w_dend.shape},  norm = {float(jnp.linalg.norm(params.w_dend)):.4f},  range [{float(jnp.min(params.w_dend)):.4f}, {float(jnp.max(params.w_dend)):.4f}]")
    print(f"  w_soma:    shape {params.w_soma.shape},  norm = {float(jnp.linalg.norm(params.w_soma)):.4f},  range [{float(jnp.min(params.w_soma)):.4f}, {float(jnp.max(params.w_soma)):.4f}]")
    print(f"  w_readout: shape {params.w_readout.shape},  norm = {float(jnp.linalg.norm(params.w_readout)):.4f},  range [{float(jnp.min(params.w_readout)):.4f}, {float(jnp.max(params.w_readout)):.4f}]")
    print()

    print("Loading test data...", flush=True)
    try:
        test_data = load_test_data(T, test_samples_per_class=args.test_per_class)
    except Exception as e:
        print(f"Failed to load test data: {e}")
        return
    print(f"  Test set size: {len(test_data)}", flush=True)

    # Accumulate: correct count, and per-sample gradient norms / sigma' norm
    correct = 0
    n_samples = len(test_data)
    grad_dend_norms = []
    grad_soma_norms = []
    grad_readout_norms = []
    grad_readout_per_neuron_norms = []  # list of (n_outputs,) arrays
    sigma_prime_readout_norms = []
    # Components: soma (sigma', E_soma, eff_err), dendrite (h', dmu_tprime_dw), readout (E_readout)
    sigma_prime_hidden_norms = []
    E_soma_norms = []
    effective_error_norms = []
    h_prime_norms = []
    dmu_tprime_dw_norms = []  # E_dendrite (dendritic sensitivity)
    E_readout_norms = []
    # Mean (signed) per sample: gradients and all components (average over elements / "per weight")
    mean_grad_dend_list = []
    mean_grad_soma_list = []
    mean_grad_readout_list = []
    mean_sigma_list = []
    mean_E_soma_list = []
    mean_eff_list = []
    mean_h_prime_list = []
    mean_dmu_list = []
    mean_E_readout_list = []
    mean_sigma_readout_list = []
    mean_abs_grad_dend_list = []
    mean_abs_grad_soma_list = []
    mean_abs_grad_readout_list = []
    elig_soma_one_list = []
    elig_dend_one_list = []
    elig_readout_one_list = []

    for i, (x_input, target) in enumerate(test_data):
        target = jnp.array(target, dtype=jnp.int32)
        out = network.compute_gradients_one_sample(params, x_input, target, return_soma_components=True)
        grad_dend, grad_soma, grad_readout, sigma_prime_readout, prediction, comp = out
        (n_sigma, n_E_soma, n_eff, n_h_prime, n_dmu_tprime_dw, n_E_readout,
         mean_grad_dend, mean_grad_soma, mean_grad_readout,
         mean_sigma, mean_E_soma, mean_eff, mean_h_prime, mean_dmu, mean_E_readout, mean_sigma_readout,
         mean_abs_grad_dend, mean_abs_grad_soma, mean_abs_grad_readout,
         elig_soma_one_synapse, elig_dend_one_synapse, elig_readout_one_synapse) = comp
        if prediction == int(np.array(target).item()):
            correct += 1

        grad_dend_norms.append(float(jnp.linalg.norm(grad_dend)))
        grad_soma_norms.append(float(jnp.linalg.norm(grad_soma)))
        grad_readout_norms.append(float(jnp.linalg.norm(grad_readout)))
        grad_readout_per_neuron_norms.append(np.array([float(jnp.linalg.norm(grad_readout[j, :])) for j in range(n_outputs)]))
        sigma_prime_readout_norms.append(float(jnp.linalg.norm(sigma_prime_readout)))
        sigma_prime_hidden_norms.append(n_sigma)
        E_soma_norms.append(n_E_soma)
        effective_error_norms.append(n_eff)
        h_prime_norms.append(n_h_prime)
        dmu_tprime_dw_norms.append(n_dmu_tprime_dw)
        E_readout_norms.append(n_E_readout)
        mean_grad_dend_list.append(mean_grad_dend)
        mean_grad_soma_list.append(mean_grad_soma)
        mean_grad_readout_list.append(mean_grad_readout)
        mean_sigma_list.append(mean_sigma)
        mean_E_soma_list.append(mean_E_soma)
        mean_eff_list.append(mean_eff)
        mean_h_prime_list.append(mean_h_prime)
        mean_dmu_list.append(mean_dmu)
        mean_E_readout_list.append(mean_E_readout)
        mean_sigma_readout_list.append(mean_sigma_readout)
        mean_abs_grad_dend_list.append(mean_abs_grad_dend)
        mean_abs_grad_soma_list.append(mean_abs_grad_soma)
        mean_abs_grad_readout_list.append(mean_abs_grad_readout)
        elig_soma_one_list.append(elig_soma_one_synapse)
        elig_dend_one_list.append(elig_dend_one_synapse)
        elig_readout_one_list.append(elig_readout_one_synapse)

        if (i + 1) % 100 == 0 or (i + 1) == n_samples:
            print(f"  Processed {i + 1}/{n_samples} samples...", flush=True)

    # Convert to arrays
    grad_dend_norms = np.array(grad_dend_norms)
    grad_soma_norms = np.array(grad_soma_norms)
    grad_readout_norms = np.array(grad_readout_norms)
    sigma_prime_readout_norms = np.array(sigma_prime_readout_norms)
    sigma_prime_hidden_norms = np.array(sigma_prime_hidden_norms)
    E_soma_norms = np.array(E_soma_norms)
    effective_error_norms = np.array(effective_error_norms)
    h_prime_norms = np.array(h_prime_norms)
    dmu_tprime_dw_norms = np.array(dmu_tprime_dw_norms)
    E_readout_norms = np.array(E_readout_norms)
    mean_grad_dend_arr = np.array(mean_grad_dend_list)
    mean_grad_soma_arr = np.array(mean_grad_soma_list)
    mean_grad_readout_arr = np.array(mean_grad_readout_list)
    mean_sigma_arr = np.array(mean_sigma_list)
    mean_E_soma_arr = np.array(mean_E_soma_list)
    mean_eff_arr = np.array(mean_eff_list)
    mean_h_prime_arr = np.array(mean_h_prime_list)
    mean_dmu_arr = np.array(mean_dmu_list)
    mean_E_readout_arr = np.array(mean_E_readout_list)
    mean_sigma_readout_arr = np.array(mean_sigma_readout_list)
    mean_abs_grad_dend_arr = np.array(mean_abs_grad_dend_list)
    mean_abs_grad_soma_arr = np.array(mean_abs_grad_soma_list)
    mean_abs_grad_readout_arr = np.array(mean_abs_grad_readout_list)
    elig_soma_one_arr = np.array(elig_soma_one_list)
    elig_dend_one_arr = np.array(elig_dend_one_list)
    elig_readout_one_arr = np.array(elig_readout_one_list)
    # (n_samples, n_outputs)
    grad_readout_per_neuron = np.array(grad_readout_per_neuron_norms)

    accuracy = 100.0 * correct / n_samples
    print()
    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy: {correct}/{n_samples} = {accuracy:.2f}%")
    print()
    print("Gradient norms (over full test set):")
    print(f"  Hidden dendrite (full):  mean = {np.mean(grad_dend_norms):.6f},  std = {np.std(grad_dend_norms):.6f},  min = {np.min(grad_dend_norms):.6f},  max = {np.max(grad_dend_norms):.6f}")
    print(f"  Hidden soma (full):      mean = {np.mean(grad_soma_norms):.6f},  std = {np.std(grad_soma_norms):.6f},  min = {np.min(grad_soma_norms):.6f},  max = {np.max(grad_soma_norms):.6f}")
    print()
    print("Average value (signed) per gradient element (mean over weights; shows typical update direction/magnitude):")
    print(f"  grad_dend (per weight):  mean over samples = {np.mean(mean_grad_dend_arr):.6e},  std = {np.std(mean_grad_dend_arr):.6e}")
    print(f"  grad_soma (per weight):  mean over samples = {np.mean(mean_grad_soma_arr):.6e},  std = {np.std(mean_grad_soma_arr):.6e}")
    print(f"  grad_readout (per weight): mean over samples = {np.mean(mean_grad_readout_arr):.6e},  std = {np.std(mean_grad_readout_arr):.6e}")
    print()
    print("--- Per-synapse (one eligibility trace per weight) ---")
    print("Typical |gradient| for one weight (mean over synapses of |grad[i,j]|) → how much that weight changes after one sample, before lr:")
    print(f"  soma:     mean over samples = {np.mean(mean_abs_grad_soma_arr):.6e},  std = {np.std(mean_abs_grad_soma_arr):.6e}")
    print(f"  dendrite: mean over samples = {np.mean(mean_abs_grad_dend_arr):.6e},  std = {np.std(mean_abs_grad_dend_arr):.6e}")
    print(f"  readout:  mean over samples = {np.mean(mean_abs_grad_readout_arr):.6e},  std = {np.std(mean_abs_grad_readout_arr):.6e}")
    print("Typical eligibility trace magnitude for one synapse (mean over synapses of ||trace|| over time):")
    print(f"  soma (one trace per input j, shared across neurons): mean = {np.mean(elig_soma_one_arr):.6e},  std = {np.std(elig_soma_one_arr):.6e}")
    print(f"  dendrite (one trace per (i,j)):                     mean = {np.mean(elig_dend_one_arr):.6e},  std = {np.std(elig_dend_one_arr):.6e}")
    print(f"  readout (one trace per hidden j):                   mean = {np.mean(elig_readout_one_arr):.6e},  std = {np.std(elig_readout_one_arr):.6e}")
    print()
    print("  Somatic gradient = (1/T) sum_t sigma'_hidden * E_soma * effective_error  (component norms, which drive grad_soma size):")
    print(f"    sigma'_hidden (Frob):   mean = {np.mean(sigma_prime_hidden_norms):.6f},  std = {np.std(sigma_prime_hidden_norms):.6f},  max = {np.max(sigma_prime_hidden_norms):.6f}")
    print(f"    E_soma (Frob):          mean = {np.mean(E_soma_norms):.6f},  std = {np.std(E_soma_norms):.6f},  max = {np.max(E_soma_norms):.6f}")
    print(f"    effective_error (Frob): mean = {np.mean(effective_error_norms):.6f},  std = {np.std(effective_error_norms):.6f},  max = {np.max(effective_error_norms):.6f}")
    print("  Dendritic gradient = (1/T) sum_t gamma * sigma'_hidden * h' * dmu_tprime_dw * effective_error:")
    print(f"    h' hidden (Frob):       mean = {np.mean(h_prime_norms):.6f},  std = {np.std(h_prime_norms):.6f},  max = {np.max(h_prime_norms):.6f}")
    print(f"    E_dendrite / dmu_tprime_dw (Frob): mean = {np.mean(dmu_tprime_dw_norms):.6f},  std = {np.std(dmu_tprime_dw_norms):.6f},  max = {np.max(dmu_tprime_dw_norms):.6f}")
    print("  Readout gradient = (1/T) sum_t sigma'_readout * E_readout * global_error:")
    print(f"    E_readout (Frob):       mean = {np.mean(E_readout_norms):.6f},  std = {np.std(E_readout_norms):.6f},  max = {np.max(E_readout_norms):.6f}")
    print()
    print("Average value (signed) per element for each gradient component (typical value per (t,i) or (t,i,j); drives product magnitude):")
    print(f"    sigma'_hidden (per elem):  mean over samples = {np.mean(mean_sigma_arr):.6e},  std = {np.std(mean_sigma_arr):.6e}")
    print(f"    E_soma (per weight, per t): mean over samples = {np.mean(mean_E_soma_arr):.6e},  std = {np.std(mean_E_soma_arr):.6e}")
    print(f"    effective_error (per elem): mean over samples = {np.mean(mean_eff_arr):.6e},  std = {np.std(mean_eff_arr):.6e}")
    print(f"    h' hidden (per elem):       mean over samples = {np.mean(mean_h_prime_arr):.6e},  std = {np.std(mean_h_prime_arr):.6e}")
    print(f"    dmu_tprime_dw / E_dendrite (per weight): mean over samples = {np.mean(mean_dmu_arr):.6e},  std = {np.std(mean_dmu_arr):.6e}")
    print(f"    E_readout (per weight, per t): mean over samples = {np.mean(mean_E_readout_arr):.6e},  std = {np.std(mean_E_readout_arr):.6e}")
    print(f"    sigma'_readout (per elem):   mean over samples = {np.mean(mean_sigma_readout_arr):.6e},  std = {np.std(mean_sigma_readout_arr):.6e}")
    print()
    print(f"  Readout (full):          mean = {np.mean(grad_readout_norms):.6f},  std = {np.std(grad_readout_norms):.6f},  min = {np.min(grad_readout_norms):.6f},  max = {np.max(grad_readout_norms):.6f}")
    print(f"  sigma' readout (Frob):  mean = {np.mean(sigma_prime_readout_norms):.6f},  std = {np.std(sigma_prime_readout_norms):.6f},  min = {np.min(sigma_prime_readout_norms):.6f},  max = {np.max(sigma_prime_readout_norms):.6f}")
    print()
    print("Per-readout-neuron gradient norm (mean over samples):")
    mean_per_readout = np.mean(grad_readout_per_neuron, axis=0)
    std_per_readout = np.std(grad_readout_per_neuron, axis=0)
    for j in range(n_outputs):
        print(f"  Readout neuron {j:2d}:  mean = {mean_per_readout[j]:.6f},  std = {std_per_readout[j]:.6f}")
    print()
    print("(Full gradient = hidden dendrite + hidden soma + readout; norms above are for each part separately.)")
    print("=" * 60)


if __name__ == "__main__":
    main()
