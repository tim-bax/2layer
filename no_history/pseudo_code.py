# Version 3 (carry-only forward pass; no full histories)
# Carry at step t: mu_prev, v_prev, h_prev, t_prime_prev, mu_at_tprime_prev
t_prime = jnp.where(t == 0, 0, jnp.where(h_prev == 1, t_prime_prev, t))
mu = jnp.where(t > 0, alpha * mu_prev + (1 - h_prev) * dend_in[t], dend_in[t])
mu_at_tprime = jnp.where((t == 0) | (h_prev == 0), mu, mu_at_tprime_prev)

plateau_duration = t - t_prime
h = jnp.where(
    (mu_at_tprime >= config.mu_th) &
    (plateau_duration <= T_p) &
    (plateau_duration >= 0),
    1,
    0,
)

v = jnp.where(t > 0, alpha_s * v_prev + soma_in[t], soma_in[t])
o = jnp.where(v >= config.v_th - config.gamma * h, 1, 0)
v = v * (1 - o)

# Next carry state
mu_next = mu
v_next = v
h_next = h
t_prime_next = t_prime
mu_at_tprime_next = mu_at_tprime






# Readout LIF neuron forward pass (from 2comp_uniform.py)
# Inputs: spike_inputs[t] from hidden layer, readout weights w_readout
# Carry at step t: v_readout_prev, readout_counts_prev
readout_in[t] = spike_inputs[t] @ w_readout.T
v_readout = alpha_m * v_readout_prev + readout_in[t]
o_readout = jnp.where(v_readout >= config.v_th, 1, 0)
v_readout = v_readout * (1 - o_readout) 
readout_counts = readout_counts_prev + o_readout

# Next carry state
v_readout_next = v_readout
readout_counts_next = readout_counts


# Backward pass (start): readout prediction, loss, global error
# Inputs: final readout_counts from forward carry, target class index
scaled_logits = readout_counts / loss_temperature + loss_count_bias
probs = jnp.exp(scaled_logits - jnp.max(scaled_logits))
probs = probs / jnp.sum(probs)

target_one_hot = jnp.zeros(n_outputs).at[target].set(1.0)
target_smoothed = (
    target_one_hot * (1 - loss_label_smoothing)
    + loss_label_smoothing / n_outputs
)

# Prediction and cross-entropy loss
prediction = jnp.argmax(readout_counts)
loss = -jnp.sum(target_smoothed * jnp.log(probs + 1e-8))

# Global output error used for readout/hidden updates
global_error = target_smoothed - probs


# Surrogate gradient function (SuperSpike)
# beta controls sharpness around threshold
def surrogate_sigma(x, beta):
    return 1.0 / (1.0 + beta * jnp.abs(x)) ** 2


# Somatic eligibility trace (online, carry-only)
# Generic recurrence: E_soma[t] = alpha_s * E_soma[t-1] + pre_spike[t]
# Example pre_spike:
#   - hidden soma synapses: pre_spike[t] = extra_o[t]
#   - extra soma synapses:  pre_spike[t] = x_input[t]
def update_somatic_eligibility(E_soma_prev, pre_spike_t, alpha_s):
    E_soma = alpha_s * E_soma_prev + pre_spike_t
    return E_soma


# Dendritic eligibility trace at plateau initiation (online, carry-only)
# Returns:
#   dmu_dw_t:          running dendritic eligibility at current time
#   dmu_dw_at_tprime_t:eligibility latched at plateau initiation t'
def update_dendritic_eligibility(
    dmu_dw_prev,
    dmu_dw_at_tprime_prev,
    pre_spike_t,
    h_hidden_prev,
    alpha_d_hidden,
):
    dmu_dw_t = alpha_d_hidden * dmu_dw_prev + (1 - h_hidden_prev[:, None]) * pre_spike_t[None, :]
    dmu_dw_at_tprime_t = jnp.where(
        (h_hidden_prev == 0)[:, None],
        dmu_dw_t,
        dmu_dw_at_tprime_prev,
    )
    return dmu_dw_t, dmu_dw_at_tprime_t


# Readout gradient (online accumulation, then apply global error)
# Full-history form:
#   grad_readout[i, j] = (1/T) * sum_t global_error[i] * sigma_prime_readout[t, i] * E_readout[t, j]
# Since global_error is time-independent, accumulate basis over time first.

# Per-step (inside sequence loop):
# Inputs at step t:
#   v_readout_t (pre-reset membrane), hidden_o_t, E_readout_prev
E_readout_t = update_somatic_eligibility(E_readout_prev, hidden_o_t, alpha_readout)
sigma_prime_readout_t = surrogate_sigma(v_readout_t - config.v_th, beta_readout)  # shape: (n_outputs,)
A_readout += jnp.einsum("i,j->ij", sigma_prime_readout_t, E_readout_t)            # shape: (n_outputs, n_hidden)
E_readout_next = E_readout_t

# End of sequence:
grad_readout = (global_error[:, None] * A_readout) / T


# Hidden 2-comp soma gradient (online accumulation, then apply global error)

# Per-step (inside sequence loop):
# Inputs at step t:
#   sigma_prime_readout_t (already computed), v_hidden_pre_reset_t, h_hidden_t, extra_o_t, E_soma_hidden_prev, w_readout
E_soma_hidden_t = update_somatic_eligibility(E_soma_hidden_prev, extra_o_t, alpha_s_hidden)
sigma_prime_hidden_t = surrogate_sigma(
    v_hidden_pre_reset_t + config.gamma * h_hidden_t - config.v_th,
    beta_hidden_soma,
)
A_soma_hidden += jnp.einsum(
    "j,ji,i,k->jik",
    sigma_prime_readout_t,
    w_readout,
    sigma_prime_hidden_t,
    E_soma_hidden_t,
)  # shape: (n_outputs, n_hidden, n_extra)
E_soma_hidden_next = E_soma_hidden_t

# End of sequence:
grad_soma_hidden = jnp.einsum("j,jik->ik", global_error, A_soma_hidden) / T


# Hidden 2-comp dendritic gradient (online accumulation, then apply global error)
# Reuses:
#   sigma_prime_readout_t  (from readout block)
#   sigma_prime_hidden_t   (from hidden soma block)
#
# Per-step (inside sequence loop):
# Inputs at step t:
#   mu_at_tprime_hidden_t, h_hidden_prev, extra_o_t, dmu_dw_hidden_prev, dmu_dw_at_tprime_hidden_prev
h_prime_hidden_t = surrogate_sigma(mu_at_tprime_hidden_t - config.mu_th, beta_hidden_dend)
dmu_dw_hidden_t, dmu_dw_at_tprime_hidden_t = update_dendritic_eligibility(
    dmu_dw_hidden_prev,
    dmu_dw_at_tprime_hidden_prev,
    extra_o_t,
    h_hidden_prev,
    alpha_d_hidden,
)
A_dend_hidden += jnp.einsum(
    "j,ji,i,i,ik->jik",
    sigma_prime_readout_t,
    w_readout,
    sigma_prime_hidden_t,
    h_prime_hidden_t * config.gamma,
    dmu_dw_at_tprime_hidden_t,
)  # shape: (n_outputs, n_hidden, n_extra)
dmu_dw_hidden_next = dmu_dw_hidden_t
dmu_dw_at_tprime_hidden_next = dmu_dw_at_tprime_hidden_t

# End of sequence:
grad_dend_hidden = jnp.einsum("j,jik->ik", global_error, A_dend_hidden) / T
