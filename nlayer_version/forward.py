o[t] = jnp.where(v[t] >= config.v_th - config.gamma * h[t], 1, 0)
h[t] = jnp.where(mu[t_prime[t]] >= config.mu_th, 1, 0)
v[t] = alpha_s * v[t-1] + soma_in[t] * (1 - o[t-1])
mu[t] = alpha * mu[t-1] + (1 - h[t]) * dend_in[t]
t_prime[t] = jnp.where(t[t] == 0, 0, jnp.where(h[t-1] == 1, t_prime[t-1], t[t]))

if h[t] == 0:
    t_prime[t] = t
elif t - t_prime[t-1] < T_plateau:
    t_prime[t] = t_prime[t-1]
else:
    t_prime[t] = t


# Version 2 (no mu_history; carry mu_at_tprime_prev)
t_prime[t] = jnp.where(t == 0, 0, jnp.where(h_prev == 1, t_prime_prev, t))
mu[t] = jnp.where(t > 0, alpha * mu_prev + (1 - h_prev) * dend_in[t], dend_in[t])
mu_at_tprime[t] = jnp.where((t == 0) | (h_prev == 0), mu[t], mu_at_tprime_prev)

mu_at_tprime = mu_at_tprime[t]
plateau_duration = t - t_prime[t]
h[t] = jnp.where(
    (mu_at_tprime >= config.mu_th) &
    (plateau_duration <= T_p) &
    (plateau_duration >= 0),
    1,
    0,
)

v_pre_reset[t] = jnp.where(t > 0, alpha_s * v_prev + soma_in[t], soma_in[t])
o[t] = jnp.where(v_pre_reset[t] >= config.v_th - config.gamma * h[t], 1, 0)
v[t] = v_pre_reset[t] * (1 - o[t])
t_prime_next = t_prime[t]
mu_at_tprime_next = mu_at_tprime[t]


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

v_pre_reset = jnp.where(t > 0, alpha_s * v_prev + soma_in[t], soma_in[t])
o = jnp.where(v_pre_reset >= config.v_th - config.gamma * h, 1, 0)
v = v_pre_reset * (1 - o)

# Next carry state
mu_next = mu
v_next = v
h_next = h
t_prime_next = t_prime
mu_at_tprime_next = mu_at_tprime