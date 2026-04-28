import jax.numpy as jnp
from jax import random, jit, lax, vmap

from config import NeuronConfig, surrogate_sigma
from two_comp_neuron import TwoCompNeuron
from lif_neuron import LIFNeuron


# ══════════════════════════════════════════════════════════════════════
#  Two-layer e-prop, no_history-style.
#
#  Input x(t) ──► extra 2-comp layer ──► extra_o(t)
#                  (n_extra)                 │
#                                            ▼
#                                  hidden 2-comp layer ──► hidden_o(t)
#                                       (n_hidden)             │
#                                                              ▼
#                                                       LIF readout (n_outputs)
#
#  No T-axis tensors are ever materialised — gradients accumulate online
#  inside one lax.scan (same idea as no_history/network.py). The one
#  cross-layer wrinkle: all "extra-layer quantity at hidden's plateau-init
#  time t'_h" tensors are replaced by mirrors in the carry that get
#  refreshed exactly when h_h_prev[i] == 0 (i.e. when hidden i is not in
#  plateau, so t'_h(t,i) = t).
#
#  Memory budget per sample (SHD-sized: T=700, N=64, n_extra=10, K=700):
#    - mirrors:                                     ~4 MB
#    - online accumulators (J-axis kept):           ~3.3 MB
#    - per-step scan outputs (forwarded post-scan): ~5 MB
#    - everything else (neuron state, eligibilities): <1 MB
#  Independent of T → no OOM, vmap-friendly.
# ══════════════════════════════════════════════════════════════════════


def _forward_and_accum(
    x_input,
    w_dend_e, w_soma_e, w_dend_h, w_soma_h, w_readout,
    alpha_s, alpha_d, alpha_m, T_p_e, T_p_h, config,
    e_carry_init, h_carry_init, r_carry_init,
    mirrors_init, accums_init,
    rng_key, dropout_extra, dropout_hidden,
):
    """Forward pass + gradient accumulator bookkeeping for one sample.

    Returns:
        readout_counts (J,)
        per_step tensors used for post-scan reductions:
            sp_r (T,J), sp_h (T,N), sp_e (T,n_extra),
            r_E (T,N), E_soma_h (T,n_extra), E_soma_e (T,K_in)
        accums:
            A_dend_h     (J, N, n_extra)        — for w_dend_hidden
            A_soma_e_p2  (J, n_extra, K_in)     — for w_soma_extra (path 2)
            A_dend_e_p1  (J, n_extra, K_in)     — for w_dend_extra (path 1)
            A_dend_e_p2  (J, n_extra, K_in)     — for w_dend_extra (path 2)
    """
    n_extra = w_dend_e.shape[0]
    n_hidden = w_dend_h.shape[0]
    T = x_input.shape[0]

    time_indices = jnp.arange(T, dtype=jnp.int32)
    drop_keys = random.split(rng_key, 2 * T)
    drop_keys_e = drop_keys[:T]
    drop_keys_h = drop_keys[T:]
    scale_e = 1.0 / (1.0 - dropout_extra)
    scale_h = 1.0 / (1.0 - dropout_hidden)

    def step(carry, inputs):
        e_carry, h_carry, r_carry, mirrors, accums = carry
        x_t, t, dk_e, dk_h = inputs
        x_t_f = x_t.astype(jnp.float64)

        # ─────────────────────────────  EXTRA LAYER  ─────────────────────────────
        dend_in_e = x_t_f @ w_dend_e.T
        soma_in_e = x_t_f @ w_soma_e.T
        e_carry_new, e_o, e_v_pre, e_h, e_h_prev, e_mu_at_tp = TwoCompNeuron.forward_step(
            e_carry, dend_in_e, soma_in_e, t, alpha_s, alpha_d, T_p_e, config,
        )
        mu_e_, v_e_, h_e_, tp_e_, matp_e_, E_soma_e_old, dmu_dw_e_old, dmu_dw_e_atp_old = e_carry_new
        E_soma_e_new = TwoCompNeuron.update_somatic_eligibility(
            E_soma_e_old, x_t_f, alpha_s,
        )
        dmu_dw_e_new, dmu_dw_e_atp_new = TwoCompNeuron.update_dendritic_eligibility(
            dmu_dw_e_old, dmu_dw_e_atp_old, x_t_f, e_h_prev, alpha_d,
        )
        e_carry_new = (mu_e_, v_e_, h_e_, tp_e_, matp_e_,
                       E_soma_e_new, dmu_dw_e_new, dmu_dw_e_atp_new)

        # Dropout on extra spikes BEFORE they reach the hidden layer.
        # This means the hidden layer sees a noisy version of extra_o, and
        # its eligibility traces / dmu reflect that. (Exactly analogous to
        # no_history's dropout on hidden_o on its way to the readout.)
        e_o_float = e_o.astype(jnp.float64)
        mask_e = random.bernoulli(dk_e, 1.0 - dropout_extra, (n_extra,)).astype(jnp.float64)
        e_o_float = e_o_float * mask_e * scale_e

        # ─────────────────────────────  HIDDEN LAYER  ────────────────────────────
        dend_in_h = e_o_float @ w_dend_h.T
        soma_in_h = e_o_float @ w_soma_h.T
        h_carry_new, h_o, h_v_pre, h_h, h_h_prev, h_mu_at_tp = TwoCompNeuron.forward_step(
            h_carry, dend_in_h, soma_in_h, t, alpha_s, alpha_d, T_p_h, config,
        )
        mu_h_, v_h_, hh_, tp_h_, matp_h_, E_soma_h_old, dmu_dw_h_old, dmu_dw_h_atp_old = h_carry_new
        E_soma_h_new = TwoCompNeuron.update_somatic_eligibility(
            E_soma_h_old, e_o_float, alpha_s,
        )
        dmu_dw_h_new, dmu_dw_h_atp_new = TwoCompNeuron.update_dendritic_eligibility(
            dmu_dw_h_old, dmu_dw_h_atp_old, e_o_float, h_h_prev, alpha_d,
        )
        h_carry_new = (mu_h_, v_h_, hh_, tp_h_, matp_h_,
                       E_soma_h_new, dmu_dw_h_new, dmu_dw_h_atp_new)

        # Dropout on hidden spikes BEFORE they reach the readout
        # (this is the exact mechanism from no_history).
        h_o_float = h_o.astype(jnp.float64)
        mask_h = random.bernoulli(dk_h, 1.0 - dropout_hidden, (n_hidden,)).astype(jnp.float64)
        h_o_float = h_o_float * mask_h * scale_h

        # ─────────────────────────────  READOUT  ─────────────────────────────────
        r_carry_new, r_o, r_v_pre, r_E = LIFNeuron.forward_step(
            r_carry, h_o_float, w_readout, alpha_m, config.v_th,
        )

        # ─────────────────────────────  SURROGATES  ──────────────────────────────
        sp_r = surrogate_sigma(r_v_pre - config.v_th, config.beta_s)                          # (J,)
        sp_h = surrogate_sigma(h_v_pre + config.gamma * h_h - config.v_th, config.beta_s)     # (N,)
        hp_h = surrogate_sigma(h_mu_at_tp - config.mu_th, config.beta_d)                      # (N,)
        sp_e = surrogate_sigma(e_v_pre + config.gamma * e_h - config.v_th, config.beta_s)    # (n_extra,)
        hp_e = surrogate_sigma(e_mu_at_tp - config.mu_th, config.beta_d)                      # (n_extra,)

        # ─────────────────────────  REFRESH CROSS-LAYER MIRRORS  ─────────────────
        # Mirror update rule: a hidden neuron i has t'_h(t,i) = t exactly when
        # h_h_prev[i] == 0 (no plateau ongoing into this step). When that
        # holds, snapshot the current values of the extra-layer quantities
        # for this hidden neuron. Otherwise, hold the previous snapshot.
        # This replaces the (T, n_hidden, …) "lookup at t'_h" tensors of the
        # original 2layer.py with O(1) carries.
        sp_extra_at_tph_old, hp_extra_at_tph_old, E_extra_at_tph_old, dmu_extra_at_tph_old = mirrors
        refresh = (h_h_prev == 0)                                # (N,) bool
        sp_extra_at_tph = jnp.where(
            refresh[:, None], sp_e[None, :], sp_extra_at_tph_old,
        )                                                        # (N, n_extra)
        hp_extra_at_tph = jnp.where(
            refresh[:, None], hp_e[None, :], hp_extra_at_tph_old,
        )                                                        # (N, n_extra)
        E_extra_at_tph = jnp.where(
            refresh[:, None], E_soma_e_new[None, :], E_extra_at_tph_old,
        )                                                        # (N, K_in)
        # NOTE on dmu_dw_e_atp_new vs dmu_dw_e_new: dmu_dw_e_atp_new is the
        # frozen-at-extra's-t'_e value, which is exactly what the original
        # `dmu_dw_extra` (post `compute_dmu_tprime_dw` plateau fill) holds.
        # This is the right thing to mirror across to t'_h.
        dmu_extra_at_tph = jnp.where(
            refresh[:, None, None], dmu_dw_e_atp_new[None, :, :], dmu_extra_at_tph_old,
        )                                                        # (N, n_extra, K_in)
        mirrors_new = (sp_extra_at_tph, hp_extra_at_tph, E_extra_at_tph, dmu_extra_at_tph)

        # ───────────────────────  ONLINE GRADIENT ACCUMULATION  ──────────────────
        # All accumulators keep a J=n_outputs axis; global_error is folded in
        # post-scan. This mirrors no_history's A_d (J,N,K) construction.
        A_dend_h, A_soma_e_p2, A_dend_e_p1, A_dend_e_p2 = accums

        # A_dend_h[j, i, k] += sp_r[j]·w_readout[j,i]·sp_h[i]·hp_h[i]·γ · dmu_dw_h_atp[i,k]
        eta_d_h = (sp_r[:, None] * w_readout                       # (J, N)
                   * sp_h[None, :] * (hp_h * config.gamma)[None, :])
        A_dend_h = A_dend_h + jnp.einsum("ji,ik->jik", eta_d_h, dmu_dw_h_atp_new)

        # A_dend_e_p1[j, k, m] += coeff_p1[j, k] · dmu_dw_e_atp[k, m]
        # coeff_p1[j, k] = sp_r[j] · γ · sp_e[k] · hp_e[k] · Σ_i w_readout[j,i]·sp_h[i]·w_soma_h[i,k]
        inner_p1 = jnp.einsum("ji,i,ik->jk", w_readout, sp_h, w_soma_h)        # (J, n_extra)
        coeff_p1 = (sp_r[:, None] * inner_p1
                    * (sp_e * hp_e * config.gamma)[None, :])                    # (J, n_extra)
        A_dend_e_p1 = A_dend_e_p1 + jnp.einsum("jk,km->jkm", coeff_p1, dmu_dw_e_atp_new)

        # per_t_p2_soma[j, i, k] = sp_r[j]·w_readout[j,i]·sp_h[i]·hp_h[i]·w_dend_h[i,k]
        #                          ·sp_extra_at_tph[i,k] · γ
        # used for both A_soma_e_p2 and A_dend_e_p2.
        per_t_p2_soma = (sp_r[:, None, None] * w_readout[:, :, None]            # (J, N, 1)
                         * sp_h[None, :, None] * hp_h[None, :, None]            # × (1, N, 1)
                         * w_dend_h[None, :, :]                                  # × (1, N, n_extra)
                         * sp_extra_at_tph[None, :, :]                           # × (1, N, n_extra)
                         * config.gamma)                                         # → (J, N, n_extra)

        # A_soma_e_p2[j, k, m] += Σ_i per_t_p2_soma[j, i, k] · E_extra_at_tph[i, m]
        A_soma_e_p2 = A_soma_e_p2 + jnp.einsum("jik,im->jkm", per_t_p2_soma, E_extra_at_tph)

        # per_t_p2_dend[j, i, k] = per_t_p2_soma[j, i, k] · hp_extra_at_tph[i, k] · γ
        per_t_p2_dend = per_t_p2_soma * hp_extra_at_tph[None, :, :] * config.gamma  # (J, N, n_extra)

        # A_dend_e_p2[j, k, m] += Σ_i per_t_p2_dend[j, i, k] · dmu_extra_at_tph[i, k, m]
        A_dend_e_p2 = A_dend_e_p2 + jnp.einsum("jik,ikm->jkm", per_t_p2_dend, dmu_extra_at_tph)

        accums_new = (A_dend_h, A_soma_e_p2, A_dend_e_p1, A_dend_e_p2)

        new_carry = (e_carry_new, h_carry_new, r_carry_new, mirrors_new, accums_new)
        per_step = (sp_r, sp_h, sp_e, r_E, E_soma_h_new, E_soma_e_new)
        return new_carry, per_step

    init_carry = (e_carry_init, h_carry_init, r_carry_init, mirrors_init, accums_init)
    scan_inputs = (x_input, time_indices, drop_keys_e, drop_keys_h)
    final_carry, per_step_all = lax.scan(step, init_carry, scan_inputs)

    sp_r_T, sp_h_T, sp_e_T, r_E_T, E_soma_h_T, E_soma_e_T = per_step_all
    _, _, r_carry_f, _, accums_f = final_carry
    readout_counts = r_carry_f[1]

    return readout_counts, (sp_r_T, sp_h_T, sp_e_T, r_E_T, E_soma_h_T, E_soma_e_T), accums_f


def _predict_only(
    x_input,
    w_dend_e, w_soma_e, w_dend_h, w_soma_h, w_readout,
    alpha_s, alpha_d, alpha_m, T_p_e, T_p_h, config,
):
    """Forward pass only — no gradient bookkeeping, no dropout. Returns counts (J,)."""
    n_extra = w_dend_e.shape[0]
    n_hidden = w_dend_h.shape[0]
    n_outputs = w_readout.shape[0]
    T = x_input.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)

    def two_comp_step(state, dend_in, soma_in, t, T_p):
        mu, v, h, t_prime, mu_at_tp = state
        t_prime_new = jnp.where(t == 0, 0, jnp.where(h == 1, t_prime, t))
        mu_new = jnp.where(t > 0, alpha_d * mu + (1 - h) * dend_in, dend_in)
        mu_at_tp_new = jnp.where(h == 0, mu_new, mu_at_tp)
        plat_dur = t - t_prime_new
        h_new = jnp.where(
            (mu_at_tp_new >= config.mu_th) & (plat_dur <= T_p) & (plat_dur >= 0),
            1, 0,
        ).astype(jnp.int32)
        v_pre = jnp.where(t > 0, alpha_s * v + soma_in, soma_in)
        o = jnp.where(v_pre >= config.v_th - config.gamma * h_new, 1, 0).astype(jnp.int32)
        v_new = v_pre * (1 - o)
        return (mu_new, v_new, h_new, t_prime_new, mu_at_tp_new), o

    def step(carry, inputs):
        e_state, h_state, r_v, r_counts = carry
        x_t, t = inputs
        x_t_f = x_t.astype(jnp.float64)

        dend_in_e = x_t_f @ w_dend_e.T
        soma_in_e = x_t_f @ w_soma_e.T
        e_state, e_o = two_comp_step(e_state, dend_in_e, soma_in_e, t, T_p_e)
        e_o_f = e_o.astype(jnp.float64)

        dend_in_h = e_o_f @ w_dend_h.T
        soma_in_h = e_o_f @ w_soma_h.T
        h_state, h_o = two_comp_step(h_state, dend_in_h, soma_in_h, t, T_p_h)

        r_in = h_o.astype(jnp.float64) @ w_readout.T
        r_v_new = alpha_m * r_v + r_in
        r_o = jnp.where(r_v_new >= config.v_th, 1, 0).astype(jnp.int32)
        r_v_new = r_v_new * (1 - r_o)
        r_counts_new = r_counts + r_o

        return (e_state, h_state, r_v_new, r_counts_new), None

    e_init = (jnp.zeros(n_extra), jnp.zeros(n_extra),
              jnp.zeros(n_extra, dtype=jnp.int32), jnp.zeros(n_extra, dtype=jnp.int32),
              jnp.zeros(n_extra))
    h_init = (jnp.zeros(n_hidden), jnp.zeros(n_hidden),
              jnp.zeros(n_hidden, dtype=jnp.int32), jnp.zeros(n_hidden, dtype=jnp.int32),
              jnp.zeros(n_hidden))
    init = (e_init, h_init, jnp.zeros(n_outputs), jnp.zeros(n_outputs))
    final, _ = lax.scan(step, init, (x_input, time_indices))
    return final[3]


def _loss_and_grads(
    readout_counts, per_step_tensors, accums,
    w_readout, w_soma_h,
    target_smoothed, T, loss_temperature, loss_count_bias,
):
    """Compute loss and weight gradients for one sample (post-scan reductions)."""
    sp_r_T, sp_h_T, sp_e_T, r_E_T, E_soma_h_T, E_soma_e_T = per_step_tensors
    A_dend_h, A_soma_e_p2, A_dend_e_p1, A_dend_e_p2 = accums

    scaled_logits = readout_counts / loss_temperature + loss_count_bias
    probs = jnp.exp(scaled_logits - jnp.max(scaled_logits))
    probs = probs / jnp.sum(probs)
    prediction = jnp.argmax(readout_counts)
    loss = -jnp.sum(target_smoothed * jnp.log(probs + 1e-8))
    global_error = target_smoothed - probs                                       # (J,)

    # eps_h[t, i] = Σ_j sp_r[t,j] · global_error[j] · w_readout[j, i]
    ge_w = global_error[:, None] * w_readout                                      # (J, N)
    eps_h = sp_r_T @ ge_w                                                         # (T, N)

    # ── Readout: post-scan ──
    A_readout = jnp.einsum("tj,ti->ji", sp_r_T, r_E_T)                            # (J, N)
    grad_readout = global_error[:, None] * A_readout / T

    # ── Hidden soma: post-scan ──
    grad_soma_h = jnp.einsum("ti,ti,tk->ik", sp_h_T, eps_h, E_soma_h_T) / T       # (N, n_extra)

    # ── Hidden dend: online (uses dmu_dw_h_atp from carry) ──
    grad_dend_h = jnp.einsum("j,jik->ik", global_error, A_dend_h) / T             # (N, n_extra)

    # ── Extra soma path 1: post-scan (no carry-only term) ──
    # coeff_soma_p1[t, k] = sp_e[t,k] · Σ_i w_soma_h[i,k] · sp_h[t,i] · eps_h[t,i]
    inner_soma_p1 = jnp.einsum("ik,ti,ti->tk", w_soma_h, sp_h_T, eps_h)           # (T, n_extra)
    coeff_soma_p1 = sp_e_T * inner_soma_p1                                        # (T, n_extra)
    grad_soma_e_p1 = jnp.einsum("tk,tm->km", coeff_soma_p1, E_soma_e_T) / T       # (n_extra, K_in)

    # ── Extra soma path 2: online (uses E_extra_at_tph from carry) ──
    grad_soma_e_p2 = jnp.einsum("j,jkm->km", global_error, A_soma_e_p2) / T       # (n_extra, K_in)

    grad_soma_e = grad_soma_e_p1 + grad_soma_e_p2

    # ── Extra dend path 1: online (uses dmu_dw_e_atp from carry) ──
    grad_dend_e_p1 = jnp.einsum("j,jkm->km", global_error, A_dend_e_p1) / T       # (n_extra, K_in)

    # ── Extra dend path 2: online (uses dmu_extra_at_tph mirror from carry) ──
    grad_dend_e_p2 = jnp.einsum("j,jkm->km", global_error, A_dend_e_p2) / T       # (n_extra, K_in)

    grad_dend_e = grad_dend_e_p1 + grad_dend_e_p2

    return (loss, prediction,
            grad_readout, grad_soma_h, grad_dend_h, grad_soma_e, grad_dend_e)


def _apply_grads_sgd(
    w_dend_e, w_soma_e, w_dend_h, w_soma_h, w_readout,
    g_dend_e, g_soma_e, g_dend_h, g_soma_h, g_readout,
    lr, clip_value, weight_decay,
):
    """SGD with decoupled weight decay (same as no_history, 5 weight groups)."""
    g_readout = jnp.clip(g_readout, -clip_value, clip_value)
    g_soma_h = jnp.clip(g_soma_h, -clip_value, clip_value)
    g_dend_h = jnp.clip(g_dend_h, -clip_value, clip_value)
    g_soma_e = jnp.clip(g_soma_e, -clip_value, clip_value)
    g_dend_e = jnp.clip(g_dend_e, -clip_value, clip_value)
    return (
        w_dend_e + lr * g_dend_e - lr * weight_decay * w_dend_e,
        w_soma_e + lr * g_soma_e - lr * weight_decay * w_soma_e,
        w_dend_h + lr * g_dend_h - lr * weight_decay * w_dend_h,
        w_soma_h + lr * g_soma_h - lr * weight_decay * w_soma_h,
        w_readout + lr * g_readout - lr * weight_decay * w_readout,
    )


def _apply_grads_adam(
    w_de, w_se, w_dh, w_sh, w_r,
    g_de, g_se, g_dh, g_sh, g_r,
    m_de, m_se, m_dh, m_sh, m_r,
    v_de, v_se, v_dh, v_sh, v_r,
    step, lr, beta1, beta2, eps, clip_value, weight_decay,
):
    """AdamW-style decoupled weight decay (same as no_history, 5 weight groups)."""
    def update_one(w, g, m, v):
        g = jnp.clip(g, -clip_value, clip_value)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        w = w + lr * m_hat / (jnp.sqrt(v_hat) + eps) - lr * weight_decay * w
        return w, m, v

    w_de, m_de, v_de = update_one(w_de, g_de, m_de, v_de)
    w_se, m_se, v_se = update_one(w_se, g_se, m_se, v_se)
    w_dh, m_dh, v_dh = update_one(w_dh, g_dh, m_dh, v_dh)
    w_sh, m_sh, v_sh = update_one(w_sh, g_sh, m_sh, v_sh)
    w_r, m_r, v_r = update_one(w_r, g_r, m_r, v_r)
    return (w_de, w_se, w_dh, w_sh, w_r,
            m_de, m_se, m_dh, m_sh, m_r,
            v_de, v_se, v_dh, v_sh, v_r)


# ══════════════════════════════════════════════════════════════════════
#  vmap in_axes: which args are per-sample (0) vs shared (None).
# ══════════════════════════════════════════════════════════════════════

_FWD_AXES = (
    0,                                       # x_input
    None, None, None, None, None,            # 5 weight matrices (shared)
    None, None, None, None, None, None,      # alpha_s, alpha_d, alpha_m, T_p_e, T_p_h, config
    (0, 0, 0, 0, 0, 0, 0, 0),                # e_carry_init (8-tuple, batched)
    (0, 0, 0, 0, 0, 0, 0, 0),                # h_carry_init
    (0, 0, 0),                               # r_carry_init
    (0, 0, 0, 0),                            # mirrors_init (4-tuple)
    (0, 0, 0, 0),                            # accums_init (4-tuple)
    0,                                       # rng_key (per-sample)
    None, None,                              # dropout rates (shared)
)

_PRED_AXES = (
    0,                                       # x_input
    None, None, None, None, None,            # weights
    None, None, None, None, None, None,      # alphas, T_p, config
)


_fwd_single = jit(_forward_and_accum)
_pred_single = jit(_predict_only)
_loss_single = jit(_loss_and_grads)
_apply_sgd = jit(_apply_grads_sgd)
_apply_adam = jit(_apply_grads_adam)

_fwd_batch = jit(vmap(_forward_and_accum, in_axes=_FWD_AXES))
_pred_batch = jit(vmap(_predict_only, in_axes=_PRED_AXES))
# loss/grads vmap: counts(0), per_step(tuple of 0s), accums(tuple of 0s),
#                  w_readout(None), w_soma_h(None), target(0), T(None), temp(None), bias(None)
_LOSS_AXES = (
    0,                                       # readout_counts
    (0, 0, 0, 0, 0, 0),                      # per_step_tensors (6-tuple)
    (0, 0, 0, 0),                            # accums (4-tuple)
    None, None,                              # w_readout, w_soma_h (shared)
    0,                                       # target_smoothed
    None, None, None,                        # T, loss_temperature, loss_count_bias
)
_loss_batch = jit(vmap(_loss_and_grads, in_axes=_LOSS_AXES))


# ══════════════════════════════════════════════════════════════════════
#  Network class — ties everything together (mirrors no_history.Network).
# ══════════════════════════════════════════════════════════════════════

class Network:
    def __init__(
        self,
        key: jnp.ndarray,
        n_inputs: int,
        n_extra: int,
        n_hidden: int,
        n_outputs: int,
        config: NeuronConfig,
        optimizer: str = "sgd",
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
        dropout_extra: float = 0.0,
        dropout_hidden: float = 0.0,
        weight_decay: float = 0.0,
    ):
        self.n_inputs = n_inputs
        self.n_extra = n_extra
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.config = config
        self.optimizer = optimizer
        self.dropout_extra = dropout_extra
        self.dropout_hidden = dropout_hidden
        self.weight_decay = weight_decay

        key_e, key_h, key_r, key_rng = random.split(key, 4)
        self.extra = TwoCompNeuron(key_e, n_extra, n_inputs, config)
        self.hidden = TwoCompNeuron(key_h, n_hidden, n_extra, config)
        self.readout = LIFNeuron(key_r, n_outputs, n_hidden, config)
        self.rng_key = key_rng

        if optimizer == "adam":
            self.beta1 = beta1
            self.beta2 = beta2
            self.adam_eps = adam_eps
            self.adam_step = jnp.array(0, dtype=jnp.int32)
            self.m_dend_e = jnp.zeros_like(self.extra.w_dend)
            self.m_soma_e = jnp.zeros_like(self.extra.w_soma)
            self.m_dend_h = jnp.zeros_like(self.hidden.w_dend)
            self.m_soma_h = jnp.zeros_like(self.hidden.w_soma)
            self.m_readout = jnp.zeros_like(self.readout.w)
            self.v_dend_e = jnp.zeros_like(self.extra.w_dend)
            self.v_soma_e = jnp.zeros_like(self.extra.w_soma)
            self.v_dend_h = jnp.zeros_like(self.hidden.w_dend)
            self.v_soma_h = jnp.zeros_like(self.hidden.w_soma)
            self.v_readout = jnp.zeros_like(self.readout.w)

    # ── Carries ──

    def _e_carry(self, B=None):
        """Extra-layer 8-tuple carry (zeros)."""
        n, k = self.n_extra, self.n_inputs
        s = (B, n) if B else (n,)
        sk = (B, k) if B else (k,)
        snk = (B, n, k) if B else (n, k)
        return (
            jnp.zeros(s), jnp.zeros(s),
            jnp.zeros(s, dtype=jnp.int32), jnp.zeros(s, dtype=jnp.int32),
            jnp.zeros(s), jnp.zeros(sk),
            jnp.zeros(snk), jnp.zeros(snk),
        )

    def _h_carry(self, B=None):
        """Hidden-layer 8-tuple carry (zeros)."""
        n, k = self.n_hidden, self.n_extra
        s = (B, n) if B else (n,)
        sk = (B, k) if B else (k,)
        snk = (B, n, k) if B else (n, k)
        return (
            jnp.zeros(s), jnp.zeros(s),
            jnp.zeros(s, dtype=jnp.int32), jnp.zeros(s, dtype=jnp.int32),
            jnp.zeros(s), jnp.zeros(sk),
            jnp.zeros(snk), jnp.zeros(snk),
        )

    def _r_carry(self, B=None):
        """Readout 3-tuple carry (zeros)."""
        j, n = self.n_outputs, self.n_hidden
        sj = (B, j) if B else (j,)
        sn = (B, n) if B else (n,)
        return (jnp.zeros(sj), jnp.zeros(sj), jnp.zeros(sn))

    def _mirrors(self, B=None):
        """Cross-layer mirrors (zeros). Refreshed when h_h_prev[i] == 0."""
        n_h, n_e, k = self.n_hidden, self.n_extra, self.n_inputs
        sne = (B, n_h, n_e) if B else (n_h, n_e)
        snk = (B, n_h, k) if B else (n_h, k)
        snek = (B, n_h, n_e, k) if B else (n_h, n_e, k)
        return (jnp.zeros(sne), jnp.zeros(sne), jnp.zeros(snk), jnp.zeros(snek))

    def _accums(self, B=None):
        """Online gradient accumulators (J-axis kept). Folded with global_error post-scan."""
        J, N, n_e, K = self.n_outputs, self.n_hidden, self.n_extra, self.n_inputs
        s_dh = (B, J, N, n_e) if B else (J, N, n_e)
        s_e = (B, J, n_e, K) if B else (J, n_e, K)
        return (jnp.zeros(s_dh), jnp.zeros(s_e), jnp.zeros(s_e), jnp.zeros(s_e))

    def _weights(self):
        return (self.extra.w_dend, self.extra.w_soma,
                self.hidden.w_dend, self.hidden.w_soma,
                self.readout.w)

    def _params(self):
        # alpha_s, alpha_d are identical for both 2-comp layers (same NeuronConfig).
        return (self.extra.alpha_s, self.extra.alpha_d, self.readout.alpha_m,
                self.extra.T_p, self.hidden.T_p, self.config)

    def _smooth_targets(self, targets):
        cfg = self.config
        one_hot = jnp.eye(self.n_outputs)[targets]
        return one_hot * (1 - cfg.loss_label_smoothing) + cfg.loss_label_smoothing / self.n_outputs

    def _next_key(self):
        self.rng_key, subkey = random.split(self.rng_key)
        return subkey

    # ── Weight update ──

    def _update_weights(self, g_de, g_se, g_dh, g_sh, g_r, lr, clip_value):
        if self.optimizer == "adam":
            self.adam_step = self.adam_step + 1
            result = _apply_adam(
                self.extra.w_dend, self.extra.w_soma,
                self.hidden.w_dend, self.hidden.w_soma, self.readout.w,
                g_de, g_se, g_dh, g_sh, g_r,
                self.m_dend_e, self.m_soma_e, self.m_dend_h, self.m_soma_h, self.m_readout,
                self.v_dend_e, self.v_soma_e, self.v_dend_h, self.v_soma_h, self.v_readout,
                self.adam_step, lr, self.beta1, self.beta2, self.adam_eps,
                clip_value, self.weight_decay,
            )
            (self.extra.w_dend, self.extra.w_soma,
             self.hidden.w_dend, self.hidden.w_soma, self.readout.w,
             self.m_dend_e, self.m_soma_e, self.m_dend_h, self.m_soma_h, self.m_readout,
             self.v_dend_e, self.v_soma_e, self.v_dend_h, self.v_soma_h, self.v_readout) = result
        else:
            (self.extra.w_dend, self.extra.w_soma,
             self.hidden.w_dend, self.hidden.w_soma, self.readout.w) = _apply_sgd(
                *self._weights(), g_de, g_se, g_dh, g_sh, g_r, lr, clip_value, self.weight_decay,
            )

    # ── Single-sample API ──

    def train_step(self, x_input, target, lr=1e-3, clip_value=1.0):
        """Train on one sample. Returns (loss, prediction, grad_norms_dict)."""
        T = x_input.shape[0]

        counts, per_step, accums = _fwd_single(
            x_input, *self._weights(), *self._params(),
            self._e_carry(), self._h_carry(), self._r_carry(),
            self._mirrors(), self._accums(),
            self._next_key(), self.dropout_extra, self.dropout_hidden,
        )

        loss, pred, g_r, g_sh, g_dh, g_se, g_de = _loss_single(
            counts, per_step, accums,
            self.readout.w, self.hidden.w_soma,
            self._smooth_targets(target), T,
            self.config.loss_temperature, self.config.loss_count_bias,
        )

        gnorms = {
            "readout": float(jnp.linalg.norm(g_r)),
            "soma_h": float(jnp.linalg.norm(g_sh)),
            "dend_h": float(jnp.linalg.norm(g_dh)),
            "soma_e": float(jnp.linalg.norm(g_se)),
            "dend_e": float(jnp.linalg.norm(g_de)),
        }

        self._update_weights(g_de, g_se, g_dh, g_sh, g_r, lr, clip_value)
        return float(loss), int(pred), gnorms

    def predict(self, x_input):
        counts = _pred_single(x_input, *self._weights(), *self._params())
        return int(jnp.argmax(counts))

    # ── Batched API ──

    def batch_train_step(self, x_batch, targets, lr=1e-3, clip_value=1.0):
        """Train on B samples in parallel. Returns (mean_loss, predictions, grad_norms_dict)."""
        B = x_batch.shape[0]
        T = x_batch.shape[1]
        batch_keys = random.split(self._next_key(), B)

        counts, per_step, accums = _fwd_batch(
            x_batch, *self._weights(), *self._params(),
            self._e_carry(B), self._h_carry(B), self._r_carry(B),
            self._mirrors(B), self._accums(B),
            batch_keys, self.dropout_extra, self.dropout_hidden,
        )

        losses, preds, g_r, g_sh, g_dh, g_se, g_de = _loss_batch(
            counts, per_step, accums,
            self.readout.w, self.hidden.w_soma,
            self._smooth_targets(targets), T,
            self.config.loss_temperature, self.config.loss_count_bias,
        )

        g_r_avg = jnp.mean(g_r, axis=0)
        g_sh_avg = jnp.mean(g_sh, axis=0)
        g_dh_avg = jnp.mean(g_dh, axis=0)
        g_se_avg = jnp.mean(g_se, axis=0)
        g_de_avg = jnp.mean(g_de, axis=0)

        gnorms = {
            "readout": float(jnp.linalg.norm(g_r_avg)),
            "soma_h": float(jnp.linalg.norm(g_sh_avg)),
            "dend_h": float(jnp.linalg.norm(g_dh_avg)),
            "soma_e": float(jnp.linalg.norm(g_se_avg)),
            "dend_e": float(jnp.linalg.norm(g_de_avg)),
        }

        self._update_weights(g_de_avg, g_se_avg, g_dh_avg, g_sh_avg, g_r_avg, lr, clip_value)
        return float(jnp.mean(losses)), preds, gnorms

    def batch_predict(self, x_batch):
        counts = _pred_batch(x_batch, *self._weights(), *self._params())
        return jnp.argmax(counts, axis=1)
