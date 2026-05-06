import jax
import jax.numpy as jnp
from jax import random, jit, lax, vmap
from jax.nn import sigmoid

from config import NeuronConfig, surrogate_sigma
from two_comp_neuron import TwoCompNeuron
from lif_neuron import LIFNeuron


# ══════════════════════════════════════════════════════════════════════
#  Core functions — each processes ONE sample.
#
#  These have no decorators so we can wrap them two ways:
#    jit(fn)            → fast single-sample execution
#    jit(vmap(fn, ...)) → fast batched execution (B samples in parallel)
# ══════════════════════════════════════════════════════════════════════

def _forward_and_accum(
    x_input, dend_in_BN, soma_in_BN, z_norm_d, z_norm_s, w_readout,
    alpha_s, alpha_d, alpha_m, T_p, gamma_h, config,
    h_carry_init, r_carry_init,
    A_d_init, A_g_d_init, A_b_d_init,
    rng_key, dropout_rate,
):
    """Forward pass + gradient accumulator bookkeeping for one sample.

    Inputs:
      x_input:      (T, K)  raw input spike train (still needed for the
                            W_soma / W_dend eligibility traces).
      dend_in_BN:   (T, N)  per-neuron dendritic input AFTER batch-norm
                            (i.e. γ_BN_d·z_norm_d + β_BN_d). Replaces what
                            the scan used to compute as x @ w_dend.T.
      soma_in_BN:   (T, N)  per-neuron somatic input AFTER batch-norm.
      z_norm_d:     (T, N)  per-step normalized dendritic projection
                            (= (z_d − μ_d)/√(σ²_d+ε)). Used to build the
                            γ_BN_d eligibility trace; gradients of W_dend
                            ignore the centering/rescaling Jacobian
                            (frozen-stats e-prop approximation), so we
                            only need this signal for ∂L/∂γ_BN_d.
      z_norm_s:     (T, N)  per-step normalized somatic projection.
      gamma_h:      (N,)    per-neuron threshold-reduction factor.
      A_g_d_init,
      A_b_d_init:   (J, N)  zero-initialised online accumulators for
                            γ_BN_d and β_BN_d (mirrors of A_d).
      rng_key:      PRNG key for dropout masks.
      dropout_rate: fraction of hidden spikes to drop (0.0 = no dropout).

    Returns:
      readout_counts (J,),
      A_readout      (J, N),
      A_soma         (J, N, K)         -- existing W_soma accumulator (un-scaled).
      A_dend         (J, N, K)         -- existing W_dend accumulator (un-scaled).
      A_gamma_h      (J, N),
      A_gamma_BN_s   (J, N),
      A_beta_BN_s    (J, N),
      A_gamma_BN_d   (J, N),
      A_beta_BN_d    (J, N).
    """
    T = x_input.shape[0]
    n_hidden = soma_in_BN.shape[1]
    time_indices = jnp.arange(T, dtype=jnp.int32)
    dropout_keys = random.split(rng_key, T)
    dropout_scale = 1.0 / (1.0 - dropout_rate)
    dtype = soma_in_BN.dtype

    # Per-step BN-parameter eligibility traces. Soma-side traces are simple
    # exponential filters with α_s; dend-side traces are gated by (1-h_prev)
    # and have an "at t'_p" mirror, mirroring the existing dmu_dw / dmu_atp
    # structure.
    E_g_s_init = jnp.zeros(n_hidden, dtype=dtype)
    E_b_s_init = jnp.zeros(n_hidden, dtype=dtype)
    E_g_d_init = jnp.zeros(n_hidden, dtype=dtype)
    E_b_d_init = jnp.zeros(n_hidden, dtype=dtype)
    E_g_d_atp_init = jnp.zeros(n_hidden, dtype=dtype)
    E_b_d_atp_init = jnp.zeros(n_hidden, dtype=dtype)

    def step(carry, inputs):
        (h_carry, r_carry, A_d, A_g_d, A_b_d,
         E_g_s, E_b_s, E_g_d, E_b_d, E_g_d_atp, E_b_d_atp) = carry
        dend_in, soma_in, zn_d, zn_s, x_t, t, drop_key = inputs

        h_carry, h_o, h_v_pre, h_h, h_h_prev, h_mu_at_tp = TwoCompNeuron.forward_step(
            h_carry, dend_in, soma_in, t, alpha_s, alpha_d, T_p, gamma_h, config,
        )
        hidden_o_float = h_o.astype(dtype)

        mask = random.bernoulli(drop_key, 1.0 - dropout_rate, (n_hidden,)).astype(dtype)
        hidden_o_float = hidden_o_float * mask * dropout_scale

        r_carry, r_o, r_v_pre, r_E = LIFNeuron.forward_step(
            r_carry, hidden_o_float, w_readout, alpha_m, config.v_th,
        )

        # ── W_soma / W_dend eligibility (raw input, frozen-stats approx for BN) ──
        mu_c, v_c, h_c, tp_c, matp_c, E_soma_c, dmu_c, dmu_atp_c = h_carry
        x_t_f = x_t.astype(dtype)
        E_soma_new = TwoCompNeuron.update_somatic_eligibility(E_soma_c, x_t_f, alpha_s)
        dmu_new, dmu_atp_new = TwoCompNeuron.update_dendritic_eligibility(
            dmu_c, dmu_atp_c, x_t_f, h_h_prev, alpha_d,
        )
        h_carry = (mu_c, v_c, h_c, tp_c, matp_c, E_soma_new, dmu_new, dmu_atp_new)

        # ── BN-parameter eligibility traces ──
        # Soma side: y(t,i) = γ_s[i]·z_norm_s(t,i) + β_s[i] feeds v(t,i).
        E_g_s_new = alpha_s * E_g_s + zn_s
        E_b_s_new = alpha_s * E_b_s + 1.0
        # Dend side: y(t,i) feeds μ(t,i) gated by (1-h_prev). Mirror at t'_p.
        not_h_prev = (1.0 - h_h_prev.astype(dtype))
        E_g_d_new = alpha_d * E_g_d + not_h_prev * zn_d
        E_b_d_new = alpha_d * E_b_d + not_h_prev * 1.0
        refresh = (h_h_prev == 0)
        E_g_d_atp_new = jnp.where(refresh, E_g_d_new, E_g_d_atp)
        E_b_d_atp_new = jnp.where(refresh, E_b_d_new, E_b_d_atp)

        # ── Surrogates ──
        sp_readout = surrogate_sigma(r_v_pre - config.v_th, config.beta_s)
        sp_hidden = surrogate_sigma(
            h_v_pre + gamma_h * h_h - config.v_th, config.beta_s,
        )
        hp_hidden = surrogate_sigma(h_mu_at_tp - config.mu_th, config.beta_d)

        # ── Online accumulators ──
        eta = sp_readout[:, None] * w_readout * sp_hidden[None, :]            # (J, N)
        eta_d = eta * (hp_hidden * gamma_h)[None, :]                          # (J, N)
        A_d = A_d + jnp.einsum("ji,ik->jik", eta_d, dmu_atp_new)
        A_g_d = A_g_d + eta_d * E_g_d_atp_new[None, :]
        A_b_d = A_b_d + eta_d * E_b_d_atp_new[None, :]

        h_h_f = h_h.astype(dtype)
        new_carry = (h_carry, r_carry, A_d, A_g_d, A_b_d,
                     E_g_s_new, E_b_s_new,
                     E_g_d_new, E_b_d_new,
                     E_g_d_atp_new, E_b_d_atp_new)
        per_step = (sp_readout, sp_hidden, r_E, E_soma_new, h_h_f,
                    E_g_s_new, E_b_s_new)
        return new_carry, per_step

    init_carry = (h_carry_init, r_carry_init, A_d_init, A_g_d_init, A_b_d_init,
                  E_g_s_init, E_b_s_init,
                  E_g_d_init, E_b_d_init,
                  E_g_d_atp_init, E_b_d_atp_init)
    scan_inputs = (dend_in_BN, soma_in_BN, z_norm_d, z_norm_s,
                   x_input, time_indices, dropout_keys)
    final_carry, per_step_all = lax.scan(step, init_carry, scan_inputs)

    (sp_r, sp_h, E_r, E_s, h_h_T,
     E_g_s_T, E_b_s_T) = per_step_all
    (_, r_carry_f, A_d_f, A_g_d_f, A_b_d_f,
     _, _, _, _, _, _) = final_carry
    readout_counts = r_carry_f[1]

    A_readout = jnp.einsum("ti,tj->ij", sp_r, E_r)
    C_soma = jnp.einsum("tj,ti,tk->jik", sp_r, sp_h, E_s)
    A_soma = w_readout[:, :, None] * C_soma                                   # (J, N, K)

    # γ_h accumulator (same as before).
    C_gamma = jnp.einsum("tj,ti,ti->ji", sp_r, sp_h, h_h_T)
    A_gamma_h = w_readout * C_gamma

    # Soma-side BN accumulators: post-scan reductions analogous to A_soma but
    # with the per-input E_soma replaced by the per-neuron filter E_g_s / E_b_s.
    C_g_s = jnp.einsum("tj,ti,ti->ji", sp_r, sp_h, E_g_s_T)
    C_b_s = jnp.einsum("tj,ti,ti->ji", sp_r, sp_h, E_b_s_T)
    A_gamma_BN_s = w_readout * C_g_s
    A_beta_BN_s = w_readout * C_b_s

    return (readout_counts, A_readout, A_soma, A_d_f, A_gamma_h,
            A_gamma_BN_s, A_beta_BN_s, A_g_d_f, A_b_d_f)


def _predict_only(
    x_input, dend_in_BN, soma_in_BN, w_readout,
    alpha_s, alpha_d, alpha_m, T_p, gamma_h, config,
):
    """Forward pass only — no gradient bookkeeping. Returns readout_counts (J,)."""
    dend_inputs = dend_in_BN
    soma_inputs = soma_in_BN
    T = x_input.shape[0]
    n_hidden = dend_inputs.shape[1]
    n_outputs = w_readout.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)

    def step(carry, inputs):
        mu, v, h, t_prime, mu_at_tp, r_v, r_counts = carry
        dend_in, soma_in, t = inputs

        t_prime_new = jnp.where(t == 0, 0, jnp.where(h == 1, t_prime, t))
        mu_new = jnp.where(t > 0, alpha_d * mu + (1 - h) * dend_in, dend_in)
        mu_at_tp_new = jnp.where(h == 0, mu_new, mu_at_tp)

        plat_dur = t - t_prime_new
        h_new = jnp.where(
            (mu_at_tp_new >= config.mu_th) & (plat_dur <= T_p) & (plat_dur >= 0),
            1, 0,
        ).astype(jnp.int32)

        v_pre = jnp.where(t > 0, alpha_s * v + soma_in, soma_in)
        o_h = jnp.where(v_pre >= config.v_th - gamma_h * h_new, 1, 0).astype(jnp.int32)
        v_new = v_pre * (1 - o_h)

        r_in = o_h.astype(jnp.float64) @ w_readout.T
        r_v_new = alpha_m * r_v + r_in
        r_o = jnp.where(r_v_new >= config.v_th, 1, 0).astype(jnp.int32)
        r_v_new = r_v_new * (1 - r_o)
        r_counts_new = r_counts + r_o

        return (mu_new, v_new, h_new, t_prime_new, mu_at_tp_new, r_v_new, r_counts_new), None

    init = (
        jnp.zeros(n_hidden), jnp.zeros(n_hidden),
        jnp.zeros(n_hidden, dtype=jnp.int32), jnp.zeros(n_hidden, dtype=jnp.int32),
        jnp.zeros(n_hidden), jnp.zeros(n_outputs), jnp.zeros(n_outputs),
    )
    final, _ = lax.scan(step, init, (dend_inputs, soma_inputs, time_indices))
    return final[6]


def _loss_and_grads(
    readout_counts, A_readout, A_soma, A_dend, A_gamma_h,
    A_gamma_BN_s, A_beta_BN_s, A_gamma_BN_d, A_beta_BN_d,
    bn_scale_s, bn_scale_d,
    target_smoothed, T, loss_temperature, loss_count_bias,
):
    """Compute loss and weight gradients for one sample.

    `bn_scale_s` and `bn_scale_d` are the per-neuron γ_BN_s[i]/σ_s[i] and
    γ_BN_d[i]/σ_d[i] factors that come from the BN forward (frozen-stats
    e-prop approximation: gradients of W_soma / W_dend pick this up
    row-wise, while γ_BN's and β_BN's gradients are exact in the same
    approximation).

    Returns gradients for: readout, W_soma, W_dend, gamma_h, γ_BN_s,
    β_BN_s, γ_BN_d, β_BN_d. The chain-rule conversion grad_gamma_h ->
    grad_rho_h happens outside (it depends on current gamma_h).
    """
    scaled_logits = readout_counts / loss_temperature + loss_count_bias
    probs = jnp.exp(scaled_logits - jnp.max(scaled_logits))
    probs = probs / jnp.sum(probs)

    prediction = jnp.argmax(readout_counts)
    loss = -jnp.sum(target_smoothed * jnp.log(probs + 1e-8))
    global_error = target_smoothed - probs

    grad_readout = (global_error[:, None] * A_readout) / T
    # Frozen-stats BN scaling for the underlying weights: row i is scaled
    # by γ_BN[i]/σ[i]. When use_bn=False these are 1, so the math is unchanged.
    grad_soma = jnp.einsum("j,jik->ik", global_error, A_soma) / (T * 8.0)
    grad_soma = grad_soma * bn_scale_s[:, None]
    grad_dend = jnp.einsum("j,jik->ik", global_error, A_dend) / T
    grad_dend = grad_dend * bn_scale_d[:, None]
    grad_gamma_h = jnp.einsum("j,ji->i", global_error, A_gamma_h) / T
    grad_gamma_BN_s = jnp.einsum("j,ji->i", global_error, A_gamma_BN_s) / T
    grad_beta_BN_s = jnp.einsum("j,ji->i", global_error, A_beta_BN_s) / T
    grad_gamma_BN_d = jnp.einsum("j,ji->i", global_error, A_gamma_BN_d) / T
    grad_beta_BN_d = jnp.einsum("j,ji->i", global_error, A_beta_BN_d) / T

    return (loss, prediction, grad_readout, grad_soma, grad_dend, grad_gamma_h,
            grad_gamma_BN_s, grad_beta_BN_s, grad_gamma_BN_d, grad_beta_BN_d)


def _apply_grads(
    w_dend, w_soma, w_readout, rho_h,
    gamma_BN_s, beta_BN_s, gamma_BN_d, beta_BN_d,
    g_dend, g_soma, g_readout, g_rho,
    g_gamma_BN_s, g_beta_BN_s, g_gamma_BN_d, g_beta_BN_d,
    lr, lr_gamma, lr_bn, clip_value, weight_decay,
):
    """SGD with decoupled weight decay.

    Weight params (W_*) use lr + weight_decay. rho_h uses lr_gamma without
    weight decay. BN params use lr_bn without weight decay.
    """
    def clip(g):
        return jnp.clip(g, -clip_value, clip_value)
    g_readout = clip(g_readout); g_soma = clip(g_soma); g_dend = clip(g_dend)
    g_rho = clip(g_rho)
    g_gamma_BN_s = clip(g_gamma_BN_s); g_beta_BN_s = clip(g_beta_BN_s)
    g_gamma_BN_d = clip(g_gamma_BN_d); g_beta_BN_d = clip(g_beta_BN_d)
    return (
        w_dend + lr * g_dend - lr * weight_decay * w_dend,
        w_soma + lr * g_soma - lr * weight_decay * w_soma,
        w_readout + lr * g_readout - lr * weight_decay * w_readout,
        rho_h + lr_gamma * g_rho,
        gamma_BN_s + lr_bn * g_gamma_BN_s,
        beta_BN_s + lr_bn * g_beta_BN_s,
        gamma_BN_d + lr_bn * g_gamma_BN_d,
        beta_BN_d + lr_bn * g_beta_BN_d,
    )


def _adam_apply(
    w_d, w_s, w_r, rho_h, g_BN_s, b_BN_s, g_BN_d, b_BN_d,
    g_d, g_s, g_r, g_rho, gg_s, gb_s, gg_d, gb_d,
    m_d, m_s, m_r, m_rho, m_g_s, m_b_s, m_g_d, m_b_d,
    v_d, v_s, v_r, v_rho, v_g_s, v_b_s, v_g_d, v_b_d,
    step, lr, lr_gamma, lr_bn, beta1, beta2, eps, clip_value, weight_decay,
):
    """AdamW-style decoupled weight decay (no decay on rho_h / BN params)."""
    def update_one(w, g, m, v, lr_used, wd):
        g = jnp.clip(g, -clip_value, clip_value)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        w = w + lr_used * m_hat / (jnp.sqrt(v_hat) + eps) - lr_used * wd * w
        return w, m, v

    w_d, m_d, v_d = update_one(w_d, g_d, m_d, v_d, lr, weight_decay)
    w_s, m_s, v_s = update_one(w_s, g_s, m_s, v_s, lr, weight_decay)
    w_r, m_r, v_r = update_one(w_r, g_r, m_r, v_r, lr, weight_decay)
    rho_h, m_rho, v_rho = update_one(rho_h, g_rho, m_rho, v_rho, lr_gamma, 0.0)
    g_BN_s, m_g_s, v_g_s = update_one(g_BN_s, gg_s, m_g_s, v_g_s, lr_bn, 0.0)
    b_BN_s, m_b_s, v_b_s = update_one(b_BN_s, gb_s, m_b_s, v_b_s, lr_bn, 0.0)
    g_BN_d, m_g_d, v_g_d = update_one(g_BN_d, gg_d, m_g_d, v_g_d, lr_bn, 0.0)
    b_BN_d, m_b_d, v_b_d = update_one(b_BN_d, gb_d, m_b_d, v_b_d, lr_bn, 0.0)
    return (w_d, w_s, w_r, rho_h, g_BN_s, b_BN_s, g_BN_d, b_BN_d,
            m_d, m_s, m_r, m_rho, m_g_s, m_b_s, m_g_d, m_b_d,
            v_d, v_s, v_r, v_rho, v_g_s, v_b_s, v_g_d, v_b_d)


# ══════════════════════════════════════════════════════════════════════
#  vmap in_axes: which args get a batch dimension (0) vs stay shared (None)
#
#  For _forward_and_accum:
#    x_input → batched (B,T,K)       init carries → batched (B,...)
#    weights → shared                 config/alphas → shared
# ══════════════════════════════════════════════════════════════════════

_FWD_AXES = (
    0, 0, 0, 0, 0,                       # x_input, dend_in_BN, soma_in_BN, z_norm_d, z_norm_s
    None,                                # w_readout (shared)
    None, None, None, None, None, None,  # alpha_s, alpha_d, alpha_m, T_p, gamma_h, config
    (0, 0, 0, 0, 0, 0, 0, 0),            # h_carry_init (8-tuple, each batched)
    (0, 0, 0),                           # r_carry_init (3-tuple, each batched)
    0, 0, 0,                             # A_d_init, A_g_d_init, A_b_d_init
    0,                                   # rng_key (per-sample)
    None,                                # dropout_rate (shared)
)

_PRED_AXES = (
    0, 0, 0,                             # x_input, dend_in_BN, soma_in_BN
    None,                                # w_readout (shared)
    None, None, None, None, None, None,  # alphas, T_p, gamma_h, config
)

_LOSS_AXES = (
    0, 0, 0, 0, 0,                       # counts, A_r, A_s, A_d, A_gamma_h (per-sample)
    0, 0, 0, 0,                          # A_gamma_BN_s, A_beta_BN_s, A_gamma_BN_d, A_beta_BN_d
    None, None,                          # bn_scale_s, bn_scale_d (shared)
    0,                                   # target_smoothed (per-sample)
    None, None, None,                    # T, loss_temperature, loss_count_bias
)


# ══════════════════════════════════════════════════════════════════════
#  Pre-compiled versions:
#    _*_single = jit(core_fn)                  → one sample
#    _*_batch  = jit(vmap(core_fn, in_axes=…)) → B samples in parallel
# ══════════════════════════════════════════════════════════════════════

_fwd_single = jit(_forward_and_accum)
_pred_single = jit(_predict_only)
_loss_single = jit(_loss_and_grads)
_apply = jit(_apply_grads)
_adam = jit(_adam_apply)

_fwd_batch = jit(vmap(_forward_and_accum, in_axes=_FWD_AXES))
_pred_batch = jit(vmap(_predict_only, in_axes=_PRED_AXES))
_loss_batch = jit(vmap(_loss_and_grads, in_axes=_LOSS_AXES))


# ══════════════════════════════════════════════════════════════════════
#  Network class — ties everything together
# ══════════════════════════════════════════════════════════════════════

class Network:
    def __init__(
        self,
        key: jnp.ndarray,
        n_inputs: int,
        n_hidden: int,
        n_outputs: int,
        config: NeuronConfig,
        optimizer: str = "sgd",
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
        dropout_rate: float = 0.0,
        weight_decay: float = 0.0,
        train_gamma: bool = False,
        lr_gamma: float | None = None,
        use_bn: bool = False,
        train_bn: bool = True,
        lr_bn: float | None = None,
    ):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.config = config
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.train_gamma = train_gamma
        self.lr_gamma = lr_gamma
        # Batch-norm: use_bn enables BN forward/backward; train_bn allows
        # γ_BN/β_BN to update. Disabling use_bn keeps γ_BN=1, β_BN=0,
        # μ=0, σ=1 so the math reduces to the no-BN case exactly.
        self.use_bn = use_bn
        self.train_bn = train_bn
        self.lr_bn = lr_bn

        key_h, key_r, key_rng = random.split(key, 3)
        self.hidden = TwoCompNeuron(key_h, n_hidden, n_inputs, config)
        self.readout = LIFNeuron(key_r, n_outputs, n_hidden, config)
        self.rng_key = key_rng

        ratio = float(jnp.clip(config.gamma / config.v_th, 1e-3, 1.0 - 1e-3))
        rho_init = float(jnp.log(ratio / (1.0 - ratio)))
        dtype = self.hidden.w_dend.dtype
        self.rho_h = jnp.full((n_hidden,), rho_init, dtype=dtype)

        # BN parameters (one set per pre-activation: somatic and dendritic).
        # Initialise gamma=1, beta=0 so the BN forward is identity until the
        # first running-stats update.
        self.gamma_BN_s = jnp.ones((n_hidden,), dtype=dtype)
        self.beta_BN_s = jnp.zeros((n_hidden,), dtype=dtype)
        self.gamma_BN_d = jnp.ones((n_hidden,), dtype=dtype)
        self.beta_BN_d = jnp.zeros((n_hidden,), dtype=dtype)
        self.running_mean_s = jnp.zeros((n_hidden,), dtype=dtype)
        self.running_var_s = jnp.ones((n_hidden,), dtype=dtype)
        self.running_mean_d = jnp.zeros((n_hidden,), dtype=dtype)
        self.running_var_d = jnp.ones((n_hidden,), dtype=dtype)

        if optimizer == "adam":
            self.beta1 = beta1
            self.beta2 = beta2
            self.adam_eps = adam_eps
            self.adam_step = jnp.array(0, dtype=jnp.int32)
            self.m_dend = jnp.zeros_like(self.hidden.w_dend)
            self.m_soma = jnp.zeros_like(self.hidden.w_soma)
            self.m_readout = jnp.zeros_like(self.readout.w)
            self.m_rho = jnp.zeros_like(self.rho_h)
            self.m_g_BN_s = jnp.zeros_like(self.gamma_BN_s)
            self.m_b_BN_s = jnp.zeros_like(self.beta_BN_s)
            self.m_g_BN_d = jnp.zeros_like(self.gamma_BN_d)
            self.m_b_BN_d = jnp.zeros_like(self.beta_BN_d)
            self.v_dend = jnp.zeros_like(self.hidden.w_dend)
            self.v_soma = jnp.zeros_like(self.hidden.w_soma)
            self.v_readout = jnp.zeros_like(self.readout.w)
            self.v_rho = jnp.zeros_like(self.rho_h)
            self.v_g_BN_s = jnp.zeros_like(self.gamma_BN_s)
            self.v_b_BN_s = jnp.zeros_like(self.beta_BN_s)
            self.v_g_BN_d = jnp.zeros_like(self.gamma_BN_d)
            self.v_b_BN_d = jnp.zeros_like(self.beta_BN_d)

    @property
    def gamma_h(self) -> jnp.ndarray:
        """Per-neuron gamma (shape (n_hidden,)), bounded in (0, v_th)."""
        return self.config.v_th * sigmoid(self.rho_h)

    # ── Batch-norm helpers (Python-side; called outside vmap) ──

    def _bn_train(self, z_raw, gamma, beta, running_mean, running_var):
        """BN with batch stats. z_raw is (B, T, N) or (T, N).

        Returns (z_BN, z_norm, batch_mean, batch_var). The batch stats are
        treated as stop_gradient downstream (we never differentiate through
        them — that's the frozen-stats e-prop approximation). Running stats
        update is applied by the caller using batch_mean/batch_var.
        """
        if not self.use_bn:
            z_norm = z_raw
            z_BN = z_raw
            return z_BN, z_norm, running_mean, running_var
        # Reduce over all axes except the last (per-neuron stats over (B, T)).
        reduce_axes = tuple(range(z_raw.ndim - 1))
        batch_mean = jnp.mean(z_raw, axis=reduce_axes)
        batch_var = jnp.var(z_raw, axis=reduce_axes)
        std = jnp.sqrt(batch_var + self.config.bn_eps)
        z_norm = (z_raw - batch_mean) / std
        z_BN = gamma * z_norm + beta
        return z_BN, z_norm, batch_mean, batch_var

    def _bn_eval(self, z_raw, gamma, beta, running_mean, running_var):
        """BN with running stats (used at predict time)."""
        if not self.use_bn:
            return z_raw, z_raw
        std = jnp.sqrt(running_var + self.config.bn_eps)
        z_norm = (z_raw - running_mean) / std
        z_BN = gamma * z_norm + beta
        return z_BN, z_norm

    def _bn_scale(self, gamma, running_var):
        """Per-neuron γ_BN/σ that scales W gradients (frozen-stats Jacobian).

        Returns ones when use_bn=False so existing math is unchanged.
        """
        if not self.use_bn:
            return jnp.ones((self.n_hidden,), dtype=gamma.dtype)
        return gamma / jnp.sqrt(running_var + self.config.bn_eps)

    def _bn_update_running(self, batch_mean, batch_var, running_mean, running_var):
        m = self.config.bn_momentum
        return (
            (1.0 - m) * running_mean + m * batch_mean,
            (1.0 - m) * running_var + m * batch_var,
        )

    # ── Helpers to build zero-initialized carries ──

    def _h_carry(self, B=None):
        """Hidden neuron carry. B=None → single sample, B=int → batched."""
        n, k = self.n_hidden, self.n_inputs
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
        """Readout neuron carry."""
        j, n = self.n_outputs, self.n_hidden
        sj = (B, j) if B else (j,)
        sn = (B, n) if B else (n,)
        return (jnp.zeros(sj), jnp.zeros(sj), jnp.zeros(sn))

    def _A_d_zeros(self, B=None):
        """Dendritic accumulator zeros (J, N, K)."""
        base = (self.n_outputs, self.n_hidden, self.n_inputs)
        return jnp.zeros((B,) + base if B else base)

    def _A_BN_d_zeros(self, B=None):
        """BN-dend accumulator zeros (J, N) — for γ_BN_d and β_BN_d."""
        base = (self.n_outputs, self.n_hidden)
        return jnp.zeros((B,) + base if B else base)

    def _weights(self):
        return self.hidden.w_dend, self.hidden.w_soma, self.readout.w

    def _params(self):
        return (self.hidden.alpha_s, self.hidden.alpha_d, self.readout.alpha_m,
                self.hidden.T_p, self.gamma_h, self.config)

    def _project_and_bn(self, x, training):
        """Project x with W and apply BN (batch or running stats).

        x: (T, K) for single-sample, (B, T, K) for batched.
        Returns soma_in_BN, dend_in_BN, z_norm_s, z_norm_d, var_used_s,
        var_used_d. The `var_used_*` outputs are the variance the FORWARD
        actually divided by — batch_var when training, running_var when
        eval. The frozen-stats BN backward uses these so the W-gradient
        scale matches the forward (otherwise running/batch mismatch
        early in training distorts the gradient magnitude).
        """
        z_soma_raw = jnp.einsum("...k,ik->...i", x, self.hidden.w_soma)
        z_dend_raw = jnp.einsum("...k,ik->...i", x, self.hidden.w_dend)
        if training:
            soma_BN, z_norm_s, mean_s, var_s = self._bn_train(
                z_soma_raw, self.gamma_BN_s, self.beta_BN_s,
                self.running_mean_s, self.running_var_s,
            )
            dend_BN, z_norm_d, mean_d, var_d = self._bn_train(
                z_dend_raw, self.gamma_BN_d, self.beta_BN_d,
                self.running_mean_d, self.running_var_d,
            )
            if self.use_bn:
                self.running_mean_s, self.running_var_s = self._bn_update_running(
                    mean_s, var_s, self.running_mean_s, self.running_var_s,
                )
                self.running_mean_d, self.running_var_d = self._bn_update_running(
                    mean_d, var_d, self.running_mean_d, self.running_var_d,
                )
            return soma_BN, dend_BN, z_norm_s, z_norm_d, var_s, var_d
        else:
            soma_BN, z_norm_s = self._bn_eval(
                z_soma_raw, self.gamma_BN_s, self.beta_BN_s,
                self.running_mean_s, self.running_var_s,
            )
            dend_BN, z_norm_d = self._bn_eval(
                z_dend_raw, self.gamma_BN_d, self.beta_BN_d,
                self.running_mean_d, self.running_var_d,
            )
            return (soma_BN, dend_BN, z_norm_s, z_norm_d,
                    self.running_var_s, self.running_var_d)

    def _smooth_targets(self, targets):
        """Scalar label or (B,) labels → smoothed one-hot vector(s)."""
        cfg = self.config
        one_hot = jnp.eye(self.n_outputs)[targets]
        return one_hot * (1 - cfg.loss_label_smoothing) + cfg.loss_label_smoothing / self.n_outputs

    def _grad_rho_h(self, grad_gamma_h):
        """Chain rule: d(gamma)/d(rho) = gamma * (1 - gamma/v_th).

        If train_gamma is False we zero the gradient so rho_h stays put.
        """
        if not self.train_gamma:
            return jnp.zeros_like(self.rho_h)
        gh = self.gamma_h
        return grad_gamma_h * gh * (1.0 - gh / self.config.v_th)

    def _update_weights(self, g_d, g_s, g_r, g_gamma,
                        g_g_BN_s, g_b_BN_s, g_g_BN_d, g_b_BN_d,
                        lr, clip_value):
        """Apply gradients using the configured optimizer (SGD or Adam).

        Each non-W parameter group has its own learning rate (lr_gamma for
        rho_h, lr_bn for the four BN params). When train_gamma / train_bn
        flags are off, the corresponding gradients are zeroed before applying
        so the parameters stay frozen.
        """
        g_rho = self._grad_rho_h(g_gamma)
        if not self.train_bn:
            g_g_BN_s = jnp.zeros_like(self.gamma_BN_s)
            g_b_BN_s = jnp.zeros_like(self.beta_BN_s)
            g_g_BN_d = jnp.zeros_like(self.gamma_BN_d)
            g_b_BN_d = jnp.zeros_like(self.beta_BN_d)
        lr_gamma = lr if self.lr_gamma is None else self.lr_gamma
        lr_bn = lr if self.lr_bn is None else self.lr_bn
        if self.optimizer == "adam":
            self.adam_step = self.adam_step + 1
            result = _adam(
                self.hidden.w_dend, self.hidden.w_soma, self.readout.w, self.rho_h,
                self.gamma_BN_s, self.beta_BN_s, self.gamma_BN_d, self.beta_BN_d,
                g_d, g_s, g_r, g_rho, g_g_BN_s, g_b_BN_s, g_g_BN_d, g_b_BN_d,
                self.m_dend, self.m_soma, self.m_readout, self.m_rho,
                self.m_g_BN_s, self.m_b_BN_s, self.m_g_BN_d, self.m_b_BN_d,
                self.v_dend, self.v_soma, self.v_readout, self.v_rho,
                self.v_g_BN_s, self.v_b_BN_s, self.v_g_BN_d, self.v_b_BN_d,
                self.adam_step, lr, lr_gamma, lr_bn,
                self.beta1, self.beta2, self.adam_eps,
                clip_value, self.weight_decay,
            )
            (self.hidden.w_dend, self.hidden.w_soma, self.readout.w, self.rho_h,
             self.gamma_BN_s, self.beta_BN_s, self.gamma_BN_d, self.beta_BN_d,
             self.m_dend, self.m_soma, self.m_readout, self.m_rho,
             self.m_g_BN_s, self.m_b_BN_s, self.m_g_BN_d, self.m_b_BN_d,
             self.v_dend, self.v_soma, self.v_readout, self.v_rho,
             self.v_g_BN_s, self.v_b_BN_s, self.v_g_BN_d, self.v_b_BN_d) = result
        else:
            (self.hidden.w_dend, self.hidden.w_soma, self.readout.w, self.rho_h,
             self.gamma_BN_s, self.beta_BN_s, self.gamma_BN_d, self.beta_BN_d) = _apply(
                *self._weights(), self.rho_h,
                self.gamma_BN_s, self.beta_BN_s, self.gamma_BN_d, self.beta_BN_d,
                g_d, g_s, g_r, g_rho, g_g_BN_s, g_b_BN_s, g_g_BN_d, g_b_BN_d,
                lr, lr_gamma, lr_bn, clip_value, self.weight_decay,
            )

    # ── Single-sample API ──

    def _next_key(self):
        """Advance the PRNG and return a fresh subkey for dropout."""
        self.rng_key, subkey = random.split(self.rng_key)
        return subkey

    # ── Single-sample API ──

    def train_step(self, x_input, target, lr=1e-3, clip_value=1.0):
        """Train on one sample (with dropout during forward pass).
        Returns: (loss, prediction, grad_norms_dict)
        """
        T = x_input.shape[0]

        (soma_BN, dend_BN, z_norm_s, z_norm_d,
         var_used_s, var_used_d) = self._project_and_bn(x_input, training=True)
        bn_scale_s = self._bn_scale(self.gamma_BN_s, var_used_s)
        bn_scale_d = self._bn_scale(self.gamma_BN_d, var_used_d)

        (counts, A_r, A_s, A_d, A_gamma,
         A_g_BN_s, A_b_BN_s, A_g_BN_d, A_b_BN_d) = _fwd_single(
            x_input, dend_BN, soma_BN, z_norm_d, z_norm_s,
            self.readout.w, *self._params(),
            self._h_carry(), self._r_carry(),
            self._A_d_zeros(), self._A_BN_d_zeros(), self._A_BN_d_zeros(),
            self._next_key(), self.dropout_rate,
        )

        (loss, pred, g_r, g_s, g_d, g_gamma,
         g_g_BN_s, g_b_BN_s, g_g_BN_d, g_b_BN_d) = _loss_single(
            counts, A_r, A_s, A_d, A_gamma,
            A_g_BN_s, A_b_BN_s, A_g_BN_d, A_b_BN_d,
            bn_scale_s, bn_scale_d,
            self._smooth_targets(target), T,
            self.config.loss_temperature, self.config.loss_count_bias,
        )

        gnorms = {
            "readout": float(jnp.linalg.norm(g_r)),
            "soma": float(jnp.linalg.norm(g_s)),
            "dend": float(jnp.linalg.norm(g_d)),
            "gamma_h": float(jnp.linalg.norm(g_gamma)),
            "gamma_BN_s": float(jnp.linalg.norm(g_g_BN_s)),
            "beta_BN_s": float(jnp.linalg.norm(g_b_BN_s)),
            "gamma_BN_d": float(jnp.linalg.norm(g_g_BN_d)),
            "beta_BN_d": float(jnp.linalg.norm(g_b_BN_d)),
        }

        self._update_weights(g_d, g_s, g_r, g_gamma,
                             g_g_BN_s, g_b_BN_s, g_g_BN_d, g_b_BN_d,
                             lr, clip_value)
        return float(loss), int(pred), gnorms

    def predict(self, x_input):
        """Predict one sample (no dropout, BN with running stats)."""
        soma_BN, dend_BN, _, _, _, _ = self._project_and_bn(x_input, training=False)
        counts = _pred_single(x_input, dend_BN, soma_BN, self.readout.w, *self._params())
        return int(jnp.argmax(counts))

    # ── Batched API ──

    def batch_train_step(self, x_batch, targets, lr=1e-3, clip_value=1.0):
        """Train on B samples in parallel (with dropout)."""
        B = x_batch.shape[0]
        T = x_batch.shape[1]

        batch_keys = random.split(self._next_key(), B)

        # BN forward (across (B, T) per neuron) is computed *outside* the
        # vmap so batch stats are well-defined. The vmap then sees per-sample
        # (T, N) BN'd inputs.
        (soma_BN, dend_BN, z_norm_s, z_norm_d,
         var_used_s, var_used_d) = self._project_and_bn(x_batch, training=True)
        bn_scale_s = self._bn_scale(self.gamma_BN_s, var_used_s)
        bn_scale_d = self._bn_scale(self.gamma_BN_d, var_used_d)

        (counts, A_r, A_s, A_d, A_gamma,
         A_g_BN_s, A_b_BN_s, A_g_BN_d, A_b_BN_d) = _fwd_batch(
            x_batch, dend_BN, soma_BN, z_norm_d, z_norm_s,
            self.readout.w, *self._params(),
            self._h_carry(B), self._r_carry(B),
            self._A_d_zeros(B), self._A_BN_d_zeros(B), self._A_BN_d_zeros(B),
            batch_keys, self.dropout_rate,
        )

        (losses, preds, g_r, g_s, g_d, g_gamma,
         g_g_BN_s, g_b_BN_s, g_g_BN_d, g_b_BN_d) = _loss_batch(
            counts, A_r, A_s, A_d, A_gamma,
            A_g_BN_s, A_b_BN_s, A_g_BN_d, A_b_BN_d,
            bn_scale_s, bn_scale_d,
            self._smooth_targets(targets), T,
            self.config.loss_temperature, self.config.loss_count_bias,
        )

        g_r_avg = jnp.mean(g_r, axis=0)
        g_s_avg = jnp.mean(g_s, axis=0)
        g_d_avg = jnp.mean(g_d, axis=0)
        g_gamma_avg = jnp.mean(g_gamma, axis=0)
        g_g_BN_s_avg = jnp.mean(g_g_BN_s, axis=0)
        g_b_BN_s_avg = jnp.mean(g_b_BN_s, axis=0)
        g_g_BN_d_avg = jnp.mean(g_g_BN_d, axis=0)
        g_b_BN_d_avg = jnp.mean(g_b_BN_d, axis=0)

        gnorms = {
            "readout": float(jnp.linalg.norm(g_r_avg)),
            "soma": float(jnp.linalg.norm(g_s_avg)),
            "dend": float(jnp.linalg.norm(g_d_avg)),
            "gamma_h": float(jnp.linalg.norm(g_gamma_avg)),
            "gamma_BN_s": float(jnp.linalg.norm(g_g_BN_s_avg)),
            "beta_BN_s": float(jnp.linalg.norm(g_b_BN_s_avg)),
            "gamma_BN_d": float(jnp.linalg.norm(g_g_BN_d_avg)),
            "beta_BN_d": float(jnp.linalg.norm(g_b_BN_d_avg)),
        }

        self._update_weights(g_d_avg, g_s_avg, g_r_avg, g_gamma_avg,
                             g_g_BN_s_avg, g_b_BN_s_avg, g_g_BN_d_avg, g_b_BN_d_avg,
                             lr, clip_value)
        return float(jnp.mean(losses)), preds, gnorms

    def batch_predict(self, x_batch):
        """Predict B samples in parallel. x_batch: (B,T,K) → (B,) int labels."""
        soma_BN, dend_BN, _, _, _, _ = self._project_and_bn(x_batch, training=False)
        counts = _pred_batch(
            x_batch, dend_BN, soma_BN, self.readout.w, *self._params(),
        )
        return jnp.argmax(counts, axis=1)
