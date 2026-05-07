import jax.numpy as jnp
from jax import random, jit, lax, vmap

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

def _sliding_argmax_mu(mu, T_p):
    """For each timestep t and hidden neuron i, index of the maximum μ in the window.

    g[t, i] = argmax_{s ∈ [max(0, t − T_p[i] + 1), t]} μ[s, i].

    Used only in the backward (dendritic credit heuristic); forward dynamics unchanged.
    """
    T = mu.shape[0]

    def argmax_up_to_t(t_cur):
        lo = jnp.maximum(0, t_cur - T_p + 1)
        rows = jnp.arange(T, dtype=jnp.int32)[:, None]
        mask = (rows >= lo[None, :]) & (rows <= t_cur)
        neg = jnp.array(-jnp.finfo(mu.dtype).max, dtype=mu.dtype)
        masked = jnp.where(mask, mu, neg)
        return jnp.argmax(masked, axis=0)

    return vmap(argmax_up_to_t)(jnp.arange(T, dtype=jnp.int32))


def _forward_and_accum(
    x_input, w_dend, w_soma, w_readout,
    alpha_s, alpha_d, alpha_m, T_p, config,
    h_carry_init, r_carry_init,
    rng_key, dropout_rate,
):
    """Forward pass for one sample; stack trajectories for max-over-time loss / e-prop.

    Readout: leaky integrate-and-fire *without* thresholding or reset (paper-style).

    Returns stacked (time-first):
      v_readout (T, J), mu_tr (T, N), dmu_tr (T, N, K), sp_hidden (T, N),
      E_soma_tr (T, K), E_readout_tr (T, N).
    """
    dend_inputs = x_input @ w_dend.T
    soma_inputs = x_input @ w_soma.T
    T = x_input.shape[0]
    n_hidden = w_dend.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)
    dropout_keys = random.split(rng_key, T)
    dropout_scale = 1.0 / (1.0 - dropout_rate)

    def step(carry, inputs):
        h_carry, r_carry = carry
        dend_in, soma_in, x_t, t, drop_key = inputs

        h_carry, h_o, h_v_pre, h_h, h_h_prev, h_mu_at_tp = TwoCompNeuron.forward_step(
            h_carry, dend_in, soma_in, t, alpha_s, alpha_d, T_p, config,
        )
        hidden_o_float = h_o.astype(jnp.float64)

        mask = random.bernoulli(drop_key, 1.0 - dropout_rate, (n_hidden,)).astype(jnp.float64)
        hidden_o_float = hidden_o_float * mask * dropout_scale

        r_carry, r_v, r_E = LIFNeuron.forward_step_integrate_only(
            r_carry, hidden_o_float, w_readout, alpha_m,
        )

        mu_c, v_c, h_c, tp_c, matp_c, E_soma_c, dmu_c, dmu_atp_c = h_carry
        E_soma_new = TwoCompNeuron.update_somatic_eligibility(
            E_soma_c, x_t.astype(jnp.float64), alpha_s,
        )
        dmu_new, dmu_atp_new = TwoCompNeuron.update_dendritic_eligibility(
            dmu_c, dmu_atp_c, x_t.astype(jnp.float64), h_h_prev, alpha_d,
        )
        h_carry = (mu_c, v_c, h_c, tp_c, matp_c, E_soma_new, dmu_new, dmu_atp_new)

        mu_out = mu_c
        sp_hidden = surrogate_sigma(
            h_v_pre + config.gamma * h_h - config.v_th, config.beta_s,
        )

        new_carry = (h_carry, r_carry)
        per_step = (r_v, sp_hidden, E_soma_new, mu_out, dmu_new, r_E)
        return new_carry, per_step

    init_carry = (h_carry_init, r_carry_init)
    scan_inputs = (dend_inputs, soma_inputs, x_input, time_indices, dropout_keys)
    _, per_step_all = lax.scan(step, init_carry, scan_inputs)

    v_readout, sp_hidden, E_soma_tr, mu_tr, dmu_tr, E_readout_tr = per_step_all
    return v_readout, mu_tr, dmu_tr, sp_hidden, E_soma_tr, E_readout_tr


def _predict_only(
    x_input, w_dend, w_soma, w_readout,
    alpha_s, alpha_d, alpha_m, T_p, config,
):
    """Forward only: max-over-time readout voltage per class → logits; returns class index."""
    dend_inputs = x_input @ w_dend.T
    soma_inputs = x_input @ w_soma.T
    T = x_input.shape[0]
    n_hidden = w_dend.shape[0]
    n_outputs = w_readout.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)
    dtype = jnp.float64

    def step(carry, inputs):
        h_carry, r_carry = carry
        dend_in, soma_in, t = inputs

        h_carry, h_o, h_v_pre, h_h, h_h_prev, h_mu_at_tp = TwoCompNeuron.forward_step(
            h_carry, dend_in, soma_in, t, alpha_s, alpha_d, T_p, config,
        )
        r_carry, r_v, _ = LIFNeuron.forward_step_integrate_only(
            r_carry, h_o.astype(dtype), w_readout, alpha_m,
        )
        new_carry = (h_carry, r_carry)
        return new_carry, r_v

    n_inputs = x_input.shape[1]
    h_carry_init = (
        jnp.zeros(n_hidden), jnp.zeros(n_hidden),
        jnp.zeros(n_hidden, dtype=jnp.int32), jnp.zeros(n_hidden, dtype=jnp.int32),
        jnp.zeros(n_hidden), jnp.zeros(n_inputs),
        jnp.zeros((n_hidden, n_inputs)), jnp.zeros((n_hidden, n_inputs)),
    )
    r_carry_init = (
        jnp.zeros(n_outputs),
        jnp.zeros(n_outputs),
        jnp.zeros(n_hidden),
    )
    init_carry = (h_carry_init, r_carry_init)
    _, v_traj = lax.scan(step, init_carry, (dend_inputs, soma_inputs, time_indices))
    t_star = jnp.argmax(v_traj, axis=0)
    j_idx = jnp.arange(n_outputs)
    v_max = v_traj[t_star, j_idx]
    return jnp.argmax(v_max)


def _max_loss_and_grads(
    v_readout,
    mu_tr,
    dmu_tr,
    sp_hidden,
    E_soma_tr,
    E_readout_tr,
    w_readout,
    T_p,
    alpha_m,
    target_smoothed,
    loss_temperature,
    loss_logit_bias,
    config,
):
    """Cross-entropy on max readout voltages; gradients (readout exact, hidden e-prop)."""
    T = v_readout.shape[0]
    J = v_readout.shape[1]

    t_star = jnp.argmax(v_readout, axis=0)
    j_idx = jnp.arange(J)
    v_max = v_readout[t_star, j_idx]

    logits = v_max / loss_temperature + loss_logit_bias
    probs = jnp.exp(logits - jnp.max(logits))
    probs = probs / jnp.sum(probs)
    prediction = jnp.argmax(v_max)
    loss = -jnp.sum(target_smoothed * jnp.log(probs + 1e-8))
    global_error = target_smoothed - probs

    grad_readout = global_error[:, None] * E_readout_tr[t_star, :]

    t_grid = jnp.arange(T, dtype=jnp.int32)[:, None]
    t_star_row = t_star[None, :]
    M = (t_grid <= t_star_row).astype(v_readout.dtype) * (
        alpha_m ** jnp.maximum(0, t_star_row - t_grid)
    )
    phi = jnp.einsum("j,tj,ji->ti", global_error, M, w_readout)

    G = _sliding_argmax_mu(mu_tr, T_p)
    n_broadcast = jnp.broadcast_to(
        jnp.arange(mu_tr.shape[1], dtype=jnp.int32), G.shape,
    )
    mu_at_g = mu_tr[G, n_broadcast]
    hp_heur = surrogate_sigma(mu_at_g - config.mu_th, config.beta_d)
    dmu_pick = dmu_tr[G, n_broadcast, :]

    T_scale = jnp.maximum(
        jnp.array(1.0, dtype=phi.dtype), jnp.asarray(T, dtype=phi.dtype),
    )
    grad_soma = jnp.einsum("ti,ti,tk->ik", phi, sp_hidden, E_soma_tr) / T_scale
    grad_dend = (
        config.gamma
        * jnp.einsum("ti,ti,ti,tik->ik", phi, sp_hidden, hp_heur, dmu_pick)
        / T_scale
    )

    return loss, prediction, grad_readout, grad_soma, grad_dend


def _apply_grads(
    w_dend, w_soma, w_readout, g_dend, g_soma, g_readout,
    lr, clip_value, weight_decay,
):
    """SGD with decoupled weight decay.

    Gradients are clipped first, then weights are updated as
        w ← w + lr·g − lr·λ·w,
    which for plain SGD is mathematically equivalent to adding
    (λ/2)·||w||² to the loss (the L2 penalty), but keeps the
    weight-decay term outside the clip so it can't be capped by
    the gradient clip.
    """
    g_readout = jnp.clip(g_readout, -clip_value, clip_value)
    g_soma = jnp.clip(g_soma, -clip_value, clip_value)
    g_dend = jnp.clip(g_dend, -clip_value, clip_value)
    return (
        w_dend + lr * g_dend - lr * weight_decay * w_dend,
        w_soma + lr * g_soma - lr * weight_decay * w_soma,
        w_readout + lr * g_readout - lr * weight_decay * w_readout,
    )


def _adam_apply(
    w_d, w_s, w_r,
    g_d, g_s, g_r,
    m_d, m_s, m_r,
    v_d, v_s, v_r,
    step, lr, beta1, beta2, eps, clip_value, weight_decay,
):
    """AdamW-style decoupled weight decay.

    The data gradient flows through the moment estimates and the
    adaptive 1/√v rescaling, but the λ·w term does not — it is
    subtracted directly from w after the Adam update. This matches
    Loshchilov & Hutter (2017) and avoids the parameter-dependent
    decay strength that arises if λ·w is added to the loss before
    Adam normalisation.
    """
    def update_one(w, g, m, v):
        g = jnp.clip(g, -clip_value, clip_value)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        w = w + lr * m_hat / (jnp.sqrt(v_hat) + eps) - lr * weight_decay * w
        return w, m, v

    w_d, m_d, v_d = update_one(w_d, g_d, m_d, v_d)
    w_s, m_s, v_s = update_one(w_s, g_s, m_s, v_s)
    w_r, m_r, v_r = update_one(w_r, g_r, m_r, v_r)
    return (w_d, w_s, w_r, m_d, m_s, m_r, v_d, v_s, v_r)


# ══════════════════════════════════════════════════════════════════════
#  vmap in_axes: which args get a batch dimension (0) vs stay shared (None)
#
#  For _forward_and_accum:
#    x_input → batched (B,T,K)       init carries → batched (B,...)
#    weights → shared                 config/alphas → shared
# ══════════════════════════════════════════════════════════════════════

_FWD_AXES = (
    0,                            # x_input
    None, None, None,             # w_dend, w_soma, w_readout
    None, None, None, None, None, # alpha_s, alpha_d, alpha_m, T_p, config
    (0, 0, 0, 0, 0, 0, 0, 0),    # h_carry_init (8-tuple, each batched)
    (0, 0, 0),                    # r_carry_init (3-tuple, each batched)
    0,                            # rng_key (per-sample)
    None,                         # dropout_rate (shared)
)

_PRED_AXES = (
    0,                            # x_input
    None, None, None,             # weights
    None, None, None, None, None, # alphas, T_p, config
)

_LOSS_AXES = (
    0, 0, 0, 0, 0, 0,            # v_readout, mu, dmu, sp, E_soma, E_readout
    None,                         # w_readout
    None,                         # T_p (shared network)
    None,                         # alpha_m
    0,                            # target_smoothed
    None, None, None,            # loss_temperature, loss_logit_bias, config
)


# ══════════════════════════════════════════════════════════════════════
#  Pre-compiled versions:
#    _*_single = jit(core_fn)                  → one sample
#    _*_batch  = jit(vmap(core_fn, in_axes=…)) → B samples in parallel
# ══════════════════════════════════════════════════════════════════════

_fwd_single = jit(_forward_and_accum)
_pred_single = jit(_predict_only)
_loss_single = jit(_max_loss_and_grads)
_apply = jit(_apply_grads)
_adam = jit(_adam_apply)

_fwd_batch = jit(vmap(_forward_and_accum, in_axes=_FWD_AXES))
_pred_batch = jit(vmap(_predict_only, in_axes=_PRED_AXES))
_loss_batch = jit(vmap(_max_loss_and_grads, in_axes=_LOSS_AXES))


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
    ):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.config = config
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

        key_h, key_r, key_rng = random.split(key, 3)
        self.hidden = TwoCompNeuron(key_h, n_hidden, n_inputs, config)
        self.readout = LIFNeuron(key_r, n_outputs, n_hidden, config)
        self.rng_key = key_rng

        if optimizer == "adam":
            self.beta1 = beta1
            self.beta2 = beta2
            self.adam_eps = adam_eps
            self.adam_step = jnp.array(0, dtype=jnp.int32)
            self.m_dend = jnp.zeros_like(self.hidden.w_dend)
            self.m_soma = jnp.zeros_like(self.hidden.w_soma)
            self.m_readout = jnp.zeros_like(self.readout.w)
            self.v_dend = jnp.zeros_like(self.hidden.w_dend)
            self.v_soma = jnp.zeros_like(self.hidden.w_soma)
            self.v_readout = jnp.zeros_like(self.readout.w)

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

    def _weights(self):
        return self.hidden.w_dend, self.hidden.w_soma, self.readout.w

    def _params(self):
        return (self.hidden.alpha_s, self.hidden.alpha_d, self.readout.alpha_m,
                self.hidden.T_p, self.config)

    def _smooth_targets(self, targets):
        """Scalar label or (B,) labels → smoothed one-hot vector(s)."""
        cfg = self.config
        one_hot = jnp.eye(self.n_outputs)[targets]
        return one_hot * (1 - cfg.loss_label_smoothing) + cfg.loss_label_smoothing / self.n_outputs

    def _update_weights(self, g_d, g_s, g_r, lr, clip_value):
        """Apply gradients using the configured optimizer (SGD or Adam).

        Both branches use decoupled weight decay (subtract lr·λ·w
        after the gradient step). For SGD this is equivalent to an
        L2 loss penalty (λ/2)·||w||²; for Adam this is the AdamW
        recipe — the decay does not pass through the 1/√v rescaling.
        """
        if self.optimizer == "adam":
            self.adam_step = self.adam_step + 1
            result = _adam(
                self.hidden.w_dend, self.hidden.w_soma, self.readout.w,
                g_d, g_s, g_r,
                self.m_dend, self.m_soma, self.m_readout,
                self.v_dend, self.v_soma, self.v_readout,
                self.adam_step, lr, self.beta1, self.beta2, self.adam_eps,
                clip_value, self.weight_decay,
            )
            (self.hidden.w_dend, self.hidden.w_soma, self.readout.w,
             self.m_dend, self.m_soma, self.m_readout,
             self.v_dend, self.v_soma, self.v_readout) = result
        else:
            self.hidden.w_dend, self.hidden.w_soma, self.readout.w = _apply(
                *self._weights(), g_d, g_s, g_r, lr, clip_value, self.weight_decay,
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
        v_readout, mu_tr, dmu_tr, sp_hidden, E_soma_tr, E_readout_tr = _fwd_single(
            x_input, *self._weights(), *self._params(),
            self._h_carry(), self._r_carry(),
            self._next_key(), self.dropout_rate,
        )

        loss, pred, g_r, g_s, g_d = _loss_single(
            v_readout, mu_tr, dmu_tr, sp_hidden, E_soma_tr, E_readout_tr,
            self.readout.w, self.hidden.T_p, self.readout.alpha_m,
            self._smooth_targets(target),
            self.config.loss_temperature, self.config.loss_count_bias, self.config,
        )

        gnorms = {
            "readout": float(jnp.linalg.norm(g_r)),
            "soma": float(jnp.linalg.norm(g_s)),
            "dend": float(jnp.linalg.norm(g_d)),
        }

        self._update_weights(g_d, g_s, g_r, lr, clip_value)
        return float(loss), int(pred), gnorms

    def predict(self, x_input):
        """Predict one sample (no dropout). x_input: (T,K) → int class label."""
        return int(_pred_single(x_input, *self._weights(), *self._params()))

    # ── Batched API ──

    def batch_train_step(self, x_batch, targets, lr=1e-3, clip_value=1.0):
        """Train on B samples in parallel (with dropout).
        Returns: (mean_loss, predictions_array (B,), grad_norms_dict)
        """
        B = x_batch.shape[0]

        batch_keys = random.split(self._next_key(), B)

        v_readout, mu_tr, dmu_tr, sp_hidden, E_soma_tr, E_readout_tr = _fwd_batch(
            x_batch, *self._weights(), *self._params(),
            self._h_carry(B), self._r_carry(B),
            batch_keys, self.dropout_rate,
        )

        losses, preds, g_r, g_s, g_d = _loss_batch(
            v_readout, mu_tr, dmu_tr, sp_hidden, E_soma_tr, E_readout_tr,
            self.readout.w, self.hidden.T_p, self.readout.alpha_m,
            self._smooth_targets(targets),
            self.config.loss_temperature, self.config.loss_count_bias, self.config,
        )

        g_r_avg = jnp.mean(g_r, axis=0)
        g_s_avg = jnp.mean(g_s, axis=0)
        g_d_avg = jnp.mean(g_d, axis=0)

        gnorms = {
            "readout": float(jnp.linalg.norm(g_r_avg)),
            "soma": float(jnp.linalg.norm(g_s_avg)),
            "dend": float(jnp.linalg.norm(g_d_avg)),
        }

        self._update_weights(g_d_avg, g_s_avg, g_r_avg, lr, clip_value)
        return float(jnp.mean(losses)), preds, gnorms

    def batch_predict(self, x_batch):
        """Predict B samples in parallel. x_batch: (B,T,K) → (B,) int labels."""
        return _pred_batch(x_batch, *self._weights(), *self._params())
