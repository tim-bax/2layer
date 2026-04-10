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

def _forward_and_accum(
    x_input, w_dend, w_soma, w_readout,
    alpha_s, alpha_d, alpha_m, T_p, config,
    h_carry_init, r_carry_init, A_d_init,
):
    """Forward pass + gradient accumulator bookkeeping for one sample.

    x_input:      (T, K)  input spike train
    h_carry_init: 8-tuple of hidden-neuron state zeros
    r_carry_init: 3-tuple of readout-neuron state zeros
    A_d_init:     (J, N, K) dendritic accumulator zeros

    Returns: readout_counts (J,), A_readout (J,N), A_soma (J,N,K), A_dend (J,N,K)
    """
    dend_inputs = x_input @ w_dend.T
    soma_inputs = x_input @ w_soma.T
    T = x_input.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)

    def step(carry, inputs):
        h_carry, r_carry, A_d = carry
        dend_in, soma_in, x_t, t = inputs

        h_carry, h_o, h_v_pre, h_h, h_h_prev, h_mu_at_tp = TwoCompNeuron.forward_step(
            h_carry, dend_in, soma_in, t, alpha_s, alpha_d, T_p, config,
        )
        hidden_o_float = h_o.astype(jnp.float64)

        r_carry, r_o, r_v_pre, r_E = LIFNeuron.forward_step(
            r_carry, hidden_o_float, w_readout, alpha_m, config.v_th,
        )

        mu_c, v_c, h_c, tp_c, matp_c, E_soma_c, dmu_c, dmu_atp_c = h_carry
        E_soma_new = TwoCompNeuron.update_somatic_eligibility(
            E_soma_c, x_t.astype(jnp.float64), alpha_s,
        )
        dmu_new, dmu_atp_new = TwoCompNeuron.update_dendritic_eligibility(
            dmu_c, dmu_atp_c, x_t.astype(jnp.float64), h_h_prev, alpha_d,
        )
        h_carry = (mu_c, v_c, h_c, tp_c, matp_c, E_soma_new, dmu_new, dmu_atp_new)

        sp_readout = surrogate_sigma(r_v_pre - config.v_th, config.beta_s)
        sp_hidden = surrogate_sigma(
            h_v_pre + config.gamma * h_h - config.v_th, config.beta_s,
        )
        hp_hidden = surrogate_sigma(h_mu_at_tp - config.mu_th, config.beta_d)

        eta = sp_readout[:, None] * w_readout * sp_hidden[None, :]
        eta_d = eta * (hp_hidden * config.gamma)[None, :]
        A_d = A_d + jnp.einsum("ji,ik->jik", eta_d, dmu_atp_new)

        new_carry = (h_carry, r_carry, A_d)
        per_step = (sp_readout, sp_hidden, r_E, E_soma_new)
        return new_carry, per_step

    init_carry = (h_carry_init, r_carry_init, A_d_init)
    scan_inputs = (dend_inputs, soma_inputs, x_input, time_indices)
    final_carry, per_step_all = lax.scan(step, init_carry, scan_inputs)

    sp_r, sp_h, E_r, E_s = per_step_all
    _, r_carry_f, A_d_f = final_carry
    readout_counts = r_carry_f[1]

    A_readout = jnp.einsum("ti,tj->ij", sp_r, E_r)
    C_soma = jnp.einsum("tj,ti,tk->jik", sp_r, sp_h, E_s)
    A_soma = w_readout[:, :, None] * C_soma

    return readout_counts, A_readout, A_soma, A_d_f


def _predict_only(
    x_input, w_dend, w_soma, w_readout,
    alpha_s, alpha_d, alpha_m, T_p, config,
):
    """Forward pass only — no gradient bookkeeping. Returns readout_counts (J,)."""
    dend_inputs = x_input @ w_dend.T
    soma_inputs = x_input @ w_soma.T
    T = x_input.shape[0]
    n_hidden = w_dend.shape[0]
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
        o_h = jnp.where(v_pre >= config.v_th - config.gamma * h_new, 1, 0).astype(jnp.int32)
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
    readout_counts, A_readout, A_soma, A_dend,
    target_smoothed, T, loss_temperature, loss_count_bias,
):
    """Compute loss and weight gradients for one sample."""
    scaled_logits = readout_counts / loss_temperature + loss_count_bias
    probs = jnp.exp(scaled_logits - jnp.max(scaled_logits))
    probs = probs / jnp.sum(probs)

    prediction = jnp.argmax(readout_counts)
    loss = -jnp.sum(target_smoothed * jnp.log(probs + 1e-8))
    global_error = target_smoothed - probs

    grad_readout = (global_error[:, None] * A_readout) / T
    grad_soma = jnp.einsum("j,jik->ik", global_error, A_soma) / T
    grad_dend = jnp.einsum("j,jik->ik", global_error, A_dend) / T

    return loss, prediction, grad_readout, grad_soma, grad_dend


def _apply_grads(w_dend, w_soma, w_readout, g_dend, g_soma, g_readout, lr, clip_value):
    """Clip gradients and update weights."""
    g_readout = jnp.clip(g_readout, -clip_value, clip_value)
    g_soma = jnp.clip(g_soma, -clip_value, clip_value)
    g_dend = jnp.clip(g_dend, -clip_value, clip_value)
    return (
        w_dend + lr * g_dend,
        w_soma + lr * g_soma,
        w_readout + lr * g_readout,
    )


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
    0,                            # A_d_init
)

_PRED_AXES = (
    0,                            # x_input
    None, None, None,             # weights
    None, None, None, None, None, # alphas, T_p, config
)

_LOSS_AXES = (
    0, 0, 0, 0,                  # counts, A_r, A_s, A_d (per-sample)
    0,                            # target_smoothed (per-sample)
    None, None, None,             # T, loss_temperature, loss_count_bias
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
    ):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.config = config

        key_h, key_r = random.split(key)
        self.hidden = TwoCompNeuron(key_h, n_hidden, n_inputs, config)
        self.readout = LIFNeuron(key_r, n_outputs, n_hidden, config)

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
        """Dendritic accumulator zeros."""
        base = (self.n_outputs, self.n_hidden, self.n_inputs)
        return jnp.zeros((B,) + base if B else base)

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

    # ── Single-sample API ──

    def train_step(self, x_input, target, lr=1e-3, clip_value=1.0):
        """Train on one sample. x_input: (T,K), target: int."""
        T = x_input.shape[0]

        counts, A_r, A_s, A_d = _fwd_single(
            x_input, *self._weights(), *self._params(),
            self._h_carry(), self._r_carry(), self._A_d_zeros(),
        )

        loss, pred, g_r, g_s, g_d = _loss_single(
            counts, A_r, A_s, A_d,
            self._smooth_targets(target), T,
            self.config.loss_temperature, self.config.loss_count_bias,
        )

        self.hidden.w_dend, self.hidden.w_soma, self.readout.w = _apply(
            *self._weights(), g_d, g_s, g_r, lr, clip_value,
        )
        return float(loss), int(pred)

    def predict(self, x_input):
        """Predict one sample. x_input: (T,K) → int class label."""
        counts = _pred_single(x_input, *self._weights(), *self._params())
        return int(jnp.argmax(counts))

    # ── Batched API ──
    #
    # vmap runs the SAME single-sample logic on B samples in parallel.
    # At each of the T=700 time steps, all B samples are processed
    # simultaneously — the GPU fills its cores with B parallel computations
    # instead of idling on one.

    def batch_train_step(self, x_batch, targets, lr=1e-3, clip_value=1.0):
        """Train on B samples in parallel.

        x_batch: (B, T, K)  — B input spike trains stacked
        targets: (B,)       — integer labels

        Returns: (mean_loss, predictions_array (B,))
        """
        B = x_batch.shape[0]
        T = x_batch.shape[1]

        # Step 1: forward pass — B copies of the lax.scan run in lockstep
        #   Every (N,) vector becomes (B,N), every (J,N,K) tensor becomes (B,J,N,K)
        counts, A_r, A_s, A_d = _fwd_batch(
            x_batch, *self._weights(), *self._params(),
            self._h_carry(B), self._r_carry(B), self._A_d_zeros(B),
        )

        # Step 2: per-sample loss + gradients (also vmapped)
        #   Each sample gets its own global_error and gradient matrices
        losses, preds, g_r, g_s, g_d = _loss_batch(
            counts, A_r, A_s, A_d,
            self._smooth_targets(targets), T,
            self.config.loss_temperature, self.config.loss_count_bias,
        )

        # Step 3: average gradients across the batch → one update
        #   g_r is (B, J, N) → mean → (J, N), same for g_s and g_d
        g_r_avg = jnp.mean(g_r, axis=0)
        g_s_avg = jnp.mean(g_s, axis=0)
        g_d_avg = jnp.mean(g_d, axis=0)

        # Step 4: apply averaged gradients to weights (single update)
        self.hidden.w_dend, self.hidden.w_soma, self.readout.w = _apply(
            *self._weights(), g_d_avg, g_s_avg, g_r_avg, lr, clip_value,
        )
        return float(jnp.mean(losses)), preds

    def batch_predict(self, x_batch):
        """Predict B samples in parallel. x_batch: (B,T,K) → (B,) int labels."""
        counts = _pred_batch(x_batch, *self._weights(), *self._params())
        return jnp.argmax(counts, axis=1)
