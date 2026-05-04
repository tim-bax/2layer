"""
Feedforward multi-layer LIF with LIF readout. E-prop uses the soma (readout) path only:
sigma' on (v_pre - v_th), eligibility E = alpha_m E + pre_spikes (same alpha_m as membrane).

Performance: one lax.scan per layer (eligibility + membrane fused), jitted forward+grad,
and vmap over the batch (no Python loop per sample).
"""
from __future__ import annotations

import os
import sys
from functools import partial
from typing import List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from jax import lax, random
from jax import jit, vmap

from config import NeuronConfig, surrogate_sigma
from lif_neuron import LIFNeuron


# ── one scan: eligibility + LIF (halves kernel launches vs separate scans) ───


def lif_eligibility_and_forward(
    x_in: jnp.ndarray,
    w: jnp.ndarray,
    alpha_m: jnp.ndarray,
    v_th,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns (E_series (T, n_in), v_pre (T, n_out), spikes (T, n_out)).
    """
    n_in = x_in.shape[1]
    n_out = w.shape[0]

    def step(carry, x_t):
        e_prev, v_prev = carry
        e_new = alpha_m * e_prev + x_t
        inj = jnp.dot(x_t, w.T)
        v = alpha_m * v_prev + inj
        spikes = jnp.where(v >= v_th, 1.0, 0.0)
        v_pre = v
        v_new = v * (1.0 - spikes)
        return (e_new, v_new), (e_new, v_pre, spikes)

    init = (
        jnp.zeros(n_in, dtype=jnp.float64),
        jnp.zeros(n_out, dtype=jnp.float64),
    )
    _, (e_series, v_pre, spikes) = lax.scan(step, init, x_in.astype(jnp.float64))
    return e_series, v_pre, spikes


def lif_forward_only(
    x_in: jnp.ndarray,
    w: jnp.ndarray,
    alpha_m: jnp.ndarray,
    v_th,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Inference: only v_pre and spikes (one scan per layer)."""
    n_out = w.shape[0]

    def step(v_prev, x_t):
        inj = jnp.dot(x_t, w.T)
        v = alpha_m * v_prev + inj
        spikes = jnp.where(v >= v_th, 1.0, 0.0)
        v_pre = v
        v_new = v * (1.0 - spikes)
        return v_new, (v_pre, spikes)

    init_v = jnp.zeros(n_out, dtype=jnp.float64)
    _, (v_pre, spikes) = lax.scan(step, init_v, x_in.astype(jnp.float64))
    return v_pre, spikes


def softmax_loss_from_counts(
    readout_counts: jnp.ndarray,
    target: int,
    n_outputs: int,
    loss_temperature: float,
    loss_count_bias: float,
    loss_label_smoothing: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    scaled_logits = readout_counts / loss_temperature + loss_count_bias
    probs = jnp.exp(scaled_logits - jnp.max(scaled_logits))
    probs = probs / jnp.sum(probs)
    prediction = jnp.argmax(readout_counts)
    t_sm = jnp.zeros(n_outputs).at[target].set(1.0)
    t_sm = t_sm * (1 - loss_label_smoothing) + loss_label_smoothing / n_outputs
    loss = -jnp.sum(t_sm * jnp.log(probs + 1e-8))
    return loss, prediction, t_sm


def global_errors_from_counts(
    readout_counts: jnp.ndarray,
    target: int,
    n_outputs: int,
    loss_temperature: float,
    loss_count_bias: float,
    loss_label_smoothing: float,
) -> jnp.ndarray:
    scaled_logits = readout_counts / loss_temperature + loss_count_bias
    probs = jnp.exp(scaled_logits - jnp.max(scaled_logits))
    probs = probs / jnp.sum(probs)
    t_sm = jnp.zeros(n_outputs).at[target].set(1.0)
    t_sm = t_sm * (1 - loss_label_smoothing) + loss_label_smoothing / n_outputs
    return t_sm - probs


# ── jitted forward + grad (static H / dropout branch) ────────────────────────


def _forward_and_grad_impl(
    x_input: jnp.ndarray,
    target: jnp.ndarray,
    rng_key: jnp.ndarray,
    w_hidden: Tuple[jnp.ndarray, ...],
    w_readout: jnp.ndarray,
    alpha_m: jnp.ndarray,
    clip_value,
    *,
    H: int,
    n_outputs: int,
    use_dropout: bool,
    v_th,
    beta_s,
    loss_temperature,
    loss_count_bias,
    loss_label_smoothing,
    dropout_p,
):
    """Pure JAX: one fused scan per layer + readout; then e-prop grads."""
    scale = jnp.asarray(1.0 / (1.0 - dropout_p), jnp.float64) if use_dropout else jnp.asarray(1.0, jnp.float64)
    keys = random.split(rng_key, H + 1)

    inps = x_input.astype(jnp.float64)
    E_layers = []
    v_pres_h = []

    for ell in range(H):
        E_ell, v_pre, o = lif_eligibility_and_forward(inps, w_hidden[ell], alpha_m, v_th)
        E_layers.append(E_ell)
        v_pres_h.append(v_pre)
        if use_dropout:
            mask = random.bernoulli(keys[ell], 1.0 - dropout_p, o.shape).astype(jnp.float64)
            o_d = o.astype(jnp.float64) * mask * scale
        else:
            o_d = o.astype(jnp.float64)
        inps = o_d

    E_readout, v_pre_r, o_r = lif_eligibility_and_forward(inps, w_readout, alpha_m, v_th)

    T = jnp.asarray(o_r.shape[0], dtype=jnp.float64)
    readout_counts = jnp.sum(o_r, axis=0)

    ge = global_errors_from_counts(
        readout_counts,
        target,
        n_outputs,
        loss_temperature,
        loss_count_bias,
        loss_label_smoothing,
    )

    sig_r = surrogate_sigma(v_pre_r - v_th, beta_s)
    grad_readout = jnp.einsum("j,tj,ti->ji", ge, sig_r, E_readout) / T
    grad_readout = jnp.clip(grad_readout, -clip_value, clip_value)

    loss, pred, _ = softmax_loss_from_counts(
        readout_counts,
        target,
        n_outputs,
        loss_temperature,
        loss_count_bias,
        loss_label_smoothing,
    )

    effective = jnp.einsum("tj,j,ji->ti", sig_r, ge, w_readout)
    grad_hidden_list = []

    for ell in range(H - 1, -1, -1):
        sig_h = surrogate_sigma(v_pres_h[ell] - v_th, beta_s)
        g_ell = jnp.einsum("ti,ti,tk->ik", sig_h, effective, E_layers[ell]) / T
        grad_hidden_list.append(jnp.clip(g_ell, -clip_value, clip_value))
        if ell > 0:
            effective = jnp.einsum("ik,ti,ti->tk", w_hidden[ell], sig_h, effective)

    grad_hidden = tuple(reversed(grad_hidden_list))
    return loss, pred, grad_hidden, grad_readout


def _forward_eval_impl(
    x_input: jnp.ndarray,
    w_hidden: Tuple[jnp.ndarray, ...],
    w_readout: jnp.ndarray,
    alpha_m: jnp.ndarray,
    v_th,
    *,
    H: int,
):
    inp = x_input.astype(jnp.float64)
    for ell in range(H):
        _, o = lif_forward_only(inp, w_hidden[ell], alpha_m, v_th)
        inp = o.astype(jnp.float64)
    _, o_r = lif_forward_only(inp, w_readout, alpha_m, v_th)
    return jnp.sum(o_r, axis=0).astype(jnp.float64)


class Network:
    """
    x -> [LIF x H] -> LIF readout. Hidden sizes from `hidden_sizes`; readout has n_outputs.
    """

    def __init__(
        self,
        key: jnp.ndarray,
        n_inputs: int,
        hidden_sizes: List[int],
        n_outputs: int,
        config: NeuronConfig,
        optimizer: str = "sgd",
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
        dropout_hidden: float = 0.0,
        weight_decay: float = 0.0,
    ):
        if len(hidden_sizes) < 1:
            raise ValueError("Need at least one hidden layer (--hidden).")
        self.n_inputs = n_inputs
        self.hidden_sizes = list(hidden_sizes)
        self.H = len(hidden_sizes)
        self.n_outputs = n_outputs
        self.config = config
        self.optimizer = optimizer
        self.dropout_hidden = dropout_hidden
        self.weight_decay = weight_decay
        self.use_dropout = dropout_hidden > 0.0

        self.alpha_m = jnp.exp(-config.dt / config.tau_m)

        key_w, self.rng_key = random.split(key)
        keys = random.split(key_w, self.H + 1)
        self.w_hidden: List[jnp.ndarray] = []
        prev = n_inputs
        for ell in range(self.H):
            n_ell = hidden_sizes[ell]
            self.w_hidden.append(LIFNeuron(keys[ell], n_ell, prev, config).w)
            prev = n_ell
        self.w_readout = LIFNeuron(keys[self.H], n_outputs, prev, config).w

        cfg = config
        _train_fn = partial(
            _forward_and_grad_impl,
            H=self.H,
            n_outputs=n_outputs,
            use_dropout=self.use_dropout,
            v_th=jnp.asarray(cfg.v_th),
            beta_s=jnp.asarray(cfg.beta_s),
            loss_temperature=jnp.asarray(cfg.loss_temperature),
            loss_count_bias=jnp.asarray(cfg.loss_count_bias),
            loss_label_smoothing=jnp.asarray(cfg.loss_label_smoothing),
            dropout_p=jnp.asarray(self.dropout_hidden, dtype=jnp.float64),
        )
        self._jit_train = jit(_train_fn)

        _pred_fn = partial(
            _forward_eval_impl,
            H=self.H,
            v_th=jnp.asarray(cfg.v_th),
        )
        self._jit_predict = jit(_pred_fn)

        self._batched_train = jit(vmap(_train_fn, in_axes=(0, 0, 0, None, None, None, None)))

        self._batched_predict = jit(vmap(_pred_fn, in_axes=(0, None, None, None)))

        if optimizer == "adam":
            self.beta1 = beta1
            self.beta2 = beta2
            self.adam_eps = adam_eps
            self.adam_step = jnp.array(0, dtype=jnp.int32)
            self.m_readout = jnp.zeros_like(self.w_readout)
            self.v_readout = jnp.zeros_like(self.w_readout)
            self.m_hidden = [jnp.zeros_like(w) for w in self.w_hidden]
            self.v_hidden = [jnp.zeros_like(w) for w in self.w_hidden]

    def _weights_tuple(self) -> Tuple[jnp.ndarray, ...]:
        return tuple(self.w_hidden)

    def _next_key(self):
        self.rng_key, sub = random.split(self.rng_key)
        return sub

    def _update_sgd(self, grad_hidden: Tuple, grad_readout, lr: float, clip_value: float):
        wd = self.weight_decay
        self.w_readout = self.w_readout + lr * grad_readout - lr * wd * self.w_readout
        for ell in range(self.H):
            self.w_hidden[ell] = (
                self.w_hidden[ell] + lr * grad_hidden[ell] - lr * wd * self.w_hidden[ell]
            )

    def _update_adam(self, grad_hidden: Tuple, grad_readout, lr: float, clip_value: float):
        self.adam_step = self.adam_step + 1
        t = self.adam_step.astype(jnp.float32)
        b1, b2, eps = self.beta1, self.beta2, self.adam_eps

        def one(w, g, m, v):
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g ** 2
            m_hat = m / (1 - b1 ** t)
            v_hat = v / (1 - b2 ** t)
            w = w + lr * m_hat / (jnp.sqrt(v_hat) + eps) - lr * self.weight_decay * w
            return w, m, v

        self.w_readout, self.m_readout, self.v_readout = one(
            self.w_readout, grad_readout, self.m_readout, self.v_readout
        )
        for ell in range(self.H):
            self.w_hidden[ell], self.m_hidden[ell], self.v_hidden[ell] = one(
                self.w_hidden[ell], grad_hidden[ell], self.m_hidden[ell], self.v_hidden[ell]
            )

    def _apply_grads(self, grad_hidden, grad_readout, lr, clip_value):
        if self.optimizer == "adam":
            self._update_adam(grad_hidden, grad_readout, lr, clip_value)
        else:
            self._update_sgd(grad_hidden, grad_readout, lr, clip_value)

    def train_step(self, x_input, target, lr=1e-3, clip_value=5.0):
        clip = jnp.asarray(clip_value, dtype=jnp.float64)
        tgt = jnp.asarray(int(target), dtype=jnp.int32)
        loss, pred, g_h, g_r = self._jit_train(
            x_input,
            tgt,
            self._next_key(),
            self._weights_tuple(),
            self.w_readout,
            self.alpha_m,
            clip,
        )
        self._apply_grads(g_h, g_r, lr, float(clip_value))
        gnorms = {"readout": float(jnp.linalg.norm(g_r))}
        for ell in range(self.H):
            gnorms[f"hidden_{ell}"] = float(jnp.linalg.norm(g_h[ell]))
        return float(loss), int(pred), gnorms

    def predict(self, x_input):
        counts = self._jit_predict(
            x_input,
            self._weights_tuple(),
            self.w_readout,
            self.alpha_m,
        )
        return int(jnp.argmax(counts))

    def batch_train_step(self, x_batch, targets, lr=1e-3, clip_value=5.0):
        B = x_batch.shape[0]
        keys = random.split(self._next_key(), B)
        clip = jnp.asarray(clip_value, dtype=jnp.float64)
        tgts = jnp.asarray(targets, dtype=jnp.int32)

        losses, preds, g_h_batched, g_r_batched = self._batched_train(
            x_batch,
            tgts,
            keys,
            self._weights_tuple(),
            self.w_readout,
            self.alpha_m,
            clip,
        )

        g_h_avg = tuple(jnp.mean(g_h_batched[ell], axis=0) for ell in range(self.H))
        g_r_avg = jnp.mean(g_r_batched, axis=0)
        self._apply_grads(g_h_avg, g_r_avg, lr, clip_value)

        gnorms = {"readout": float(jnp.linalg.norm(g_r_avg))}
        for ell in range(self.H):
            gnorms[f"hidden_{ell}"] = float(jnp.linalg.norm(g_h_avg[ell]))

        return (
            float(jnp.mean(losses)),
            preds,
            gnorms,
        )

    def batch_predict(self, x_batch):
        counts = self._batched_predict(
            x_batch,
            self._weights_tuple(),
            self.w_readout,
            self.alpha_m,
        )
        return jnp.argmax(counts, axis=1)


def parse_hidden_sizes(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --hidden")
    return [int(x) for x in parts]
