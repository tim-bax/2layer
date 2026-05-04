"""
Feedforward multi-layer LIF with LIF readout. E-prop uses the soma (readout) path only:
sigma' on (v_pre - v_th), eligibility E = alpha_m E + pre_spikes (same alpha_m as membrane).
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import jax
import jax.numpy as jnp
from jax import lax, random

from config import NeuronConfig, surrogate_sigma
from lif_neuron import LIFNeuron


# ── scan helpers (materialize full (T, …) tensors) ───────────────────────────


def eligibility_trace(x_tf: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    """E[t] = alpha * E[t-1] + x[t]; x_tf (T, n_in)."""

    def step(e_prev, x_t):
        e = alpha * e_prev + x_t
        return e, e

    init = jnp.zeros(x_tf.shape[1], dtype=jnp.float64)
    _, e_series = lax.scan(step, init, x_tf.astype(jnp.float64))
    return e_series


def lif_layer_forward(
    x_in: jnp.ndarray,
    w: jnp.ndarray,
    alpha_m: jnp.ndarray,
    v_th: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns v_pre (T, n_out), spikes (T, n_out) float 0/1 (matches readout numerics)."""
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
    """Returns (loss, prediction, target_smoothed (n_outputs,))."""
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

        if optimizer == "adam":
            self.beta1 = beta1
            self.beta2 = beta2
            self.adam_eps = adam_eps
            self.adam_step = jnp.array(0, dtype=jnp.int32)
            self.m_readout = jnp.zeros_like(self.w_readout)
            self.v_readout = jnp.zeros_like(self.w_readout)
            self.m_hidden = [jnp.zeros_like(w) for w in self.w_hidden]
            self.v_hidden = [jnp.zeros_like(w) for w in self.w_hidden]

    def _next_key(self):
        self.rng_key, sub = random.split(self.rng_key)
        return sub

    def _smooth_target(self, target):
        cfg = self.config
        return jnp.zeros(self.n_outputs).at[target].set(1.0) * (
            1 - cfg.loss_label_smoothing
        ) + cfg.loss_label_smoothing / self.n_outputs

    def _smooth_targets_batch(self, targets):
        return jnp.array([self._smooth_target(int(t)) for t in targets])

    def _apply_dropout_to_spikes(self, o, key, scale):
        if self.dropout_hidden <= 0:
            return o.astype(jnp.float64)
        mask = random.bernoulli(key, 1.0 - self.dropout_hidden, o.shape).astype(jnp.float64)
        return o.astype(jnp.float64) * mask * scale

    def _forward_train(self, x_input: jnp.ndarray, rng_key: jnp.ndarray):
        """Returns readout counts, intermediates for grads, readout_o (T, n_out)."""
        cfg = self.config
        scale = 1.0 / (1.0 - self.dropout_hidden) if self.dropout_hidden > 0 else 1.0
        keys = random.split(rng_key, self.H + 1)

        inps: List[jnp.ndarray] = [x_input.astype(jnp.float64)]
        E_layers: List[jnp.ndarray] = []
        v_pres_h: List[jnp.ndarray] = []
        o_hs: List[jnp.ndarray] = []

        for ell in range(self.H):
            inp = inps[ell]
            E_layers.append(eligibility_trace(inp, self.alpha_m))
            v_pre, o = lif_layer_forward(inp, self.w_hidden[ell], self.alpha_m, cfg.v_th)
            v_pres_h.append(v_pre)
            o_hs.append(o)
            o_d = self._apply_dropout_to_spikes(o, keys[ell], scale)
            if ell < self.H - 1:
                inps.append(o_d)
            else:
                read_in = o_d

        E_readout = eligibility_trace(read_in, self.alpha_m)
        v_pre_r, o_r = lif_layer_forward(read_in, self.w_readout, self.alpha_m, cfg.v_th)
        readout_counts = jnp.sum(o_r, axis=0).astype(jnp.float64)
        return readout_counts, E_layers, v_pres_h, E_readout, v_pre_r, o_r

    def _forward_eval(self, x_input: jnp.ndarray):
        cfg = self.config
        inp = x_input.astype(jnp.float64)
        for ell in range(self.H):
            _, o = lif_layer_forward(inp, self.w_hidden[ell], self.alpha_m, cfg.v_th)
            inp = o.astype(jnp.float64)
        v_pre_r, o_r = lif_layer_forward(inp, self.w_readout, self.alpha_m, cfg.v_th)
        return jnp.sum(o_r, axis=0).astype(jnp.float64)

    def _grads(
        self,
        E_layers: List[jnp.ndarray],
        v_pres_h: List[jnp.ndarray],
        E_readout: jnp.ndarray,
        v_pre_r: jnp.ndarray,
        readout_o: jnp.ndarray,
        target: int,
        clip_value: float,
    ):
        cfg = self.config
        T = float(readout_o.shape[0])

        readout_counts = jnp.sum(readout_o, axis=0)
        ge = global_errors_from_counts(
            readout_counts,
            target,
            self.n_outputs,
            cfg.loss_temperature,
            cfg.loss_count_bias,
            cfg.loss_label_smoothing,
        )

        sig_r = surrogate_sigma(v_pre_r - cfg.v_th, cfg.beta_s)
        # grad_readout[j, m] = (1/T) sum_t g[j] * sig_r[t,j] * E_readout[t,m]
        grad_readout = jnp.einsum("j,tj,ti->ji", ge, sig_r, E_readout) / T
        grad_readout = jnp.clip(grad_readout, -clip_value, clip_value)

        loss, pred, _ = softmax_loss_from_counts(
            readout_counts,
            target,
            self.n_outputs,
            cfg.loss_temperature,
            cfg.loss_count_bias,
            cfg.loss_label_smoothing,
        )

        effective = jnp.einsum("tj,j,ji->ti", sig_r, ge, self.w_readout)
        grad_hidden: List[jnp.ndarray] = [None] * self.H

        for ell in range(self.H - 1, -1, -1):
            sig_h = surrogate_sigma(v_pres_h[ell] - cfg.v_th, cfg.beta_s)
            g_ell = jnp.einsum("ti,ti,tk->ik", sig_h, effective, E_layers[ell]) / T
            grad_hidden[ell] = jnp.clip(g_ell, -clip_value, clip_value)
            if ell > 0:
                effective = jnp.einsum("ik,ti,ti->tk", self.w_hidden[ell], sig_h, effective)

        return loss, pred, tuple(grad_hidden), grad_readout

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
        counts, E_l, v_h, E_r, v_pr, o_r = self._forward_train(x_input, self._next_key())
        loss, pred, g_h, g_r = self._grads(E_l, v_h, E_r, v_pr, o_r, int(target), clip_value)
        self._apply_grads(g_h, g_r, lr, clip_value)
        gnorms = {
            "readout": float(jnp.linalg.norm(g_r)),
        }
        for ell in range(self.H):
            gnorms[f"hidden_{ell}"] = float(jnp.linalg.norm(g_h[ell]))
        return float(loss), int(pred), gnorms

    def predict(self, x_input):
        counts = self._forward_eval(x_input)
        return int(jnp.argmax(counts))

    def batch_train_step(self, x_batch, targets, lr=1e-3, clip_value=5.0):
        B = x_batch.shape[0]
        keys = random.split(self._next_key(), B)
        losses = []
        preds = []
        g_h_acc = [jnp.zeros_like(w) for w in self.w_hidden]
        g_r_acc = jnp.zeros_like(self.w_readout)

        for b in range(B):
            _, E_l, v_h, E_r, v_pr, o_r = self._forward_train(x_batch[b], keys[b])
            loss, pred, g_h, g_r = self._grads(
                E_l, v_h, E_r, v_pr, o_r, int(targets[b]), clip_value
            )
            losses.append(loss)
            preds.append(pred)
            for ell in range(self.H):
                g_h_acc[ell] = g_h_acc[ell] + g_h[ell]
            g_r_acc = g_r_acc + g_r

        g_h_avg = tuple(g_h_acc[ell] / B for ell in range(self.H))
        g_r_avg = g_r_acc / B
        self._apply_grads(g_h_avg, g_r_avg, lr, clip_value)

        gnorms = {"readout": float(jnp.linalg.norm(g_r_avg))}
        for ell in range(self.H):
            gnorms[f"hidden_{ell}"] = float(jnp.linalg.norm(g_h_avg[ell]))
        return float(jnp.mean(jnp.stack(losses))), jnp.stack(preds), gnorms

    def batch_predict(self, x_batch):
        return jnp.array([self.predict(x_batch[b]) for b in range(x_batch.shape[0])])


def parse_hidden_sizes(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --hidden")
    return [int(x) for x in parts]
