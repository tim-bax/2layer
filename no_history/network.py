import jax.numpy as jnp
from jax import random, jit, lax

from config import NeuronConfig, surrogate_sigma
from two_comp_neuron import TwoCompNeuron
from lif_neuron import LIFNeuron


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

    # ------------------------------------------------------------------
    # Training forward pass: hybrid scan.
    # - A_dend accumulated inside scan (avoids emitting large dmu_dw_at_tprime)
    # - Small per-step factors emitted for A_readout and A_soma (built post-scan)
    # ------------------------------------------------------------------

    @staticmethod
    @jit
    def _run_sequence(
        x_input,
        w_dend, w_soma, w_readout,
        alpha_s, alpha_d, alpha_m,
        T_p, config,
        h_carry_init, r_carry_init,
        A_d_init,
    ):
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

            # A_dend: accumulated in-scan (needs dmu_atp_new which is large)
            eta = sp_readout[:, None] * w_readout * sp_hidden[None, :]
            eta_d = eta * (hp_hidden * config.gamma)[None, :]
            A_d = A_d + jnp.einsum("ji,ik->jik", eta_d, dmu_atp_new)

            new_carry = (h_carry, r_carry, A_d)
            # Emit only small per-step factors for A_readout and A_soma
            per_step = (sp_readout, sp_hidden, r_E, E_soma_new, h_o)
            return new_carry, per_step

        init_carry = (h_carry_init, r_carry_init, A_d_init)
        scan_inputs = (dend_inputs, soma_inputs, x_input, time_indices)
        final_carry, per_step_all = lax.scan(step, init_carry, scan_inputs)

        sp_r, sp_h, E_r, E_s, o_all = per_step_all
        _, r_carry_f, A_d_f = final_carry
        readout_counts = r_carry_f[1]

        # Build A_readout and A_soma from small emitted arrays
        A_readout = jnp.einsum("ti,tj->ij", sp_r, E_r)
        C_soma = jnp.einsum("tj,ti,tk->jik", sp_r, sp_h, E_s)
        A_soma = w_readout[:, :, None] * C_soma

        return readout_counts, A_readout, A_soma, A_d_f, o_all

    def run_sequence(self, x_input):
        h_carry_init = self.hidden.init_carry()
        r_carry_init = self.readout.init_carry()
        A_d_init = jnp.zeros((self.n_outputs, self.n_hidden, self.n_inputs))

        return Network._run_sequence(
            x_input,
            self.hidden.w_dend, self.hidden.w_soma, self.readout.w,
            self.hidden.alpha_s, self.hidden.alpha_d, self.readout.alpha_m,
            self.hidden.T_p, self.config,
            h_carry_init, r_carry_init,
            A_d_init,
        )

    # ------------------------------------------------------------------
    # Predict-only forward pass (no eligibility / accumulation)
    # ------------------------------------------------------------------

    @staticmethod
    @jit
    def _predict_forward(
        x_input,
        w_dend, w_soma, w_readout,
        alpha_s, alpha_d, alpha_m,
        T_p, config,
    ):
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

    # ------------------------------------------------------------------
    # JIT-compiled loss + gradient computation
    # ------------------------------------------------------------------

    @staticmethod
    @jit
    def _compute_loss_and_grads(
        readout_counts, A_readout, A_soma, A_dend,
        target_smoothed, T,
        loss_temperature, loss_count_bias,
    ):
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

    def compute_loss_and_grads(self, readout_counts, A_readout, A_soma, A_dend, target, T):
        cfg = self.config
        target_one_hot = jnp.zeros(self.n_outputs).at[target].set(1.0)
        target_smoothed = (
            target_one_hot * (1 - cfg.loss_label_smoothing)
            + cfg.loss_label_smoothing / self.n_outputs
        )
        return Network._compute_loss_and_grads(
            readout_counts, A_readout, A_soma, A_dend,
            target_smoothed, T,
            cfg.loss_temperature, cfg.loss_count_bias,
        )

    # ------------------------------------------------------------------
    # JIT-compiled parameter update
    # ------------------------------------------------------------------

    @staticmethod
    @jit
    def _apply_grads(w_dend, w_soma, w_readout, g_dend, g_soma, g_readout, lr, clip_value):
        g_readout = jnp.clip(g_readout, -clip_value, clip_value)
        g_soma = jnp.clip(g_soma, -clip_value, clip_value)
        g_dend = jnp.clip(g_dend, -clip_value, clip_value)
        return (
            w_dend + lr * g_dend,
            w_soma + lr * g_soma,
            w_readout + lr * g_readout,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_step(self, x_input, target, lr=1e-3, clip_value=1.0):
        T = x_input.shape[0]
        readout_counts, A_readout, A_soma, A_dend, _ = self.run_sequence(x_input)

        loss, prediction, g_readout, g_soma, g_dend = self.compute_loss_and_grads(
            readout_counts, A_readout, A_soma, A_dend, target, T,
        )

        self.hidden.w_dend, self.hidden.w_soma, self.readout.w = Network._apply_grads(
            self.hidden.w_dend, self.hidden.w_soma, self.readout.w,
            g_dend, g_soma, g_readout, lr, clip_value,
        )

        return float(loss), int(prediction)

    def predict(self, x_input):
        counts = Network._predict_forward(
            x_input,
            self.hidden.w_dend, self.hidden.w_soma, self.readout.w,
            self.hidden.alpha_s, self.hidden.alpha_d, self.readout.alpha_m,
            self.hidden.T_p, self.config,
        )
        return int(jnp.argmax(counts))
