from typing import Optional, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, random, lax
from flax import struct

from extra_layer import ExtraLayer
from hidden_layer import HiddenLayer
from neuron import JAXLIFLayer, JAXTwoCompartmentalLayer, NeuronConfig


@struct.dataclass
class NetworkParamsTwoLayer:
    w_dend_extra: jnp.ndarray
    w_soma_extra: jnp.ndarray
    w_dend_hidden: jnp.ndarray
    w_soma_hidden: jnp.ndarray
    w_readout: jnp.ndarray


class TwoLayerEProp:
    def __init__(
        self,
        key,
        n_inputs: int = 700,
        n_extra: int = 42,
        n_hidden: int = 40,
        n_outputs: int = 20,
        T: int = 700,
        neuron_config=None,
        learning_rate_extra_dendritic: float = 0.05,
        learning_rate_extra_soma: float = 0.0025,
        learning_rate_hidden_dendritic: float = 0.05,
        learning_rate_hidden_somatic: float = 0.0025,
        learning_rate_readout: float = 0.025,
        weight_decay: float = 1e-5,
        gradient_clip: float = 5.0,
        loss_temperature: float = 5.0,
        loss_count_bias: float = 0.1,
        loss_label_smoothing: float = 0.2,
        use_low_memory: bool = True,
    ):
        self.n_inputs = n_inputs
        self.n_extra = n_extra
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.T = T
        self.config = NeuronConfig() if neuron_config is None else neuron_config
        self.learning_rate_extra_dendritic = float(learning_rate_extra_dendritic)
        self.learning_rate_extra_soma = float(learning_rate_extra_soma)
        self.learning_rate_hidden_dendritic = float(learning_rate_hidden_dendritic)
        self.learning_rate_hidden_somatic = float(learning_rate_hidden_somatic)
        self.learning_rate_readout = float(learning_rate_readout)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = float(gradient_clip)
        self.loss_temperature = float(loss_temperature)
        self.loss_count_bias = float(loss_count_bias)
        self.loss_label_smoothing = float(loss_label_smoothing)
        self.use_low_memory = bool(use_low_memory)

        key_e, key_h, key_r = random.split(key, 3)
        self.extra_layer = ExtraLayer(key_e, n_extra, n_inputs, self.config)
        self.hidden_layer = HiddenLayer(key_h, n_hidden, n_extra, self.config)
        self.readout_layer = JAXLIFLayer(key_r, n_outputs, n_hidden, self.config)
        self._train_step_compiled = jit(self._train_step_impl, static_argnums=(2,))

    def get_params(self) -> NetworkParamsTwoLayer:
        return NetworkParamsTwoLayer(
            w_dend_extra=self.extra_layer.w_dend,
            w_soma_extra=self.extra_layer.w_soma,
            w_dend_hidden=self.hidden_layer.w_dend,
            w_soma_hidden=self.hidden_layer.w_soma,
            w_readout=self.readout_layer.w,
        )

    def set_params(self, params: NetworkParamsTwoLayer):
        self.extra_layer.w_dend = params.w_dend_extra
        self.extra_layer.w_soma = params.w_soma_extra
        self.hidden_layer.w_dend = params.w_dend_hidden
        self.hidden_layer.w_soma = params.w_soma_hidden
        self.readout_layer.w = params.w_readout

    def _forward_with_params(self, params: NetworkParamsTwoLayer, x_input: jnp.ndarray):
        mu_e, v_e, h_e, extra_o, mu_hist_e, tpe, v_hist_e = JAXTwoCompartmentalLayer.forward_pass(
            x_input, params.w_dend_extra, params.w_soma_extra, self.config, self.T, self.extra_layer.T_p
        )
        mu_h, v_h, h_h, hidden_o, mu_hist_h, tph, v_hist_h = JAXTwoCompartmentalLayer.forward_pass(
            extra_o, params.w_dend_hidden, params.w_soma_hidden, self.config, self.T, self.hidden_layer.T_p
        )
        readout_v, readout_o = JAXLIFLayer.forward_pass(hidden_o, params.w_readout, self.config, self.T)
        return (
            mu_e, v_e, h_e, extra_o, mu_hist_e, tpe, v_hist_e,
            mu_h, v_h, h_h, hidden_o, mu_hist_h, tph, v_hist_h,
            readout_v, readout_o,
        )

    def compute_global_errors(self, readout_o: jnp.ndarray, target: int) -> jnp.ndarray:
        readout_counts = jnp.sum(readout_o, axis=0)
        scaled = readout_counts / self.loss_temperature + self.loss_count_bias
        probs = jnp.exp(scaled - jnp.max(scaled))
        probs = probs / jnp.sum(probs)
        target_one_hot = jnp.zeros(self.n_outputs).at[target].set(1.0)
        target_one_hot = target_one_hot * (1 - self.loss_label_smoothing) + self.loss_label_smoothing / self.n_outputs
        return target_one_hot - probs

    # ----------------------------------------------------------------
    # Full-history gradient computation
    # ----------------------------------------------------------------

    def _compute_gradients(self, params: NetworkParamsTwoLayer, x_input: jnp.ndarray, target: int, clip_value: float):
        (
            _, _, h_e, extra_o, mu_hist_e, tp_e, v_hist_e,
            _, _, h_h, hidden_o, mu_hist_h, tp_h, v_hist_h,
            readout_v, readout_o,
        ) = self._forward_with_params(params, x_input)

        global_errors = self.compute_global_errors(readout_o, target)
        n_hidden = self.n_hidden
        n_extra = self.n_extra

        E_readout = JAXTwoCompartmentalLayer.compute_eligibility_traces(hidden_o, self.readout_layer.alpha)
        sigma_prime_readout = JAXTwoCompartmentalLayer.surrogate_sigma(readout_v - self.config.v_th, self.config.beta_s)
        grad_readout = jnp.einsum("ti,tj,i->ij", sigma_prime_readout, E_readout, global_errors) / self.T

        effective_error_hidden = jnp.einsum("tj,j,ji->ti", sigma_prime_readout, global_errors, params.w_readout)
        sigma_prime_hidden = JAXTwoCompartmentalLayer.surrogate_sigma(
            v_hist_h + self.config.gamma * h_h - self.config.v_th, self.config.beta_s
        )
        tp_h_int = tp_h.astype(jnp.int32)
        mu_at_tprime_h = mu_hist_h[tp_h_int, jnp.arange(n_hidden)[None, :]]
        h_prime_hidden = JAXTwoCompartmentalLayer.surrogate_sigma(mu_at_tprime_h - self.config.mu_th, self.config.beta_d)
        E_soma_hidden = JAXTwoCompartmentalLayer.compute_eligibility_traces(extra_o, self.hidden_layer.alpha_s)
        dmu_dw_hidden = JAXTwoCompartmentalLayer.compute_dmu_tprime_dw(extra_o, h_h, tp_h, self.hidden_layer.alpha)
        grad_soma_hidden = jnp.einsum("ti,tj,ti->ij", sigma_prime_hidden, E_soma_hidden, effective_error_hidden) / self.T
        grad_dend_hidden = jnp.einsum(
            "ti,ti,tij,ti->ij",
            sigma_prime_hidden,
            h_prime_hidden,
            dmu_dw_hidden,
            effective_error_hidden * self.config.gamma,
        ) / self.T

        sigma_prime_extra = JAXTwoCompartmentalLayer.surrogate_sigma(
            v_hist_e + self.config.gamma * h_e - self.config.v_th, self.config.beta_s
        )
        tp_e_int = tp_e.astype(jnp.int32)
        mu_at_tprime_e = mu_hist_e[tp_e_int, jnp.arange(n_extra)[None, :]]
        h_prime_extra = JAXTwoCompartmentalLayer.surrogate_sigma(mu_at_tprime_e - self.config.mu_th, self.config.beta_d)
        E_soma_extra = JAXTwoCompartmentalLayer.compute_eligibility_traces(x_input, self.extra_layer.alpha_s)
        dmu_dw_extra = JAXTwoCompartmentalLayer.compute_dmu_tprime_dw(x_input, h_e, tp_e, self.extra_layer.alpha)

        coeff_extra_soma_p1 = jnp.einsum(
            "ik,ti,tk,ti->tk", params.w_soma_hidden, sigma_prime_hidden, sigma_prime_extra, effective_error_hidden
        )
        grad_extra_soma = jnp.einsum("tk,tj->kj", coeff_extra_soma_p1, E_soma_extra) / self.T
        sigma_extra_at_tprime_h = sigma_prime_extra[tp_h_int]
        E_extra_at_tprime_h = E_soma_extra[tp_h_int]
        term_p2 = (
            params.w_dend_hidden
            * sigma_prime_hidden[:, :, None]
            * h_prime_hidden[:, :, None]
            * sigma_extra_at_tprime_h
            * effective_error_hidden[:, :, None]
        )
        grad_extra_soma = grad_extra_soma + (self.config.gamma / self.T) * jnp.einsum(
            "tik,tij->kj", term_p2, E_extra_at_tprime_h
        )

        coeff_extra_dend_p1 = jnp.einsum(
            "ik,ti,tk,ti->tk",
            params.w_soma_hidden,
            sigma_prime_hidden,
            sigma_prime_extra * h_prime_extra,
            effective_error_hidden,
        )
        grad_extra_dend = jnp.einsum("tk,tkj->kj", coeff_extra_dend_p1 * self.config.gamma, dmu_dw_extra) / self.T
        h_extra_at_tprime_h = h_prime_extra[tp_h_int]
        term_d2 = (
            params.w_dend_hidden
            * sigma_prime_hidden[:, :, None]
            * h_prime_hidden[:, :, None]
            * sigma_extra_at_tprime_h
            * h_extra_at_tprime_h
            * effective_error_hidden[:, :, None]
            * self.config.gamma
        )

        def _accumulate_dend_p2(carry, t):
            grad_acc = carry
            dmu_t = dmu_dw_extra[tp_h_int[t]]
            inc = (self.config.gamma / self.T) * jnp.einsum("ik,ikj->kj", term_d2[t], dmu_t)
            return grad_acc + inc, None

        grad_extra_dend_p2, _ = lax.scan(_accumulate_dend_p2, jnp.zeros((n_extra, self.n_inputs)), jnp.arange(self.T))
        grad_extra_dend = grad_extra_dend + grad_extra_dend_p2

        grad_extra_dend = jnp.clip(grad_extra_dend, -clip_value, clip_value)
        grad_extra_soma = jnp.clip(grad_extra_soma, -clip_value, clip_value)
        grad_dend_hidden = jnp.clip(grad_dend_hidden, -clip_value, clip_value)
        grad_soma_hidden = jnp.clip(grad_soma_hidden, -clip_value, clip_value)
        grad_readout = jnp.clip(grad_readout, -clip_value, clip_value)

        readout_counts = jnp.sum(readout_o, axis=0)
        scaled = readout_counts / self.loss_temperature + self.loss_count_bias
        probs = jnp.exp(scaled - jnp.max(scaled))
        probs = probs / jnp.sum(probs)
        target_one_hot = jnp.zeros(self.n_outputs).at[target].set(1.0)
        target_one_hot = target_one_hot * (1 - self.loss_label_smoothing) + self.loss_label_smoothing / self.n_outputs
        loss = -jnp.sum(target_one_hot * jnp.log(probs + 1e-8))
        return grad_extra_dend, grad_extra_soma, grad_dend_hidden, grad_soma_hidden, grad_readout, loss, readout_o

    def _train_step_impl_full_history(
        self,
        params: NetworkParamsTwoLayer,
        x_input: jnp.ndarray,
        target: int,
        lr_ed: float,
        lr_es: float,
        lr_hd: float,
        lr_hs: float,
        lr_r: float,
        clip_value: float = 5.0,
    ):
        g_ed, g_es, g_hd, g_hs, g_r, loss, readout_o = self._compute_gradients(params, x_input, target, clip_value)
        new_params = NetworkParamsTwoLayer(
            w_dend_extra=jnp.clip(params.w_dend_extra * (1 - self.weight_decay) + lr_ed * g_ed, -1.0, 1.0),
            w_soma_extra=jnp.clip(params.w_soma_extra * (1 - self.weight_decay) + lr_es * g_es, -1.0, 1.0),
            w_dend_hidden=jnp.clip(params.w_dend_hidden * (1 - self.weight_decay) + lr_hd * g_hd, -1.0, 1.0),
            w_soma_hidden=jnp.clip(params.w_soma_hidden * (1 - self.weight_decay) + lr_hs * g_hs, -1.0, 1.0),
            w_readout=jnp.clip(params.w_readout * (1 - self.weight_decay) + lr_r * g_r, -1.0, 1.0),
        )
        return new_params, loss, readout_o

    # ----------------------------------------------------------------
    # Low-memory online gradient computation
    # Single scan: forward dynamics + eligibility/sensitivity updates
    # + snapshotting extra state at hidden plateau starts
    # + output-conditioned basis gradient accumulation.
    # Memory: O(n_hidden * n_extra * n_inputs) carry instead of
    #         O(T * n_extra * n_inputs) full history.
    # ----------------------------------------------------------------

    def _train_step_impl_low_memory(
        self,
        params: NetworkParamsTwoLayer,
        x_input: jnp.ndarray,
        target: int,
        lr_ed: float,
        lr_es: float,
        lr_hd: float,
        lr_hs: float,
        lr_r: float,
        clip_value: float = 5.0,
    ):
        n_in = self.n_inputs
        n_e = self.n_extra
        n_h = self.n_hidden
        n_o = self.n_outputs
        cfg = self.config
        alpha_d_e = self.extra_layer.alpha
        alpha_s_e = self.extra_layer.alpha_s
        alpha_d_h = self.hidden_layer.alpha
        alpha_s_h = self.hidden_layer.alpha_s
        alpha_r = self.readout_layer.alpha
        T_p_e = self.extra_layer.T_p
        T_p_h = self.hidden_layer.T_p

        def scan_step(carry, x_t):
            (
                # Extra forward state
                mu_e, v_e, h_e, plat_age_e, mu_init_e,
                # Hidden forward state
                mu_h, v_h, h_h, plat_age_h, mu_init_h,
                # Readout state
                readout_v,
                # Extra sensitivities
                dmu_dw_e, dmu_init_e, E_soma_e,
                # Hidden sensitivities
                dmu_dw_h, dmu_init_h, E_soma_h, E_readout,
                # Snapshots of extra state at hidden plateau start (per hidden neuron)
                snap_sigma_e, snap_hprime_e, snap_E_soma_e, snap_dmu_eff_e,
                # Output-conditioned basis accumulators
                A_readout, A_soma_h, A_dend_h, B_soma_e, B_dend_e,
                # Readout spike count
                readout_counts,
            ) = carry

            # ==================== FORWARD DYNAMICS ====================

            # --- Extra layer ---
            dend_in_e = x_t @ params.w_dend_extra.T
            soma_in_e = x_t @ params.w_soma_extra.T
            mu_e_new = alpha_d_e * mu_e + (1 - h_e) * dend_in_e

            new_start_e = (h_e == 0)
            mu_init_e_new = jnp.where(new_start_e, mu_e_new, mu_init_e)
            plat_age_e_new = jnp.where(new_start_e, 0, plat_age_e + 1)
            h_e_new = jnp.where(
                (mu_init_e_new >= cfg.mu_th) & (plat_age_e_new <= T_p_e), 1, 0
            ).astype(jnp.int64)

            v_e_new = alpha_s_e * v_e + soma_in_e
            extra_o = jnp.where(
                v_e_new >= cfg.v_th - cfg.gamma * h_e_new, 1, 0
            ).astype(jnp.int64)
            v_e_post = v_e_new * (1 - extra_o)

            # --- Hidden layer ---
            extra_o_f = extra_o.astype(jnp.float64)
            dend_in_h = extra_o_f @ params.w_dend_hidden.T
            soma_in_h = extra_o_f @ params.w_soma_hidden.T
            mu_h_new = alpha_d_h * mu_h + (1 - h_h) * dend_in_h

            new_start_h = (h_h == 0)
            mu_init_h_new = jnp.where(new_start_h, mu_h_new, mu_init_h)
            plat_age_h_new = jnp.where(new_start_h, 0, plat_age_h + 1)
            h_h_new = jnp.where(
                (mu_init_h_new >= cfg.mu_th) & (plat_age_h_new <= T_p_h), 1, 0
            ).astype(jnp.int64)

            v_h_new = alpha_s_h * v_h + soma_in_h
            hidden_o = jnp.where(
                v_h_new >= cfg.v_th - cfg.gamma * h_h_new, 1, 0
            ).astype(jnp.int64)
            v_h_post = v_h_new * (1 - hidden_o)

            # --- Readout layer ---
            hidden_o_f = hidden_o.astype(jnp.float64)
            readout_in = hidden_o_f @ params.w_readout.T
            readout_v_new = alpha_r * readout_v + readout_in
            readout_o = jnp.where(readout_v_new >= cfg.v_th, 1, 0).astype(jnp.int64)
            readout_v_post = readout_v_new * (1 - readout_o) + cfg.v_reset * readout_o
            readout_counts_new = readout_counts + readout_o

            # ==================== SENSITIVITY / ELIGIBILITY UPDATES ====================

            # Extra dmu_dw (dendritic sensitivity to input weights)
            x_b = jnp.broadcast_to(x_t[None, :], (n_e, n_in))
            dmu_dw_e_new = alpha_d_e * dmu_dw_e + (1 - h_e[:, None]) * x_b
            dmu_init_e_new = jnp.where(new_start_e[:, None], dmu_dw_e_new, dmu_init_e)
            dmu_eff_e = jnp.where(h_e_new[:, None] == 1, dmu_init_e_new, dmu_dw_e_new)

            E_soma_e_new = alpha_s_e * E_soma_e + x_t

            # Hidden dmu_dw (dendritic sensitivity to extra->hidden weights)
            extra_o_b = jnp.broadcast_to(extra_o_f[None, :], (n_h, n_e))
            dmu_dw_h_new = alpha_d_h * dmu_dw_h + (1 - h_h[:, None]) * extra_o_b
            dmu_init_h_new = jnp.where(new_start_h[:, None], dmu_dw_h_new, dmu_init_h)
            dmu_eff_h = jnp.where(h_h_new[:, None] == 1, dmu_init_h_new, dmu_dw_h_new)

            E_soma_h_new = alpha_s_h * E_soma_h + extra_o_f
            E_readout_new = alpha_r * E_readout + hidden_o_f

            # ==================== SURROGATE DERIVATIVES ====================

            sigma_prime_readout = JAXTwoCompartmentalLayer.surrogate_sigma(
                readout_v_new - cfg.v_th, cfg.beta_s
            )
            sigma_prime_hidden = JAXTwoCompartmentalLayer.surrogate_sigma(
                v_h_new + cfg.gamma * h_h_new - cfg.v_th, cfg.beta_s
            )
            h_prime_hidden = JAXTwoCompartmentalLayer.surrogate_sigma(
                mu_init_h_new - cfg.mu_th, cfg.beta_d
            )
            sigma_prime_extra = JAXTwoCompartmentalLayer.surrogate_sigma(
                v_e_new + cfg.gamma * h_e_new - cfg.v_th, cfg.beta_s
            )
            h_prime_extra = JAXTwoCompartmentalLayer.surrogate_sigma(
                mu_init_e_new - cfg.mu_th, cfg.beta_d
            )

            # ==================== SNAPSHOT EXTRA STATE AT HIDDEN PLATEAU START ====================
            # When hidden neuron i transitions from h=0 to a new candidate plateau,
            # freeze the current extra-layer quantities for the p2 gradient path.

            snap_sigma_e_new = jnp.where(
                new_start_h[:, None],
                jnp.broadcast_to(sigma_prime_extra[None, :], (n_h, n_e)),
                snap_sigma_e,
            )
            snap_hprime_e_new = jnp.where(
                new_start_h[:, None],
                jnp.broadcast_to(h_prime_extra[None, :], (n_h, n_e)),
                snap_hprime_e,
            )
            snap_E_soma_e_new = jnp.where(
                new_start_h[:, None],
                jnp.broadcast_to(E_soma_e_new[None, :], (n_h, n_in)),
                snap_E_soma_e,
            )
            snap_dmu_eff_e_new = jnp.where(
                new_start_h[:, None, None],
                jnp.broadcast_to(dmu_eff_e[None, :, :], (n_h, n_e, n_in)),
                snap_dmu_eff_e,
            )

            # ==================== BASIS GRADIENT ACCUMULATION ====================
            # Accumulate output-conditioned basis tensors so that after the scan:
            #   grad_X = (1/T) * einsum("j, j...->...", global_errors, Basis_X)

            # eta[j,i] = sigma'_readout[j] * w_readout[j,i]
            eta = sigma_prime_readout[:, None] * params.w_readout          # (n_o, n_h)
            # chi[j,i] = eta[j,i] * sigma'_hidden[i]  (soma error propagation)
            chi = eta * sigma_prime_hidden[None, :]                        # (n_o, n_h)
            # chi_h[j,i] = chi[j,i] * h'_hidden[i]    (dendrite error propagation)
            chi_h = chi * h_prime_hidden[None, :]                          # (n_o, n_h)

            # --- Readout basis: A_readout[j,i] ---
            A_readout_new = A_readout + jnp.einsum(
                "j,i->ji", sigma_prime_readout, E_readout_new
            )

            # --- Hidden soma basis: A_soma_h[j,i,k] ---
            soma_base_h = jnp.einsum("i,k->ik", sigma_prime_hidden, E_soma_h_new)
            A_soma_h_new = A_soma_h + jnp.einsum("ji,ik->jik", eta, soma_base_h)

            # --- Hidden dend basis: A_dend_h[j,i,k] ---
            dend_base_h = jnp.einsum(
                "i,ik->ik", sigma_prime_hidden * h_prime_hidden * cfg.gamma, dmu_eff_h
            )
            A_dend_h_new = A_dend_h + jnp.einsum("ji,ik->jik", eta, dend_base_h)

            # --- Extra soma basis: B_soma_e[j,k,m] = p1 + p2 ---
            # p1: through hidden soma weights
            #   phi_soma[j,k] = sum_i chi[j,i]*w_soma_h[i,k]  * sigma'_extra[k]
            phi_soma = (
                jnp.einsum("ji,ik->jk", chi, params.w_soma_hidden)
                * sigma_prime_extra[None, :]
            )                                                               # (n_o, n_e)
            B_soma_e_p1 = jnp.einsum("jk,m->jkm", phi_soma, E_soma_e_new)

            # p2: through hidden dend weights, using snapshots
            #   sum_i chi_h[j,i]*gamma*w_dend_h[i,k]*snap_sigma_e[i,k] * snap_E_soma_e[i,m]
            factor_sp2 = params.w_dend_hidden * snap_sigma_e_new            # (n_h, n_e)
            weighted_sp2 = (
                chi_h[:, :, None] * (cfg.gamma * factor_sp2)[None, :, :]
            )                                                               # (n_o, n_h, n_e)
            B_soma_e_p2 = jnp.einsum("jik,im->jkm", weighted_sp2, snap_E_soma_e_new)

            B_soma_e_new = B_soma_e + B_soma_e_p1 + B_soma_e_p2

            # --- Extra dend basis: B_dend_e[j,k,m] = p1 + p2 ---
            # p1: through hidden soma weights
            phi_dend = (
                jnp.einsum("ji,ik->jk", chi, params.w_soma_hidden)
                * (sigma_prime_extra * h_prime_extra * cfg.gamma)[None, :]
            )                                                               # (n_o, n_e)
            B_dend_e_p1 = jnp.einsum("jk,km->jkm", phi_dend, dmu_eff_e)

            # p2: through hidden dend weights, using snapshots (gamma^2 total)
            factor_dp2 = (
                params.w_dend_hidden * snap_sigma_e_new * snap_hprime_e_new
            )                                                               # (n_h, n_e)
            weighted_dp2 = (
                chi_h[:, :, None]
                * (cfg.gamma * cfg.gamma * factor_dp2)[None, :, :]
            )                                                               # (n_o, n_h, n_e)
            B_dend_e_p2 = jnp.einsum("jik,ikm->jkm", weighted_dp2, snap_dmu_eff_e_new)

            B_dend_e_new = B_dend_e + B_dend_e_p1 + B_dend_e_p2

            # ==================== PACK CARRY ====================

            next_carry = (
                mu_e_new, v_e_post, h_e_new, plat_age_e_new, mu_init_e_new,
                mu_h_new, v_h_post, h_h_new, plat_age_h_new, mu_init_h_new,
                readout_v_post,
                dmu_dw_e_new, dmu_init_e_new, E_soma_e_new,
                dmu_dw_h_new, dmu_init_h_new, E_soma_h_new, E_readout_new,
                snap_sigma_e_new, snap_hprime_e_new, snap_E_soma_e_new, snap_dmu_eff_e_new,
                A_readout_new, A_soma_h_new, A_dend_h_new, B_soma_e_new, B_dend_e_new,
                readout_counts_new,
            )
            return next_carry, readout_o

        # ==================== INITIAL CARRY ====================

        init_carry = (
            jnp.zeros(n_e),                           # mu_e
            jnp.zeros(n_e),                           # v_e
            jnp.zeros(n_e, dtype=jnp.int64),          # h_e
            jnp.zeros(n_e, dtype=jnp.int64),          # plat_age_e
            jnp.zeros(n_e),                           # mu_init_e
            jnp.zeros(n_h),                           # mu_h
            jnp.zeros(n_h),                           # v_h
            jnp.zeros(n_h, dtype=jnp.int64),          # h_h
            jnp.zeros(n_h, dtype=jnp.int64),          # plat_age_h
            jnp.zeros(n_h),                           # mu_init_h
            jnp.zeros(n_o),                           # readout_v
            jnp.zeros((n_e, n_in)),                   # dmu_dw_e
            jnp.zeros((n_e, n_in)),                   # dmu_init_e
            jnp.zeros(n_in),                          # E_soma_e
            jnp.zeros((n_h, n_e)),                    # dmu_dw_h
            jnp.zeros((n_h, n_e)),                    # dmu_init_h
            jnp.zeros(n_e),                           # E_soma_h
            jnp.zeros(n_h),                           # E_readout
            jnp.zeros((n_h, n_e)),                    # snap_sigma_e
            jnp.zeros((n_h, n_e)),                    # snap_hprime_e
            jnp.zeros((n_h, n_in)),                   # snap_E_soma_e
            jnp.zeros((n_h, n_e, n_in)),              # snap_dmu_eff_e
            jnp.zeros((n_o, n_h)),                    # A_readout
            jnp.zeros((n_o, n_h, n_e)),               # A_soma_h
            jnp.zeros((n_o, n_h, n_e)),               # A_dend_h
            jnp.zeros((n_o, n_e, n_in)),              # B_soma_e
            jnp.zeros((n_o, n_e, n_in)),              # B_dend_e
            jnp.zeros(n_o),                           # readout_counts
        )

        final_carry, readout_o_hist = lax.scan(scan_step, init_carry, x_input)

        # Unpack basis accumulators and spike counts
        A_readout = final_carry[22]
        A_soma_h = final_carry[23]
        A_dend_h = final_carry[24]
        B_soma_e = final_carry[25]
        B_dend_e = final_carry[26]
        readout_counts = final_carry[27]

        # ==================== GLOBAL ERROR & LOSS ====================

        scaled = readout_counts / self.loss_temperature + self.loss_count_bias
        probs = jnp.exp(scaled - jnp.max(scaled))
        probs = probs / jnp.sum(probs)
        target_one_hot = jnp.zeros(n_o).at[target].set(1.0)
        target_one_hot = (
            target_one_hot * (1 - self.loss_label_smoothing)
            + self.loss_label_smoothing / n_o
        )
        global_errors = target_one_hot - probs
        loss = -jnp.sum(target_one_hot * jnp.log(probs + 1e-8))

        # ==================== CONTRACT BASIS WITH GLOBAL ERROR ====================

        grad_readout = jnp.einsum("j,jk->jk", global_errors, A_readout) / self.T
        grad_soma_hidden = jnp.einsum("j,jik->ik", global_errors, A_soma_h) / self.T
        grad_dend_hidden = jnp.einsum("j,jik->ik", global_errors, A_dend_h) / self.T
        grad_soma_extra = jnp.einsum("j,jkm->km", global_errors, B_soma_e) / self.T
        grad_dend_extra = jnp.einsum("j,jkm->km", global_errors, B_dend_e) / self.T

        grad_readout = jnp.clip(grad_readout, -clip_value, clip_value)
        grad_soma_hidden = jnp.clip(grad_soma_hidden, -clip_value, clip_value)
        grad_dend_hidden = jnp.clip(grad_dend_hidden, -clip_value, clip_value)
        grad_soma_extra = jnp.clip(grad_soma_extra, -clip_value, clip_value)
        grad_dend_extra = jnp.clip(grad_dend_extra, -clip_value, clip_value)

        # ==================== WEIGHT UPDATE ====================

        new_params = NetworkParamsTwoLayer(
            w_dend_extra=jnp.clip(
                params.w_dend_extra * (1 - self.weight_decay) + lr_ed * grad_dend_extra, -1.0, 1.0
            ),
            w_soma_extra=jnp.clip(
                params.w_soma_extra * (1 - self.weight_decay) + lr_es * grad_soma_extra, -1.0, 1.0
            ),
            w_dend_hidden=jnp.clip(
                params.w_dend_hidden * (1 - self.weight_decay) + lr_hd * grad_dend_hidden, -1.0, 1.0
            ),
            w_soma_hidden=jnp.clip(
                params.w_soma_hidden * (1 - self.weight_decay) + lr_hs * grad_soma_hidden, -1.0, 1.0
            ),
            w_readout=jnp.clip(
                params.w_readout * (1 - self.weight_decay) + lr_r * grad_readout, -1.0, 1.0
            ),
        )
        return new_params, loss, readout_o_hist

    # ----------------------------------------------------------------
    # Dispatcher
    # ----------------------------------------------------------------

    def _train_step_impl(
        self,
        params: NetworkParamsTwoLayer,
        x_input: jnp.ndarray,
        target: int,
        lr_ed: float,
        lr_es: float,
        lr_hd: float,
        lr_hs: float,
        lr_r: float,
        clip_value: float = 5.0,
    ):
        if self.use_low_memory:
            return self._train_step_impl_low_memory(
                params, x_input, target, lr_ed, lr_es, lr_hd, lr_hs, lr_r, clip_value
            )
        return self._train_step_impl_full_history(
            params, x_input, target, lr_ed, lr_es, lr_hd, lr_hs, lr_r, clip_value
        )

    def train_step(self, x_input: jnp.ndarray, target: int):
        if not isinstance(x_input, jnp.ndarray):
            x_input = jnp.array(x_input)
        params = self.get_params()
        new_params, loss, readout_o = self._train_step_compiled(
            params,
            x_input,
            target,
            self.learning_rate_extra_dendritic,
            self.learning_rate_extra_soma,
            self.learning_rate_hidden_dendritic,
            self.learning_rate_hidden_somatic,
            self.learning_rate_readout,
            self.gradient_clip,
        )
        self.set_params(new_params)
        return float(loss), readout_o

    def predict(self, x_input: jnp.ndarray) -> int:
        if not isinstance(x_input, jnp.ndarray):
            x_input = jnp.array(x_input)
        *_, readout_o = self._forward_with_params(self.get_params(), x_input)
        return int(jnp.argmax(jnp.sum(readout_o, axis=0)))
