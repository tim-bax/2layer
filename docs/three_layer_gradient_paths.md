# Three-Layer E-Prop: Gradient Paths and Times (t vs t′)

This document spells out how gradients flow through soma (s) and dendrite (d) in 1-layer, 2-layer, and 3-layer 2-compartment networks, and **at which times** (all `t` vs plateau time `t′`) each term is evaluated.

## Notation

- **Layer indexing**: L1 = bottom (input), L2 = middle, L3 = top (next to readout). Readout is LIF (no dendrite).
- **Quantities**:
  - `σ'` = somatic surrogate (∂spike/∂v), evaluated at **current time t** (or at upper layer’s t′ when backprop through that layer).
  - `h'` = dendritic surrogate (∂plateau/∂μ), evaluated at **plateau initiation time t′** of that layer (μ at t′).
  - `E` = somatic eligibility trace (recurrent over input spikes); stored as (T, n_inputs).
  - `dμ/dw` = dendritic eligibility (∂μ/∂w); uses plateau logic, so at time t we use the value **at t′[t, neuron]** (or equivalent). Stored as (T, n_neurons, n_inputs).
- **t′**: For each neuron, the time at which its dendrite **entered** the current plateau. Shape (T, n_neurons). During the plateau, t′ is constant; when not in plateau, t′ = t.

---

## General rule for L layers (input → L1 → … → L_L → readout)

**1. Effective error (recursive)**  
- **Top 2comp layer (L_L)**:  
  `e_L(t) = σ'_readout(t) · global_errors · w_readout`  
  (one step back from the readout; sum over all `t` in the gradient.)  
- **Any lower layer ℓ (ℓ = 1, …, L−1)**:  
  Effective error `e_ℓ` comes from the layer above (ℓ+1) along **two routes**:
  - **Soma route**: use layer-(ℓ+1) at **all t** (σ′_{ℓ+1}(t), e_{ℓ+1}(t)) and weights w_soma_{ℓ+1}.
  - **Dend route**: use layer-(ℓ+1) at **t′_{ℓ+1}** only (σ′_{ℓ+1}, h′_{ℓ+1}, e_{ℓ+1} at t′_{ℓ+1}) and weights w_dend_{ℓ+1}.  
  Combine these two to get the time series (and possibly point-in-time contributions) that act as effective error for layer ℓ.

**2. Gradient for each 2comp layer ℓ**  
Once you have the effective error `e_ℓ` (from the layer above as above), the **local** update for layer ℓ is always the same:

- **Soma weights**:  
  `grad_soma_ℓ ∝ Σ_t σ′_ℓ(t) · E_ℓ(t) · [effective signal to this layer](t)`  
  Eligibility `E_ℓ` is over this layer’s **input** (spikes from ℓ−1 or from the dataset). Time: **all t**.

- **Dend weights**:  
  `grad_dend_ℓ ∝ γ · Σ_t σ′_ℓ(t) · h′_ℓ(t′_ℓ) · (dμ_ℓ/dw)(t) · [effective signal](t)`  
  Same effective signal; h′_ℓ and dμ_ℓ/dw use this layer’s **plateau time t′_ℓ**. Time: sum over **all t**, with dendritic terms at **t′_ℓ**.

**3. Time rule for the effective signal (which t or t′ to use)**  
- When the path from the layer above goes through that layer’s **soma**: the layer above contributes at **all t**; the current layer ℓ uses its traces at **all t** (soma) or at **t′_ℓ** (dend) as in step 2.  
- When the path from the layer above goes through that layer’s **dendrite**: the layer above is evaluated at **t′_{ℓ+1}**, and **the current layer ℓ is also evaluated at t′_{ℓ+1}** (the time when the dendrite above “saw” layer ℓ).  
  So: **through upper dendrite ⇒ evaluate lower layer at upper’s t′.**

**4. Summary in one sentence**  
Each layer gets an effective error from the layer above (soma path: all t; dend path: at t′ of the layer above). It then applies the same local e-prop rule: soma gradient = σ′ · E · effective_error (all t), dend gradient = γ · σ′ · h′(t′) · dμ/dw · effective_error (dendritic terms at this layer’s t′; when effective error came through the upper dendrite, that effective error and this layer’s σ′, E, h′, dμ/dw are evaluated at the upper’s t′).

---

## Mathematical formulation (for a paper)

**Architecture.** Input \(\mathbf{x}(t)\) drives a stack of \(L\) two-compartment layers; layer \(\ell\) has somatic spike output \(\mathbf{o}^\ell(t)\), dendritic plateau \(h^\ell(t)\), and plateau-onset times \(t'^\ell(t,i)\) per neuron \(i\). The top layer \(\ell=L\) feeds a LIF readout with output \(\mathbf{o}^{\mathrm{out}}\); the loss yields a global error vector \(\mathbf{e}^{\mathrm{global}}\).

**Notation.**  
\(\sigma'^\ell_i(t)\) = somatic surrogate \(\partial o^\ell_i/\partial v^\ell_i\);  
\(h'^\ell_i(t)\) = dendritic surrogate \(\partial h^\ell_i/\partial \mu^\ell_i\), evaluated at plateau onset (so at \(t'^\ell(t,i)\));  
\(E^\ell(t)\) = somatic eligibility of layer \(\ell\) for its input (recurrent filter on pre-synaptic spikes);  
\(\partial\mu^\ell/\partial\mathbf{W}^\ell_{\mathrm{d}}\) = dendritic eligibility of layer \(\ell\) (with plateau logic: at time \(t\) use value at \(t'^\ell(t,i)\)).  
Weights: \(\mathbf{W}^\ell_{\mathrm{s}}\) (soma), \(\mathbf{W}^\ell_{\mathrm{d}}\) (dendrite) for layer \(\ell\); readout weights \(\mathbf{W}^{\mathrm{out}}\).

**Effective error (recursive).**  
For the top hidden layer (\(\ell=L\)) and for any lower layer \(\ell \in \{1,\ldots,L-1\}\):

\[
\mathbf{e}^L(t)
\;=\;
\sigma'^{\mathrm{out}}(t)^{\top} \mathbf{e}^{\mathrm{global}} \,\mathbf{W}^{\mathrm{out}}
\quad\text{(vector of size } n_L\text{)}
\]

\[
\mathbf{e}^\ell(t)
\;=\;
\underbrace{\mathbf{W}^{\ell+1}_{\mathrm{s}}^{\top}\,
  \big(\sigma'^{\ell+1}(t) \odot \mathbf{e}^{\ell+1}(t)\big)}_{\text{soma path: all } t}
\;+\;
\underbrace{\mathbf{W}^{\ell+1}_{\mathrm{d}}^{\top}\,
  \big(\sigma'^{\ell+1}(t'^{\ell+1}) \odot h'^{\ell+1}(t'^{\ell+1}) \odot \mathbf{e}^{\ell+1}(t'^{\ell+1})\big)}_{\text{dend path: at } t'^{\ell+1}}
\]

(\(\odot\) = element-wise product; \(t'^{\ell+1}\) denotes evaluation at \(t'^{\ell+1}(t,i)\) for each neuron \(i\) of layer \(\ell+1\). The second term contributes only at the time indices where the layer-above dendrite is at plateau.)

**Weight updates (local e-prop at layer \(\ell\)).**  
Once \(\mathbf{e}^\ell\) is defined as above, the updates are the same for every layer \(\ell\):

\[
\Delta\mathbf{W}^\ell_{\mathrm{s}}
\;\propto\;
\frac{1}{T}\sum_t
\sigma'^\ell(t) \odot \mathbf{e}^\ell(t)
\;\otimes\; E^\ell(t)
\qquad\text{(soma: all } t\text{)}
\]

\[
\Delta\mathbf{W}^\ell_{\mathrm{d}}
\;\propto\;
\frac{\gamma}{T}\sum_t
\sigma'^\ell(t) \odot h'^\ell(t'^\ell) \odot \frac{\partial\mu^\ell}{\partial\mathbf{W}^\ell_{\mathrm{d}}}(t)
\odot \mathbf{e}^\ell(t)
\qquad\text{(dend: } h',\partial\mu/\partial W \text{ at } t'^\ell\text{)}
\]

(\(\otimes\) = outer product so that the product of a vector “\(\sigma'\odot e\)” with eligibility \(E\) gives the gradient matrix.)

**Time rule.**  
When forming \(\mathbf{e}^\ell\) from layer \(\ell+1\), the **soma** term uses layer \(\ell+1\) at **all** \(t\); the **dendrite** term uses layer \(\ell+1\) (and thus the pre-synaptic activity that drove it) **only at** \(t'^{\ell+1}\). So: *through the upper dendrite \(\Rightarrow\) evaluate the lower layer at the upper layer’s plateau time \(t'^{\ell+1}\).*

**One-line summary.**  
Effective error propagates backward layer-by-layer via a soma path (all times) and a dendrite path (at each layer’s \(t'\)); each layer then applies the same local e-prop rule with somatic eligibility \(E^\ell(t)\) and dendritic eligibility at \(t'^\ell\).

---

## 1-Layer (input → 1× 2comp → readout)

**Effective error** (from readout only):

- `e_1(t) = σ'_readout(t) · global_errors · w_readout`  →  (T, n_1).

**Gradients for the single 2comp layer:**

| Path (readout→hidden) | Formula (per time) | Time used |
|----------------------|--------------------|-----------|
| **Soma**             | `σ'_1(t) · E_1(t) · e_1(t)` | All t. E_1 = eligibility over input. |
| **Dend**             | `γ · σ'_1(t) · h'_1(t′_1) · (dμ_1/dw)(t) · e_1(t)` | σ', e_1 at t; h' and dμ/dw at **t′_1** (plateau of this layer). |

So: one path “readout → soma” and one “readout → dend”, both with the same effective error; soma uses E(t), dend uses h'(t′) and dμ/dw(t′) at the **same** layer’s plateau time.

---

## 2-Layer (input → L1 “extra” → L2 “hidden” → readout)

**Effective error for L2 (hidden):**

- `e_2(t) = σ'_readout(t) · global_errors · w_readout`  →  (T, n_2).

**L2 gradients** (same as 1-layer hidden):

| Path (readout→L2) | Time used |
|-------------------|-----------|
| **Soma**          | Sum over **all t**: σ'_2(t), E_2(t), e_2(t). |
| **Dend**          | Sum over **all t**; inside the sum, h'_2 and dμ_2/dw at **t′_2** (L2’s plateau). |

**Effective error for L1 (extra)** comes from L2 via two routes: through L2’s **soma** (all t) and through L2’s **dendrite** (at L2’s t′). So we get four path combinations for L1:

| Path (readout→L2→L1) | L2 side      | L1 side | Times used |
|----------------------|-------------|---------|------------|
| **Soma–Soma**        | Through w_soma_L2 | L1 soma (E_1) | **All t**: σ'_2(t), σ'_1(t), e_2(t); E_1(t). |
| **Soma–Dend**       | Through w_soma_L2 | L1 dend (dμ_1/dw, h'_1) | **All t**: σ'_2(t), σ'_1(t)·h'_1(t), e_2(t); dμ_1/dw(t) (with t′_1 logic). |
| **Dend–Soma**       | Through w_dend_L2 at **t′_2** | L1 soma (E_1) | L2 at **t′_2[t,i]** (per t, L2 neuron i): σ'_2, h'_2, e_2. L1: **σ'_1(t′_2)**, **E_1(t′_2)**. |
| **Dend–Dend**       | Through w_dend_L2 at **t′_2** | L1 dend | L2 at **t′_2**. L1: **σ'_1(t′_2)**, **h'_1(t′_2)**, **dμ_1/dw(t′_2)**. |

So when the gradient goes through the **upper** layer’s dendrite, the **lower** layer’s quantities are evaluated at the **upper** layer’s plateau time t′_upper (when the upper dendrite “saw” the lower layer’s activity).

---

## 3-Layer (input → L1 → L2 → L3 → readout)

### Effective errors

- **L3**: `e_3(t) = σ'_readout(t) · global_errors · w_readout`  →  (T, n_3).
- **L2**: Obtained by backprop through L3 (soma path + dend path at t′_3). So:
  - From L3 **soma** (all t): coefficient ∝ w_soma_L3^T · (σ'_L3(t) · e_3(t)) · σ'_L2(t).
  - From L3 **dend** at **t′_3**: coefficient ∝ w_dend_L3 · σ'_L3 · h'_L3 · e_3 evaluated at **t′_3**; then times σ'_L2(t′_3).
- **L1**: Obtained by backprop through L2 (soma path + dend path at t′_2), using the effective error e_2(t) (and e_2(t′_2) for the dend path).

### L3 gradients (top 2comp)

Same as “hidden” in 2-layer:

| Path | Time used |
|------|-----------|
| **Soma** | Sum over **all t**: σ'_L3(t), E_L3(t), e_3(t). |
| **Dend** | Sum over **all t**; h'_L3 and dμ_L3/dw at **t′_3**. |

### L2 gradients (middle 2comp)

Effective error e_2 has two parts: from L3 soma (all t) and from L3 dend (at t′_3). So again four path combinations:

| Path (L3→L2)   | L3 side        | L2 side | Times |
|----------------|----------------|---------|--------|
| **Soma–Soma**  | w_soma_L3, σ'_L3(t), e_3(t) | σ'_L2(t), E_L2(t) | **All t**. |
| **Soma–Dend**  | w_soma_L3, σ'_L3(t), e_3(t) | σ'_L2(t), h'_L2(t′_2), dμ_L2/dw | **All t** for L3; L2 dend at **t′_2**. |
| **Dend–Soma**  | w_dend_L3 at **t′_3** (σ'_L3, h'_L3, e_3) | σ'_L2(**t′_3**), E_L2(**t′_3**) | L3 and L2 evaluated at **t′_3** (when L3 dend saw L2). |
| **Dend–Dend**  | w_dend_L3 at **t′_3** | σ'_L2(**t′_3**), h'_L2(**t′_3**), dμ_L2/dw(**t′_3**) | All at **t′_3**. |

### L1 gradients (bottom 2comp)

Effective error e_1 comes from L2 (soma path all t + dend path at t′_2). Same four path types:

| Path (L2→L1)   | L2 side        | L1 side | Times |
|----------------|----------------|---------|--------|
| **Soma–Soma**  | w_soma_L2, σ'_L2(t), e_2(t) | σ'_L1(t), E_L1(t) | **All t**. |
| **Soma–Dend**  | w_soma_L2, σ'_L2(t), e_2(t) | σ'_L1(t), h'_L1(t′_1), dμ_L1/dw | **All t** for L2; L1 dend at **t′_1**. |
| **Dend–Soma**  | w_dend_L2 at **t′_2** | σ'_L1(**t′_2**), E_L1(**t′_2**) | At **t′_2**. |
| **Dend–Dend**  | w_dend_L2 at **t′_2** | σ'_L1(**t′_2**), h'_L1(**t′_2**), dμ_L1/dw(**t′_2**) | At **t′_2**. |

---

## Summary: When is each quantity at “all t” vs “at t′”?

- **E (soma eligibility)**  
  - Used in **soma** gradient of that layer: sum over **all t** (E(t)).  
  - When backprop **from** the layer above: if the path goes through the **upper** layer’s **dendrite**, the **lower** layer’s E is evaluated at the **upper** layer’s **t′** (e.g. E_L1(t′_2)).

- **dμ/dw, h′ (dendritic)**  
  - In the **same** layer’s dend gradient: always at that layer’s **t′** (and often stored/used as (T, n, n_inputs) with t′ logic inside).  
  - When backprop from the layer above through the upper **dendrite**: the **lower** layer’s σ', h', and dμ/dw are evaluated at the **upper** layer’s **t′** (e.g. t′_2 for L1 when L2 is above).

- **σ' (soma surrogate)**  
  - In the same layer: at **current t** (or at upper t′ when that layer is the “upper” in a backprop step).

So the only “special” times are the **plateau times t′_ℓ** of each layer ℓ. Any gradient path that goes through a **dendrite** of layer ℓ uses that layer’s t′_ℓ; and when that path continues to the layer below, the **lower** layer’s activations/surrogates/eligibilities are evaluated at that **same** t′_ℓ.

**Rule of thumb:** If the path goes through “layer-above’s **dendrite**”, then the **layer-below** is evaluated at the **layer-above’s t′** (the time when that dendrite was at plateau and received from the layer below). So: L3 dend → L2 at t′_3; L2 dend → L1 at t′_2.

---

## Path enumeration (3-layer, readout → L3 → L2 → L1)

Viewed as “readout → … → target weight”:

1. **Readout → L3 soma** → all t.  
2. **Readout → L3 dend** → L3 at t′_3.  
3. **Readout → L3 soma → L2 soma** → all t.  
4. **Readout → L3 soma → L2 dend** → all t for L3; L2 dend at t′_2.  
5. **Readout → L3 dend → L2 soma** → at t′_3.  
6. **Readout → L3 dend → L2 dend** → at t′_3.  
7. **Readout → L3 soma → L2 soma → L1 soma** → all t.  
8. **Readout → L3 soma → L2 soma → L1 dend** → all t for L3,L2; L1 dend at t′_1.  
9. **Readout → L3 soma → L2 dend → L1 soma** → all t for L3; L2 at t′_2; L1 at t′_2.  
10. **Readout → L3 soma → L2 dend → L1 dend** → all t for L3; L2 at t′_2; L1 at t′_2.  
11. **Readout → L3 dend → L2 soma → L1 soma** → at t′_3 (L3); L2 and L1 at t′_3.  
12. **Readout → L3 dend → L2 soma → L1 dend** → at t′_3 for L3,L2; L1 soma/dend at t′_3, L1 dendritic h′/dμ at t′_1.  
13. **Readout → L3 dend → L2 dend → L1 soma** → L3 at t′_3; L2 dend at t′_2 (when L2’s dend saw L1); **L1 at t′_2**.  
14. **Readout → L3 dend → L2 dend → L1 dend** → L3 at t′_3; L2 dend at t′_2; **L1 at t′_2**.  

So in 3-layer you get **soma–soma–soma**, **soma–soma–dend**, **soma–dend–soma**, **soma–dend–dend**, **dend–soma–soma**, **dend–soma–dend**, **dend–dend–soma**, **dend–dend–dend**, each with the correct t / t′ as above.

---

## Tree: three hidden layers (L1 → L2 → L3 → readout)

Each node is “effective error into this layer”; edges are “through that layer’s **soma**” or “through that layer’s **dendrite**”. Times are read **outer → inner**: e.g. **t′₁ at t′₂ at t** means at wall-clock **t**, at parent time **t′₂**, we use the value from **t′₁** (layer’s own plateau).

**Time inheritance:**

- **Soma edge**: pass down **all t** (time context unchanged).
- **Dendrite edge**: pass down **t′_{ℓ+1} at t**; everything below is evaluated at the parent’s t′. In the **final dendritic gradient** we use **t′_layer at t′_parent at t** (value from that layer’s plateau, read at parent’s time).

---

### Tree with times (three hidden layers)

```
                              [global error δ]
                                      │
                  ┌───────────────────┴───────────────────┐
                  │                                       │
            L3 soma                                L3 dend
            time: t                                time: t′₃ at t
                  │                                       │
          L3 grad: σ′₃,E₃ at t                   L3 grad: σ′₃,h′₃,dμ₃/dw at t′₃
          dend: h′₃,dμ₃ at t′₃ at t              (all at t′₃)
                  │                                       │
        to L2 at t                    to L2 at t′₃ at t
                  │                                       │
        ┌─────────┴─────────┐                 ┌───────────┴───────────┐
        │                   │                 │                       │
   L2 soma             L2 dend           L2 soma                 L2 dend
   time: t             time: t           time: t′₃ at t           time: t′₃ at t
        │                   │                 │                       │
   L2: σ′₂,E₂ at t    L2 dend:          L2: σ′₂,E₂ at t′₃      L2 dend:
   L2 dend: t′₂ at t  t′₂ at t               L2 dend: t′₂ at t′₃   t′₂ at t′₃ at t
        │                   │                 │                       │
   to L1 at t          to L1 at t         to L1 at t′₃          to L1 at t′₂
        │                   │                 │                       │
   ┌────┴────┐         ┌────┴────┐       ┌────┴────┐             ┌────┴────┐
   │         │         │         │       │         │             │         │
 L1 s     L1 d       L1 s     L1 d     L1 s     L1 d           L1 s     L1 d
   t    t′₁ at t       t    t′₁ at t   t′₃   t′₁ at t′₃       t′₂   t′₁ at t′₂ at t
```

**Leaf legend (L1):**

- **L1 soma**: σ′₁, E₁ at the time shown.
- **L1 dend**: dendritic gradient uses **t′₁ at (parent time) at t** — i.e. value from L1’s plateau (t′₁), read at the parent’s time (t or t′₂ or t′₃), outer t.

---

### Compact tree (times only)

L2 **soma** is always evaluated at the inherited time: **t** when we came through L3 soma (S–S), **t′₃ at t** when we came through L3 dend (D–S). L2 **dend** uses that same inherited time, with L2’s own dendritic terms at t′₂ in that context.

```
                         [δ]
                          │
              ┌───────────┴───────────┐
              │                       │
         L3 S                      L3 D
          t                      t′₃ at t
              │                       │
        ┌─────┴─────┐           ┌─────┴─────┐
        │           │           │           │
   L2 S       L2 D         L2 S       L2 D
    t          t         t′₃ at t   t′₃ at t
             t′₂ at t              t′₂ at t′₃ at t
        │           │           │           │
   ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐
   │         │ │         │ │         │ │         │
 L1 s   L1 d L1 s   L1 d L1 s   L1 d L1 s   L1 d
  t   t′₁@t   t   t′₁@t  t′₃  t′₁@t′₃ t′₂  t′₁@t′₂@t
```

- **Left (L3 S)**: L2 S at **t**, L2 D at **t** with dend **t′₂ at t**.  
- **Right (L3 D)**: L2 S at **t′₃ at t**, L2 D at **t′₃ at t** with dend **t′₂ at t′₃ at t**.

Notation: **t′₁@t** = t′₁ at t (L1 dend uses own t′₁ in context t); **t′₁@t′₂@t** = t′₁ at t′₂ at t (L1 dend: value from t′₁, read at t′₂, outer t).

---

### Summary table (all 8 paths)

| Path (R → L3 → L2 → L1) | L3 time   | L2 time      | L1 time (soma / dend gradient) | L1 dend gradient (h′, dμ/dw) |
|--------------------------|-----------|--------------|---------------------------------|------------------------------|
| S → S → S                | t         | t            | t                               | **t′₁ at t**                 |
| S → S → D                | t         | t            | t                               | **t′₁ at t**                 |
| S → D → S                | t         | t; dend t′₂  | **t′₂**                         | —                            |
| S → D → D                | t         | t; dend t′₂  | **t′₂** (σ′,E at t′₂)           | **t′₁ at t′₂ at t**          |
| D → S → S                | t′₃ at t  | t′₃          | t′₃                             | —                            |
| D → S → D                | t′₃ at t  | t′₃          | t′₃ (σ′,E at t′₃)               | **t′₁ at t′₃ at t**          |
| D → D → S                | t′₃ at t  | t′₃; dend t′₂ | **t′₂**                         | —                            |
| D → D → D                | t′₃ at t  | t′₃; dend t′₂ | **t′₂** (σ′,E at t′₂)           | **t′₁ at t′₂ at t**          |

So: **dendrite path** locks the layer below to the parent’s **t′**; in the **final dendritic gradient** we always use **t′_layer at t′_parent at t** (value from that layer’s plateau, read at parent’s time, outer t).
