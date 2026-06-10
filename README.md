# CAL-L4 — Meta-Inference Layer

**Part of:** [CAL — Cognitive Abstraction Layers](https://github.com/jpcpol/Cognitive-Abstraction-Layer-CAL) — the research starts at CAL; L4 is its Meta-Inference Layer.  
**Author:** Juan Pablo Chancay · Aural Syncro  
**Status:** gate-C CLOSED (2026-06) — conditions (a)+(b) met; L4-A operator validated; at-scale κ vs n² contrast active  
**Target venue:** NeurIPS / ICML  
**Collaboration:** AMD-Instinct Labs (`fa_dme` on MI300X)  
**License:** CC BY-NC 4.0 (docs) · AGPL-3.0 (src)

---

## What is L4?

L4 is the **Meta-Inference Layer** of the CAL architecture. It defines the inference function that maps a compressed tensor volume (L3 output) to actionable governance decisions — without requiring human working memory as a substrate.

```
L4:  M(V) → {decisions, predictions, adaptations}
```

Where:
- `V` — tensor volume produced by the L3 composition operator C
- `M(V)` — meta-inference function; operates on compressed structure, not raw artifacts
- Output — governance signals: deploy/block decisions, system-wide predictions, policy adaptations

---

## L4 Efficiency Hypothesis (§6.2 CAL pre-paper)

> There exists an inference architecture such that the cost of M(V) scales with κ(V) — the structural complexity of V (effective rank, attractor entropy, causal graph size) — where κ(V) grows **significantly slower than O(n²)** in n (raw artifact count at L0).

Proving this requires three simultaneous conditions:

| Condition | Status |
|-----------|--------|
| **(a)** C defined + κ(V) concrete (L3 gate) | ✅ **MET** — L3 closed; C = C_causal ∘ C_compress; κ(V)=1296 (195.6×) |
| **(b)** Cost comparison M(V) vs flat-context O(n²) | ✅ **MET** — AMD baseline measured: n^1.90, R²=0.996 |
| **(c)** Governance accuracy under both approaches | Pending — remaining gate (RCT-adjacent) |

With (a) and (b) met, the **κ vs n² cost contrast is now runnable** — the L4-A operator delivers the `(κ, G_pruned, U≈0.86)` object the contrast consumes. **The hypothesis is not yet proven:** condition (c) — that M(V) decisions match or beat flat-context — remains.

### L4-A — the operator that closed gate-C

L3 delivered C as a **dual representation** `V = (V_Tucker, G_pruned)`: a Tucker core (κ=1296, the cost object) plus a pruned causal graph (|E|=2, the governance object). A reference `M_ref` was shown to operate on this pair at (κ+|E|) cost **without re-running causal discovery** (checks C1/C2/C5 pass). Honest limitation, declared: **κ does not reflect the prune** — collapsing the two into a single volume V′ is **L4-B**, future work gated on characterizing L3's residual 25%. See [`experiments/efficiency_hypothesis/`](experiments/efficiency_hypothesis/).

---

## Representational Convergence Conjecture — RCC (§6.4)

> The optimal governance state is extractable directly from attention activations during pre-fill — without a second LLM-QA pass.

AMD-Instinct's `probe_mfma_mapping.hip` already characterized the lane↔output mapping of `v_mfma_f32_16x16x16f16` — the low-level register access this would require. With C now defined at L3 (gate-C closed), the RCC is empirically approachable from the hardware side.

---

## AMD-Instinct Collaboration

`fa_dme` (Flash Attention with DME async, validated on MI300X at D=64, 82.4 µs, max_err < 0.0001 — an 18% end-to-end speedup) has a dual role in L4. (Throughput note: the MFMA-tile kernel reaches 10.45 TFLOPS at D=128 vs 6.19 at D=64; the baseline sweep uses the D=128 LLM-realistic path.)

| Role | When | Description |
|------|------|-------------|
| **Rol 1 — Baseline** | ✅ Done | Measured the flat-context O(n²) curve (n^1.90, R²=0.996, confirmed quadratic) |
| **Rol 2 — Proxy M(V)** | Active (gate-C closed) | Kernel from which the RCC extracts governance signal V during pre-fill |

**Scope discipline:**
- ✅ O(n²) curve measured on MI300X (seqLen sweep, log-log fit, exponent ≈ 2 confirmed)
- ✅ C characterized on synthetic + L4-A operator delivers κ(V) — gate-C closed
- ❌ Do NOT claim the L4 Efficiency Hypothesis proven until condition (c) and the at-scale contrast are run
- C validated on synthetic is *preliminary evidence* — same epistemic status as n=40 in L2

---

## Roadmap with Gates

| Task | Owner | Status | Blocker |
|------|-------|--------|---------|
| Baseline flat-context O(n²) (`fa_robust` seqLen sweep 512→4k) | AMD | ✅ Done — n^1.90, R²=0.996 | — |
| Confirm quadratic regime (log-log, exponent ≈ 2) | AMD | ✅ Done — confirmed | — |
| Composition operator C validated (L3) | L3 | ✅ Done — L3 closed, κ(V)=1296 | — |
| L4-A operator: dual V operable at κ-bounded cost | L4 | ✅ Done — C1/C2/C5 pass | — |
| Publish citable baseline note | AMD | Pending | (log-log result ready) |
| At-scale κ vs n² contrast (seqLen 512→4k, D=128) | AMD | Active — unblocked | — |
| L4 Efficiency Hypothesis — full test (condition c) | Both | Pending — (a),(b) met | condition (c) |
| L4-B (single-V via inverse projection) | L4 | Frozen — after AMD contrast freeze | residual-25% characterization |

---

## Repository Structure

```
L4/
├── README.md
├── paper/                  ← L4 paper (in development)
├── src/
│   └── meta_inference/     ← M(V) implementation (post gate-C)
├── benchmarks/
│   ├── baseline_quadratic/ ← O(n²) empirical curve from AMD-Instinct
│   └── efficiency_contrast/ ← O(n²) flat vs O(κ) comparison
└── experiments/
    └── efficiency_hypothesis/ ← L4 Efficiency Hyp. tests
```

---

## Related Repos

| Repo | Role |
|------|------|
| [CAL](https://github.com/jpcpol/Cognitive-Abstraction-Layer-CAL) | Framework root — pre-paper, architecture |
| [L2 — TCO](https://github.com/jpcpol/TENSOR-BASED-COGNITIVE-OVERSIGHT-TCO) | Provides governance accuracy baseline (condition c) |
| [L3 — Tensor Volume](https://github.com/jpcpol/Tensor-Volume-Layer-L3) | Provides V and κ(V)=1296 (gate-C closed); causal conservation = sparsity preservation |
