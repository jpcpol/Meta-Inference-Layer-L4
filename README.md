# CAL-L4 — Meta-Inference Layer

**Part of:** [CAL — Cognitive Abstraction Layers](https://github.com/jpcpol/Cognitive-Abstraction-Layer-CAL)  
**Author:** Juan Pablo Chancay · Aural Syncro  
**Status:** In development — O(n²) baseline active; M(V) deferred until gate-C  
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
| **(a)** C defined + κ(V) concrete (L3 gate) | Pending — blocked on L3 |
| **(b)** Cost comparison M(V) vs flat-context O(n²) | AMD baseline active |
| **(c)** Governance accuracy under both approaches | L2 corpus provides this |

**This hypothesis cannot be claimed as proven until all three conditions are met.**

---

## Representational Convergence Conjecture — RCC (§6.4)

> The optimal governance state is extractable directly from attention activations during pre-fill — without a second LLM-QA pass.

AMD-Instinct's `probe_mfma_mapping.hip` already characterized the lane↔output mapping of `v_mfma_f32_16x16x16f16` — the low-level register access this would require. This makes the RCC empirically approachable once C exists.

---

## AMD-Instinct Collaboration

`fa_dme` (Flash Attention with DME async, validated on MI300X at D=128, max_err < 0.0001) has a dual role in L4:

| Role | When | Description |
|------|------|-------------|
| **Rol 1 — Baseline** | Now | Measures the flat-context O(n²) attention cost curve that condition (b) requires |
| **Rol 2 — Proxy M(V)** | Post gate-C | Kernel from which the RCC extracts governance signal V during pre-fill |

**Scope discipline:**
- ✅ Measure real O(n²) curve on MI300X (seqLen sweep 512→4k, log-log fit, confirm exponent ≈ 2)
- ✅ Publish citable note: "flat-context attention cost on MI300X as CAL-L4 baseline"
- ❌ Do NOT claim L4 Efficiency Hypothesis proven without C + M(V) implemented
- Tucker C validated on synthetic is *preliminary evidence* — same epistemic status as n=40 in L2

---

## Roadmap with Gates

| Task | Owner | Status | Blocker |
|------|-------|--------|---------|
| Baseline flat-context O(n²) (`fa_robust` seqLen sweep 512→4k) | AMD | Ready (cold) | — |
| Confirm quadratic regime (log-log, exponent ≈ 2) | AMD | Next VM session | — |
| Publish citable baseline note | AMD | Pending | log-log result |
| Composition operator C validated (L3) | L3 | Pending | L3 gate |
| Kernel proxy M(V): O(n²) flat vs O(κ) | AMD | Deferred | C / L3 |
| L4 Efficiency Hypothesis — synthetic test | Both | Deferred | M(V) + C |

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
| [L3 — Tensor Volume](https://github.com/jpcpol/Tensor-Volume-Layer-L3) | Provides V and κ(V) — gate for Roles 1→2 transition |
