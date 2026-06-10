# CAL-L4 — Meta-Inference Layer

**Part of:** [CAL — Cognitive Abstraction Layers](https://github.com/jpcpol/Cognitive-Abstraction-Layer-CAL) — the research starts at CAL; L4 is its Meta-Inference Layer.  
**Author:** Juan Pablo Chancay · Aural Syncro  
**Status:** gate-C CLOSED · **S5 mechanism confirmed on MI300X** (κ decouples, D(n)→52.8×) · **L4-B0 NO-GO** (residual 81% non-linear → dual is terminal) · condition (c) governance accuracy is the sole open gate (2026-06)  
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
| **(b)** Cost comparison M(V) vs flat-context O(n²) | ✅ **MET** — AMD baseline measured: n^1.90–1.91, R²≈0.997 |
| **(c)** Governance accuracy under both approaches | **OPEN — the sole remaining gate** (RCT-bound; corpus from L2) |

Conditions (a)+(b) are met and the cost **mechanism is confirmed on hardware (S5)**: combining the two laws, the decoupling ratio D(n)=Cost_flat(n)/Cost_gov reaches **52.8× at seqLen 4096** while the governance-state cost stays bounded by κ(V) and independent of n (§5.4 of the paper). This shows the *mechanism* (coupling vs decoupling), not a production-scale speedup or an accuracy claim. **The hypothesis is not yet proven:** condition (c) — that M(V) decisions match or beat flat-context — is the only gate left.

### L4-A — the operator that closed gate-C

L3 delivered C as a **dual representation** `V = (V_Tucker, G_pruned)`: a Tucker core (κ=1296, the cost object) plus a pruned causal graph (|E|=2, the governance object). A reference `M_ref` was shown to operate on this pair at (κ+|E|) cost **without re-running causal discovery** (checks C1/C2/C5 pass). Honest limitation, declared: **κ does not reflect the prune** — whether the two can be collapsed into a single volume V′ (L4-B) was gated on characterizing L3's residual.

### L4-B0 — residual characterized → the dual is terminal (NO-GO)

With S5 frozen, the residual `ΔU≈0.138` was characterized (`PRE_REGISTRATION_L4B0_RESIDUAL.md`). Attribution in the continuous signed Φ-space: only **19% is linear-edge-representable** (magnitude + sign), while **81% is non-linearity** a single linear volume cannot carry (lag>1 structure ≈0%). Decision `share_lin=0.193` → **NO-GO**: L4-B (single linear V′) is **not opened**; the dual `(V_Tucker, G_pruned)` is the **terminal** representation at this rank. A clean negative result — the second time *causality ≻ reconstruction* prevents a collapse (after L3's S3-bis). See [`experiments/efficiency_hypothesis/`](experiments/efficiency_hypothesis/) (paper §5.5).

---

## Representational Convergence Conjecture — RCC (§6.4)

> The optimal governance state is extractable directly from attention activations during pre-fill — without a second LLM-QA pass.

AMD-Instinct's `probe_mfma_mapping.hip` already characterized the lane↔output mapping of `v_mfma_f32_16x16x16f16` — the low-level register access this would require. The RCC remains a long-horizon conjecture, now **bounded by L4-B0**: since 81% of the governance state's residual causal content is non-linear, no single *low-rank linear* state is simultaneously κ-minimal and causally complete here — any convergence, if it exists, is not a linear folding at this rank.

---

## AMD-Instinct Collaboration

`fa_dme` (Flash Attention with DME async, validated on MI300X at D=64, 82.4 µs, max_err < 0.0001 — an 18% end-to-end speedup) has a dual role in L4. (Throughput note: the MFMA-tile kernel reaches 10.45 TFLOPS at D=128 vs 6.19 at D=64; the baseline sweep uses the D=128 LLM-realistic path.)

| Role | When | Description |
|------|------|-------------|
| **Rol 1 — Baseline + S5 contrast** | ✅ Done | Flat-context O(n²) curve (n^1.90–1.91, R²≈0.997); S5 κ vs n² contrast run & frozen (D(n)→52.8× @4k) |
| **Research→application arc (2-A/2-C)** | ✅ Done | Research kernel runs a full LLM (Qwen2.5-0.5B, 24/24 layers, top-1 preserved — 2-A PASS); gap vs production SDPA 8.5–15× measured honestly (2-C) |
| **Rol 2 — Proxy M(V)** | Long-horizon | Kernel from which the RCC would extract V during pre-fill — bounded by L4-B0 (no single linear convergent state) |

**Scope discipline:**
- ✅ O(n²) curve measured on MI300X; S5 mechanism contrast run and frozen
- ✅ C characterized on synthetic + L4-A operator delivers κ(V); L4-B0 closed the representation question (dual terminal)
- ❌ Do NOT claim the L4 Efficiency Hypothesis proven until condition (c) (governance accuracy) is run — the mechanism is shown, accuracy is not
- ❌ Do NOT read 2-C as a performance win — production wins 8.5–15×; the deliverable is the *gap* and what is portable upstream
- C validated on synthetic is *preliminary evidence* — same epistemic status as n=40 in L2

---

## Roadmap with Gates

| Task | Owner | Status | Blocker |
|------|-------|--------|---------|
| Baseline flat-context O(n²) (`fa_robust` seqLen sweep 512→4k) | AMD | ✅ Done — n^1.90, R²=0.996 | — |
| Confirm quadratic regime (log-log, exponent ≈ 2) | AMD | ✅ Done — confirmed | — |
| Composition operator C validated (L3) | L3 | ✅ Done — L3 closed, κ(V)=1296 | — |
| L4-A operator: dual V operable at κ-bounded cost | L4 | ✅ Done — C1/C2/C5 pass | — |
| Citable baseline note (`NOTE_flat_context_baseline.md`) | AMD | ✅ Done | — |
| κ vs n² cost contrast — mechanism (S5, seqLen 512→4k, D=128) | AMD | ✅ Done — D(n)→52.8×; frozen | — |
| Research→application arc (2-A full LLM, 2-C vs production) | AMD | ✅ Done — 2-A PASS; 2-C gap measured | — |
| L4-B0 residual characterization | L3/L4 | ✅ Done — NO-GO (81% non-linear); dual terminal | — |
| L4-B (single-V via inverse projection) | L4 | ❌ Not opened — refuted by L4-B0 | — |
| **L4 Efficiency Hypothesis — full test (condition c)** | Both | **Open — sole gate; needs RCT governance corpus** | condition (c) |

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
