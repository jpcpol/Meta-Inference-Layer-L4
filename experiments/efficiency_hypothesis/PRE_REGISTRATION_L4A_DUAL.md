# L4-A — Dual-Representation Operational Baseline (Pre-Registration)

**Commit this document BEFORE writing or running any L4-A code.** Its commit SHA is
the pre-registration timestamp. L4-A formalizes the **Architecture A** operator
(dual representation) as the operational baseline of L4 — it closes the L3→L4 bridge
enough for AMD-Instinct to run the cost contrast, **without** solving the inverse
graph→tensor projection (which is L4-B, a research hypothesis, not this experiment).

- **Date:** 2026-06-10
- **Reuses:** S1-bis corpus (t=48), the Tucker operator (`tucker_operator.py`), the
  TCI metric U, the Form-1 structural mask (`run_form1.py`), the S3-bis PCMCI
  machinery. The AMD O(n²) flat-context baseline is already measured (n^1.90,
  R²=0.9964). No new corpus, no new estimator.
- **Builds on:** Form 1 (L3, commit 35d4cb0) confirmed the structural thesis —
  pruning the recovered causal graph to the raw support lifts U 0.441→0.862 (75% of
  the headroom) with no ground truth (raw↔GT gap = 0.000).

---

## 0. Where we are going, what we need, how we build it (the analysis)

### 0.1 Destination
L4 consumes `V = C(T)` and runs `M(V)` at cost `O(κ(V)) ≪ O(n²)`. For that claim to
be non-trivial, V must be simultaneously **causally clean** (free of Tucker's
spurious edges — what L3 proved matters) and **structurally compact** (low κ(V) —
what makes inference cheap). The RCC north star adds: the *same* V serves human
governance and generator compute.

### 0.2 The dissociation we found (the reason this prereg is split A/B)
A design audit established that **κ(V) = rank(Tucker core) = 1296 is fixed by
C_compress, BEFORE any causal pruning**, and the causal graph is a *derived*
property of the reconstruction, not stored in V. Form 1 therefore produced a new
**graph** `Ĝ_pruned`, not a new **volume**: it measured `U(Ĝ_pruned)`, not
`U(V̂_pruned)`. Pruning improved U but did **not** lower κ.

Consequence — two architectures, not interchangeable:

| | **A (this prereg)** | **B (future, L4-B)** |
|---|---|---|
| V | `(V_Tucker, G_pruned)` — dual | `V'` with pruning folded into the volume — single |
| κ(V) | κ(V_Tucker)=1296, prune does not lower it | reflects causal sparsity |
| M(V) | reads core + pruned graph (graph as metadata) | operates on V' directly, already clean |
| RCC | dual representation (operational ≠ explanatory) | single representation (RCC-coherent) |
| cost to build | ~free (Form 1 already does it) | needs inverse graph→tensor projection (a new math problem, weeks) |
| status | **EXPERIMENT** | **RESEARCH HYPOTHESIS** |

### 0.3 Why A first (methodological)
B requires an **inverse projection** (pruned graph → tensor), which depends on first
characterizing where the residual 25% lives (weights / non-linearity / cycles /
second-order). That characterization is not done. Designing B now would repeat the
pattern S3-bis punished: **build the operator before understanding the object.**
A is validated indirectly by Form 1, introduces no new math, and lets AMD run now.
**CAL's final goal is B; the next experiment is A.**

---

## 1. The operator under test (fixed a priori)

```
C_compress :  T → V_Tucker = Tucker(stack({T⁽ˢ⁾}), rank=(8,3,3,3,6))    # κ(V)=1296, validated
C_causal_A :  V_Tucker → G_pruned                                        # Form-1 prune, formalized
C_A        :  T → V = (V_Tucker, G_pruned)                               # the DUAL volume
```

`C_causal_A` is the Form-1 structural mask formalized as a **reproducible
operator** (not a measurement): reconstruct V̂ from V_Tucker, recover the per-graph
PCMCI edge set, intersect it with the **raw causal support** (self-referential, no
ground truth), and emit the pruned graph `G_pruned` alongside the unchanged core.
The output volume carries both objects.

This is the dual representation. It is explicitly NOT the single-V operator (B).

## 2. What L4-A delivers to L4 / AMD (fixed a priori)

For each graph G1/G2/G3 over the n=30 sessions of S1-bis, the operator emits:

- **κ(V_Tucker)** — the effective rank (=1296 at this rank), the L4 Efficiency
  Hypothesis cost quantity. Reported per the L4 paper §3.4 candidate 1
  (rank of Tucker core).
- **G_pruned** — the clean causal graph (|E|≈2, the true support), the object a
  human governs with.
- **U(G_pruned)** — the causal-conservation score (≈0.862, known from Form 1),
  reported as the SID(L2→L3) surrogate.
- **|E|, R, C, S** — the Ω₀ invariants on the pruned graph (the governance-relevant
  structure).

These are exactly the inputs AMD needs to run the cost contrast: κ(V) for the
O(κ) side, the raw artifact count n for the O(n²) side (baseline already measured).

## 3. The question L4-A answers (fixed a priori)

The success criterion is NOT recovering more U (75% is already recovered). It is
**operability of the dual representation** (consultant, adopted):

> Can a meta-inference function M operate on the pair `(V_Tucker, G_pruned)` at cost
> close to κ(V_Tucker) while conserving Ω₀?

This is **not** the efficiency-hypothesis test itself (the full O(κ) vs O(n²)
contrast needs AMD's hardware run, condition (b)+(c)). L4-A is the **operator that
makes that test runnable** AND a software-level demonstration that M can *consume*
the dual object at κ-bounded work. Concretely: a reference `M_ref(V_A)` that reads
the κ(V_Tucker)-sized core and the |E|≈2 pruned graph, performs the four L4
capabilities (causal discovery / drift / conflict / policy) **without re-running
PCMCI on the full reconstruction**, and whose work scales with κ + |E|, not with the
n² raw-artifact pairs. If M must re-derive the graph from scratch, the dual
representation failed its purpose (the prune would not be *carried*, only *measured*).

## 4. Confirmatory checks (fixed a priori)

- **C1 — Reproducibility of the prune.** Running `C_causal_A` twice on the same
  corpus yields the identical `G_pruned` (deterministic given seeds). PASS if
  byte-identical edge sets across two runs.
- **C2 — Governance sufficiency (Ω₀ held).** On `G_pruned`: U ≥ 0.80, C ≥ 0.95,
  S = 1.0 (the Form-1 safeguard, re-checked as an operator property not a one-off
  measurement). PASS if all three hold per graph.
- **C3 — κ exposure.** κ(V_Tucker) is emitted and equals the Tucker core size
  (=1296 at rank (8,3,3,3,6)); the compression ratio |T|/κ is reported. This is a
  reporting check, not a gate.
- **C4 — Dual-representation honesty (the declared limitation).** The prereg states
  explicitly that κ does NOT reflect the prune (κ is the Tucker core's, the prune
  lives in G_pruned). L4-A does not claim a κ reduction from pruning. Claiming
  otherwise would be the A-vs-B confusion this split exists to prevent.
- **C5 — Operability at κ-bounded cost (PRIMARY, the consultant's success
  criterion).** A reference `M_ref(V_A)` consumes the dual pair and performs the
  four L4 capabilities reading ONLY the κ(V_Tucker)-sized core + the |E|-edge pruned
  graph — it must NOT re-run PCMCI on the full reconstruction. PASS if (i) M_ref's
  work scales with (κ + |E|), not with the n² raw-artifact pairs, and (ii) the Ω₀ it
  reports off `G_pruned` matches C2's values (the carried graph is sufficient for the
  inference). This is what makes the representation *operable*, not merely
  *well-formed* — the actual point of L4-A.

## 5. Verdict logic (fixed a priori)

| C1 | C2 | C5 | Verdict |
|----|----|----|---------|
| PASS | PASS | PASS | **L4-A validated: the dual representation is operable at κ-bounded cost.** L3 fully closed; AMD can run the cost contrast on (κ(V_Tucker), n); RCC gets an operational representation. L4-B (single-V via inverse projection) becomes the next research question, gated on characterizing the residual. |
| PASS | PASS | FAIL | M cannot operate on the dual pair without re-deriving the graph — the prune is *measured* but not *carried*. The dual representation is insufficient; this itself motivates L4-B sooner. |
| PASS | FAIL | — | The pruned graph does NOT preserve Ω₀ as an operator (vs as a one-off) — investigate determinism/aggregation before handing to AMD. |
| FAIL | — | — | The prune is not reproducible as an operator — fix before any downstream use. |

## 6. What L4-A explicitly does NOT do

- **No inverse projection.** `V'` (pruning folded into the volume) is NOT built.
  That is L4-B, and it is gated on first characterizing the residual 25% (a separate
  prereg: does it live in edge weights, non-linearity, cycles, or second-order
  structure?).
- **No κ-reduction claim.** κ stays the Tucker core's; the dual representation is
  honest about carrying two objects.
- **No governance-accuracy claim.** Whether M(V) decisions beat flat-context belongs
  to condition (c) / the RCT, not here.
- **No new corpus.** Drift/conflict (Q_L3.2B) and any S-Ω corpus stay out.

## 7. Fixed parameters (summary)

- Operator: `C_A: T → (V_Tucker, G_pruned)`, Tucker rank (8,3,3,3,6), prune to raw
  support (self-referential), PCMCI ParCorr tau=1 pc_alpha=0.01 (S3-bis/TCI machinery).
- Delivered triple: (κ(V_Tucker)=1296, G_pruned, U≈0.862) + Ω₀(G_pruned).
- C1 reproducibility (byte-identical), C2 Ω₀ sufficiency (U≥0.80 ∧ C≥0.95 ∧ S=1.0),
  C3 κ exposure (report), C4 honesty (κ ≠ prune, declared), **C5 operability
  (PRIMARY): M_ref reads only core+pruned-graph at (κ+|E|) cost, no PCMCI re-run.**
- Corpus S1-bis (t=48). AMD O(n²) baseline already measured (n^1.90, R²=0.9964).
- Architecture A only. B is a separately pre-registered research hypothesis.

## 8. Execution sequence (consultant, fixed)

1. Run L4-A exactly as pre-registered.
2. Deliver the dual object `(V_Tucker, G_pruned)` to AMD.
3. Obtain the κ vs n² cost contrast.
4. **Freeze results.**
5. ONLY THEN open the prereg characterizing the residual 25% (where it lives:
   weights / non-linearity / cycles / higher-order) — the precondition for L4-B.

The largest methodological risk now is **not** falling short — it is getting ahead
of L4-B before knowing what the residual 25% actually contains. The freeze enforces
that order.
