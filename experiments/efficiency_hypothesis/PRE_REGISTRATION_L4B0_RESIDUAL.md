# L4-B0 — Residual Characterization (Pre-Registration)

**Commit this document BEFORE writing or running any L4-B0 code.** Its commit SHA is
the pre-registration timestamp. L4-B0 is the **characterization** prereg that the
L4-A sequence (`PRE_REGISTRATION_L4A_DUAL.md` §0.3, §6, §8.5) gates the L4-B research
hypothesis on. It is explicitly **descriptive, not constructive**: it does NOT build
the inverse graph→tensor projection (`V'`). It answers one prior question —
**where does the residual ΔU live?** — so that L4-B, if opened, is designed against a
known object instead of repeating the S3-bis failure (*build the operator before
understanding the object*).

- **Date:** 2026-06-10
- **Gated on:** AMD S5 cost contrast frozen (commit 447e58d / a762f39, 2026-06-10).
  The freeze condition of L4-A §8.4 is **met**; this prereg may open.
- **Reuses (verbatim, no new machinery):** S1-bis corpus (t=48), the Tucker operator
  `tucker_operator.py` (rank (8,3,3,3,6), κ=1296), the TCI metric U (`run_tci.U`), the
  flow matrix Φ (`run_tci.flow_matrix` / `val_matrix_one_session`), the Form-1 prune
  primitives (`run_form1.masked_flow`, `run_form1.measure`,
  `run_s3_run2.discover_edges_majority`), all with the S3-bis/TCI machinery
  (ParCorr, tau=1, pc_alpha=0.01, vote≥0.5). **No new corpus.** One declared auxiliary
  PCMCI run for B3 (tau_max=3, §1) — same estimator, different tau_max only.

---

## 0. The object to characterize (the analysis)

### 0.1 What the residual is, precisely — and in which space U lives
U is **not** a function of an edge set. `run_tci.U(Φ_ref, Φ_test)` is the **Pearson
correlation of the off-diagonal entries of the continuous flow matrix** Φ, where
`Φ[i,j] = median_s val_matrix_s[i,j,lag=1]` (signed partial-correlation strength,
ParCorr). The residual therefore lives in the **continuous, signed Φ-space**, not in
edge presence/absence. This is the central correction over the first draft: every
bucket below is a *difference in Φ entries*, not a count of edges.

L4-A delivered `V = (V_Tucker, G_pruned)`. Two U values bracket the residual:

- `U(Φ_tucker)` — U of the **raw Tucker reconstruction** Φ, before the Form-1 prune
  (`measure(tucker_series)` → `base["U"]`).
- `U(Φ_masked_raw)` ≈ **0.862** — after masking Φ to the raw causal support
  (`measure(tucker_series, raw_edges)` → `masked_raw["U"]`; l4a_results.json
  U∈{0.840,0.864,0.883}).

The **residual** is `ΔU = 1 − U(Φ_masked_raw)` ≈ 0.138: the headroom to the raw
reference (U=1 by construction, `raw_ref["U"]`). Because the Form-1 prune is a mask on
Φ (zeroes off-support entries — `masked_flow`), it can only *remove* off-support flow;
it cannot *restore* an on-support entry the Tucker core reconstructed with the wrong
magnitude or sign. So the residual is, by construction, one of:

1. **B1 — Support mismatch (magnitude).** On-support entries where `Φ_raw[i,j] ≠ 0`
   but `Φ_tucker[i,j]` is attenuated/inflated in **magnitude** (same sign). The core's
   rank-(8,3,3,3,6) truncation distorts the strength.
2. **B2 — Sign error.** On-support entries where `sign(Φ_tucker[i,j]) ≠
   sign(Φ_raw[i,j])`. Reported as its **own** bucket because U is Pearson over *signed*
   values: a flip moves a point across the correlation axis, costing U non-linearly and
   far more than a same-sign magnitude error of equal size. Conflating it with B1 (the
   first-draft error) understates its impact.
3. **B3-lin — Linear-residual (the complement).** The fraction of ΔU that B1+B2
   *cannot* recover even when fully corrected, **within the linear ParCorr regime at
   lag 1**. By exhaustion this is the non-edge-representable part a single **linear**
   V' cannot carry: non-linearity (ParCorr cannot see it) plus any lag-1 structure the
   masking still misses. It is measured as a **residual**, not by a non-linear test
   (see §0.3).
4. **B4 — Higher-order temporal (lag>1).** Structure at lag∈{2,3} the lag-1 machinery
   drops entirely. Measured via one declared auxiliary PCMCI run at tau_max=3 (§1).

B1+B2 are **linear-edge-representable**: structure an inverse linear projection could
fold into a volume V'. B3-lin+B4 are **not** carriable by a single linear V'. The
go/no-go for L4-B turns on the B1+B2 share (§5).

### 0.2 Why characterize, not construct (methodological — user-confirmed 2026-06-10)
S3-bis punished building a low-rank operator that scored well statistically (98%
variance) but destroyed causality (semantic collapse). The lesson — encoded as
`Causality ≻ Topology ≻ Reconstruction` — is to not build `V'` before knowing what it
must preserve. L4-B0 spends one cheap descriptive cycle to decide whether L4-B is even
the right next experiment, or whether the dual representation is the honest terminal
form.

### 0.3 Why B3 is a residual, not a non-linear test (environment-verified 2026-06-10)
A direct non-linear measurement of B3 was attempted and is **not feasible in the frozen
environment**: GPDC fails (`ModuleNotFoundError: dcor`) and CMIknn fails
(`TypeError: corrcoef() ddof` — numpy↔tigramite version skew). Installing `dcor` or
patching numpy would mutate the exact environment that validated the ParCorr machinery
— a non-declared change a pre-registration must not smuggle in. **It is also
unnecessary:** L4-B proposes a *linear* V', so the decision only needs how much of ΔU
is linear-edge-representable (B1+B2, measured with the validated ParCorr machinery).
Whatever B1+B2 cannot recover **is** the non-linear-or-higher residual, by exhaustion —
exactly the part L4-B's linear projection could not carry anyway. Measuring B3 with a
non-linear test would be redundant with the complement of B1+B2. B3-lin is therefore
reported as `ΔU − ΔU_{B1} − ΔU_{B2} − ΔU_{B4}` (D2), not as a separate estimator call.

---

## 1. The decomposition under test (fixed a priori)

All quantities reuse the existing functions; symbols match `run_form1.py`.

```
Φ_raw      := flow_matrix(raw_series[g])                       # raw reference, U=1
Φ_tucker   := flow_matrix(tucker_series[g])                    # unmasked Tucker recon
raw_edges  := discover_edges_majority(raw_series[g])           # raw causal support
Φ_masked   := masked_flow(tucker_series[g], raw_edges)         # = G_pruned's Φ (L4-A)
U0         := U(Φ_raw, Φ_masked)  ≈ 0.862                      # the post-prune U
ΔU         := 1 − U0  ≈ 0.138                                  # the residual

# Bucket corrections — inject the RAW Φ value for that bucket's entries into Φ_masked,
# recompute U with the SAME run_tci.U, and read the recovery.
S          := raw_edges (the on-support entry set, i≠j)
B1_set     := { (i,j)∈S : sign(Φ_tucker)=sign(Φ_raw) ∧ |Φ_tucker−Φ_raw| > δ_w }
B2_set     := { (i,j)∈S : sign(Φ_tucker) ≠ sign(Φ_raw) }       # incl. one side ≈ 0
B4 (lag>1) : auxiliary PCMCI at tau_max=3 (ParCorr, pc_alpha=0.01) → Φ^{(2)},Φ^{(3)};
             B4 recovery = U gain from adding the lag-2/3 raw support to the lag-1 Φ,
             via an extended flow comparison (declared aux run; same estimator).
```

- `δ_w` = 0.10 on the Φ scale U is computed on (the normalized val_matrix). Fixed.
- B1 and B2 partition the on-support entries (sign-equal vs sign-flipped); they do not
  overlap by construction. B4 is on a different lag axis. B3-lin is the complement.

## 2. ΔU attribution metric (fixed a priori)

For each measured bucket B_k ∈ {B1, B2, B4}, the **U-recovery if that bucket were
corrected to its raw value**:

```
ΔU_k := U( Φ_raw , correct(Φ_masked, B_k) ) − U0
correct(Φ_masked, B_k): copy Φ_raw[i,j] into Φ_masked[i,j] for (i,j) ∈ B_k_set
                        (for B4: add the lag-2/3 raw entries on the extended axis)
```

Reported per graph G1/G2/G3 and as the mean. Attribution per bucket = `ΔU_k / ΔU`.
B3-lin := `(ΔU − ΔU_{B1} − ΔU_{B2} − ΔU_{B4}) / ΔU` (the complement, §0.3). Buckets
need not be exactly additive (Pearson is non-linear in its inputs); §D2 reports the
sum and any super/sub-additivity explicitly.

## 3. The question L4-B0 answers (fixed a priori)

> Of the residual ΔU ≈ 0.138, what fraction is **linear-edge-representable**
> (B1 magnitude + B2 sign — structure a linear inverse projection could fold into V')
> versus **not carriable by a single linear V'** (B3-lin non-linearity + B4 lag>1)?

This is the **go/no-go prior for L4-B.** It is NOT a claim that L4-B works.

## 4. Confirmatory / descriptive checks (fixed a priori)

- **D1 — Reproducibility (gate).** Bucket attribution is deterministic given the TCI
  seeds (shuffle_seed 20260609) and the frozen ParCorr machinery. PASS if every `ΔU_k`
  is byte-identical across two runs. (The aux tau_max=3 run is likewise deterministic.)
- **D2 — Exhaustiveness honesty (report, not gate).** Report `ΔU_{B1}+ΔU_{B2}+ΔU_{B4}`
  vs `ΔU`, and B3-lin as the complement. State super/sub-additivity. There is no PASS
  to game; the complement carries the non-linear/unmeasured mass openly.
- **D3 — Machinery-scope honesty (report).** No non-linear estimator is run (GPDC/CMIknn
  infeasible in-env, §0.3); B3-lin is a residual, not a measurement. The only run beyond
  the frozen lag-1 machinery is the declared tau_max=3 PCMCI for B4 (same ParCorr,
  same pc_alpha). Stated so the prereg cannot be mis-cited as "L4 measured non-linearity."

## 5. Decision logic (fixed a priori) — this prereg's whole point

Let `share_lin = (ΔU_{B1} + ΔU_{B2}) / ΔU` (the linear-edge-representable share).

| `share_lin` | Decision |
|---|---|
| **≥ 0.70** | **GO for L4-B.** The residual is dominantly linear structure an inverse projection can fold into V'. Open the L4-B prereg (inverse graph→tensor projection), targeting B1+B2 specifically. |
| **0.40 – 0.70** | **CONDITIONAL.** A linear V' recovers part of ΔU but cannot close it; the dual representation stays the operational form. Open L4-B only with an explicit partial-recovery target, not a single-V claim. |
| **< 0.40** | **NO-GO for L4-B as a single linear V'.** The residual lives in non-linearity/lag>1 a linear core cannot carry. The dual `(V_Tucker, G_pruned)` is the honest terminal representation; RCC's single-V north star is refuted at this rank. Publish as a negative result. |

The NO-GO branch is a **publishable negative result** — the second time the project's
`Causality ≻ Reconstruction` principle prevents a collapse.

## 6. What L4-B0 explicitly does NOT do

- **No inverse projection.** `V'` is not built. That is L4-B, gated on §5.
- **No new operator, no new corpus.** Tucker rank, prune, PCMCI machinery unchanged;
  S1-bis (t=48) only.
- **No non-linear estimator, no env mutation.** No `dcor` install, no numpy patch; the
  ParCorr environment that validated L3 stays frozen (§0.3).
- **No κ change.** κ stays 1296 (descriptive; touches no cost quantity).
- **No governance-accuracy claim.** Condition (c) untouched (still RCT-gated).

## 7. Fixed parameters (summary)

- Corpus: S1-bis (t=48), n=30 sessions, graphs G1/G2/G3.
- Operator under inspection: L4-A dual `C_A`, Tucker rank (8,3,3,3,6), κ=1296.
- Φ / U: `run_tci.flow_matrix`, `run_tci.U` (Pearson of off-diagonal signed Φ).
- Prune primitive: `run_form1.masked_flow` / `masked_edges`; support =
  `discover_edges_majority` (vote≥0.5).
- PCMCI: ParCorr, tau=1, pc_alpha=0.01 (main); ParCorr, tau_max=3, pc_alpha=0.01
  (declared aux, B4 only).
- δ_w = 0.10 on normalized Φ. shuffle_seed = 20260609.
- Buckets: B1 magnitude (sign-equal, |ΔΦ|>δ_w), B2 sign-flip, B4 lag>1 (aux),
  B3-lin = complement. Attribution by §2 correct-and-recompute-U.
- Checks: D1 reproducibility (gate), D2 exhaustiveness (report), D3 machinery-scope
  (report). Decision §5: `share_lin = (ΔU_B1+ΔU_B2)/ΔU` → GO / CONDITIONAL / NO-GO.

## 8. Execution sequence (fixed)

1. Commit this prereg (timestamp).
2. Recompute Φ_raw, Φ_tucker, Φ_masked and U0 from the L4-A operator, reusing
   `run_form1.measure` (no new machinery for the lag-1 quantities).
3. Partition on-support entries into B1/B2; attribute ΔU_{B1}, ΔU_{B2} by §2.
4. Run the declared tau_max=3 aux PCMCI; attribute ΔU_{B4}.
5. Compute B3-lin as the complement (D2); run D1, D3.
6. Apply §5 → GO / CONDITIONAL / NO-GO for L4-B.
7. **Only on GO/CONDITIONAL** open `PRE_REGISTRATION_L4B_INVERSE.md` (the inverse
   projection). On NO-GO, write the negative-result note and freeze the dual as terminal.

The methodological risk this prereg manages: opening L4-B (weeks of inverse-projection
math) against a residual a single linear V' provably cannot carry. §5 makes that
decision *before* the cost is incurred.
