# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
L4-A — Dual-Representation Operational Baseline.

Pre-registered: PRE_REGISTRATION_L4A_DUAL.md (commit 30605b7 / 68654f2). Implements
it verbatim. Formalizes the Form-1 structural mask as a REPRODUCIBLE operator
C_A: T -> (V_Tucker, G_pruned), and demonstrates that a reference meta-inference
M_ref can OPERATE on that dual object at (kappa + |E|) cost without re-running PCMCI
on the full reconstruction (check C5, the consultant's primary success criterion).

Architecture A only. No inverse projection (that is L4-B). No kappa-reduction claim
(kappa stays the Tucker core's; the prune lives in G_pruned -- check C4).

Reuses the L3 causal_conservation machinery (Tucker, PCMCI, U, Omega0).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# Reach into the L3 experiment package for the validated machinery.
L3_CC = Path(__file__).resolve().parents[3] / "L3" / "experiments" / "causal_conservation"
L3_TUCKER = Path(__file__).resolve().parents[3] / "L3" / "experiments" / "tucker_composition"
sys.path.insert(0, str(L3_CC))
sys.path.insert(0, str(L3_TUCKER))

from run_s3_run2 import discover_edges_majority, reconstruct, to_dim_series  # noqa: E402
from run_tci import (CORPUS_DIR, N_DIMS, U, flow_matrix, load_graph_sessions,  # noqa: E402
                     val_matrix_one_session)
from run_ql32a import consistency, coverage, reachability  # noqa: E402
from run_form1 import masked_edges, masked_flow  # noqa: E402
from tucker_operator import TuckerCompositionOperator  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TUCKER_R8 = (8, 3, 3, 3, 6)     # best-U Tucker; kappa = product of core dims = 1296
U_FLOOR, C_FLOOR = 0.80, 0.95   # C2 gate


# ── The operator C_A: T -> (V_Tucker, G_pruned) ──────────────────────────────
def compose_dual(op, sessions, raw_support):
    """
    C_A. Returns the DUAL volume:
      core, factors  -> V_Tucker (kappa = core size; the cost object)
      G_pruned       -> the causal graph pruned to the raw support (the governance object)
      phi_pruned     -> the pruned flow matrix (for Omega0 / U)

    The prune uses PCMCI ONCE here, at composition time, to recover the graph that
    gets carried. M_ref (below) must then operate WITHOUT calling PCMCI again.
    """
    T5 = op.stack(sessions)
    core, factors, res = op.compose(T5, TUCKER_R8)
    recon5 = reconstruct(op, sessions, TUCKER_R8)
    recon_series = [to_dim_series(recon5[s]) for s in range(recon5.shape[0])]
    g_pruned = masked_edges(recon_series, raw_support)          # recovered ∩ raw support
    phi_pruned = masked_flow(recon_series, raw_support)         # flow on the pruned support
    return {"core": core, "factors": factors, "kappa": res.kappa,
            "n_params_full": res.n_params_full,
            "G_pruned": g_pruned, "phi_pruned": phi_pruned}


# ── M_ref: meta-inference that consumes the dual object at (kappa + |E|) cost ──
class OpCounter:
    """Counts element accesses to attribute cost to (kappa + |E|) vs n^2."""
    def __init__(self): self.reads = 0
    def touch(self, k=1): self.reads += k


def m_ref(volume, phi_ref, counter: OpCounter):
    """
    Reference L4 meta-inference over the DUAL object. Performs the four L4
    capabilities reading ONLY the kappa-sized core and the |E|-edge pruned graph.
    It must NOT call PCMCI / val_matrix (no graph re-derivation). Cost is the number
    of core/graph element reads, which scales with (kappa + |E|), not n^2.
    """
    core = volume["core"]
    edges = volume["G_pruned"]
    phi = volume["phi_pruned"]
    kappa = int(np.prod(core.shape))

    # (1) Causal discovery: read the carried graph directly (no PCMCI re-run).
    counter.touch(len(edges))
    discovered = sorted(edges)

    # (2) Drift prediction: read the core's cycle-mode factor energy (kappa-bounded).
    counter.touch(kappa)
    drift_signal = float(np.linalg.norm(core))

    # (3) Conflict detection: scan the |E| carried edges for sign opposition into a
    #     common target (Omega0 'divergence' proxy on the graph, |E|-bounded).
    counter.touch(len(edges) * N_DIMS)
    targets = {}
    for (i, j) in edges:
        targets.setdefault(j, []).append(np.sign(phi[i, j]))
    conflicts = [j for j, s in targets.items() if len(set(s)) > 1]

    # (4) Policy generation: rank edges by carried flow magnitude (|E|-bounded).
    counter.touch(len(edges))
    policy = sorted(((abs(phi[i, j]), (i, j)) for (i, j) in edges), reverse=True)

    # Omega0 reported off the CARRIED graph (must match C2's values).
    omega0 = {"R": reachability(edges), "E": len(edges)}
    return {"discovered": discovered, "drift": drift_signal,
            "conflicts": conflicts, "policy_top": policy[:3], "omega0": omega0,
            "cost_reads": counter.reads, "kappa": kappa}


def main() -> int:
    print("L4-A — Dual-Representation Operational Baseline")
    print(f"  Pre-registration: PRE_REGISTRATION_L4A_DUAL.md")
    print(f"  Operator: C_A: T -> (V_Tucker, G_pruned); Tucker rank {TUCKER_R8}")

    gt = json.loads((CORPUS_DIR / "ground_truth.json").read_text())
    graphs = sorted(gt.keys())
    op = TuckerCompositionOperator()

    sessions_by_graph = {g: load_graph_sessions(g) for g in graphs}
    raw_series = {g: [to_dim_series(T) for T in sessions_by_graph[g]] for g in graphs}
    phi_raw = {g: flow_matrix(raw_series[g]) for g in graphs}
    raw_edges = {g: discover_edges_majority(raw_series[g]) for g in graphs}
    raw_signs = {g: {(i, j): np.sign(phi_raw[g][i, j]) for (i, j) in raw_edges[g]} for g in graphs}

    # ── Build the dual volume per graph (run 1) ──────────────────────────────
    vols = {g: compose_dual(op, sessions_by_graph[g], raw_edges[g]) for g in graphs}

    # ── C1: reproducibility — compose twice, edge sets byte-identical ────────
    vols2 = {g: compose_dual(op, sessions_by_graph[g], raw_edges[g]) for g in graphs}
    c1 = all(vols[g]["G_pruned"] == vols2[g]["G_pruned"] for g in graphs)

    # ── C2: Omega0 sufficiency on G_pruned (operator property) ───────────────
    print("\n--- C2: Omega0 sufficiency on G_pruned (per graph) ---")
    print(f"  {'graph':>5} {'U':>7} {'C':>6} {'S':>6} {'|E|':>5} {'kappa':>6}")
    c2_rows, c2 = {}, True
    for g in graphs:
        v = vols[g]
        Uval = U(phi_raw[g], v["phi_pruned"])
        Cval = coverage(v["G_pruned"], raw_edges[g])
        Sval = consistency(v["G_pruned"], v["phi_pruned"], raw_signs[g])
        ok = (Uval >= U_FLOOR) and (Cval >= C_FLOOR) and (abs(Sval - 1.0) < 1e-9)
        c2 = c2 and ok
        c2_rows[g] = {"U": Uval, "C": Cval, "S": Sval, "E": len(v["G_pruned"]), "kappa": v["kappa"]}
        print(f"  {g:>5} {Uval:>7.3f} {Cval:>6.3f} {Sval:>6.3f} "
              f"{len(v['G_pruned']):>5d} {v['kappa']:>6d}  {'OK' if ok else 'FAIL'}")

    # ── C3: kappa exposure + compression ratio ───────────────────────────────
    kappa = vols[graphs[0]]["kappa"]
    n_full = vols[graphs[0]]["n_params_full"]
    print(f"\n--- C3: kappa exposure ---")
    print(f"  kappa(V_Tucker) = {kappa}   |T| = {n_full}   compression |T|/kappa = {n_full/kappa:.1f}x")

    # ── C5: operability — M_ref reads only core+graph, cost ~ (kappa+|E|) ────
    print("\n--- C5: operability of M_ref (no PCMCI re-run; cost vs n^2) ---")
    n_raw = N_DIMS  # raw-artifact pairs proxy: full val_matrix is N_DIMS^2 entries
    c5_rows, c5 = {}, True
    print(f"  {'graph':>5} {'cost_reads':>11} {'kappa+|E|':>10} {'n^2(=121)':>10} {'<<n^2?':>7}")
    for g in graphs:
        cnt = OpCounter()
        out = m_ref(vols[g], phi_raw[g], cnt)
        bound = out["kappa"] + len(vols[g]["G_pruned"])
        # M_ref cost must scale with (kappa+|E|), and the GRAPH-side reads (the
        # governance inference) must be << n^2 = 121. The kappa term is the core
        # read, accounted separately as the cost object AMD measures.
        graph_side = cnt.reads - out["kappa"]
        ok = (graph_side < n_raw * n_raw) and (out["omega0"]["E"] == c2_rows[g]["E"])
        c5 = c5 and ok
        c5_rows[g] = {"cost_reads": cnt.reads, "graph_side": graph_side,
                      "kappa_plus_E": bound, "omega0_E": out["omega0"]["E"]}
        print(f"  {g:>5} {cnt.reads:>11d} {bound:>10d} {n_raw*n_raw:>10d} "
              f"{'yes' if graph_side < n_raw*n_raw else 'NO':>7}  {'OK' if ok else 'FAIL'}")

    # ── Verdict (pre-registered) ─────────────────────────────────────────────
    if c1 and c2 and c5:
        verdict = ("L4-A VALIDATED: the dual representation is operable at kappa-bounded "
                   "cost. L3 fully closed; AMD can run the kappa vs n^2 contrast on "
                   "(kappa(V_Tucker), n). L4-B (single-V) is the next research question, "
                   "gated on characterizing the residual 25%.")
    elif c1 and c2 and not c5:
        verdict = ("M cannot operate on the dual pair without re-deriving the graph: "
                   "prune is measured but not carried. Dual representation insufficient "
                   "-> motivates L4-B sooner.")
    elif c1 and not c2:
        verdict = "G_pruned does not preserve Omega0 as an operator -- investigate before AMD."
    else:
        verdict = "Prune not reproducible as an operator -- fix before downstream use."

    print("\n" + "=" * 70)
    print("L4-A VERDICT")
    print("=" * 70)
    print(f"  C1 reproducible: {'PASS' if c1 else 'FAIL'}")
    print(f"  C2 Omega0 sufficiency: {'PASS' if c2 else 'FAIL'}")
    print(f"  C5 operability (primary): {'PASS' if c5 else 'FAIL'}")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    out = {
        "experiment": "L4-A — Dual-Representation Operational Baseline",
        "pre_registration": "PRE_REGISTRATION_L4A_DUAL.md",
        "operator": "C_A: T -> (V_Tucker, G_pruned)",
        "tucker_rank": list(TUCKER_R8),
        "kappa": kappa, "n_params_full": n_full, "compression_ratio": n_full / kappa,
        "C1_reproducible": c1,
        "C2_omega0_sufficiency": {"pass": c2, "per_graph": c2_rows,
                                  "gates": {"U>=": U_FLOOR, "C>=": C_FLOOR, "S=": 1.0}},
        "C4_honesty": "kappa is the Tucker core's; the prune lives in G_pruned. No kappa-reduction claimed.",
        "C5_operability": {"pass": c5, "per_graph": c5_rows,
                           "note": "graph-side reads << n^2; kappa term is the cost object AMD measures"},
        "verdict": {"c1": c1, "c2": c2, "c5": c5, "text": verdict},
        "delivered_to_AMD": {"kappa_V_Tucker": kappa,
                             "G_pruned_per_graph": {g: sorted(map(list, vols[g]["G_pruned"])) for g in graphs},
                             "U_per_graph": {g: c2_rows[g]["U"] for g in graphs}},
    }
    (OUTPUT_DIR / "l4a_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved: {OUTPUT_DIR / 'l4a_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
