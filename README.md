# CAL-L4 — Meta-Inference Layer

**Parte de:** [CAL — Cognitive Abstraction Layers](../)  
**Autor:** Juan Pablo Chancay (Aural Syncro)  
**Estado:** En desarrollo — baseline O(n²) activo; M(V) diferido hasta gate-C  
**Venue objetivo:** NeurIPS / ICML  
**Colaboración:** AMD-Instinct Labs (`fa_dme` en MI300X)

## Definición formal

```
L4: M(V) → {decisions, predictions, adaptations}
```

Meta-inferencia sobre volúmenes tensoriales comprimidos. Operación sin memoria
de trabajo humana como sustrato.

## L4 Efficiency Hypothesis (§6.2 CAL pre-paper)

> Existe una arquitectura de inferencia tal que el costo de M(V) escala con κ(V)
> — complejidad estructural de V — donde κ(V) crece significativamente más lento
> que O(n²) en n (conteo de artefactos del espacio L0 crudo).

Probar esto requiere: (a) C definido + κ(V) concreto, (b) comparación M(V) vs
flat-context O(n²), (c) accuracy de gobernanza bajo ambos enfoques.

## Representational Convergence Conjecture — RCC (§6.4)

El estado de gobernanza óptimo puede ser extraíble directamente de las
activaciones de atención durante pre-fill — sin segunda pasada LLM-QA.
El `probe_mfma_mapping.hip` de AMD-Instinct ya caracterizó el mapeo
lane↔output de `v_mfma_f32_16x16x16f16`, el acceso de bajo nivel que requeriría.

## Roadmap con gates

| Tarea | Estado | Bloqueante |
| --- | --- | --- |
| Baseline flat-context O(n²) (`fa_robust` seqLen sweep) | AMD — listo en frío | — |
| Confirmar régimen cuadrático (log-log, exp ≈ 2) | Pendiente (próxima VM AMD) | — |
| Operador C definido (L3) | Pendiente | gate L3 |
| Kernel proxy M(V): O(n²) flat vs O(κ) | Diferido (AMD post-gate) | C/L3 |
| Probar L4 Efficiency Hypothesis en sintético | Diferido | M(V) + C |

## Estructura (en construcción)

```
L4/
├── README.md
├── paper/              ← paper L4 (en desarrollo)
├── src/                ← implementación M(V) (post-gate-C)
├── benchmarks/         ← baseline O(n²) + contraste O(κ)
└── experiments/        ← L4 Efficiency Hyp. tests
```

## Colaboración AMD-Instinct

`fa_dme` (Flash Attention DME, validado MI300X) tiene doble rol en L4:
- **Rol 1 (ahora):** medir la curva flat-context O(n²) — baseline empírico de la hipótesis
- **Rol 2 (post-gate-C):** kernel del cual la RCC extrae señal V en pre-fill

Registro canónico de la colaboración: `Obsidian/wiki/proyectos/cal-collaboration.md`
