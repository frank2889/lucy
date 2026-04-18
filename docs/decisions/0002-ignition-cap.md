# ADR 0002 — Harde cap op `ignition_strength`

**Status:** Uitgerold (v2.6.12, 2026-04-15)
**Auteur:** Frank + Claude
**Gerelateerd:** [Incident 500K collapse](../incidents/2026-04-15-500k-collapse.md), [ADR 0001 — Top-K sparsity](0001-top-k-sparsity.md), [Invariant: competitie](../invariants/competition.md)

## Context

Na deploy van [ADR 0001 (Top-K sparsity)](0001-top-k-sparsity.md) was de differentiatie (`score_cv`) hersteld, maar de `ignition_strength`-waarde rende verder weg:

- **Pre-fix:** `ignition_strength = 21030`
- **Post-ADR-0001 (v2.6.11):** `ignition_strength = 24326`

Top-K selecteert per definitie de *hoogste* 5% scores. De gemiddelde winner-score stijgt daardoor, en omdat:

```
ignition_product = winner_score * (1 + avg_support)
new_is = (1 - fe) * IS + fe * ignition_product    // fe ≈ 0.02-0.05
```

krijgt `IS` elke tick een zet omhoog en decayt slechts langzaam. Resultaat: positive-feedback runaway die door differentiatie-herstel alleen niet gestopt wordt.

## Besluit

Harde clamp `[0.0, 500.0]` op `ignition_strength` na de EMA-update in `global_workspace_broadcast`. Géén softmax, géén goal-aanpassing, géén escape-wijziging — strikt één variabele per fase zodat causaliteit bewijsbaar blijft.

```rust
let new_is_raw = (1.0 - fe) * field.ignition_strength + fe * ignition_product;
let new_is_safe = if new_is_raw.is_finite() { new_is_raw } else { 0.0 };
let was_capped = new_is_safe > IGNITION_CAP;
let clamped = new_is_safe.clamp(0.0, IGNITION_CAP);
field.ignition_strength = clamped;

IGNITION_SAMPLES.fetch_add(1, Ordering::Relaxed);
if was_capped { IGNITION_CAP_HITS.fetch_add(1, Ordering::Relaxed); }
IGNITION_LAST_RAW_BITS.store(new_is_safe.to_bits(), Ordering::Relaxed);
```

**`IGNITION_CAP = 500.0`** — bewuste keuze:

| Referentiepunt               | Waarde  |
|------------------------------|---------|
| Huidige runaway (v2.6.11)    | 24.326  |
| Pre-fix runaway (v2.6.5)     | 21.030  |
| Pre-gedefinieerde Fase 2 trigger | 5.000   |
| Normaal dynamisch bereik     | ~50-100 |
| **Cap**                      | **500** |

≈ 5-10× normaal bereik. Als 500 te laag blijkt: evidence-based ophogen. Harde clamp (niet sigmoid) omdat het kleiner, beter interpreteerbaar en directer zichtbaar is in telemetry.

**Observability (atomics — géén `Field`-wijziging, dus snapshot-compat):**

Module-level `AtomicU64`/`AtomicU32` in `compute.rs:50-66`. Exposed in `/telemetry.gnw_stats`:

- `ignition_cap` (constante 500.0)
- `ignition_strength_raw` (meest recente pre-clamp)
- `ignition_strength_clamped` (veld-waarde, kan achterlopen door ArcSwap)
- `ignition_cap_hits_total` / `ignition_samples_total` / `ignition_cap_hit_ratio`

Log WARN bij cap-hit, throttled elke 200 ticks.

## Gevolgen

**Positief:**
- Tick-progressie hervat (gemeten: 1.25 tps).
- `/telemetry` (tick 503.950, v2.6.13): `ignition_cap_hit_ratio=0.875`, `ignition_strength_raw=1594.9`, `ignition_strength_clamped=500.0`.
- Turnover en `score_cv` ongewijzigd gezond → geen regressie op ADR 0001.

**Negatief / aandachtspunten:**
- 87,5% cap-hit ratio betekent: de onderliggende drive naar ~1600 is niet afgenomen, de cap vangt het alleen. Systeem opereert permanent op ceiling.
- Harde clamp is niet-differentieerbaar. Voor toekomstig gradient-based leren mogelijk relevant (nu niet).
- Gekozen getal (500) is ruw geschat, niet afgeleid uit fundamentele dimensie.

## Verificatie

- Metric `ignition_cap_hit_ratio > 0` → cap wordt getriggerd ✓
- Tick-progressie ≥ 1 tps ✓
- `winner_turnover_ratio > 0.15` → geen collapse door de cap ✓
- Geen terugval naar `collapse_condition_met=true` ✓

## Open vragen

1. **Volstaat 500 op lange termijn?** Als de "drive" blijft groeien met systeem-complexiteit, loopt de cap-hit ratio vast op 100% en wordt `ignition_strength` effectief een constante. Dan verliest de variabele zijn informatieve waarde.
2. **Sigmoid vs adaptieve cap?** Alternatief: cap = `noise_amplitude * n_cand` of `avg_phi_norm * n_cand` (adaptief). Nadeel: minder voorspelbaar, verborgen fluctuaties.
3. **Fase 3 (softmax) nodig?** Beslisrapport 500K (zie [CHANGELOG v2.6.13](../CHANGELOG.md)) concludeerde "niet nodig". Herzien als cap-hit ratio structureel > 95% wordt.
