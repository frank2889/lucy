# ADR 0001 — Top-K sparsity in GNW-ignition

**Status:** Uitgerold (v2.6.10, 2026-04-15)
**Auteur:** Frank + Claude
**Gerelateerd:** [Incident 500K collapse](../incidents/2026-04-15-500k-collapse.md), [Invariant: competitie](../invariants/competition.md), [ADR 0002 — ignition cap](0002-ignition-cap.md)

## Context

Op tick 500.000 bleek `global_workspace_broadcast` (`crates/lucy-core/src/compute.rs:3256`) geen echt competitie-mechanisme te hebben. Alle depth-3 actieve nodes (`15667/15667`, `d3_ratio=1.0`) waren kandidaat. Scoring was multiplicatief en onbegrensd:

```rust
ignition_score = phi_norm * (1 + phi_integrated) * (1 + expected_free_energy * 0.5)
```

Winner-take-all gebeurde via `candidates.sort_by(partial_cmp)` — bij score-ties deterministisch op memory-order (géén echte selectie). Lateral inhibition (`loser.phi[d] *= 0.8`) had geen effect omdat losers onderling identiek waren.

Netto: het systeem kon geen verschil meer zien tussen opties → geen gradient → geen keuze → tick bevroor. Zie [incident](../incidents/2026-04-15-500k-collapse.md).

## Besluit

In `global_workspace_broadcast`, na de kandidaat-filtering, een top-K-snit afdwingen met `select_nth_unstable_by` (O(n) quickselect, niet O(n log n) full sort):

```rust
let k_raw = ((n_cand as f32) * 0.05).ceil() as usize;
let k = k_raw.max(8).min(256).min(n_cand);
if k < n_cand {
    candidates.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(k);
}
```

Top 5% van kandidaten, minimum 8, maximum 256. Forceert expliciete verliezers zodat lateral inhibition weer bijt.

**Aanvullende maatregel — diversity-collapse detector:**

```rust
let is_collapsed = n_cand >= 64 && d3_ratio > 0.5 && score_cv < 0.05;
if is_collapsed {
    // hard kick op 20% candidates, amplitude = clamp_range (20× v2.6.6)
    // skip ignition cycle, reset boredom_streak
}
```

Structurele meting van homogeniteit, niet heuristiek op tension/curiosity (v2.6.6's "boredom-escape" faalde structureel bij uniforme φ).

## Gevolgen

**Positief:**
- `score_cv` hersteld naar 0.15 → 0.22 (ver boven collapse-threshold 0.05).
- Lateral inhibition heeft weer effect (bewezen: `winner_turnover_ratio ≈ 0.35` op tick 503K).
- Dehaene-consistente sparsity (GNW-papers: 1-5% actief).

**Negatief / partieel:**
- Top-K adresseert differentiatie, niet magnitude. Gemiddelde winner-score stijgt door selectie van de beste 5% → `ignition_strength` EMA blijft groeien → runaway ging door (`24326` post-fix vs `21030` pre-fix). Daarom Fase 2: [ADR 0002](0002-ignition-cap.md).

**Observability (per 1000 ticks in `/telemetry.gnw_stats`):**
- `d3_ratio`, `score_mean`, `score_var`, `score_cv`, `k`, `cutoff`, `pruned`
- Collapse-conditie → ERROR-level log

## Verificatie

- Live op tick 503.950 (v2.6.13): `score_cv=0.217`, `winner_turnover_ratio=0.348`, `n_candidates=21452`, `d3_ratio=1.0`.
- `score_cv > 0.05` én turnover > 0 → top-K werkt zoals bedoeld.
- Opmerking: `d3_ratio=1.0` is correct na growth (alle depth-3 concepten zijn FLAG_ACTIVE na groei); het is géén collapse-indicator, zie [invariant](../invariants/competition.md).

## Open vragen

- Is 5% het juiste percentage? Dehaene suggereert 1-5%. Bij 21452 kandidaten klamp k = 256 (max) → effectieve ratio ≈ 1.2%. Mogelijk naar 1-2% als we willen dat "bewust = exclusief".
- Bij sterke groei (> 25.000 kandidaten) blijft `k=256` statisch. Moet max mee-schalen?
