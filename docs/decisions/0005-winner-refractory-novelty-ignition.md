# ADR 0005 — Winner-refractory + novelty-gated ignition boost

**Status:** Uitgerold in v2.6.15
**Auteur:** Frank + Claude (Opus code review)
**Gerelateerd:** Groep B1 fixes (lucy-state, compute.rs), [ADR 0001](0001-top-k-sparsity.md)

## Context

Code review (Opus) ontdekte: ondanks top-K sparsity (ADR 0001) en cap (ADR 0002) blijft één GNW-concept structureel domineren.

**Live data tick 503K:**
- `winner_turnover_ratio: 0.348` — Lucy wisselt wel, maar hetzelfde concept herhaalt zich 15× vs. 8 switches
- `ignition_cap_hit_ratio: 0.875` — 87,5% van ticks raakt de cap, ruw mechanisme domineert
- `score_cv: 0.217` — gezond verschil, dus competitie werkt _formeel_, maar één concept "gaat niet weg"

**Gat:** RND-curiosity (novelty-detectie) was berekend maar niet gekoppeld aan GNW-selectie. Nieuwe informatie triggerde geen herignition van een ander concept.

## Besluit

Twee wijzigingen, één in Field struct, één in `global_workspace_broadcast`:

### 1. Field struct uitbreiding (`lucy-state/src/lib.rs`)

```rust
#[serde(default)]
pub last_winner_id: u32,

#[serde(default)]
pub winner_refractory_until: u64,
```

Met `#[serde(default)]` → bestaande snapshots laden zonder probleem (backwards-compatible).

### 2. Refractory + novelty boost in `global_workspace_broadcast` (compute.rs)

**Refractory check** (na top-K, vóór score-berekening):
```rust
if field.winner_refractory_until > field.tick {
    candidates.retain(|(id, _, _)| *id as u32 != field.last_winner_id);
}
```

Verwijdert de vorige winner uit de competitie voor minimaal 50 ticks. Dwingt ander concept aan de beurt.

**Novelty-boost op ignition_score**:
```rust
let novelty_boost = 1.0 + field.points[id].curiosity * 0.3;
ignition_score *= novelty_boost;
```

Punten met hoge RND-curiosity (detected novelty) krijgen multiplicatieve boost. Sluit loop: `verrassing → hoger curiosity → hogere kans op ignition → ander concept wint`.

**Winner registration**:
```rust
field.last_winner_id = winner_u32;
field.winner_refractory_until = field.tick + 50;
```

Na winst, stel refractory period.

## Gevolgen

**Competitie-gedrag:**
- Zelfde concept kan niet 50+ ticks achter elkaar herignite worden
- Plotselinge novelty → multiplicatieve ignition-boost → verrassing domineert selectie
- `winner_turnover_ratio` stijgt van ~0.35 naar verwacht ~0.5+

**Lerenleren:**
- Rood informatie die `curiosity` oprekt (RND-piek) verstoort het dominant concept → nieuw concept krijgt kans
- Dit geeft Lucy echte **verrassing-respons**: onverwachtse input verandert werkelijk haar bewuste focus

**Schema-impact:** Twee `#[serde(default)]` velden → snapshot-compat, geen FieldV2 bridge nodig

## Verificatie

**Na deploy:**
1. `curl .../telemetry | jq '.gnw_stats.winner_turnover_ratio'` → moet > 0.4 stijgen
2. Injecteer hoog-novelty input → monitor `ignition_strength` → moet even naar iets lager concept springen
3. Regressie check: `score_cv > 0.1` (competitie-invariant intact), `tick` beweegt normal

## Architectural impact

- `lucy-state/src/lib.rs` — Field struct, 2 velden
- `crates/lucy-core/src/compute.rs` — `global_workspace_broadcast`, 3 logische blokken (refractory filter, novelty boost, registration)
- Geen verandering aan apprentice-API, geen hardcoded constants (behalve 50-tick refract_period)
