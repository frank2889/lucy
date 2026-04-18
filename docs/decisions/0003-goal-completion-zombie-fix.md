# ADR 0003 — Goal-completion zombie-bug

**Status:** Uitgerold in v2.6.14 (2026-04-18)
**Auteur:** Frank + Claude
**Gerelateerd:** [Incident 500K collapse](../incidents/2026-04-15-500k-collapse.md), [ADR 0002 — ignition cap](0002-ignition-cap.md)

## Context

Op tick 503.950 (v2.6.13, 2026-04-18, tijdens routine status-check) viel op:

```
emotions.frustration: 0.99
goal_stats.complete_total: 0
goal_stats.mean_latency_ticks: 0.0
goals: [
  { description: "learn:redirect",         priority: 1.0,  progress: 1.0 },
  { description: "explore:new_sources",    priority: 0.24, progress: 1.0 },
  { description: "think:meta_learning",    priority: 0.10, progress: 0.?  },
  ...
]
```

Paradox: goals met `progress == 1.0` zichtbaar, maar `complete_total == 0`.

Root cause in `crates/lucy-core/src/thinker.rs:317`:

```rust
for (i, goal) in field.goals.iter_mut().enumerate() {
    if goal.progress >= 1.0 { continue; }   // ← skip + NIET toevoegen aan completed_indices

    let age = tick.saturating_sub(goal.created_tick);

    // Onbereikbare dode code — zou de echte completion-afhandeling moeten zijn:
    if goal.progress >= 1.0 {
        completed_indices.push(i);
        continue;
    }
    ...
```

Goals die in een vorige `reflect()`-cyclus `progress=1.0` bereikten, werden bij élke volgende cyclus via line 317 weggefilterd vóór ze aan `completed_indices` konden worden toegevoegd. Gevolg:

- `GOAL_COMPLETE_TOTAL` werd nooit opgehoogd → `complete_total=0`
- `GOAL_LATENCY_SUM_TICKS` bleef op 0 → `mean_latency_ticks=0.0`
- Zombie-goals bleven eeuwig in `field.goals` met `progress=1.0`
- Emergency knock-on: `control_belief` (`compute.rs:506-509`) stijgt bij completion — zonder completions convergeerde EMA van stuckness naar 1.0 → `frustration = 0.99` vastgepind

Deze bug was al aanwezig vóór het 500K-incident, maar werd gemaskeerd doordat Lucy vóór v2.6.10 sowieso nauwelijks goals voltooide. Pas na ADR 0001+0002 (gezond gedrag) werd zichtbaar dat completions niet werden geregistreerd.

## Besluit

Verwijder de vroege-skip op regel 317. De al bestaande check op regel 322 (`if goal.progress >= 1.0 → push + continue`) was dode code; na deze fix is hij weer de feitelijke afhandeling.

```rust
for (i, goal) in field.goals.iter_mut().enumerate() {
    let age = tick.saturating_sub(goal.created_tick);

    // v2.5: Voltooide goals meteen opruimen (ook goals die al eerder 1.0
    // bereikten — v2.6.14 fix: vroege-skip verwijderd die deze goals
    // onbereikbaar maakte, waardoor ze noch geteld noch opgeruimd werden).
    if goal.progress >= 1.0 {
        completed_indices.push(i);
        continue;
    }
    ...
```

**Latency-interpretatie:** `goal_age` wordt gemeten tot de *detection-reflect*, niet tot het exacte tick waar `progress` 1.0 werd. Accuratesse binnen één `reflect_interval` — acceptabel voor observability.

## Gevolgen

**Verwacht na deploy:**
- `goal_stats.complete_total` > 0 binnen enkele reflect-cycles
- `goals` lijst verliest de zombie-entries met `progress=1.0`
- `frustration` decayt (EMA in `compute.rs:513-517`) naarmate `control_belief` stijgt → verwacht < 0.9 binnen 15-30 min
- `satisfaction` beweegt naar boven

**Niet-wijzigingen:**
- Geen wijziging aan score- of ignition-logica (geen overlap met ADR 0001/0002).
- Geen schema-wijziging, snapshot-compat behouden.
- Geen aanpassing aan spawning-logica van goals; nieuwe goals komen normaal op als oude verdwijnen.

## Verificatie

Na `bash deploy.sh`:

1. `curl -s https://lucy.world/telemetry | jq '.goal_stats'` → `complete_total > 0` binnen ~5 min.
2. `mean_latency_ticks > 0` (niet-lege deler).
3. `goals[].progress == 1.0` entries moeten verdwijnen.
4. `emotions.frustration` daalt onder 0.9 binnen 15-30 min.
5. Regressie-check: `winner_turnover_ratio > 0.15`, `score_cv > 0.1`, `ignition_cap_hit_ratio` stabiel (~0.87), tick blijft lopen.

## Open vragen

- **`learn:redirect` met priority=1.0 + progress=1.0** — dit doel bestaat niet expliciet in code, het wordt dynamisch gespawned uit `explorer.interests`. Waarom pint het zich op prio=1.0 vast? Waarschijnlijk `priority = (avg_curiosity + avg_tension).min(1.0)` bij spawn. Verdient follow-up als dit opnieuw zichtbaar wordt.
- **Frustration-clamp op 0.99** — ontwerp-asymptoot van de EMA, geen bug. Documenteren in [invariant](../invariants/competition.md)-style doc als we meer emotie-regels vastleggen.
