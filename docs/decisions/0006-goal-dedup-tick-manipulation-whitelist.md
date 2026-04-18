# ADR 0006 — Goal dedup + tick manipulation fix + cp/mv whitelist

**Status:** Uitgerold in v2.6.15
**Auteur:** Frank + Claude (Opus code review)
**Gerelateerd:** Groep A fixes (3 bugs: thinker.rs, main.rs, self_modify.rs)

## Context

Opus code review ontdekte drie losse bugs die systeem-gedrag ondergraven:

1. **Goal-spiraal:** `RECENT_GOAL_DESCS` buffert duplicaten maar blokkeert spawn niet (A3, thinker.rs:274-278)
2. **Tick-sprong:** `/decide_action` overschrijft `field.tick` direct, slaat ticks over (A5, main.rs:1202-1204)
3. **Zelf-evolutie gebroken:** `restart_self` gebruikt `cp/mv`, maar whitelist blokkeert beide (A4, self_modify.rs:65)

Alle drie dwingen systeem-patronen op die geen volledige controleverhouding geven.

## Besluit

### Fix A3: Goal-deduplicatie in `reflect()`

```rust
// Huidig (incomplete):
let occurrences = recent.iter().filter(|d| d == &description).count();
if occurrences >= 2 {
    GOAL_REPEAT_EVENTS.fetch_add(1, Ordering::Relaxed);
    // maar geen return — spawn gaat door!
}
recent.push(description.clone());

// Fixed:
let occurrences = recent.iter().filter(|d| d == &description).count();
if occurrences >= 2 {
    GOAL_REPEAT_EVENTS.fetch_add(1, Ordering::Relaxed);
    return None;  // ← dit ontbrak
}
recent.push(description.clone());
```

Eén `return None` toevoegen. Effect: als een goal type al 2× in de laatste 5 reflects is gespawnd, blokkeert hij. Voorkomt herhaald spawn van `think:meta_learning` zonder progress.

### Fix A4: `cp` en `mv` whitelist

```rust
// ALLOWED_COMMANDS toevoegen:
"cp ",      // v2.6.15: enable restart_self binary swap
"mv ",      // v2.6.15: enable restart_self binary swap
```

Twee strings. Effect: `restart_self` kan nu daadwerkelijk de new binary op de old verwisselen. Zelf-evolutie (recompile → restart) werkt.

### Fix A5: Tick-manipulatie uit `decide_action` verwijderen

```rust
// Huidig (springt ticks over):
if tick % field.explore_interval != 0 {
    field.tick = (field.tick / field.explore_interval + 1) * field.explore_interval;
    ...
}

// Fixed:
if tick % field.explore_interval != 0 {
    return;  // geef de keuze door, manipuleer niet tick
}
```

Verwijder de tick-overwrite. Effect: Lucy's keuze om te exploreren wordt gelogged maar verandert niet de tick-loop tempo. Interval-triggers (boredom check, logging, risk-zone detection) vuren op de normale moments, niet overgeslagen.

## Gevolgen

**Goal-systeem:**
- Geen meer `think:meta_learning` spirals (onbeperkt herhaling)
- Goals moeten echt progress maken om opnieuw te spawnen
- Completetion-logica wordt consequenter

**Tick-loop timing:**
- Alle interval-triggers (bijv. risk-zone logging rond 500K, 1M) vuren op het correcte moment
- Geen grote gaten in de telemetrie waar een tick-sprong heeft plaatsgevonden
- Ensemble behavior meer predictable

**Zelf-evolutie:**
- `/self-modify` → compile → `restart_self` → binary swap werkt end-to-end
- Lucy kan daadwerkelijk haar eigen code herladen zonder handmatige tussenkomst

## Architectural impact

- `thinker.rs:274-278` — één regel toevoegen (early return)
- `main.rs:1202-1204` — één regel verwijderen (tick-overwrite)
- `self_modify.rs:65` — twee strings in ALLOWED_COMMANDS
- Geen schema-impact, geen API-wijziging

## Notes

- Fix A5: voorzichtig getest tegen explore-logica (logging happens, decide_action returns, tick continues naturally)
- Goal-dedup kan later verfijnd worden (bv. description-fuzzy-match voor varianten), maar hard block op duplicaten voorkomt escalatie
