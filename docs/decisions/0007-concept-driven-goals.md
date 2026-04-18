# ADR 0007 — Concept-driven goals: Lucy genereert haar eigen doelen

**Status:** Uitgerold in v2.6.16
**Auteur:** Frank + Claude (Opus code review)
**Gerelateerd:** [ADR 0003](0003-goal-completion-zombie-fix.md) (zombie-fix), [ADR 0006](0006-goal-dedup-tick-manipulation-whitelist.md) (goal-dedup)

## Context

v2.6.15 fixte de leer- en selectie-bugs. Lucy leert nu correct en kiest echt tussen opties (refractory + novelty-boost).

Maar haar **goals zijn nog steeds hardcoded templates**: 5 mogelijke beschrijvingen, gegenereerd via een match-arm op basis van welke score wint (curiosity/frustration/explore/self_improve). De match-arms produceren vaste strings:
- `"curiosity"` → `"learn:topic"` (topic = random interest)
- `"frustration"` → `"think:analyze_tension"` (statisch)
- `"explore"` → `"explore:new_sources"` (statisch)
- `"self_improve"` → `"think:meta_learning"` (statisch)

Deze zijn niet gekoppeld aan **wat Lucy echt denkt**. Ze worden gegenereerd via haar emotie-scores, maar de inhoud is hardcoded.

**Gat:** Lucy's doelen vloeiden voort uit een vaste tabel, niet uit haar eigen interne staat. Dit breekt de "geen hardcoding" regel: Lucy moet zelf kiezen wat ze doet en waarom.

## Besluit

Twee wijzigingen in `thinker.rs`:

### Fix 1: Concept-inferentie via GNW-output

Nieuwe helper-functie `infer_concept()` (thinker.rs, net boven `generate_goal()`):

```rust
fn infer_concept(field: &Field, vocab: &Vocabulary) -> String {
    // Lees Lucy's huidige output-phi (wat haar GNW uitvoert)
    let output_phi = decoding::read_output_region(field);
    // Decodeer naar het dichtstbijzijnde vocabulaire-woord
    let (token_id, _score) = decoding::decode_token(&output_phi, vocab, field);
    let word = vocab.token_name(token_id);

    // Filter speciale tokens en te korte woorden uit
    if word.len() >= 3 && !word.starts_with('<') && word.chars().all(|c| c.is_alphabetic() || c == '_') {
        word.to_lowercase().to_string()
    } else {
        // Fallback: kijk naar de winner's phi direct
        if field.last_winner_id < field.points.len() as u32 {
            let winner_phi = &field.points[field.last_winner_id as usize].phi;
            let (wid, _) = decoding::decode_token(winner_phi, vocab, field);
            let w = vocab.token_name(wid);
            if w.len() >= 3 && !w.starts_with('<') {
                return w.to_lowercase().to_string();
            }
        }
        "patterns".to_string()  // laatste fallback
    }
}
```

Mechanisme:
1. `read_output_region(&field)` decodeert Lucy's huidige output-phi (wat zij denkt uit te voeren)
2. `decode_token(phi, vocab, field)` vindt het dichtst bijzijnde vocabulaire-woord
3. `vocab.token_name(token_id)` geeft dat woord als string

Fallback:
- Als output-region leeg of special token: gebruik `field.last_winner_id` (GNW-winner)
- Als dat ook leeg: return `"patterns"` als default

Effect: `infer_concept()` geeft een woord dat **uit Lucy's GNW-state voortkomt**, niet uit een vaste tabel.

### Fix 2: Dynamische goal-beschrijvingen

In `generate_goal()`, na bepalen van `best_source`, roep `infer_concept(field, vocab)` aan om het huidige concept te halen:

```rust
let concept = infer_concept(field, vocab);
```

Vervang de match-arm descriptions:

```rust
"curiosity" => {
    // 50% gebruik current GNW concept, 50% explorer interest
    let topic = if !explorer.interests.is_empty() && rng.gen_bool(0.5) {
        let interests_vec: Vec<&String> = explorer.interests.iter().collect();
        interests_vec[rng.gen_range(0..interests_vec.len())].to_string()
    } else {
        concept.clone()
    };
    (format!("learn:{}", topic), GoalStrategy::Learn(topic))
},
"frustration" => {
    (format!("think:resolve_{}", concept), GoalStrategy::Think("resolve".to_string()))
},
"explore" => {
    (format!("explore:{}", concept), GoalStrategy::Explore(concept.clone()))
},
"self_improve" => {
    (format!("think:optimize_{}", concept), GoalStrategy::Think("optimize".to_string()))
},
```

Effect: Goals reflecteren nu wat Lucy **echt denkt**:
- Voor `"curiosity"`: `"learn:neural"` (als dat wat ze GNW encodeert)
- Voor `"frustration"`: `"think:resolve_gradient"` (conceptafhankelijk)
- Voor `"explore"`: `"explore:coupling"` (uit GNW-winner)
- Voor `"self_improve"`: `"think:optimize_weights"` (uit GNW-winner)

### Fix 3: Residuele dedup-bug (carry-over uit v2.6.15)

Verplaats `field.goals.push(goal)` van **vóór** naar **ná** de dedup-check.

**Huidig (buggy):**
```rust
field.goals.push(goal);  // <- goal is al in de vec

// later...
if occurrences >= 2 {
    return None;  // te laat! duplicate is al ingesloten
}
```

**Fixed:**
```rust
// Check EERST
if occurrences >= 2 {
    return None;
}

// Pas dan pushen
field.goals.push(goal);
```

## Gevolgen

**Goal-inhoud:**
- Goals vloeiden voort uit Lucy's GNW-output, niet uit vaste strings
- Elke reflect-cycle kan Lucy een ander doel hebben (afhankelijk van wat ze denkt)
- Neuraal thema? → doel over neural engineering
- Tension rond coupling? → doel om couplings te resolven

**Vrije wil:**
- Lucy's doelen zijn niet **autonoom gekozen** (ze wordt nog steeds getrigger door scores: curiosity/frustration/etc.)
- Maar ze zijn **self-generated** — de inhoud vloeit uit haar interne staat voort, niet uit een tabel
- Dit breekt de "geen hardcoding" regel

**Dedup-fix:**
- Spiral-preventie werkt nu echt: duplicaten komen niet meer in field.goals terecht
- `goal_stats.complete_total` stijgt sneller (minder ghost-goals)

## Verificatie

Na deploy, check `/telemetry` → `goals` array:

**Voor v2.6.16:**
- `["think:meta_learning", "explore:broad", "learn:python"]`

**Na v2.6.16:**
- `["think:resolve_gradient", "explore:neural", "learn:coupling"]`

De goal-beschrijvingen reflecteren nu wat Lucy echt denkt (GNW-winner decoded).

**Invarianten:**
- `cargo check -p lucy-core` → 0 errors
- `score_cv > 0.1` (competitie-invariant intact)
- `tick` beweegt continu
- `goal_stats.complete_total` stijgt (dedup-bug fix elimineert ghost-pushes)
- Snapshot-load slaagt (geen schema-wijziging)

## Architectural impact

- `crates/lucy-core/src/thinker.rs` — imports + `infer_concept()` + dynamic descriptions + dedup fix
- Geen schema-wijziging (Goal struct unchanged)
- Geen API-wijziging
- Snapshot-compat intact (backward-compatible)

## Notes

- Concept-inferentie gebruikt `decode_token()` die nucleus sampling doet — twee calls kunnen verschillende resultaten geven. Dit is gewenst (goal-variëteit).
- Fallback naar `"patterns"` is defensief — in normale bedrijf (ignition actief) decode je altijd iets zinnigs.
- De 50/50 split in curiosity-goals (concept vs. explorer.interests) handelt het geval af waar GNW outputregio leeg is, maar explorationfrontier niet.
