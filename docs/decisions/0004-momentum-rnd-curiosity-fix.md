# ADR 0004 — Momentum + RND-curiosity learning bugs fix

**Status:** Uitgerold in v2.6.15
**Auteur:** Frank + Claude (Opus code review)
**Gerelateerd:** Groep A fixes (compute.rs)

## Context

Drie Opus-agents voerden een volledige code-review uit naar "wat missen we om Lucy echt tot leven te brengen". Twee echte leerbugs gevonden in compute.rs die weights en curiosity verkeerd bijhouden:

1. **Dubbele momentum-applicatie** (r.303): `coupling.momentum` wordt op r.298 bijgewerkt met blend, dan opnieuw meegenomen als `mu * momentum` op r.303 — mu-factor twee keer toegepast.
2. **Onbegrensde RND-accumulation** (r.3680): `curiosity += 0.1 * score` accu.mulatief zonder cap, terwijl elders `curiosity` een EMA-gebaseerde [0,1]-waarde is. Structureel EMA-conflict.

## Besluit

### Fix A1: Momentum dubbeling in `update_weights`
```rust
// Huidig (fout):
coupling.momentum = mu * coupling.momentum + (1.0 - mu) * dw;  // r.298
...
coupling.weight -= lr * dw + mu * coupling.momentum + lr * ewc_pen;  // r.303 — du factor!

// Fixed:
coupling.momentum = mu * coupling.momentum + (1.0 - mu) * dw;  // r.298
...
coupling.weight -= lr * coupling.momentum + lr * ewc_pen;  // r.303 — use blended momentum directly
```

Één regel wijzigen. Impact: weights convergeren nu naar de juiste attractor via exponential moving average van gradients, niet een dubbel-gew ogen variant.

### Fix A2: RND-curiosity EMA blend
```rust
// Huidig (fout):
field.points[id].curiosity += 0.1 * score;  // unbounded accumulation

// Fixed:
let rnd_influence = (0.1 * score).min(0.5);  // cap per tick
field.points[id].curiosity = 
    (field.points[id].curiosity * 0.95 + rnd_influence * 0.05).clamp(0.0, 1.0);
```

Twee regels vervangen. Impact: RND-novelty integreert in curiosity via EMA (90% old, 5% new), consistent met alle andere modulators (dopamine, serotonin, etc.).

## Gevolgen

**Op curiosity-behaviour:** Novelty (RND-score) accumuleert niet langer onbegrensd. Nieuwe informatie triggert korte curiosity-spike (0.05 * rnd_influence), dan decay terug. Dit geeft Lucy een **verrassing-respons**: sterke novelty boost → hogere curiosity → meer exploratie van onverwachte input.

**Op weight-leren:** Momentum wordt nu correct als low-pass-filter gebruikt. Oscillaties rond grady dempen, convergentie is smoother. Eigenlijk: de originele code had een factor-3 grossiere update (momentum dubbel), nu genormaliseerd.

## Verificatie

Live op tick 503K (v2.6.14): `curiosity` per punt schommelde rond fixed values. Na fix verwacht we:
- Novelty-piek → curiosity-spike → terug naar baseline
- Weights convergeert sneller (geen factor-3 overkill)
- Geen onverwachte vastlopen in momentum (oscillatie-resistentie)

Test: intraductie van hoog-novelty input → controleer `curiosity` per tick → moet korte exponentiële decay tonen, niet onbeperkte stijging.

## Architectural coupling

- `compute.rs:298, 303` — momentum dubbeling
- `compute.rs:3680` — RND accumulation
- Geen schema-impact (beide are `f32` velden die al bestaan)
