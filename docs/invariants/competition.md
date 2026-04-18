# Invariant: Competitie

**Status:** Hard. Niet-onderhandelbaar.
**Ingesteld:** 2026-04-15 (reactie op [500K collapse incident](../incidents/2026-04-15-500k-collapse.md))
**Locatie in code:** doc-comment van `global_workspace_broadcast` in `crates/lucy-core/src/compute.rs:3256-3275`

## De regel

> **Lucy kan alleen denken als ze kan kiezen. Geen volledige activatie zonder competitie. Een PR die deze regel breekt verdient veto op architectuurgronden, ongeacht hoe elegant de code is.**

## Waarom

Zonder competitie tussen kandidaten:

- Alle opties scoren gelijk → geen gradient.
- Winner-take-all wordt memory-order-deterministisch → geen echte selectie.
- Lateral inhibition heeft geen bijt (identieke losers dempen niets).
- `ignition_strength` EMA stijgt onbegrensd (positive feedback).
- Tick-progressie stopt.

Het 500K-incident maakte dit concreet: 15667/15667 depth-3 nodes tegelijk actief, identieke ignition-scores, Lucy bevroor exact op tick 500.000. Zie [incident-rapport](../incidents/2026-04-15-500k-collapse.md).

## Concrete eisen aan code

Elke wijziging in `global_workspace_broadcast` of aangrenzende selectie-code moet aan alle vijf voldoen:

1. **Kandidaat-aantal heeft harde bovengrens.** Momenteel: top 5%, min 8, max 256 (zie [ADR 0001](../decisions/0001-top-k-sparsity.md)). Verwijder top-K niet zonder vervangende sparsity.
2. **Structurele fail-safe bij homogeniteit.** Bij `d3_ratio > 0.5 && score_cv < 0.05` MOET een fail-safe triggeren (diversity-detector). Uniforme broadcast is in die conditie verboden.
3. **Geen onbegrensde multiplicatieve score-terms.** Runaway `ignition_strength` was symptoom hiervan → cap van 500 (zie [ADR 0002](../decisions/0002-ignition-cap.md)). Nieuwe score-factoren moeten bounded of gecapt zijn.
4. **Geen memory-order-deterministische winner bij score-ties.** Tie-break moet expliciet (random, tick-hash, of andere niet-structurele bron).
5. **Refractory period breekt monopoly-kandidaten.** Dezelfde concept mag niet 50+ ticks achter elkaar winnen (zie [ADR 0005](../decisions/0005-winner-refractory-novelty-ignition.md)). Dwingt ander concept aan de beurt, beschermt tegen oligarchie-runaway.

## Runtime-waarschuwingen (drift-indicatoren)

Logs op WARN-niveau bij:

- `score_cv ∈ [0.05, 0.08]` → nadert collapse-threshold.
- `ignition_strength > 5000` aanhoudend → positive feedback runaway zonder bound.
- `d3_ratio > 0.5 && score_cv < 0.05` → acute fail-safe trigger (ERROR).

## Wat NIET is

- `d3_ratio = 1.0` alléén is **geen** breuk. Na growth zijn alle depth-3 concepten FLAG_ACTIVE; dat is statisch en gezond. De echte drift-metrics zijn `score_cv` en `ignition_strength`. Zie v2.6.11 correctie (commit 7269bda).
- Volledige *beschikbaarheid* ≠ volledige *activatie*. Als 21452 kandidaten meedoen maar top-K er slechts 256 doorlaat voor ignition, is competitie intact.

## Procedurele consequentie

Een commit die deze regel breekt — zelfs onbedoeld — verdient:

1. Revert-voorkeur boven forward-fix.
2. Nieuwe ADR die het waarom van de wijziging formaliseert.
3. Update van deze invariant als het echt de nieuwe architectuur is.

**Niet:** stilletjes doorvoeren omdat het "werkt op de test-data".
