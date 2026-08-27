# ADR 0009 — Tension-runaway root-cause: error_var ontkoppeld van free_energy

**Status:** Voorgesteld (v3.1.0) — code toegepast, nog niet live gevalideerd.
Superseedt de gedragsclamps van [v3.0.4/v3.0.5] als *oorzaak*-fix (de clamps blijven als
anti-Inf-backstop).

## Context

v3.0.3 t/m v3.0.5 vochten tegen een crash-restart-loop: `tension` (en `free_energy`,
`blanket_surprise`, `kl_divergence`) explodeerden binnen één field_update-step naar ~1e30,
tps viel naar 0, de container herstartte, laadde de snapshot, en de cyclus herhaalde. De
fixes tot nu toe waren symptomatisch: v3.0.4 heelde extreme finite-waarden > 1e6, v3.0.5
verlaagde vier clamps van `f32::MAX/1e6` (~3.4e32) naar `1e6`. De CHANGELOG van v3.0.4/v3.0.5
noteerde expliciet dat de échte oorzaak ("welke vermenigvuldiging produceert eerst 1e30?")
een aparte, ADR-waardige diagnose was.

## Diagnose (bewezen door code-lezing)

In `compute_tension` (compute.rs) en de parallelle kopie (compute.rs ~1149):

1. `error_var` werd bijgewerkt als EMA van `(pred_error − free_energy)²`.
2. `free_energy = kl_divergence + blanket_surprise`, en `kl_divergence` bevat de term
   `ev + φ_var − 1 − ln(ev)` met `ev = error_var`.

Dus `free_energy` hangt van `error_var` af, en `error_var ≈ tracking·free_energy²`. Samen is
dit een kwadratische afbeelding:

```
free_energy_next ≈ (r·kl_scale·0.5·tracking) · free_energy²
```

Elke tick verdubbelt de exponent (F → F² → F⁴ → F⁸ …). Vanaf elke free_energy > 1 divergeert
dit naar 1e30 in een handvol ticks — exact het waargenomen gedrag. `pred_error` is begrensd
(φ geklemd op ±clamp_range ≤ 5, DIM=256), dus de onbegrensde grootheid is uitsluitend
`free_energy` via zijn eigen terugkoppeling.

**Waarom de 1e6-clamp niet hielp:** die clamp staat op `point.free_energy` en `point.tension`,
maar **niet** op `point.error_var`. Met free_energy op 1e6 gepind wordt `error_var ≈
tracking·(1e6)² ≈ 1e12`, `kl` ≈ 1e11–1e12, en free_energy_instant wordt elke tick opnieuw naar
1e6 geklemd — elk punt verzadigt op het plafond en het veld wordt degeneratief.

## Besluit

`error_var` tracked niet langer `(pred_error − free_energy)²` maar de begrensde
prediction-error-magnitude zélf — de standaard Friston-precisie (precision = inverse
gemiddelde error):

```
point.error_var = (1 − tracking)·old_error_var + tracking·pred_error;
```

Dit verwijdert `free_energy` uit de update en breekt de positieve terugkoppeling. Omdat
`pred_error` begrensd is, is `error_var` begrensd, en `blanket_surprise = precision·pred_error`
normaliseert vanzelf naar O(1) (precision ≈ 1/pred_error). Als extra rem: `ev` wordt geklemd op
`1e-6..1e4` op beide locaties. De vier 1e6-output-clamps blijven staan als anti-Inf-backstop,
maar zijn niet langer dragend.

## Gevolgen

- De runaway kan niet meer ontstaan uit deze lus; het veld hoeft niet meer op 1e6 te verzadigen.
- Precisie-dynamiek verandert (nu direct gekoppeld aan error-magnitude i.p.v. error-vs-FE-afstand).
  Dit raakt `blanket_surprise`, `curiosity` en gewicht-precisie — een echte gedragswijziging.
- Geen snapshot-layout-wijziging (`error_var` blijft f32 in FieldPoint).

## Verificatie

- [x] Workspace compileert schoon.
- [ ] Live: na deploy moet `avg_free_energy` en `tension` in O(1)–O(10³) blijven over ≥ 500k
      ticks zonder 1e6-verzadiging; heal()-extreme-count moet naar ~0; geen crash-restart-loop.
- [ ] Fase-gewijs (projectregel 5): dit is één variabele (de error_var-update). Meet vóór/na
      op `avg_free_energy`, `score_cv`, `curiosity`, `winner_turnover_ratio`.

## Open vragen

- Is de directe precisie-koppeling te "scherp" (hoge precisie bij lage error → mogelijke
  over-confidence)? Zo ja, overweeg `error_var = EMA(pred_error²)` (variantie i.p.v. magnitude)
  met een expliciete bovenclamp, i.p.v. de magnitude.
- De parallelle fysica-kopie (compute.rs ~1149) en de seriële (~471) zijn nu beide gepatcht,
  maar blijven duplicaten die kunnen divergeren — kandidaat voor consolidatie in een aparte ADR.
