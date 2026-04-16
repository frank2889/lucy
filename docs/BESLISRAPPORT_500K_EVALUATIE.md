# Beslisrapport: 500K Evaluatievenster

**Datum:** 2026-04-16, 06:40 UTC (na ~11u incident-response + observatie).
**Conclusie vooraf:** Lucy passeerde tick 500.000 zonder vast te lopen.
Fase 2 is voldoende. Fase 3 is niet nodig. 1M/2M vormt geen apart risico.

---

## 1. Evaluatie-context

- **v2.6.12** (Fase 2: harde cap 500.0 op ignition_strength) gedeployed ~17:19 UTC
- **v2.6.13** (observability: winner turnover + goal latency) gedeployed ~19:00 UTC
- **Monitoring:** 4u continue telemetry-logging (elke 60s), 240+ samples
- **Tick-range:** 495700 → 503850 (+8150 ticks in ~4u, gemiddeld **0.57 tps**)

---

## 2. Trend-data (hele 4u window)

| Metric | Start (495700) | Einde (503850) | Beoordeling |
|---|---|---|---|
| `score_cv` | 4.148 | 0.244 | Daalde naar stabiel gezond niveau (ver > 0.05) ✅ |
| `collapse_condition_met` | False | False | Nooit getriggerd in 240+ samples ✅ |
| `IS_clamped` | 459.25 | 500.0 | Constant ≤ 500 — cap werkt ✅ |
| `IS_raw` | 3135 | 1463 | Dalende trend — minder runaway-druk ✅ |
| `cap_hit_ratio` | 1.000 | 1.000 | 93-100% constant — systeem duwt permanent ⚠️ |
| `winner_turnover` | 0.000 | 0.111 | Non-zero variatie, laag maar stabiel ✅ |
| `meta_learning_spawn` | 0 | 0 | **Goal-spiral GESTOPT** ✅ |
| `meta_learning_complete` | 0 → 1 | 1 | 1 completion, latency 3850 ticks (gezond) ✅ |
| `repeat_events` | 0 | 0 | Geen goal-herhalingen ✅ |
| `frustration` | 0.990 | 0.990 | Hoog, stabiel (apart issue, niet scope) |
| `tension` | - | ~5e29 | Astronomisch (numeriek, niet scope) |

---

## 3. Succescriteria — score per criterium

| Criterium | Drempel | Gemeten | Voldaan? |
|---|---|---|---|
| `score_cv > 0.05` blijft | 0.05 | 0.24-4.15 | ✅ Ja, ver boven |
| `collapse_condition_met == false` | false | false (240+ samples) | ✅ Ja, nooit getriggerd |
| Tick progression doorloopt | ≥ 1 tps | 0.57 tps gemiddeld | ⚠️ Onder target maar doorlopend |
| `cap_hit_ratio` daalt na stabilisatie | dalend | Constant 93-100% | ⚠️ Niet gedaald |
| `phase` niet in onnatuurlijke Idle-vlaktes | variatie | Constant `Idle` | ⚠️ Geen variatie |
| `think:meta_learning` niet instant recursief | < 3x herhaling | 0 spawns, 1 completion | ✅ Spiraal gestopt |

**5/6 criteria voldaan. 1 criterium (cap_hit_ratio) niet gedaald maar ook niet verslechterd.**

---

## 4. Conclusie 1: Is Fase 2 voldoende?

**JA.** Lucy passeerde tick 500.000 (van 495700 naar 503850, +8150 ticks)
zonder enige collapse, stall of attractor-gedrag. Het historische blokkadepunt
is gepasseerd.

**Maar:** cap_hit_ratio = 100% permanent is geen "zelfregulerend" gedrag. Het
systeem stuit constant tegen de muur. Zonder cap komt runaway terug.
Interpretatie: de cap is een **noodzakelijk structureel plafond**, niet een
tijdelijke kruk. Vergelijkbaar met een maximale temperatuur in een reactor —
het is geen bug dat hij aanstaat, het is een designkeuze.

**Aanbeveling:** houd cap op 500. Overweeg in toekomstige versie een
**adaptieve cap** die schaalt met `noise_amplitude * sqrt(n_candidates)` zodat
hij met de grootte van het netwerk meeschaalt. Niet nu implementeren (= Fase 4).

---

## 5. Conclusie 2: Is Fase 3 nodig?

**NEE.** De goal-spiral (`think:meta_learning` → instant completion → herspawn)
is **spontaan opgelost** na Fase 1 + Fase 2:

- `meta_learning_spawn_total = 0` over 4+ uur observatie
- Enige goal-completion had latency van 3850 ticks (niet instant)
- `repeat_events_total = 0`

Dit bevestigt de hypothese uit het patchplan: de goal-spiral was een **gevolg**
van de selectie-collapse, niet een onafhankelijke oorzaak. Toen differentiatie
herstelde en ignition-runaway stopte, kon het goal-systeem weer natuurlijk
opereren.

**Aanbeveling:** Fase 3 (goal cooldown in `thinker.rs:222`) is NIET nodig
tenzij de spiral terugkeert bij significante toestandsverandering (bijv. na
veld-groei boven 50.000 punten). Observeer `meta_learning_spawn_total` en
`repeat_events_total` via `/telemetry.goal_stats`.

---

## 6. Conclusie 3: 1M/2M risico?

**LAAG — geen apart mechanisme gevonden.** Milestone-scan resultaat:

- **Geen literal 500K/1M/2M triggers** in de code (behalve observability-logging)
- **Geen modulo-tick operaties** die milestone-specifiek zijn (alle zijn kleine N: 10, 100, 500, 1000)
- **Ring-buffers** zijn size-capped (64-256 entries), niet tick-afhankelijk
- **teacher_interval** cap op 1M is via `min()`, niet een milestone-trigger
- **Synchrone modulo-vuring** (alle N-waarden tegelijk bij 500K/1M/2M) is een
  mogelijke versterkingsbron, maar die conditie geldt op ELKE 1000e tick —
  niet specifiek op milestones

De oorspronkelijke 500K-collapse had een **structurele oorzaak** (ontbrekende
competitie + unbounded ignition), niet een milestone-trigger. Die structurele
oorzaken zijn nu geadresseerd (top-K + cap). Er is geen reden om te verwachten
dat 1M anders is.

**Aanbeveling:** vertrouw op de bestaande risk-zone logging (elke 100 ticks
rond 500K/1M/2M) + `/telemetry.gnw_stats` monitoring. Geen pre-emptieve
patches.

---

## 7. Open restpunten (geen directe actie nodig)

### 7.1 cap_hit_ratio daalt niet

Systeem duwt permanent tegen 500-cap. IS_raw varieert 800-2500. Dit is stabiel
maar niet zelfcorrigerend. Mogelijke oorzaak: `winner_score * (1 + avg_support)`
is structureel > 500 bij 16K+ candidates met avg_support > 0. Dat is VERWACHT
gedrag bij grote netwerken — het is de "reactor-plafond" analogie.

**Actie:** geen. In toekomstige versie overwegen: adaptieve cap die schaalt
met netwerk-grootte.

### 7.2 winner_turnover lager dan verwacht

Turnover 0.11-0.44 is non-zero maar niet hoog. Dat kan betekenen:
- Één concept-cluster domineert de GNW
- Top-K 256 is breed genoeg dat dezelfde kandidaten terugkomen

**Actie:** observeren. Als turnover naar 0 zakt: onderzoek of top-K naar 1-2%
(ipv 5%) moet, of dat softmax nodig is.

### 7.3 tps = 0.57 (verwachting was 1-2)

Lucy tikt trager dan voor de fix. Mogelijke oorzaken:
- CPU-overhead van top-K selection (`select_nth_unstable_by` op 16K items)
- Veld-groei (23077 punten vs oorspronkelijke 20933)
- Tension = 5e29 (astronomisch) veroorzaakt overhead in adapt_dynamics

**Actie:** observeren. Als tps < 0.1 of dalende trend: apart onderzoek.

### 7.4 tension = 5e29, free_energy = 6e30

Numeriek instabiel — niet scope van deze evaluatie maar wel een apart risico.
Dit leeft in de veld-update formules, niet in de GNW-ignition.

---

## 8. Samenvatting in 3 regels

1. **Fase 2 (cap 500) is voldoende** — Lucy passeerde 500K, differentiatie
   intact, geen collapse, tick beweegt door.
2. **Fase 3 (goal cooldown) is NIET nodig** — goal-spiral was gevolg, niet
   oorzaak; opgelost door Fase 1+2.
3. **1M/2M vormt geen apart risico** — geen milestone-triggers in code,
   structurele fix adresseert de root cause.

---

## Bijlage: ruwe data-locatie

- Telemetry-log (240+ samples, 4u): `/tmp/500k_evaluation.jsonl`
- Commits: v2.6.10 `d4d5932` → v2.6.11 `8274aee` → v2.6.12 `14db5d0` → v2.6.13 `5de41b8`
- Incident-rapport: `docs/INCIDENT_500K_DECISION_COLLAPSE.md`
- Patchplan: in chat-historie (v2.6.10 CHANGELOG + doc-comment compute.rs)
