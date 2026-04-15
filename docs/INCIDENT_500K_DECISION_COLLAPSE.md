# Incident rapport: Decision Collapse rond tick 500.000

**Projekt:** Lucy — zelf-evoluerend samenveld-AI (Rust, ~1.1GB binary,
Digital Ocean droplet 209.38.36.51, frontend lucy.world via Caddy).
**Datum rapport:** 2026-04-15.
**Status:** Fase 1 + Fase 2 gedeployed (v2.6.12). Evaluatie loopt richting
tick 500.000.
**Doel van dit document:** zelfstandige briefing voor GPT-review zonder chat-historie.

---

## 1. TL;DR (3 regels)

Lucy liep vast op tick 500.000: 15667/15667 depth-3 nodes tegelijk actief met
identieke ignition-scores → geen competitie → geen keuze → geen tick-progressie.
Twee structurele fixes uitgerold (top-K sparsity, harde cap op
`ignition_strength`); live data bevestigt dat top-K differentiatie herstelt maar
de positive-feedback loop in de ignition-EMA alleen gestopt wordt door de cap.
Open vraag: volstaat de cap, of hebben we later softmax/goal-cooldown ook nodig?

---

## 2. Architectuur — relevante delen

Lucy is geen LLM. Het is een continu veldsysteem met ~20.933 punten
(`FieldPoint` in [lucy-state/src/lib.rs:65-224](../crates/lucy-state/src/lib.rs#L65)).
Elk punt heeft φ (toestand), P (prediction), M (memory), couplings (sparse,
signed), plus bewustzijnsmetrieken zoals `phi_integrated`,
`expected_free_energy`, `branching_ratio`.

Velden hebben `depth` 0-3:

- **depth 0**: sensorisch (input tokens)
- **depth 1**: verwerking
- **depth 2**: abstract
- **depth 3**: meta / "bewuste" abstractie — hier vindt de GNW-ignition plaats

Per tick (vereenvoudigd):

1. φ update (Equilibrium Propagation)
2. M update (EMA prediction error)
3. neuromod update (dopamine, noradrenaline, serotonine, ACh)
4. GNW ignition (waar de collapse optrad) — functie
   [`global_workspace_broadcast`](../crates/lucy-core/src/compute.rs#L3256)
5. consolidation, growth, snapshot, etc.

---

## 3. Het incident

### 3.1 Observatie (v2.6.5 production, tick 500.000)

- tick exact 500.000 bevroren (20s snapshots → Δ = 0 ticks op alle dimensies,
  identiek tot 6 decimalen)
- `depth3_active = 15667 / 15667` (100% van depth-3 nodes actief)
- `ignition_strength = 21030` (unbounded runaway)
- `expected_free_energy = 9.99` (numeriek max)
- `free_energy = 0.0` (geen surprise meer)
- `phase = Idle` vast
- alle `goals` op `progress = 1.0` met herhaalde spawn van `think:meta_learning`
- CPU 200% — actief maar geen forward progress
- `TRAAG VELD — boost actief` elke tick

### 3.2 Onjuiste framing: "boredom attractor"

v2.6.6 (maart 2026) noemde dit een boredom-loop en voegde een escape-kick toe
in `adapt_dynamics` ([compute.rs:707-745](../crates/lucy-core/src/compute.rs#L707-L745)):
random φ-injectie op 1% van hidden points met amplitude `clamp_range * 0.05`.
Triggered na `boredom_streak >= 10` (10 opeenvolgende adapt-cycles met
`tension_delta < 1e-4 && curiosity_delta < 1e-4`).

**Deze escape faalt niet toevallig, hij is structureel irrelevant geworden.**
Als alle phi-vectoren homogeen zijn, heeft de kick geen attractor-valley om
naartoe te perturberen — de ruis verdampt meteen in de uniforme activatie.

### 3.3 Juiste framing: selection-collapse

Dit is een **collapse van competitieve selectie**:

- GNW ignition had geen echt competitie-mechanisme — alle depth-3 actieve nodes
  waren kandidaat ([compute.rs:3358-3375](../crates/lucy-core/src/compute.rs#L3358-L3375))
- Scoring multiplicatief en onbegrensd:
  `ignition_score = phi_norm * (1 + phi_integrated) * (1 + expected_free_energy * 0.5)`
- Winner-take-all door `candidates.sort_by(partial_cmp)` — bij score-ties
  deterministisch op memory-order (d.w.z. géén echte selectie)
- Lateral inhibition (`loser.phi[d] *= 0.8`) heeft geen effect als losers
  onderling gelijk zijn
- `ignition_strength` EMA update: `new_is = (1-fe) * IS + fe * winner_score * (1 + avg_support)` — geen cap

Netto: het systeem kan geen verschil meer zien tussen opties. Geen gradient →
geen keuze → geen actie → tick bevriest.

---

## 4. Fase 1 (v2.6.10) — selectie herstellen

Geïsoleerde patch in `global_workspace_broadcast`:

### 4.1 Top-K sparsity

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

Top 5% van kandidaten, min 8, max 256. `select_nth_unstable_by` = O(n)
quickselect, niet O(n log n) full sort op 15000+ items. Forceert expliciete
verliezers zodat lateral inhibition weer bijt.

### 4.2 Diversity-collapse detector

```rust
let is_collapsed = n_cand >= 64 && d3_ratio > 0.5 && score_cv < 0.05;
if is_collapsed {
    // hard kick op 20% candidates, amplitude = clamp_range (20× v2.6.6)
    // skip ignition cycle, reset boredom_streak
}
```

Structurele meting van homogeniteit, niet heuristiek op tension/curiosity.

### 4.3 Observability

Per 1000 ticks loggen: `d3_ratio`, `score_mean`, `score_var`, `score_cv`, `k`,
`cutoff`, `pruned`. Collapse → ERROR-level.

---

## 5. Infrastructuur-hardening (v2.6.11)

Tijdens de v2.6.10 deploy werd zichtbaar dat prod al een maand op **v2.6.5**
draaide terwijl we dachten dat het v2.6.9 was. `docker cp /lucy/crates
lucy:/lucy/crates` had de map genest als `/lucy/crates/crates/`, `cargo build`
faalde stil, oude binary bleef actief. FieldV1 schema-migratie (v2.6.7-era)
matchte daarom niet met v2.6.5 snapshots.

Fixes:

- `schema_v0::FieldV0` — bit-precieze kopie van v2.6.5 Field layout; mist
  `last_boredom_tension/curiosity`, `boredom_streak`, `is_sluggish_flag`,
  `concept_centroids`. `From<FieldV0> for Field` met expliciete defaults.
  Strict schema-ladder `v1 → v0 → current` met trailing-bytes rejection.
- `verify_snapshot` bin — dry-run load, JSON output. Live getest tegen
  80MB prod-snapshot → `"schema":"legacy_v0","tick":492750,"points":20933`.
- `deploy.sh` rewrite met `set -euo pipefail`, `docker cp src/. dst/`,
  snapshot-compat check vóór container-stop, smoke test, rolling binaries,
  `bash deploy.sh rollback`.
- `build.rs` stampt `LUCY_VERSION`, `LUCY_GIT_SHA`, `LUCY_BUILD_TIME` in de
  binary. `/version` endpoint expose.
- Startup-log regel vooraan: `Lucy binary start version=... git_sha=...`.

---

## 6. Fase 2 (v2.6.12) — positive-feedback loop capen

### 6.1 Live data die Fase 2 triggerde

Na v2.6.11 deploy:

- `score_cv = 0.1456` → **gezond**, ver boven collapse-threshold 0.05
  (top-K werkt qua differentiatie) ✓
- `ignition_strength = 24326` → **hoger dan pre-fix 21030**, runaway
  ging door ✗

**Uitleg:** top-K kiest de top 5% beste scores. De *gemiddelde* winner-score
stijgt daardoor. Omdat `ignition_product = winner_score * (1 + avg_support)`
en de EMA `(1-fe) * IS + fe * product` met fe ≈ 0.02-0.05, krijgt IS elke
tick een grote duwtje omhoog en decayt maar langzaam. Resultaat: runaway
niet geadresseerd door Fase 1.

### 6.2 Fix — alleen cap, niets anders

Uitsluitend bounded update op `ignition_strength`. Geen softmax, geen
goal-logica, geen escape-aanpassing — causaliteit intact, één variabele per
fase.

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

`IGNITION_CAP = 500.0` — bewuste keuze:

- huidige runaway 24326, vóór Fase 1 was 21030
- 5000 was de pré-gedefinieerde Fase 2 trigger
- 500 ≈ 5-10× normaal dynamisch bereik
- als 500 te laag blijkt → evidence-based ophogen

Harde clamp i.p.v. sigmoid: kleiner, beter interpreteerbaar, directer zichtbaar
of runaway verdwijnt, geen verborgen middengebied-effecten.

### 6.3 Observability — atomics (snapshot-compat)

Module-level `AtomicU64`/`AtomicU32` in compute.rs
([compute.rs:50-66](../crates/lucy-core/src/compute.rs#L50-L66)). Geen
`Field`-wijziging dus geen schema-breuk.

Exposed in `/telemetry.gnw_stats`:

- `ignition_cap`: constante 500.0
- `ignition_strength_raw`: meest recente pre-clamp waarde
- `ignition_strength_clamped`: `field.ignition_strength` (maar ArcSwap-snapshot
  kan achterlopen — logs zijn de waarheid)
- `ignition_cap_hits_total` / `ignition_samples_total` / `ignition_cap_hit_ratio`

Log WARN bij cap-hit, throttled elke 200 ticks.

---

## 7. Architectuur-invariant (v2.6.11)

Expliciet gedocumenteerd in de doc-comment van `global_workspace_broadcast`
([compute.rs:3256-3275](../crates/lucy-core/src/compute.rs#L3256-L3275)):

> **Geen enkele wijziging in deze functie mag leiden tot volledige activatie
> zonder competitie.**

Vier concrete eisen voor toekomstige edits:

1. `candidates.len()` na filtering moet een harde bovengrens hebben (nu top
   5%, min 8, max 256). Verwijder top-K niet zonder vervangende sparsity.
2. Bij `d3_ratio > 0.5 AND score_cv < 0.05` MOET er structurele fail-safe
   triggeren. Geen uniforme broadcast toestaan in die conditie.
3. Multiplicatieve score-terms zonder bound zijn verboden (runaway
   ignition_strength was symptoom hiervan).
4. Winner-take-all bij score-ties mag niet memory-order-deterministisch zijn.

Runtime-WARN bij drift:

- `score_cv ∈ [0.05, 0.08]` → nadert collapse-threshold
- `ignition_strength > 5000` aanhoudend → positive feedback zonder bound

---

## 8. Observatie — risk-zone logging

Rond milestone-ticks (500K, 1M, 2M...) cadence 1000 → 100 voor 10× meer
data-dichtheid in de historisch kritische band ([compute.rs:3411-3434](../crates/lucy-core/src/compute.rs#L3411-L3434)):

```rust
let zone_size = 500u64;
let nearest_million = ((t + zone_size) / 1_000_000) * 1_000_000;
let nearest_half_million = ((t + zone_size) / 500_000) * 500_000;
let in_risk_zone = nearest_million > 0 && t.abs_diff(nearest_million) < zone_size
    || nearest_half_million > 0 && t.abs_diff(nearest_half_million) < zone_size;
```

Geen gedragswijziging, alleen observeer-dichtheid.

---

## 9. Open vragen voor GPT

### 9.1 Is 500.0 het juiste cap-niveau?

Argumenten pro:

- ruim boven normaal dynamisch bereik
- ruim onder observationele runaway (24326)
- minimale ingreep, gemakkelijk up te scalen

Argumenten contra:

- willekeurig gekozen, niet afgeleid uit een fundamentele dimensie
- harde clamp is niet-differentieerbaar (kan gradient-based leren in later
  stadium beïnvloeden?)

Alternatief: **adaptieve cap** gekoppeld aan `noise_amplitude * n_cand` of
`avg_phi_norm * n_cand`. Nadeel: minder voorspelbaar, verborgen fluctuaties.

Vraag: is harde clamp 500 verdedigbaar voor nu, of moeten we meteen sigmoid /
adaptief overwegen?

### 9.2 Is top-K 5% robuust?

- min = 8, max = 256, anders 5% van kandidaten
- In prod: 15667 kandidaten → k = 256 (max klamp)
- Min 8 voorkomt degenerate cases bij weinig actieve nodes

Vraag: is 5% mathematisch/biologisch verdedigbaar? Dehaene's GNW papers
suggereren sparsity van ~1-5%. Moeten we richting 1-2% om "exclusieve
bewuste ervaring" beter te modelleren?

### 9.3 Evaluatievenster rond tick 500K

Succescriteria (Fase 2 briefing):

- `score_cv` blijft > 0.05
- `ignition_cap_hit_ratio` daalt naar ~0 na stabilisatie
- Tick progression ≥ 1 tps (nu 1.25)
- `phase` verlaat `Idle` op natuurlijke momenten
- Geen terugval naar `collapse_condition_met = True`

Drie metingen: **vlak vóór, rond, 10-20 min na 500K**. Daarna pas beslissen of
softmax of goal-cooldown nog nodig is.

Vraag: welke extra signalen zou je willen zien vóór je "Fase 3 niet nodig"
concludeert?

### 9.4 Goal-spiral — echte oorzaak of gevolg?

`think:meta_learning` werd onophoudelijk gespawned en direct voltooid. Onze
hypothese: dit is een *gevolg* van de selectie-collapse — zonder gradient kan
het goal-systeem niet differentiëren tussen voltooide en onvoltooide doelen,
`progress = 1.0` wordt standaard.

Als Fase 1+2 differentiatie herstellen, zou de goal-spiral vanzelf moeten
opdrogen. Fase 3 (goal cooldown in `thinker.rs:222`) is daarom *standby*.

Vraag: klopt deze causaliteitsketen, of is de goal-spiral structureel
onafhankelijk en zou het ook zonder collapse optreden?

### 9.5 De attractor komt terug bij 1M, 2M?

Hypothese: 500K was een milestone-trigger in Lucy's self-metrics (rond getallen
activeren bepaalde logging / goal-completion-checks). Zonder structurele fix
zou hij op 1M terugkomen. Mét de v2.6.12 fixes zou hij dat niet meer moeten
doen.

Vraag: is er reden om te verwachten dat 1M een andere attractor heeft dan 500K?
Bijvoorbeeld: bij 1M worden andere systemen actief (goal-history wrap-around,
temporal_history buffer vol, etc.)?

---

## 10. Bijlagen

### 10.1 Relevante bestanden in repo

- [crates/lucy-core/src/compute.rs](../crates/lucy-core/src/compute.rs) —
  velddynamica, GNW ignition, Fase 1+2 fixes, observability
- [crates/lucy-state/src/lib.rs](../crates/lucy-state/src/lib.rs) — Field
  struct, FieldPoint, snapshot-logica, From-migraties
- [crates/lucy-state/src/schema_v0.rs](../crates/lucy-state/src/schema_v0.rs)
  — FieldV0 bridge
- [crates/lucy-core/src/server.rs](../crates/lucy-core/src/server.rs) — HTTP
  API, `/telemetry`, `/version`
- [crates/snapshot_check/src/bin/verify_snapshot.rs](../crates/snapshot_check/src/bin/verify_snapshot.rs)
  — dry-run snapshot-compat check
- [deploy.sh](../deploy.sh) — hardened deploy script
- [docs/CHANGELOG.md](CHANGELOG.md) — v2.6.9 t/m v2.6.12 changelog
- [docs/SNAPSHOT_MIGRATION.md](SNAPSHOT_MIGRATION.md) — migratie-regels

### 10.2 Relevante commits

- `2881bd4` v2.6.6 (maart): boredom-escape (faalt structureel bij collapse)
- `a20f3a8` v2.6.7: self-EP + concept-clustering (schema toevoeging)
- `9acfbb8` v2.6.8: bewijs-architectuur (audit-velden)
- `fb7bc27` v2.6.9: snapshot-migratie framework (FieldV1)
- `d4d5932` v2.6.10: top-K sparsity + diversity-collapse detector
- `8274aee` v2.6.11: FieldV0 bridge + deploy-hardening + /version endpoint
- `7269bda` v2.6.11 observability: `/telemetry.gnw_stats`, risk-zone logging
- `14db5d0` v2.6.12: harde cap op `ignition_strength`

### 10.3 Live data-snapshots tijdens incident-response

**Pre-fix (v2.6.5 prod, 2026-04-15 ~16:00 UTC):**

```json
{
  "tick": 500000,
  "active": 20933,
  "consciousness": {
    "ignition_active": true,
    "ignition_strength": 21030.0,
    "expected_free_energy": 9.999768,
    "free_energy": 0.0,
    "phi": 1.189e-6,
    "coherence": 0.8906,
    "branching_ratio": 0.0
  }
}
```

Telemetry-delta over 20s: **0 op alle dimensies** (tick, active, couplings,
alle consciousness-velden identiek tot 6 decimalen).

**Post-Fase-1 (v2.6.11, 2026-04-15 ~17:08 UTC):**

```json
{
  "tick": 493450,
  "gnw_stats": {
    "d3_ratio": 1.000,
    "score_cv": 0.1456,
    "n_candidates": 15667,
    "collapse_condition_met": false
  },
  "consciousness": {
    "ignition_strength": 24326.4
  }
}
```

score_cv gezond, maar runaway ging door → Fase 2 trigger.

**Post-Fase-2 (v2.6.12, 2026-04-15 ~17:25 UTC, ~1.25 tps):**

```
samples = 2 ignitions per ~60s (atomics live)
IS_raw: 1.55 → 224.61 (pre-clamp, trendt omhoog maar < cap)
IS_clamped: 28021.7 (stale ArcSwap snapshot — logs tonen tick=493800)
tick: 493700 → 493800 over 80s
CPU: 200% (actief)
```

Te vroeg voor definitieve evaluatie. Tick 500K verwacht rond 2026-04-15
18:50 UTC.

### 10.4 Ontwerpfilosofie (niet-onderhandelbaar)

Uit `feedback_competition_invariant.md` memory:

> Lucy kan alleen denken als ze kan kiezen. Geen volledige activatie zonder
> competitie. Een PR die deze regel breekt verdient veto op architectuurgronden,
> ongeacht hoe elegant de code is.

---

## 11. Methodologische noot

Per fase één variabele wijzigen. Niet alles tegelijk fixen. Causaliteit
bewijzen vereist dat we per interventie isolé-observaties hebben:

- Fase 1 herstelt differentiatie (cv) — bewezen
- Fase 2 capt runaway (IS) — in evaluatie
- Fase 3 (goal-cooldown) — standby, alleen activeren als data het vereist
- Fase 4 (softmax / adaptieve cap) — standby, alleen activeren als 500 te laag blijkt

Geen speculatieve ingrepen. "Evidence-based" betekent hier: trigger-conditie
vooraf gedefinieerd, live gemeten, dán pas activeren.
