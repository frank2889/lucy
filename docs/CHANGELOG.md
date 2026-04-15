# Lucy Changelog

## v2.6.10 (2026-04-15) — Decision-collapse fix: top-K sparsity + diversity detector

**Doel:** Lucy zat op tick 500.000 bevroren in een **harde attractor** (0 ticks in
20s, alle telemetrie-waarden identiek tot op 6 decimalen). Structurele diagnose:
geen boredom, maar een **decision collapse** — het veld kon geen verschil meer
zien tussen kandidaten, dus geen gradient, dus geen keuze, dus geen actie.

**Symptomen in prod:**

- `depth3_active = 15667/15667` (100%) → alle kandidaten tegelijk actief, geen sparsity
- `ignition_strength = 21030` (unbounded, geen cap)
- `expected_free_energy = 9.99` (max), `free_energy = 0.0` (geen surprise meer)
- Alle goals `progress = 1.0` → systeem dacht dat alles al bereikt was
- `phase=Idle`, `tps=1`, maar CPU 200% (draaide wel, kwam nergens)
- v2.6.6 boredom-escape triggerde wel, maar 5%-kick verdampte in homogeen veld

**Root cause (compute.rs `global_workspace_broadcast`):**

1. Alle actieve depth-3 nodes waren kandidaat voor ignition — geen top-K filter
2. Lateral inhibition (`loser.phi *= 0.8`) heeft geen effect als losers onderling
   gelijk zijn — winner-take-all wordt willekeurig (sort-op-memory-order bij ties)
3. Boredom-detectie keek alleen naar tension/curiosity delta's (heuristiek), niet
   naar de structurele degeneratie (spread van ignition-scores zelf)

### Fix (twee lagen, structureel)

**1. Top-K sparsity op ignition-kandidaten** (`compute.rs` ~3260)
Vervangt "alle actieve depth-3" door **top 5%** (min 8, max 256). O(n)
`select_nth_unstable_by` i.p.v. full sort op 15000+ items. Forceert competitie:
er zijn altijd expliciet meer verliezers dan winnaars. Verandert semantiek van
doelen niet — alleen de selectie.

**2. Diversity-collapse detector** (zelfde fn, vóór top-K)
Meet per tick:

- `d3_ratio = active_depth3 / total_depth3`
- `score_cv = std(ignition_scores) / mean(ignition_scores)`

Als `n_cand ≥ 64 AND d3_ratio > 0.5 AND score_cv < 0.05` → **hard kick**:
random phi-injectie op 20% van candidates, amplitude `field.clamp_range` (20×
sterker dan v2.6.6 boredom-kick), skip de ignition-cycle, reset boredom_streak.
Niet gebaseerd op emotie-heuristiek maar op structurele homogeniteit van het
kandidaat-veld zelf.

### Observability (elke 1000 ticks)

Log "GNW ignition check" breidt uit met `d3_ratio`, `score_mean`, `score_var`,
`score_cv`, `depth3_total`. Nieuwe log "GNW top-K sparsity" toont `k`, `cutoff`,
`pruned`. Collapse-events loggen met `ERROR`-niveau zodat ze zichtbaar boven
reguliere INFO-stroom verschijnen.

### Niet in deze release (gepland als v2.6.11 als attractor terugkomt)

- Softmax-normalisatie op ignition-scores (grotere dynamiek-ingreep)
- Goal cooldown / repeated-goal blacklist in thinker.rs

Rationale: eerst observeren of top-K + collapse-detector volstaat; de
goal-spiral is vermoedelijk een **gevolg** van de collapse en zou met hersteld
onderscheidingsvermogen vanzelf moeten normaliseren.

### Deploy-pad

Geen snapshot-schema wijziging → migratie onnodig. Standaard `bash deploy.sh`
(code-sync + hercompilatie in container). Data intact, Lucy herstart met v2.6.9
data en v2.6.10 logica.

---

## v2.6.9 (2026-04-13) — Snapshot-migratie framework [COMMITTED, READY TO DEPLOY]

**Doel:** Oplossen van de v2.6.8 deploy-crisis — maak struct-wijzigingen aan
Field/Explorer/RetrievalStore voortaan safe via expliciete versioned snapshots
en migratie-impls.

**Status:** Code gecommitteerd. End-to-end getest:
legacy snapshot (zonder header) → `FieldV1` → migratie → current `Field` ✓
headered roundtrip: save → load ✓.

### Nieuwe infrastructuur

- **`SnapshotHeader`** (`lucy-state/src/snapshot_header.rs`) — 20-byte fixed-size
  header: magic `b"LUCY"` + `schema_version: u32` + `payload_size: u64` + reserved.
  Zero-cost detectie: 4-byte magic check vóór bincode.
- **`schema_v1::FieldV1`** — exact kopie van v2.6.7-era Field struct (pre-v2.6.8
  velden alleen). Wordt gebruikt voor legacy-load fallback.
- **`From<FieldV1> for Field`** — expliciete migratie: 1-op-1 veld-mapping voor
  behouden velden, nieuwe v2.6.8+ velden krijgen `Default`.
- **`Field::load`** nieuw versioned pad:
  1. Detecteer magic bytes
  2. **Headered v1**: direct deserialize current Field
  3. **Legacy**: try `FieldV1` → migrate via `From` → current Field
  4. **Fallback**: probeer direct current (voor pre-v2.6.7 snapshots)
- **`Field::save`** schrijft altijd met current header (`SnapshotHeader::current`).

### Nieuwe tools

- **`crates/snapshot_check/src/bin/migrate_snapshot.rs`** — CLI:
  ```
  cargo run -p snapshot_check --bin migrate_snapshot -- input.snapshot output.snapshot
  ```
  Leest any versie, schrijft current-headered output. Offline migratie voor
  backups of deploy-migraties.
- **`test_v269`** — end-to-end integration test (`cargo run -p snapshot_check --bin test_v269`).

### Ontwerp-regels (aangescherpt t.o.v. v2.6.8)

1. **Bump `CURRENT_SCHEMA_VERSION`** bij elke breaking Field/Explorer/RetrievalStore wijziging.
2. **Nieuw schema = nieuwe module**: `schema_v2.rs`, `schema_v3.rs`, ... met exacte
   layout + `From<schema_vN::Field> for current::Field` impl.
3. **Pre-deploy verplicht**: download live snapshot, run `cargo run -p snapshot_check`.
   Als geen 🟢 → NIET deployen (bouw eerst compat-struct).
4. **Header-magic is stabiel** (`b"LUCY"`). Versie-field is het enige wat bump't.
5. **CRC32 zit op RAW bytes** (incl. header) voor atomische integriteit.

### Bekende beperking

De HUIDIGE live lucy.world snapshot (geschreven door v2.6.7 binary in de container,
vermoedelijk pre-hotfix `49d051f`) laadt NIET met onze `FieldV1` struct — zelfs niet
met een verse checkout van commit `49d051f`. Dit impliceert dat de container-binary
een subtly andere struct layout heeft gehad dan wat in git staat. Voor v2.6.9 deploy:

- **Optie 1 (voorkeur):** live Lucy schrijft nieuwe snapshot via lopende v2.6.7 binary →
  `sigterm graceful shutdown` → stop → `bash deploy.sh` → v2.6.9 leest via legacy pad.
  Als dat faalt: voer `migrate_snapshot` op `/lucy/data-safety-backup/lucy.snapshot` uit
  nadat die eenmaal succesvol door v2.6.7 is gelezen + opnieuw opgeslagen.
- **Optie 2:** Accepteer fresh-bootstrap (vocab + journal behouden, couplings verloren).
- **Optie 3:** Build `dump_v267_state` binary die via `docker exec` de live v2.6.7 binary
  aanroept met speciale env-vars om state als JSON te exporteren. Complex, uitgesteld.

---

## v2.6.8 (2026-04-13) — Bewijs-architectuur [COMMITTED, NIET GEDEPLOYED]

**Doel:** Observability + auditability voor v2.6.7 claims. Geen nieuwe cognitieve
lagen; alle code is meet-instrumentatie om te bewijzen dat zelfmodel-vorming,
concept-compressie en teacher fade-out daadwerkelijk gebeuren.

**Status:** Code gecommitteerd (`9acfbb8`, `4593d30`). **Niet actief op lucy.world**
vanwege bincode snapshot-incompatibiliteit. Zie [SNAPSHOT_MIGRATION.md](./SNAPSHOT_MIGRATION.md).

### Nieuwe features (in code)

- **P1.1** — `RetrievalEntry.kind` (u8: 0=external, 1=self, 2=teacher, 3=interaction).
  Primary truth voor provenance-classificatie, geen string-heuristiek.
- **P1.2** — Response provenance + confidence in `/ask`:
  `fallback | field_generated | retrieval_{self,teacher,external} | mixed`.
  AtomicU64 counters op AppState voor cumulatieve tracking.
- **P1.3** — `/self_metrics` endpoint:
  `self_cluster_density`, `self_cluster_drift`, `self_centroid_strength`,
  `self_query_hit_rate`, `self_response_fallback_rate`, `self_state_reuse_rate`.
- **P1.4** — `/teacher_metrics` endpoint met delayed effect-outcome:
  `TeacherEvent.pending_until_tick` + `effect_outcome: Option<bool>`.
  Tick-loop scant elke 500 ticks en vult outcome in op basis van sat_delta + reuse.
- **P2.1** — `ConceptAudit` ring-buffer (cap 64): per clustering-cycle
  `k`, `avg_intra/inter_cosine`, `silhouette_sampled`, `priors_bias_mean_norm`.
- **P2.2** — Ablation window endpoint `POST /debug/toggle_priors?enabled=false&ticks=N`.
  Auto-restore na N ticks via `concept_priors_disabled_until_tick`.
- **P2.3** — `SelfEpAudit` ring-buffer (cap 64): pre/post tension delta rond
  self-EP in `reflect_tick`, WM-seq pairs, dream-replay count.
- **P3.1** — Self-test runner in audit-only modus (nieuw `self_test.rs`):
  5 vaste probe-vragen (`"wie ben je"`, etc), elke 5000 ticks via `field.clone()`.
  Geen micro_ep / retrieval.insert side-effects — pure gedrag-meting.
  `process_query_full(audit_only: bool)` propagateert flag.
- **P3.2** — `/self_history` endpoint: tijdreeks uit `data/self_test.jsonl`
  (appendlog, max 240 recent rows).

### Nieuwe bestanden

- `crates/lucy-core/src/ai_teacher.rs` — DDG AI Chat + DDG Lite rotatie (feature uit v2.6.7)
- `crates/lucy-core/src/self_test.rs` — Audit-only probe runner
- `crates/snapshot_check/` — Offline snapshot validator (nieuw tooling)
- `docs/SNAPSHOT_MIGRATION.md` — Lessons learned + migratiepaden
- `docs/CHANGELOG.md` — Dit bestand

### Bekende issue

**Bincode snapshot-incompatibiliteit.** Het toevoegen van ANY nieuw veld aan
`Field`, `Explorer` of `RetrievalStore` breekt de deserialisatie van bestaande
snapshots, ook met `#[serde(default)]` en ook bij toevoeging aan einde-struct.

**Mitigatie:** voor v2.6.9+ moet een `FieldV(n-1)` compat-struct of expliciete
migratie-header worden toegevoegd — zie [SNAPSHOT_MIGRATION.md](./SNAPSHOT_MIGRATION.md).

---

## v2.6.7 (2026-04-13) — Stabiel + Self-EP + AI-bootstrap + Depth-3 clustering [GEDEPLOYED]

Commits: `a20f3a8`, `49d051f`. **Live op lucy.world** vanaf ~20:00 CEST.

### Fase A — Numerieke stabiliteit

- **A1** — Phi clamp in `update_phi` + parallelle versie (direct, niet pas na 10 ticks via layer_norm)
- **A2** — Tension anti-Inf safety clamp (`f32::MAX / 1e6`)
- **A3** — Boredom-kick magnitude relateren aan `field.clamp_range * 0.05` (was vaste 0.3)
- **A4** — Noise amplitude relateren aan `field.clamp_range` voor bored/sluggish modi

### Fase B — Self-EP loop

- **B1** — Auto-encoder in `reflect_tick`: `micro_ep(field, &[self_state_input], &[self_state_input])`.
  Lucy traint zichzelf om haar eigen staat te voorspellen — pure weight-update, geen storage.
- **B2** — WM-sequence training: `micro_ep` op opeenvolgende working-memory slots elke reflect.
- **B3** — Dream-replay: `consolidation.replay_one()` extra bij `is_sluggish_flag`.

### Fase C — AI-teacher bootstrap (zelfafbouwend)

- Nieuwe module `ai_teacher.rs` — DDG AI Chat via vqd-token + SSE stream
- Provider-rotatie met EMA `success_rate` per provider
- Vraagformulering uit `explorer.interests` — géén hardcoded prompts
- `teacher_interval`: ×2 na 3 succes, ÷2 bij fail → zelfuitfaden

### Fase D — Bron-balans (organisch)

- `diversify_frontier` strenger (10% per domein) bij monocultuur > 50% top-domein
- `Explorer.domain_score: HashMap<String, f32>` met satisfaction-EMA per domein
- `fresh_source_url` skip 50% wanneer ≥ 50 unieke visited-domeinen

### Fase E — Depth-3 concept-clustering

- **E1** — `compute_concept_centroids`: mini-batch k-means (k=32-256) op depth-3 phi's
- **E2** — `hierarchical_prediction` uitgebreid: concept-centroid als extra top-down prior
  voor depth-2 punten (50% hier_strength)
- **E3** — `consolidate_weights` uitgebreid: intra-concept couplings ×1.5 importance,
  cross-concept ×0.7
- **E4** — `RetrievalStore` cap 100K → 15K, FIFO → score-based eviction
  (`hits/(tick-last_access+1) + learning_value`)
- **E5** — Self-state archive elke 500 reflects + `/ask` interaction-log

### Fase B1 was onvoldoende — zie v2.6.7 hotfix

Hotfix `49d051f`: NaN-safe free_energy update in `tick_all_parallel` na
snapshot-load. Oude NaN-values bleven in point.free_energy zitten omdat parallelle
tick nooit naar point.free_energy schreef. Fix: `PointUpdate.free_energy` veld
toegevoegd, EMA update met `is_finite()` guard + clamp 0..f32::MAX/1e6.

---

## v2.6.6 (2026-04-13) — Boredom-attractor escape [GEDEPLOYED]

Commit: `2881bd4`. **Vervangen door v2.6.7** binnen uren.

- Detecteert fixed-point in tension/curiosity (10 opeenvolgende identieke
  adapt_dynamics calls = 500 ticks onveranderd)
- Bij detectie: random phi-injectie (±0.3 per dim) in 1% van hidden points
- Verhoogt noise van 0.02 → 0.12 bij bored

**Probleem geïntroduceerd:** de ±0.3 × 256 dims = magnitude ~4.8 per tick zonder
phi-clamp cascadeerde via `phi → phi_var → KL → free_energy → tension → ∞`.
Live site bereikte `tension = 1.4e34`. Gerepareerd in v2.6.7 Fase A.

---

## v2.6.5 (2026-04-12) — Neuromodulator bugs [GEDEPLOYED]

Commits: `cabf463`, `13be5d7`, `01ec598`, `b89ab6e`.

- `ema_decay` 0.99 → 0.90 (bewustzijn was dood door te trage EMA)
- Dopamine circulair-dependency fix — was vast op 0.01
- Ignition NaN guard
- `√DIM` ipv `DIM` normalisatie in weight updates (16× sterker signaal)

---

## v2.6.4 (2026-04-10) — Stabiliteitsfix phi-explosie

- v2.6.4 `sluggish_signals >= 2` detector (niet 1 meer, was false positives)
- NaN neighbors skip in update_phi gradient
- Oscillatie-amplitude gecapt

---

## v2.6.1 (2026-04-07) — Homeostasis death spiral fix

- `synaptic_homeostasis` skipt bij avg_couplings < 4.0
- Pre-check survival count vóór prune
- Min 6 couplings per point
- Auto-rewire bij startup als total_couplings == 0

---

## v0.1 - v2.6.0 — Pre-changelog periode

Zie `git log` en `HANDOFF.md` voor eerdere geschiedenis.

---

## Deploy-safety checklist (toegevoegd na v2.6.8 incident)

Voor elke deploy die `Field`/`Explorer`/`RetrievalStore` struct wijzigt:

1. ☐ Download live snapshot: `ssh root@server 'docker cp lucy:/lucy/data/lucy.snapshot /tmp/'`
2. ☐ `cargo run -p snapshot_check --release -- /tmp/lucy.snapshot` → moet 🟢 geven
3. ☐ Bij rood: schrijf `FieldV(n-1)` compat-struct + `From` impl
4. ☐ Deploy via `bash deploy.sh`
5. ☐ Post-deploy: verify `curl /telemetry | jq .tick` matcht pre-deploy tick ± 200
6. ☐ Post-deploy: verify `curl /retrieval_metrics | jq .total_entries` ≥ pre-deploy count
7. ☐ Bij fail: `docker stop lucy && cp /lucy/data-safety-backup/* /lucy/data/ && docker start lucy`

## Rollback-procedure

Als deploy faalt en data verloren dreigt:

```bash
# Stop Lucy
docker stop lucy

# Restore uit data-safety-backup (altijd gemaakt in deploy stap 0)
docker cp /lucy/data-safety-backup/lucy.snapshot     lucy:/lucy/data/lucy.snapshot
docker cp /lucy/data-safety-backup/lucy.crc32        lucy:/lucy/data/lucy.crc32
docker cp /lucy/data-safety-backup/retrieval.bincode lucy:/lucy/data/retrieval.bincode
docker cp /lucy/data-safety-backup/retrieval.crc32   lucy:/lucy/data/retrieval.crc32
docker cp /lucy/data-safety-backup/explorer.bincode  lucy:/lucy/data/explorer.bincode
docker cp /lucy/data-safety-backup/explorer.crc32    lucy:/lucy/data/explorer.crc32

# Als binary ook rollback nodig: kopieer oude binary terug
# (Houd /tmp/lucy-core-v2XX-backup op de server voor elke deploy)
docker cp /tmp/lucy-core-vXXX lucy:/lucy/lucy-core

# Start
docker start lucy

# Verifieer
sleep 10 && docker logs lucy --tail 20 | grep -i "snapshot geladen"
```
