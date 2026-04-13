# Snapshot-migratie — lessons learned & vooruitpad

**Status:** v2.6.8 code gecommitteerd (9acfbb8, 4593d30) maar NIET gedeployed.
lucy.world draait v2.6.7 met volledige data intact (tick ≥ 492750, 20933 points,
431K couplings, 11.321 retrieval entries, vocab 11984, 368 interests).

## Probleem in een zin

**Bincode 1.x is strict positioneel.** Elke nieuwe veld in `Field`/`Explorer`/
`RetrievalStore` breekt backward-compatibility met bestaande snapshots, ook met
`#[serde(default)]` en zelfs bij toevoeging aan het einde van de struct.

## Wat er gebeurd is (2026-04-13)

### Timeline

1. **19:32 CEST** — v2.6.7 commit `a20f3a8` (self-EP + AI-bootstrap + concept-clustering).
   Voegt Field-velden toe: `is_sluggish_flag`, `concept_centroids`, `concept_assignments`,
   `boredom_streak`, etc.
2. **19:40 CEST** — v2.6.7 hotfix `49d051f` (NaN-safe free_energy).
3. **~20:00 CEST** — Deploy v2.6.7 op lucy.world. Lucy draait stabiel op tick ≥ 492750.
4. **20:04 CEST** — v2.6.8 commit `9acfbb8` (bewijs-architectuur).
   Voegt toe aan Field: `concept_priors_enabled: bool`, `concept_priors_disabled_until_tick: u64`,
   `concept_audit_log: Vec<ConceptAudit>`, `self_ep_audit: Vec<SelfEpAudit>`,
   `provenance_counts: [u64; 6]`. Voegt toe aan Explorer: `teacher_audit`.
5. **20:08 CEST** — Deploy v2.6.8. **Primary snapshot load faalt** met
   `invalid u8 while decoding bool, expected 0 or 1, found 64`. Alle 6 backups falen.
   Lucy start blank (tick=0, fresh bootstrap). Retrieval store wordt tick=100 (leeg)
   weggeschreven als primary → **data-verlies cascade**.
6. **20:13 CEST** — v2.6.8 hotfix `4593d30`: alle nieuwe velden `#[serde(skip)]`.
   Deploy. **Snapshot load blijft falen** — ook met alle velden skip.
7. **20:20 CEST** — Rollback naar v2.6.7 binary. `lucy.snapshot` primary is gecorrumpeerd
   (tick=100 dump), maar backup #2 is nog goed (tick=492950). Lucy recovered van backup.
8. **20:30 CEST** — Retrieval store en explorer waren óók tick=100 fresh geschreven
   (geen backups zoals field). **Gered via `/lucy/data-safety-backup/` van deploy stap 0.**

### Eindtoestand

- Field tick=492750 (geaccepteerd verlies van 200 ticks) ✓
- Vocab 11984 tokens ✓
- Retrieval 11321 entries ✓
- Explorer 591 frontier, 368 interests ✓
- 431K couplings ✓
- v2.6.8 code in git, niet gedeployed

## Waarom #[serde(default)] niet helpt bij bincode

Bincode 1.x gebruikt een positional visitor (`visit_tuple`). Bij deserialiseren leest
het veld-voor-veld in declaratie-volgorde, exact N bytes per veld. Bij EOF vóór alle
velden gelezen zijn: **`io::Error::UnexpectedEof`** — ook al heeft het veld `serde(default)`.

JSON/YAML gebruiken `visit_map`, waar `serde(default)` wél werkt (ontbrekende key →
default). Bincode's tuple-visitor negeert missende velden niet.

Experimenteel bewijs (zie `crates/snapshot_check/src/bin/try_read.rs`):
- `bincode::deserialize(&bytes)` → faalt
- `bincode::options().with_fixint_encoding().allow_trailing_bytes().deserialize(&bytes)` → faalt identiek

## Waarom ook `#[serde(skip)]` niet volstond

In theorie: skip = geen bytes geschreven of gelezen → identiek aan v2.6.7 stream.

In praktijk: de LIVE snapshot op disk bleek ook incompatibel met een verse
checkout van v2.6.7 commit `49d051f` (lucy-state/lib.rs). Dat impliceert dat de
draaiende v2.6.7 binary op de server een *iets* andere Field layout had dan wat
in git staat — vermoedelijk een lokale dev-build met extra velden die nooit
gecommitteerd zijn, of een timestamping-mismatch tussen de `/tmp/lucy-core-v267`
rollback-binary en de git tag.

Dit is niet diagnostisch gesloten. De pragmatische conclusie:
**elke struct-wijziging aan `Field`/`Explorer`/`RetrievalStore` is een breaking change
tenzij expliciet gemigreerd.**

## Veilige migratiepaden — voor volgende deploy

### Optie A: FieldV267 compat-struct (aanbevolen)

Definieer in `lucy-state`:

```rust
// FieldV267 = EXACT kopie van v2.6.7 Field struct
#[derive(Deserialize)]
pub struct FieldV267 {
    // ... alle v2.6.7 velden in exacte volgorde ...
}

impl From<FieldV267> for Field {
    fn from(v1: FieldV267) -> Self {
        Self {
            // ... map v1 velden naar v2 ...
            // nieuwe v2.6.8 velden → default
            concept_priors_enabled: true,
            concept_audit_log: Vec::new(),
            // ...
        }
    }
}

impl Field {
    pub fn load(path: &Path) -> Result<Self, ...> {
        let bytes = std::fs::read(path)?;
        // Try v2 first
        match bincode::deserialize::<Field>(&bytes) {
            Ok(f) => return Ok(f),
            Err(_) => {}
        }
        // Fallback to v1 compat
        match bincode::deserialize::<FieldV267>(&bytes) {
            Ok(v1) => Ok(v1.into()),
            Err(e) => Err(format!("Both v2 and v1 load failed: {}", e).into()),
        }
    }
}
```

**Vereisten:**
- `FieldV267` struct moet BIT-PRECIES matchen met wat v2.6.7 schreef.
- Wordt vastgelegd door v2.6.7 git-commit te pakken (`git show 49d051f:crates/lucy-state/src/lib.rs`).
- Onderhoudt de `impl From` wanneer Field verder evolueert.

### Optie B: Expliciete snapshot_version + header

Schrijf header als eerste bytes:

```rust
struct SnapshotHeader {
    magic: u32,      // 0xLUCY
    version: u32,    // 1=v2.6.7, 2=v2.6.8, ...
    payload_size: u64,
}
// Dan payload als bincode Field
```

Bij load: eerst header lezen, dan pad kiezen per versie.

**Vereist** schema-bumps bij elke Field wijziging. Zwaar maar robuust.

### Optie C: Export/import via JSON

Gebruik JSON als tussenformaat voor migraties:

```bash
# Op oude binary
curl -X POST http://localhost:3000/admin/export_state > state.json

# Op nieuwe binary (fresh boot)
curl -X POST http://localhost:3000/admin/import_state -d @state.json
```

**Makkelijk te schrijven**, maar vereist dat Lucy API-endpoints heeft tijdens migratie.
Kost ~500MB voor 20K points x 256 DIM (= 5M floats).

### Optie D: Fresh re-train acceptance (laatste redmiddel)

Bij elke struct-wijziging:
1. Pre-save: export vocab, journal, sources.txt (deze zijn text/json → versiestabiel)
2. Fresh bootstrap met nieuwe binary
3. Retrain op sources.txt + replay van vocab
4. Kost enkele dagen "warming-up"

**Alleen gebruiken** als data-verlies van couplings acceptabel is. NIET voor productie.

## Ontwerp-regels vanaf nu

1. **Geen nieuwe velden** aan `Field`, `Explorer`, of `RetrievalStore` zonder:
   - Offline test met live snapshot via `cargo run -p snapshot_check`
   - Of: FieldV(n-1) compat struct geschreven + `From<FieldV(n-1)> for Field` impl
2. **Audit-velden bij voorkeur op `AppState` (RAM)**, niet op `Field` (persistent).
   Voorbeeld: `provenance_counts` zitten nu op `AppState` als `AtomicU64` — reset bij reboot,
   maar breken geen snapshot.
3. **Ring-buffers op Field mogen `#[serde(skip)]`** als data-verlies bij reboot acceptabel is.
4. **Nooit bool toevoegen mid-struct**, alleen aan eind + documented migration.
5. **Altijd `/lucy/data-safety-backup/`** checken bij rollback-scenario's.

## Tooling toegevoegd in deze sessie

- `crates/snapshot_check/` — standalone binary voor offline snapshot validatie.
  Gebruik: `cargo run -p snapshot_check --release -- /path/to/lucy.snapshot`
  - `snapshot_check` (main): volledige validatie + roundtrip
  - `try_read`: probeert met meerdere bincode options
  - `roundtrip`: sanity-test dat Field::new() succesvol round-trips
  - `try_diff`: hex-inspectie van snapshot
- `crates/lucy-core/src/ai_teacher.rs` — nieuw, provider-rotatie (gecommit, niet gedeployed)
- `crates/lucy-core/src/self_test.rs` — nieuw, audit-only probe suite (gecommit, niet gedeployed)

## Verificatie-workflow (VOOR elke v2.6.9+ deploy)

```bash
# 1. Download LIVE snapshot
ssh root@209.38.36.51 'docker cp lucy:/lucy/data/lucy.snapshot /tmp/lucy.snapshot'
scp root@209.38.36.51:/tmp/lucy.snapshot /tmp/snap-test/

# 2. Build + test offline
cargo build -p snapshot_check --release
cargo run -p snapshot_check --release -- /tmp/snap-test/lucy.snapshot
# Must print "🟢 ALL CHECKS PASSED"

# 3. Als fail: lever migratie via Optie A/B hierboven. Anders NIET deployen.

# 4. Pas na pass: deploy via bash deploy.sh
```

## Referenties

- Bincode 1.x positional: https://github.com/bincode-org/bincode/issues/167
- Serde field reordering compatibility: https://serde.rs/field-attrs.html#default
- Commit `9acfbb8` (v2.6.8 features, gecommitteerd niet gedeployed)
- Commit `4593d30` (v2.6.8 hotfix skip, niet gedeployed)
